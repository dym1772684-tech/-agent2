"""黑板多智能体(supervisor + 三专员 + 汇总),LangGraph Send 并行扇出版。

【架构】(对照批准方案)
    supervisor:一次性结构化输出"任务清单"(plan-and-execute,不跑 ReAct 循环)
        ↓ 内置权限过滤器:无权限的任务直接标记 blocked,不派发
    dispatch:Send 图内扇出(并行,不自己开线程)→ 各专员写自己的黑板字段
        🩺 health    → health_report    (老人数据,工具走 MCP 网关)
        🌿 knowledge → knowledge_report (RAG 养生知识)
        🧰 life      → life_report      (天气/计算)
    aggregate:强模型汇总黑板 → ParsedOutput + 追加 AIMessage(多轮兼容)

【字段所有权】(黑板纪律:每个专员只写自己的字段,避免并行写冲突)
    health_report / knowledge_report / life_report / final 各归其主,
    messages 只有 aggregate 追加。
"""
from __future__ import annotations

from typing import Annotated, Any, Dict, List, Literal, Optional

from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy
from langchain.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.messages import BaseMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.types import Send
from pydantic import BaseModel, ConfigDict, Field

from allagent2.agent.agent import create_chat_model
from allagent2.agent.middleware import identity_guard
from allagent2.agent.state import CareAgentState, ParsedOutput, PermissionContext
from allagent2.guance.logging_helper import logger
from allagent2.prompts.prompts import EMERGENCY_REPLY
from allagent2.tools.knowledge_tool import search_knowledge_base
from allagent2.tools.tool import tools_name as _registry

# 三类专员的工具清单从注册表拿(MCP 版本由 bootstrap 注入后就在这里);
# 拿不到(如 MCP 服务未启动被去重删除)→ 专员无工具,仍可回答"能力不可用"
ELDER_TOOLS = ["query_elder_profile", "query_health_records", "analyze_health_trend"]
LIFE_TOOLS = ["calculator", "get_real_weather", "fetch_webpage",
              "analyze_health_trend", "compute_bmi"]


# ============================================================
# 结构化 schema
# ============================================================
class Task(BaseModel):
    agent: Literal["health", "knowledge", "life"]
    instruction: str = Field(default="", description="交给该专员的具体任务说明")
    requires_elder_access: bool = Field(
        default=False, description="是否需要访问老人数据(用于权限过滤)"
    )


class Plan(BaseModel):
    need_user_input: Optional[str] = Field(
        default=None, description="信息不足以执行时,要向用户追问的问题(此时不派发任务)"
    )
    tasks: List[Task] = Field(default_factory=list)


class HealthReport(BaseModel):
    summary: str = ""
    risk_level: str = "低"
    emergency_flag: bool = False
    health_advice: str = ""
    medication_advice: str = ""


class KnowledgeReport(BaseModel):
    knowledge_answer: str = ""
    diet_advice: str = ""
    exercise_advice: str = ""
    health_advice: str = ""


class LifeReport(BaseModel):
    answer: str = ""


class TeamState(BaseModel):
    """黑板 = 共享 state(字段所有权分区,见模块 docstring)。"""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    messages: Annotated[List[BaseMessage], add_messages]
    question: str = ""
    permissions: Optional[PermissionContext] = None
    # 【记忆系统】链路一召回的记忆注入块(bootstrap._prepare 填充;空串=无记忆)
    memory_block: str = ""
    # 【技能系统】本轮启用技能渲染出的 prompt 块(空串=无技能);supervisor 与专员共用
    skills_block: str = ""
    # 【零信任·S3】可信身份引用(identity:{id}),identity_guard 每次模型调用前复验
    identity_id: str = ""
    tasks: List[Task] = Field(default_factory=list)
    blocked_notes: List[str] = Field(default_factory=list)
    # 黑板字段(各专员只写自己的)
    health_report: Optional[Dict[str, Any]] = None
    knowledge_report: Optional[Dict[str, Any]] = None
    life_report: Optional[Dict[str, Any]] = None
    final: Optional[Any] = None  # ParsedOutput,用 Any 避免递归校验开销


def _history_text(messages: List[BaseMessage], k: int = 6) -> str:
    """最近 k 条消息拼成文本(给 planner/专员看的会话上下文)。"""
    parts = []
    for m in messages[-k:]:
        role = getattr(m, "type", "unknown")
        content = getattr(m, "content", "")
        if isinstance(content, str) and content:
            parts.append(f"{role}: {content[:200]}")
    return "\n".join(parts)


# ============================================================
# 节点 1:supervisor —— 一次性任务清单 + 权限过滤(plan-and-execute)
# ============================================================
PLANNER_PROMPT = """你是养老平台的调度主管。根据用户问题规划任务清单,交给专员执行。
可选专员:
- health:查询老人档案/健康记录、评估健康风险(需要 elder_id 且必须访问老人数据)
- knowledge:养生知识解答(饮食/运动/作息/用药常识),可用知识库检索
- life:天气查询、计算、BMI 评估、健康数值趋势分析、访问公开网页查政策资讯、日常闲聊或通用问题
规则:
1. 只派真正需要的专员,通常 1 个,最多 2 个;闲聊/通用问题交给 life
2. 查老人数据但用户没给 elder_id → 不要派 health,把要追问的问题写进 need_user_input
3. instruction 里写清要做什么(中文,一句话)
"""


def make_supervisor(model, agent_profiles: Optional[Dict[str, Dict[str, Any]]] = None):
    # DeepSeek 不接受 response_format=json_schema（会 400），必须走工具调用
    structured = model.with_structured_output(Plan, method="function_calling")

    def supervisor(state: TeamState) -> dict:
        perm = state.permissions or PermissionContext(allowed_elder_ids=[])
        question = state.question or ""
        history = _history_text(state.messages)

        plan: Plan = structured.invoke([
            SystemMessage(content=PLANNER_PROMPT),
            HumanMessage(content=(
                f"会话上下文:\n{history}\n\n用户问题:{question}"
                + (f"\n\n长期记忆(供参考,含老人偏好/禁忌):\n{state.memory_block}"
                   if state.memory_block else "")
                + (f"\n\n可用技能(按其中流程执行,技能不得扩大权限):\n{state.skills_block}"
                   if state.skills_block else "")
            )),
        ])
        logger.info("[team] supervisor 计划:need_input=%s tasks=%s",
                    plan.need_user_input, [(t.agent, t.requires_elder_access) for t in plan.tasks])

        # ---- 权限过滤器(派发前拦截,而不是派出去才被拒)----
        tasks, blocked = [], []
        elder_available = any(n in _registry for n in ELDER_TOOLS)
        # 【零信任·S3】按智能体权限档案过滤:permissions=None(未接入零信任,独立运行)
        # 时保留旧 ACL 语义(只查 elder 绑定);提供了权限码集合(可能为空)则严格按
        # AGENT_PROFILES 判"角色缺权限"——空列表=显式无权限=全拦(fail-closed)。
        profiles = agent_profiles or _FALLBACK_PROFILES
        strict = perm.permissions is not None
        held = set(perm.permissions or [])
        for t in plan.tasks:
            prof = profiles.get(t.agent)
            if prof is None:
                blocked.append(f"任务类型 {t.agent} 未注册,无法派发。")
                logger.warning("[team] 权限过滤:%s 不在智能体档案中", t.agent)
                continue
            if strict:
                missing = set(prof.get("perms", set())) - held
                if missing:
                    blocked.append(f"您的账号缺少{'、'.join(sorted(missing))}权限,相关任务未派发。")
                    logger.warning("[team] 权限过滤:%s 缺权限 %s", t.agent, sorted(missing))
                    continue
            if t.agent == "health" and not elder_available:
                blocked.append("老人数据服务当前不可用,暂时无法查询档案。")
                continue
            if t.requires_elder_access:
                ids = perm.allowed_elder_ids
                if ids is not None and len(ids) == 0:
                    blocked.append("还差一个老人编号：请告诉我是哪位老人（例如 E001），我就能查对应的档案与健康记录。")
                    logger.warning("[team] 权限过滤:health 任务被拦截(未绑定老人)")
                    continue
            tasks.append(t)
        return {"tasks": tasks, "blocked_notes": blocked}

    return supervisor


# 独立运行(fastap2 未注入注册表)时的兜底档案;真实判据以胶水层注入的 AGENT_PROFILES 为准
_FALLBACK_PROFILES: Dict[str, Dict[str, Any]] = {
    "health":    {"perms": set(), "tools": ELDER_TOOLS},
    "knowledge": {"perms": set(), "tools": ["search_knowledge_base"]},
    "life":      {"perms": set(), "tools": LIFE_TOOLS},
}


# ============================================================
# 节点 2(条件边):dispatch —— Send 图内并行扇出
# ============================================================
def dispatch(state: TeamState):
    if not state.tasks:
        return "aggregate"  # 全被拦截/需追问 → 直接汇总(由 aggregate 处理追问)
    payload_base = {
        "question": state.question,
        "history": _history_text(state.messages),
        "permissions": state.permissions,
        "identity_id": state.identity_id,  # 【零信任·S3】随任务下发,专员内 identity_guard 复验
        "skills_block": state.skills_block,  # 【技能】专员也要看到同一份技能约束
    }
    return [
        Send(t.agent, {**payload_base, "instruction": t.instruction})
        for t in state.tasks
    ]


# ============================================================
# 专员:每个是小型 create_agent(弱职责,只写自己的黑板字段)
# ============================================================

def _specialist_middleware() -> list:
    """专员共用防线：身份复验 + 工具白名单/老人越权。

    与 agent/middleware.build_middleware 使用同一批守卫实现，只是装配成列表
    给嵌套 create_agent 用。三个专员共用一份装配，避免「某个专员漏挂防线」。
    """
    from allagent2.agent.middleware import elder_access_guard

    return [identity_guard, elder_access_guard]



def _pick_tools(names: List[str]) -> list:
    return [_registry[n] for n in names if n in _registry]


def make_health_agent(model):
    agent = create_agent(
        model=model,
        tools=_pick_tools(ELDER_TOOLS),
        system_prompt=(
            "你是养老健康专员。调用工具查询老人档案与健康记录后做风险评估。"
            "若用户没提供 elder_id 或查询被拒,如实说明,不要编造数据。"
            "风险等级(低/中/高)与 emergency_flag(胸痛/意识模糊/跌倒等急症才为 true)必须谨慎。"
        ),
        response_format=ToolStrategy(HealthReport),
        state_schema=CareAgentState,   # 承载 identity_id/permissions,identity_guard 才读得到
        middleware=_specialist_middleware(),   # 【零信任·S3】第二道纵深:模型调用前复验身份
    )

    def health_agent(payload: dict) -> dict:
        result = agent.invoke({
            "messages": [
                SystemMessage(content=(
                    (f"{payload.get('skills_block')}\n\n" if payload.get("skills_block") else "")
                    + f"任务:{payload.get('instruction') or '评估老人健康状况'}"
                )),
                HumanMessage(content=f"用户问题:{payload.get('question')}(会话:\n{payload.get('history', '')[:500]})"),
            ],
            "identity_id": payload.get("identity_id") or None,
            "permissions": payload.get("permissions"),
        })
        sr = result.get("structured_response")
        report = sr.model_dump() if hasattr(sr, "model_dump") else dict(sr or {})
        logger.info("[team] health 完成:risk=%s emergency=%s",
                    report.get("risk_level"), report.get("emergency_flag"))
        return {"health_report": report}

    return health_agent


def make_knowledge_agent(model):
    agent = create_agent(
        model=model,
        tools=[search_knowledge_base],
        system_prompt=(
            "你是养生知识专员。优先用知识库检索结果回答饮食/运动/作息/用药常识,"
            "检索不到再凭常识回答并注明'仅供参考,不替代医嘱'。"
        ),
        response_format=ToolStrategy(KnowledgeReport),
        state_schema=CareAgentState,
        middleware=_specialist_middleware(),
    )

    def knowledge_agent(payload: dict) -> dict:
        result = agent.invoke({
            "messages": [
                SystemMessage(content=(
                    (f"{payload.get('skills_block')}\n\n" if payload.get("skills_block") else "")
                    + f"任务:{payload.get('instruction') or '解答养生问题'}"
                )),
                HumanMessage(content=f"用户问题:{payload.get('question')}(会话:\n{payload.get('history', '')[:500]})"),
            ],
            "identity_id": payload.get("identity_id") or None,
            "permissions": payload.get("permissions"),
        })
        sr = result.get("structured_response")
        report = sr.model_dump() if hasattr(sr, "model_dump") else dict(sr or {})
        logger.info("[team] knowledge 完成:diet=%s exercise=%s",
                    bool(report.get("diet_advice")), bool(report.get("exercise_advice")))
        return {"knowledge_report": report}

    return knowledge_agent


def make_life_agent(model):
    agent = create_agent(
        model=model,
        tools=_pick_tools(LIFE_TOOLS),
        system_prompt=(
            "你是生活助手专员,负责天气查询、计算、BMI 评估、健康数值趋势分析、"
            "访问公开网页查证养老政策/医保资讯,以及日常问答。回答简洁友好,"
            "引用网页内容时注明来源链接。"
        ),
        response_format=ToolStrategy(LifeReport),
        state_schema=CareAgentState,
        middleware=_specialist_middleware(),
    )

    def life_agent(payload: dict) -> dict:
        result = agent.invoke({
            "messages": [
                SystemMessage(content=(
                    (f"{payload.get('skills_block')}\n\n" if payload.get("skills_block") else "")
                    + f"任务:{payload.get('instruction') or '回答用户问题'}"
                )),
                HumanMessage(content=f"用户问题:{payload.get('question')}(会话:\n{payload.get('history', '')[:500]})"),
            ],
            "identity_id": payload.get("identity_id") or None,
            "permissions": payload.get("permissions"),
        })
        sr = result.get("structured_response")
        report = sr.model_dump() if hasattr(sr, "model_dump") else dict(sr or {})
        return {"life_report": report}

    return life_agent


# ============================================================
# 节点 3:aggregate —— 强模型汇总黑板 → ParsedOutput
# ============================================================
AGGREGATE_PROMPT = """你是养老平台的首席顾问,汇总各专员的报告给用户最终答复。
要求:
1. answer 用自然、温暖的中文,面向老人或家属,综合各专员结论,不要暴露内部字段名
2. 有 blocked_notes 时,在 answer 里礼貌转述(如"未绑定老人,请联系管理员绑定")
3. need_clarify 有值时,answer 就是这个追问本身,其他字段留空
4. emergency=true 时,answer 必须以紧急就医提示开头
5. 各业务字段只填有依据的内容,没有就留空,不要编造
"""


def make_aggregator(model):
    structured = model.with_structured_output(ParsedOutput, method="function_calling")

    def aggregate(state: TeamState) -> dict:
        reports = {
            "健康专员": state.health_report,
            "知识专员": state.knowledge_report,
            "生活专员": state.life_report,
        }
        lines = []
        for name, r in reports.items():
            if r:
                lines.append(f"[{name}] {r}")
        if state.blocked_notes:
            lines.append("[权限提示] " + ";".join(state.blocked_notes))
        need_clarify = ""
        if not lines:
            need_clarify = "请问您想咨询哪位老人(需先绑定)、或哪方面的养生/生活问题?"

        sr = structured.invoke([
            SystemMessage(content=AGGREGATE_PROMPT),
            HumanMessage(content=(
                f"用户问题:{state.question}\n\n各专员报告:\n" +
                ("\n".join(lines) if lines else "(无,需要向用户澄清)") +
                (f"\nneed_clarify:{need_clarify}" if need_clarify else "")
            )),
        ])

        # 【修复】结构化失败兜底：report 内容丰富时 function call 参数 JSON
        # 容易被截断（曾因 max_tokens=1024 必现），sr 返回 None / answer 为空。
        # 此时降级为纯文本生成，业务字段从各专员报告里搬——保证用户一定拿得到回复。
        final = sr if isinstance(sr, ParsedOutput) else ParsedOutput(**(sr or {}))
        if not (final.answer or "").strip():
            logger.warning("[team] aggregate 结构化输出为空，降级纯文本生成")
            plain = model.invoke([
                SystemMessage(content=(
                    AGGREGATE_PROMPT +
                    "\n直接输出最终答复正文，不要输出任何字段名或 JSON。"
                )),
                HumanMessage(content=(
                    f"用户问题:{state.question}\n\n各专员报告:\n" +
                    ("\n".join(lines) if lines else "(无,需要向用户澄清)")
                )),
            ])
            final.answer = (plain.content or "").strip() or need_clarify or "抱歉，本轮未能生成回复，请换个问法重试。"
            # 业务字段从最相关的专员报告回填（有依据才填）
            best = state.knowledge_report or state.health_report or state.life_report or {}
            for fld in ("diet_advice", "exercise_advice", "health_advice", "medication_advice"):
                if not getattr(final, fld, None) and best.get(fld):
                    setattr(final, fld, best[fld])

        # 急症兜底:健康专员判定急症 → 强制置位并前置急救提示
        if (state.health_report or {}).get("emergency_flag"):
            final.emergency_flag = True
            if not (final.answer or "").startswith("【紧急"):
                final.answer = f"{EMERGENCY_REPLY}\n\n{final.answer or ''}"

        logger.info("[team] aggregate 完成:answer长度=%d emergency=%s",
                    len(final.answer or ""), final.emergency_flag)
        # messages 只由 aggregate 追加(黑板纪律),保证多轮对话兼容
        return {"final": final, "messages": [AIMessage(content=final.answer or "")]}

    return aggregate


# ============================================================
# 组装
# ============================================================
def make_team_graph(checkpointer=None, agent_profiles: Optional[Dict[str, Dict[str, Any]]] = None):
    """构建黑板多智能体图(编译版)。checkpointer 由 bootstrap 传入(Redis)。

    agent_profiles:【零信任·S3】智能体权限档案( fastap2 core.identity.AGENT_PROFILES),
    由胶水层注入作为派发过滤判据;None 时用本文件兜底档案(独立运行场景)。
    """
    # 【修复】显式传 env 的 key/base_url:传 None 会让 init_chat_model 解析成
    # ChatDeepSeek 默认构造,要求 DEEPSEEK_API_KEY(项目实际用的是 OPENAI_API_KEY)。
    import os

    chat_model = create_chat_model(
        os.getenv("OPENAI_API_KEY"), os.getenv("OPENAI_BASE_URL")
    )

    from allagent2.agent.observe import team_node

    _RESULT_KEYS = (
        "tasks", "need_user_input", "blocked_notes",
        "health_report", "knowledge_report", "life_report", "final",
    )

    def _summarize(out) -> dict:
        """从节点返回值里挑出"这一步产出了什么"，供 node_end 事件展示。"""
        if not isinstance(out, dict):
            return {}
        picked = {}
        for key in _RESULT_KEYS:
            value = out.get(key)
            if value in (None, "", [], {}):
                continue
            picked[key] = value
        return picked

    def _traced(name: str, fn):
        """包一层事件流：team_node 会给节点注入 runtime，节点本身只吃 state。

        包了之后每次节点进出都会写 node_start / node_end / interrupted /
        error 事件，前端「思考过程」才看得到"这一步是谁、花了多久、产出了什么"。
        """

        def _call(state, runtime=None):
            out = fn(state)
            if isinstance(out, dict):
                out = dict(out)
                out["_trace"] = {"result": _summarize(out)}
            return out

        return team_node(name)(_call)

    builder = StateGraph(state_schema=TeamState)
    builder.add_node("supervisor",
                     _traced("supervisor", make_supervisor(chat_model, agent_profiles)))
    builder.add_node("health", _traced("health", make_health_agent(chat_model)))
    builder.add_node("knowledge", _traced("knowledge", make_knowledge_agent(chat_model)))
    builder.add_node("life", _traced("life", make_life_agent(chat_model)))
    builder.add_node("aggregate", _traced("aggregate", make_aggregator(chat_model)))

    builder.add_edge(START, "supervisor")
    # 条件边:有任务 → Send 扇出到各专员;无任务 → 直接汇总(追问/全拦截)
    builder.add_conditional_edges("supervisor", dispatch,
                                  path_map=["health", "knowledge", "life", "aggregate"])
    for spec in ("health", "knowledge", "life"):
        builder.add_edge(spec, "aggregate")  # 并行分支全部完成才进 aggregate
    builder.add_edge("aggregate", END)

    agent = builder.compile(checkpointer=checkpointer)
    logger.info("[team] 黑板多智能体编译完成")
    return agent
