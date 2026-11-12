"""养老智能体中间件层（全部基于 LangChain 1.x 官方 middleware API）。

【改造说明 · 总览】
------------------------------------------------------------------
怎么改的：
  原项目 `allagent2/middleware/middleware.py` 里那批中间件有两个致命问题：
    1. 用的是 `create_agent` 的 middleware 装饰器，而当时 graph.py 是手写
       StateGraph —— 两者不兼容，那批中间件从头到尾没有一次被执行过；
    2. 函数体本身是半成品（delete_messages 只有一行 pass、check_rag_privacy
       循环里写 pass、add_ai_think 访问 state.ai_content 越界）。

  现在链路整体迁到 `create_agent`（见 agent/graph.py），中间件才真正挂得上去。
  本文件全部使用官方 API：
       @dynamic_prompt / @before_model / @after_model / @wrap_tool_call
       PIIMiddleware / SummarizationMiddleware / ToolCallLimitMiddleware
       ModelCallLimitMiddleware / ToolRetryMiddleware / HumanInTheLoopMiddleware

改的意义（一一对应此前的断点）：
  ① 人设与安全红线接入   → 解决「prompts.py 是死代码」
  ② 急救关键词短路       → 解决「emergency_flag 无短路」
  ③ 槽位缺失追问         → 解决「next_need 不参与决策、模型硬答」
  ④ 老人数据越权拦截     → 解决「权限校验是空壳分支」
  ⑤ PII 脱敏             → 官方实现替换手写正则，双保险
  ⑥ 长会话自动摘要       → 解决「messages 只增不减、context 爆炸」
  ⑦ 工具/模型调用上限    → 成本闸，替代手写的轮数计数
  ⑧ 工具失败自动重试     → 天气 API 抖动不再直接失败
  ⑨ 可观测日志           → 每次模型调用后留痕
------------------------------------------------------------------
"""
from __future__ import annotations

import re
from typing import Any

from langchain.agents.middleware import (
    HumanInTheLoopMiddleware,
    ModelCallLimitMiddleware,
    ModelFallbackMiddleware,
    ModelRequest,
    PIIMiddleware,
    SummarizationMiddleware,
    ToolCallLimitMiddleware,
    ToolRetryMiddleware,
    after_model,
    before_model,
    dynamic_prompt,
    wrap_tool_call,
)
from langchain.messages import AIMessage, SystemMessage, ToolMessage

from allagent2.agent.state import CareAgentState, PermissionContext
from allagent2.guance.logging_helper import logger
from allagent2.prompts.prompts import (
    EMERGENCY_KEYWORDS,
    EMERGENCY_REPLY,
    build_system_prompt,
)

# 涉及老人个人数据的工具：需要 elder_id 越权校验
_ELDER_DATA_TOOLS = ("query_elder_profile", "query_health_records")

# 出现这些词说明用户在问「个人数据」，此时必须有老人编号
_PROFILE_HINTS = ("档案", "健康记录", "血压", "血糖", "心率", "体重", "体检", "监测")

# 追问指令的幂等标记：避免每一轮都重复注入
_SLOT_ASKED_MARK = "[槽位追问已发出]"

_ELDER_ID_PATTERN = re.compile(r"\b[Ee]\d{2,}\b")


# ============================================================
# 工具函数
# ============================================================
def _resolve_permissions(state: Any, runtime: Any) -> PermissionContext:
    """取权限上下文：state 优先（随 checkpoint 持久化），回退 runtime.context。"""
    perm = None
    try:
        perm = state.get("permissions") if hasattr(state, "get") else None
    except Exception:
        perm = None
    if perm is None:
        ctx = getattr(runtime, "context", None)
        perm = getattr(ctx, "permissions", None) if ctx is not None else None
    if perm is None:
        perm = PermissionContext()
    if isinstance(perm, dict):
        try:
            perm = PermissionContext(**perm)
        except Exception:
            perm = PermissionContext()
    return perm


def _last_human_text(state: Any) -> str:
    messages = state.get("messages", []) if hasattr(state, "get") else []
    for msg in reversed(messages):
        if getattr(msg, "type", "") == "human":
            content = getattr(msg, "content", "")
            return content if isinstance(content, str) else str(content)
    return ""


# ============================================================
# ① 人设与安全红线注入（官方 @dynamic_prompt）
# -----------------------------------------------------------
# 怎么改的：原 prompts.py 定义后无人引用，链路里用的是 bootstrap 里一句
#           临时 SystemMessage。现在由官方 dynamic_prompt 在「每次模型调用前」
#           重新生成 system prompt，成为人设的唯一来源。
# 改的意义：医疗场景的人设 = 安全边界。不接入，模型就能诊断、开药、编造数据。
#           用 dynamic_prompt 而非静态 system_prompt，是因为人设需要按
#           当前用户角色 / 关联老人动态变化。
# ============================================================
@dynamic_prompt
def care_system_prompt(request: ModelRequest) -> str:
    ctx = getattr(request.runtime, "context", None)
    perm = _resolve_permissions(request.state, request.runtime)
    prompt = build_system_prompt(
        user_role=perm.user_role or "common",
        allow_sensitive=bool(perm.allow_sensitive),
        current_elder_id=getattr(ctx, "current_elder_id", None),
        user_name=getattr(ctx, "user_name", None),
    )
    # 【记忆系统】链路一召回的记忆块拼在人设之后（bootstrap._prepare 写入 state；
    # 空串=无记忆不注入；块内自带来源标注，与档案冲突时以档案为准）
    try:
        mem = request.state.get("memory_block") or "" if hasattr(request.state, "get") else ""
    except Exception:
        mem = ""
    if mem:
        prompt += "\n\n# 关于这位用户的长期记忆（仅供参考，与档案冲突时以档案为准）\n" + mem
    return prompt


# ============================================================
# ② 急救短路（官方 @before_model + can_jump_to）
# -----------------------------------------------------------
# 怎么改的：原链路解析出 emergency_flag 后没有任何节点响应，急救问题照样
#           走完 RAG + 多轮 LLM。这里用官方 before_model 的 jump 能力，
#           命中关键词直接 append 一条急救回答并 jump_to="end"，跳过模型调用。
# 改的意义：a) 零 token、零延迟，急救场景不能等；
#           b) 规则判定不依赖 LLM，不会出现"模型今天心情不好没识别出急救"；
#           c) 与 bootstrap 入口短路共用 prompts.EMERGENCY_KEYWORDS，两处判定一致。
# ============================================================
@before_model(can_jump_to=["end"], state_schema=CareAgentState)
def emergency_triage(state: CareAgentState, runtime) -> dict[str, Any] | None:
    text = _last_human_text(state)
    if not text:
        return None
    hits = [kw for kw in EMERGENCY_KEYWORDS if kw in text]
    if not hits:
        return None
    logger.warning("[guard] 命中急救关键词 %s，链路内短路，直接输出急救指引", hits)
    return {
        "messages": [AIMessage(content=EMERGENCY_REPLY)],
        "emergency_flag": True,
        "jump_to": "end",
    }


# ============================================================
# ③ 槽位缺失追问（官方 @before_model）
# -----------------------------------------------------------
# 怎么改的：原 state 里解析了 next_need（含 required_fields），但 route_by_intent
#           只看 intent，next_need 从没参与过决策 —— 信息不全时模型只能硬答，
#           与提示词"不完整要追问"的要求完全相反。这里改成：检测到问题涉及
#           个人数据、却没有老人编号时，注入一条追问指令 SystemMessage。
# 改的意义：把"追问"从"写在提示词里的愿望"变成"链路里真实存在的分支"。
#           幂等标记保证同一轮只注入一次，不会每轮堆积。
# ============================================================
@before_model(state_schema=CareAgentState)
def slot_guard(state: CareAgentState, runtime) -> dict[str, Any] | None:
    ctx = getattr(runtime, "context", None)
    current_elder = getattr(ctx, "current_elder_id", None)
    text = _last_human_text(state)

    if not text or not any(hint in text for hint in _PROFILE_HINTS):
        return None
    if current_elder or _ELDER_ID_PATTERN.search(text):
        return None

    # 幂等：最近几条里已经问过就不再重复注入
    for msg in reversed((state.get("messages", []) or [])[-4:]):
        if getattr(msg, "type", "") == "system" and _SLOT_ASKED_MARK in (
            getattr(msg, "content", "") or ""
        ):
            return None

    logger.info("[guard] 缺少老人编号，注入追问指令，本轮不调工具")
    return {
        "messages": [
            SystemMessage(
                content=(
                    _SLOT_ASKED_MARK
                    + " 用户的问题涉及个人档案或健康记录，但你手上没有老人编号。"
                    "请先用一句亲切的话询问老人编号（或老人姓名），"
                    "本轮不要调用任何工具、不要编造任何数据。"
                )
            )
        ]
    }


# ============================================================
# ④ 老人数据越权拦截（官方 @wrap_tool_call）
# -----------------------------------------------------------
# 怎么改的：原实现写在 tool_use 节点里，靠 _check_node_permission(state, "tool:xxx")。
#           但那个函数只处理 "rag" 和 "tool:*" 两个标签，"plan"/"llm"/"summarize"
#           全是空分支 —— 看着有 5 个权限标签，实际只有 2 个生效（假安全感）。
#           现在用官方 wrap_tool_call，在工具真正执行前统一拦截。
# 改的意义：a) 拦截点从"某个节点"上移到"所有工具调用"，无论模型走哪条路径都拦得住；
#           b) 拒绝时直接返回 ToolMessage 不执行 handler —— 越权查询根本不落库；
#           c) 与后端 DBAccessMiddleware 形成双保险（这里是图内，那里是数据网关）。
# ============================================================
@wrap_tool_call
def elder_access_guard(request, handler):
    call = request.tool_call
    name = call.get("name")
    args = call.get("args") or {}
    perm = _resolve_permissions(request.state, request.runtime)

    # 工具白名单
    if perm.allowed_tools and name not in perm.allowed_tools:
        logger.warning("[guard] 工具越权拦截：%s 不在白名单内", name)
        return ToolMessage(
            content=f"无权调用工具 {name}（不在当前用户的白名单内）",
            tool_call_id=call.get("id"),
        )

    # 老人数据：必须有编号，且必须在授权范围内
    if name in _ELDER_DATA_TOOLS:
        elder_id = args.get("elder_id")
        if not elder_id:
            return ToolMessage(
                content="缺少老人编号 elder_id，请先向用户询问后再查询",
                tool_call_id=call.get("id"),
            )
        allowed = perm.allowed_elder_ids
        # 【安全】allowed 为 None 表示未限定；为 [] 表示显式禁止（fail closed）
        if allowed is not None and elder_id not in allowed:
            logger.warning("[guard] 老人数据越权拦截：尝试访问 %s", elder_id)
            return ToolMessage(
                content=f"无权访问老人 {elder_id} 的数据（超出授权范围）",
                tool_call_id=call.get("id"),
            )

    return handler(request)


# ============================================================
# ⑤ 可观测：每次模型调用后留痕（官方 @after_model）
# -----------------------------------------------------------
# 怎么改的：原来只有节点级 logger，且 phase 靠手写字符串。现在用 after_model
#           统一记录模型调用后的状态，配合 /agent/trace 做回放。
# 改的意义：出问题时能定位"是哪一次模型调用开始跑偏的"，而不是只看最终结果。
# ============================================================
@after_model(state_schema=CareAgentState)
def trace_model_call(state: CareAgentState, runtime) -> None:
    messages = state.get("messages", []) or []
    last = messages[-1] if messages else None
    tool_calls = getattr(last, "tool_calls", None) or []
    logger.info(
        "[chain] 模型调用完成：消息总数=%d，最后消息类型=%s，tool_calls=%s",
        len(messages),
        getattr(last, "type", "-"),
        [c.get("name") for c in tool_calls] if tool_calls else None,
    )
    # 【新增·token可观测】每次模型调用后实时打一行本次 token 消耗。
    # 数据来源是 usage_metadata（API 返回，权威）；这一层回答的是
    # "一轮对话里 5 次模型调用，是哪一次把 token 吃掉的"——定位问题用。
    # 注意这里只打日志不累加：汇总去重由 bootstrap.collect_usage 负责，
    # 两处分工 = 单步定位（这里）+ 单轮账单（bootstrap）。
    try:
        from allagent2.guance.token_accounting import extract_usage

        usage = extract_usage(last) if last is not None else None
        if usage:
            logger.info(
                "[token] 单步调用：input=%d (cache_hit=%d/miss=%d) output=%d",
                usage["input_tokens"],
                usage["cache_read"],
                usage["cache_miss"],
                usage["output_tokens"],
            )
    except Exception:
        pass  # 观测失败绝不影响主链路
    return None


# ============================================================
# ⑩ 零信任·S3：身份复验（官方 @before_model + can_jump_to）
# -----------------------------------------------------------
# 设计：FastAPI 侧装配 TrustedContext（identity:{id} 存 Redis，TTL 900s）后，
#       把 identity_id 随 permissions 注入图状态。本中间件在【每次模型调用前】
#       复验身份仍然可信（未过期 + perm_version 未变 + 属主匹配）——
#       登出/降权/锁定后，正在跑的长链路在下一个模型调用点即被斩断。
# 解耦：allagent2 不 import fastap2。校验器由胶水层（bootstrap）通过
#       set_identity_validator 注入；未注入（独立运行/单测）且有身份时
#       按 fail-closed 拒绝——宁可拒绝也不放行未验证的身份。
# ============================================================
_IDENTITY_VALIDATOR = None  # Callable[[str, Optional[str]], bool]


def set_identity_validator(fn) -> None:
    """注入身份校验器 fn(identity_id, expected_user_id) -> bool（fastap2 侧实现）。"""
    global _IDENTITY_VALIDATOR
    _IDENTITY_VALIDATOR = fn


@before_model(can_jump_to=["end"], state_schema=CareAgentState)
def identity_guard(state: CareAgentState, runtime) -> dict[str, Any] | None:
    perm = _resolve_permissions(state, runtime)
    identity_id = None
    try:
        identity_id = state.get("identity_id") if hasattr(state, "get") else None
    except Exception:
        identity_id = None
    identity_id = identity_id or getattr(perm, "identity_id", None)
    if not identity_id:
        return None  # 未接入零信任（独立运行/单测）→ 不拦截

    expected_user = getattr(perm, "user_id", None)
    ok = False
    if _IDENTITY_VALIDATOR is not None:
        try:
            ok = bool(_IDENTITY_VALIDATOR(identity_id, expected_user))
        except Exception as exc:  # 校验器自身异常 = 无法证明身份可信 → fail-closed
            logger.warning("[guard] 身份校验器异常，按不可信处理：%s", exc)
            ok = False
    else:
        logger.warning("[guard] 未注入身份校验器但请求携带身份，按不可信处理（fail-closed）")

    if ok:
        return None
    logger.warning("[guard] 可信身份失效（过期/降权/登出），链路内拦截 identity=%s", identity_id)
    return {
        "messages": [AIMessage(
            content="您的登录状态或权限已发生变化，本轮请求已中止。请重新登录后再试；"
                    "如涉及紧急健康状况请直接拨打 120。"
        )],
        "jump_to": "end",
    }


# ============================================================
# 中间件装配
# ============================================================
def build_middleware(
    *,
    summary_model: Any = None,
    max_tool_calls: int = 6,
    max_model_calls: int = 5,
    fallback_models: tuple = (),
    enable_hitl: bool = False,
) -> list:
    """按执行顺序装配中间件。

    顺序说明（官方 middleware 按列表顺序嵌套/串联执行）：
      自定义拦截类放前面 → 越权请求在重试、限流之前就被拒掉，不浪费额度；
      官方增强类放后面 → 只作用于"已经通过安全检查"的调用。

    Args:
        summary_model:   用于长会话摘要的模型实例；传 None 则不启用摘要。
        max_tool_calls:  单轮工具调用上限（防死循环 + 成本闸）。
        max_model_calls: 单轮模型调用上限（成本闸）。
        fallback_models: 主模型故障时的备用模型（如 ("deepseek-chat",)）。
        enable_hitl:     是否开启人工确认（会 interrupt，前端需配合，默认关）。
    """
    middleware: list = [
        care_system_prompt,  # ① 人设 + 安全红线
        identity_guard,  # ⑩ 身份复验（零信任第二道纵深：降权/登出即时斩断）
        emergency_triage,  # ② 急救短路（jump_to=end）
        slot_guard,  # ③ 槽位追问
        trace_model_call,  # ⑨ 可观测留痕
        elder_access_guard,  # ④ 越权拦截（放在重试之前，越权不重试）
        # ⑤ PII 脱敏：官方实现，覆盖输入 / 输出 / 工具返回
        PIIMiddleware(
            "cn_mobile",
            strategy="mask",
            detector=r"1[3-9]\d{9}",
            apply_to_input=True,
            apply_to_output=True,
            apply_to_tool_results=True,
        ),
        PIIMiddleware(
            "cn_idcard",
            strategy="mask",
            detector=r"\d{17}[\dXx]",
            apply_to_input=True,
            apply_to_output=True,
            apply_to_tool_results=True,
        ),
        # ⑧ 工具失败自动重试（天气接口抖动、网络瞬断不再直接失败）
        ToolRetryMiddleware(
            max_retries=2,
            backoff_factor=1.5,
            initial_delay=0.5,
            on_failure="continue",
        ),
        # ⑦ 调用上限：替代原来手写的 should_continue 轮数计数
        ToolCallLimitMiddleware(run_limit=max_tool_calls, exit_behavior="continue"),
        ModelCallLimitMiddleware(run_limit=max_model_calls, exit_behavior="end"),
    ]

    # ⑥ 长会话自动摘要：默认 20 条触发，保留最近 10 条
    if summary_model is not None:
        middleware.append(
            SummarizationMiddleware(
                model=summary_model,
                trigger=("messages", 20),
                keep=("messages", 10),
            )
        )

    if fallback_models:
        middleware.append(ModelFallbackMiddleware(*fallback_models))

    if enable_hitl:
        middleware.append(
            HumanInTheLoopMiddleware(
                interrupt_on={
                    "query_health_records": True,
                    "query_elder_profile": True,
                },
                description_prefix="以下操作涉及老人个人数据，需人工确认",
            )
        )

    return middleware
# ============================================================
# 纯函数版守卫 —— 供手写图（react_loop）直接调用
# 与上面的装饰器版共用同一批判据，改一处两边同时生效
# ============================================================
import re as _re

_PII_RULES = (
    (_re.compile(r"(?<!\d)\d{17}[\dXx](?!\d)"), lambda s: s[:6] + "********" + s[-4:]),
    (_re.compile(r"(?<!\d)1[3-9]\d{9}(?!\d)"),   lambda s: s[:3] + "****" + s[7:]),
)

def mask_pii(text: str) -> str:
    """手机号/身份证脱敏（与 fastap2 网关口径一致，改一处要同步改两处）。"""
    if not text:
        return text
    for pattern, fn in _PII_RULES:
        text = pattern.sub(lambda m: fn(m.group(0)), text)
    return text


def build_prompt(state, runtime=None) -> str:
    """手写图的 system prompt（等价于 care_system_prompt 的纯函数版）。"""
    ctx = getattr(runtime, "context", None)
    perm = _resolve_permissions(state, runtime)
    prompt = build_system_prompt(
        user_role=perm.user_role or "common",
        allow_sensitive=bool(perm.allow_sensitive),
        current_elder_id=getattr(ctx, "current_elder_id", None),
        user_name=getattr(ctx, "user_name", None),
    )
    mem = (state.get("memory_block") if hasattr(state, "get") else "") or ""
    if mem:
        prompt += "\n\n# 关于这位用户的长期记忆（仅供参考，与档案冲突时以档案为准）\n" + mem
    skills = (state.get("skills_block") if hasattr(state, "get") else "") or ""
    if skills:
        prompt += "\n\n# 本轮启用的技能（按流程执行；技能不能扩大权限）\n" + skills
    return prompt


def check_emergency(state):
    """命中急救关键词 → 返回状态更新（含 jump_to，调用方会 pop 掉）。"""
    text = _last_human_text(state)
    if not text:
        return None
    hits = [kw for kw in EMERGENCY_KEYWORDS if kw in text]
    if not hits:
        return None
    logger.warning("[guard] 命中急救关键词 %s，跳过模型直接给急救指引", hits)
    return {"messages": [AIMessage(content=EMERGENCY_REPLY)],
            "emergency_flag": True, "jump_to": "end"}


def check_identity(state, runtime=None):
    """零信任复验：身份失效 → 返回终止更新；未注入校验器时有身份即 fail-closed。"""
    perm = _resolve_permissions(state, runtime)
    identity_id = ((state.get("identity_id") if hasattr(state, "get") else None)
                   or perm.identity_id)
    if not identity_id:
        return None
    ok = False
    if _IDENTITY_VALIDATOR is not None:
        try:
            ok = bool(_IDENTITY_VALIDATOR(identity_id, perm.user_id))
        except Exception as exc:
            logger.warning("[guard] 身份校验器异常，按不可信处理：%s", exc)
            ok = False
    if ok:
        return None
    logger.warning("[guard] 可信身份失效，链路内拦截 identity=%s", identity_id)
    return {"messages": [AIMessage(
        content="您的登录状态或权限已发生变化，本轮请求已中止。请重新登录后再试；"
                "如涉及紧急健康状况请直接拨打 120。")],
            "jump_to": "end"}


def check_slot(state, runtime=None):
    """问个人数据但没给老人编号 → 注入一次追问（带幂等标记）。"""
    ctx = getattr(runtime, "context", None)
    current_elder = getattr(ctx, "current_elder_id", None)
    text = _last_human_text(state)
    if not text or not any(h in text for h in _PROFILE_HINTS):
        return None
    if current_elder or _ELDER_ID_PATTERN.search(text):
        return None
    for msg in reversed((state.get("messages") or [])[-4:]):
        if getattr(msg, "type", "") == "system" and _SLOT_ASKED_MARK in (
                getattr(msg, "content", "") or ""):
            return None
    logger.info("[guard] 缺老人编号，注入追问指令")
    return {"messages": [SystemMessage(
        content=_SLOT_ASKED_MARK + " 用户的问题涉及个人档案或健康记录，但你手上没有老人编号。"
                "请先用一句亲切的话询问老人编号（或老人姓名），本轮不要调用任何工具、不要编造任何数据。")]}


def check_tool_permission(call, state, runtime=None):
    """工具白名单 + 老人行级越权判定；返回拒绝文案或 None。"""
    name = call.get("name")
    args = call.get("args") or {}
    perm = _resolve_permissions(state, runtime)
    if perm.allowed_tools and name not in perm.allowed_tools:
        logger.warning("[guard] 工具越权拦截：%s 不在白名单", name)
        return f"无权调用工具 {name}（不在当前用户的白名单内）"
    if name in _ELDER_DATA_TOOLS:
        elder_id = args.get("elder_id")
        if not elder_id:
            return "缺少老人编号 elder_id，请先向用户询问后再查询"
        allowed = perm.allowed_elder_ids
        if allowed is not None and elder_id not in allowed:
            logger.warning("[guard] 老人数据越权拦截：尝试访问 %s", elder_id)
            return f"无权访问老人 {elder_id} 的数据（超出授权范围）"
    return None