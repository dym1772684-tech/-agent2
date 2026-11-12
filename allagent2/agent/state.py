# state.py
# 【改进】依据 LangGraph 最新文档（State 推荐用 Pydantic BaseModel，支持默认值/递归校验），
# 将 state / config / 输出结构全部重构为 Pydantic，并补齐元数据：
#   1) query（QueryPlan）：模型明确表达"此刻要查什么、查哪个老人、什么指标"——解决输出没有 query 的问题
#   2) next_need（NextNeed）：模型明确表达"下一步需要什么"（工具查询 / 用户补充信息 / 直接回答）
#   3) phase：智能体当前所处阶段（intent_plan / rag_retrieval / llm_reasoning / tool_calling / finalizing），
#      让每次 stream 都能看出"智能体现在在哪一步"
#   4) config_schema：携带会话级元数据（当前老人、用户角色、语言、工具循环上限）
# 同时移除从未被读写的装饰字段（query_time/update_time/session/diagnostic/ai_content），避免约束死板。
from __future__ import annotations

from typing import Annotated, Any, NotRequired, Optional, TypedDict

from langchain.agents import AgentState
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, ConfigDict, Field


# ============================================================
# 元数据 ①：查询计划（QueryPlan）——模型表达"此刻要查什么"
# ============================================================
class QueryPlan(BaseModel):
    intent: str = Field(
        default="general",
        description="意图分类：health_record / elder_profile / weather / knowledge / general",
    )
    elder_id: Optional[str] = Field(
        default=None,
        description="目标老人编号（涉及个人档案 / 健康记录时必填）",
    )
    metric_type: Optional[str] = Field(
        default=None,
        description="健康指标类型：血压 / 血糖 / 心率等；不填表示查全部最新记录",
    )
    question: Optional[str] = Field(
        default=None,
        description="需要检索或查询的问题原文",
    )
    keywords: Optional[list[str]] = Field(
        default=None,
        description="知识库检索关键词（intent=knowledge 时使用）",
    )


# ============================================================
# 元数据 ②：下一步需求（NextNeed）——模型表达"下一步需要什么"
# ============================================================
class NextNeed(BaseModel):
    kind: str = Field(
        description="下一步动作类型：tool_query（调工具）/ user_input（向用户要信息）/ rag_retrieval（检索知识库）/ answer（直接回答）/ finish（结束）",
    )
    description: str = Field(
        default="",
        description="下一步需要做什么（人可读描述）",
    )
    required_fields: Optional[list[str]] = Field(
        default=None,
        description="需要用户补充的信息字段（kind=user_input 时使用，如 elder_id / 指标类型）",
    )
    tool_name: Optional[str] = Field(
        default=None,
        description="计划调用的工具名（kind=tool_query 时使用）",
    )
    tool_args: Optional[dict] = Field(
        default=None,
        description="计划调用的工具参数（kind=tool_query 时使用）",
    )


# ============================================================
# 输出结构：养生养老业务字段 + 查询/下一步元数据
# ============================================================
class ParsedOutput(BaseModel):
    """最终结构化输出。

    query / next_need / answer 是"元数据三件套"：
    - query：这次对话智能体查了什么（复盘用）
    - next_need：回答完用户后，下一步还需要什么（衔接用）
    - answer：最终人话回复
    其余为业务字段：养生养老维度（健康/饮食/运动/用药/风险/紧急标志）。
    """
    query: Optional[QueryPlan] = None
    next_need: Optional[NextNeed] = None
    answer: Optional[str] = Field(default=None, description="最终回复正文")
    health_advice: Optional[str] = Field(default=None, description="养生建议（作息/情志/日常调护）")
    diet_advice: Optional[str] = Field(default=None, description="饮食建议（食疗/忌口/营养）")
    exercise_advice: Optional[str] = Field(default=None, description="运动建议（适合老人的运动方式与强度）")
    medication_advice: Optional[str] = Field(default=None, description="用药/服药提醒（仅提醒，不替代医嘱）")
    risk_level: Optional[str] = Field(default=None, description="健康风险等级：低 / 中 / 高")
    emergency_flag: Optional[bool] = Field(default=False, description="是否触发紧急就医建议（胸痛/意识模糊等）")


# ============================================================
# 元数据 ③：权限上下文（PermissionContext）——"这一步能调用什么权限"
# ============================================================
class PermissionContext(BaseModel):
    """由外部（FastAPI 集成，基于登录用户）注入到 config，再写入 state。

    每个节点执行前会按"权限标签"校验（见 graph.py _check_node_permission）：
    - intent_plan   → plan       ：只允许 LLM 解析，无工具权限
    - check_rag     → rag        ：需 allow_rag=True 才允许检索知识库
    - first_llm_call→ llm        ：只允许 LLM 决策，无执行权限
    - tool_use      → tool:<name>：工具名必须在 allowed_tools 白名单内；
                                  涉及老人数据的工具，参数 elder_id 必须在 allowed_elder_ids 内（防越权）
    - final_output  → summarize  ：只允许汇总，无工具权限
    """
    user_role: Optional[str] = "common"
    # 【零信任·S3】身份引用与 RBAC 码集合（fastap2 侧 TrustedContext 装配后传入）：
    # identity_id 指向 Redis 里的可信身份（identity_guard 据此复验 perm_version）；
    # permissions 是功能权限码集合（team supervisor 按 AGENT_PROFILES 判断任务可派性）；
    # user_id 供链路内校验属主。三者缺失时按"无身份"降级——图内逻辑必须 fail-closed。
    user_id: Optional[str] = None
    permissions: Optional[list[str]] = Field(
        default=None,
        description="功能权限码集合（如 elder:health_read）；None=未知（按无权限处理）",
    )
    identity_id: Optional[str] = Field(
        default=None,
        description="可信身份引用（identity:{id}，Redis TTL 内有效）；None=未接入零信任",
    )
    allowed_tools: Optional[list[str]] = Field(
        default_factory=lambda: [
            "calculator",
            "get_real_weather",
            "query_elder_profile",
            "query_health_records",
        ],
        description="当前用户可调用的工具白名单",
    )
    allowed_elder_ids: Optional[list[str]] = Field(
        default=None,
        description="当前用户可访问的老人编号范围；None=未限定（外部可注入受限列表）",
    )
    allow_rag: Optional[bool] = True
    allow_sensitive: Optional[bool] = Field(
        default=False,
        description="是否允许返回敏感信息（联系电话、完整病史）；默认 False 即脱敏",
    )


# ============================================================
# 会话级元数据：config_schema（LangGraph RunnableConfig 的内容）
# ============================================================
class config_schema(BaseModel):
    """进入图时的会话上下文，由外部（FastAPI 集成 / graph.py 交互入口）注入。

    【改进】在原 role/session_id/checkpoint_id 基础上新增养生养老场景元数据：
    current_elder_id（当前对话默认关联的老人）、user_name、locale、max_tool_rounds（防死循环）、
    以及 permissions（权限上下文，控制每一步能调用什么）。
    """
    role: Optional[str] = "common"
    session_id: Optional[str] = None
    checkpoint_id: Optional[str] = None
    current_elder_id: Optional[str] = Field(
        default=None,
        description="当前对话默认关联的老人编号（后端鉴权后可自动注入）",
    )
    user_name: Optional[str] = Field(default=None, description="当前用户姓名")
    locale: Optional[str] = "zh-CN"
    max_tool_rounds: Optional[int] = Field(
        default=3,
        description="工具调用最大轮数，超过则强制收尾，防止无限循环",
    )
    permissions: Optional[PermissionContext] = Field(
        default=None,
        description="权限上下文；为 None 时在图内回退为默认权限（见 graph.py）",
    )


# ============================================================
# 图状态：state_schema
# ============================================================
class state_schema(BaseModel):
    """LangGraph 图状态（Pydantic 版）。

    【改进】
    - messages：对话消息（add_messages reducer，自动追加）
    - query / intent / next_need / phase：状态元数据——"此刻智能体在哪一步、要什么、下一步要做什么"
    - rag_context：知识库检索结果（独立字段，不污染 messages，避免多条 SystemMessage 堆叠）
    - tool_results：工具查询结果统一回填（key=工具名，value=返回文本）
    - 业务字段：养生养老维度（健康/饮食/运动/用药/风险/紧急标志）
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)

    messages: Annotated[list[BaseMessage], add_messages]
    query: Optional[QueryPlan] = None
    intent: Optional[str] = None
    phase: Optional[str] = Field(
        default=None,
        description="当前阶段：intent_plan / rag_retrieval / llm_reasoning / tool_calling / finalizing",
    )
    next_need: Optional[NextNeed] = None
    rag_context: Optional[str] = None
    tool_results: Optional[dict[str, Any]] = None
    permissions: Optional[PermissionContext] = Field(
        default=None,
        description="权限上下文：每个节点按自己的权限标签校验（graph.py _check_node_permission）",
    )

    # 养生养老业务字段（已删除原旅游字段）
    health_advice: Optional[str] = None
    diet_advice: Optional[str] = None
    exercise_advice: Optional[str] = None
    medication_advice: Optional[str] = None
    risk_level: Optional[str] = None
    emergency_flag: Optional[bool] = False


# 图的最终输出形状：直接使用 ParsedOutput（query + answer + 业务字段）
output_schema = ParsedOutput

# 供 trace / 结果读取的业务字段清单（与 ParsedOutput 对齐，仅养生养老维度）
PARSED_FIELDS = [
    "answer",
    "health_advice",
    "diet_advice",
    "exercise_advice",
    "medication_advice",
    "risk_level",
    "emergency_flag",
]


# ============================================================
# 【改造】create_agent 兼容状态：CareAgentState
# -----------------------------------------------------------
# 怎么改的：
#   原 state_schema 是 Pydantic BaseModel，只能配手写 StateGraph。
#   LangChain 1.x 的 create_agent 要求 state_schema 必须是
#   langchain.agents.AgentState 的子类（TypedDict 语义），否则
#   middleware / response_format / structured_response 都挂不上。
#   这里新增一个继承 AgentState 的 TypedDict，把权限上下文等元数据
#   作为「可选字段」带进去，原 state_schema / ParsedOutput 全部保留。
#
# 改的意义：
#   1) 这是本轮改造的前提：只有换成 AgentState 子类，才能用官方
#      middleware 体系（dynamic_prompt / before_model / wrap_tool_call）；
#   2) messages 交给 AgentState 自带的 add_messages reducer 管理，
#      删掉自己维护消息列表的代码，减少出错面；
#   3) permissions 留在 state 里并随 checkpoint 持久化，middleware
#      每次执行都能读到，权限校验不会因为换线程/换节点而丢失；
#   4) structured_response 由 create_agent 在 response_format=ParsedOutput
#      时自动写入 —— 这就是「接住结构化输出」的落点（原实现把 invoke
#      返回值整个丢了，只能靠最后一条 AI 消息兜底）。
# ============================================================
class CareAgentState(AgentState):
    """LangChain `create_agent` 使用的图状态（TypedDict 语义）。"""

    # 权限上下文：middleware 做越权拦截时读取（None = 走默认，见 _get_permissions）
    permissions: NotRequired[Optional[PermissionContext]]
    # 【零信任·S3】可信身份引用：identity_guard 每次模型调用前复验（空/None=未接入，跳过）
    identity_id: NotRequired[Optional[str]]
    # 意图：由 LLM 自主决策后不再强制路由，仅作 trace/复盘用
    intent: NotRequired[Optional[str]]
    # 急救标志：命中急救规则时置 True，供外部与审计识别
    emergency_flag: NotRequired[bool]
    # 本轮是否命中知识库检索（供可观测性统计）
    rag_used: NotRequired[bool]
    # 【记忆系统】链路一召回的记忆注入块（bootstrap._prepare 填充；
    # dynamic_prompt 拼进 system prompt / team 图 supervisor 读取；空串=无记忆）
    memory_block: NotRequired[str]
    # 【技能系统】本轮启用技能渲染出的 prompt 块（bootstrap._prepare 填充；
    # 空串=没有启用任何技能）。技能只能收窄工具集，不能扩权。
    skills_block: NotRequired[str]
