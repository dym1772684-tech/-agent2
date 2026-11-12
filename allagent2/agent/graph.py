"""养老智能体图（create_agent + 官方 middleware 版）。

【改造说明 · 总览】
------------------------------------------------------------------
怎么改的：
  原实现是手写 StateGraph，五个节点 + 硬编码意图路由：
      intent_plan → route_by_intent → (check_rag | first_llm_call) ⇄ tool_use → final_output
  现在整体换成 LangChain 1.x 官方的 create_agent：
      create_agent(model, tools, system_prompt, middleware=[...],
                   response_format=ParsedOutput, state_schema=CareAgentState,
                   context_schema=config_schema, checkpointer=..., store=...)

  对应的变化：
    1. 【删除】intent_plan 节点 —— 由模型自己通过 tool_calls 决定"查档案还是
       查知识库"，不再靠一个字符串意图硬编码路由（原 route_by_intent 判错就
       永远走错分支）。
    2. 【删除】check_rag 节点 —— RAG  封装成 search_knowledge_base 工具，
       按需检索（原来 knowledge 意图必检索、其他意图永不检索，两头不讨好）。
    3. 【删除】tool_use / should_continue —— 工 具循环交给 create_agent，
       轮数上限交给 ToolCallLimitMiddleware（官方中间件）。
    4. 【删除】final_output 节点 —— 交给 response_format=ParsedOutput，
       结构化结果落在 state["structured_response"]。
    5. 【新增】middleware 管道 —— 人设注入、急救短路、槽位追问、越权拦截、
       PII 脱敏、长会话摘要、调用上限、失败重试，全部走官方 API。

改的意义：
  - 「意图路由」这类硬编码分支是链路里最脆的一环：加一个意图就要改路由函数，
    判错一次整轮答案就废了。交给模型做工具选择是现在的主流做法，扩展性更好。
  - 结构化输出由框架保证（response_format），不再依赖"最后一条 AI 消息兜底"。
  - 所有横切关注点（安全、成本、可观测）收敛到 middleware，业务节点不再掺杂
    权限判断和日志代码。

⚠️ 注意：make_graph() 现在直接返回「已编译的 agent」，调用方不要再 .compile()。
------------------------------------------------------------------
"""
import os
from pathlib import Path

from dotenv import find_dotenv, load_dotenv
from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore

from allagent2.agent.agent import create_chat_model, create_parser_model
from allagent2.agent.middleware import build_middleware
from allagent2.agent.react_loop import build_react_graph
from allagent2.agent.state import (
    CareAgentState,
    ParsedOutput,
    config_schema,
)
from allagent2.guance.logging_helper import logger
from allagent2.prompts.prompts import CARE_SYSTEM_PROMPT
from allagent2.tools.knowledge_tool import search_knowledge_base
from allagent2.tools.tool import tools

BASE_DIR = Path(__file__).resolve().parent.parent
ENV_PATH = BASE_DIR / "core" / "key.env"
load_dotenv(find_dotenv(str(ENV_PATH)))

api_key = os.getenv("OPENAI_API_KEY")
base_url = os.getenv("OPENAI_BASE_URL")

# ============================================================
# 引擎开关
# ------------------------------------------------------------
# react（默认）：手写 guard→think→act→observe 循环，每个节点写事件流，
#               日志与 /agent/trace 能看到"第几步、想了什么、token 多少、
#               下一步做什么"；
# create_agent：LangChain 官方封装，节点只有 model/tools，链路不可见，
#               仅用于出问题时一键回退（AGENT_ENGINE=create_agent）。
# ============================================================
AGENT_ENGINE = os.getenv("AGENT_ENGINE", "react").strip().lower()

# parser_model：保留给需要结构化解析的场景（如外部单独调用）
parser_model = create_parser_model(api_key, base_url)
# 【改造】裸模型：交给 create_agent，由框架绑定工具
chat_model = create_chat_model(api_key, base_url)
# 【改造】摘要专用模型实例：长会话压缩用，与对话模型隔离，避免互相污染
summary_model = create_chat_model(api_key, base_url)


def build_tool_list() -> list:
    """组装工具列表（按 name 去重）。

    【改造】bootstrap 会在 import 本模块之前，把后端只读工具
    （query_elder_profile / query_health_records）追加进 allagent2.tools.tool.tools。
    这里在函数内重新读取并合并知识库工具，保证注入顺序不影响结果。
    """
    merged: dict = {t.name: t for t in list(tools)}
    # 【MCP 改造】bootstrap 已把 MCP 版工具(含 search_knowledge_base)注入 tools;
    # 本地 knowledge_tool 仅在对应 server 未连接时兜底,不覆盖 MCP 版本
    if "search_knowledge_base" not in merged:
        merged["search_knowledge_base"] = search_knowledge_base
    return list(merged.values())


def _make_create_agent(checkpointer=None, tool_list=None):
    """官方 create_agent 版（保留作回退路径）。

    编译后节点只有 model / tools 两个，链路过程不可见——需要逐步可观测时
    用 AGENT_ENGINE=react 切到手写循环。
    """
    tool_list = tool_list if tool_list is not None else build_tool_list()
    agent = create_agent(
        model=chat_model,
        tools=tool_list,
        # 静态人设：dynamic_prompt 中间件会按上下文覆盖它，这里作为兜底基线
        system_prompt=CARE_SYSTEM_PROMPT,
        # 官方中间件管道（人设/急救/追问/越权/PII/摘要/限流/重试）
        middleware=build_middleware(summary_model=summary_model),
        # 结构化输出：结果落在 state["structured_response"]
        # 【兼容】ToolStrategy=走工具调用；默认的 ProviderStrategy 会发
        # response_format:json_schema，DeepSeek 不支持（400）。
        response_format=ToolStrategy(ParsedOutput),
        state_schema=CareAgentState,
        # 运行时上下文（当前老人 / 用户角色 / 权限）走官方 context_schema
        context_schema=config_schema,
        checkpointer=checkpointer if checkpointer is not None else InMemorySaver(),
        # 长期记忆存储：为后续沉淀老人画像 / 用户偏好预留
        store=InMemoryStore(),
        name="elder_care_agent",
    )
    logger.info("agent 编译完成（create_agent 引擎）")
    return agent


def make_graph(checkpointer=None):
    """构建并编译养老智能体。

    返回的是 CompiledStateGraph（已编译），不要再调用 .compile()。
    checkpointer:由 bootstrap 传入(Redis 持久化);缺省 InMemorySaver(命令行本地跑)。

    引擎由环境变量 AGENT_ENGINE 决定：
      react（默认）—— 手写 guard→think→act→observe 循环，每步写事件流，
                     日志与 /agent/trace 能看到"第几步、想了什么、token 多少"；
      create_agent —— 官方封装，链路不可见，仅在需要回退时使用。
    """
    tool_list = build_tool_list()
    logger.info(
        "编译 agent：engine=%s tools=%s", AGENT_ENGINE, [t.name for t in tool_list]
    )

    if AGENT_ENGINE in ("create_agent", "agent", "official"):
        return _make_create_agent(checkpointer=checkpointer, tool_list=tool_list)

    return build_react_graph(
        model=chat_model,
        tools=tool_list,
        checkpointer=checkpointer if checkpointer is not None else InMemorySaver(),
        max_rounds=int(os.getenv("AGENT_MAX_ROUNDS", "6")),
    )


if __name__ == "__main__":
    agent = make_graph()
    logger.info("agent graph 已编译完成")
    print("agent ready")
