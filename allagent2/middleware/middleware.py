"""⚠️ 本文件已废弃（保留仅作学习笔记对照），请勿再引用。

【废弃原因】
  1. 本文件用的是 `create_agent` 体系的 middleware 装饰器（@before_model /
     @wrap_tool_call / @after_model），而当时 agent/graph.py 是手写 StateGraph，
     两者不兼容 —— 这批中间件从头到尾没有一次真正被执行过。
  2. 函数体本身是半成品：delete_messages 只有一行 pass、check_rag_privacy 的
     循环体是 pass、add_ai_think 访问 state.ai_content（该字段已从 state 删除）。

【替代方案】
  全部中间件已重写在 `allagent2/agent/middleware.py`，并且链路已迁到
  create_agent（见 agent/graph.py），中间件现在真正生效：
      care_system_prompt  / emergency_triage / slot_guard / elder_access_guard
      trace_model_call    / PIIMiddleware / SummarizationMiddleware
      ToolCallLimitMiddleware / ModelCallLimitMiddleware / ToolRetryMiddleware

  其中：
  - 本文件的 trim_messages      → 由官方 SummarizationMiddleware 接管（更稳）
  - 本文件的 check_rag_privacy  → 由官方 PIIMiddleware 接管
  - 本文件的 add_ai_think       → 由 trace_model_call + /agent/trace 接管
  - 本文件的 HumanInTheLoopMiddleware → 在 build_middleware(enable_hitl=True) 中提供
"""
from langchain.agents.middleware import wrap_tool_call, SummarizationMiddleware,HumanInTheLoopMiddleware
from langchain.messages import RemoveMessage
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from langgraph.checkpoint.memory import InMemorySaver
from langchain.agents import create_agent, AgentState
from langchain.agents.middleware import before_model,after_model,hook_config
from langgraph.runtime import Runtime
from langchain.messages import HumanMessage,AIMessage
from langchain_core.runnables import RunnableConfig
from typing import Any
from allagent2.agent.state import state_schema
@before_model
def trim_messages(state: state_schema, runtime: Runtime) -> dict[str, Any] | None:
    """Keep only the last few messages to fit context window."""
    messages = state["messages"]

    if len(messages) <= 3:
        return None  # No changes needed

    first_msg = messages[0] 
    recent_messages = messages[-3:] if len(messages) % 2 == 0 else messages[-4:]
    new_messages = [first_msg] + recent_messages

    return {
        "messages": [
            RemoveMessage(id=REMOVE_ALL_MESSAGES),
            *new_messages
        ]
    }
@after_model
def delete_messages(state:state_schema,runtime:Runtime):
    messages=state["messages"]
@wrap_tool_call
def check_rag_privacy(state:state_schema):
    for i in state["messages"]:
        if isinstance(i,HumanMessage):
            pass
@after_model
def add_ai_think(state:state_schema,response):###GraphOutput
    aithink=[b for b in response.content_blocks if b["type"]=="reasoning"]
    complete_aithink="".join(step["reasoning"] for step in aithink)
    state.ai_content=state.ai_content.append(complete_aithink)#########用来充当中间件 提取每一次的aithink ，进入任何节点之后的
# before_agent	代理人开始前（每次调用一次）
# before_model	每次模型调用前
# after_model	每次模型响应后
# after_agent	代理完成后（每次调用一次）
# 每个调用都环绕着 Wrap 风格的钩子，让你控制执行：
# 胡克	当它运行时
# wrap_model_call	围绕每个模型调用
# wrap_tool_call	围绕每个工具调用


