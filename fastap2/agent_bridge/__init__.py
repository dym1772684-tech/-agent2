"""agent_bridge：把 allagent2（LangGraph 智能体）接入 fastap2（FastAPI 后端）的集成层（胶水层）。

导入该包即完成接线：加载配置 -> 注册 allagent2 包 -> 注入后端工具 -> 编译 graph。
上述动作均为惰性、不在导入期联网；真正的模型推理在每次请求时发生。
"""
from .bootstrap import (
    AGENT_DIR,
    FASTAP2_DIR,
    ABORT_KEY_PREFIX,
    bootstrap,
    clear_abort,
    get_status,
    get_trace,
    resume_agent,
    run_agent,
    run_agent_stream,
    set_abort,
)

# 导入该包即完成一次接线（幂等）

__all__ = ["bootstrap", "run_agent", "run_agent_stream", "resume_agent",
           "set_abort", "clear_abort", "ABORT_KEY_PREFIX",
           "get_status", "get_trace",
           "AGENT_DIR", "FASTAP2_DIR"]
