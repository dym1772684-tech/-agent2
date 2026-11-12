"""MCP 客户端运行时:后台事件循环 + 常驻会话 + 同步工具包装。

【为什么这么设计】三个现实约束:
1. create_agent 的工具节点会在 worker 线程里同步 invoke 工具 → 必须提供同步入口;
2. MCP 连接是 async 且必须常驻(不能每次调用重连)→ 独立后台事件循环线程(daemon);
3. 权限上下文(contextvars)必须随每次调用跨进程传给 MCP Server →
   同步包装器在调用前从 contextvars 读上下文、用 HMAC 签成 agent_context 塞进参数。

生命周期:bootstrap 启动时 connect_servers() 一次,连接保持到进程退出。
"""
import asyncio
import json
import logging
import threading
from typing import Any, Dict, List, Optional

from langchain_core.tools import StructuredTool
from pydantic import Field, create_model

from core.agent_context import get_access_context, make_context_token

logger = logging.getLogger("mcp_runtime")
if not logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s [%(name)s] %(message)s",
                                      datefmt="%Y-%m-%d %H:%M:%S"))
    logger.addHandler(_h)
    logger.setLevel(logging.INFO)
    logger.propagate = False

_LOOP: Optional[asyncio.AbstractEventLoop] = None
_SESSIONS: Dict[str, Any] = {}   # server_name -> ClientSession(常驻)
_CMS: Dict[str, Any] = {}        # server_name -> streamablehttp_client 上下文管理器
                                 # 【关键】必须持有引用:被 GC 时其 cancel scope 会被
                                 # 在错误的任务里退出,炸掉后台事件循环
_TOOLS: List[StructuredTool] = []

_CALL_TIMEOUT = 60      # 单次工具调用超时(秒)
_CONNECT_TIMEOUT = 10   # 连接超时(秒)

_TYPE_MAP = {"string": str, "number": float, "integer": int, "boolean": bool}


def _start_loop() -> asyncio.AbstractEventLoop:
    global _LOOP
    if _LOOP is None or _LOOP.is_closed():
        _LOOP = asyncio.new_event_loop()
        threading.Thread(target=_LOOP.run_forever, daemon=True, name="mcp-loop").start()
    return _LOOP


def _schema_to_pydantic(tool_name: str, schema: dict):
    """把服务端 JSON Schema 转成 pydantic 模型(供 LangChain 工具参数校验)。

    agent_context 由运行时注入,从对外 schema 中剔除 —— LLM 看不到它。
    """
    props = dict((schema or {}).get("properties") or {})
    required = set((schema or {}).get("required") or [])
    props.pop("agent_context", None)
    fields: Dict[str, Any] = {}
    for name, spec in props.items():
        py_t = _TYPE_MAP.get((spec or {}).get("type"), str)
        desc = (spec or {}).get("description", "")
        if name in required:
            fields[name] = (py_t, Field(description=desc))
        else:
            fields[name] = (Optional[py_t], Field(default=None, description=desc))
    return create_model(f"{tool_name}_args", **fields)


def _wrap_tool(server_name: str, mcp_tool: Any) -> StructuredTool:
    """把 MCP 远端工具包装成 LangChain 同步 StructuredTool。

    调用链:LLM 发起 tool_call → 同步 func → 后台事件循环上的 async 会话
    → 注入 agent_context 令牌 → MCP Server(验签+ACL+审计)。
    """

    async def _acall(token: str, **kwargs) -> str:
        # 【修复】token 由同步 _call 在"工具执行线程"里生成后传入。
        # 原实现在这里(后台事件循环线程)调 get_access_context(),但
        # asyncio.run_coroutine_threadsafe 不传播调用方线程的 contextvars,
        # 导致鉴权上下文永远读不到 → elder 工具 100% 被 fail-closed 拒掉。
        if token:
            kwargs["agent_context"] = token
        # 无令牌:不注入 → 服务端 fail-closed 返回 403(对模型可见)
        result = await _SESSIONS[server_name].call_tool(mcp_tool.name, kwargs)
        parts = []
        for c in (result.content or []):
            text = getattr(c, "text", None)
            if text:
                parts.append(text)
        return "\n".join(parts) or "(工具返回空)"

    def _call(**kwargs) -> str:
        # 在调用方线程(langgraph executor 已 copy_context)读取鉴权上下文并签名,
        # 再把令牌作为普通参数跨线程传给后台循环 —— 绕开 contextvars 不跨线程的问题。
        ctx = get_access_context()
        token = make_context_token(ctx) if getattr(ctx, "authenticated", False) else ""
        fut = asyncio.run_coroutine_threadsafe(_acall(token, **kwargs), _LOOP)
        return fut.result(timeout=_CALL_TIMEOUT)

    return StructuredTool(
        name=mcp_tool.name,
        description=mcp_tool.description or "",
        args_schema=_schema_to_pydantic(mcp_tool.name, mcp_tool.inputSchema),
        func=_call,
    )


def connect_servers(urls: Dict[str, str]) -> None:
    """连接所有 MCP Server 并拉取工具清单(阻塞,启动时调用一次)。

    某个服务连接失败 → 跳过该服务的工具并告警(该类能力自然不可用 = fail-closed),
    不阻断整个 agent 启动。
    """
    loop = _start_loop()
    for name, url in urls.items():
        async def _do(url=url, name=name):
            from mcp import ClientSession
            from mcp.client.streamable_http import streamablehttp_client

            cm = streamablehttp_client(url)
            read, write, _ = await cm.__aenter__()
            _CMS[name] = cm  # 【保引用】见 _CMS 注释
            session = ClientSession(read, write)
            await session.__aenter__()
            await session.initialize()
            listed = await session.list_tools()
            _SESSIONS[name] = session
            for t in listed.tools:
                _TOOLS.append(_wrap_tool(name, t))
            return [t.name for t in listed.tools]

        fut = asyncio.run_coroutine_threadsafe(_do(), loop)
        try:
            tool_names = fut.result(timeout=_CONNECT_TIMEOUT)
            logger.info("MCP[%s] 已连接 %s,tools=%s", name, url, tool_names)
        except Exception as e:
            try:  # 【降级计数】MCP 不可用是最常见的静默降级来源
                from observability.metrics import note_degradation

                note_degradation("mcp", type(e).__name__, f"{name}={url}")
            except Exception:
                pass
            logger.warning("MCP[%s] 连接失败(%s),该服务工具不可用:%s", name, url, e)


def get_mcp_tools() -> List[StructuredTool]:
    return list(_TOOLS)
