"""请求级追踪 ID（request_id）—— 让一条请求的所有日志能串起来。

============================================================
为什么要有
------------------------------------------------------------
一条 /chat 请求的日志会分散在 HTTP 入口、线程池、LangGraph 节点、MCP
子进程、SSE 出口五处。没有关联 ID 时，并发一高日志就"串味"，
只能靠 thread_id 手工拼，而 thread_id 在多轮会话里重复出现、无法定位单次请求。

============================================================
实现要点
------------------------------------------------------------
1. **contextvar**：一次请求一个值，asyncio 任务与 anyio/starlette 的
   run_in_threadpool 都会自动复制上下文，不需要手动传参；
2. **logging.Filter**：所有 handler 挂一个 Filter，把当前 rid 塞进
   LogRecord，formatter 就能用 `%(request_id)s`；
3. **重写 formatter**：已有的 handler 是 import 期写死的纯文本格式，
   只加 Filter 不会显示 rid，所以 install 时必须把格式换掉（幂等）；
4. **兜底 "-"**：后台线程（线程池复用、MCP 后台 loop）里没有请求上下文，
   rid 会是 "-"，这是预期行为，不报错。

为什么不用 `logging.setLogRecordFactory`：会和第三方库/uvicorn 自己的
factory 相互覆盖，出问题极难排查。Filter 是每个 handler 独立的，安全。
"""
from __future__ import annotations

import contextvars
import logging
import uuid
from typing import Any, Iterable, Optional

_RID: contextvars.ContextVar[str] = contextvars.ContextVar("request_id", default="-")
_MISSING = "-"


class RequestIdFilter(logging.Filter):
    """把当前请求 ID 写进 LogRecord（已有值则不覆盖，便于子线程显式继承）。"""

    def filter(self, record: logging.LogRecord) -> bool:  # noqa: A003
        if not getattr(record, "request_id", None):
            record.request_id = _RID.get() or _MISSING
        return True


def new_request_id(prefix: str = "r") -> str:
    return f"{prefix}-{uuid.uuid4().hex[:12]}"


def get_request_id() -> str:
    return _RID.get() or _MISSING


def set_request_id(value: str) -> contextvars.Token:
    return _RID.set(value or _MISSING)


def reset_request_id(token: contextvars.Token) -> None:
    try:
        _RID.reset(token)
    except Exception:
        _RID.set(_MISSING)


# ============================================================
# 日志注入
# ============================================================
def _iter_handlers() -> Iterable[logging.Handler]:
    """遍历 root + 所有具名 logger 的 handler（去重）。"""
    seen: set[int] = set()
    loggers = [logging.getLogger()]
    for obj in list(logging.Logger.manager.loggerDict.values()):
        if isinstance(obj, logging.Logger):
            loggers.append(obj)
    for lg in loggers:
        for handler in list(getattr(lg, "handlers", []) or []):
            if id(handler) in seen:
                continue
            seen.add(id(handler))
            yield handler


def _build_like(
    original: logging.Formatter, fmt: str, datefmt: Optional[str]
) -> logging.Formatter:
    """用「原 formatter 的同一个类」重建格式串。

    【为什么不能直接用 logging.Formatter】**uvicorn 的 DefaultFormatter /
    AccessFormatter 是在 formatMessage 里动态往 record 上塞
    %(levelprefix)s / %(client_addr)s / %(status_code)s 等自定义字段的**。
    一旦换成标准 logging.Formatter，这些占位符就找不到值，每条日志都抛
    KeyError: 'levelprefix'，终端刷满 "--- Logging error ---"。
    所以必须保留原类，只替换格式串。
    """
    cls = type(original)
    if cls is not logging.Formatter:
        try:
            return cls(fmt, datefmt=datefmt)
        except Exception:
            pass
    try:
        return logging.Formatter(fmt, datefmt=datefmt)
    except Exception:
        return original


def _rewrite_formatter(original: Optional[logging.Formatter]) -> logging.Formatter:
    """在原格式里插入 rid 字段，保留时间格式与其余字段。"""
    if original is None:
        return logging.Formatter(
            "%(asctime)s %(levelname)s [%(name)s] rid=%(request_id)s %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    # 只处理 % 风格（标准 logging 和 uvicorn 都是）；{} / $ 风格不碰，避免拼坏
    style = getattr(original, "_style", None)
    if style is not None and type(style) is not logging.PercentStyle:
        return original

    datefmt = getattr(original, "datefmt", None)
    fmt = getattr(original, "_fmt", None)
    if not fmt:
        fmt = "%(asctime)s %(levelname)s [%(name)s] rid=%(request_id)s %(message)s"
        return _build_like(original, fmt, datefmt or "%Y-%m-%d %H:%M:%S")
    if "request_id" in fmt:
        return original  # 已经是新格式
    if "%(message)s" in fmt:
        new_fmt = fmt.replace("%(message)s", "rid=%(request_id)s %(message)s", 1)
    else:
        new_fmt = fmt + " rid=%(request_id)s"
    return _build_like(original, new_fmt, datefmt)


def install_logging() -> int:
    """给所有 handler 挂 Filter 并重写 formatter。返回处理的 handler 数。

    幂等：重复调用只处理尚未安装过的 handler。
    """
    count = 0
    for handler in _iter_handlers():
        if getattr(handler, "_request_id_ready", False):
            continue
        try:
            handler.addFilter(RequestIdFilter())
            handler.setFormatter(_rewrite_formatter(handler.formatter))
            handler._request_id_ready = True  # type: ignore[attr-defined]
            count += 1
        except Exception:
            continue
    logging.getLogger("agent_bridge").info(
        "[observability] request_id 日志注入完成：handlers=%d", count
    )
    return count


def bind_to_record(record: logging.LogRecord, request_id: Any = None) -> logging.LogRecord:
    """给单条 record 手动绑定 rid（后台线程补日志时可显式用）。"""
    if not getattr(record, "request_id", None):
        record.request_id = str(request_id) if request_id else get_request_id()
    return record
