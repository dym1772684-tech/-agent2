"""事件级可观测：把智能体每一步拆成可读的事件流。

============================================================
目标形态（对齐常见 agent 事件流）
------------------------------------------------------------
    [event 1] node_start  guard
    [event 2] node_end    guard 3ms | 下一步：进入思考
    [event 3] node_start  think · 第1轮
    [event 4] llm_start   messages=3 prompt_est=1180
    [event 5] llm_end     1240ms | token in=1280(cache 800/miss 480) out=86 total=1366 | cost=¥0.000316
    [event 5]   thinking: 用户问天气，需要先查实时天气…
    [event 5]   text: 我先帮你查一下天气。
    [event 5]   tool_call: get_real_weather({"city":"北京"})
    [event 6] node_end    think 1245ms | 下一步：调用工具 get_real_weather(city=北京)
    [event 7] node_start  act · 第1轮
    [event 8] tool_start  get_real_weather {"city":"北京"}
    [event 9] tool_end    get_real_weather 420ms isError=false | 北京 晴 24℃
    [event 10] node_end   act 425ms | 下一步：把工具结果交回模型
    ...

  - 每个事件是一个 plain dict，按顺序 append 到 `state["trace"]`
    （operator.add 归约，随 checkpoint 落 Redis，`/agent/trace` 直读）；
  - 用 dict 而不是 pydantic：RedisSaver 的 JsonPlusRedisSerializer
    白名单只认精确符号路径，用模型每加一个就要改白名单；
  - 日志一行一个事件，缩进表示从属关系，肉眼能看出"一步步走"。

============================================================
事件类型
------------------------------------------------------------
  node_start / node_end   节点进出（含耗时、下一步动作）
  llm_start / llm_end     模型调用（含 usage、cost、thinking、text、tool_calls）
  tool_start / tool_end   工具执行（含参数、耗时、isError、结果预览）
  guard                   规则命中（急救/身份/槽位）
  structured_end          结构化收尾（成功/降级）
  error                   节点异常（记录后仍抛出，让上层兜底）
============================================================
"""
from __future__ import annotations

import contextvars
import functools
import inspect
import json
import os
import threading
import time
from collections import deque
from typing import Any, Callable, Dict, List, Optional

from allagent2.guance.logging_helper import logger
from allagent2.guance.metrics_hook import meter
from allagent2.guance.token_accounting import cost_of, extract_usage

_SEP = "─" * 68
_PREVIEW_LIMIT = 60

# 旁路事故记录（不进 checkpoint，只落内存 + JSONL）
_LOG_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "guance", "logs"
)
_INCIDENT_LOG = os.path.join(_LOG_DIR, "trace_incidents.jsonl")
_INCIDENTS: Dict[str, deque] = {}
_INCIDENT_MAX = 20          # 每个 thread 最多保留多少条旁路事件


class AgentInterrupted(Exception):
    """用户主动中断：链路停在上一快照，可 resume 续跑（不是错误）。"""


# ============================================================
# 文本 / 属性小工具
# ============================================================
def preview(value: Any, limit: int = _PREVIEW_LIMIT) -> str:
    """压成一行并截断——日志里不要出现换行，否则一个事件会占很多行。"""
    text = "" if value is None else str(value)
    text = " ".join(text.split())
    return text if len(text) <= limit else text[:limit] + "…"


def attr(obj: Any, name: str, default: Any = None) -> Any:
    """同时兼容 pydantic 对象与 dict（兜底模型会返回 dict）。"""
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def message_text(msg: Any) -> str:
    """取一条消息的纯文本（content 可能是字符串，也可能是 content blocks）。"""
    content = attr(msg, "content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                if block.get("type") == "text":
                    parts.append(str(block.get("text", "")))
        return "".join(parts)
    return str(content or "")


def message_thinking(msg: Any) -> str:
    """取模型的思考内容（DeepSeek / Qwen 的 reasoning_content）。

    不同兼容端点落点不同，这里按常见位置依次尝试：
      additional_kwargs.reasoning_content → response_metadata.reasoning_content
      → content blocks 里的 thinking 块
    """
    for holder in ("additional_kwargs", "response_metadata"):
        data = attr(msg, holder, None) or {}
        if isinstance(data, dict):
            for key in ("reasoning_content", "reasoning", "thinking"):
                value = data.get(key)
                if isinstance(value, str) and value.strip():
                    return value.strip()

    content = attr(msg, "content", None)
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") in ("thinking", "reasoning"):
                parts.append(str(block.get("thinking") or block.get("text") or ""))
        if parts:
            return "".join(parts).strip()
    return ""


def tool_calls_of(msg: Any) -> List[Dict[str, Any]]:
    calls = attr(msg, "tool_calls", None) or []
    return [c for c in calls if isinstance(c, dict) and c.get("name")]


def token_line(tokens: Optional[Dict[str, int]]) -> str:
    if not tokens:
        return "token 未返回（本地模型/兼容端点可能不带 usage）"
    return (
        f"token in={tokens.get('input_tokens', 0)}"
        f"(cache {tokens.get('cache_read', 0)} / miss {tokens.get('cache_miss', 0)})"
        f" out={tokens.get('output_tokens', 0)}"
        f" total={tokens.get('total_tokens', 0)}"
    )


# ============================================================
# 会话标识 / 中断开关
# ============================================================
def thread_id_of(runtime: Any) -> str:
    """从节点运行时取 thread_id（中断标记与旁路事故都按会话隔离）。

    LangGraph 的 Runtime.execution_info.thread_id 是权威来源
    （已验证 ExecutionInfo 带 thread_id 字段）；context / config 作兜底，
    全都拿不到时返回 "-"（不参与隔离，避免把事故串到别的会话）。
    """
    info = getattr(runtime, "execution_info", None)
    tid = getattr(info, "thread_id", None) if info is not None else None
    if tid:
        return str(tid)

    ctx = getattr(runtime, "context", None)
    for name in ("thread_id", "session_id"):
        value = getattr(ctx, name, None) if ctx is not None else None
        if value:
            return str(value)

    cfg = getattr(runtime, "config", None)
    if isinstance(cfg, dict):
        value = (cfg.get("configurable") or {}).get("thread_id")
        if value:
            return str(value)
    return "-"


_ABORT_CHECK: Optional[Callable[..., bool]] = None
# 注入的回调是否接受第二个参数（node）：支持 fn(tid) 与 fn(tid, node) 两种签名，
# 后者让调用方可以「只在某个节点前中断」（测试与精细风控都要用）。
_ABORT_TAKES_NODE = False


def set_abort_check(fn: Optional[Callable[..., bool]]) -> None:
    """注入"该会话是否已被要求中断"的判定（fastap2 侧读 Redis 标记）。

    与 identity_validator 同一套路：allagent2 不直接依赖 Redis / 不反向依赖
    fastap2；未注入时中断功能自然关闭，不影响主链路。

    回调签名两种都支持：
      fn(thread_id)          只管会话（生产注入的就是这种，一个 Redis 标记）
      fn(thread_id, node)    连节点一起判定（测试里要精确断在第 N 个节点前）
    """
    global _ABORT_CHECK, _ABORT_TAKES_NODE
    _ABORT_CHECK = fn
    _ABORT_TAKES_NODE = False
    if fn is not None:
        try:
            params = [
                p
                for p in inspect.signature(fn).parameters.values()
                if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
            ]
            _ABORT_TAKES_NODE = len(params) >= 2
        except (TypeError, ValueError):  # 内建函数 / 无法取签名 → 按单参处理
            pass


# ============================================================
# 实时事件订阅：把事件推给 SSE / 前端思考面板
# ------------------------------------------------------------
# 与 identity_validator / abort_check 同一套路：allagent2 不认识 Redis、
# WebSocket、SSE，只暴露一个订阅函数，谁需要谁注册进来。
# 按 thread_id 隔离 —— 并发多个会话时不会串台。
# ============================================================
_EVENT_SINKS: Dict[str, Callable[[Dict[str, Any]], None]] = {}
_SINK_LOCK = threading.Lock()


def register_event_sink(thread_id: str, fn: Callable[[Dict[str, Any]], None]) -> None:
    """注册某会话的实时事件订阅（同一线程后注册的覆盖前一个）。"""
    if not thread_id:
        return
    with _SINK_LOCK:
        _EVENT_SINKS[thread_id] = fn


def unregister_event_sink(thread_id: str) -> None:
    """注销订阅；必须放在 finally 里，否则线程对象会一直留在表里。"""
    with _SINK_LOCK:
        _EVENT_SINKS.pop(thread_id, None)


def _dispatch_events(thread_id: str, events: List[Dict[str, Any]]) -> None:
    """把事件分发给订阅者；订阅方异常不影响主链路。"""
    if not thread_id or thread_id == "-" or not events:
        return
    with _SINK_LOCK:
        fn = _EVENT_SINKS.get(thread_id)
    if fn is None:
        return
    for event in events:
        try:
            fn(event)
        except Exception:  # 订阅方自己的问题，不能反向影响图执行
            pass


def should_abort(thread_id: str, node: Optional[str] = None) -> bool:
    if _ABORT_CHECK is None or not thread_id or thread_id == "-":
        return False
    try:
        if _ABORT_TAKES_NODE:
            return bool(_ABORT_CHECK(thread_id, node))
        return bool(_ABORT_CHECK(thread_id))
    except Exception:  # 判定失败 → 不中断（可用性优先）
        return False


# ============================================================
# 旁路事故（node error / interrupt）—— 补进 /agent/trace
# ============================================================
def _incident_bucket(thread_id: str) -> deque:
    bucket = _INCIDENTS.get(thread_id)
    if bucket is None:
        bucket = deque(maxlen=_INCIDENT_MAX)
        _INCIDENTS[thread_id] = bucket
    return bucket


def record_incident(thread_id: str, events: List[Dict[str, Any]]) -> None:
    """把"进不了 state 的事件"留痕：内存（本进程立刻可读）+ JSONL（重启可查）。

    为什么需要：节点抛异常时 LangGraph 丢弃节点返回值，error 事件永远进不了
    state["trace"]，于是 /agent/trace 里"炸的那一次"是空白的。中断同理。
    """
    if not events:
        return
    bucket = _incident_bucket(thread_id or "-")
    for event in events:
        bucket.append(event)
    try:
        os.makedirs(_LOG_DIR, exist_ok=True)
        with open(_INCIDENT_LOG, "a", encoding="utf-8") as f:
            for event in events:
                f.write(
                    json.dumps({"thread_id": thread_id, **event}, ensure_ascii=False)
                    + "\n"
                )
    except Exception as exc:  # 留痕失败不能影响主链路
        logger.warning("[trace] 事故记录写入失败：%s", exc)


def read_incidents(thread_id: str) -> List[Dict[str, Any]]:
    """读某会话的旁路事件（内存优先；进程重启后从 JSONL 恢复）。"""
    got = list(_INCIDENTS.get(thread_id) or [])
    if got:
        return got
    if not os.path.exists(_INCIDENT_LOG):
        return []
    out: List[Dict[str, Any]] = []
    try:
        with open(_INCIDENT_LOG, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                except Exception:
                    continue
                if item.get("thread_id") == thread_id:
                    item.pop("thread_id", None)
                    out.append(item)
    except Exception as exc:
        logger.warning("[trace] 事故记录读取失败：%s", exc)
        return []
    return out[-_INCIDENT_MAX:]


# ============================================================
# 事件构造与日志
# ============================================================
def make_event(
    kind: str,
    *,
    node: str = "",
    round_no: int = 0,
    duration_ms: float = 0.0,
    **fields: Any,
) -> Dict[str, Any]:
    """构造一个事件（seq 由 traced_node 统一编号）。

    注意参数名用 round_no 而不是 round：后者会遮蔽内置 round()，
    之前就在这里踩过 TypeError。
    """
    event: Dict[str, Any] = {
        "type": kind,
        "node": node,
        "round": round_no,
        "ts": round(time.time(), 3),
    }
    if duration_ms:
        event["duration_ms"] = round(float(duration_ms), 1)
    event.update({k: v for k, v in fields.items() if v not in (None, "", [], {})})
    return event


def _meter_event(event: Dict[str, Any]) -> None:
    """事件 → 指标（唯一埋点处，覆盖全部事件类型）。

    meter 在未注入 sink 时是空操作，所以缺 prometheus-client / 单独跑
    allagent2 都不会有任何副作用。
    """
    kind = event.get("type")
    try:
        if kind == "node_end":
            node = event.get("node") or "-"
            meter("agent_node_seconds",
                  float(event.get("duration_ms") or 0) / 1000.0, node=node)
            meter("agent_node_total", 1, node=node)
        elif kind == "llm_end":
            meter("agent_llm_seconds", float(event.get("duration_ms") or 0) / 1000.0)
            tokens = event.get("tokens") or {}
            for label, key in (("input", "input_tokens"),
                               ("output", "output_tokens"),
                               ("cache_read", "cache_read")):
                value = int(tokens.get(key) or 0)
                if value:
                    meter("agent_llm_tokens_total", value, kind=label)
            cost = float(event.get("cost") or 0)
            if cost:
                meter("agent_llm_cost_yuan_total", cost)
        elif kind == "tool_end":
            name = event.get("name") or "-"
            meter("agent_tool_seconds",
                  float(event.get("duration_ms") or 0) / 1000.0, tool=name)
            meter("agent_tool_total", 1, tool=name,
                  status="error" if event.get("is_error") else "ok")
        elif kind == "guard":
            meter("agent_guard_total", 1, rule=event.get("rule") or "-")
        elif kind == "error":
            meter("agent_node_errors_total", 1, node=event.get("node") or "-")
        elif kind == "interrupted":
            meter("agent_interrupt_total", 1, node=event.get("node") or "-")
        elif kind == "structured_end":
            if event.get("degraded"):
                meter("agent_structured_degraded_total", 1)
    except Exception:  # 指标失败绝不影响主链路
        pass


def emit_event(event: Dict[str, Any]) -> None:
    """一个事件 = 一行日志 + 一组指标。"""
    _log_event(event)
    _meter_event(event)


def _log_event(event: Dict[str, Any]) -> None:
    """一个事件一行日志；子信息缩进，肉眼能看出层级。"""
    seq = event.get("seq", "?")
    kind = event.get("type")
    node = event.get("node") or ""

    if kind == "node_start":
        round_tag = f" · 第{event['round']}轮" if event.get("round") else ""
        logger.info("%s [event %s] %s 开始%s", _SEP, seq, node, round_tag)
        return

    if kind == "node_end":
        logger.info(
            "[event %s] %s 结束 %.0fms | 下一步：%s",
            seq,
            node,
            event.get("duration_ms", 0),
            event.get("next_action") or "-",
        )
        return

    if kind == "llm_start":
        logger.info(
            "[event %s] llm_start  messages=%s prompt_est=%s",
            seq,
            event.get("messages", "-"),
            event.get("prompt_est_tokens", "-"),
        )
        return

    if kind == "llm_end":
        logger.info(
            "[event %s] llm_end    %.0fms | %s | cost=¥%.6f",
            seq,
            event.get("duration_ms", 0),
            token_line(event.get("tokens")),
            float(event.get("cost", 0) or 0),
        )
        if event.get("thinking"):
            logger.info("[event %s]   thinking: %s", seq, event["thinking"])
        if event.get("text"):
            logger.info("[event %s]   text: %s", seq, event["text"])
        for call in event.get("tool_calls") or []:
            logger.info("[event %s]   tool_call: %s(%s)", seq, call.get("name"), call.get("args"))
        return

    if kind == "tool_start":
        logger.info("[event %s] tool_start %s %s", seq, event.get("name"), event.get("args"))
        return

    if kind == "tool_end":
        logger.info(
            "[event %s] tool_end   %s %.0fms isError=%s | %s",
            seq,
            event.get("name"),
            event.get("duration_ms", 0),
            event.get("is_error"),
            event.get("content") or "-",
        )
        return

    if kind == "structured_end":
        logger.info(
            "[event %s] structured_end ok=%s%s | answer_len=%s",
            seq,
            event.get("ok"),
            f" | {event['error']}" if event.get("error") else "",
            event.get("answer_len", 0),
        )
        return

    if kind == "guard":
        logger.info("[event %s] guard       命中规则 %s", seq, event.get("rule"))
        return

    if kind == "observe":
        logger.info(
            "[event %s] observe     %s（成功 %s / 异常 %s）",
            seq,
            event.get("summary"),
            event.get("ok_count", 0),
            event.get("failed_count", 0),
        )
        return

    if kind == "interrupted":
        logger.warning(
            "%s [event %s] %s 已中断 | %s（快照保留，可 resume 续跑）",
            _SEP,
            seq,
            node,
            event.get("reason") or "用户中断",
        )
        return

    if kind == "error":
        logger.error(
            "[event %s] %s 异常 %.0fms | %s",
            seq,
            node,
            event.get("duration_ms", 0),
            event.get("error"),
        )
        return

    logger.info("[event %s] %s %s", seq, kind, preview(event, 120))


# ============================================================
# 节点装饰器：node_start → 节点自定义事件 → node_end
# ============================================================
def _max_seq(trace: List[Dict[str, Any]]) -> int:
    seqs = [e.get("seq", 0) for e in trace if isinstance(e, dict)]
    return max(seqs) if seqs else 0


def _incident_seq(thread_id: str) -> int:
    """旁路事故里的 seq 水位。

    中断/异常事件进不了 state["trace"]，但它们的 seq 已经被用掉了。
    不算进来会导致续跑时 seq 回到旧值、和旁路事件撞号（报告里会看到两个 27）。
    """
    if not thread_id or thread_id == "-":
        return 0
    bucket = _INCIDENTS.get(thread_id) or []
    seqs = [int(e.get("seq") or 0) for e in bucket if isinstance(e, dict)]
    return max(seqs) if seqs else 0


def traced_node(node: str) -> Callable[[Callable], Callable]:
    """给节点套一层事件流。

    节点返回值约定：
      `_events`  本步内部产生的事件列表（llm_start/llm_end/tool_start/...）
      `_end`     node_end 的附加信息（next_action / thought / detail）
    两个键都会被消费掉，不会污染图状态。

    进入节点前先检查中断标记：命中即抛 AgentInterrupted（不是 error），
    图停在上一 superstep，checkpoint 保留，/agent/resume 可续跑。
    """

    def decorator(fn: Callable) -> Callable:
        @functools.wraps(fn)
        def wrapper(state: Any, runtime: Any = None) -> Dict[str, Any]:
            trace: List[Dict[str, Any]] = (
                list(state.get("trace") or []) if hasattr(state, "get") else []
            )
            rnd = int(state.get("round", 0) or 0) if hasattr(state, "get") else 0
            thread_id = thread_id_of(runtime)
            # 旁路事件也占用了 seq，一并算进水位，避免续跑时撞号
            base = max(_max_seq(trace), _incident_seq(thread_id))

            # ① 中断检查：放在 node_start 之前，seq 仍然连续
            if should_abort(thread_id, node):
                stop = make_event(
                    "interrupted",
                    node=node,
                    round_no=rnd,
                    reason="收到中断标记，链路停在上一快照",
                )
                stop["seq"] = base + 1
                emit_event(stop)
                record_incident(thread_id, [stop])
                raise AgentInterrupted(f"会话 {thread_id} 已在节点 {node} 前中断")

            start = make_event("node_start", node=node, round_no=rnd)
            start["seq"] = base + 1
            emit_event(start)

            t0 = time.perf_counter()
            try:
                out = fn(state, runtime) or {}
            except AgentInterrupted:
                raise
            except Exception as exc:
                duration = (time.perf_counter() - t0) * 1000
                error = make_event(
                    "error",
                    node=node,
                    round_no=rnd,
                    duration_ms=duration,
                    error=f"{type(exc).__name__}: {exc}",
                )
                error["seq"] = base + 2
                emit_event(error)
                # 【可恢复·可见】节点抛异常时 LangGraph 丢弃返回值，error 事件
                # 进不了 state["trace"]（炸的那次在 /agent/trace 里查不到）→ 落旁路，
                # get_trace 读时再合并回来。异常仍原样抛出：上层已有兜底文案，
                # 这里只负责把"哪一步炸了"留在日志与轨迹里。
                record_incident(thread_id, [start, error])
                raise

            duration = (time.perf_counter() - t0) * 1000
            out = dict(out)
            raw_events = list(out.pop("_events", None) or [])
            end_meta = dict(out.pop("_end", None) or {})

            end = make_event(
                "node_end",
                node=node,
                round_no=rnd,
                duration_ms=duration,
                **end_meta,
            )

            stream = [start] + raw_events + [end]
            for offset, event in enumerate(stream):
                event["seq"] = base + 1 + offset
                # 内部事件（llm_start/llm_end/tool_start...）默认没有节点名，
                # 这里补上，日志/报告里才能看出"这次模型调用属于哪个节点"
                if not event.get("node"):
                    event["node"] = node
                if event is not start:
                    emit_event(event)

            # 实时推送：手写图（react）的事件也要能进前端思考面板
            _dispatch_events(thread_id, stream)

            # trace channel 的归约器是 operator.add：这里只返回**本步新增**事件，
            # 带上历史会重复追加（踩过：一次请求里事件数指数膨胀）。
            out["trace"] = stream
            return out

        return wrapper

    return decorator


# ============================================================
# 供节点复用的两个小函数
# ============================================================
def llm_events(ai: Any, duration_ms: float, messages_count: int, prompt_est: int):
    """把一次模型调用的结果转成 llm_start + llm_end 两个事件。"""
    usage = extract_usage(ai)
    calls = tool_calls_of(ai)
    start = make_event(
        "llm_start", messages=messages_count, prompt_est_tokens=prompt_est
    )
    end = make_event(
        "llm_end",
        duration_ms=duration_ms,
        tokens=usage,
        cost=round(cost_of(usage), 8) if usage else 0.0,
        thinking=preview(message_thinking(ai), 120),
        text=preview(message_text(ai), 120),
        tool_calls=[
            {"name": c.get("name"), "args": preview(c.get("args"), 60)} for c in calls
        ],
    )
    return [start, end], usage, calls


def tool_events(
    name: str,
    args: Any,
    duration_ms: float,
    content: Any,
    is_error: bool,
    reused: bool = False,
):
    """把一次工具执行转成 tool_start + tool_end 两个事件。

    reused=True 表示这次没有真的执行工具，而是从 tool_ledger 复用了上次结果
    （act 节点重放 / 续跑场景）。日志和 trace 里必须能看出这个区别，
    否则"工具没跑但发生了事件"会让人误判。
    """
    start = make_event("tool_start", name=name, args=preview(args, 90))
    if reused:
        start["reused"] = True
    end = make_event(
        "tool_end",
        name=name,
        duration_ms=duration_ms,
        is_error=is_error,
        content=preview(content, 90),
    )
    if reused:
        end["reused"] = True
    return [start, end]


# ============================================================
# team 拓扑的链路：节点级 + 工具级（走旁路通道，不进 state）
# ============================================================
# 为什么 team 不能用 traced_node：
#   1) TeamState 没有 trace 字段（也就没有 operator.add 归约）；
#   2) 专员是 Send 并行扇出的，每个分支拿到的是独立 payload（不含 trace），
#      各自从 0 开始编号 → 合并后 seq 撞号；
#   3) 专员内部是嵌套 create_agent，工具调用发生在子图里，外层黑板根本看不到。
# 所以 team 链路统一走「线程级旁路通道」：线程级原子序号 + deque + JSONL，
# get_trace 读的时候再和 state["trace"] / 事故记录合并。
_TEAM_TRACE: Dict[str, deque] = {}
_TEAM_TRACE_MAX = 800
_TEAM_TRACE_LOG = os.path.join(_LOG_DIR, "trace_team.jsonl")
_SEQ_LOCK = threading.Lock()
# 当前正在执行的链路节点名。team_node 进入时 set、退出时 reset；
# TraceCallback 读它，给嵌套子图里的 llm/tool 事件标上"属于哪个模块"。
_TRACE_NODE: contextvars.ContextVar[str] = contextvars.ContextVar("trace_node", default="-")
_SEQ: Dict[str, int] = {}


def read_team_trace(thread_id: str) -> List[Dict[str, Any]]:
    """读某会话的 team 链路事件（内存优先；进程重启后从 JSONL 恢复）。"""
    tid = thread_id or "-"
    got = list(_TEAM_TRACE.get(tid) or [])
    if got:
        return got
    if not os.path.exists(_TEAM_TRACE_LOG):
        return []
    out: List[Dict[str, Any]] = []
    try:
        with open(_TEAM_TRACE_LOG, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                except Exception:
                    continue
                if item.get("thread_id") == tid:
                    item.pop("thread_id", None)
                    out.append(item)
    except Exception as exc:
        logger.warning("[trace] team 链路读取失败：%s", exc)
        return []
    return out[-_TEAM_TRACE_MAX:]


def _next_seq(thread_id: str) -> int:
    """线程级原子序号：多分支并行也不会撞号。首次使用时按已有事件播种。"""
    with _SEQ_LOCK:
        cur = _SEQ.get(thread_id)
        if cur is None:
            cur = max(_max_seq(read_team_trace(thread_id)), _incident_seq(thread_id))
        cur += 1
        _SEQ[thread_id] = cur
        return cur


def record_trace(thread_id: str, events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """把 team 链路事件写进旁路通道：内存 + JSONL，并打日志 / 记指标。"""
    if not events:
        return []
    tid = thread_id or "-"
    bucket = _TEAM_TRACE.get(tid)
    if bucket is None:
        bucket = deque(maxlen=_TEAM_TRACE_MAX)
        _TEAM_TRACE[tid] = bucket
    out: List[Dict[str, Any]] = []
    for event in events:
        if not event.get("seq"):
            event["seq"] = _next_seq(tid)
        bucket.append(event)
        try:
            emit_event(event)
        except Exception:  # 日志/指标失败不影响主链路
            pass
        out.append(event)
    _dispatch_events(tid, out)

    try:
        os.makedirs(_LOG_DIR, exist_ok=True)
        with open(_TEAM_TRACE_LOG, "a", encoding="utf-8") as f:
            for event in out:
                f.write(
                    json.dumps({"thread_id": tid, **event}, ensure_ascii=False) + "\n"
                )
    except Exception as exc:
        logger.warning("[trace] team 链路写入失败：%s", exc)
    return out


def team_node(node: str) -> Callable[[Callable], Callable]:
    """给 team 黑板节点套事件流（node_start / node_end / error / interrupted）。

    被装饰的节点签名统一为 `fn(state, runtime=None)`；返回值原样透传，
    不往 state 里塞 trace（原因见上面的注释块）。
    """

    def decorator(fn: Callable) -> Callable:
        @functools.wraps(fn)
        def wrapper(state: Any, runtime: Any = None) -> Any:
            tid = thread_id_of(runtime) if runtime is not None else "-"
            rnd = int(state.get("round", 0) or 0) if hasattr(state, "get") else 0

            if should_abort(tid, node):
                stop = make_event(
                    "interrupted",
                    node=node,
                    round_no=rnd,
                    reason="收到中断标记，链路停在上一快照",
                )
                record_trace(tid, [stop])
                raise AgentInterrupted(f"会话 {tid} 已在节点 {node} 前中断")

            record_trace(tid, [make_event("node_start", node=node, round_no=rnd)])
            t0 = time.perf_counter()
            # 【归属】把当前节点名放进 contextvar：嵌套 create_agent 子图里的
            # llm / tool 回调会读它，标出"这次调用属于哪个模块"。
            node_token = _TRACE_NODE.set(node)
            try:
                out = fn(state, runtime)
            except AgentInterrupted:
                raise
            except Exception as exc:
                record_trace(tid, [make_event(
                    "error",
                    node=node,
                    round_no=rnd,
                    duration_ms=(time.perf_counter() - t0) * 1000,
                    error=f"{type(exc).__name__}: {exc}",
                )])
                raise
            finally:
                _TRACE_NODE.reset(node_token)
            # 【节点产出可见】节点可以用返回值的 `_trace` 键附带本步结果
            # （supervisor 的任务清单 / 专员的报告 / 聚合的答案）。
            # 它在进图前被弹出，不会污染 TeamState。
            meta = {}
            if isinstance(out, dict) and "_trace" in out:
                out = dict(out)
                meta = out.pop("_trace") or {}
            record_trace(tid, [make_event(
                "node_end",
                node=node,
                round_no=rnd,
                duration_ms=(time.perf_counter() - t0) * 1000,
                **meta,
            )])
            return out

        return wrapper

    return decorator


def _usage_of_response(response: Any) -> Dict[str, int]:
    """从 LLMResult 里尽量抠出 token 用量（不同 provider 字段不一样）。"""
    try:
        gens = getattr(response, "generations", None) or []
        if gens and gens[0]:
            msg = getattr(gens[0][0], "message", None)
            if msg is not None:
                usage = extract_usage(msg)
                if usage:
                    return usage
        out = getattr(response, "llm_output", None) or {}
        usage = out.get("token_usage") or out.get("usage") or {}
        if usage:
            return {
                "input_tokens": int(usage.get("prompt_tokens") or usage.get("input_tokens") or 0),
                "output_tokens": int(usage.get("completion_tokens") or usage.get("output_tokens") or 0),
                "total_tokens": int(usage.get("total_tokens") or 0),
            }
    except Exception:
        pass
    return {}


try:  # 回调基类缺失时不影响主链路（纯观测能力）
    from langchain_core.callbacks import BaseCallbackHandler as _BaseCallbackHandler
except Exception:  # pragma: no cover
    _BaseCallbackHandler = object  # type: ignore


class TraceCallback(_BaseCallbackHandler):  # type: ignore[misc]
    """把嵌套 create_agent 子图里的 LLM / 工具调用记进 team 链路。

    team 拓扑下专员是嵌套 agent，外层黑板节点只能看到"进场/出场"，
    看不到它内部调了哪个工具、模型花了多久。LangGraph 会把 config.callbacks
    传播进所有子图，所以在这里统一捕获，单点埋。
    """

    def __init__(self, thread_id: str):
        self._tid = thread_id or "-"
        self._t0: Dict[str, float] = {}
        self._names: Dict[str, str] = {}

    # ---- LLM ----
    def _node(self, fallback: str) -> str:
        """当前归属模块（team_node 用 contextvar 标好了，单机跑时回落）。"""
        try:
            cur = _TRACE_NODE.get()
            return cur if cur and cur != "-" else fallback
        except Exception:
            return fallback

    def _llm_start(self, run_id: Any, messages: Any = None, prompts: Any = None):
        try:
            batch = []
            if messages:
                batch = messages[0] if isinstance(messages[0], (list, tuple)) else messages
            n = len(batch) or len(prompts or [])
            roles = ",".join(str(getattr(m, "type", "?")) for m in batch[-8:]) if batch else ""
            last = ""
            if batch:
                last = preview(message_text(batch[-1]), 300)
            elif prompts:
                last = preview(prompts[-1], 300)
            self._t0[f"llm:{run_id}"] = time.perf_counter()
            record_trace(self._tid, [make_event(
                "llm_start",
                node=self._node("llm"),
                messages=n or None,
                roles=roles or None,
                input=last or None,
            )])
        except Exception:
            pass

    def on_chat_model_start(self, serialized, messages, **kwargs):
        self._llm_start(kwargs.get("run_id"), messages=messages)

    def on_llm_start(self, serialized, prompts, **kwargs):
        self._llm_start(kwargs.get("run_id"), prompts=prompts)

    def on_llm_end(self, response, **kwargs):
        try:
            key = f"llm:{kwargs.get('run_id')}"
            t0 = self._t0.pop(key, None)
            usage = _usage_of_response(response)

            # 【思考可读】把这次模型调用真正"说了什么"记下来：
            # thinking = reasoning_content（DeepSeek/Qwen 的思维链）
            # text     = 正文输出
            # tool_calls = 它决定调哪个工具、参数是什么
            msg = None
            try:
                gens = getattr(response, "generations", None) or []
                if gens and gens[0]:
                    msg = getattr(gens[0][0], "message", None)
            except Exception:
                msg = None
            text = preview(message_text(msg), 800) if msg is not None else ""
            thinking = preview(message_thinking(msg), 800) if msg is not None else ""
            calls = tool_calls_of(msg) if msg is not None else []

            record_trace(self._tid, [make_event(
                "llm_end",
                node=self._node("llm"),
                duration_ms=(time.perf_counter() - t0) * 1000 if t0 else 0.0,
                tokens=usage or None,
                cost=round(cost_of(usage), 8) if usage else None,
                thinking=thinking or None,
                text=text or None,
                tool_calls=[
                    {"name": c.get("name"), "args": preview(c.get("args"), 160)}
                    for c in calls
                ] or None,
            )])
        except Exception:
            pass

    def on_llm_error(self, error, **kwargs):
        try:
            key = f"llm:{kwargs.get('run_id')}"
            t0 = self._t0.pop(key, None)
            record_trace(self._tid, [make_event(
                "llm_end",
                node=self._node("llm"),
                duration_ms=(time.perf_counter() - t0) * 1000 if t0 else 0.0,
                is_error=True,
                error=f"{type(error).__name__}: {error}",
            )])
        except Exception:
            pass

    # ---- 工具 ----
    def on_tool_start(self, serialized, input_str, **kwargs):
        try:
            run_id = kwargs.get("run_id")
            name = (serialized or {}).get("name") or "tool"
            self._names[str(run_id)] = name
            self._t0[f"tool:{run_id}"] = time.perf_counter()
            record_trace(self._tid, [make_event(
                "tool_start", node=self._node("tool"), name=name, args=preview(input_str, 400)
            )])
        except Exception:
            pass

    def on_tool_end(self, output, **kwargs):
        try:
            run_id = kwargs.get("run_id")
            name = self._names.pop(str(run_id), "tool")
            t0 = self._t0.pop(f"tool:{run_id}", None)
            content = getattr(output, "content", output)
            record_trace(self._tid, [make_event(
                "tool_end",
                node=self._node("tool"),
                name=name,
                duration_ms=(time.perf_counter() - t0) * 1000 if t0 else 0.0,
                content=preview(content, 1500),
            )])
        except Exception:
            pass

    def on_tool_error(self, error, **kwargs):
        try:
            run_id = kwargs.get("run_id")
            name = self._names.pop(str(run_id), "tool")
            t0 = self._t0.pop(f"tool:{run_id}", None)
            record_trace(self._tid, [make_event(
                "tool_end",
                node=self._node("tool"),
                name=name,
                duration_ms=(time.perf_counter() - t0) * 1000 if t0 else 0.0,
                is_error=True,
                content=preview(f"{type(error).__name__}: {error}", 800),
            )])
        except Exception:
            pass
