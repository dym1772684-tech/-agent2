"""手写 ReAct 思考循环（每步可见、可追踪）。

============================================================
图结构
------------------------------------------------------------
    START → guard → think ─┬─(有 tool_calls) → act → observe ─┐
                           └─(无 tool_calls) → finalize → END │
                           ↑──────────────────────────────────┘

  guard     急救短路 / 身份复验 / 槽位追问（全部复用 middleware 的纯函数）
  think     调模型思考：想了什么、要不要调工具、这步花了多少 token
  act       按 tool_calls 逐个执行：权限校验 → 重试 → 脱敏 → 回填
  observe   汇总本步观察（成功几条、失败几条），回到 think
  finalize  结构化输出 ParsedOutput，失败降级为纯文本

============================================================
为什么不用 create_agent
------------------------------------------------------------
`create_agent` 把 model→tools→model 的循环封在图内部，编译后节点名只有
model / tools。结果是能埋"钩子"、不能埋"步骤"：

  - 不知道第几次思考跑偏
  - 不知道那次思考想了什么、为什么选这个工具
  - 不知道每步 token（缓存命中多少）
  - 工具参数/返回/耗时/是否被拦，全都看不见

这里把循环拆成显式节点，每个节点由 `traced_node` 包一层，把一条结构化
记录写进 `state["trace"]` 并打成多行日志（步间有分隔线）。

============================================================
校验为什么还走 middleware
------------------------------------------------------------
急救关键词、身份复验、槽位追问、工具越权——这些是**横切规则**，不是循环
逻辑。本文件直接调用 middleware.py 抽出的 `check_*` 纯函数，与官方
middleware 版共用同一份判定，换引擎不会让防线松掉。
"""
from __future__ import annotations

import os
import time
from typing import Any, Dict, List, Optional

from langchain.messages import SystemMessage, ToolMessage
from langgraph.graph import END, START, StateGraph

from allagent2.agent.middleware import (
    build_prompt,
    check_emergency,
    check_identity,
    check_slot,
    check_tool_permission,
    mask_pii,
)
from allagent2.agent.observe import (
    attr,
    llm_events,
    make_event,
    message_text,
    preview,
    tool_calls_of,
    tool_events,
    traced_node,
    thread_id_of,
)
from allagent2.agent.documents import make_snapshot
from allagent2.agent.state import CareAgentState, ParsedOutput
from allagent2.guance.logging_helper import logger
from allagent2.guance.token_accounting import (
    cost_of,
    estimate_messages_tokens,
    extract_usage,
)

# 消息压缩 / 工具结果 / 重试的默认值。
# 运行时以 allagent2.config 的配置快照为准（agent.compress_trigger 等），
# 这里的常量只是配置不可用时的兜底 —— 保证"配置读失败也能跑"。
_COMPRESS_TRIGGER = 30
_COMPRESS_KEEP = 16
# 单条工具结果写进上下文的字符上限（防一次查询撑爆窗口）
_TOOL_RESULT_LIMIT = 4000
# 工具执行重试次数（网络抖动/接口超时）
_TOOL_RETRIES = 2
# 工具结果里出现这些前缀 = 没真正拿到数据
_FAIL_PREFIXES = ("无权", "缺少", "工具执行失败", "工具 ")


def _limits() -> Dict[str, int]:
    """从配置快照取运行参数（失败即回落到模块常量）。"""
    try:
        from allagent2.config import get_settings

        agent = get_settings().settings.agent
        return {
            "compress_trigger": int(agent.compress_trigger),
            "compress_keep": int(agent.compress_keep),
            "tool_result_limit": int(agent.tool_result_limit),
            "tool_retries": int(agent.tool_retries),
            "max_rounds": int(agent.max_rounds),
        }
    except Exception:
        return {
            "compress_trigger": _COMPRESS_TRIGGER,
            "compress_keep": _COMPRESS_KEEP,
            "tool_result_limit": _TOOL_RESULT_LIMIT,
            "tool_retries": _TOOL_RETRIES,
            "max_rounds": 6,
        }


# ============================================================
# 小工具
# ============================================================
def _args_preview(args: Any, limit: int = 60) -> str:
    try:
        text = ", ".join(f"{k}={preview(v, 24)}" for k, v in (args or {}).items())
    except Exception:
        text = preview(args, limit)
    return text if len(text) <= limit else text[:limit] + "…"


def _last_message(state: Any) -> Any:
    msgs = state.get("messages") or []
    return msgs[-1] if msgs else None


def _last_ai_text(state: Any) -> str:
    """最近一条有文本的 AI 消息（finalize 降级时当 answer 用）。"""
    for msg in reversed(state.get("messages") or []):
        if attr(msg, "type", "") == "ai":
            text = message_text(msg).strip()
            if text:
                return text
    return ""


# 写工具白名单：环境变量 AGENT_WRITE_TOOLS="record_medication,set_reminder"
# 现在装配的工具全是只读（query/检索/计算/天气/抓取），白名单默认为空；
# 一旦接入写工具，把名字加进来即可获得「不盲目重试 + 自动注入幂等键」的保护。
_WRITE_TOOLS_ENV = "AGENT_WRITE_TOOLS"


def write_tools() -> set:
    """当前配置为「写操作」的工具名集合（小写无关，精确匹配）。"""
    raw = os.getenv(_WRITE_TOOLS_ENV, "") or ""
    return {item.strip() for item in raw.split(",") if item.strip()}


def supports_idempotency_key(tool: Any) -> bool:
    """工具签名里是否有 idempotency_key 参数。

    有才注入：StructuredTool 的参数 schema 是严格的，多传一个未声明参数会直接
    ValidationError，反而把本来能跑的工具调崩。
    """
    try:
        return "idempotency_key" in (getattr(tool, "args", None) or {})
    except Exception:
        return False


def tool_idempotency_key(thread_id: str, turn_id: str, call_id: str) -> str:
    """写工具的幂等键：同一次用户动作 + 同一个 tool_call 恒定。

    为什么包含 tool_call_id：一次提问里可能先后调两次同名工具（先查后写），
    用 tool_name 会让第二次被误判成重复；tool_call_id 是模型给的、断点续跑时不变。
    """
    return f"{thread_id}:{turn_id}:{call_id}"


def _invoke_with_retry(
    tool: Any,
    args: Dict[str, Any],
    retries: int = _TOOL_RETRIES,
    *,
    write: bool = False,
):
    """执行工具，指数退避重试。返回 (结果, 错误文案)。

    【幂等·L3】写操作默认**不重试**：工具可能已经把副作用落到库里，只是响应超时，
    重试等于写两次。只有两类例外允许重试：
      1. 工具声明了 idempotency_key 参数（服务端能按 key 去重）；
      2. 调用方显式传 write=False（确认是读操作）。
    """
    if write and not supports_idempotency_key(tool):
        retries = 0
    last_error = None
    for attempt in range(retries + 1):
        try:
            return tool.invoke(args), None
        except Exception as exc:
            last_error = exc
            if attempt < retries:
                time.sleep(0.3 * (2**attempt))
    return None, f"{type(last_error).__name__}: {last_error}"


def _bind_tools(model: Any, tools: List[Any]) -> Any:
    try:
        return model.bind_tools(tools)
    except Exception as exc:  # 兜底模型 / 不支持 tools 的端点
        logger.warning("[react] bind_tools 失败，按无工具模式运行：%s", exc)
        return model


def _structured_model(model: Any):
    try:
        # 【兼容】langchain-openai>=1.1 默认 json_schema，DeepSeek 不支持 → 400；
        # 显式 function_calling（工具调用）兼容性最好。
        return model.with_structured_output(
            ParsedOutput, include_raw=True, method="function_calling"
        )
    except Exception as exc:
        logger.warning("[react] with_structured_output 不可用，finalize 将走纯文本：%s", exc)
        return None


# ============================================================
# 主构建函数
# ============================================================
def build_react_graph(
    *,
    model: Any,
    tools: List[Any],
    checkpointer: Any = None,
    max_rounds: int = 0,
    name: str = "elder_care_agent",
):
    """构建手写 ReAct 图（返回已编译对象，调用方不要再 .compile()）。
                                                                                                                                  
    max_rounds=0 表示用配置里的 agent.max_rounds（默认 6）；显式传值优先。
    """
    tool_map: Dict[str, Any] = {t.name: t for t in tools}
    limits = _limits()
    rounds_cap = int(max_rounds or limits["max_rounds"])
    # 成本分桶的模型键（"provider/model" 形式）；配置缺失时退回 unknown
    try:
        from allagent2.config import get_settings

        _cfg_model = get_settings().settings.llm.model
    except Exception:
        _cfg_model = ""
    model_key = (
        getattr(model, "model_name", None)
        or getattr(model, "model", None)
        or _cfg_model
        or "unknown"
    )
    tool_model = _bind_tools(model, tools)
    structured = _structured_model(model)

    # --------------------------------------------------------
    # 公共：消息装配（system prompt + 历史 + 压缩）
    # --------------------------------------------------------
    def _build_messages(state: Any, runtime: Any):
        history = list(state.get("messages") or [])
        stats = None
        before = estimate_messages_tokens(history)
        if len(history) > limits["compress_trigger"]:
            dropped = len(history) - limits["compress_keep"]
            history = [
                SystemMessage(content=f"（较早的 {dropped} 条对话已省略，仅保留最近上下文）")
            ] + history[-limits["compress_keep"]:]
            stats = {
                "dropped": dropped,
                "kept": limits["compress_keep"],
                "tokens_before": before,
                "tokens_after": estimate_messages_tokens(history),
                "reason": f"超过 {limits['compress_trigger']} 条触发压缩",
            }
        return [SystemMessage(content=build_prompt(state, runtime))] + history, stats

    # --------------------------------------------------------
    # 节点 1：guard —— 横切校验（规则，不花 token）
    # --------------------------------------------------------
    def guard_node(state: Any, runtime: Any) -> Dict[str, Any]:
        # ① 急救：命中关键词直接给急救指引，不进模型
        hit = check_emergency(state)
        if hit:
            hit.pop("jump_to", None)
            return {
                **hit,
                "phase": "emergency",
                "_events": [make_event("guard", rule="emergency")],
                "_end": {"next_action": "命中急救规则，直接输出急救指引（未调用模型）"},
            }

        # ② 零信任：身份失效（过期/降权/登出）立即斩断
        blocked = check_identity(state, runtime)
        if blocked:
            blocked.pop("jump_to", None)
            return {
                **blocked,
                "phase": "identity_blocked",
                "_events": [make_event("guard", rule="identity")],
                "_end": {"next_action": "身份不可信，中止本轮（fail-closed）"},
            }

        # ③ 槽位：问个人数据却没给老人编号 → 注入一次追问指令
        slot = check_slot(state, runtime)
        if slot:
            return {
                **slot,
                "phase": "slot_ask",
                "_events": [make_event("guard", rule="slot")],
                "_end": {"next_action": "缺少老人编号，先向用户追问"},
            }

        return {
            "phase": "guard_pass",
            "_end": {"next_action": "校验通过，进入思考"},
        }

    # --------------------------------------------------------
    # 节点 2：think —— 模型思考（本步产出 thought / tool_calls / token）
    # --------------------------------------------------------
    def think_node(state: Any, runtime: Any) -> Dict[str, Any]:
        rnd = int(state.get("round", 0) or 0) + 1
        messages, ctx_stats = _build_messages(state, runtime)

        t0 = time.perf_counter()
        ai = tool_model.invoke(messages)  # 唯一的模型调用点
        latency = (time.perf_counter() - t0) * 1000

        # 事件：llm_start（上下文规模）→ llm_end（耗时 / token / cost / thinking / text / tool_calls）
        events, usage, calls = llm_events(
            ai, latency, len(messages), estimate_messages_tokens(messages)
        )
        text = message_text(ai)

        if calls:
            next_action = "调用工具 " + "、".join(
                f"{c.get('name')}({_args_preview(c.get('args'))})" for c in calls
            )
        else:
            next_action = "信息已足够，进入结构化收尾"

        # 【care.usage】成本按模型分桶累加（排障要看"哪个模型烧的钱"）
        usage_models = dict(state.get("usage_models") or {})
        if usage:
            bucket = dict(usage_models.get(model_key) or {})
            for key in ("calls", "input_tokens", "output_tokens", "total_tokens",
                        "cache_read", "cache_miss"):
                add = 1 if key == "calls" else int(usage.get(key) or 0)
                bucket[key] = int(bucket.get(key) or 0) + add
            bucket["cost_yuan"] = round(
                float(bucket.get("cost_yuan") or 0) + cost_of(usage), 8
            )
            usage_models[model_key] = bucket

        # 【care.trace】快照锚点：本轮想到第几步、事件到第几条
        base_trace_len = len(state.get("trace") or [])
        snapshot = make_snapshot(
            node="think",
            batch_seq=base_trace_len + len(events) + 2,
            trace_len=base_trace_len + len(events) + 2,
            detail={
                "round": rnd,
                "tool_calls": [c.get("name") for c in calls],
                "model": model_key,
            },
        )

        return {
            "messages": [ai],
            "round": rnd,
            "phase": "think",
            "usage_models": usage_models,
            "snapshot": snapshot,
            "context_stats": ctx_stats,
            "_events": events,
            "_end": {
                "next_action": next_action,
                "detail": {
                    "tool_calls": [c.get("name") for c in calls],
                    "compressed": (ctx_stats or {}).get("dropped"),
                    "has_text": bool((text or "").strip()),
                },
            },
        }

    # --------------------------------------------------------
    # 节点 3：act —— 逐个执行工具（含权限校验 / 重试 / 脱敏）
    # --------------------------------------------------------
    def act_node(state: Any, runtime: Any) -> Dict[str, Any]:
        calls = tool_calls_of(_last_message(state))
        if not calls:
            return {
                "phase": "act",
                "_end": {"next_action": "没有待执行的工具调用，直接回到思考"},
            }

        out_messages: List[Any] = []
        events: List[Dict[str, Any]] = []
        board = dict(state.get("tool_results") or {})
        # 【幂等·L2】工具调用台账：key 是 tool_call_id（不是工具名）。
        # tool_results 的 key 是工具名，同名工具调两次会互相覆盖，回答不了
        # 「这个具体调用执行过没有」——而幂等恰恰要这个粒度。
        ledger = dict(state.get("tool_ledger") or {})
        thread_id = thread_id_of(runtime)
        turn_id = str(state.get("turn_id") or "")
        writes = write_tools()
        executed = 0
        reused = 0
        cur_round = int(state.get("round", 0) or 0)

        def _remember(call_id: str, name: str, status: str, text: str) -> None:
            ledger[call_id] = {
                "name": name,
                "status": status,
                "text": text,
                "round": cur_round,
                "ts": round(time.time(), 3),
            }

        for call in calls:
            name = call.get("name")
            args = call.get("args") or {}
            call_id = call.get("id") or f"call_{name}"

            # ⓪【幂等·L2】这个 tool_call 已经执行过 → 复用台账里的结果，不再碰工具。
            # 触发场景：checkpoint 恢复后 act 重放、interrupt 后续跑、图上重试。
            # 没有这一层，写工具会被执行两次；读工具则白烧一次 RPC 与延迟。
            prev = ledger.get(call_id)
            if isinstance(prev, dict) and prev.get("status"):
                text = str(prev.get("text") or "")
                out_messages.append(
                    ToolMessage(content=text, tool_call_id=call_id, name=name)
                )
                board[name] = text
                events.extend(
                    tool_events(
                        name, args, 0, text, prev.get("status") != "ok", reused=True
                    )
                )
                reused += 1
                continue

            # ① 权限校验（复用 middleware 的越权判定）
            denied = check_tool_permission(call, state, runtime)
            if denied:
                out_messages.append(
                    ToolMessage(content=denied, tool_call_id=call_id, name=name)
                )
                board[name] = denied
                _remember(call_id, name, "denied", denied)
                events.extend(tool_events(name, args, 0, denied, True))
                continue

            # ② 工具不存在（MCP 服务未启动时会被去重删除）
            tool = tool_map.get(name)
            if tool is None:
                text = f"工具 {name} 当前不可用（服务未连接或未注册）"
                out_messages.append(
                    ToolMessage(content=text, tool_call_id=call_id, name=name)
                )
                board[name] = text
                _remember(call_id, name, "missing", text)
                events.extend(tool_events(name, args, 0, text, True))
                continue

            # ③ 执行（读工具带重试 / 写工具默认不重试）→ ④ 脱敏 → ⑤ 截断 → ⑥ 回填
            is_write = name in writes
            call_args = args
            if is_write:
                if supports_idempotency_key(tool):
                    # 同一次用户动作 + 同一个 tool_call → 恒定的幂等键，
                    # 无论重试几次，MCP 服务端按这个键去重，只生效一次
                    call_args = {
                        **args,
                        "idempotency_key": tool_idempotency_key(
                            thread_id, turn_id, call_id
                        ),
                    }
                else:
                    logger.warning(
                        "[idem] 写工具 %s 未声明 idempotency_key 参数，已禁用自动重试"
                        "（改名或加参数可开启服务端去重）",
                        name,
                    )

            t0 = time.perf_counter()
            result, error = _invoke_with_retry(
                tool, call_args, limits["tool_retries"], write=is_write
            )
            latency = (time.perf_counter() - t0) * 1000

            if error:
                safe = f"工具执行失败：{error}"
                is_error = True
            else:
                safe = mask_pii(result if isinstance(result, str) else str(result))
                safe = safe[: limits["tool_result_limit"]]
                is_error = False

            board[name] = safe
            # 失败也记台账：宁可这次答得保守，也不要因为"想重试成功"而重复写
            _remember(call_id, name, "error" if is_error else "ok", safe)
            events.extend(tool_events(name, args, latency, safe, is_error))
            out_messages.append(
                ToolMessage(content=safe, tool_call_id=call_id, name=name)
            )
            executed += 1

        return {
            "messages": out_messages,
            "tool_results": board,
            "tool_ledger": ledger,
            "phase": "act",
            "_events": events,
            "_end": {
                "next_action": (
                    f"执行 {executed} 个工具、复用 {reused} 个已执行结果，"
                    "交回模型继续判断"
                )
            },
        }

    # --------------------------------------------------------
    # 节点 4：observe —— 汇总观察（不产生消息，只统计与留痕）
    # --------------------------------------------------------
    def observe_node(state: Any, runtime: Any) -> Dict[str, Any]:
        recent = [
            m for m in (state.get("messages") or [])[-10:] if attr(m, "type", "") == "tool"
        ]
        ok = sum(
            1 for m in recent if not message_text(m).startswith(_FAIL_PREFIXES)
        )
        failed = len(recent) - ok
        names = [attr(m, "name", "tool") for m in recent]
        summary = f"本轮收到 {len(recent)} 条工具结果" + (
            f"：{', '.join(n for n in names if n)}" if names else ""
        )
        return {
            "phase": "observe",
            "_events": [
                make_event(
                    "observe", summary=summary, ok_count=ok, failed_count=failed
                )
            ],
            "_end": {"next_action": "回到思考：判断信息是否足够，不够就继续调工具"},
        }

    # --------------------------------------------------------
    # 节点 5：finalize —— 结构化收尾（失败降级纯文本）
    # --------------------------------------------------------
    def finalize_node(state: Any, runtime: Any) -> Dict[str, Any]:
        messages, _ = _build_messages(state, runtime)
        parsed: Optional[ParsedOutput] = None
        error: Optional[str] = None

        if structured is not None:
            try:
                res = structured.invoke(messages)
                parsed = attr(res, "parsed", None)
                if parsed is None and isinstance(res, ParsedOutput):
                    parsed = res
                if parsed is None:
                    raw = attr(res, "raw", None)
                    error = "结构化解析为空（常见原因：JSON 被 max_tokens 截断）"
                    text = message_text(raw).strip() if raw is not None else ""
                    if text:
                        parsed = ParsedOutput(answer=text)
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"

        if parsed is None:
            # 降级：用最近一条自然语言回复兜底，保证用户一定拿得到话
            error = error or "结构化输出不可用"
            parsed = ParsedOutput(
                answer=_last_ai_text(state) or "抱歉，本轮未能生成回复，请换个问法重试。"
            )
        if isinstance(parsed, dict):
            try:
                parsed = ParsedOutput(**parsed)
            except Exception:
                parsed = ParsedOutput(answer=str(parsed))

        emergency = bool(
            getattr(parsed, "emergency_flag", False) or state.get("emergency_flag")
        )
        parsed.emergency_flag = emergency

        return {
            "structured_response": parsed,
            "phase": "finalize",
            "emergency_flag": emergency,
            "_events": [
                make_event(
                    "structured_end",
                    ok=error is None,
                    error=error,
                    answer_len=len(parsed.answer or ""),
                    degraded=error is not None,
                )
            ],
            "_end": {"next_action": "结束本轮，返回结构化结果"},
        }

    # --------------------------------------------------------
    # 条件边
    # --------------------------------------------------------
    def route_after_guard(state: Any) -> str:
        if state.get("phase") in ("emergency", "identity_blocked"):
            return END
        return "think"

    def route_after_think(state: Any) -> str:
        if not tool_calls_of(_last_message(state)):
            return "finalize"
        if int(state.get("round", 0) or 0) >= rounds_cap:
            logger.warning(
                "[react] 达到工具轮次上限 %s（已完成 %s 轮思考），强制收尾",
                rounds_cap,
                state.get("round"),
            )
            return "finalize"
        return "act"

    # --------------------------------------------------------
    # 组装
    # --------------------------------------------------------
    builder = StateGraph(CareAgentState)
    builder.add_node("guard", traced_node("guard")(guard_node))
    builder.add_node("think", traced_node("think")(think_node))
    builder.add_node("act", traced_node("act")(act_node))
    builder.add_node("observe", traced_node("observe")(observe_node))
    builder.add_node("finalize", traced_node("finalize")(finalize_node))

    builder.add_edge(START, "guard")
    builder.add_conditional_edges(
        "guard", route_after_guard, {"think": "think", END: END}
    )
    builder.add_conditional_edges(
        "think", route_after_think, {"act": "act", "finalize": "finalize"}
    )
    builder.add_edge("act", "observe")
    builder.add_edge("observe", "think")
    builder.add_edge("finalize", END)

    agent = builder.compile(checkpointer=checkpointer, name=name)
    logger.info(
        "[react] 手写思考循环编译完成：tools=%s rounds_cap=%s model=%s",
        list(tool_map),
        rounds_cap,
        model_key,
    )
    return agent
