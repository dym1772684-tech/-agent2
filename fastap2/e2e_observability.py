"""全链路走查：把「问答 → 中断 → 重启续跑」的每一步写成一份可读报告。

============================================================
它回答什么问题
------------------------------------------------------------
1. 用户问一句，智能体内部**每一步**到底做了什么（思考/工具/token/下一步）；
2. 每一步落盘后，checkpoint 快照长什么样（哪个节点、多少消息、trace 多长）；
3. 用户中途中断后，断点还在不在、停在哪、丢了什么；
4. 重启对话时，断点内容是怎么重新注入的（消息数/trace 长度/seq 水位）；
5. 这一整套跑下来，指标涨了多少。

输出：fastap2/observability_walkthrough.md

============================================================
为什么用假模型
------------------------------------------------------------
走查要保持"随时可重跑、零 token、不依赖 vLLM/MCP/MySQL"。假模型走的是
**真实的图、真实的事件流、真实的 checkpoint**，只有 LLM 的文本是脚本化的。
事件、耗时、token、快照、中断、续跑全部来自真实代码路径。

运行：
    cd <项目根>            # 练手agent
    python fastap2/e2e_observability.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

FASTAP2_DIR = Path(__file__).resolve().parent
WORKSPACE = FASTAP2_DIR.parent
if str(WORKSPACE) not in sys.path:
    sys.path.insert(0, str(WORKSPACE))
if str(FASTAP2_DIR) not in sys.path:
    sys.path.insert(0, str(FASTAP2_DIR))

from langchain.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver

from allagent2.agent.documents import CURRENT_STATE_VERSION
from allagent2.agent.observe import (
    AgentInterrupted,
    read_incidents,
    set_abort_check,
)
from allagent2.agent.react_loop import build_react_graph
from allagent2.agent.state import ParsedOutput, PermissionContext
from allagent2.guance.metrics_hook import set_sink

OUT_FILE = FASTAP2_DIR / "observability_walkthrough.md"
THREAD = "demo-thread-001"


# ============================================================
# 假模型 / 假工具
# ============================================================
def _usage(prompt=820, completion=96, cache=520):
    return {
        "input_tokens": prompt,
        "output_tokens": completion,
        "total_tokens": prompt + completion,
        "input_token_details": {"cache_read": cache},
        "output_token_details": {},
    }


class FakeStructured:
    def __init__(self, value=None):
        self.value = value

    def invoke(self, messages, **kwargs):
        return {"parsed": self.value, "raw": None, "parsing_error": None}


class ScriptedLLM:
    """按脚本返回，并可在第 N 次调用时"替用户按下停止键"。"""

    def __init__(self, responses, structured=None, interrupt_at: int = 0):
        self.responses = list(responses)
        self.index = 0
        self.calls = 0
        self.structured = structured or FakeStructured()
        self.interrupt_at = interrupt_at
        self.model_name = "demo-model-7b"
        self.fired_interrupt = False

    def bind_tools(self, tools):
        return self

    def with_structured_output(self, schema, include_raw=True):
        return self.structured

    def invoke(self, messages, **kwargs):
        self.calls += 1
        pos = min(self.index, len(self.responses) - 1)
        self.index += 1
        # 【模拟用户中断】第 N 次思考返回后立刻落中断标记，
        # 于是图在下一个节点（act）入口就会停下 —— 与真实 /interrupt 一致
        if self.interrupt_at and self.calls >= self.interrupt_at and not self.fired_interrupt:
            self.fired_interrupt = True
            set_abort_check(lambda tid: True)
        return self.responses[pos]


class FakeTool:
    def __init__(self, name, result):
        self.name = name
        self.result = result
        self.calls = 0

    def invoke(self, args):
        self.calls += 1
        time.sleep(0.01)  # 让耗时字段看起来真实
        return self.result


def _tool_msg(name, args):
    return AIMessage(
        content="",
        tool_calls=[{"name": name, "args": args, "id": f"call_{name}_1", "type": "tool_call"}],
        usage_metadata=_usage(),
        additional_kwargs={"reasoning_content": f"用户问的是个人健康数据，需要先调 {name} 拿到近期记录。"},
    )


def _text_msg(text, thinking=""):
    return AIMessage(
        content=text,
        usage_metadata=_usage(760, 130, 480),
        additional_kwargs={"reasoning_content": thinking} if thinking else {},
    )


# ============================================================
# 指标收集（替代 prometheus，直接内存聚合）
# ============================================================
class MetricRecorder:
    def __init__(self):
        self.rows: List[Dict[str, Any]] = []

    def __call__(self, name: str, value: float, labels: Dict[str, Any]) -> None:
        self.rows.append({"name": name, "value": value, "labels": labels})

    def summary(self) -> Dict[str, Dict[str, float]]:
        out: Dict[str, Dict[str, float]] = {}
        for row in self.rows:
            key = row["name"]
            if row["labels"]:
                key += "{" + ",".join(f"{k}={v}" for k, v in row["labels"].items()) + "}"
            bucket = out.setdefault(key, {"count": 0, "sum": 0.0})
            bucket["count"] += 1
            bucket["sum"] = round(bucket["sum"] + float(row["value"]), 4)
        return dict(sorted(out.items()))


# ============================================================
# 渲染工具
# ============================================================
def _truncate(text: Any, limit: int = 68) -> str:
    if text is None:
        return ""
    s = " ".join(str(text).split())
    return s if len(s) <= limit else s[:limit] + "…"


def render_event(event: Dict[str, Any]) -> List[str]:
    """一个事件 → 一到多行 markdown 表格行。"""
    kind = event.get("type", "?")
    seq = event.get("seq", "-")
    node = event.get("node") or "-"
    dur = event.get("duration_ms")
    dur_s = f"{dur:.0f}ms" if isinstance(dur, (int, float)) else ""

    if kind == "node_start":
        rnd = f" 第{event['round']}轮" if event.get("round") else ""
        return [f"| {seq} | ▶ 进入 | `{node}`{rnd} | {dur_s} | 节点开始执行 |"]
    if kind == "node_end":
        return [f"| {seq} | ✔ 结束 | `{node}` | {dur_s} | 下一步：{_truncate(event.get('next_action'), 56)} |"]
    if kind == "llm_start":
        return [f"| {seq} | 🧠 思考 | `{node}` | - | 上下文 {event.get('messages')} 条 / 预估 {event.get('prompt_est_tokens')} tokens |"]
    if kind == "llm_end":
        tokens = event.get("tokens") or {}
        rows = [
            f"| {seq} | 🧠 模型返回 | `{node}` | {dur_s} | token in={tokens.get('input_tokens', 0)}"
            f"(cache {tokens.get('cache_read', 0)}) out={tokens.get('output_tokens', 0)} "
            f"cost=¥{float(event.get('cost') or 0):.6f} |"
        ]
        if event.get("thinking"):
            rows.append(f"| | ↳ 思考内容 | | | {_truncate(event['thinking'], 90)} |")
        if event.get("text"):
            rows.append(f"| | ↳ 输出文本 | | | {_truncate(event['text'], 90)} |")
        for call in event.get("tool_calls") or []:
            rows.append(f"| | ↳ 决定调用 | | | `{call.get('name')}({call.get('args')})` |")
        return rows
    if kind == "tool_start":
        return [f"| {seq} | 🔧 工具开始 | `{event.get('name')}` | - | 参数 {event.get('args')} |"]
    if kind == "tool_end":
        flag = "异常" if event.get("is_error") else "成功"
        return [f"| {seq} | 🔧 工具结束 | `{event.get('name')}` | {dur_s} | {flag}：{_truncate(event.get('content'), 56)} |"]
    if kind == "guard":
        return [f"| {seq} | 🛡 规则 | `guard` | - | 命中 {event.get('rule')} |"]
    if kind == "observe":
        return [f"| {seq} | 👁 观察 | `observe` | - | {_truncate(event.get('summary'), 60)}（成功 {event.get('ok_count')} / 异常 {event.get('failed_count')}） |"]
    if kind == "structured_end":
        state = "成功" if event.get("ok") else f"降级（{event.get('error')}）"
        return [f"| {seq} | 📦 结构化收尾 | `finalize` | - | {state}，答案 {event.get('answer_len')} 字 |"]
    if kind == "interrupted":
        return [f"| {seq} | ⏸ **用户中断** | `{node}` | - | {event.get('reason')} |"]
    if kind == "error":
        return [f"| {seq} | ✖ 异常 | `{node}` | {dur_s} | {_truncate(event.get('error'), 70)} |"]
    return [f"| {seq} | {kind} | `{node}` | {dur_s} | {_truncate(event, 60)} |"]


def render_events(events: List[Dict[str, Any]]) -> str:
    lines = ["| # | 类型 | 节点 | 耗时 | 详情 |", "|---|---|---|---|---|"]
    for event in events:
        lines.extend(render_event(event))
    return "\n".join(lines)


# ============================================================
# 主流程
# ============================================================
def snapshot_rows(graph, config) -> List[Dict[str, Any]]:
    """把 checkpoint 历史整理成"快照时间线"。"""
    rows: List[Dict[str, Any]] = []
    try:
        history = list(graph.get_state_history(config))
    except Exception as exc:
        return [{"error": str(exc)}]
    prev_msgs = 0
    for i, snap in enumerate(reversed(history)):
        values = snap.values or {}
        data = values.model_dump() if hasattr(values, "model_dump") else dict(values)
        msgs = data.get("messages") or []
        rows.append({
            "step": i + 1,
            "checkpoint_id": (getattr(snap, "config", {}) or {}).get("configurable", {}).get("checkpoint_id", "")[:12],
            "next": ",".join(getattr(snap, "next", None) or ()) or "(结束)",
            "messages": len(msgs),
            "new_messages": len(msgs) - prev_msgs,
            "trace_len": len(data.get("trace") or []),
            "round": data.get("round") or 0,
            "phase": data.get("phase") or "-",
        })
        prev_msgs = len(msgs)
    return rows


def render_snapshots(rows: List[Dict[str, Any]]) -> str:
    if rows and rows[0].get("error"):
        return f"快照读取失败：{rows[0]['error']}"
    lines = [
        "| 步 | checkpoint | 下一步节点 | 消息数 | 新增 | trace 长度 | 轮次 | 阶段 |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['step']} | `{row['checkpoint_id']}` | {row['next']} | {row['messages']} "
            f"| +{row['new_messages']} | {row['trace_len']} | {row['round']} | {row['phase']} |"
        )
    return "\n".join(lines)


def _idem_demo():
    """跑一遍 HTTP 幂等三态，返回 [(时刻, 状态, 说明)] 与「是否真 Redis」。"""
    import asyncio

    async def _run():
        from core import idempotency as idem
        from core.redis_client import get_redis_async, redis_available

        live = bool(redis_available())
        redis = get_redis_async() if live else None
        key = idem.normalize_key("walkthrough-user", "demo-k1")
        rows = []

        state, _ = await idem.acquire(redis, key)
        rows.append(("第一次请求", state, "SET NX 抢到执行权 → 真正跑模型"))
        state, _ = await idem.acquire(redis, key)
        rows.append(("同键并发（如用户连点）", state, "被挡住 → 返回 409，不会双扣 token / 双写库"))

        await idem.commit(redis, key, {"answer": "缓存下来的答案", "thread_id": "demo"})
        state, cached = await idem.acquire(redis, key)
        rows.append(
            ("执行完再重试（客户端超时重试）", state,
             f"回放缓存 `{str((cached or {}).get('answer'))[:12]}…`，零 token、零工具副作用")
        )

        # 清理演示键，避免污染真实 Redis
        if live:
            try:
                await redis.delete(f"{key}:done")
                await redis.delete(f"{key}:lock")
            except Exception:
                pass
        return rows, live

    return asyncio.run(_run())


def main() -> int:
    started = time.strftime("%Y-%m-%d %H:%M:%S")
    recorder = MetricRecorder()
    set_sink(recorder)

    # ---------------- 装配 ----------------
    tools = [
        FakeTool("query_health_records", "老人 E001 近 7 天血压：150/95、148/92、152/96（收缩压持续偏高）"),
        FakeTool("search_knowledge_base", "高血压饮食建议：每日食盐 <5g；增加富钾食物（香蕉/菠菜）；限制饮酒。"),
    ]
    llm = ScriptedLLM(
        responses=[
            _tool_msg("query_health_records", {"elder_id": "E001"}),
            _text_msg(
                "老人近 7 天收缩压 148~152，持续偏高。建议：① 每日固定时间测量并记录；"
                "② 食盐控制在 5g 以内，少吃腌制食品；③ 若出现头痛、胸闷，立即就医。",
                thinking="已拿到近 7 天血压记录，收缩压均超过 140，属于持续偏高。先给出测量与饮食建议，再提示red flag。",
            ),
            _tool_msg("search_knowledge_base", {"query": "高血压老人 太极 运动强度"}),
            _text_msg(
                "太极适合老人，但要注意：① 避开清晨血压高峰，改到上午 9 点后；"
                "② 每次 20~30 分钟，以微微出汗、能正常说话为准；③ 出现头晕立即停下休息。",
                thinking="用户问运动，先查知识库拿有依据的建议，再结合太极这种低强度运动给具体强度。",
            ),
        ],
        structured=FakeStructured(
            ParsedOutput(
                answer="老人近 7 天收缩压 148~152，持续偏高，建议规律测量并控制食盐摄入。",
                health_advice="每日固定时间测血压并记录，保证睡眠。",
                diet_advice="每日食盐 <5g，少吃腌制食品，增加富钾食物。",
                exercise_advice="以散步为主，每次 20~30 分钟，避免清晨剧烈活动。",
                risk_level="中",
                emergency_flag=False,
            )
        ),
        interrupt_at=3,   # 第二轮第 1 次思考后落中断标记
    )
    checkpointer = InMemorySaver()
    graph = build_react_graph(model=llm, tools=tools, checkpointer=checkpointer, max_rounds=4)
    config = {"configurable": {"thread_id": THREAD}}
    perm = PermissionContext(
        allowed_tools=["query_health_records", "search_knowledge_base"],
        allowed_elder_ids=["E001"],
    )

    # ---------------- 第 1 幕：完整问答 ----------------
    q1 = "帮我看看 E001 最近的血压情况，饮食上要注意什么？"
    result1 = graph.invoke(
        {"messages": [HumanMessage(content=q1)], "permissions": perm,
         "schema_version": CURRENT_STATE_VERSION},
        config=config,
    )
    state1 = result1 if isinstance(result1, dict) else dict(result1)
    events1 = [e for e in (state1.get("trace") or []) if isinstance(e, dict)]
    snaps1 = snapshot_rows(graph, config)
    parsed = state1.get("structured_response")
    parsed_dump = parsed.model_dump() if hasattr(parsed, "model_dump") else (parsed or {})
    usage1 = state1.get("usage_models") or {}

    # ---------------- 第 2 幕：追问 + 中断 ----------------
    q2 = "那运动方面呢？他早上喜欢去公园打太极。"
    interrupted = False
    interrupt_error = ""
    try:
        graph.invoke(
            {"messages": [HumanMessage(content=q2)], "permissions": perm},
            config=config,
        )
    except AgentInterrupted as exc:
        interrupted = True
        interrupt_error = str(exc)
    except Exception as exc:  # 兜底：其他异常也要在报告里体现
        interrupt_error = f"{type(exc).__name__}: {exc}"

    snap_after_stop = graph.get_state(config)
    stop_values = snap_after_stop.values or {}
    stop_data = stop_values.model_dump() if hasattr(stop_values, "model_dump") else dict(stop_values)
    incidents = read_incidents(THREAD)
    events_before = [e for e in (stop_data.get("trace") or []) if isinstance(e, dict)]
    next_nodes = list(getattr(snap_after_stop, "next", None) or ())
    seq_before = max([int(e.get("seq") or 0) for e in events_before] or [0])
    msgs_before = len(stop_data.get("messages") or [])

    # ---------------- 第 3 幕：重启续跑 ----------------
    set_abort_check(lambda tid: False)   # 清除中断标记（等价 /agent/resume 的前置动作）
    resume_error = ""
    try:
        result2 = graph.invoke(None, config=config)   # ★ input=None = 从断点继续
        state2 = result2 if isinstance(result2, dict) else dict(result2)
    except Exception as exc:
        resume_error = f"{type(exc).__name__}: {exc}"
        state2 = {}
    events_all = [e for e in (state2.get("trace") or []) if isinstance(e, dict)]
    events_after = [e for e in events_all if int(e.get("seq") or 0) > seq_before]
    snap_resume = graph.get_state(config)
    resume_values = snap_resume.values or {}
    resume_data = resume_values.model_dump() if hasattr(resume_values, "model_dump") else dict(resume_values)

    # 对照实验：新会话问同一问题，看模型能拿到多少历史
    fresh_config = {"configurable": {"thread_id": "demo-thread-fresh"}}
    graph.invoke(
        {"messages": [HumanMessage(content=q2)], "permissions": perm},
        config=fresh_config,
    )
    fresh_values = graph.get_state(fresh_config).values or {}
    fresh_data = fresh_values.model_dump() if hasattr(fresh_values, "model_dump") else dict(fresh_values)

    metrics_summary = recorder.summary()

    # ========================================================
    # 渲染报告
    # ========================================================
    md: List[str] = []
    md.append("# 养老智能体 · 全链路走查报告\n")
    md.append(f"> 生成时间：{started}　｜　线程：`{THREAD}`　｜　引擎：`react`（手写思考循环）\n")
    md.append(
        "> 本报告由 `fastap2/e2e_observability.py` 自动生成：走的是真实的图、真实的事件流、"
        "真实的 checkpoint 与中断/续跑代码路径，只有 LLM 的文本是脚本化的（零 token、可重复）。\n"
    )

    md.append("\n## 0. 装配（这一轮到底加载了什么）\n")
    md.append("| 项 | 值 |\n|---|---|")
    md.append(f"| 状态结构版本 | `v{CURRENT_STATE_VERSION}`（documents.py，含逐级迁移） |")
    md.append(f"| 状态文档域 | care.session / care.conversation / care.turn / care.usage / care.trace |")
    md.append(f"| 可用工具 | {', '.join(f'`{t.name}`' for t in tools)} |")
    md.append(f"| checkpointer | `InMemorySaver`（生产用 RedisSaver，语义相同） |")
    md.append(f"| 轮次上限 | 4（来自配置 `agent.max_rounds`） |")
    md.append(f"| 线程 | `{THREAD}` |")

    md.append("\n## 1. 用户提问\n")
    md.append("```text")
    md.append(f"用户：{q1}")
    md.append("```\n")

    md.append("## 2. 智能体的逐步思考（事件流）\n")
    md.append(
        "每个事件一行，`#` 是全局序号（续跑时从这里接着编，不重头）。"
        "`🧠 模型返回` 后面的缩进行是模型的思考内容、输出文本和它决定调用的工具。\n"
    )
    md.append(render_events(events1))

    md.append("\n## 3. 最终回答\n")
    md.append("```text")
    md.append(f"{parsed_dump.get('answer') or '(无)'}")
    md.append("```\n")
    md.append("| 结构化字段 | 值 |\n|---|---|")
    for key in ("health_advice", "diet_advice", "exercise_advice", "medication_advice",
                "risk_level", "emergency_flag"):
        md.append(f"| {key} | {parsed_dump.get(key)} |")
    md.append("\n**本轮成本（按模型分桶）**\n")
    md.append("| 模型 | 调用次数 | 输入 | 输出 | 缓存命中 | 成本（元） |\n|---|---|---|---|---|---|")
    for model_key, bucket in usage1.items():
        md.append(
            f"| `{model_key}` | {bucket.get('calls')} | {bucket.get('input_tokens')} "
            f"| {bucket.get('output_tokens')} | {bucket.get('cache_read')} "
            f"| {bucket.get('cost_yuan')} |"
        )

    md.append("\n## 4. checkpoint 快照时间线\n")
    md.append(
        "LangGraph 每跑完一个节点就落一次 checkpoint。下表是**每一步落盘后的状态**："
        "「下一步节点」为空表示这一轮已经跑完。\n"
    )
    md.append(render_snapshots(snaps1))

    md.append("\n## 5. 用户中断\n")
    md.append("```text")
    md.append(f"用户（第二轮）：{q2}")
    md.append("```\n")
    md.append(
        "模型已经完成第 1 次思考并决定调工具，此时用户点了「停止」。"
        "中断不是杀线程，而是给会话打标记 —— 图在**下一个节点入口**看到标记就停下。\n"
    )
    md.append("| 项 | 值 |\n|---|---|")
    md.append(f"| 是否中断 | {'是' if interrupted else '否'} |")
    md.append(f"| 中断异常 | `{_truncate(interrupt_error, 80)}` |")
    md.append(f"| 停在哪一步 | 已完成 {len(events_before)} 条事件，seq 水位 = {seq_before} |")
    md.append(f"| 快照还在吗 | {'在' if next_nodes else '不在'}（下一步待执行：`{','.join(next_nodes) or '(无)'}`） |")
    md.append(f"| 已保留消息数 | {msgs_before} 条 |")
    md.append(f"| 旁路事故记录 | {len(incidents)} 条（error/interrupted 不进 state，落旁路） |\n")

    if incidents:
        md.append("**中断事件（来自旁路，state 里看不到 —— 这正是 P0 修复的点）**\n")
        md.append(render_events(incidents))

    md.append("\n## 6. 重启对话：断点内容重新注入\n")
    md.append(
        "续跑时调用 `graph.invoke(None, config)` —— 第一个参数是 `None`，"
        "表示「不要新的用户消息，从最后一个 checkpoint 加载状态继续」，"
        "所以 `next` 里的节点会接着跑，历史消息、trace、工具黑板全部被重新注入。\n"
    )
    md.append("| 项 | 中断时 | 续跑后 | 说明 |\n|---|---|---|---|")
    md.append(
        f"| 消息数 | {msgs_before} | {len(resume_data.get('messages') or [])} "
        f"| 续跑把工具结果 + AI 回复补进了上下文 |"
    )
    md.append(
        f"| trace 长度 | {len(events_before)} | {len(events_all)} "
        f"| 事件继续追加，不清空 |"
    )
    md.append(
        f"| seq 水位 | {seq_before} | "
        f"{max([int(e.get('seq') or 0) for e in events_all] or [0])} "
        f"| 从断点接着编号，不重头 |"
    )
    md.append(f"| 轮次 | {stop_data.get('round')} | {resume_data.get('round')} | 轮次连续 |")
    md.append(f"| 续跑错误 | - | `{resume_error or '无'}` | - |\n")

    md.append("**续跑产生的新事件（seq 接在上次水位之后）**\n")
    md.append(render_events(events_after) if events_after else "_（无新事件）_")

    md.append("\n### 6.1 对照：新开一个会话问同样的问题\n")
    md.append(
        "同一个问题，如果是**新线程**（`demo-thread-fresh`），模型只看得到「这一问 + 这一答」；"
        "而续接原线程时，历史消息 + 断点状态全部被重新注入。这就是「断点内容重新注入」的实际差别。\n"
    )
    md.append("| 场景 | 线程 | 模型可见消息数 | 能否接着上一轮说 |\n|---|---|---|---|")
    md.append(f"| 新会话 | `demo-thread-fresh` | {len(fresh_data.get('messages') or [])} | 不能（从零开始） |")
    md.append(f"| 续接 | `{THREAD}` | {len(resume_data.get('messages') or [])} | 能（历史 + 断点全部注入） |")

    md.append("\n### 6.2 幂等：同一请求「重放」只生效一次\n")
    md.append(
        "第 6 幕是用户**有意**的中断后续跑；客户端超时重试、用户连点是**无意**的重放，"
        "靠下面四层挡住。这里实测的是 L1 的三态状态机：\n"
    )
    try:
        idem_rows, idem_live = _idem_demo()
    except Exception as exc:
        idem_rows, idem_live = [], False
        md.append(f"_幂等演示跳过：{type(exc).__name__}: {exc}_\n")
    if idem_rows:
        md.append(
            f"演示后端：{'真 Redis' if idem_live else '**Redis 不可用 → fail-open 降级（全部放行）**'}\n"
        )
        md.append("| 时刻 | 状态 | 会发生什么 |\n|---|---|---|")
        for label, state, note in idem_rows:
            md.append(f"| {label} | `{state}` | {note} |")
        md.append("")

    md.append("**四层各管一段，互补而非重复**\n")
    md.append("| 层 | 位置 | 挡什么 | 单测 |\n|---|---|---|---|")
    md.append(
        "| L1 HTTP 幂等键 | `core/idempotency.py` + `/chat` `/chat/stream` "
        "| 客户端重试、用户连点：`done` 回放 / `inflight` 409 | 13 例 |"
    )
    md.append(
        "| L2 工具台账 | `react_loop.act_node` 的 `tool_ledger`（key=`tool_call_id`） "
        "| checkpoint 重放 / 续跑：同一调用不重复执行 | 4 例 |"
    )
    md.append(
        "| L3 写工具 | `_invoke_with_retry(write=True)` + `mcp_servers/idempotency.py` "
        "| 写操作不盲目重试；声明了 `idempotency_key` 就由 MCP 端 SETNX 去重 | 9 例 |"
    )
    md.append(
        "| L4 线程登记 | `core/thread_registry.py` "
        "| 首次并发登记同一 thread 撞 unique → 复查，而不是 500 | 7 例 |"
    )
    md.append("")

    md.append("\n## 7. 指标增量（/metrics 里会看到的东西）\n")
    md.append("走查过程中注入的内存指标收集器捕获到的全部数据：\n")
    md.append("| 指标 | 次数 | 累计值 |\n|---|---|---|")
    for key, bucket in metrics_summary.items():
        md.append(f"| `{key}` | {bucket['count']} | {bucket['sum']} |")

    md.append("\n## 8. 这套东西对应到线上怎么看\n")
    md.append("| 想知道的事 | 看哪里 |\n|---|---|")
    md.append("| 这次请求都干了什么 | 日志按 `rid=` 过滤（HTTP 头 / SSE 首帧 `meta` 事件都有） |")
    md.append("| 某轮对话的完整步骤 | `GET /api/v1/agent/trace/{thread_id}`（含失败/中断的旁路事件） |")
    md.append("| 哪个模型烧的钱、哪个工具最慢 | `GET /api/v1/agent/metrics` + `GET /metrics` |")
    md.append("| 当前用什么配置、值从哪来 | `GET /api/v1/agent/config`（密钥已脱敏） |")
    md.append("| 中断 / 续跑 | `POST /interrupt` → `POST /resume` → `GET /trace` 核对 seq 接续 |")
    md.append("| 服务是不是持久化模式 | `GET /api/v1/agent/status` 的 `persistent` 字段 |")

    md.append("\n---\n")
    md.append(
        f"状态结构版本 v{CURRENT_STATE_VERSION}　｜　"
        f"事件总数 {len(events1) + len(events_after)}　｜　"
        f"中断 {1 if interrupted else 0} 次　｜　"
        f"续跑 {'成功' if state2 and not resume_error else '失败'}　｜　"
        f"报告文件 `{OUT_FILE.name}`"
    )

    OUT_FILE.write_text("\n".join(md), encoding="utf-8")
    set_sink(None)
    print(f"[walkthrough] 报告已生成：{OUT_FILE}")
    print(f"[walkthrough] 事件 {len(events1) + len(events_after)} 条 / 中断 {interrupted} / 续跑 {bool(state2)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
