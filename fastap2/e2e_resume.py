"""断点续跑场景批量验证：把 datasets/resume_cases.jsonl 的 8 个场景真跑一遍。

============================================================
它和 e2e_observability.py 的分工
------------------------------------------------------------
e2e_observability.py   跑「一条」完整故事(问答→中断→续跑)，产出可读的走查报告。
本脚本                 跑「一组」中断点不同的场景，逐条断言续跑行为是否符合预期，
                       产出对照表 —— 用来回归"中断/续跑"这块能力本身有没有坏。

8 个场景覆盖：think 前断 / act 前断(工具零执行) / observe 前断(工具已执行) /
finalize 前断 / 中断后改发新消息 / 中断后换新线程 / 连续两次中断 / seq 接续不撞号。

============================================================
为什么用假模型
------------------------------------------------------------
零 token、不依赖 vLLM/MCP/MySQL，随时可重跑。走的是**真实的图、真实的
checkpoint、真实的中断与续跑代码路径**，只有 LLM 的文本是脚本化的。

运行：
    cd <项目根>            # 练手agent
    python fastap2/e2e_resume.py
输出：fastap2/resume_test_report.md
"""
from __future__ import annotations

import collections
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

FASTAP2_DIR = Path(__file__).resolve().parent
WORKSPACE = FASTAP2_DIR.parent
for p in (str(WORKSPACE), str(FASTAP2_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

from langchain.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver

from allagent2.agent.documents import CURRENT_STATE_VERSION
from allagent2.agent.observe import AgentInterrupted, set_abort_check
from allagent2.agent.react_loop import build_react_graph
from allagent2.agent.state import ParsedOutput, PermissionContext

CASES_FILE = FASTAP2_DIR / "eval" / "datasets" / "resume_cases.jsonl"
OUT_FILE = FASTAP2_DIR / "resume_test_report.md"

TOOL_NAME = "query_health_records"
TOOL_RESULT = "老人 E001 近 7 天血压：150/95、148/92、152/96（收缩压持续偏高）"


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


def _tool_msg(name, args, call_id="call_1"):
    from langchain.messages import AIMessage

    return AIMessage(
        content="",
        tool_calls=[{"name": name, "args": args, "id": call_id, "type": "tool_call"}],
        usage_metadata=_usage(),
        response_metadata={"model_name": "demo-model-7b"},
    )


def _text_msg(text, thinking=""):
    from langchain.messages import AIMessage

    extra = {"reasoning_content": thinking} if thinking else {}
    return AIMessage(
        content=text,
        usage_metadata=_usage(prompt=760, completion=130, cache=480),
        response_metadata={"model_name": "demo-model-7b"},
        additional_kwargs=extra,
    )


class FakeStructured:
    def __init__(self, value=None):
        self.value = value

    def invoke(self, messages, **kwargs):
        return {"parsed": self.value, "raw": None, "parsing_error": None}


class ScriptedLLM:
    """按脚本返回固定响应（不模拟中断：中断由 abort 钩子在节点入口精确控制）。"""

    def __init__(self, responses, structured=None):
        self.responses = list(responses)
        self.index = 0
        self.structured = structured or FakeStructured()
        self.model_name = "demo-model-7b"

    def bind_tools(self, tools):
        return self

    def with_structured_output(self, schema, include_raw=True):
        return self.structured

    def invoke(self, messages, **kwargs):
        pos = min(self.index, len(self.responses) - 1)
        self.index += 1
        return self.responses[pos]


class FakeTool:
    def __init__(self, name, result):
        self.name = name
        self.result = result
        self.calls = 0

    def invoke(self, args):
        self.calls += 1
        time.sleep(0.005)
        return self.result


def _responses(two_tool_rounds: bool = False):
    """首轮：决定调工具；次轮：给答案。多给几条，续跑重新进 think 也有得用。

    two_tool_rounds=True 时前两次都调工具，用来造出「第二轮 act」——
    RC08 要断在第 2 次 act 之前，验证大事件量下 seq 接续。
    """
    if two_tool_rounds:
        return [
            _tool_msg(TOOL_NAME, {"elder_id": "E001"}, call_id="call_1"),
            _tool_msg(TOOL_NAME, {"elder_id": "E001"}, call_id="call_2"),
            _text_msg(
                "老人近 7 天收缩压 148~152，持续偏高。建议规律测量并控制食盐摄入。",
                thinking="两次记录已交叉核对，收缩压均超过 140，属于持续偏高。",
            ),
        ]
    return [
        _tool_msg(TOOL_NAME, {"elder_id": "E001"}),
        _text_msg(
            "老人近 7 天收缩压 148~152，持续偏高。建议规律测量并控制食盐摄入。",
            thinking="已拿到近 7 天血压记录，收缩压均超过 140，属于持续偏高。",
        ),
        _text_msg("建议每日食盐控制在 5g 以内，少吃腌制食品，多吃富钾食物。"),
    ]


def _structured():
    return FakeStructured(
        ParsedOutput(
            answer="老人近 7 天收缩压 148~152，持续偏高，建议规律测量并控制食盐摄入。",
            health_advice="每日固定时间测量并记录。",
            diet_advice="每日食盐 <5g，少吃腌制食品。",
            exercise_advice="以散步为主，每次 20~30 分钟。",
            risk_level="中",
            emergency_flag=False,
        )
    )


# ============================================================
# 中断钩子：只在「指定节点的第 N 次进入」返回 True
# ============================================================
def make_abort(thread_id: str, spec: Dict[str, Any], once: bool = True):
    """spec = {"node": "act", "occurrence": 1}。

    abort 回调签名是 fn(tid, node) —— observe.set_abort_check 会自动识别
    两参签名并把节点名传进来，所以能精确断在"第 N 次进入 act 之前"。
    """
    counts = collections.Counter()
    fired = {"v": False}

    def _check(tid: str, node: str | None = None) -> bool:
        if tid != thread_id or node is None:
            return False
        if once and fired["v"]:
            return False
        if node != spec["node"]:
            return False
        counts[node] += 1
        if counts[node] >= int(spec.get("occurrence", 1)):
            fired["v"] = True
            return True
        return False

    return _check


def _values(graph, config) -> Dict[str, Any]:
    snap = graph.get_state(config)
    v = snap.values if snap is not None else {}
    return v.model_dump() if hasattr(v, "model_dump") else dict(v or {})


def _next_nodes(graph, config) -> List[str]:
    snap = graph.get_state(config)
    return list(getattr(snap, "next", None) or ())


def _seq_stats(events: List[Dict[str, Any]]):
    seqs = [int(e.get("seq") or 0) for e in events if isinstance(e, dict)]
    return (max(seqs) if seqs else 0), len(seqs) - len(set(seqs))


# ============================================================
# 跑一个场景
# ============================================================
def run_case(case: Dict[str, Any]) -> Dict[str, Any]:
    tid = case["thread_id"]
    spec = case["interrupt_at"]
    resume_mode = case.get("resume_mode", "resume")

    tool = FakeTool(TOOL_NAME, TOOL_RESULT)
    llm = ScriptedLLM(_responses(bool(case.get("two_tool_rounds"))), _structured())
    graph = build_react_graph(
        model=llm, tools=[tool], checkpointer=InMemorySaver(), max_rounds=6
    )
    perm = PermissionContext(allowed_tools=[TOOL_NAME], allowed_elder_ids=["E001"])
    config = {"configurable": {"thread_id": tid}}

    def _init(question: str, with_perm: bool = True) -> Dict[str, Any]:
        s: Dict[str, Any] = {
            "messages": [HumanMessage(content=question)],
            "schema_version": CURRENT_STATE_VERSION,
        }
        if with_perm:
            s["permissions"] = perm
        return s

    row: Dict[str, Any] = {
        "id": case["id"],
        "scenario": case["scenario"],
        "expect": case.get("expect") or {},
        "resume_mode": resume_mode,
        "interrupted": False,
        "stop_node": "-",
        "seq_before": 0,
        "msgs_before": 0,
        "seq_after": 0,
        "msgs_after": 0,
        "dup_seq": 0,
        "tool_calls": 0,
        "error": "",
        "reasons": [],
    }

    # ---------- 第一轮：跑到中断为止 ----------
    set_abort_check(make_abort(tid, spec))
    try:
        graph.invoke(_init(case["turns"][0]), config=config)
    except AgentInterrupted:
        row["interrupted"] = True
    except Exception as exc:
        row["error"] = f"{type(exc).__name__}: {exc}"

    stop_data = _values(graph, config)
    stop_events = [e for e in (stop_data.get("trace") or []) if isinstance(e, dict)]
    seq_before, _ = _seq_stats(stop_events)
    row["stop_node"] = ",".join(_next_nodes(graph, config)) or "-"
    row["seq_before"] = seq_before
    row["msgs_before"] = len(stop_data.get("messages") or [])
    row["tool_calls_after_stop"] = tool.calls

    # ---------- 续跑 ----------
    # 真实链路里 /resume 会先删 Redis 中断标记；这里等价于把钩子清掉。
    set_abort_check(None)

    if resume_mode == "fresh_thread":
        fresh_config = {"configurable": {"thread_id": f"{tid}-fresh"}}
        try:
            graph.invoke(_init(case["turns"][0]), config=fresh_config)
        except Exception as exc:
            row["error"] = f"{type(exc).__name__}: {exc}"
        fresh = _values(graph, fresh_config)
        row["msgs_after"] = len(fresh.get("messages") or [])
        fevents = [e for e in (fresh.get("trace") or []) if isinstance(e, dict)]
        row["seq_after"], row["dup_seq"] = _seq_stats(fevents)
    elif resume_mode == "new_message":
        follow = case["turns"][1] if len(case["turns"]) > 1 else "那饮食上要注意什么"
        try:
            graph.invoke({"messages": [HumanMessage(content=follow)]}, config=config)
        except Exception as exc:
            row["error"] = f"{type(exc).__name__}: {exc}"
        data = _values(graph, config)
        events = [e for e in (data.get("trace") or []) if isinstance(e, dict)]
        row["msgs_after"] = len(data.get("messages") or [])
        row["seq_after"], row["dup_seq"] = _seq_stats(events)
    else:
        # RC07：续跑时再断一次（第二次中断），验证中断标记可重复设置
        again = case.get("interrupt_again_at")
        if again:
            set_abort_check(make_abort(tid, again))
            try:
                graph.invoke(None, config=config)
            except AgentInterrupted:
                row["interrupted_again"] = True
            except Exception as exc:
                row["error"] = f"{type(exc).__name__}: {exc}"
            stop2 = _values(graph, config)
            row["stop_node"] = ",".join(_next_nodes(graph, config)) or "-"
            row["seq_before"], _ = _seq_stats(
                [e for e in (stop2.get("trace") or []) if isinstance(e, dict)]
            )
            set_abort_check(None)

        try:
            graph.invoke(None, config=config)   # ★ input=None = 从断点继续
        except Exception as exc:
            row["error"] = f"{type(exc).__name__}: {exc}"
        data = _values(graph, config)
        events = [e for e in (data.get("trace") or []) if isinstance(e, dict)]
        row["msgs_after"] = len(data.get("messages") or [])
        row["seq_after"], row["dup_seq"] = _seq_stats(events)

    row["tool_calls"] = tool.calls
    set_abort_check(None)

    # ---------- 判定 ----------
    exp = row["expect"]
    if exp.get("stopped_before"):
        if row["stop_node"] != exp["stopped_before"]:
            row["reasons"].append(
                f"停在了 {row['stop_node']}，预期 {exp['stopped_before']}"
            )
    if exp.get("seq_continuous") and row["seq_after"] <= row["seq_before"]:
        row["reasons"].append(f"seq 没接续：{row['seq_before']} → {row['seq_after']}")
    if exp.get("seq_unique") and row["dup_seq"]:
        row["reasons"].append(f"seq 撞号 {row['dup_seq']} 处")
    if exp.get("messages_grow") is True and row["msgs_after"] <= row["msgs_before"]:
        row["reasons"].append(
            f"消息没增长：{row['msgs_before']} → {row['msgs_after']}"
        )
    if exp.get("messages_grow") is False and row["msgs_after"] != row["msgs_before"]:
        row["reasons"].append(
            f"消息数应为 {row['msgs_before']}，实际 {row['msgs_after']}"
        )
    if exp.get("tool_replay") is False and row["tool_calls"] > 1:
        row["reasons"].append(f"工具重复执行 {row['tool_calls']} 次（应 ≤1）")
    if row["error"]:
        row["reasons"].append(f"异常：{row['error']}")

    row["status"] = "PASS" if not row["reasons"] else "FAIL"
    return row


# ============================================================
# 报告
# ============================================================
def render(rows: List[Dict[str, Any]], started: str) -> str:
    ok = sum(1 for r in rows if r["status"] == "PASS")
    md: List[str] = []
    md.append("# 断点续跑场景验证报告\n")
    md.append(
        f"> 生成时间：{started}　｜　场景数：{len(rows)}　｜　通过：{ok}/{len(rows)}　"
        f"｜　引擎：`react`（手写思考循环）\n"
    )
    md.append(
        "> 由 `fastap2/e2e_resume.py` 自动生成：走真实的图、真实的 checkpoint、"
        "真实的中断与续跑代码路径，只有 LLM 文本是脚本化的（零 token、可重复）。\n"
    )

    md.append("## 一、总览\n")
    md.append("| # | 场景 | 中断 | 停在 | seq 中断前→续跑后 | 消息数 | 工具执行 | 结论 |")
    md.append("|---|---|---|---|---|---|---|---|")
    for r in rows:
        mark = "✅ PASS" if r["status"] == "PASS" else "❌ FAIL"
        md.append(
            f"| {r['id']} | {r['scenario']} | {'是' if r['interrupted'] else '否'} "
            f"| `{r['stop_node']}` | {r['seq_before']} → {r['seq_after']} "
            f"| {r['msgs_before']} → {r['msgs_after']} "
            f"| {r['tool_calls']} 次 | {mark} |"
        )
    md.append("")

    md.append("## 二、逐条断言\n")
    for r in rows:
        exp = r["expect"]
        md.append(f"### {r['id']} · {r['scenario']}\n")
        if exp.get("note"):
            md.append(f"**预期要点**：{exp['note']}\n")
        md.append("| 断言 | 预期 | 实际 |")
        md.append("|---|---|---|")
        if exp.get("stopped_before"):
            md.append(
                f"| 中断位置 | 停在 `{exp['stopped_before']}` 前 | `{r['stop_node']}` |"
            )
        if exp.get("seq_continuous"):
            md.append(
                f"| seq 接续 | 续跑后水位 > 中断前 | {r['seq_before']} → {r['seq_after']} |"
            )
        if exp.get("seq_unique"):
            md.append(f"| seq 唯一 | 无重复 | 重复 {r['dup_seq']} 处 |")
        if exp.get("messages_grow") is not None:
            md.append(
                f"| 消息数 | {'增长' if exp['messages_grow'] else '不变'} "
                f"| {r['msgs_before']} → {r['msgs_after']} |"
            )
        if exp.get("tool_replay") is False:
            md.append(f"| 工具不重放 | ≤1 次 | {r['tool_calls']} 次 |")
        md.append("")
        if r["reasons"]:
            md.append("**失败原因**：\n")
            for x in r["reasons"]:
                md.append(f"- {x}")
            md.append("")
        else:
            md.append("全部断言通过。\n")

    md.append("## 三、这 8 个场景在证明什么\n")
    md.append("| 中断点 | 证明的能力 |")
    md.append("|---|---|")
    md.append("| `think` 前 | 图还没算就被停下，checkpoint 仍保留在上一 superstep |")
    md.append("| `act` 前 | 最常见的场景：断在工具执行前，续跑要真的执行一次、且只执行一次 |")
    md.append("| `observe` 前 | 工具已完成，续跑**不能重放工具**（L2 台账生效） |")
    md.append("| `finalize` 前 | 答案已生成，续跑只需补结构化，消息数不变 |")
    md.append("| 改发新消息 | 不续跑也能带上历史 → 证明 checkpoint 同时是「多轮记忆」 |")
    md.append("| 换新线程 | 对照组：新线程没有历史 → 证明「断点注入」是真实发生的 |")
    md.append("| 连续两次中断 | 中断标记可重复设置，每次都不污染 seq 水位 |")
    md.append("| seq 接续 | 中断/异常事件占号但不进 state，靠 `_incident_seq` 补水，不撞号 |")
    md.append("")
    md.append(
        "---\n\n续跑成功 "
        f"{ok}/{len(rows)}　｜　报告文件 `resume_test_report.md`\n"
    )
    return "\n".join(md)


def main() -> int:
    started = time.strftime("%Y-%m-%d %H:%M:%S")
    cases = [
        json.loads(line)
        for line in CASES_FILE.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    print(f"待跑 {len(cases)} 个续跑场景…")
    rows = []
    for case in cases:
        r = run_case(case)
        rows.append(r)
        icon = "✅" if r["status"] == "PASS" else "❌"
        print(f"  {icon} {r['id']} {r['scenario'][:26]} "
              f"停在={r['stop_node']} seq {r['seq_before']}→{r['seq_after']}")
        if r["reasons"]:
            for x in r["reasons"]:
                print(f"      - {x}")
    OUT_FILE.write_text(render(rows, started), encoding="utf-8")
    ok = sum(1 for r in rows if r["status"] == "PASS")
    print(f"\n通过 {ok}/{len(rows)}，报告：{OUT_FILE}")
    return 0 if ok == len(rows) else 1


if __name__ == "__main__":
    sys.exit(main())
