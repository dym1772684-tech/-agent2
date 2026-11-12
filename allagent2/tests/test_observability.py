"""可观测 / 可恢复 的零 LLM 单测。

覆盖：
  1) request_id：contextvar 传播到线程池子线程（starlette 实际路径）
  2) 失败旁路：节点抛异常时 error 事件进不了 state，但能被 read_incidents 读到
  3) 中断：命中标记 → AgentInterrupted → checkpoint 停在上一 superstep
  4) 续跑：invoke(None) 从断点继续，且 trace seq 接着编（不重头）
  5) 指标：emit_event 统一打点，sink 能收到 node/llm/tool 三类
  6) 状态迁移：v1（无 schema_version）→ v2，且不丢业务字段
  7) 配置：来源追踪 + 密钥打码 + 非法值逐字段回退

运行：
    cd <项目根>
    python allagent2/tests/test_observability.py
"""
from __future__ import annotations

import sys
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
# fastap2 侧模块（core.request_id / observability）也在测，需把 fastap2 加进来；
# 注意 fastap2 必须在 allagent2 之后插入（bootstrap 里同样是这个顺序，
# 保证 `core` 解析到 fastap2/core 而不是 allagent2/core）
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "fastap2"))

from langchain.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver

from allagent2.agent.documents import CURRENT_STATE_VERSION, migrate_state_dict
from allagent2.agent.observe import (
    AgentInterrupted,
    read_incidents,
    set_abort_check,
)
from allagent2.agent.react_loop import build_react_graph
from allagent2.agent.state import ParsedOutput, PermissionContext
from allagent2.guance.metrics_hook import set_sink


def _usage(prompt=100, completion=10, cache=40):
    return {
        "input_tokens": prompt,
        "output_tokens": completion,
        "total_tokens": prompt + completion,
        "input_token_details": {"cache_read": cache},
        "output_token_details": {},
    }


class FakeStructured:
    def invoke(self, messages, **kwargs):
        return {"parsed": ParsedOutput(answer="好的"), "raw": None, "parsing_error": None}


class FakeModel:
    def __init__(self, responses):
        self.responses = list(responses)
        self.index = 0
        self.structured = FakeStructured()
        self.model_name = "fake-model"

    def bind_tools(self, tools):
        return self

    def with_structured_output(self, schema, include_raw=True):
        return self.structured

    def invoke(self, messages, **kwargs):
        pos = min(self.index, len(self.responses) - 1)
        self.index += 1
        return self.responses[pos]


class FakeTool:
    def __init__(self, name, result="ok"):
        self.name = name
        self.result = result
        self.calls = 0

    def invoke(self, args):
        self.calls += 1
        return self.result


def _tool_call(name="t", args=None):
    return AIMessage(
        content="",
        tool_calls=[{"name": name, "args": args or {"x": 1}, "id": f"c_{name}", "type": "tool_call"}],
        usage_metadata=_usage(),
    )


def _text(text="好的"):
    return AIMessage(content=text, usage_metadata=_usage(90, 12, 30))


def _perm(tools=("t",)):
    return PermissionContext(allowed_tools=list(tools), allowed_elder_ids=["E1"])


def _invoke(graph, question, thread, perm=None):
    return graph.invoke(
        {
            "messages": [HumanMessage(content=question)],
            "permissions": perm or _perm(),
            "schema_version": CURRENT_STATE_VERSION,
        },
        config={"configurable": {"thread_id": thread}},
    )


class T1RequestId(unittest.TestCase):
    def test_thread_pool_propagation(self):
        """starlette 的 run_in_threadpool 底层是 anyio.to_thread.run_sync，
        它会复制 context → rid 必须跟着进 worker 线程（/chat 的真实路径）。
        """
        import anyio

        from core.request_id import get_request_id, new_request_id, set_request_id

        rid = new_request_id()
        set_request_id(rid)

        async def _run():
            return await anyio.to_thread.run_sync(get_request_id)

        self.assertEqual(anyio.run(_run), rid)

    def test_bare_thread_pool_does_not_propagate(self):
        """反例：裸 ThreadPoolExecutor 不复制 context，rid 会退化成 "-"。

        写这条是为了把"哪些路径能传播"钉死在测试里 —— 将来换执行器时
        如果忘了显式 copy_context，这里会提醒。
        """
        from core.request_id import get_request_id, new_request_id, set_request_id

        set_request_id(new_request_id())
        with ThreadPoolExecutor(max_workers=1) as pool:
            got = list(pool.map(lambda _: get_request_id(), [None]))[0]
        self.assertEqual(got, "-")

    def test_default_placeholder(self):
        from core.request_id import get_request_id

        self.assertTrue(get_request_id(), "未设置时也必须回落成可打印的值，不能是 None")


class T2IncidentBuffer(unittest.TestCase):
    def test_error_lands_in_incidents(self):
        """节点异常：error 事件进不了 state["trace"]，必须能从旁路读回来。"""
        thread = "t-fail-buffer"

        def boom(state, runtime):
            raise RuntimeError("模拟节点崩溃")

        from allagent2.agent.observe import traced_node

        with self.assertRaises(RuntimeError):
            traced_node("boom")(boom)(
                {"trace": []}, _Runtime(thread)
            )
        got = read_incidents(thread)
        self.assertTrue(got, "旁路里应该有事故记录")
        self.assertEqual(got[-1]["type"], "error")
        self.assertIn("模拟节点崩溃", got[-1]["error"])


class _Runtime:
    """最小 Runtime 替身：只需要 execution_info.thread_id。"""

    class _Info:
        def __init__(self, tid):
            self.thread_id = tid

    def __init__(self, tid):
        self.execution_info = self._Info(tid)


class T3InterruptResume(unittest.TestCase):
    def tearDown(self):
        set_abort_check(None)   # 全局状态，别污染其他用例

    def _graph(self, thread):
        tool = FakeTool("t")
        model = FakeModel([_tool_call("t"), _text("查到了")])
        graph = build_react_graph(
            model=model, tools=[tool],
            checkpointer=InMemorySaver(), max_rounds=3,
        )
        return graph, tool

    def test_interrupt_keeps_checkpoint(self):
        thread = "t-interrupt"
        graph, tool = self._graph(thread)
        set_abort_check(lambda tid: True)   # 一进去就中断

        with self.assertRaises(AgentInterrupted):
            _invoke(graph, "帮我查一下", thread)

        snap = graph.get_state({"configurable": {"thread_id": thread}})
        self.assertTrue(getattr(snap, "next", None), "断点应该保留待执行节点")
        self.assertEqual(tool.calls, 0, "中断后工具不应被执行")
        self.assertEqual(read_incidents(thread)[-1]["type"], "interrupted")

    def test_resume_continues_without_new_message(self):
        thread = "t-resume"
        graph, tool = self._graph(thread)
        set_abort_check(lambda tid: True)

        with self.assertRaises(AgentInterrupted):
            _invoke(graph, "帮我查一下", thread)

        before = graph.get_state({"configurable": {"thread_id": thread}})
        msgs_before = len((before.values or {}).get("messages") or [])
        seq_before = max(
            [int(e.get("seq") or 0) for e in ((before.values or {}).get("trace") or [])]
            or [0]
        )
        next_nodes = list(getattr(before, "next", None) or ())
        self.assertIn("guard", next_nodes)

        set_abort_check(lambda tid: False)
        result = graph.invoke(None, config={"configurable": {"thread_id": thread}})

        after = graph.get_state({"configurable": {"thread_id": thread}})
        data = after.values or {}
        trace = [e for e in (data.get("trace") or []) if isinstance(e, dict)]
        self.assertGreaterEqual(len(data.get("messages") or []), msgs_before)
        # 续跑后 seq 单调递增，且大于中断前的水位
        seqs = [int(e.get("seq") or 0) for e in trace]
        self.assertEqual(seqs, sorted(seqs), "seq 必须单调递增")
        if seqs:
            self.assertGreater(max(seqs), seq_before)
        self.assertIsNotNone(result)


class T4Metrics(unittest.TestCase):
    def test_event_to_metric(self):
        rows = []
        set_sink(lambda name, value, labels: rows.append((name, value, dict(labels))))
        try:
            thread = "t-metrics"
            graph = build_react_graph(
                model=FakeModel([_tool_call("t"), _text("好了")]),
                tools=[FakeTool("t")],
                checkpointer=InMemorySaver(),
                max_rounds=3,
            )
            _invoke(graph, "查一下", thread)
        finally:
            set_sink(None)

        names = {r[0] for r in rows}
        self.assertIn("agent_node_total", names)
        self.assertIn("agent_llm_seconds", names)
        self.assertIn("agent_tool_total", names)
        self.assertIn("agent_llm_tokens_total", names)
        # label 必须带出来，否则指标无法按节点/工具聚合
        node_rows = [r for r in rows if r[0] == "agent_node_total"]
        self.assertTrue(all("node" in r[2] for r in node_rows))


class T5Migration(unittest.TestCase):
    def test_v1_to_current(self):
        old = {
            "messages": ["m"],
            "trace": [{"type": "node_start", "seq": 5}],
            "round": 2,
            "usage": {"total": 1},
            "session": "dead-field",
        }
        new, frm, notes = migrate_state_dict(old)
        self.assertEqual(frm, 1)
        self.assertEqual(new["schema_version"], CURRENT_STATE_VERSION)
        self.assertEqual(new["round"], 2)
        self.assertNotIn("session", new, "历史垃圾字段应被清掉")
        self.assertIn("usage_models", new)
        self.assertIn("interrupt", new)
        self.assertTrue(notes)

    def test_garbage_input_is_safe(self):
        new, _, notes = migrate_state_dict("不是 dict")
        self.assertEqual(new, {})
        self.assertTrue(notes)


class T6Config(unittest.TestCase):
    def test_source_and_mask(self):
        from allagent2.config.loader import load

        tmp = Path(__file__).parent / "_tmp_cfg.env"
        tmp.write_text(
            "OPENAI_API_KEY=sk-test-1234567890\n"
            "OPENAI_BASE_URL=${MISSING_BASE_URL}\n"
            "AGENT_MAX_ROUNDS=not-a-number\n",
            encoding="utf-8",
        )
        try:
            loaded = load((str(tmp),))
        finally:
            tmp.unlink(missing_ok=True)

        self.assertEqual(loaded.sources["llm.api_key"], "env_file")
        # 引用了不存在的变量 → 整值按未配置处理（不做半截替换）
        self.assertEqual(loaded.sources["llm.base_url"], "missing")
        # 类型非法 → 只回退这一项，其他项照常生效
        self.assertEqual(loaded.sources["agent.max_rounds"], "default")
        self.assertEqual(loaded.settings.agent.max_rounds, 6)
        self.assertEqual(loaded.settings.agent.engine, "react")
        self.assertTrue(loaded.llm_ready)

        shown = loaded.describe()["llm.api_key"]["value"]
        self.assertNotIn("1234567890", shown, "密钥不能回传真值")
        self.assertIn("len=", shown)


if __name__ == "__main__":
    unittest.main(verbosity=2)
