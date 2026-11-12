"""手写 ReAct 循环的零 LLM 单测。

用假模型（不联网、不调 vLLM）验证：
  1) 五节点流转与事件流顺序（node_start → llm_start → llm_end → node_end）
  2) 工具执行事件（tool_start / tool_end）与结果回填
  3) 工具异常不炸链路，is_error=True 落进事件
  4) 工具白名单越权拦截，且工具本体没有被执行
  5) 急救规则短路：不产生 llm 事件、emergency_flag=True
  6) 轮次上限：达到 max_rounds 直接 finalize，不再执行工具
  7) finalize 结构化失败降级为纯文本
  8) token / cost 写进 llm_end
  9) trace 序号连续

运行：
    cd <项目根>
    python allagent2/tests/test_react_loop.py
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from langchain.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver

from allagent2.agent.react_loop import build_react_graph
from allagent2.agent.state import ParsedOutput, PermissionContext
from allagent2.prompts.prompts import EMERGENCY_KEYWORDS


# ============================================================
# 假模型 / 假工具
# ============================================================
def _usage(prompt=100, completion=10, cache=40):
    return {
        "input_tokens": prompt,
        "output_tokens": completion,
        "total_tokens": prompt + completion,
        "input_token_details": {"cache_read": cache},
        "output_token_details": {},
    }


class FakeStructured:
    """模拟 with_structured_output(include_raw=True) 的返回。"""

    def __init__(self, value=None, boom: bool = False):
        self.value = value
        self.boom = boom
        self.calls = 0

    def invoke(self, messages, **kwargs):
        self.calls += 1
        if self.boom:
            raise RuntimeError("structured boom")
        return {"parsed": self.value, "raw": None, "parsing_error": None}


class FakeModel:
    """按脚本轮转返回 AIMessage。"""

    def __init__(self, responses, structured=None):
        self.responses = list(responses)
        self.index = 0
        self.structured = structured if structured is not None else FakeStructured()
        self.invoke_count = 0

    def bind_tools(self, tools):
        return self

    def with_structured_output(self, schema, include_raw=True):
        return self.structured

    def invoke(self, messages, **kwargs):
        self.invoke_count += 1
        pos = min(self.index, len(self.responses) - 1)
        self.index += 1
        return self.responses[pos]


class FakeTool:
    def __init__(self, name, result="ok", error=None):
        self.name = name
        self.result = result
        self.error = error
        self.calls = []

    def invoke(self, args):
        self.calls.append(args)
        if self.error:
            raise self.error
        return self.result


def _tool_call_msg(name="fake_tool", args=None, call_id="call_1", usage=None):
    return AIMessage(
        content="",
        tool_calls=[
            {"name": name, "args": args or {"x": 1}, "id": call_id, "type": "tool_call"}
        ],
        usage_metadata=usage or _usage(),
    )


def _text_msg(text="好的", usage=None):
    return AIMessage(content=text, usage_metadata=usage or _usage())


def _perm(tools=("fake_tool",), elders=None):
    return PermissionContext(allowed_tools=list(tools), allowed_elder_ids=elders)


def _run(graph, question="帮我算一下", perm=None, thread="t1"):
    return graph.invoke(
        {
            "messages": [HumanMessage(content=question)],
            "permissions": perm if perm is not None else _perm(),
        },
        config={"configurable": {"thread_id": thread}},
    )


def _types(trace):
    return [e.get("type") for e in trace]


# ============================================================
# 用例
# ============================================================
class TestReactLoop(unittest.TestCase):
    def _graph(self, model, tools=None, **kwargs):
        return build_react_graph(
            model=model,
            tools=tools if tools is not None else [FakeTool("fake_tool", "42")],
            checkpointer=InMemorySaver(),
            **kwargs,
        )

    def test_single_turn_no_tool(self):
        """无工具调用：guard → think → finalize，事件流完整。"""
        model = FakeModel(
            [_text_msg("你好")],
            structured=FakeStructured(ParsedOutput(answer="你好")),
        )
        result = _run(self._graph(model), question="打个招呼")
        trace = result["trace"]
        types = _types(trace)

        self.assertEqual(types.count("node_start"), 3)  # guard / think / finalize
        self.assertIn("llm_start", types)
        self.assertIn("llm_end", types)
        self.assertIn("structured_end", types)
        self.assertNotIn("tool_start", types)
        self.assertEqual(result["structured_response"].answer, "你好")

    def test_tool_call_round_trip(self):
        """一次工具调用：think → act → observe → think → finalize。"""
        model = FakeModel(
            [_tool_call_msg("fake_tool", {"x": 1}), _text_msg("结果是 42")],
            structured=FakeStructured(ParsedOutput(answer="结果是 42")),
        )
        tool = FakeTool("fake_tool", "42")
        result = _run(self._graph(model, [tool]))
        types = _types(result["trace"])

        self.assertEqual(types.count("tool_start"), 1)
        self.assertEqual(types.count("tool_end"), 1)
        self.assertEqual(types.count("llm_end"), 2)  # 两次思考
        self.assertEqual(tool.calls, [{"x": 1}])
        # 工具结果进了消息历史
        self.assertTrue(
            any(getattr(m, "type", "") == "tool" for m in result["messages"])
        )

    def test_tool_error_is_recorded(self):
        """工具抛错：不炸链路，tool_end.is_error=True。"""
        model = FakeModel(
            [_tool_call_msg("fake_tool"), _text_msg("工具挂了，先记下")],
            structured=FakeStructured(ParsedOutput(answer="工具挂了")),
        )
        tool = FakeTool("fake_tool", error=RuntimeError("boom"))
        result = _run(self._graph(model, [tool]))
        ends = [e for e in result["trace"] if e.get("type") == "tool_end"]
        self.assertEqual(len(ends), 1)
        self.assertTrue(ends[0]["is_error"])

    def test_tool_permission_denied(self):
        """白名单外的工具：拦截且不执行工具本体。"""
        model = FakeModel(
            [_tool_call_msg("dangerous_tool"), _text_msg("无权调用")],
            structured=FakeStructured(ParsedOutput(answer="无权调用")),
        )
        tool = FakeTool("dangerous_tool", "secret")
        graph = build_react_graph(
            model=model,
            tools=[tool],
            checkpointer=InMemorySaver(),
        )
        result = _run(graph, perm=_perm(tools=["fake_tool"]))
        ends = [e for e in result["trace"] if e.get("type") == "tool_end"]
        self.assertEqual(len(ends), 1)
        self.assertTrue(ends[0]["is_error"])
        self.assertEqual(tool.calls, [])  # 根本没执行

    def test_emergency_short_circuit(self):
        """急救关键词：guard 短路，不产生任何 llm 事件。"""
        keyword = EMERGENCY_KEYWORDS[0]
        model = FakeModel([_text_msg("不该被调用")])
        result = _run(self._graph(model), question=f"老人{keyword}了怎么办")
        types = _types(result["trace"])

        self.assertTrue(result.get("emergency_flag"))
        self.assertNotIn("llm_start", types)
        self.assertEqual(model.invoke_count, 0)
        self.assertEqual(types.count("node_start"), 1)  # 只有 guard

    def test_round_limit(self):
        """轮次上限：达到上限直接收尾，不再重复调工具。"""
        model = FakeModel(
            [_tool_call_msg("fake_tool"), _tool_call_msg("fake_tool")],
            structured=FakeStructured(ParsedOutput(answer="先到这儿")),
        )
        tool = FakeTool("fake_tool", "42")
        result = _run(self._graph(model, [tool], max_rounds=1))
        types = _types(result["trace"])

        self.assertEqual(types.count("llm_end"), 1)
        self.assertNotIn("tool_start", types)  # 第 1 轮就到上限，工具未执行

    def test_finalize_degrades(self):
        """结构化失败：降级为最近一条 AI 文本。"""
        model = FakeModel(
            [_text_msg("这是自然语言回答")],
            structured=FakeStructured(boom=True),
        )
        result = _run(self._graph(model))
        parsed = result["structured_response"]
        self.assertEqual(parsed.answer, "这是自然语言回答")
        ends = [e for e in result["trace"] if e.get("type") == "structured_end"]
        self.assertFalse(ends[0]["ok"])

    def test_tokens_and_cost_recorded(self):
        """llm_end 记录 token 与 cost。"""
        model = FakeModel(
            [_text_msg("hi")],
            structured=FakeStructured(ParsedOutput(answer="hi")),
        )
        result = _run(self._graph(model))
        llm_end = [e for e in result["trace"] if e.get("type") == "llm_end"][0]
        self.assertEqual(llm_end["tokens"]["input_tokens"], 100)
        self.assertEqual(llm_end["tokens"]["cache_read"], 40)
        self.assertIn("cost", llm_end)

    def test_seq_is_continuous(self):
        """事件序号从 1 开始且连续。"""
        model = FakeModel(
            [_tool_call_msg("fake_tool"), _text_msg("done")],
            structured=FakeStructured(ParsedOutput(answer="done")),
        )
        result = _run(self._graph(model, [FakeTool("fake_tool", "42")]))
        seqs = [e["seq"] for e in result["trace"]]
        self.assertEqual(seqs, list(range(1, len(seqs) + 1)))


if __name__ == "__main__":
    unittest.main(verbosity=2)
