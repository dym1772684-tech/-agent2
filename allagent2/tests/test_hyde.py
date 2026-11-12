"""HyDE + 三路融合零 LLM 单测：不依赖 MySQL / Chroma / 网络 / API key。

覆盖：
  - relative 三路融合：权重归一化、hyde_score 字段、HyDE 命中提分
  - hyde 空路回退：与原两路结果完全一致（回归保障）
  - rrf / weighted_rrf 接受第三路不炸
  - generate_hypothesis：禁用 / 无 key / mock 客户端成功 / 异常降级

运行：python allagent2/tests/test_hyde.py（直接跑，不需要 pytest）
外部依赖全部 mock 在模块属性上，进程内零副作用。
"""
from __future__ import annotations

import os
import sys
import unittest
from unittest import mock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from allagent2.rag import fusion, hyde  # noqa: E402
from allagent2.rag.config import RAGConfig  # noqa: E402


def _hit(cid: str, score: float) -> dict:
    return {"id": cid, "score": score, "text": f"t{cid}", "heading_path": "",
            "parent_id": f"p{cid}", "doc_source": ""}


class TestRelativeThreeWay(unittest.TestCase):
    def setUp(self):
        self.cfg = RAGConfig(W_LEXICAL=0.3, W_VECTOR=0.7, W_HYDE=0.3,
                             FUSION_MODE="relative")
        self.lex = [_hit("a", 2.0), _hit("b", 1.0)]   # norm: a=1, b=0
        self.vec = [_hit("b", 0.9), _hit("c", 0.3)]   # norm: b=1, c=0

    def test_no_hyde_keeps_two_way_semantics(self):
        out = fusion.fuse(self.lex, self.vec, self.cfg, None)
        scores = {d["id"]: d["score"] for d in out}
        self.assertAlmostEqual(scores["a"], 0.3, places=6)
        self.assertAlmostEqual(scores["b"], 0.7, places=6)
        self.assertAlmostEqual(scores["c"], 0.0, places=6)

    def test_empty_hyde_same_as_none(self):
        out1 = fusion.fuse(self.lex, self.vec, self.cfg, [])
        out2 = fusion.fuse(self.lex, self.vec, self.cfg, None)
        self.assertEqual([d["id"] for d in out1], [d["id"] for d in out2])
        for d1, d2 in zip(out1, out2):
            self.assertAlmostEqual(d1["score"], d2["score"], places=9)

    def test_hyde_route_participates(self):
        hy = [_hit("c", 0.8), _hit("d", 0.2)]  # norm: c=1, d=0
        out = fusion.fuse(self.lex, self.vec, self.cfg, hy)
        by_id = {d["id"]: d for d in out}
        # 三权归一化 0.3/1.3, 0.7/1.3, 0.3/1.3
        self.assertAlmostEqual(by_id["c"]["hyde_score"], 1.0, places=6)
        self.assertAlmostEqual(by_id["c"]["score"], 0.3 / 1.3, places=6)
        self.assertAlmostEqual(by_id["d"]["hyde_score"], 0.0, places=6)
        # c 从零分变为非零：HyDE 路确实注入了贡献
        self.assertGreater(by_id["c"]["score"], 0.0)

    def test_hyde_hit_can_boost_ranking(self):
        # c 同时被向量路与 HyDE 路命中 → 排名应超过仅词法命中的 a
        lex = [_hit("a", 2.0)]                 # norm: a=1
        vec = [_hit("c", 0.9), _hit("a", 0.2)]  # norm: c=1, a=0
        hy = [_hit("c", 0.9)]                  # norm: c=1（单条命中按 1.0 计）
        out = fusion.fuse(lex, vec, self.cfg, hy)
        self.assertEqual(out[0]["id"], "c")    # 1.0/1.3 > 0.3/1.3


class TestRRFWithHyde(unittest.TestCase):
    def setUp(self):
        self.cfg = RAGConfig(FUSION_MODE="rrf")
        self.lex = [_hit("a", 2.0)]
        self.vec = [_hit("b", 0.9)]
        self.hy = [_hit("c", 0.8)]

    def test_rrf_accepts_hyde(self):
        out = fusion.rrf_fusion(self.lex, self.vec, self.cfg, self.hy)
        self.assertEqual({d["id"] for d in out}, {"a", "b", "c"})

    def test_weighted_rrf_accepts_hyde(self):
        out = fusion.weighted_rrf_fusion(self.lex, self.vec, self.cfg, self.hy)
        self.assertEqual({d["id"] for d in out}, {"a", "b", "c"})

    def test_fuse_dispatches_hyde(self):
        out = fusion.fuse(self.lex, self.vec, self.cfg, self.hy)
        self.assertEqual({d["id"] for d in out}, {"a", "b", "c"})


class TestGenerateHypothesis(unittest.TestCase):
    def test_disabled_returns_empty(self):
        cfg = RAGConfig(HYDE_ENABLED=False)
        self.assertEqual(hyde.generate_hypothesis("血压高吃什么", cfg), "")

    def test_empty_query_returns_empty(self):
        self.assertEqual(hyde.generate_hypothesis("  ", RAGConfig()), "")

    def test_no_key_returns_empty(self):
        with mock.patch.dict(os.environ):
            os.environ.pop("HYDE_API_KEY", None)
            self.assertEqual(hyde.generate_hypothesis("血压高吃什么", RAGConfig()), "")

    def test_success_via_mock_client(self):
        fake = mock.MagicMock()
        fake.chat.completions.create.return_value = mock.MagicMock(
            choices=[mock.MagicMock(message=mock.MagicMock(content="  假设回答文本  "))]
        )
        with mock.patch.dict(os.environ, {"HYDE_API_KEY": "sk-test",
                                          "HYDE_BASE_URL": "http://x/v1",
                                          "HYDE_MODEL": "m-test"}):
            with mock.patch.object(hyde, "_get_client", return_value=fake):
                text = hyde.generate_hypothesis("血压高吃什么", RAGConfig())
        self.assertEqual(text, "假设回答文本")
        # prompt 携带了原始问题
        sent = fake.chat.completions.create.call_args.kwargs["messages"][0]["content"]
        self.assertIn("血压高吃什么", sent)

    def test_exception_degrades_to_empty(self):
        fake = mock.MagicMock()
        fake.chat.completions.create.side_effect = RuntimeError("boom")
        with mock.patch.dict(os.environ, {"HYDE_API_KEY": "sk-test"}):
            with mock.patch.object(hyde, "_get_client", return_value=fake):
                self.assertEqual(hyde.generate_hypothesis("q", RAGConfig()), "")


if __name__ == "__main__":
    unittest.main(verbosity=2)
