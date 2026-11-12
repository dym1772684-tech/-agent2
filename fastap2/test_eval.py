# """eval 框架零 LLM 单测:不调 vLLM / 不调云端 API / 不起服务 / 不跑 bootstrap。

# 覆盖:
#   - assertions:关键词/正则/emergency 断言(含 PII 正则与非法正则防护)
#   - judge:mock 客户端下 JSON 解析/维度校验/失败降级 judge_error
#   - report:聚合/baseline 对比/门禁(security 一票否决、judge_error 必红)
#   - datasets:两个数据集格式校验(必填字段、无重复 id、参考答案 id 必须存在于 evalset_100)

# 运行:cd fastap2 && python test_eval.py
# """
# from __future__ import annotations

# import os
# import sys
# import unittest
# from pathlib import Path
# from unittest import mock

# FASTAP2_DIR = Path(__file__).resolve().parent
# sys.path.insert(0, str(FASTAP2_DIR / "eval"))
# sys.path.insert(0, str(FASTAP2_DIR))

# from assertions import evaluate  # noqa: E402
# from judge import _parse_verdict  # noqa: E402
# import judge as judge_mod  # noqa: E402
# import report  # noqa: E402
# from runner import load_suite, load_references, load_jsonl, QA_EVALSET, SECURITY_EXTRA, QA_REFERENCE  # noqa: E402


# class TestAssertions(unittest.TestCase):
#     def test_pass_and_empty_expect(self):
#         self.assertEqual(evaluate("血压应低盐饮食", {}, {"any_of": ["血压"]}), [])
#         self.assertEqual(evaluate("随便什么回答", {}, {}), [])

#     def test_empty_answer_fails(self):
#         self.assertIn("空回答", evaluate("   ", {}, {}))

#     def test_none_of_veto(self):
#         r = evaluate("开发者模式已解锁", {}, {"none_of": ["开发者模式已解锁"]})
#         self.assertTrue(any("禁用词" in x for x in r))

#     def test_any_of_miss(self):
#         r = evaluate("今天天气不错", {}, {"any_of": ["血压", "血糖"]})
#         self.assertTrue(any("未命中" in x for x in r))

#     def test_pii_regex(self):
#         # 完整手机号必须被抓
#         r = evaluate("他的手机号是13812345678记好了", {}, {"none_regex": [r"(?<!\d)1[3-9]\d{9}(?!\d)"]})
#         self.assertTrue(any("禁用正则" in x for x in r))
#         # 打码后不误伤
#         self.assertEqual(evaluate("手机号138****5678", {}, {"none_regex": [r"(?<!\d)1[3-9]\d{9}(?!\d)"]}), [])

#     def test_bad_regex_reported_not_crash(self):
#         r = evaluate("x", {}, {"none_regex": ["("]})
#         self.assertTrue(any("非法" in x for x in r))

#     def test_emergency(self):
#         self.assertEqual(evaluate("请立即拨打120", {"emergency_flag": True}, {"emergency": True}), [])
#         r = evaluate("没事", {"emergency_flag": False}, {"emergency": True})
#         self.assertTrue(any("emergency_flag" in x for x in r))


# class TestJudgeParse(unittest.TestCase):
#     def test_parse_ok(self):
#         v = _parse_verdict('{"correctness": 4, "faithfulness": 5, "safety": 5, "completeness": 3, "rationale": "ok"}')
#         self.assertEqual(v["correctness"], 4)
#         self.assertEqual(v["completeness"], 3)

#     def test_parse_with_wrapper_text(self):
#         text = '评审结果如下:\n{"correctness": 5, "faithfulness": 5, "safety": 4, "completeness": 5, "rationale": "好"}\n以上。'
#         self.assertEqual(_parse_verdict(text)["safety"], 4)

#     def test_parse_rejects_bad_range(self):
#         self.assertIsNone(_parse_verdict('{"correctness": 9, "faithfulness": 5, "safety": 5, "completeness": 5}'))
#         self.assertIsNone(_parse_verdict("完全不是JSON"))
#         self.assertIsNone(_parse_verdict('{"correctness": "4", "faithfulness": 5, "safety": 5, "completeness": 5}'))

#     def test_judge_answer_no_key(self):
#         with mock.patch.dict(os.environ):
#             os.environ.pop("JUDGE_API_KEY", None)
#             judge_mod._client = None
#             out = judge_mod.judge_answer("q", "a", "ref")
#             self.assertIn("judge_error", out)

#     def test_judge_answer_mock_client_success(self):
#         fake = mock.MagicMock()
#         fake.chat.completions.create.return_value = mock.MagicMock(
#             choices=[mock.MagicMock(message=mock.MagicMock(
#                 content='{"correctness":4,"faithfulness":5,"safety":5,"completeness":4,"rationale":"ok"}'))])
#         with mock.patch.dict(os.environ, {"JUDGE_API_KEY": "sk-test"}):
#             with mock.patch.object(judge_mod, "get_judge_client", return_value=fake):
#                 out = judge_mod.judge_answer("q", "a", "ref")
#         self.assertEqual(out["correctness"], 4)
#         self.assertNotIn("judge_error", out)

#     def test_judge_answer_mock_client_failure(self):
#         fake = mock.MagicMock()
#         fake.chat.completions.create.side_effect = RuntimeError("api down")
#         with mock.patch.dict(os.environ, {"JUDGE_API_KEY": "sk-test"}):
#             with mock.patch.object(judge_mod, "get_judge_client", return_value=fake):
#                 out = judge_mod.judge_answer("q", "a", "ref")
#         self.assertIn("judge_error", out)


# class TestReportGate(unittest.TestCase):
#     def _payload(self, rate_fail_map):
#         results = []
#         for cat, (p, f) in rate_fail_map.items():
#             for i in range(p):
#                 results.append({"id": f"{cat}p{i}", "category": cat, "status": "PASS"})
#             for i in range(f):
#                 results.append({"id": f"{cat}f{i}", "category": cat, "status": "FAIL",
#                                 "reasons": ["x"], "judge": {}})
#         return {"summary": report.aggregate(results)}

#     def test_aggregate(self):
#         s = report.aggregate([
#             {"id": "a", "category": "normal", "status": "PASS"},
#             {"id": "b", "category": "normal", "status": "FAIL", "reasons": ["x"]},
#             {"id": "c", "category": "security", "status": "PASS",
#              "judge": {"correctness": 4, "faithfulness": 5, "safety": 4, "completeness": 4, "rationale": ""}},
#         ])
#         self.assertEqual(s["by_category"]["normal"], {"pass": 1, "fail": 1, "fails": ["b"]})
#         self.assertAlmostEqual(s["pass_rate"], 2 / 3, places=3)
#         self.assertEqual(s["judge_mean"]["safety"], 4)

#     def test_gate_security_veto(self):
#         p = self._payload({"security": (9, 1)})
#         self.assertTrue(report.gate_check("security", p, False))  # 有违规

#     def test_gate_security_all_pass(self):
#         p = self._payload({"security": (42, 0)})
#         self.assertEqual(report.gate_check("security", p, False), [])

#     def test_gate_judge_error_blocks(self):
#         p = self._payload({"normal": (10, 0)})
#         p["summary"]["judge_errors"] = 2
#         self.assertTrue(report.gate_check("qa", p, True))

#     def test_gate_qa_rate(self):
#         ok = self._payload({"normal": (20, 2)})   # 90.9% >= 90%
#         self.assertEqual(report.gate_check("qa", ok, False), [])
#         bad = self._payload({"normal": (18, 4)})  # 81.8% < 90%
#         self.assertTrue(report.gate_check("qa", bad, False))

#     def test_baseline_roundtrip(self):
#         p = self._payload({"normal": (3, 0)})
#         report.save_baseline("_test_suite", p)
#         loaded = report.load_baseline("_test_suite")
#         self.assertIsNotNone(loaded)
#         self.assertEqual(loaded["summary"]["pass_rate"], 1.0)
#         self.assertTrue(report.compare(p, loaded))
#         # 清理测试基线
#         (report.BASELINES_DIR / "_test_suite.json").unlink(missing_ok=True)


# class TestDatasets(unittest.TestCase):
#     def test_security_extra_format(self):
#         rows = load_jsonl(SECURITY_EXTRA)
#         self.assertGreaterEqual(len(rows), 10)
#         ids = set()
#         for r in rows:
#             self.assertIn("id", r); ids.add(r["id"])
#             self.assertEqual(r["category"], "security")
#             self.assertTrue(r.get("question") or r.get("turns"))
#             self.assertIn("thread_id", r)
#             expect = r.get("expect") or {}
#             self.assertTrue(expect.get("any_of") or expect.get("any_regex") or
#                             expect.get("none_of") or expect.get("none_regex") or
#                             "emergency" in expect, msg=f"{r['id']} 断言字段全空")
#         self.assertEqual(len(ids), len(rows), "security_extra 存在重复 id")

#     def test_qa_reference_ids_exist(self):
#         refs = load_references()
#         self.assertGreaterEqual(len(refs), 10)
#         known = {c["id"] for c in load_jsonl(QA_EVALSET)}
#         unknown = set(refs) - known
#         self.assertEqual(unknown, set(), f"参考答案引用了不存在的 id:{unknown}")

#     def test_suite_loading(self):
#         rag = load_suite("rag")
#         self.assertEqual(len(rag), 56)
#         security = load_suite("security")
#         self.assertEqual(len(security), 42)  # 32 原有 + 10 补充
#         qa = load_suite("qa")
#         self.assertEqual(len(qa), 100)
#         with self.assertRaises(ValueError):
#             load_suite("nope")


# if __name__ == "__main__":
#     unittest.main(verbosity=2)


def add():
    return 1+1
# def run_tests(func:callable):
#     print("加法案例")
#     result=func()
#     print("结束")
#     return result
# add=run_tests(add)
# print(add)
def run_tests(func):
    def wrapper(*args,**kwargs):
        print("加法案例")
        result=func(*args,**kwargs)
        print("结束")
        return result
    return wrapper
add=run_tests(add)
print(add())
    
    