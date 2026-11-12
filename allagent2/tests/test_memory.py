"""记忆系统零 LLM 单测：不依赖 MySQL / Chroma / vLLM / 网络。

覆盖（对应"还没做完的清单 #2"）：
  - 六因子评分 compute / R 门控 passes_gate / 时间衰减 time_factor（F 抵消）
  - 频率因子 frequency_factor / 置信基线 initial_factors / 反馈信号 apply_feedback
  - PII 脱敏 mask_pii
  - 读路径路由 route_layers / 运行时纠错 extract_correction / fail-open 全空块
  - 预算装配 _assemble（token 预算，不是 Top-K 计数）
  - 轮次日志 build_turn_log（纯函数）
  - 写路径门槛 should_reflect / on_turn_finished 登记+异步投递
  - 画像冲突裁决 upsert_field（invalid/added/overwrite/coexist 兜底/discard）
  - 技能 2 次晋升 upsert_skill（invalid/added/晋升转正）

运行：python allagent2/tests/test_memory.py（直接跑，不需要 pytest）
外部依赖全部 mock 在 store / 模块属性上，进程内零副作用。
"""
from __future__ import annotations

import os
import sys
import unittest
from datetime import datetime, timedelta
from types import SimpleNamespace
from unittest import mock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from allagent2.session import profile, procedural, recall, reflect, scoring, store  # noqa: E402
from allagent2.session.config import memory_config  # noqa: E402
from allagent2.session.schema import MemoryRecord, build_turn_log  # noqa: E402


def _rec(**kw) -> MemoryRecord:
    kw.setdefault("layer", "episodic")
    kw.setdefault("updated_at", datetime.now())
    return MemoryRecord(**kw)


# ============================================================
# ① 评分：R 门控 + 六因子
# ============================================================
class TestScoring(unittest.TestCase):
    def test_gate_blocks_irrelevant_even_with_high_importance(self):
        """★① 不相关(R 低)的记忆，I/F 再高也必须被门控淘汰。"""
        r = _rec(relevance=0.1, importance=0.9, hit_count=20, specificity=0.9)
        self.assertFalse(scoring.passes_gate(r))
        self.assertEqual(scoring.compute(r), 0.0)

    def test_gate_passes_relevant(self):
        r = _rec(relevance=0.8, importance=0.5)
        self.assertTrue(scoring.passes_gate(r))
        self.assertGreater(scoring.compute(r), 0.0)

    def test_profile_layer_never_gated(self):
        """必带项不走门控（★⑧）。"""
        r = _rec(layer="profile", relevance=0.0, importance=0.8)
        self.assertTrue(scoring.passes_gate(r))

    def test_time_decay_recent_beats_old(self):
        now = datetime.now()
        fresh = scoring.time_factor("episodic", now - timedelta(hours=1))
        stale = scoring.time_factor("episodic", now - timedelta(days=60))
        self.assertGreater(fresh, stale)
        self.assertGreater(stale, 0.0)          # 不衰减到 0
        self.assertLessEqual(fresh, 1.2)        # 封顶

    def test_frequency_compensates_decay(self):
        """④ 高频访问的旧记忆不该被衰减淹死。"""
        now = datetime.now()
        old_idle = scoring.time_factor("episodic", now - timedelta(days=60), hit_count=0)
        old_hot = scoring.time_factor("episodic", now - timedelta(days=60), hit_count=15)
        self.assertGreater(old_hot, old_idle)

    def test_frequency_factor_bounds(self):
        self.assertEqual(scoring.frequency_factor(0), 0.0)
        self.assertGreater(scoring.frequency_factor(100), 0.99)
        self.assertEqual(scoring.frequency_factor(-5), 0.0)

    def test_confidence_baseline_by_source(self):
        """★③ 用户确认 > 工具实证 > LLM 推断。"""
        imp_u, conf_u, _ = scoring.initial_factors("user_confirmed", 0.6, 0.5)
        _, conf_t, _ = scoring.initial_factors("tool_verified", 0.6, 0.5)
        _, conf_l, _ = scoring.initial_factors("llm_inferred", 0.6, 0.5)
        self.assertGreater(conf_u, conf_t)
        self.assertGreater(conf_t, conf_l)
        self.assertEqual(conf_u, 1.0)

    def test_feedback_adjusts_confidence_with_clamp(self):
        r = _rec(confidence=0.95, importance=0.5, hit_count=0)
        scoring.apply_feedback(r, adopted=True)
        self.assertEqual(r.confidence, 1.0)     # clamp 上限
        self.assertEqual(r.hit_count, 1)
        scoring.apply_feedback(r, adopted=False)
        self.assertEqual(r.confidence, 0.8)
        r2 = _rec(confidence=0.15, importance=0.05)
        scoring.apply_feedback(r2, adopted=False)
        self.assertEqual(r2.confidence, 0.1)    # clamp 下限
        self.assertEqual(r2.importance, 0.0)


# ============================================================
# ② PII 脱敏（★⑭）
# ============================================================
class TestMaskPii(unittest.TestCase):
    def test_phone_masked(self):
        out = store.mask_pii("联系人电话13812345678请记下")
        self.assertNotIn("13812345678", out)
        self.assertIn("138", out) and self.assertIn("5678", out)

    def test_idcard_masked(self):
        out = store.mask_pii("身份证11010119900307891X")
        self.assertNotIn("11010119900307891X", out)
        self.assertTrue(out.startswith("身份证110"))

    def test_plain_text_untouched(self):
        s = "老人喜欢清淡饮食，对青霉素过敏"
        self.assertEqual(store.mask_pii(s), s)

    def test_empty(self):
        self.assertEqual(store.mask_pii(""), "")


# ============================================================
# ③ 读路径：路由 / 纠错 / fail-open / 预算装配
# ============================================================
class TestRecall(unittest.TestCase):
    def test_route_rules(self):
        self.assertIn("episodic", recall.route_layers("上次我们聊过什么？"))
        self.assertIn("procedural", recall.route_layers("血压监测怎么做？"))
        self.assertIn("semantic", recall.route_layers("我妈对什么过敏"))
        # 无命中 → 默认层
        self.assertEqual(recall.route_layers("今天天气如何"), ["episodic", "semantic"])

    def test_extract_correction(self):
        self.assertIn("我不吃海鲜了", recall.extract_correction("我不吃海鲜了，推荐个晚餐"))
        self.assertEqual(recall.extract_correction("推荐个晚餐"), "")

    def test_recall_fail_open_empty_block(self):
        """记忆库不可用/无数据 → 空块、不抛异常（反噪声铁律：空串不注入）。"""
        res = recall.recall("ghost-user", "随便聊聊今天的天气")
        self.assertEqual(res.context_block, "")
        self.assertIsInstance(res.injected, list)

    def test_recall_captures_correction_even_without_db(self):
        res = recall.recall("ghost-user", "我现在不吃海鲜了，帮我配一周食谱")
        self.assertIn("不吃海鲜", res.correction)

    def test_assemble_respects_token_budget(self):
        """⑦ 装不下的条目跳过（不是硬 Top-K 截断，短的高分条仍可入选）。"""
        long_c = _rec(content="很" * 400, score=0.9)     # 远超预算
        short_c = _rec(content="短记忆", score=0.5)
        winners = recall._assemble([long_c, short_c], budget_tokens=60)
        self.assertEqual([w.content for w in winners], ["短记忆"])

    def test_assemble_truncates_overlong_single_item(self):
        c = _rec(content="长" * 300, score=0.95)
        winners = recall._assemble([c], budget_tokens=250)
        if winners:  # 截断后若仍装不下则不选；装得下则必须是被截断的
            self.assertTrue(winners[0].content.endswith("…"))


# ============================================================
# ④ 轮次日志（工作记忆 → 反思输入的纯函数）
# ============================================================
class TestTurnLog(unittest.TestCase):
    def test_build_turn_log_filters_and_numbers(self):
        msgs = [
            SimpleNamespace(type="system", content="人设"),          # 过滤
            SimpleNamespace(type="human", content="我血压高怎么办？"),
            SimpleNamespace(type="ai", content="", tool_calls=[{"name": "query_health_records"}]),
            SimpleNamespace(type="tool", content='{"bp": "150/95"}'),
            SimpleNamespace(type="ai", content="建议低盐饮食并监测血压。", tool_calls=None),
        ]
        turns = build_turn_log(msgs)
        roles = [t["role"] for t in turns]
        self.assertEqual(roles, ["user", "agent", "tool", "agent"])
        self.assertEqual(turns[0]["turn"], 1)
        self.assertEqual(turns[1]["tool_calls"], ["query_health_records"])

    def test_truncation(self):
        long = "字" * 1000
        turns = build_turn_log([SimpleNamespace(type="human", content=long)])
        self.assertEqual(len(turns[0]["text"]), 400)


# ============================================================
# ⑤ 写路径门槛与异步投递
# ============================================================
class TestReflectGate(unittest.TestCase):
    def test_thresholds(self):
        self.assertFalse(reflect.should_reflect({"turn_count": 1, "tool_calls": 5}))
        self.assertFalse(reflect.should_reflect({"turn_count": 2, "tool_calls": 0}))  # 无工具且 <3 轮
        self.assertTrue(reflect.should_reflect({"turn_count": 2, "tool_calls": 1}))
        self.assertTrue(reflect.should_reflect({"turn_count": 3, "tool_calls": 0}))

    def test_on_turn_finished_registers_and_submits(self):
        with mock.patch.object(store, "available", return_value=True), \
             mock.patch.object(store, "register_turn",
                               return_value={"turn_count": 5, "tool_calls": 2, "reflected": 0}) as m_reg, \
             mock.patch.object(reflect, "_EXECUTOR") as m_exec:
            reflect.on_turn_finished("u1", "t1", [], question="q", answer="a")
            m_reg.assert_called_once()
            m_exec.submit.assert_called_once()

    def test_on_turn_finished_below_threshold_no_submit(self):
        with mock.patch.object(store, "available", return_value=True), \
             mock.patch.object(store, "register_turn",
                               return_value={"turn_count": 1, "tool_calls": 0, "reflected": 0}), \
             mock.patch.object(reflect, "_EXECUTOR") as m_exec:
            reflect.on_turn_finished("u1", "t1", [], question="q", answer="a")
            m_exec.submit.assert_not_called()

    def test_on_turn_finished_noop_when_store_unavailable(self):
        with mock.patch.object(store, "available", return_value=False):
            reflect.on_turn_finished("u1", "t1", [])  # 不应抛异常

    def test_noop_without_user(self):
        with mock.patch.object(store, "available", return_value=True), \
             mock.patch.object(store, "register_turn") as m_reg:
            reflect.on_turn_finished("", "t1", [])
            m_reg.assert_not_called()


# ============================================================
# ⑥ 画像：UPSERT + 冲突裁决（store 全 mock，LLM 裁决替换为桩）
# ============================================================
class TestProfileUpsert(unittest.TestCase):
    def test_invalid_category_or_content(self):
        self.assertEqual(profile.upsert_field("u1", "爱好", "下棋"), "invalid")   # 非法类别
        self.assertEqual(profile.upsert_field("u1", "偏好", "咸"), "invalid")     # 内容过短

    def test_added_when_no_existing(self):
        with mock.patch.object(store, "list_items", return_value=[]), \
             mock.patch.object(store, "add_item", side_effect=lambda r: setattr(r, "id", 1) or r) as m_add:
            self.assertEqual(profile.upsert_field("u1", "偏好", "清淡饮食"), "added")
            m_add.assert_called_once()

    def test_overwrite_supersedes_old(self):
        """⑫ 冲突裁决=overwrite → 旧值软失效指向新值。"""
        old = _rec(layer="profile", category="禁忌", content="不能吃海鲜", id=7)
        with mock.patch.object(store, "list_items", return_value=[old]), \
             mock.patch.object(store, "index_query", return_value={}), \
             mock.patch.object(profile, "_adjudicate", return_value="overwrite"), \
             mock.patch.object(store, "add_item", side_effect=lambda r: setattr(r, "id", 8) or r), \
             mock.patch.object(store, "supersede") as m_sup:
            self.assertEqual(profile.upsert_field("u1", "禁忌", "对海鲜过敏不能吃"), "overwrote")
            m_sup.assert_called_once()

    def test_adjudicate_failure_falls_back_to_coexist(self):
        """★⑩ LLM 裁决失败保守并存，绝不丢信息。

        注意：兜底 try/except 在 _adjudicate 内部——必须让真实的 _adjudicate
        跑起来、只把 LLM 构造器打挂，才能测到兜底路径。
        """
        with mock.patch("allagent2.agent.agent.create_chat_model",
                        side_effect=RuntimeError("llm down")):
            self.assertEqual(profile._adjudicate("偏好", "喜欢散步", "喜欢太极"), "coexist")

    def test_discard(self):
        old = _rec(layer="profile", category="偏好", content="喜欢散步", id=7)
        with mock.patch.object(store, "list_items", return_value=[old]), \
             mock.patch.object(store, "index_query", return_value={}), \
             mock.patch.object(profile, "_adjudicate", return_value="discard"), \
             mock.patch.object(store, "add_item") as m_add:
            self.assertEqual(profile.upsert_field("u1", "偏好", "不喜欢散步了吧"), "discarded")
            m_add.assert_not_called()


# ============================================================
# ⑦ 技能：2 次晋升 + 生命周期
# ============================================================
class TestProceduralUpsert(unittest.TestCase):
    def test_invalid(self):
        self.assertEqual(procedural.upsert_skill("u1", "", ["a", "b"]), "invalid")
        self.assertEqual(procedural.upsert_skill("u1", "触发", ["只有一步"]), "invalid")

    def test_added_as_candidate(self):
        with mock.patch.object(store, "index_query", side_effect=RuntimeError("no index")), \
             mock.patch.object(store, "add_item", side_effect=lambda r: setattr(r, "id", 1) or r) as m_add:
            out = procedural.upsert_skill("u1", "测血压", ["洗手", "绑袖带", "读数"])
            self.assertEqual(out, "added")
            rec = m_add.call_args[0][0]
            self.assertEqual(rec.confidence, memory_config.skill_candidate_confidence)  # 候选 0.5

    def test_promote_on_second_similar(self):
        """同类再现(sim≥0.8) → confidence 提到晋升阈值、hit+1、步骤合并。"""
        existing = _rec(layer="procedural", id=3, content="场景:测血压 步骤:a→b",
                        confidence=0.5, importance=0.5, hit_count=1)
        with mock.patch.object(store, "index_query", return_value={"3": 0.9}), \
             mock.patch.object(store, "list_items", return_value=[existing]), \
             mock.patch.object(store, "update_item") as m_upd:
            out = procedural.upsert_skill("u1", "在家测血压", ["洗手", "绑袖带", "读数"])
            self.assertEqual(out, "promoted")
            m_upd.assert_called_once()
            self.assertGreaterEqual(existing.confidence, memory_config.skill_promote_confidence)
            self.assertEqual(existing.hit_count, 2)

    def test_demote_retires_on_repeated_failure(self):
        low = _rec(layer="procedural", id=5, confidence=0.32)
        with mock.patch.object(store, "list_items", return_value=[low]), \
             mock.patch.object(store, "update_item") as m_upd:
            procedural.demote("u1", 5, worked=False)
            self.assertEqual(low.confidence, 0.12)   # 0.32-0.2
            self.assertFalse(low.active)             # <0.3 退役
            m_upd.assert_called_once()

    def test_ready_skills_filters_candidates(self):
        cand = _rec(layer="procedural", confidence=0.5)
        ready = _rec(layer="procedural", confidence=0.8)
        with mock.patch.object(store, "list_items", return_value=[cand, ready]):
            out = procedural.ready_skills("u1")
            self.assertEqual([r.id for r in out], [ready.id])


if __name__ == "__main__":
    unittest.main(verbosity=2)
