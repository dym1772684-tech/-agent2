"""记忆系统分层测试（allagent2/session 全模块，复用 tests/test_memory.py 主体）。

这里做进程内冒烟：真实 MySQL 记忆库写入→召回→清理（不调 LLM）。
全量 35 项零 LLM 单测见 allagent2/tests/test_memory.py，由 run_all.py 一并执行。
"""
from _lib import check, load_env, results


def run():
    print("[t09] 记忆系统(真实库冒烟)")
    load_env()
    try:
        from allagent2.session import store
        from allagent2.session.schema import MemoryRecord
    except Exception as e:
        return [("记忆: 模块导入", False, str(e)[:80])]

    if not store.available():
        return [("记忆: 存储层可用(MySQL)", None, "SKIP：MySQL 未启动")]

    uid = "u-suite-test"
    try:
        # ---- 1) 写入三层的种子 ----
        s = []
        s.append(store.add_item(MemoryRecord(
            user_id=uid, layer="profile", category="禁忌", content="海鲜过敏，忌口",
            importance=0.9, confidence=1.0, specificity=0.9, source_thread="t-suite")))
        s.append(store.add_item(MemoryRecord(
            user_id=uid, layer="semantic", category="知识", content="低盐饮食适合高血压老人",
            importance=0.6, confidence=0.8, specificity=0.6, source_thread="t-suite")))
        s.append(store.add_item(MemoryRecord(
            user_id=uid, layer="episodic", category="事件", content="曾咨询过血压监测方法",
            importance=0.5, confidence=0.7, specificity=0.5, source_thread="t-suite")))
        check("记忆: 三层种子写入", all(x.id for x in s), f"ids={[x.id for x in s]}")

        # ---- 2) 召回：profile 必带 + 门控行为 ----
        from allagent2.session.recall import recall
        rr = recall(uid, "帮我看看饮食注意什么")
        block = getattr(rr, "context_block", "") or ""
        layers = {getattr(x, "layer", "") for x in (getattr(rr, "selected", None) or [])}
        check("记忆: 召回注入画像(必带项)", "profile" in layers or "过敏" in block,
              f"layers={layers}")
        check("记忆: 召回块非空", bool(str(block).strip()), f"len={len(str(block))}")

    finally:
        # ---- 3) 删除权自清理 ----
        n = store.delete_user_memories(uid)
        check("记忆: 删除权清理", n >= 3, f"deleted={n}")

    return results()


if __name__ == "__main__":
    run()
