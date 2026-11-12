"""core/cache.py 升级版单测（真实 Redis，零 LLM）。

覆盖：三态语义、包装协议、旧格式兼容、脏 key 自愈、get_or_load 收敛
（含 single-flight 与空哨兵防穿透）、TTL 抖动、熔断开关与恢复。
"""
import sys
import time
import unittest
from unittest.mock import patch

sys.path.insert(0, __file__.rsplit("\\", 1)[0])

from core import cache as c


class CacheTests(unittest.TestCase):
    def setUp(self):
        # 每个用例前复位熔断器与待重放队列,避免互相影响
        c._circuit = c._Circuit()
        with c._pending_lock:
            c._pending_invalidations.clear()

    # ---------- 三态 + 协议 ----------
    def test_roundtrip_and_three_states(self):
        key = "cache:test:t1"
        c.invalidate_cache_sync(key)
        self.assertIsNone(c.cache_get_json_sync(key))                 # 未命中
        c.cache_set_json_sync(key, {"name": "张三", "age": 80}, ttl=30)
        self.assertEqual(c.cache_get_json_sync(key), {"name": "张三", "age": 80})  # 数据
        c.cache_set_json_sync(key, None, ttl=30)                      # 空结果
        self.assertEqual(c.cache_get_json_sync(key), c.NULL_SENTINEL)  # 空哨兵三态

    def test_legacy_raw_value_compatible(self):
        """旧格式(非包装值)仍可读——平滑过渡。"""
        import json as _json
        from core.redis_client import get_redis_sync
        key = "cache:test:t2"
        get_redis_sync().set(key, _json.dumps({"legacy": 1}))
        self.assertEqual(c.cache_get_json_sync(key), {"legacy": 1})

    def test_dirty_key_self_heals(self):
        """脏数据：返回 None(当未命中)且 key 被自愈删除,不打挂业务。"""
        from core.redis_client import get_redis_sync
        key = "cache:test:t3"
        get_redis_sync().set(key, b"{corrupted!!")
        self.assertIsNone(c.cache_get_json_sync(key))
        self.assertIsNone(get_redis_sync().get(key))  # 已被删除

    # ---------- get_or_load ----------
    def test_get_or_load_miss_loader_hit(self):
        key = "cache:test:t4"
        c.invalidate_cache_sync(key)
        calls = []

        def loader():
            calls.append(1)
            return {"elder": "E001"}

        self.assertEqual(c.get_or_load_sync(key, loader, ttl=30), {"elder": "E001"})
        self.assertEqual(c.get_or_load_sync(key, loader, ttl=30), {"elder": "E001"})
        self.assertEqual(len(calls), 1)  # 第二次走缓存,loader 不再执行

    def test_get_or_load_null_sentinel(self):
        """loader 返回 None → 空哨兵入库,后续不再回源(防穿透)。"""
        key = "cache:test:t5"
        c.invalidate_cache_sync(key)
        calls = []

        def loader():
            calls.append(1)
            return None

        self.assertIsNone(c.get_or_load_sync(key, loader, ttl=30))
        self.assertIsNone(c.get_or_load_sync(key, loader, ttl=30))
        self.assertEqual(len(calls), 1)  # 空结果也只回源一次
        self.assertEqual(c.cache_get_json_sync(key), c.NULL_SENTINEL)

    def test_single_flight_sync(self):
        """同 key 并发回源只放行一个(threading.Lock;asyncio 版逻辑同构不重复测)。"""
        import threading
        key = "cache:test:t6"
        c.invalidate_cache_sync(key)
        barrier = threading.Event()
        calls = []

        def slow_loader():
            calls.append(1)
            barrier.wait(timeout=3)  # 持锁期间放行其他线程
            return "v1"

        results = []

        def worker():
            lock = c._flight_lock_sync(key)
            with lock:
                cached = c.cache_get_json_sync(key)
                if cached is not None:
                    results.append(cached)
                    return
                data = slow_loader()
                c.cache_set_json_sync(key, data, ttl=30)
                results.append(data)

        t1 = threading.Thread(target=worker)
        t1.start()
        barrier.set()
        t2 = threading.Thread(target=worker)  # 等 t1 写完后才拿锁 → 双检命中缓存
        t2.start()
        t2.join(timeout=3)
        t1.join(timeout=3)
        self.assertEqual(results[0], "v1")
        self.assertEqual(len(calls), 1)

    # ---------- TTL 抖动 ----------
    def test_jitter_range(self):
        for _ in range(50):
            j = c._jittered(300)
            self.assertTrue(300 <= j <= 300 + c.JITTER_SPAN, j)

    # ---------- 熔断 ----------
    def test_circuit_opens_after_consecutive_failures(self):
        """连续 5 次异常 → 熔断打开,读写直接降级(不再等 2s 超时)。"""
        key = "cache:test:t7"
        boom = type("Boom", (c.redis_lib.RedisError,), {})()

        with patch.object(c, "get_redis_sync", side_effect=boom):
            for _ in range(c._FAILURE_THRESHOLD):
                c.cache_get_json_sync(key)
        self.assertFalse(c._circuit.allow())          # 已熔断
        with patch.object(c, "get_redis_sync", side_effect=AssertionError("不应再碰 Redis")):
            self.assertIsNone(c.cache_get_json_sync(key))   # 直接降级,不碰客户端
            c.cache_set_json_sync(key, "x", ttl=30)         # 写也直接跳过
            c.invalidate_cache_sync(key)

    def test_circuit_recovers_on_success(self):
        """异常后恢复:record_success 清零,半开窗口恢复正常读写。"""
        key = "cache:test:t8"
        boom = type("Boom", (c.redis_lib.RedisError,), {})()
        with patch.object(c, "get_redis_sync", side_effect=boom):
            for _ in range(c._FAILURE_THRESHOLD - 1):
                c.cache_get_json_sync(key)
        c.cache_set_json_sync(key, "ok", ttl=30)      # 成功 → 清零
        self.assertTrue(c._circuit.allow())
        self.assertEqual(c.cache_get_json_sync(key), "ok")

    def test_big_value_not_cached(self):
        """超大 value:不缓存(防内存打爆),也不报错。"""
        key = "cache:test:t9"
        c.cache_set_json_sync(key, "x" * (c._MAX_VALUE_BYTES + 1), ttl=30)
        self.assertIsNone(c.cache_get_json_sync(key))  # 未写入

    # ---------- 熔断半开(修复:到期只放行单个探测) ----------
    def test_open_state_rejects_all(self):
        """OPEN 期间所有请求全部降级(一个都不放行)。"""
        circ = c._Circuit()
        with circ._lock:
            circ._open_until = c.time.monotonic() + 100
        self.assertEqual([circ.allow() for _ in range(5)].count(True), 0)

    def test_half_open_admits_single_probe(self):
        """开窗到期 → 半开只放行一个探测,其余仍降级(修复超时风暴)。"""
        circ = c._Circuit()
        with circ._lock:
            circ._open_until = c.time.monotonic() - 1   # 已到期但非 0(曾熔断)
            circ._next_probe_at = 0.0
        allowed = [circ.allow() for _ in range(5)]
        self.assertEqual(allowed.count(True), 1)          # 只放一个
        circ.record_success()                             # 探测成功 → 闭合
        self.assertTrue(circ.allow())

    def test_half_open_probe_failure_reopens(self):
        """探测失败 → 重新开窗,立刻又全部拒绝。"""
        circ = c._Circuit()
        with circ._lock:
            circ._consecutive_failures = c._FAILURE_THRESHOLD
            circ._open_until = c.time.monotonic() - 1
            circ._next_probe_at = 0.0
        self.assertTrue(circ.allow())        # 放到探测
        circ.record_failure()                # 探测失败 → 重开窗
        self.assertFalse(circ.allow())

    # ---------- single-flight 锁表回收(修复慢泄漏) ----------
    def test_flight_table_evicts_after_use(self):
        """get_or_load 用完后,锁表应被回收(不随 key 数无限增长)。"""
        key = "cache:test:evict"
        c.invalidate_cache_sync(key)
        c.get_or_load_sync(key, lambda: {"a": 1}, ttl=30)
        self.assertNotIn(key, c._flights_sync)

    # ---------- 熔断期失效队列(P4) ----------
    def test_invalidation_queued_during_open_and_replayed(self):
        """熔断期失效入队,恢复后首个成功请求批量重放(key 被删除)。"""
        key = "cache:test:p4"
        c.cache_set_json_sync(key, {"a": 1}, ttl=300)
        self.assertIsNotNone(c.cache_get_json_sync(key))

        circ = c._Circuit()
        with circ._lock:
            circ._open_until = c.time.monotonic() + 100   # 熔断中
        c._circuit = circ

        c.invalidate_cache_sync(key)                       # 熔断中:入队
        self.assertEqual(c.get_pending_invalidations(), 1)

        with circ._lock:                                   # 开窗到期 → 半开
            circ._open_until = c.time.monotonic() - 1
            circ._next_probe_at = 0.0
        self.assertIsNotNone(c.cache_get_json_sync(key))   # 探测成功 → 触发重放
        self.assertEqual(c.get_pending_invalidations(), 0)  # 队列已清空
        self.assertIsNone(c.cache_get_json_sync(key))       # key 已被重放删除

    # ---------- 运行指标 ----------
    def test_metrics_export(self):
        """get_metrics 返回含命中率的快照。"""
        m = c.get_metrics()
        self.assertIn("hit_rate", m)
        self.assertIsInstance(m["hit_rate"], float)


if __name__ == "__main__":
    unittest.main(verbosity=2)
