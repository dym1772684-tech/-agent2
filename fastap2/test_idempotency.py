"""幂等四层的零 LLM 单测。

覆盖：
  L1 HTTP 幂等键：三态状态机（new/inflight/done）、失败释放、Redis 挂 fail-open、
                 大响应只留标记、按 user 隔离
  L3 写工具幂等：读写区分（写工具不盲目重试）、幂等键注入判定、键格式、白名单解析
  L3 服务端装饰器：无键直通 / 命中缓存不回表 / 并发等结果 / 异常释放锁
  L4 线程登记：insert 撞 unique 后复查（并发时不再 500）

运行：
    cd fastap2
    python test_idempotency.py
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))            # 练手agent（allagent2 / fastap2 的父目录）
sys.path.insert(0, str(_ROOT / "fastap2"))

from sqlalchemy.exc import IntegrityError

from core import idempotency as idem
from core import thread_registry as registry
from mcp_servers import idempotency as mcp_idem
from models.acl import ConversationOwner


# ============================================================
# 假的 Redis（只实现本模块用到的 4 个命令）
# ============================================================
class FakeRedis:
    def __init__(self):
        self.data = {}
        self.fail = False
        self.calls = []

    async def get(self, key):
        self.calls.append(("get", key))
        if self.fail:
            raise RuntimeError("redis down")
        return self.data.get(key)

    async def set(self, key, value, nx=False, ex=None):
        self.calls.append(("set", key))
        if self.fail:
            raise RuntimeError("redis down")
        if nx and key in self.data:
            return None
        self.data[key] = value
        return True

    async def delete(self, key):
        self.calls.append(("delete", key))
        if self.fail:
            raise RuntimeError("redis down")
        self.data.pop(key, None)
        return 1


class _FakeResult:
    def __init__(self, row):
        self._row = row

    def scalar_one_or_none(self):
        return self._row


class FakeDB:
    """按顺序返回预置行；可选让第一次 commit 抛 IntegrityError（模拟并发撞索引）。"""

    def __init__(self, rows=None, commit_conflict=False):
        self._rows = list(rows or [])
        self._commit_conflict = commit_conflict
        self.added = []
        self.commits = 0
        self.rollbacks = 0

    async def execute(self, stmt):
        return _FakeResult(self._rows.pop(0) if self._rows else None)

    def add(self, obj):
        self.added.append(obj)

    async def commit(self):
        self.commits += 1
        if self._commit_conflict:
            self._commit_conflict = False
            raise IntegrityError("INSERT conversation_owner", {}, Exception("duplicate"))

    async def rollback(self):
        self.rollbacks += 1


def run(coro):
    import asyncio

    return asyncio.get_event_loop_policy().new_event_loop().run_until_complete(coro)


# ============================================================
# L1：HTTP 幂等键
# ============================================================
class T1NormalizeKey(unittest.TestCase):
    def test_empty_returns_none(self):
        self.assertIsNone(idem.normalize_key("u1", ""))
        self.assertIsNone(idem.normalize_key("u1", "   "))
        self.assertIsNone(idem.normalize_key("u1", None))

    def test_same_input_same_key(self):
        self.assertEqual(
            idem.normalize_key("u1", "req-abc"), idem.normalize_key("u1", "  req-abc  ")
        )

    def test_user_isolated(self):
        """别人拿到你的幂等键也不能回放你的响应（越权回放面）。"""
        self.assertNotEqual(
            idem.normalize_key("u1", "req-abc"), idem.normalize_key("u2", "req-abc")
        )

    def test_fixed_length(self):
        long_key = "x" * 500
        key = idem.normalize_key("u1", long_key)
        self.assertEqual(len(key), len("idem:u1:") + 32)


class T2StateMachine(unittest.TestCase):
    def test_new_then_inflight_then_done(self):
        redis = FakeRedis()
        key = idem.normalize_key("u1", "k1")

        state, cached = run(idem.acquire(redis, key))
        self.assertEqual(state, idem.ST_NEW)
        self.assertIsNone(cached)

        # 第一个请求还没跑完，第二个同键请求必须被挡住（否则双扣 token）
        state, cached = run(idem.acquire(redis, key))
        self.assertEqual(state, idem.ST_INFLIGHT)

        run(idem.commit(redis, key, {"answer": "已经答过了"}))
        state, cached = run(idem.acquire(redis, key))
        self.assertEqual(state, idem.ST_DONE)
        self.assertEqual(cached["answer"], "已经答过了")

    def test_release_allows_retry(self):
        """失败不写 done，否则用户永远拿不到这一轮结果。"""
        redis = FakeRedis()
        key = idem.normalize_key("u1", "k2")
        run(idem.acquire(redis, key))
        run(idem.release(redis, key))
        state, _ = run(idem.acquire(redis, key))
        self.assertEqual(state, idem.ST_NEW)

    def test_commit_clears_lock(self):
        redis = FakeRedis()
        key = idem.normalize_key("u1", "k3")
        run(idem.acquire(redis, key))
        run(idem.commit(redis, key, {"ok": True}))
        self.assertNotIn(f"{key}:lock", redis.data)
        self.assertIn(f"{key}:done", redis.data)

    def test_redis_down_fail_open(self):
        redis = FakeRedis()
        redis.fail = True
        state, cached = run(idem.acquire(redis, idem.normalize_key("u1", "k4")))
        self.assertEqual(state, idem.ST_NEW)  # 放行，不能把对话打死
        self.assertIsNone(cached)
        self.assertFalse(run(idem.commit(redis, "idem:u1:x", {})))
        run(idem.release(redis, "idem:u1:x"))  # 不抛异常

    def test_none_key_is_noop(self):
        redis = FakeRedis()
        self.assertEqual(run(idem.acquire(redis, None))[0], idem.ST_NEW)
        self.assertFalse(run(idem.commit(redis, None, {})))
        self.assertEqual(redis.calls, [])  # 没传键就不碰 Redis

    def test_large_response_only_marks_done(self):
        """正文过大不缓存内容，但「已完成」这个事实必须记住（防重复执行）。"""
        redis = FakeRedis()
        key = idem.normalize_key("u1", "k5")
        run(idem.acquire(redis, key))
        run(idem.commit(redis, key, {"blob": "x" * (idem.CACHE_LIMIT + 10)}))
        state, cached = run(idem.acquire(redis, key))
        self.assertEqual(state, idem.ST_DONE)
        self.assertTrue(cached.get("_no_cache"))

    def test_corrupt_payload_still_blocks_duplicate(self):
        redis = FakeRedis()
        key = idem.normalize_key("u1", "k6")
        redis.data[f"{key}:done"] = "{not json"
        state, cached = run(idem.acquire(redis, key))
        self.assertEqual(state, idem.ST_DONE)
        self.assertTrue(cached.get("_no_cache"))


# ============================================================
# L3：写工具幂等（读侧重试，写不重试）
# ============================================================
class _CountingTool:
    """每次都抛异常的工具，用来数实际执行次数。"""

    def __init__(self, args=None, always_fail=True):
        self.args = args if args is not None else {}
        self.calls = 0
        self.always_fail = always_fail

    def invoke(self, args):
        self.calls += 1
        if self.always_fail:
            raise RuntimeError("boom")
        return f"ok:{self.calls}"


class T3RetryPolicy(unittest.TestCase):
    def setUp(self):
        import allagent2.agent.react_loop as rl

        self.rl = rl
        # 重试间隔会真 sleep(0.3/0.6)，测试里去掉
        self._orig_sleep = rl.time.sleep
        rl.time.sleep = lambda *_a, **_k: None

    def tearDown(self):
        self.rl.time.sleep = self._orig_sleep

    def test_read_tool_retries(self):
        tool = _CountingTool()
        result, error = self.rl._invoke_with_retry(tool, {}, retries=2, write=False)
        self.assertEqual(tool.calls, 3)  # 1 次 + 2 次重试
        self.assertIsNone(result)
        self.assertIn("boom", error)

    def test_write_tool_does_not_retry(self):
        """写工具没声明幂等键 → 只能跑一次。这是「重复写库」的主要防线。"""
        tool = _CountingTool()
        _, error = self.rl._invoke_with_retry(tool, {}, retries=2, write=True)
        self.assertEqual(tool.calls, 1)
        self.assertIn("boom", error)

    def test_write_tool_with_idem_key_retries(self):
        """声明了 idempotency_key = 服务端能去重，此时允许重试。"""
        tool = _CountingTool(args={"elder_id": {}, "idempotency_key": {}})
        self.rl._invoke_with_retry(tool, {}, retries=2, write=True)
        self.assertEqual(tool.calls, 3)

    def test_supports_idempotency_key(self):
        self.assertFalse(self.rl.supports_idempotency_key(_CountingTool()))
        self.assertTrue(
            self.rl.supports_idempotency_key(_CountingTool(args={"idempotency_key": {}}))
        )
        self.assertFalse(self.rl.supports_idempotency_key(object()))  # 无 args 属性

    def test_idempotency_key_format(self):
        key = self.rl.tool_idempotency_key("t-abc", "turn-1", "call_x")
        self.assertEqual(key, "t-abc:turn-1:call_x")
        # 同一次动作 + 同一个 tool_call 恒定：重试多少次都是同一个键
        self.assertEqual(key, self.rl.tool_idempotency_key("t-abc", "turn-1", "call_x"))

    def test_write_tools_env(self):
        import os

        old = os.environ.get(self.rl._WRITE_TOOLS_ENV)
        try:
            os.environ[self.rl._WRITE_TOOLS_ENV] = " record_medication , set_reminder ,"
            self.assertEqual(
                self.rl.write_tools(), {"record_medication", "set_reminder"}
            )
            os.environ[self.rl._WRITE_TOOLS_ENV] = ""
            self.assertEqual(self.rl.write_tools(), set())
        finally:
            if old is None:
                os.environ.pop(self.rl._WRITE_TOOLS_ENV, None)
            else:
                os.environ[self.rl._WRITE_TOOLS_ENV] = old


# ============================================================
# L3 服务端：MCP 装饰器
# ============================================================
class T4McpDecorator(unittest.TestCase):
    def setUp(self):
        self._orig_redis = mcp_idem._redis
        self._orig_wait = mcp_idem.WAIT_ROUNDS
        self.redis = FakeRedis()
        mcp_idem._redis = lambda: _async_value(self.redis)

    def tearDown(self):
        mcp_idem._redis = self._orig_redis
        mcp_idem.WAIT_ROUNDS = self._orig_wait

    def test_no_key_passthrough(self):
        calls = []

        @mcp_idem.idempotent
        async def tool(elder_id: str, idempotency_key: str = ""):
            calls.append(elder_id)
            return "written"

        self.assertEqual(run(tool("E001")), "written")
        self.assertEqual(len(calls), 1)
        self.assertEqual(self.redis.calls, [])  # 没键不碰 Redis

    def test_second_call_returns_cache(self):
        calls = []

        @mcp_idem.idempotent
        async def tool(elder_id: str, idempotency_key: str = ""):
            calls.append(elder_id)
            return f"written#{len(calls)}"

        first = run(tool("E001", idempotency_key="t-1:turn-1:call_1"))
        second = run(tool("E001", idempotency_key="t-1:turn-1:call_1"))
        self.assertEqual(first, "written#1")
        self.assertEqual(second, "written#1")  # 回放缓存，没有第二次写
        self.assertEqual(len(calls), 1)

    def test_busy_returns_hint(self):
        """同一键正在执行 → 提示勿重复提交，而不是再执行一次。"""
        mcp_idem.WAIT_ROUNDS = 0
        key = "tool:idem:t-1:turn-1:call_2"
        self.redis.data[f"{key}:lock"] = "1"
        calls = []

        @mcp_idem.idempotent
        async def tool(idempotency_key: str = ""):
            calls.append(1)
            return "written"

        out = run(tool(idempotency_key="t-1:turn-1:call_2"))
        self.assertIn("请勿重复提交", out)
        self.assertEqual(calls, [])

    def test_exception_releases_lock(self):
        calls = []

        @mcp_idem.idempotent
        async def tool(idempotency_key: str = ""):
            calls.append(1)
            raise RuntimeError("db timeout")

        with self.assertRaises(RuntimeError):
            run(tool(idempotency_key="t-1:turn-1:call_3"))
        self.assertNotIn("tool:idem:t-1:turn-1:call_3:lock", self.redis.data)
        # 失败不落 done → 允许重试
        self.assertNotIn("tool:idem:t-1:turn-1:call_3", self.redis.data)

        @mcp_idem.idempotent
        async def ok_tool(idempotency_key: str = ""):
            calls.append(1)
            return "written"

        self.assertEqual(run(ok_tool(idempotency_key="t-1:turn-1:call_3")), "written")

    def test_redis_down_fail_open(self):
        self.redis.fail = True
        calls = []

        @mcp_idem.idempotent
        async def tool(idempotency_key: str = ""):
            calls.append(1)
            return "written"

        self.assertEqual(run(tool(idempotency_key="k")), "written")
        self.assertEqual(len(calls), 1)


async def _async_value(value):
    return value


# ============================================================
# L2：act 节点台账复用（直接调真节点函数）
# ============================================================
class _NoopModel:
    """只满足 build_react_graph 的编译期需要，永远不会被调用。"""

    def bind_tools(self, tools):
        return self

    def invoke(self, messages):
        raise RuntimeError("测试不应触发模型调用")


def _make_graph(tools):
    from langgraph.checkpoint.memory import InMemorySaver

    from allagent2.agent.react_loop import build_react_graph

    return build_react_graph(
        model=_NoopModel(), tools=tools, checkpointer=InMemorySaver()
    )


def _act_of(graph):
    """拿到未包装的 act_node：编译后 nodes['act'].bound.func 就是它（functools.wraps 保留）。"""
    return graph.nodes["act"].bound.func


def _ai_with_call(name, args, call_id):
    from langchain.messages import AIMessage

    return AIMessage(
        content="", tool_calls=[{"name": name, "args": args, "id": call_id}]
    )


class T6ActLedger(unittest.TestCase):
    def _tool(self, calls, name="query_health_records", extra=None):
        from langchain_core.tools import StructuredTool

        def _fn(**kwargs):
            calls.append(kwargs)
            return "收缩压 150/95"

        return StructuredTool.from_function(
            func=_fn, name=name, description="测试用工具"
        )

    def test_replay_reuses_ledger_without_executing(self):
        """act 重放（checkpoint 恢复 / 续跑）命中台账 → 工具一次都不执行。"""
        calls = []
        graph = _make_graph([self._tool(calls)])
        act = _act_of(graph)

        state = {
            "messages": [_ai_with_call("query_health_records", {"elder_id": "E001"}, "call_1")],
            "tool_ledger": {
                "call_1": {
                    "name": "query_health_records",
                    "status": "ok",
                    "text": "收缩压 150/95",
                }
            },
            "tool_results": {},
            "round": 2,
            "turn_id": "turn-1",
        }
        out = act(state, None)

        self.assertEqual(calls, [])  # 核心断言：没有第二次执行
        self.assertEqual(out["messages"][0].content, "收缩压 150/95")
        self.assertEqual(out["messages"][0].tool_call_id, "call_1")
        # traced_node 会把 _events 收进 trace 通道，同时给事件打上 reused 标记
        self.assertTrue(any(e.get("reused") for e in out["trace"]))

    def test_first_run_executes_and_records_ledger(self):
        calls = []
        graph = _make_graph([self._tool(calls)])
        act = _act_of(graph)

        state = {
            "messages": [_ai_with_call("query_health_records", {"elder_id": "E001"}, "call_1")],
            "tool_results": {},
            "tool_ledger": {},
            "round": 1,
            "turn_id": "turn-1",
        }
        out = act(state, None)

        self.assertEqual(len(calls), 1)
        self.assertEqual(out["tool_ledger"]["call_1"]["status"], "ok")
        self.assertIn("收缩压", out["tool_ledger"]["call_1"]["text"])
        self.assertFalse(any(e.get("reused") for e in out["trace"]))

    def test_denied_call_is_also_remembered(self):
        """越权拦截也进台账：重放时不再重复判定、不再产生重复事件语义。"""
        calls = []
        graph = _make_graph([self._tool(calls)])
        act = _act_of(graph)
        from allagent2.agent.state import PermissionContext

        state = {
            "messages": [_ai_with_call("query_health_records", {"elder_id": "E999"}, "call_1")],
            "tool_results": {},
            "tool_ledger": {
                "call_1": {"name": "query_health_records", "status": "denied", "text": "无权访问"}
            },
            "permissions": PermissionContext(allowed_elder_ids=["E001"]),
            "round": 1,
            "turn_id": "turn-1",
        }
        out = act(state, None)
        self.assertEqual(calls, [])
        self.assertEqual(out["tool_ledger"]["call_1"]["status"], "denied")

    def test_write_tool_gets_idempotency_key_injected(self):
        """写工具若声明了 idempotency_key，act 必须自动注入确定性幂等键。"""
        import os

        from langchain_core.tools import StructuredTool

        received = []

        def _record_medication(elder_id: str, drug: str, idempotency_key: str = "") -> str:
            received.append(
                {"elder_id": elder_id, "drug": drug, "idempotency_key": idempotency_key}
            )
            return "已记录"

        tool = StructuredTool.from_function(
            func=_record_medication, name="record_medication", description="记录服药"
        )
        graph = _make_graph([tool])
        act = _act_of(graph)

        old = os.environ.get("AGENT_WRITE_TOOLS")
        os.environ["AGENT_WRITE_TOOLS"] = "record_medication"
        try:
            from allagent2.agent.state import PermissionContext

            state = {
                "messages": [
                    _ai_with_call(
                        "record_medication", {"elder_id": "E001", "drug": "氨氯地平"}, "call_9"
                    )
                ],
                "tool_results": {},
                "tool_ledger": {},
                "round": 1,
                "turn_id": "turn-abc",
                # 默认白名单只有 4 个只读工具，写工具必须显式授权才不被 guard 拦
                "permissions": PermissionContext(
                    allowed_tools=["record_medication"], allowed_elder_ids=["E001"]
                ),
            }
            act(state, None)
        finally:
            if old is None:
                os.environ.pop("AGENT_WRITE_TOOLS", None)
            else:
                os.environ["AGENT_WRITE_TOOLS"] = old

        self.assertEqual(len(received), 1)
        # thread_id 在无 runtime 时为 "-"，键仍然是确定性的三段落结构
        self.assertEqual(received[0]["idempotency_key"], "-:turn-abc:call_9")


# ============================================================
# L4：线程登记并发
# ============================================================
class T5ThreadRegistry(unittest.TestCase):
    def test_new_thread_generated_and_registered(self):
        db = FakeDB()
        tid = run(registry.resolve_thread_id(db, "u1", None))
        self.assertTrue(tid.startswith("t-"))
        self.assertEqual(len(db.added), 1)
        self.assertEqual(db.added[0].user_id, "u1")

    def test_existing_owner_passes(self):
        db = FakeDB(rows=[ConversationOwner(thread_id="t1", user_id="u1")])
        self.assertEqual(run(registry.resolve_thread_id(db, "u1", "t1")), "t1")
        self.assertEqual(db.added, [])
        self.assertEqual(db.commits, 0)

    def test_other_owner_denied(self):
        from fastapi import HTTPException

        db = FakeDB(rows=[ConversationOwner(thread_id="t1", user_id="u2")])
        with self.assertRaises(HTTPException) as ctx:
            run(registry.resolve_thread_id(db, "u1", "t1"))
        self.assertEqual(ctx.exception.status_code, 403)

    def test_concurrent_insert_then_owner_is_other(self):
        """并发对手先插入了（撞 unique）→ 回滚后复查 → 按「别人的线程」403，而不是 500。"""
        from fastapi import HTTPException

        db = FakeDB(
            rows=[
                None,  # 首次 select：未登记
                ConversationOwner(thread_id="t1", user_id="u2"),  # 撞索引后复查
            ],
            commit_conflict=True,
        )
        with self.assertRaises(HTTPException) as ctx:
            run(registry.resolve_thread_id(db, "u1", "t1"))
        self.assertEqual(ctx.exception.status_code, 403)
        self.assertEqual(db.rollbacks, 1)

    def test_concurrent_insert_then_owner_is_me(self):
        """并发对手就是你自己的重试 → 复查到自己的行，正常续接。"""
        db = FakeDB(
            rows=[None, ConversationOwner(thread_id="t1", user_id="u1")],
            commit_conflict=True,
        )
        self.assertEqual(run(registry.resolve_thread_id(db, "u1", "t1")), "t1")

    def test_conflict_without_row_is_409(self):
        from fastapi import HTTPException

        db = FakeDB(rows=[None, None], commit_conflict=True)
        with self.assertRaises(HTTPException) as ctx:
            run(registry.resolve_thread_id(db, "u1", "t1"))
        self.assertEqual(ctx.exception.status_code, 409)

    def test_assert_owner(self):
        from fastapi import HTTPException

        run(registry.assert_thread_owner(
            FakeDB(rows=[ConversationOwner(thread_id="t1", user_id="u1")]), "u1", "t1"
        ))
        with self.assertRaises(HTTPException) as ctx:
            run(registry.assert_thread_owner(FakeDB(rows=[None]), "u1", "t1"))
        self.assertEqual(ctx.exception.status_code, 404)


if __name__ == "__main__":
    unittest.main(verbosity=2)
