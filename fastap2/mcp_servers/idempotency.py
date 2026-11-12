"""MCP 写工具的幂等装饰器（幂等第 3 层·服务端）。

================================================================
为什么要把幂等放在「工具内部」而不是只放在 HTTP 层
----------------------------------------------------------------
前面两层都可能失效：
  - HTTP 的 Idempotency-Key 依赖客户端老实复用同一个键，第三方/老客户端不会传；
  - act 节点的 tool_ledger 依赖 checkpoint 完整，进程重启 + InMemorySaver 降级就没了。
只有「真正产生副作用的那一端」自己认幂等键，才是最后一道可靠防线。

================================================================
协议
----------------------------------------------------------------
工具签名加一个 `idempotency_key: str = ""`：

    @mcp.tool()
    @idempotent
    async def record_medication(elder_id: str, drug: str, idempotency_key: str = "") -> str:
        ...

注意装饰顺序：`@mcp.tool()` 必须在**最外层**（FastMCP 注册的是最外层函数，
functools.wraps 保留了签名，工具 schema 里会出现 idempotency_key 参数，
fastap2 侧的 `supports_idempotency_key()` 才会自动注入）。

调用方（act 节点）生成的键是 `{thread_id}:{turn_id}:{tool_call_id}`：
同一次用户动作无论重试几次，这个键都不变 → 只生效一次。

================================================================
三态（与 core/idempotency.py 同构）
----------------------------------------------------------------
    done        → 命中缓存，直接返回上次结果（不碰数据库）
    new         → 抢到锁，执行并落结果
    busy        → 同一键正在执行，等一小会儿再查；仍没有就提示「勿重复提交」
    passthrough → Redis 不可用，放行（幂等是护栏，不能挡住业务）

================================================================
没有 idempotency_key 时
----------------------------------------------------------------
直接执行不缓存。宁可重复，也不要因为客户端没传键就跑不了 ——
这是「渐进式幂等」：老调用方行为不变，新调用方获得保护。
"""
from __future__ import annotations

import asyncio
import functools
import logging
from typing import Any, Tuple

logger = logging.getLogger("mcp.idempotency")

# 与 fastap2/core/idempotency.py 保持一致的量级
LOCK_TTL = 120        # 单次写工具执行上限（够慢查询 + 写库）
DONE_TTL = 86400      # 结果保留 24h，覆盖客户端重试窗口
WAIT_ROUNDS = 10      # 抢不到锁时最多等 2s（10 × 0.2s）
WAIT_INTERVAL = 0.2


async def _redis():
    """懒导入：MCP 服务启动时不因为 Redis 依赖而失败。"""
    from core.redis_client import get_redis_async

    return get_redis_async()


async def _begin(done_key: str) -> Tuple[str, Any]:
    """判断这次调用该「回放 / 执行 / 等待 / 放行」。"""
    try:
        redis = await _redis()
        cached = await redis.get(done_key)
        if cached is not None:
            return "done", cached

        got = await redis.set(f"{done_key}:lock", "1", nx=True, ex=LOCK_TTL)
        if got:
            return "new", None

        # 没抢到锁：可能是并发的同一调用。等它落结果，而不是直接拒绝
        for _ in range(WAIT_ROUNDS):
            await asyncio.sleep(WAIT_INTERVAL)
            cached = await redis.get(done_key)
            if cached is not None:
                return "done", cached
        return "busy", None
    except Exception as exc:
        logger.warning("[idem] Redis 不可用，工具幂等降级放行：%s", exc)
        return "passthrough", None


async def _commit(done_key: str, result: Any) -> None:
    try:
        redis = await _redis()
        await redis.set(done_key, str(result), ex=DONE_TTL)
    except Exception as exc:
        logger.warning("[idem] 工具结果落盘失败（不影响本次返回）：%s", exc)
    finally:
        await _release(done_key)


async def _release(done_key: str) -> None:
    try:
        redis = await _redis()
        await redis.delete(f"{done_key}:lock")
    except Exception:
        pass  # 锁会在 TTL 后自动过期


def idempotent(fn):
    """给写工具加「同一幂等键只生效一次」能力（async 工具专用）。"""

    @functools.wraps(fn)
    async def wrapper(*args: Any, **kwargs: Any) -> Any:
        key = str(kwargs.get("idempotency_key") or "").strip()
        if not key:
            return await fn(*args, **kwargs)

        done_key = f"tool:idem:{key}"
        verdict, cached = await _begin(done_key)

        if verdict == "done":
            logger.info("[idem] 工具命中幂等缓存，跳过执行：%s", done_key)
            return cached
        if verdict == "busy":
            logger.warning("[idem] 同一幂等键正在执行，拒绝重复提交：%s", done_key)
            return "该操作正在处理中，请勿重复提交"

        # new（已抢到锁）/ passthrough（Redis 挂了，放行）
        try:
            result = await fn(*args, **kwargs)
        except Exception:
            await _release(done_key)  # 失败不落 done，允许重试
            raise

        if verdict == "new":
            await _commit(done_key, result)
        return result

    return wrapper
