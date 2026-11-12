"""请求级幂等（Idempotency-Key）—— 让「同一请求被重放」只真正执行一次。

================================================================
为什么需要
----------------------------------------------------------------
/chat 本身是**非幂等**语义：同一条消息发两次 = 模型答两次 + 扣两次 token。
一旦接入写工具（记录服药、设置提醒），还会重复写业务表。
触发场景很常见：客户端超时重试、用户连点、网关重推、移动端弱网重连。

================================================================
状态机（Redis 两个 key + 一个锁）
----------------------------------------------------------------
    (无)
      │  SET NX idem:{u}:{h}:lock  EX 300      ← 原子抢占，只有一个请求能赢
      ▼
    inflight ──── commit ────▶  idem:{u}:{h}:done  EX 86400
      │                              （同时删锁）
      └──── release ────▶ (无)       ← 执行失败：允许客户端重试

- `done` 命中 → 直接回放上次响应（标 `Idempotent-Replay: true`），零 token；
- `inflight` 命中 → 409 + Retry-After，让客户端等而不是再跑一遍；
- `release` 只删锁不写 done —— 失败不该被当成「已完成」，否则用户永远拿不到结果。

================================================================
为什么必须是 SET NX，不能 GET 再 SET
----------------------------------------------------------------
两个并发请求同时 GET 都会 miss，然后都去跑 → 双扣 token，正是要防的事。
`SET key val NX EX ttl` 是单条原子命令，Redis 保证只有一个客户端返回 OK。

================================================================
Redis 不可用 → fail-open 放行
----------------------------------------------------------------
幂等是「成本与副作用护栏」，不是安全边界（安全边界是鉴权）。
Redis 挂了宁可多跑一次，也不能让整个对话不可用 —— 和 core/rate_limit.py 一致。

================================================================
明确不做：不用消息内容 hash 当键
----------------------------------------------------------------
用户真心会问「再讲一遍」，内容 hash 会把这种合理重复吞掉。
幂等键必须由客户端按「一次用户动作」生成（推荐 UUID），重试时复用同一个。
"""
from __future__ import annotations

import hashlib
import json
import logging
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger("agent.idempotency")

KEY_PREFIX = "idem:"

# inflight 锁存活时间：必须 >= 单次请求最长耗时。
# 本项目 RAG 检索 P95≈13.8s，加模型生成，单轮 P99 按 60s 估，留 5 倍余量。
# 太短 → 请求还在跑，锁已过期，第二个请求会重复执行（等于没做幂等）。
LOCK_TTL = 300

# 结果保留 24h：客户端重试窗口通常 ≤ 几分钟，够回放；也不至于把 Redis 撑爆。
DONE_TTL = 86400

# 单条响应超过 64KB 就不再缓存正文（只留「已完成」标记）。
# 目的是让 Redis 内存可控，同时保证「不会重复执行」这个核心语义不丢。
CACHE_LIMIT = 64 * 1024

# 三态（对外暴露成常量，调用方不用记字符串）
ST_NEW = "new"            # 抢到锁，可以执行
ST_INFLIGHT = "inflight"  # 别的请求正在跑
ST_DONE = "done"          # 已经跑完，payload 是上次的响应


def normalize_key(user_id: str, raw: Optional[str]) -> Optional[str]:
    """把客户端给的幂等键收敛成 Redis key；空键返回 None（= 本次不做幂等）。

    - 按 user 隔离：别人拿到你的幂等键也回放不了你的响应（越权回放面）；
    - 做 sha256：客户端可能传 UUID / 长串 / 带特殊字符，哈希后长度固定且安全；
    - 空串返回 None：不能让「没传键」的请求都撞到同一个 hash 上互相回放。
    """
    text = (raw or "").strip()
    if not text:
        return None
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:32]
    return f"{KEY_PREFIX}{user_id}:{digest}"


def _lock_key(key: str) -> str:
    return f"{key}:lock"


def _done_key(key: str) -> str:
    return f"{key}:done"


async def _read_done(redis: Any, key: str) -> Optional[Dict[str, Any]]:
    """读已完成的结果；解析失败按「已完成但无正文」处理（宁可少回放，不可重复执行）。"""
    raw = await redis.get(_done_key(key))
    if not raw:
        return None
    try:
        payload = json.loads(raw)
    except Exception as exc:
        logger.warning("[idem] 缓存响应解析失败，按无正文处理 key=%s err=%s", key, exc)
        return {"_no_cache": True}
    return payload if isinstance(payload, dict) else {"_no_cache": True}


async def acquire(redis: Any, key: Optional[str]) -> Tuple[str, Optional[Dict[str, Any]]]:
    """抢占一次执行权，返回 (状态, 已缓存的响应)。

    顺序很关键：
      ① 先查 done —— 已完成的请求直接回放，不必抢锁（少一次写）；
      ② SET NX 抢锁 —— 原子占位；
      ③ 抢不到时**再查一次** done —— 覆盖「刚好在 ①②之间完成」的窗口，
         否则客户端会收到不该有的 409。
    """
    if not key or redis is None:
        return ST_NEW, None
    try:
        cached = await _read_done(redis, key)
        if cached is not None:
            return ST_DONE, cached

        got = await redis.set(_lock_key(key), "1", nx=True, ex=LOCK_TTL)
        if got:
            return ST_NEW, None

        cached = await _read_done(redis, key)
        if cached is not None:
            return ST_DONE, cached
        return ST_INFLIGHT, None
    except Exception as exc:
        logger.warning("[idem] Redis 不可用，降级放行（本次可能重复执行）：%s", exc)
        return ST_NEW, None


async def commit(redis: Any, key: Optional[str], payload: Any) -> bool:
    """执行成功后落结果并释放锁。任何失败都只记日志，绝不影响本次响应。"""
    if not key or redis is None:
        return False
    try:
        text = json.dumps(payload if payload is not None else {}, ensure_ascii=False, default=str)
        if len(text) > CACHE_LIMIT:
            # 正文太大只留标记：核心语义是「不重复执行」，不是「一定回放得出内容」
            text = json.dumps(
                {"_no_cache": True, "_reason": "response_too_large", "_bytes": len(text)},
                ensure_ascii=False,
            )
        await redis.set(_done_key(key), text, ex=DONE_TTL)
        await redis.delete(_lock_key(key))
        return True
    except Exception as exc:
        logger.warning("[idem] 结果落盘失败（不影响本次响应）：%s", exc)
        return False


async def release(redis: Any, key: Optional[str]) -> None:
    """执行失败时释放锁，允许客户端用同一个键重试。

    不写 done：写 done 会让「那一次失败」变成终态，用户以后再也拿不到结果。
    """
    if not key or redis is None:
        return
    try:
        await redis.delete(_lock_key(key))
    except Exception as exc:
        logger.warning("[idem] 释放锁失败（锁会在 TTL 后自动过期）：%s", exc)


async def peek(redis: Any, key: Optional[str]) -> Optional[Dict[str, Any]]:
    """只读查询（排查/管理端点用），不抢锁、不改状态。"""
    if not key or redis is None:
        return None
    try:
        return await _read_done(redis, key)
    except Exception:
        return None
