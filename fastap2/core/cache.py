"""Redis 缓存 —— Cache-Aside(旁路缓存)模式 + 三种"缓存刷新"策略。

================================================================
【缓存·读路径】Cache-Aside 标准三步:
    1. 读缓存 → 命中直接返回(不碰数据库)
    2. 未命中 → 查数据库
    3. 把数据库结果写回缓存(带 TTL),返回

【缓存·刷新策略】(面试高频,三种都要会说)
----------------------------------------------------------------
策略 1  TTL 被动过期:写入时设过期时间,到期 Redis 自动删。
        最简单,兜底一切;缺点=数据有"TTL 时长"的陈旧窗口。
策略 2  写后删除(Delete on write)—— 本工程采用:
        数据库写成功后,主动 DEL 对应缓存 key,下次读自然回源。
        为什么是"删缓存"而不是"更新缓存"?
        → 并发写时"更新缓存"容易把旧值写回(写库快的请求先算完成,
          写缓存却后到),删缓存让下一次读去库拿最新值,天然收敛。
        顺序必须是【先更新库,再删缓存】(反过来=删完又有旧读回填)。
策略 3  延迟双删:写库前删一次,写库后延迟几百毫秒再删一次,
        兜住"删除后、库提交前,并发读把旧值回填"的窄窗口。
        本工程 TTL 短 + 场景容忍,不启用,知道即可。

【缓存穿透防护】查一个【不存在】的 elder_id:缓存永远未命中 →
每次都打到库。解法=空结果也缓存(空值哨兵 NULL_SENTINEL,短 TTL),
本工程在 DB 网关 _query_profile 里就是这么做 的。

【其他两个面试词,知道即可】
- 缓存击穿:某个热 key 过期瞬间大量请求同时回源 → 加分布式锁/逻辑过期。
- 缓雪崩:大批 key 同一时刻过期 → TTL 加随机抖动。
================================================================
"""
import json
from typing import Any

import redis as redis_lib
import redis.asyncio as aioredis

from core.redis_client import get_redis_async, get_redis_sync, redis_available

# 空值哨兵:代表"数据库里确实没有这条数据",用于防穿透,别当成业务值返回
NULL_SENTINEL = "__CACHE_NULL__"


def _note_redis_down(exc: BaseException, where: str) -> None:
    """【降级计数】缓存读写失败 = Redis 不可用；高频路径只计数不打日志。"""
    try:
        from observability.metrics import note_degradation

        note_degradation("redis", type(exc).__name__, f"cache:{where}", log=False)
    except Exception:
        pass


# key 命名规范:<业务>:<id>,前缀统一,方便 SCAN 排查和按前缀失效
USER_CACHE_KEY = "cache:user:{user_id}"        # JWT 鉴权用户信息(短 TTL)
ELDER_CACHE_KEY = "cache:elder:{elder_id}"     # 老人档案文本(网关查询用)


def user_cache_key(user_id: str) -> str:
    return USER_CACHE_KEY.format(user_id=user_id)


def elder_cache_key(elder_id: str) -> str:
    return ELDER_CACHE_KEY.format(elder_id=elder_id)


# ============================================================
# 异步版(给 FastAPI 的 async 路由/依赖用,如 check_jwt)
# ============================================================
async def cache_get_json_async(key: str) -> Any:
    """读缓存:命中返回反序列化后的值;空哨兵返回 NULL_SENTINEL;未命中/Redis 挂返回 None。

    注意三态区分:None=未命中(该回源),NULL_SENTINEL=库里有结论"不存在"(不用回源)。
    """
    try:
        raw = await get_redis_async().get(key)
    except (redis_lib.RedisError, aioredis.RedisError) as exc:
        _note_redis_down(exc, 'get_async')
        return None  # 【降级】Redis 挂了当作未命中,直接回源数据库(牺牲性能保正确)
    if raw is None:
        return None
    return json.loads(raw)


async def cache_set_json_async(key: str, value: Any, ttl: int) -> None:
    """写缓存:JSON 序列化 + TTL(策略1 被动过期兜底)。"""
    try:
        await get_redis_async().set(key, json.dumps(value, default=str), ex=ttl)
    except (redis_lib.RedisError, aioredis.RedisError) as exc:
        _note_redis_down(exc, 'set_async')
        pass  # 【降级】缓存写失败不影响主流程,只是下次还要回源


async def invalidate_user_cache(user_id: str) -> None:
    """【缓存刷新·策略2 写后删除】管理端修改角色/禁用账号后调用。"""
    try:
        await get_redis_async().delete(user_cache_key(user_id))
    except (redis_lib.RedisError, aioredis.RedisError) as exc:
        _note_redis_down(exc, 'invalidate_user')
        pass


# ============================================================
# 同步版(给线程池里的同步代码用:agent 的 DB 网关、审计)
# ============================================================
def cache_get_json_sync(key: str) -> Any:
    try:
        raw = get_redis_sync().get(key)
    except redis_lib.RedisError as exc:
        _note_redis_down(exc, 'get_sync')
        return None
    if raw is None:
        return None
    return json.loads(raw)


def cache_set_json_sync(key: str, value: Any, ttl: int) -> None:
    try:
        get_redis_sync().set(key, json.dumps(value, default=str), ex=ttl)
    except redis_lib.RedisError as exc:
        _note_redis_down(exc, 'set_sync')
        pass


def invalidate_elder_cache(elder_id: str) -> None:
    """【缓存刷新·策略2 写后删除】老人档案增/改/删后调用,保持缓存与库一致。"""
    try:
        get_redis_sync().delete(elder_cache_key(elder_id))
    except redis_lib.RedisError as exc:
        _note_redis_down(exc, 'invalidate_elder')
        pass
