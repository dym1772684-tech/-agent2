"""Redis 限流实现 —— 三种算法都写给你看,面试和实战都够用。

================================================================
算法 1:固定窗口计数器(fixed_window_allow)
----------------------------------------------------------------
思路:每个窗口期一个计数 key,INCR 后如果是第一次就设 EXPIRE。
    INCR rl:register:192.168.1.5     → 1
    EXPIRE rl:register:... 3600      → 只有 count==1 时设置
优点:内存小(O(1))、实现简单。
缺点:**窗口边界突刺**——第 59 秒打 3 次 + 第 61 秒再打 3 次,
      1 分钟内实际通过了 2 倍限额。所以只用于低敏感场景(注册)。

算法 2:滑动窗口(sliding_window_allow)—— 本项目主用
----------------------------------------------------------------
思路:用 ZSET 存"每次请求的时间戳",统计"最近 window 秒"内的次数:
    ZADD   rl:login:ip  <now> <唯一成员>
    ZREMRANGEBYSCORE rl:login:ip 0 <now-window>   ← 删掉窗口外的旧记录
    ZCARD  rl:login:ip                              ← 数窗口内还剩几次
优点:任意时刻往前看 window 秒,没有边界突刺,平滑精确。
缺点:高 QPS 下内存比计数器大一点(每次请求一个成员)。
注意:教学版用 pipeline 保证"批量发送"但不是严格原子;
      生产严格原子要用 Lua 脚本(把三步合成一个原子操作)——面试加分点。

算法 3:登录失败锁定(record_login_fail / check_login_locked)
----------------------------------------------------------------
思路:失败计数 INCR + TTL;达到阈值后 SET 一个"锁" key(带 TTL,
到期自动解锁,不需要人工清理)。这是"账号维度"的限流,和 IP 维度互补:
- IP 限流挡"一个 IP 试很多账号"(横向爆破)
- 账号锁定挡"很多 IP 试同一个账号"(分布式爆破)

【降级策略】Redis 挂了怎么办?——本工程选择 fail-open(放行 + 打告警日志):
限流是"可用性护栏"而不是"安全边界"(鉴权才是),Redis 故障时为了不把整个
网站打死,选择放行。如果要 fail-closed,把 except 分支改成 raise 即可。
================================================================
"""
import time
import uuid

import redis as redis_lib
from fastapi import HTTPException, status

from core.redis_client import get_redis_async, redis_available

# 统一密钥前缀,方便 redis-cli 里 SCAN/监控排查
RL_KEY_REGISTER = "rl:register:{ip}"     # 注册:固定窗口
RL_KEY_LOGIN_IP = "rl:login:ip:{ip}"     # 登录-IP 维度:滑动窗口
RL_KEY_LOGIN_USER_FAIL = "rl:login:fail:{username}"   # 登录-账号失败计数
RL_KEY_LOGIN_USER_LOCK = "rl:login:lock:{username}"   # 登录-账号锁


def _fail_open(e: Exception, api: str) -> None:
    """fail-open:Redis 异常时打告警日志并放行(理由见模块 docstring)。"""
    import logging
    logging.getLogger("rate_limit").warning("Redis 不可用,限流降级放行 api=%s err=%s", api, e)
    try:  # 【降级计数】限流 fail-open 应该能在 /metrics 上看到频次
        from observability.metrics import note_degradation
        note_degradation("redis", type(e).__name__, f"rate_limit:{api}", log=False)
    except Exception:
        pass


async def fixed_window_allow(key: str, limit: int, window: int) -> tuple[bool, int]:
    """【限流·算法1】固定窗口。返回 (是否放行, 当前计数)。

    用 pipeline 把 INCR/EXPIRE 一次发完,省一次网络往返。
    """
    if not redis_available():
        return True, 0          # Redis 不可用 → 放行（限流是成本护栏，不是安全边界）
    r = get_redis_async()
    try:
        pipe = r.pipeline()
        pipe.incr(key)
        pipe.expire(key, window)   # 简化写法:每次都续 TTL。副作用=滚动窗口,
                                   # 想要严格"自然窗口"就只在 count==1 时设置
        count, _ = await pipe.execute()
        return int(count) <= limit, int(count)
    except redis_lib.RedisError as e:
        _fail_open(e, "fixed_window")
        return True, 0


async def sliding_window_allow(key: str, limit: int, window: int) -> tuple[bool, int]:
    """【限流·算法2】滑动窗口(ZSET)。返回 (是否放行, 窗口内计数)。

    成员名用 <时间戳>:<随机串>:同一毫秒内的多次请求成员名也不重复,
    否则 ZADD 会互相覆盖导致少计数。
    """
    if not redis_available():
        return True, 0
    r = get_redis_async()
    now = time.time()
    member = f"{now}:{uuid.uuid4().hex[:6]}"
    try:
        pipe = r.pipeline()
        pipe.zadd(key, {member: now})                  # 记录本次请求
        pipe.zremrangebyscore(key, 0, now - window)    # 清理窗口外的旧成员
        pipe.zcard(key)                                # 统计窗口内数量
        pipe.expire(key, window)                       # 兜底 TTL:key 永远不会变僵尸
        _, _, count, _ = await pipe.execute()
        return int(count) <= limit, int(count)
    except redis_lib.RedisError as e:
        _fail_open(e, "sliding_window")
        return True, 0


# ============================================================
# 【限流·算法3】登录失败锁定:5 次失败锁 10 分钟
# ============================================================
LOGIN_MAX_FAILS = 5
LOGIN_FAIL_WINDOW = 600     # 失败计数统计窗口:10 分钟
LOGIN_LOCK_SECONDS = 600    # 锁定时长:10 分钟(到期自动解锁)


async def check_login_locked(username: str) -> bool:
    """登录前检查:账号是否处于锁定期。锁 key 存在即锁定。"""
    if not redis_available():
        return False            # 查不到锁 → 当作未锁定，允许尝试登录
    r = get_redis_async()
    try:
        return bool(await r.exists(RL_KEY_LOGIN_USER_LOCK.format(username=username)))
    except redis_lib.RedisError as e:
        _fail_open(e, "login_lock")
        return False


async def record_login_fail(username: str) -> int:
    """密码错误后调用:失败计数 +1;达到阈值写入锁 key。返回当前失败次数。"""
    if not redis_available():
        return 0
    r = get_redis_async()
    key = RL_KEY_LOGIN_USER_FAIL.format(username=username)
    try:
        count = int(await r.incr(key))
        if count == 1:                       # 第一次失败才设置统计窗口 TTL
            await r.expire(key, LOGIN_FAIL_WINDOW)
        if count >= LOGIN_MAX_FAILS:         # 达到阈值 → 上锁(TTL=锁定时长)
            await r.set(
                RL_KEY_LOGIN_USER_LOCK.format(username=username),
                "1",
                ex=LOGIN_LOCK_SECONDS,
            )
            await r.delete(key)              # 计数清零,解锁后重新数
        return count
    except redis_lib.RedisError as e:
        _fail_open(e, "login_fail")
        return 0


async def clear_login_fail(username: str) -> None:
    """登录成功后调用:清掉失败计数(成功不应累计历史失败)。"""
    if not redis_available():
        return None
    r = get_redis_async()
    try:
        await r.delete(RL_KEY_LOGIN_USER_FAIL.format(username=username))
    except redis_lib.RedisError as e:
        _fail_open(e, "login_clear")


async def enforce_rate_limit(key: str, limit: int, window: int, api: str) -> None:
    """路由里最常用的一行式封装:超限直接抛 429。"""
    ok, _ = await sliding_window_allow(key, limit, window)
    if not ok:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"请求过于频繁({api}),请 {window} 秒内降低频率",
        )
