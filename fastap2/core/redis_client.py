"""Redis 连接管理(同步 + 异步两个客户端)。

【为什么需要两个客户端】
- 异步客户端(redis.asyncio):给 FastAPI 的 async 路由/依赖用,await 不阻塞事件循环;
- 同步客户端(redis.Redis):给跑在线程池里的同步代码用(agent 的 DB 网关、审计写入)。

【连接配置】从环境变量读取(REDIS_HOST/PORT/PASSWORD/DB,写在 core/key.env):
- decode_responses=True:redis 返回 bytes 时自动解码成 str,省得到处 .decode()
- 连接池由 redis-py 内部管理(默认 max_connections=2^31,足够)
- Redis 不可用时的策略:上层(cache/rate_limit)自行决定降级,本模块只负责连接
"""
import os
import threading
import time
from pathlib import Path

import redis
import redis.asyncio as aioredis
from dotenv import load_dotenv

# 【可用性】redis-py 8.x 默认带指数退避重试：Redis 挂掉且端口被丢包时，
# 一次 ping 会等 25 秒（2s 超时 × 十几次），把登录/对话整条请求拖死。
# 这里显式关掉重试：每次调用最多一个 socket 超时，剩下的交给上层降级。
try:
    from redis.backoff import NoBackoff
    from redis.retry import Retry

    _NO_RETRY = Retry(NoBackoff(), 0)
except Exception:  # 老版本没有 retry 模块
    _NO_RETRY = None

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / "core" / "key.env")  # 幂等:重复 load 不会覆盖已有环境变量

REDIS_HOST = os.getenv("REDIS_HOST", "127.0.0.1")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD") or None  # 空串转 None = 无密码模式
REDIS_DB = int(os.getenv("REDIS_DB", "0"))

# 模块级单例:进程内只建一次连接池,所有调用方复用
_sync_client: redis.Redis | None = None
_async_client: aioredis.Redis | None = None


def get_redis_sync() -> redis.Redis:
    """同步客户端(线程池里的代码用:DB 网关、审计)。"""
    global _sync_client
    if _sync_client is None:
        _sync_client = redis.Redis(
            host=REDIS_HOST,
            port=REDIS_PORT,
            password=REDIS_PASSWORD,
            db=REDIS_DB,
            decode_responses=True,
            socket_timeout=2,        # 【可用性】读写 2 秒超时:Redis 卡死不能拖死业务线程
            socket_connect_timeout=2,
            retry=_NO_RETRY,         # 单次尝试，不指数重试
            retry_on_timeout=False,
            health_check_interval=0,
        )
    return _sync_client


def get_redis_async() -> aioredis.Redis:
    """异步客户端(FastAPI 的 async 路由/依赖用)。"""
    global _async_client
    if _async_client is None:
        _async_client = aioredis.Redis(
            host=REDIS_HOST,
            port=REDIS_PORT,
            password=REDIS_PASSWORD,
            db=REDIS_DB,
            decode_responses=True,
            socket_timeout=2,
            socket_connect_timeout=2,
            retry=_NO_RETRY,
            retry_on_timeout=False,
            health_check_interval=0,
        )
    return _async_client


_AVAIL_LOCK = threading.Lock()
_LAST_FAIL_AT = 0.0
_FAIL_COOLDOWN = 15.0   # 失败后这段时间内直接判不可用，不再尝试连接


def redis_available(force: bool = False) -> bool:
    """Redis 是否可用（带失败退避）。

    一次失败后 15 秒内直接返回 False：Redis 挂掉时每个请求都去 connect 会
    累积成秒级等待（端口被丢包时 TCP 不会快速失败），整条请求会被拖死。
    调用方拿到 False 就走各自的降级分支（限流放行、缓存跳过、身份回源）。
    """
    global _LAST_FAIL_AT
    now = time.monotonic()
    with _AVAIL_LOCK:
        if not force and _LAST_FAIL_AT and (now - _LAST_FAIL_AT) < _FAIL_COOLDOWN:
            return False
    try:
        ok = bool(get_redis_sync().ping())
    except Exception:
        ok = False
    with _AVAIL_LOCK:
        _LAST_FAIL_AT = 0.0 if ok else now
    return ok
