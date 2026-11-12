from fastapi.security import OAuth2PasswordBearer
from schemas.user import User
"""
1. 护工登录后台 → POST `/token`，账号密码校验，生成 JWT 返回
2. 护工调用接口：查询老人档案、查看健康风险记录
3. 请求头带上 Bearer token
4. `OAuth2PasswordBearer` 提取 token 字符串
5. `jwt.decode` 校验签名 + 判断是否过期，取出 user_id、角色
6. 代码判断：该护工是否有权限访问这个老人的数据
7. token 过期 → 前端提示重新登录"""
from typing import Annotated
from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from sqlalchemy.ext.asyncio import AsyncSession
from pydantic import BaseModel
from fastapi import APIRouter
from sqlalchemy.orm import Session
from sqlalchemy import select
from database.database import get_db
from passlib.context import CryptContext
from models.users import User_table
from datetime import datetime,timedelta,timezone##【优化】引入 timezone：JWT 的 exp 按规范应使用 UTC 时间
import jwt
import os
import bcrypt
from types import SimpleNamespace
from dotenv import find_dotenv, load_dotenv
from pathlib import Path

from core.cache import cache_get_json_async, cache_set_json_async, user_cache_key
BASE_DIR = Path(__file__).resolve().parent.parent
ENV_PATH = BASE_DIR / "core" / "key.env"
load_dotenv(find_dotenv(str(ENV_PATH)))

SECRET_KEY = os.getenv("SECRET_KEY")
ALGORITHM = os.getenv("ALGORITHM", "HS256")
if not SECRET_KEY:
    raise RuntimeError("JWT SECRET_KEY missing. Please confirm fastap2/core/key.env exists and contains SECRET_KEY.")

oauth2_schema = OAuth2PasswordBearer(tokenUrl="/api/v1/token")  # 【鉴权】表单登录端点（/docs Authorize 使用）


def _normalize_bcrypt_password(password: str | bytes) -> bytes:
    """bcrypt 只接受最多 72 bytes 的密码；按 UTF-8 字节长度截断。"""
    if isinstance(password, str):
        raw = password.encode("utf-8")
    else:
        raw = bytes(password)
    return raw[:72]


def hash_password(users: User):
    thing = _normalize_bcrypt_password(users.password)
    return bcrypt.hashpw(thing, bcrypt.gensalt(rounds=12)).decode("utf-8")


def decode_jwt(token):
    payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
    return payload


def create_access_token(data: dict, expires_delta: timedelta | None = None):
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + (expires_delta or timedelta(minutes=30))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


async def check_password(users: User, db: AsyncSession):
    result = await db.execute(
        select(User_table)
        .where(User_table.username == users.username)
        .order_by(User_table.id.asc())
    )
    end_result = result.scalars().first()
    if end_result is None:
        raise HTTPException(status_code=401, detail="账号或密码错误")

    user_input = _normalize_bcrypt_password(users.password)
    try:
        is_ok = bcrypt.checkpw(user_input, end_result.password.encode("utf-8"))
    except ValueError:
        raise HTTPException(status_code=401, detail="账号或密码错误")
    if not is_ok:
        raise HTTPException(status_code=401, detail="账号或密码错误")
    return end_result


async def check_jwt(token: str = Depends(oauth2_schema), db: AsyncSession = Depends(get_db)):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
    except Exception:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="token 无效或已过期")

    user_id = payload.get("user_id")
    role = payload.get("role")
    if not user_id or not role:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="token 缺少用户信息")

    # 【零信任】会话版本校验:logout/降权/锁定都会 bump perm_version。
    # 带 pv 的 token 版本不符 → 立即 401(强制重新登录);
    # 老 token(无 pv 字段)兼容放行,自然过期淘汰。
    # 注意:必须放在用户缓存之前——缓存命中直接 return 会绕过一切校验。
    pv = payload.get("pv")
    if pv is not None:
        from fastapi.concurrency import run_in_threadpool

        from core.identity import get_perm_version

        # 【性能】get_perm_version 是同步 Redis 读：每个请求都走这条路径，
        # 直接在 async 里调会在 Redis 抖动时卡住整个事件循环。
        if int(pv) != await run_in_threadpool(get_perm_version, user_id):
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED,
                                 detail="会话已失效,请重新登录")

    # 【缓存·Cache-Aside 读路径】JWT 每个请求都要校验,不能每次都打数据库:
    # 1) 先查缓存(命中=免查库,直接用上次鉴权结果)
    cached = await cache_get_json_async(user_cache_key(user_id))
    if cached is not None:
        return SimpleNamespace(**cached)

    # 2) 未命中 → 回源查库做完整校验(存在性 + 角色一致性)
    result = await db.execute(select(User_table).where(User_table.user_id == user_id))
    user = result.scalar_one_or_none()
    if user is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="用户不存在")
    if user.user_role != role:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="角色不匹配")

    # 3) 校验通过 → 写回缓存(TTL 60s):角色/禁用状态变更最迟 1 分钟生效;
    #    管理端改角色后想立即生效,调 invalidate_user_cache(user_id) 主动失效
    user_data = {
        "id": user.id,
        "username": user.username,
        "user_id": str(user.user_id),
        "user_role": user.user_role,
    }
    await cache_set_json_async(user_cache_key(user_id), user_data, ttl=60)
    # 统一返回轻量对象(不返回 ORM 实例,避免缓存路径/回源路径类型不一致)
    return SimpleNamespace(**user_data)
    #########差一个客户端报文jwt解析内容

# ============================================================
# 【鉴权】新增：当前登录用户依赖 + RBAC 角色控制依赖
# ============================================================

# 任意受保护接口把参数写成 current: CurrentUser，
# FastAPI 就会自动执行 check_jwt（验签+查库）并注入当前登录用户对象。
CurrentUser = Annotated[SimpleNamespace, Depends(check_jwt)]


def require_roles(*allowed_roles: str):
    """【鉴权】RBAC 角色控制依赖工厂。

    用法（挂到单个接口或整个 router）：
        Depends(require_roles("admin"))
        Depends(require_roles("nurse", "admin"))

    流程：先经 check_jwt 确认已登录（否则 401），再校验角色；
    角色不在允许名单内 → 403 权限不足。
    """
    async def checker(current: User_table = Depends(check_jwt)):
        if current.user_role not in allowed_roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="权限不足",
            )
        return current

    return checker
