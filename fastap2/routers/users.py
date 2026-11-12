"""用户注册/登录路由。

【限流改造】本文件挂了三道 Redis 限流(实现见 core/rate_limit.py,算法注释在那边):
1. /register:同 IP 每小时 3 次(固定窗口,防批量注册)
2. /login,/token:同 IP 每分钟 10 次(滑动窗口,防脚本撞库)
3. /login,/token:同账号密码错 5 次 → 锁 10 分钟(账号锁定,防定向爆破)
登录成功会清空失败计数(策略见 clear_login_fail)。
"""
import uuid

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from core.rate_limit import (
    RL_KEY_LOGIN_IP,
    RL_KEY_REGISTER,
    check_login_locked,
    clear_login_fail,
    fixed_window_allow,
    record_login_fail,
    sliding_window_allow,
)
from core.responses import ApiResponse
from core.settings import check_password, create_access_token, hash_password
from database.database import get_db
from models.acl import UserElderBinding
from models.users import User_table
from schemas.user import User

user_router = APIRouter()


def _client_ip(request: Request) -> str:
    """取客户端 IP。

    【注意】request.client.host 是直连 IP;将来如果前面加了 Nginx 反代,
    要改成读 X-Forwarded-For(且只信任自家反代设置的值,否则可被伪造)。
    """
    return request.client.host if request.client else "unknown"


@user_router.post("/register")
async def create_User(
    users: User,
    request: Request,
    db: AsyncSession = Depends(get_db),
):
    # 【限流】固定窗口:同 IP 每小时 3 次注册
    ok, _ = await fixed_window_allow(RL_KEY_REGISTER.format(ip=_client_ip(request)), 3, 3600)
    if not ok:
        raise HTTPException(status_code=429, detail="注册过于频繁,请稍后再试")

    existing = await db.execute(select(User_table).where(User_table.username == users.username))
    if existing.scalar_one_or_none() is not None:
        raise HTTPException(status_code=400, detail="用户名已存在")

    end_password = hash_password(users)
    users.password = end_password
    new_user = User_table(
        username=users.username,
        password=users.password,
        user_id=str(uuid.uuid4()),
        user_role="common",  # 【安全/ACL】角色服务端写死 common;原 users.role 允许注册传 admin 直接提权
    )
    db.add(new_user)
    try:
        await db.commit()
    except IntegrityError:
        # 【并发】user 表现在有 uq_user_username 唯一索引：两个请求同时通过上面的
        # 预检查时，由数据库裁决，撞索引的一方返回 400 而不是 500。
        await db.rollback()
        raise HTTPException(status_code=400, detail="用户名已存在")
    await db.refresh(new_user)
    return ApiResponse.success(msg="成功创建该用户")


# 【鉴权】双登录端点,避免 422:
# - POST /login:JSON body({"username","password"})—— 你现有客户端的调用方式
# - POST /token:标准表单(OAuth2PasswordRequestForm)—— 供 /docs 右上角 Authorize 使用
# 两者共用 _login_core(限流+锁定+校验+发 token 逻辑只有一份)。

async def _login_core(users: User, db: AsyncSession, request: Request) -> dict:
    # 【限流】滑动窗口:同 IP 每分钟 10 次登录尝试
    ok, _ = await sliding_window_allow(
        RL_KEY_LOGIN_IP.format(ip=_client_ip(request)), 10, 60
    )
    if not ok:
        raise HTTPException(status_code=429, detail="登录尝试过于频繁,请稍后再试")

    # 【限流】账号锁定检查:错 5 次后的冷却期内直接拒绝(不再碰密码校验)
    if await check_login_locked(users.username):
        raise HTTPException(status_code=429, detail="账号已锁定(密码错误次数过多),请 10 分钟后再试")

    try:
        user = await check_password(users, db)  # 内部已做 bcrypt 校验,失败抛 401
    except HTTPException:
        # 【限流】密码错误 → 失败计数 +1;达到 5 次自动上锁(record 内部处理)
        fails = await record_login_fail(users.username)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"账号或密码错误(连续失败 {fails} 次,5 次后将锁定 10 分钟)",
        )

    await clear_login_fail(users.username)  # 成功登录,清空失败计数

    # 【零信任·第①步】账户实时状态检查：disabled/locked 的账户密码对了也不放行
    if str(getattr(user, "status", "active") or "active").lower() != "active":
        from core.audit import write_audit

        write_audit(str(user.user_id), "auth:login_blocked",
                    resource=f"user:{user.user_id}", decision=False,
                    reason=f"status={user.status}")
        raise HTTPException(status_code=403, detail="账号已被禁用或锁定,请联系管理员")

    from core import identity as idm

    token = create_access_token({
        "user_id": str(user.user_id),
        "role": str(user.user_role),
        # 【零信任】会话版本:logout/降权/锁定都会 bump perm_version,
        # check_jwt 校验 pv 不符 → 旧 token 立即 401,强制重新登录
        "pv": idm.get_perm_version(str(user.user_id)),
    })

    # 【零信任·第③步】登录即预装配一份可信身份（RBAC+绑定关系+工具白名单），
    # 存 Redis 供 agent 链路复用；客户端拿到 identity_id 后续请求可携带（省一次重建）
    bound = (await db.execute(
        select(UserElderBinding.elder_id).where(UserElderBinding.user_id == str(user.user_id))
    )).scalars().all()
    ctx = idm.build_identity(str(user.user_id), str(user.user_role),
                             session_id="", bound_elder_ids=list(bound))
    idm.put_identity(ctx)
    return {
        "access_token": token, "token_type": "bearer",
        "identity_id": ctx.identity_id,
        "identity_expires_at": ctx.expires_at,
    }


@user_router.post("/login")
async def login_json(
    users: User,  # JSON body:{"username": "...", "password": "..."}
    request: Request,
    db: AsyncSession = Depends(get_db),
):
    """【鉴权】JSON 登录。请求体为 application/json:{"username","password"}"""
    return await _login_core(users, db, request)


@user_router.post("/token")
async def login_form(
    form: OAuth2PasswordRequestForm = Depends(),  # 表单登录:/docs 右上角 Authorize 直接可用
    request: Request = None,
    db: AsyncSession = Depends(get_db),
):
    """【鉴权】标准表单登录。请求体为 x-www-form-urlencoded:username=..&password=.."""
    user_in = User(username=form.username, password=form.password)
    return await _login_core(user_in, db, request)


@user_router.post("/logout")
async def logout(
    request: Request,
    db: AsyncSession = Depends(get_db),
    token: str = None,
):
    """【零信任】登出 = 撤销该用户全部在线身份（perm_version+1，即时失效）。

    从 Authorization 头（或 ?token=）取 JWT 解出 user_id；即便 token 已过期，
    撤销也是幂等安全的。身份作废后，agent 链路内的 Redis 身份缓存全部失效。
    """
    from core.audit import write_audit
    from core.settings import check_jwt

    user_id = "anonymous"
    try:
        auth = request.headers.get("Authorization", "")
        raw = token or (auth.split(" ", 1)[1] if auth.startswith("Bearer ") else None)
        if raw:
            payload = await check_jwt(raw, db)
            user_id = str(getattr(payload, "user_id", "") or "anonymous")
    except Exception:
        pass  # token 已过期/无效：撤销动作本身仍执行（幂等）

    from core import identity as idm

    ver = idm.revoke_user(user_id)
    write_audit(user_id, "auth:logout", resource=f"user:{user_id}",
                decision=True, reason=f"perm_version={ver}")
    return ApiResponse.success(data={"revoked": True, "user_id": user_id})
