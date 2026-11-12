"""ACL 访问控制核心(两层模型)。

第一层 角色→权限码(RBAC):每个资源端点声明所需权限码(如 elder:profile_read),
require_permission(...) 在依赖里校验"当前登录角色的权限集是否包含它";
未登记角色按空集处理(fail-closed:默认拒绝)。

第二层 对象级(资源行 ACL):老人/健康数据的具体数据行,由
user_elder_binding 表决定"哪个用户能访问哪个老人";admin 角色豁免行级校验。

新增资源的接入方式:
1. 定义权限码 → 2. 挂进对应角色的 ROLE_PERMISSIONS →
3. 端点参数 Depends(require_permission(PERM_XXX));
4. 涉及具体老人数据行的,函数体内再加 assert_elder_access(db, current, elder_id)。
"""
from fastapi import Depends, HTTPException, status
from fastapi.concurrency import run_in_threadpool
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from core.identity import PERM_REGISTRY, ROLE_DEFAULT_PERMS, load_role_permissions
from core.settings import CurrentUser
from models.acl import UserElderBinding

# ============================================================
# 权限码定义(资源:动作)
# ------------------------------------------------------------
# 【唯一词表】权限码与角色映射的**唯一来源**是 core/identity.py 的
# PERM_REGISTRY / ROLE_DEFAULT_PERMS（以及 role_permission 表）。
# 这里只做"取别名"，绝不自己发明新码——之前 acl.py 用 elder:read、
# identity.py 用 elder:profile_read，两套词表互不相认：
#   · 改 role_permission 表对路由鉴权完全无效；
#   · identity 里的 family/volunteer 在 acl.py 没有条目 → 这两个角色访问任何端点 403。
# ============================================================
PERM_ELDER_READ = "elder:profile_read"
PERM_ELDER_WRITE = "elder:write"
PERM_ELDER_DELETE = "elder:delete"
PERM_HEALTH_READ = "elder:health_read"
PERM_HEALTH_WRITE = "elder:health_write"
PERM_AGENT_CHAT = "agent:chat"
PERM_ACL_MANAGE = "system:bind"     # 绑定/解绑老人 = identity 的 system:bind
PERM_SYSTEM_DIAG = "system:diag"
PERM_AUDIT_READ = "audit:read"

# 防呆：常量写错（码不在注册表里）时启动即报，而不是线上悄悄 403
_UNDECLARED = [c for c in (PERM_ELDER_READ, PERM_ELDER_WRITE, PERM_ELDER_DELETE,
                           PERM_HEALTH_READ, PERM_HEALTH_WRITE, PERM_AGENT_CHAT,
                           PERM_ACL_MANAGE, PERM_SYSTEM_DIAG, PERM_AUDIT_READ)
               if c not in PERM_REGISTRY]
if _UNDECLARED:
    raise RuntimeError(f"acl.py 引用了未在 identity.PERM_REGISTRY 登记的权限码: {_UNDECLARED}")


# 角色 → 权限集（回退表镜像；真实校验走 _role_perms_async()，DB 优先）
ROLE_PERMISSIONS = {role: set(perms) for role, perms in ROLE_DEFAULT_PERMS.items()}

# 权限集缓存：避免每个请求都打一次 role_permission 表（60 秒内复用）
_PERM_TTL = 60.0
_perm_cache: dict[str, tuple[float, set]] = {}


async def _role_perms_async(role: str) -> set:
    """取角色权限集：DB(role_permission 表) 优先 → 缺失回退硬编码，带 60s 缓存。"""
    import time

    now = time.monotonic()
    hit = _perm_cache.get(role)
    if hit is not None and now - hit[0] < _PERM_TTL:
        return hit[1]
    perms = await run_in_threadpool(load_role_permissions, role)  # 同步 pymysql → 线程池
    _perm_cache[role] = (now, perms)
    return perms


def require_permission(*perms: str):
    """端点级 ACL 依赖工厂。

    用法:current = Depends(require_permission(PERM_ELDER_READ))
    流程:先经 check_jwt 登录校验(401),再比对权限码;缺任一 → 403。
    校验通过返回当前登录用户,后续对象级校验可直接复用。
    """
    async def checker(current: CurrentUser):
        owned = await _role_perms_async(current.user_role)
        missing = [p for p in perms if p not in owned]
        if missing:
            # 【审计】端点级拒绝留痕(谁、缺什么权限);审计失败不影响 403 的返回
            try:
                from core.audit import write_audit
                write_audit(str(current.user_id), "acl:check",
                            resource=",".join(perms), decision=False,
                            reason=f"缺少权限码: {','.join(missing)}")
            except Exception:
                pass
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"权限不足:缺少 {', '.join(missing)}",
            )
        return current

    return checker


async def assert_elder_access(db: AsyncSession, current, elder_id: str) -> None:
    """对象级 ACL:校验当前用户可访问该老人数据行。

    admin 全量放行;其余角色必须有 user_elder_binding 绑定记录,
    否则 403。未绑定与不存在统一按 403 处理,不向调用方泄露资源存在性。
    """
    if current.user_role == "admin":
        return
    stmt = select(UserElderBinding).where(
        UserElderBinding.user_id == str(current.user_id),
        UserElderBinding.elder_id == elder_id,
    )
    row = (await db.execute(stmt)).scalar_one_or_none()
    if row is None:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"无权访问老人 {elder_id} 的数据(未绑定)",
        )
