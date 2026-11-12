from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from core.acl import (  # 【ACL】权限码 + 依赖工厂 + 对象级校验
    PERM_ELDER_DELETE,
    PERM_ELDER_READ,
    PERM_ELDER_WRITE,
    assert_elder_access,
    require_permission,
)
from core.cache import invalidate_elder_cache  # 【缓存刷新·写后删除】
from core.responses import ApiResponse
from database.database import get_db
from schemas.elder import ElderCreate, ElderUpdate
from service.elder_services import (
    create_elder,
    get_elder_by_id,
    update_elder,
    delete_elder
)

# 【ACL】原 router 级 dependencies=[Depends(check_jwt)] 拆到端点级:
# 每个资源端点声明自己的权限码;涉及具体老人数据行的,函数内再加对象级校验。
elder_router = APIRouter(prefix="/elder_identity", tags=["老人档案"])


@elder_router.post("/create_elder")
async def make_elder(
    req: ElderCreate,
    db: AsyncSession = Depends(get_db),
    current=Depends(require_permission(PERM_ELDER_WRITE)),  # 【ACL】写档案需 elder:write(nurse/admin)
):
    record = await create_elder(db, req)
    # 【缓存刷新】新建后清掉可能存在的"空值哨兵"(以前查过这个 id 缓存过"无记录")
    invalidate_elder_cache(req.elder_id)
    return ApiResponse.success(data=record)


@elder_router.get("/select_elder")
async def select_elder(
    elder_id: str,
    db: AsyncSession = Depends(get_db),
    current=Depends(require_permission(PERM_ELDER_READ)),  # 【ACL】读档案需 elder:profile_read
):
    # 【ACL】对象级:common/nurse 只能查自己绑定的老人(admin 豁免);未绑定 → 403
    await assert_elder_access(db, current, elder_id)
    elder_data = await get_elder_by_id(db, elder_id)
    return ApiResponse.success(data=elder_data)


@elder_router.post("/update_elder")
async def update_elder_(
    new_Elder: ElderCreate,
    db: AsyncSession = Depends(get_db),
    current=Depends(require_permission(PERM_ELDER_WRITE)),
):
    await assert_elder_access(db, current, new_Elder.elder_id)
    end_elder = await update_elder(db, new_Elder.elder_id, new_Elder)
    # 【缓存刷新·策略2 写后删除】先更新库(service 内已 commit),再删缓存;
    # 下次网关查询未命中 → 回源拿到新值。为什么不"更新缓存"?见 core/cache.py 头注释。
    invalidate_elder_cache(new_Elder.elder_id)
    return end_elder


@elder_router.delete("/delete_elder")
async def delete_elder_(
    elder: ElderCreate,
    db: AsyncSession = Depends(get_db),
    current=Depends(require_permission(PERM_ELDER_DELETE)),  # 【ACL】删除仅 admin(该权限码只授予 admin)
):
    await assert_elder_access(db, current, elder.elder_id)
    result = await delete_elder(db, elder.elder_id)
    # 【缓存刷新·策略2 写后删除】删除档案后清缓存,防止网关继续返回已删除老人的旧数据
    invalidate_elder_cache(elder.elder_id)
    return result
