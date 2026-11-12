"""ACL 管理端点:维护 用户↔老人 绑定关系(仅 admin,权限码 system:bind)。"""
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from core.acl import PERM_ACL_MANAGE, require_permission
from core.audit import write_audit  # 【审计】授权变更必须留痕(谁给谁授/收了什么权)
from database.database import get_db
from models.acl import UserElderBinding
from models.users import User_table

acl_router = APIRouter(prefix="/acl", tags=["ACL 授权管理"])


class BindRequest(BaseModel):
    user_id: str
    elder_id: str
    relation: str = "caregiver"


@acl_router.post("/bind")
async def bind(
    req: BindRequest,
    current=Depends(require_permission(PERM_ACL_MANAGE)),
    db: AsyncSession = Depends(get_db),
):
    """授予:把某老人绑定到某用户(重复绑定幂等返回)。"""
    existing = (await db.execute(
        select(UserElderBinding).where(
            UserElderBinding.user_id == req.user_id,
            UserElderBinding.elder_id == req.elder_id,
        )
    )).scalar_one_or_none()
    if existing is not None:
        return {"msg": "绑定已存在"}
    db.add(UserElderBinding(user_id=req.user_id, elder_id=req.elder_id, relation=req.relation))
    await db.commit()
    write_audit(str(current.user_id), "acl:bind",              # 【审计】授权变更留痕
                resource=f"user:{req.user_id}->elder:{req.elder_id}", decision=True)
    return {"msg": "绑定成功"}


@acl_router.delete("/bind")
async def unbind(
    req: BindRequest,
    current=Depends(require_permission(PERM_ACL_MANAGE)),
    db: AsyncSession = Depends(get_db),
):
    """回收:解除 用户↔老人 绑定,该用户立即失去对该老人的访问权。"""
    row = (await db.execute(
        select(UserElderBinding).where(
            UserElderBinding.user_id == req.user_id,
            UserElderBinding.elder_id == req.elder_id,
        )
    )).scalar_one_or_none()
    if row is None:
        raise HTTPException(status_code=404, detail="绑定不存在")
    await db.delete(row)
    await db.commit()
    write_audit(str(current.user_id), "acl:unbind",            # 【审计】收权留痕
                resource=f"user:{req.user_id}->elder:{req.elder_id}", decision=True)
    return {"msg": "解绑成功"}


@acl_router.get("/bindings")
async def list_bindings(
    user_id: str,
    current=Depends(require_permission(PERM_ACL_MANAGE)),
    db: AsyncSession = Depends(get_db),
):
    """查看某用户当前被授权的全部老人。"""
    rows = (await db.execute(
        select(UserElderBinding).where(UserElderBinding.user_id == user_id)
    )).scalars().all()
    return [{"elder_id": r.elder_id, "relation": r.relation} for r in rows]


@acl_router.get("/user_id")
async def lookup_user_id(
    username: str,
    current=Depends(require_permission(PERM_ACL_MANAGE)),
    db: AsyncSession = Depends(get_db),
):
    """管理辅助:按用户名查 user_id(绑定接口用的是 user_id)。"""
    row = (await db.execute(
        select(User_table).where(User_table.username == username)
    )).scalar_one_or_none()
    if row is None:
        raise HTTPException(status_code=404, detail="用户不存在")
    return {"username": row.username, "user_id": row.user_id, "role": row.user_role}
