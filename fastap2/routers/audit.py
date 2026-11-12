"""审计日志查询端点(管理员监控台·接口②)。

与用户接口严格分离:普通用户的权限集里没有 audit:read,
调本端点一律 403。支持分页与"只看有害访问(被拒)"过滤,
供前端管理面板渲染红色告警。
"""
from fastapi import APIRouter, Depends, Query
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from core.acl import PERM_AUDIT_READ, require_permission
from core.audit import write_audit
from database.database import get_db
from models.audit import AuditLog

audit_router = APIRouter(prefix="/audit", tags=["审计日志(管理员)"])


@audit_router.get("/logs")
async def list_logs(
    only_denied: bool = Query(default=False, description="true=只看被拒的有害访问"),
    page: int = Query(default=1, ge=1),
    size: int = Query(default=20, ge=1, le=100),
    current=Depends(require_permission(PERM_AUDIT_READ)),
    db: AsyncSession = Depends(get_db),
):
    stmt = select(AuditLog)
    if only_denied:
        stmt = stmt.where(AuditLog.decision == 0)

    total = (await db.execute(
        select(func.count()).select_from(stmt.subquery())
    )).scalar_one()
    rows = (await db.execute(
        stmt.order_by(AuditLog.id.desc()).offset((page - 1) * size).limit(size)
    )).scalars().all()

    # 管理员查看审计这件事本身也留痕(审计 viewer 也要被审计)
    write_audit(str(current.user_id), "audit:view", resource=f"page={page},denied={only_denied}", decision=True)

    return {
        "total": total,
        "page": page,
        "size": size,
        "list": [
            {
                "id": r.id,
                "user_id": r.user_id,
                "action": r.action,          # guard:sensitive / rate_limited:* / gateway:* / acl:*
                "resource": r.resource,
                "decision": r.decision,      # 1=允许 0=拒绝(有害访问)
                "reason": r.reason,
                "create_time": str(r.create_time),
            }
            for r in rows
        ],
    }
