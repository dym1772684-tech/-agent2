"""技能管理端点：让用户在前端自己选技能、设常开、给反馈。

【权限】只要求 agent:chat —— 技能是"用户自己怎么用智能体"的偏好，
不涉及他人数据，不需要更细的权限码。
真正决定"能用什么工具"的仍是 RBAC + 老人绑定，技能不能扩权。
"""
from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from core.acl import PERM_AGENT_CHAT, require_permission
from core.audit import write_audit
from core.responses import ApiResponse
from core.skill_service import list_with_state, record_feedback, set_enabled
from database.database import get_db

skill_router = APIRouter(prefix="/skills", tags=["技能"])


class FeedbackRequest(BaseModel):
    helpful: bool = Field(..., description="这次技能用得有没有用")


@skill_router.get("")
async def list_skills(
    current=Depends(require_permission(PERM_AGENT_CHAT)),
    db: AsyncSession = Depends(get_db),
):
    """技能库 + 我的启用状态（前端技能面板的数据源）。"""
    items = await list_with_state(db, str(current.user_id))
    return ApiResponse.success(data={"items": items, "total": len(items)})


@skill_router.post("/{skill_id}/enable")
async def enable_skill(
    skill_id: str,
    auto: Optional[bool] = Query(default=None, description="是否设为常开（每轮都注入）"),
    current=Depends(require_permission(PERM_AGENT_CHAT)),
    db: AsyncSession = Depends(get_db),
):
    try:
        data = await set_enabled(db, str(current.user_id), skill_id, True, auto)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    write_audit(str(current.user_id), "skill:enable",
                resource=f"skill:{skill_id}", decision=True,
                reason=f"auto={data['auto']}")
    return ApiResponse.success(data=data)


@skill_router.delete("/{skill_id}")
async def disable_skill(
    skill_id: str,
    current=Depends(require_permission(PERM_AGENT_CHAT)),
    db: AsyncSession = Depends(get_db),
):
    try:
        data = await set_enabled(db, str(current.user_id), skill_id, False, False)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    write_audit(str(current.user_id), "skill:disable",
                resource=f"skill:{skill_id}", decision=True)
    return ApiResponse.success(data=data)


@skill_router.post("/{skill_id}/feedback")
async def skill_feedback(
    skill_id: str,
    req: FeedbackRequest,
    current=Depends(require_permission(PERM_AGENT_CHAT)),
    db: AsyncSession = Depends(get_db),
):
    """技能效果反馈。用于后续自动推荐排序（成功率高的优先）。"""
    try:
        data = await record_feedback(db, str(current.user_id), skill_id, req.helpful)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    write_audit(str(current.user_id), "skill:feedback",
                resource=f"skill:{skill_id}", decision=True,
                reason=f"helpful={req.helpful}")
    return ApiResponse.success(data=data)
