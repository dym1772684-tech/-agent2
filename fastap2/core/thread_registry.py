"""会话线程属主登记与校验（ACL 层的资源归属）。

================================================================
为什么从 routers/agent.py 抽到 core
----------------------------------------------------------------
1. 属主校验是「资源归属」问题，和 Agent 编排没有关系，放在一起只会让路由越来越厚；
2. routers/agent.py 一导入就会经 agent_bridge 触发图装配（加载模型/MCP），
   任何逻辑留在那儿都无法被轻量单测覆盖。抽到这里之后，本模块只依赖 orm 与
   sqlalchemy，可以零依赖跑测试。

================================================================
并发下的正确做法（幂等第 4 层）
----------------------------------------------------------------
「先 select 判断不存在，再 insert」是典型的 check-then-act 竞态：
客户端重试 / 用户双击时，两个请求同时 select 都 miss，然后都 insert，
后到者撞 unique 索引 → IntegrityError 冒成 500。

这里改成**直接 insert，让数据库唯一索引裁决**：
  - 插入成功 → 我就是第一个，正常续接；
  - 撞 IntegrityError → rollback 后**复查**：能读到行说明别人刚登记成功，
    按「已登记」路径继续（属主是自己就放行，是别人就 403）。
这样「同一 thread 首次并发」从偶发 500 变成确定性行为。
"""
from __future__ import annotations

import uuid
from typing import Optional

from fastapi import HTTPException
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from models.acl import ConversationOwner


def new_thread_id() -> str:
    """服务端生成线程 ID（不用 elder_id —— 那是可枚举的劫持入口）。"""
    return f"t-{uuid.uuid4().hex[:12]}"


async def owner_row(db: AsyncSession, thread_id: str):
    """查属主行，返回标量（None = 未登记）。失败分支要复查，所以单独暴露。"""
    result = await db.execute(
        select(ConversationOwner).where(ConversationOwner.thread_id == thread_id)
    )
    return result.scalar_one_or_none()


async def resolve_thread_id(
    db: AsyncSession, user_id: str, thread_id: Optional[str]
) -> str:
    """确定本次使用的会话线程：已登记则校验属主，未登记则登记到当前用户。"""
    if not thread_id:
        tid = new_thread_id()
        db.add(ConversationOwner(thread_id=tid, user_id=user_id))
        try:
            await db.commit()
        except IntegrityError:
            # uuid4 撞车概率可忽略，但也不能冒 500
            await db.rollback()
            raise HTTPException(status_code=409, detail="会话线程登记冲突，请重试")
        return tid

    row = await owner_row(db, thread_id)
    if row is None:
        db.add(ConversationOwner(thread_id=thread_id, user_id=user_id))
        try:
            await db.commit()
        except IntegrityError:
            # 并发对手先插进去了：回滚后复查，按「已登记」继续
            await db.rollback()
            row = await owner_row(db, thread_id)
            if row is None:
                raise HTTPException(status_code=409, detail="会话线程登记冲突，请重试")
        else:
            return thread_id

    if row.user_id != user_id:
        raise HTTPException(status_code=403, detail="无权使用该会话线程")
    return thread_id


async def assert_thread_owner(db: AsyncSession, user_id: str, thread_id: str) -> None:
    """interrupt / resume 用：不存在 404，不是本人 403（防会话劫持）。"""
    row = await owner_row(db, thread_id)
    if row is None:
        raise HTTPException(status_code=404, detail="会话不存在")
    if row.user_id != user_id:
        raise HTTPException(status_code=403, detail="无权操作该会话线程")
