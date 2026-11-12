"""显式事务边界：把 commit 从 service 收上来，交给"声明业务动作"的那一层。

================================================================
为什么要有这个模块
----------------------------------------------------------------
原实现里 get_db() 只 close 不 commit/rollback，事务边界由每个 service
函数自己 `await db.commit()` 决定 —— 粒度等于函数，多步业务被切成多段，
中间失败会留下半成品。

本模块提供两个原语：

    async with transaction(db):      # 业务事务：正常退 commit，异常 rollback
        db.add(...)
        await db.flush()             # 让约束冲突在这里暴露

    async with savepoint(db):        # 嵌套：只回滚到保存点，外层继续
        ...

约定（改代码前必读）：
1. service 层只操作 session，**一个 commit 都不写**；
2. 事务边界由 router / 用例层声明，一个业务不变量 → 一个 transaction()；
3. 事务里禁止：调 LLM、调 MCP、等 Redis、发 SSE 帧（长事务会锁行 + 占连接）；
4. 事务提交之后再删缓存、写审计、做跨存储投影。

================================================================
嵌套语义
----------------------------------------------------------------
已经处于事务中（db.in_transaction() 为 True）时，内层直接用 SAVEPOINT：
内层异常只回滚到保存点，不会把外层已写的数据一起丢掉；内层也不会提前
commit —— 外层失败时整体仍然回滚。这正是"子操作失败但主流程可继续"的
场景需要的语义（比如批量导入时单条冲突跳过）。
"""
from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncIterator

from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

# MySQL 唯一键冲突的错误码
MYSQL_ERR_DUP_ENTRY = 1062


@asynccontextmanager
async def transaction(db: AsyncSession) -> AsyncIterator[AsyncSession]:
    """业务事务边界。

    正常退出 → commit；抛任何异常（含 CancelledError）→ rollback 后原样抛出。
    已在事务中 → 降级为 SAVEPOINT，不提前提交。
    """
    if db.in_transaction():
        async with db.begin_nested():
            yield db
        return

    try:
        yield db
        await db.commit()
    except BaseException:
        # 用 BaseException 而不是 Exception：请求被取消时也要把事务关干净，
        # 否则这条连接回到池里会带着未结束的事务。
        await db.rollback()
        raise


@asynccontextmanager
async def savepoint(db: AsyncSession) -> AsyncIterator[AsyncSession]:
    """显式保存点：内层失败只回滚到这里，外层可以继续（批量导入单条跳过）。"""
    async with db.begin_nested():
        yield db


def is_duplicate(exc: BaseException) -> bool:
    """是否为唯一键冲突（MySQL 1062 / SQLite UNIQUE constraint）。"""
    orig = getattr(exc, "orig", None)
    args = getattr(orig, "args", ()) or ()
    if args and args[0] == MYSQL_ERR_DUP_ENTRY:
        return True
    text = f"{orig or exc}".lower()
    return "duplicate entry" in text or "unique constraint" in text


async def insert_or_skip(db: AsyncSession, obj) -> bool:
    """插入一条；唯一键冲突返回 False。

    用 SAVEPOINT 隔离冲突：撞键只回滚这一条，session 仍然可用，
    调用方不需要写 try/except，也不会因为一次冲突把整个事务搞脏。
    """
    try:
        async with db.begin_nested():
            db.add(obj)
            await db.flush()
        return True
    except IntegrityError as exc:
        if is_duplicate(exc):
            return False
        raise
