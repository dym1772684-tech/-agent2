"""健康记录服务层。

【并发与锁策略】锁定原则一句话：**只给"要改的行"加锁，不给"不存在的行"加锁。**

| 操作 | 锁 | 原因 |
|---|---|---|
| 读（get / list） | 不加锁 | InnoDB MVCC 快照读已能保证一致性；加 FOR UPDATE 会阻塞写，白白降低并发 |
| 新增（create） | 不加行锁 | 对"尚不存在"的行 FOR UPDATE，RR 隔离下加的是**间隙锁**，两个并发新增会互相等锁 → 死锁。改为直接插入，让主键/唯一索引裁决 |
| 更新（update） | FOR UPDATE + populate_existing | 锁住目标行，保证"读到的就是要改的那版"，防止并发覆盖 |
| 删除（delete） | FOR UPDATE + populate_existing | 与 update 的加锁顺序一致，避免"并发删改"互相等锁 |

另外两条纪律：
1. 事务里只放 SQL，绝不放网络调用（MCP/LLM/HTTP），否则行锁会被长时间持有；
2. commit 成功之后不再做任何可能失败的操作（如 refresh），否则会把"已成功"报成 500。
"""

from datetime import datetime
import logging

from fastapi import HTTPException
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from models.health import HealthRecord
from schemas.health import HealthRecord_create

logger = logging.getLogger(__name__)


def _is_duplicate(exc: BaseException) -> bool:
    """识别唯一键/主键冲突（MySQL 1062）。"""
    orig = getattr(exc, "orig", None)
    args = getattr(orig, "args", ()) or ()
    return (args and args[0] == 1062) or "duplicate entry" in str(orig or exc).lower()


def record_to_dict(rec: HealthRecord) -> dict:
    """出参结构。字段名沿用 HealthRecord_create（主键 id 对外叫 health_id）。

    router 原先用 HealthRecord_create.model_validate(ORM 对象) 序列化，
    但 schema 要求 health_id 而表列叫 id，且默认不读属性 → 必抛 ValidationError。
    统一走这里，字段与原来 model_dump() 的结果保持一致。
    """
    return {
        "health_id": rec.id,
        "elder_id": rec.elder_id,
        "device_id": rec.device_id,
        "metric_type": rec.metric_type,
        "risk_content": rec.risk_content or {},
        "value": rec.value,
        "extra": rec.extra or {},
        "record_time": rec.record_time,
    }


def _as_int_id(record_id) -> int | None:
    """主键是 int；传进来非数字时直接当"不存在"，避免 MySQL 隐式转换报 1366。"""
    try:
        return int(record_id)
    except (TypeError, ValueError):
        return None


async def get_record(id: str, db: AsyncSession):
    """按主键查单条健康记录。只读，不加锁。"""
    rid = _as_int_id(id)
    if rid is None:
        return None
    stmt = select(HealthRecord).where(HealthRecord.id == rid)
    return (await db.execute(stmt)).scalar_one_or_none()


async def create_record(db: AsyncSession, obj_in: HealthRecord_create):
    """新增一条健康记录。

    并发正确性：不预查、不加锁，直接插入让主键裁决。
    两个请求同时插同一个 health_id 时，一个成功、另一个拿到 1062 → 409，
    既不会死锁，也不会写出重复数据。
    """
    # 只有显式给了主键才写 id；否则交给数据库自增（schema 已改为可选）
    kwargs = dict(
        elder_id=obj_in.elder_id,
        device_id=obj_in.device_id,
        metric_type=obj_in.metric_type,
        value=obj_in.value,
        risk_content=obj_in.risk_content,
        extra=obj_in.extra,
        record_time=obj_in.record_time,
        create_time=datetime.now(),
    )
    if obj_in.health_id is not None:
        kwargs["id"] = obj_in.health_id
    new_record = HealthRecord(**kwargs)
    db.add(new_record)
    try:
        await db.flush()          # 立刻发 SQL，主键冲突在这里暴露
    except IntegrityError as exc:
        await db.rollback()
        if _is_duplicate(exc):
            logger.warning("重复创建健康记录, health_id=%s", obj_in.health_id)
            raise HTTPException(
                status_code=409, detail=f"健康记录 {obj_in.health_id} 已存在"
            ) from exc
        logger.exception("创建健康记录约束冲突, health_id=%s", obj_in.health_id)
        raise HTTPException(status_code=500, detail="创建健康记录失败") from exc
    except Exception as exc:
        await db.rollback()
        logger.exception("创建健康记录失败, health_id=%s", obj_in.health_id)
        raise HTTPException(status_code=500, detail="创建健康记录失败") from exc

    # flush 后主键与默认值已回填，先在 commit 前把返回体算好：
    # commit 会让对象过期，之后再读属性会触发隐式 IO（async 下 MissingGreenlet），
    # 而且 commit 成功后的失败不该再报给客户端。
    data = record_to_dict(new_record)
    try:
        await db.commit()
    except Exception as exc:
        await db.rollback()
        logger.exception("创建健康记录提交失败, health_id=%s", obj_in.health_id)
        raise HTTPException(status_code=500, detail="创建健康记录失败") from exc
    return data


async def update_record(db: AsyncSession, obj_in: HealthRecord_create):
    """按主键更新单条记录（原先按 elder_id 查，一个老人多条记录时必抛 MultipleResultsFound）。

    锁：FOR UPDATE 拦住并发覆盖；populate_existing 强制用数据库返回值刷新对象，
    否则命中 identity map 时会拿到会话里的旧值（锁加上了、数据却是旧的）。
    """
    rid = _as_int_id(obj_in.health_id)
    if rid is None:
        raise HTTPException(status_code=404, detail="记录不存在")

    stmt = (
        select(HealthRecord)
        .where(HealthRecord.id == rid)
        .with_for_update()
        .execution_options(populate_existing=True)
    )
    rec = (await db.execute(stmt)).scalar_one_or_none()
    if rec is None:
        raise HTTPException(status_code=404, detail="记录不存在")

    if rec.risk_content == obj_in.risk_content:
        # 无变化：不写库。行锁随请求结束（session close）释放，窗口只有本次请求剩余时间
        return record_to_dict(rec)

    rec.risk_content = obj_in.risk_content
    data = record_to_dict(rec)
    try:
        await db.commit()
    except Exception as exc:
        await db.rollback()
        logger.exception("更新健康记录失败, health_id=%s", obj_in.health_id)
        raise HTTPException(status_code=500, detail="更新健康记录失败") from exc
    return data


async def delete_record(db: AsyncSession, obj_in: HealthRecord_create):
    """按主键删除单条记录。

    锁：与 update 保持同一顺序（先 FOR UPDATE 定位行），
    避免"一个在改、一个在删"时互相等锁。
    """
    rid = _as_int_id(obj_in.health_id)
    if rid is None:
        raise HTTPException(status_code=404, detail="记录不存在")

    stmt = (
        select(HealthRecord)
        .where(HealthRecord.id == rid)
        .with_for_update()
        .execution_options(populate_existing=True)
    )
    rec = (await db.execute(stmt)).scalar_one_or_none()
    if rec is None:
        raise HTTPException(status_code=404, detail="记录不存在")

    try:
        await db.delete(rec)
        await db.commit()
    except Exception as exc:
        await db.rollback()
        logger.exception("删除健康记录失败, health_id=%s", obj_in.health_id)
        raise HTTPException(status_code=500, detail="删除健康记录失败") from exc
    return {"msg": "删除成功"}
