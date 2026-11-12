from sqlalchemy import select, func
from sqlalchemy.exc import IntegrityError
from models.elder import Elder
from schemas.elder import ElderCreate
from fastapi import HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
import logging
#一锁、二判、三更新、四释放
logger = logging.getLogger(__name__)


def _is_duplicate(exc: BaseException) -> bool:
    orig = getattr(exc, "orig", None)
    args = getattr(orig, "args", ()) or ()
    return (args and args[0] == 1062) or "duplicate entry" in str(orig or exc).lower()


def _elder_to_dict(elder: Elder) -> dict:
    return {
        "elder_id": elder.elder_id,
        "name": elder.name,
        "age": elder.age,
        "gender": elder.gender,
        "health_tag": elder.health_tag,
        "chronic_disease": elder.chronic_disease,
        "phone": elder.phone,
        "create_time": elder.create_time.isoformat() if elder.create_time else None,
    }


async def create_elder(db: AsyncSession, req: ElderCreate):
    new_elder = Elder(
        elder_id=req.elder_id,
        name=req.name,
        age=req.age,
        gender=req.gender,
        health_tag=req.health_tag,
        chronic_disease=req.chronic_disease,
        phone=req.phone
    )
    db.add(new_elder)
    try:
        await db.flush()
    except IntegrityError as exc:
        await db.rollback()
        if _is_duplicate(exc):
            logger.warning("重复创建老人，elder_id=%s", req.elder_id)
            raise HTTPException(status_code=409, detail=f"老人编号 {req.elder_id} 已存在") from exc
        logger.exception("创建老人 IntegrityError, elder_id=%s", req.elder_id)
        raise HTTPException(status_code=500, detail="创建老人失败") from exc
    except Exception as exc:
        await db.rollback()
        logger.exception("创建老人异常, elder_id=%s", req.elder_id)
        raise HTTPException(status_code=500, detail="创建老人失败") from exc

    # 【为什么先构造返回体再 commit】
    # flush 成功后主键与 Python 端默认值(create_time)已回填，此时对象未过期；
    # 而 commit 默认 expire_on_commit=True 会让所有属性过期，之后再读属性触发
    # 隐式 IO → async 下报 MissingGreenlet，所以原来必须 refresh（多一次 SELECT）。
    # 在 commit 前算好返回体，既省掉这次往返，也避免"commit 已成功、refresh 失败"
    # 被下面同一个 except 捕获 → rollback（无效）→ 对客户端谎报 500 但数据已落库。
    data = _elder_to_dict(new_elder)
    try:
        await db.commit()
    except Exception as exc:
        await db.rollback()
        logger.exception("创建老人提交失败, elder_id=%s", req.elder_id)
        raise HTTPException(status_code=500, detail="创建老人失败") from exc
    return data


async def get_elder_by_id(db: AsyncSession, elder_id: str):
    stmt = select(Elder).where(Elder.elder_id == elder_id)
    orm_result = await db.execute(stmt)
    result = orm_result.scalar_one_or_none()
    if not result:
        raise HTTPException(status_code=404, detail="老人不存在")
    return _elder_to_dict(result)


async def list_elder(db: AsyncSession, page: int, size: int):
    # page=0 时 (page-1)*size 为负，MySQL 会直接报语法错；size 过大则拖垮内存
    if page < 1:
        raise HTTPException(status_code=422, detail="page 必须大于等于 1")
    if size < 1 or size > 200:
        raise HTTPException(status_code=422, detail="size 必须在 1~200 之间")
    total = (await db.execute(select(func.count(Elder.id)))).scalar_one()
    stmt = select(Elder).order_by(Elder.elder_id).offset((page - 1) * size).limit(size)
    data = (await db.execute(stmt)).scalars().all()
    return {
        "total": total,
        "page": page,
        "size": size,
        "list": [_elder_to_dict(e) for e in data]
    }


async def update_elder(db: AsyncSession, elder_id: str, req: ElderCreate):
    stmt = (
        select(Elder)
        .where(Elder.elder_id == elder_id)
        .with_for_update()
        .execution_options(populate_existing=True)
    )
    result = await db.execute(stmt)
    elder = result.scalar_one_or_none()
    if not elder:
        raise HTTPException(status_code=404, detail="老人不存在")

    elder.name = req.name
    elder.age = req.age
    elder.gender = req.gender
    elder.health_tag = req.health_tag
    elder.chronic_disease = req.chronic_disease
    elder.phone = req.phone

    # 同上：字段已全部赋值，commit 前构造返回体，避免"已提交却报 500"
    data = _elder_to_dict(elder)
    try:
        await db.commit()
    except Exception as exc:
        await db.rollback()
        logger.exception("更新老人失败, elder_id=%s", elder_id)
        raise HTTPException(status_code=500, detail="更新老人失败") from exc
    return data


async def delete_elder(db: AsyncSession, elder_id: str):
    stmt = (
        select(Elder)
        .where(Elder.elder_id == elder_id)
        .with_for_update()
        .execution_options(populate_existing=True)
    )
    result = await db.execute(stmt)
    elder = result.scalar_one_or_none()
    if not elder:
        raise HTTPException(status_code=404, detail="老人不存在")
    try:
        await db.delete(elder)
        await db.commit()
    except Exception as exc:
        await db.rollback()
        logger.exception("删除老人失败, elder_id=%s", elder_id)
        raise HTTPException(status_code=500, detail="删除老人失败") from exc
    return {"msg": "删除成功"}


async def check_elder(db: AsyncSession, elder_id: str) -> bool:
    stmt = select(Elder).where(Elder.elder_id == elder_id)
    result = await db.execute(stmt)
    elder = result.scalar_one_or_none()
    return elder is not None
