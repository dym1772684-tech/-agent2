from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import desc, select
from sqlalchemy.ext.asyncio import AsyncSession

from models.health import HealthRecord

from core.acl import (  # 【ACL】权限码 + 依赖工厂 + 对象级校验
    PERM_HEALTH_READ,
    PERM_HEALTH_WRITE,
    assert_elder_access,
    require_permission,
)
from core.responses import ApiResponse
from database.database import get_db
from schemas.health import HealthRecord_create
from service.health_services import (
    get_record,
    delete_record,
    update_record,
    create_record,
    record_to_dict,  # 【修复】原用 HealthRecord_create.model_validate(ORM) 序列化：
                     # schema 要求 health_id，而表列叫 id，且默认不读属性 → 必抛 ValidationError
)

# 【ACL】原 router 级 dependencies=[Depends(check_jwt)] 拆到端点级权限码;
# 健康记录属于具体老人的数据行,读写都要过对象级绑定校验。
router = APIRouter(prefix="/health", tags=["健康"])


@router.post("/create_record")
async def post(
    data: HealthRecord_create,
    db: AsyncSession = Depends(get_db),
    current=Depends(require_permission(PERM_HEALTH_WRITE)),  # 【ACL】写记录需 elder:health_write
):
    await assert_elder_access(db, current, data.elder_id)  # 【ACL】对象级:只能给绑定的老人写记录
    await create_record(db, data)
    return ApiResponse.success(msg="成功")


@router.get("/record/{id}")
async def get__record(
    id: str,
    db: AsyncSession = Depends(get_db),
    current=Depends(require_permission(PERM_HEALTH_READ)),
):
    result = await get_record(id, db)
    if result is None:
        raise HTTPException(status_code=404, detail="资源不存在")
    # 【ACL】对象级:先按记录 id 查到行,再校验该行所属老人是否在授权范围
    await assert_elder_access(db, current, result.elder_id)
    return record_to_dict(result)


@router.get("/records/{elder_id}")
async def list_records(
    elder_id: str,
    limit: int = Query(default=100, ge=1, le=500),
    metric_type: str | None = Query(default=None, description="只取某个指标"),
    db: AsyncSession = Depends(get_db),
    current=Depends(require_permission(PERM_HEALTH_READ)),
):
    """某老人的健康记录列表（按测量时间倒序）。管理台与详情页都用这个。"""
    await assert_elder_access(db, current, elder_id)
    stmt = select(HealthRecord).where(HealthRecord.elder_id == elder_id)
    if metric_type:
        stmt = stmt.where(HealthRecord.metric_type == metric_type)
    stmt = stmt.order_by(desc(HealthRecord.record_time), desc(HealthRecord.id)).limit(limit)
    rows = (await db.execute(stmt)).scalars().all()
    return ApiResponse.success(data={
        "elder_id": elder_id,
        "total": len(rows),
        "items": [record_to_dict(r) for r in rows],
    })


# 常见指标的参考区间（仅用于规则级初筛，不构成诊断）
REFERENCE_RANGES: dict[str, tuple[float, float]] = {
    "血压": (90.0, 140.0),
    "收缩压": (90.0, 140.0),
    "高压": (90.0, 140.0),
    "舒张压": (60.0, 90.0),
    "低压": (60.0, 90.0),
    "血糖": (3.9, 7.8),
    "心率": (60.0, 100.0),
    "脉搏": (60.0, 100.0),
    "体温": (36.0, 37.3),
    "血氧": (95.0, 100.0),
    "体重": (40.0, 100.0),
}


def _reference(metric_type: str) -> tuple[float, float] | None:
    """按指标名匹配参考区间（子串匹配，兼容「收缩压(晨)」这类写法）。"""
    name = (metric_type or "").strip()
    if name in REFERENCE_RANGES:
        return REFERENCE_RANGES[name]
    for key, rng in REFERENCE_RANGES.items():
        if key in name:
            return rng
    return None


async def _load_grouped(db: AsyncSession, elder_id: str, limit: int = 300):
    """取最近记录并按指标分组（组内保持时间倒序）。"""
    stmt = (select(HealthRecord)
            .where(HealthRecord.elder_id == elder_id)
            .order_by(desc(HealthRecord.record_time), desc(HealthRecord.id))
            .limit(limit))
    rows = (await db.execute(stmt)).scalars().all()
    grouped: dict[str, list] = {}
    for r in rows:
        grouped.setdefault(r.metric_type or "未分类", []).append(r)
    return rows, grouped


@router.get("/stats/{elder_id}")
async def return_record(
    elder_id: str,
    db: AsyncSession = Depends(get_db),
    current=Depends(require_permission(PERM_HEALTH_READ)),
):
    """指标汇总：每个指标的条数 / 最新 / 最小 / 最大 / 均值，以及时间跨度。"""
    await assert_elder_access(db, current, elder_id)
    rows, grouped = await _load_grouped(db, elder_id)

    metrics = []
    for name, items in grouped.items():
        values = [float(i.value) for i in items if i.value is not None]
        if not values:
            continue
        ref = _reference(name)
        latest = values[0]
        status = "正常"
        if ref and latest > ref[1]:
            status = "偏高"
        elif ref and latest < ref[0]:
            status = "偏低"
        metrics.append({
            "metric_type": name,
            "count": len(values),
            "latest": latest,
            "min": min(values),
            "max": max(values),
            "avg": round(sum(values) / len(values), 2),
            "unit": "",
            "reference": {"low": ref[0], "high": ref[1]} if ref else None,
            "status": status if ref else "无参考区间",
            "latest_at": items[0].record_time.isoformat() if items[0].record_time else None,
        })
    metrics.sort(key=lambda m: m["latest_at"] or "", reverse=True)

    return ApiResponse.success(data={
        "elder_id": elder_id,
        "total": len(rows),
        "metric_count": len(metrics),
        "first_at": rows[-1].record_time.isoformat() if rows and rows[-1].record_time else None,
        "last_at": rows[0].record_time.isoformat() if rows and rows[0].record_time else None,
        "metrics": metrics,
    })


@router.get("/risk/{elder_id}")
async def return_risk(
    elder_id: str,
    consecutive: int = Query(default=3, ge=1, le=10, description="连续几次超限判为高风险"),
    db: AsyncSession = Depends(get_db),
    current=Depends(require_permission(PERM_HEALTH_READ)),
):
    """规则级风险初筛：最新一次超限 → 中；连续 N 次超限 → 高。

    这不是诊断，只是把「该提醒家属看一眼」的记录标出来。
    """
    await assert_elder_access(db, current, elder_id)
    rows, grouped = await _load_grouped(db, elder_id)

    LEVEL_ORDER = {"低": 0, "中": 1, "高": 2}
    items = []
    overall = "低"

    for name, group in grouped.items():
        ref = _reference(name)
        values = [float(i.value) for i in group if i.value is not None]
        if not ref or not values:
            continue
        low, high = ref
        latest = values[0]
        hits = sum(1 for v in values[:consecutive] if v > high)

        if latest > high:
            level = "高" if hits >= consecutive else "中"
            direction = "偏高"
        elif latest < low:
            level = "中"
            direction = "偏低"
        else:
            level = "低"
            direction = "正常"

        if LEVEL_ORDER[level] > LEVEL_ORDER[overall]:
            overall = level

        items.append({
            "metric_type": name,
            "latest": latest,
            "reference": {"low": low, "high": high},
            "direction": direction,
            "level": level,
            "consecutive_over": hits,
            "checked": min(len(values), consecutive),
            "latest_at": group[0].record_time.isoformat() if group[0].record_time else None,
        })

    items.sort(key=lambda i: LEVEL_ORDER[i["level"]], reverse=True)

    return ApiResponse.success(data={
        "elder_id": elder_id,
        "risk_level": overall,
        "items": items,
        "total_records": len(rows),
        "note": "规则级初筛，仅供参考，不替代医生诊断；出现急症请立即拨打 120。",
    })


@router.delete("/record/{record_id}")
async def delete_health_record(
    record_id: int,
    db: AsyncSession = Depends(get_db),
    current=Depends(require_permission(PERM_HEALTH_WRITE)),
):
    """删除一条健康记录（管理员清理脏数据用）。

    锁顺序与 service 里的 update/delete 保持一致：先 FOR UPDATE 定位行，
    再按行所属老人做对象级校验，最后删除。
    """
    stmt = (
        select(HealthRecord)
        .where(HealthRecord.id == record_id)
        .with_for_update()
        .execution_options(populate_existing=True)
    )
    rec = (await db.execute(stmt)).scalar_one_or_none()
    if rec is None:
        raise HTTPException(status_code=404, detail="记录不存在")
    await assert_elder_access(db, current, rec.elder_id)

    try:
        await db.delete(rec)
        await db.commit()
    except Exception as exc:  # 事务失败必须回滚，否则连接会带着脏状态回池
        await db.rollback()
        raise HTTPException(status_code=500, detail="删除健康记录失败") from exc
    return ApiResponse.success(msg="已删除", data={"health_id": record_id})
