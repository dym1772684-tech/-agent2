"""技能装配：决定"这一轮到底注入哪几个技能"。

================================================================
选择优先级（从强到弱）
----------------------------------------------------------------
1. **显式选择**：请求体 `skill_ids` —— 用户在前端点的那几个，最多 3 个。
   即使该技能没被 enable 也放行：用户当场选择就是授权。
2. **常开技能**：`user_skill_state.enabled=1 AND auto=1` —— 用户设的"每次都带上"。
3. **规则命中**：技能的 `when` 正则匹配本轮问题（零 token、零向量）。

三者累加，超过 MAX_ACTIVE(3) 截断。截断顺序按上面的优先级，所以"显式选择"
永远保得住，不会被规则匹配挤掉。

================================================================
为什么要规则而不是向量
----------------------------------------------------------------
技能数量是个位数，`when` 正则是人工维护的，准确率足够且零延迟。
引入 embedding 会让每轮多一次模型调用（约 20-40ms + 费用），
收益却只是"把'帮我排个表'这类说法也捞出来"——不值得。
真要多语言/口语化，再换成向量召回，接口保持不变。

================================================================
工具交集
----------------------------------------------------------------
技能可以声明 `tools`，但它只是"建议"，真正的拦截在 act 节点的
check_tool_permission 和 MCP 网关。这里做一次交集检查只为了：
  声明了用户没有的工具时，记一条日志（便于发现技能写错或权限配漏），
  不阻断——因为技能用不了那个工具时，模型会看到"工具不可用"的提示。
"""
from __future__ import annotations

import datetime
import logging
from typing import Any, Dict, List, Optional, Sequence

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from models.skill import UserSkillState

logger = logging.getLogger("agent.skill")


def _registry():
    """延迟导入：技能模块在 allagent2 里，避免 fastap2 顶层耦合。"""
    from allagent2.skills import registry as reg

    return reg


async def list_with_state(db: AsyncSession, user_id: str) -> List[Dict[str, Any]]:
    """技能库 + 当前用户的启用状态（前端技能面板的数据源）。"""
    reg = _registry()
    skills = reg.all_skills()
    rows = (await db.execute(
        select(UserSkillState).where(UserSkillState.user_id == str(user_id))
    )).scalars().all()
    state = {r.skill_id: r for r in rows}

    out: List[Dict[str, Any]] = []
    for s in skills:
        row = state.get(s.id)
        use = int(getattr(row, "use_count", 0) or 0)
        ok = int(getattr(row, "success_count", 0) or 0)
        bad = int(getattr(row, "fail_count", 0) or 0)
        out.append({
            **s.to_dict(),
            "enabled": bool(getattr(row, "enabled", 0)),
            "auto": bool(getattr(row, "auto", 0)),
            "use_count": use,
            # 没有反馈时不给 0 分，避免"从没用过"被误读成"评价很差"
            "success_rate": round(ok / (ok + bad), 3) if (ok + bad) else None,
            "last_used_at": row.last_used_at.isoformat() if getattr(row, "last_used_at", None) else None,
        })
    return out


async def set_enabled(db: AsyncSession, user_id: str, skill_id: str,
                      enabled: bool, auto: Optional[bool] = None) -> Dict[str, Any]:
    """启用 / 停用技能（不存在则先建行）。返回最新状态。"""
    reg = _registry()
    skill = reg.get_skill(skill_id)
    if skill is None:
        raise KeyError(f"技能不存在: {skill_id}")

    row = (await db.execute(
        select(UserSkillState).where(
            UserSkillState.user_id == str(user_id),
            UserSkillState.skill_id == skill.id,
        )
    )).scalar_one_or_none()

    if row is None:
        row = UserSkillState(
            user_id=str(user_id),
            skill_id=skill.id,
            enabled=1 if enabled else 0,
            auto=1 if (auto if auto is not None else False) else 0,
        )
        db.add(row)
    else:
        row.enabled = 1 if enabled else 0
        if auto is not None:
            row.auto = 1 if auto else 0
    await db.commit()
    await db.refresh(row)
    return {
        "skill_id": row.skill_id,
        "enabled": bool(row.enabled),
        "auto": bool(row.auto),
    }


async def assemble(
    db: AsyncSession,
    user_id: str,
    question: str,
    explicit_ids: Optional[Sequence[str]] = None,
    allowed_tools: Optional[Sequence[str]] = None,
) -> tuple[List[Any], Dict[str, Any]]:
    """装配本轮技能。返回 (skills, trace)，trace 用于审计与排障。"""
    reg = _registry()
    limit = reg.MAX_ACTIVE
    picked: List[Any] = []
    seen: set[str] = set()
    trace: Dict[str, Any] = {"explicit": [], "auto": [], "matched": [], "skipped": []}

    def _add(skill, source: str) -> bool:
        if skill is None or skill.id in seen:
            return False
        if len(picked) >= limit:
            trace["skipped"].append({"id": skill.id, "reason": "超过上限"})
            return False
        picked.append(skill)
        seen.add(skill.id)
        trace[source].append(skill.id)
        return True

    # ① 显式选择：用户当场点的，最高优先
    for sid in list(explicit_ids or [])[:limit]:
        _add(reg.get_skill(sid), "explicit")

    # 一次查出该用户全部技能状态：既用于"常开"判断，也用于自动匹配排序
    rows = (await db.execute(
        select(UserSkillState).where(UserSkillState.user_id == str(user_id))
    )).scalars().all()
    state_by_id = {r.skill_id: r for r in rows}

    # ② 常开技能：enabled=1 且 auto=1
    if len(picked) < limit:
        for row in rows:
            if len(picked) >= limit:
                break
            if int(row.enabled or 0) and int(row.auto or 0):
                _add(reg.get_skill(row.skill_id), "auto")

    # ③ 规则命中：补足剩余名额，按历史成功率排序（"用得越久越准"的落点）
    if len(picked) < limit and question:
        def _rank(skill) -> float:
            row = state_by_id.get(skill.id)
            ok = int(getattr(row, "success_count", 0) or 0)
            bad = int(getattr(row, "fail_count", 0) or 0)
            if not (ok + bad):
                return 0.5          # 没反馈按中性，不惩罚新技能
            return ok / (ok + bad)

        for skill in sorted(reg.match_skills(question, limit=limit + 3), key=_rank, reverse=True):
            if len(picked) >= limit:
                break
            _add(skill, "matched")

    # 工具声明与授权取交集（只告警不阻断：真正的拦截在 act / MCP 网关）
    if allowed_tools is not None:
        allowed = set(allowed_tools)
        for s in picked:
            missing = [t for t in s.tools if t not in allowed]
            if missing:
                logger.warning("[skill] %s 声明了未授权的工具 %s，本轮不保证可用", s.id, missing)
                trace.setdefault("tool_gaps", {})[s.id] = missing

    # 注入即视为一次使用：一次 SQL 批量 +1，不逐条 update
    if picked:
        ids = [s.id for s in picked]
        try:
            await db.execute(
                update(UserSkillState)
                .where(
                    UserSkillState.user_id == str(user_id),
                    UserSkillState.skill_id.in_(ids),
                )
                .values(
                    use_count=UserSkillState.use_count + 1,
                    last_used_at=datetime.datetime.now(),
                )
            )
            await db.commit()
        except Exception as exc:  # 计数失败不影响本轮注入
            logger.warning("[skill] 使用计数写入失败（不影响注入）：%s", exc)
            await db.rollback()

    if picked:
        logger.info("[skill] 本轮注入 %s（%s）", [s.id for s in picked], trace)
    return picked, trace


def assemble_sync(
    user_id: str,
    question: str,
    explicit_ids: Optional[Sequence[str]] = None,
    allowed_tools: Optional[Sequence[str]] = None,
) -> tuple[List[Any], Dict[str, Any]]:
    """assemble() 的同步版：给跑在线程池里的胶水层用（bootstrap._prepare）。

    选择规则与 async 版完全一致（显式 > 常开 > 规则命中，上限 MAX_ACTIVE），
    唯一差别是走 database.get_sync_engine() 的同步会话。
    """
    from sqlalchemy.orm import Session

    from database.database import get_sync_engine

    reg = _registry()
    limit = reg.MAX_ACTIVE
    picked: List[Any] = []
    seen: set[str] = set()
    trace: Dict[str, Any] = {"explicit": [], "auto": [], "matched": [], "skipped": []}

    def _add(skill, source: str) -> bool:
        if skill is None or skill.id in seen:
            return False
        if len(picked) >= limit:
            trace["skipped"].append({"id": skill.id, "reason": "超过上限"})
            return False
        picked.append(skill)
        seen.add(skill.id)
        trace[source].append(skill.id)
        return True

    for sid in list(explicit_ids or [])[:limit]:
        _add(reg.get_skill(sid), "explicit")

    with Session(get_sync_engine()) as db:
        rows = db.execute(
            select(UserSkillState).where(UserSkillState.user_id == str(user_id))
        ).scalars().all()
        state_by_id = {r.skill_id: r for r in rows}

        if len(picked) < limit:
            for row in rows:
                if len(picked) >= limit:
                    break
                if int(row.enabled or 0) and int(row.auto or 0):
                    _add(reg.get_skill(row.skill_id), "auto")

        if len(picked) < limit and question:
            def _rank(skill) -> float:
                row = state_by_id.get(skill.id)
                ok = int(getattr(row, "success_count", 0) or 0)
                bad = int(getattr(row, "fail_count", 0) or 0)
                return 0.5 if not (ok + bad) else ok / (ok + bad)

            for skill in sorted(reg.match_skills(question, limit=limit + 3),
                                key=_rank, reverse=True):
                if len(picked) >= limit:
                    break
                _add(skill, "matched")

        if allowed_tools is not None:
            allowed = set(allowed_tools)
            for s in picked:
                missing = [t for t in s.tools if t not in allowed]
                if missing:
                    logger.warning("[skill] %s 声明了未授权的工具 %s，本轮不保证可用",
                                   s.id, missing)
                    trace.setdefault("tool_gaps", {})[s.id] = missing

        if picked:
            try:
                db.execute(
                    update(UserSkillState)
                    .where(
                        UserSkillState.user_id == str(user_id),
                        UserSkillState.skill_id.in_([s.id for s in picked]),
                    )
                    .values(
                        use_count=UserSkillState.use_count + 1,
                        last_used_at=datetime.datetime.now(),
                    )
                )
                db.commit()
            except Exception as exc:  # 计数失败不影响本轮注入
                logger.warning("[skill] 使用计数写入失败（不影响注入）：%s", exc)
                db.rollback()

    if picked:
        logger.info("[skill] 本轮注入 %s（%s）", [s.id for s in picked], trace)
    return picked, trace


async def record_feedback(db: AsyncSession, user_id: str, skill_id: str,
                          helpful: bool) -> Dict[str, Any]:
    """用户对技能效果的评价。这是"用得越久越准"的数据来源。"""
    row = (await db.execute(
        select(UserSkillState).where(
            UserSkillState.user_id == str(user_id),
            UserSkillState.skill_id == str(skill_id),
        )
    )).scalar_one_or_none()
    if row is None:
        raise KeyError(f"用户未启用该技能: {skill_id}")

    if helpful:
        row.success_count = int(row.success_count or 0) + 1
    else:
        row.fail_count = int(row.fail_count or 0) + 1
    await db.commit()
    await db.refresh(row)
    return {
        "skill_id": skill_id,
        "success_count": int(row.success_count or 0),
        "fail_count": int(row.fail_count or 0),
        "success_rate": round(
            row.success_count / max(row.success_count + row.fail_count, 1), 3
        ),
    }
