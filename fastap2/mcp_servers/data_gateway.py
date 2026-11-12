"""数据面网关:养老数据 MCP 服务唯一的数据库出口。

【安全模型】与原 fastap2/agent_bridge/middleware.py 的 DBAccessMiddleware 同源:
    鉴权 → 查询 → 敏感扫描 → JSON 返回
区别只在鉴权上下文的来源:
    原来 = contextvars(同进程);现在 = agent_context HMAC 签名令牌(跨进程)。
fail-closed:令牌缺失/无效/过期 → 一律拒绝,不触碰数据库。

工具迁移到 MCP 后,本模块是"唯一允许访问数据库的组件"(fastap2 侧的
DBAccessMiddleware 不再被智能体链路使用)。
"""
import json
import re
from typing import Optional, Tuple

from sqlalchemy import create_engine, desc, select
from sqlalchemy.orm import Session

from core.agent_context import verify_context_token
from core.audit import write_audit

# ---- 敏感规则(与原 DBAccessMiddleware 保持一致)----
# 【修复】\b 词边界在 Unicode 模式下失效：中文是 \w，"手机138..." 里"机"和"1"
# 之间不存在 ASCII 词边界 → 中文语境的手机号/身份证从不脱敏（真实泄露 bug，
# 由 tests/t01 PII 用例抓出）。改用数字 lookaround 边界 (?<!\d)...(?!\d)。
PII_PATTERNS = [
    (re.compile(r"(?<!\d)\d{17}[\dXx](?!\d)"), "身份证号"),
    (re.compile(r"(?<!\d)1[3-9]\d{9}(?!\d)"), "手机号"),
    (re.compile(r"(?<!\d)0\d{2,3}-?\d{7,8}(?!\d)"), "座机电话"),
    (re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+"), "邮箱"),
]
STRONG_SENSITIVE_FIELDS = ("身份证", "id_card", "完整病史", "病历", "医保")

_engine = None


def _get_engine():
    global _engine
    if _engine is None:
        import os

        url = (
            f"mysql+pymysql://{os.getenv('MYSQL_USER')}:{os.getenv('MYSQL_PASSWORD')}"
            f"@{os.getenv('MYSQL_HOST', '127.0.0.1')}:{os.getenv('MYSQL_PORT', '3306')}"
            f"/{os.getenv('MYSQL_DATABASE')}?charset=utf8mb4"
        )
        _engine = create_engine(url, pool_pre_ping=True, pool_recycle=3600)
    return _engine


def _mask_pii(text: str) -> str:
    """无敏感权限时对 PII 脱敏(与原实现一致)。"""
    def mask(m: re.Match, keep_head: int, keep_tail: int) -> str:
        s = m.group(0)
        if len(s) <= keep_head + keep_tail:
            return s[:1] + "****"
        return s[:keep_head] + "****" + (s[-keep_tail:] if keep_tail else "")

    for pat, _ in PII_PATTERNS:
        text = pat.sub(lambda m: mask(m, 3, 4), text)
    return text


def execute(tool_name: str, args: dict, agent_context: str) -> str:
    """网关统一入口:验签 → 鉴权 → 查询 → 脱敏 → 审计 → JSON 返回。"""
    payload = verify_context_token(agent_context or "")
    user_id = str((payload or {}).get("user_id") or "anonymous")
    resource = f"elder:{(args or {}).get('elder_id', '-')}"

    # 1) 令牌校验(fail-closed)
    if payload is None:
        write_audit(user_id, f"mcp:{tool_name}", resource=resource,
                    decision=False, reason="agent_context 缺失/无效/过期")
        return json.dumps({"code": 403, "msg": "权限上下文无效(未登录或令牌过期)"},
                          ensure_ascii=False)

    # 2) 工具白名单
    allowed_tools = payload.get("allowed_tools")
    if allowed_tools is not None and tool_name not in allowed_tools:
        write_audit(user_id, f"mcp:{tool_name}", resource=resource,
                    decision=False, reason="工具不在白名单")
        return json.dumps({"code": 403, "msg": f"无权调用工具 {tool_name}(不在白名单内)"},
                          ensure_ascii=False)

    # 3) 老人数据越权校验(allowed_elder_ids:None=不限;列表=白名单;空=全禁)
    elder_id = (args or {}).get("elder_id")
    if not elder_id:
        return json.dumps({"code": 400, "msg": "缺少老人编号 elder_id"}, ensure_ascii=False)
    allowed_ids = payload.get("allowed_elder_ids")
    if allowed_ids is not None and elder_id not in allowed_ids:
        write_audit(user_id, f"mcp:{tool_name}", resource=resource,
                    decision=False, reason=f"越权:elder {elder_id} 不在授权范围")
        return json.dumps({"code": 403, "msg": f"无权访问老人 {elder_id} 的数据(超出授权范围)"},
                          ensure_ascii=False)

    # 4) 查询(异常不外泄细节给 model)
    try:
        if tool_name == "query_elder_profile":
            raw = _query_profile(elder_id)
        elif tool_name == "query_health_records":
            raw = _query_health(elder_id, (args or {}).get("metric_type"))
        else:
            return json.dumps({"code": 400, "msg": f"网关不支持的工具: {tool_name}"},
                              ensure_ascii=False)
    except Exception as exc:
        write_audit(user_id, f"mcp:{tool_name}", resource=resource,
                    decision=False, reason=f"db error: {type(exc).__name__}")
        return json.dumps({"code": 500, "msg": "数据库查询失败(详情已脱敏,请联系管理员查看日志)"},
                          ensure_ascii=False)

    # 5) 敏感扫描:强敏感字段命中且无权限 → 阻断;否则 PII 脱敏
    if not payload.get("allow_sensitive"):
        for f in STRONG_SENSITIVE_FIELDS:
            if f in raw:
                write_audit(user_id, f"mcp:{tool_name}", resource=resource,
                            decision=False, reason=f"强敏感字段命中:{f}")
                return json.dumps({"code": 451, "msg": "检测到敏感信息,已阻断,不进入智能体链路"},
                                  ensure_ascii=False)
        raw = _mask_pii(raw)

    # 6) 放行 + 审计留痕
    write_audit(user_id, f"mcp:{tool_name}", resource=resource, decision=True, reason="ok")
    return json.dumps({"code": 0, "data": {"tool": tool_name, "result": raw}},
                      ensure_ascii=False)


def _query_profile(elder_id: str) -> str:
    from models.elder import Elder

    with Session(_get_engine()) as session:
        elder = session.execute(
            select(Elder).where(Elder.elder_id == elder_id)
        ).scalar_one_or_none()
        if elder is None:
            return "档案里没有该老人的记录"
        return (
            f"姓名:{elder.name};年龄:{elder.age};性别:{elder.gender};"
            f"自理状态:{elder.health_tag};慢性病:{elder.chronic_disease};联系电话:{elder.phone}"
        )


def _query_health(elder_id: str, metric_type: Optional[str]) -> str:
    from models.health import HealthRecord

    with Session(_get_engine()) as session:
        stmt = select(HealthRecord).where(HealthRecord.elder_id == elder_id)
        if metric_type:
            stmt = stmt.where(HealthRecord.metric_type == metric_type)
        stmt = stmt.order_by(desc(HealthRecord.record_time)).limit(10)
        rows = session.execute(stmt).scalars().all()
        if not rows:
            return "档案里没有该老人的健康监测记录"
        return ";".join(
            f"{r.record_time if r.record_time is not None else '时间未知'} {r.metric_type}={r.value}"
            for r in rows
        )
