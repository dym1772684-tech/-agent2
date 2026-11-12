"""可信身份体系（零信任网关第③步 + 链路内硬性鉴权的数据源）。

三张注册表是本体系的"宪法"（新增功能/智能体/资源必须先注册再授权）：
  PERM_REGISTRY      功能权限键值对：权限码 → 功能描述/资源类型/敏感档位
  ROLE_DEFAULT_PERMS 角色→权限码的硬编码回退（role_permission 表缺失时生效）
  AGENT_PROFILES     智能体→所需权限码/工具白名单（team 派发过滤的判据）
  RESOURCE_REGISTRY  资源类型→归属判定/执行点（can() 只认已注册类型，未注册=拒绝）

TrustedContext 生成时机：FastAPI _authorize_chat 三步流水线（认证→清洗→装配）
走完后构建 → Redis identity:{identity_id} 缓存（TTL 900s）→ contextvars/state
只放 identity_id 引用 + 冻结副本。

失效机制：perm_version（Redis 计数器 user:permver:{user_id}）——角色/绑定/账户
状态变更即 INCR，旧身份在下次读取时版本比对失败即弃用（不等 TTL）。
Redis 不可用：回源 DB 重建身份（绝不 fail-open 放行）。
"""
from __future__ import annotations

import logging
import uuid
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set

from pydantic import BaseModel, Field

logger = logging.getLogger("agent")

# ============================================================
# ① 功能权限注册表：权限码 → {功能, 资源类型, 敏感档位}
# ============================================================
PERM_REGISTRY: Dict[str, Dict[str, str]] = {
    "agent:chat":             {"desc": "发起智能体对话",           "resource": "knowledge",  "tier": "basic"},
    "agent:memory_read":      {"desc": "读取自己的长期记忆",       "resource": "memory",     "tier": "basic"},
    "agent:memory_delete":    {"desc": "删除自己的记忆(遗忘权)",   "resource": "memory",     "tier": "basic"},
    "elder:profile_read":     {"desc": "查询老人档案",             "resource": "elder",      "tier": "sensitive"},
    "elder:health_read":      {"desc": "查询健康记录",             "resource": "elder",      "tier": "sensitive"},
    "elder:health_analyze":   {"desc": "健康趋势分析",             "resource": "elder",      "tier": "sensitive"},
    # 【权限词表统一】以下三个码原来只在 core/acl.py 里定义（elder:write / elder:delete /
    # health:write），identity 的注册表里没有 → 路由鉴权和实体层鉴权各说一套。
    # 现在统一登记到本表，acl.py 只做引用，不再自己发明权限码。
    "elder:write":            {"desc": "新增/修改老人档案",        "resource": "elder",      "tier": "sensitive"},
    "elder:delete":           {"desc": "删除老人档案",             "resource": "elder",      "tier": "sensitive"},
    "elder:health_write":     {"desc": "写入健康监测记录",         "resource": "elder",      "tier": "sensitive"},
    "tool:web_fetch":         {"desc": "访问公网网页",             "resource": "tool",       "tier": "personal"},
    "tool:weather":           {"desc": "天气查询",                 "resource": "tool",       "tier": "public"},
    "tool:calc":              {"desc": "计算器/BMI",               "resource": "tool",       "tier": "public"},
    "audit:read":             {"desc": "查看思考链路/审计日志",    "resource": "audit",      "tier": "admin"},
    "system:bind":            {"desc": "绑定/解绑老人、用户管理",  "resource": "admin",      "tier": "admin"},
    "system:diag":            {"desc": "系统诊断端点",             "resource": "admin",      "tier": "admin"},
}

# 角色→权限码硬编码回退（role_permission 表建好后以表为准，本表仅在表缺失时兜底）
# 【唯一词表】core/acl.py 的 require_permission 现在也查这里（DB 优先、60s 缓存），
# 不再维护第二份 ROLE_PERMISSIONS；新增角色/权限码只改这一处。
ROLE_DEFAULT_PERMS: Dict[str, Set[str]] = {
    "admin": set(PERM_REGISTRY.keys()),
    # 家属：可对话+管理自己记忆+查看绑定老人的健康数据；工具三档全开
    "family": {"agent:chat", "agent:memory_read", "agent:memory_delete",
               "elder:profile_read", "elder:health_read", "elder:health_analyze",
               "tool:web_fetch", "tool:weather", "tool:calc"},
    # 护理员/护工：可读写绑定老人的档案与健康记录（原来只存在于 acl.py，
    # identity 里没有这个角色 → 走 agent 链路时权限会退化成 common，这里补齐）
    "nurse": {"agent:chat", "agent:memory_read", "agent:memory_delete",
              "elder:profile_read", "elder:health_read", "elder:health_analyze",
              "elder:write", "elder:health_write",
              "tool:web_fetch", "tool:weather", "tool:calc"},
    # 志愿者：纯生活助手，无任何老人数据权限
    "volunteer": {"agent:chat", "agent:memory_read", "agent:memory_delete",
                  "tool:web_fetch", "tool:weather", "tool:calc"},
    # 未绑定普通用户：对话+公开/个人工具+（绑定后可看）老人只读
    "common": {"agent:chat", "agent:memory_read", "agent:memory_delete",
           "elder:profile_read", "elder:health_read",   # ← 补：家属场景所需
           "tool:weather", "tool:calc"},
}

# ============================================================
# ② 智能体权限档案：team 派发过滤的判据（supervisor 依此决定任务可派性）
# ============================================================
AGENT_PROFILES: Dict[str, Dict[str, Any]] = {
    "supervisor": {"perms": set(), "tools": []},  # 只规划，无数据权限
    "health":     {"perms": {"elder:profile_read", "elder:health_read", "elder:health_analyze"},
                   "tools": ["query_elder_profile", "query_health_records", "analyze_health_trend"]},
    "knowledge":  {"perms": {"agent:chat"}, "tools": ["search_knowledge_base"]},
    "life":       {"perms": {"tool:web_fetch", "tool:weather", "tool:calc"},
                   "tools": ["fetch_webpage", "get_real_weather", "calculator", "compute_bmi"]},
}

# 工具 → 所需权限码（allowed_tools 的生成规则：角色权限 ∩ 本表）
TOOL_PERM_MAP: Dict[str, str] = {
    "calculator": "tool:calc",
    "compute_bmi": "tool:calc",
    "get_real_weather": "tool:weather",
    "fetch_webpage": "tool:web_fetch",
    "analyze_health_trend": "elder:health_analyze",
    "query_elder_profile": "elder:profile_read",
    "query_health_records": "elder:health_read",
    "search_knowledge_base": "agent:chat",
}

# ============================================================
# ③ 资源注册表：资源类型 → {归属判定说明, 执行点}
#    can() 只认这里注册的类型；未注册类型一律 False（fail-closed）。
# ============================================================
RESOURCE_REGISTRY: Dict[str, Dict[str, str]] = {
    "thread":   {"owner": "conversation_owner.user_id", "enforce": "router"},
    "elder":    {"owner": "user_elder_binding",         "enforce": "mcp_gateway"},
    "health":   {"owner": "随 elder（老人可访问⇒记录可访问）", "enforce": "mcp_gateway"},
    "memory":   {"owner": "identity.user_id（仅本人）",  "enforce": "router+store"},
    "knowledge":{"owner": "登录即可读（全局共享）",      "enforce": "endpoint"},
    "audit":    {"owner": "perm audit:read",            "enforce": "router"},
    "admin":    {"owner": "perm system:*",              "enforce": "router"},
    "tool":     {"owner": "TOOL_PERM_MAP ∩ 角色权限",   "enforce": "mcp_gateway"},
}


# ============================================================
# TrustedContext 模型
# ============================================================
class TrustedContext(BaseModel):
    """双层校验通过后的唯一可信权限实体。

    FastAPI 层生成、Redis 缓存；Agent 层（identity_guard/team 派发过滤/
    MCP 网关）只依据本对象判断，不引入第二份权限来源。
    """
    identity_id: str                      # 审计串联键（itd-<uuid12>）
    user_id: str
    tenant_id: str = "t-default"          # 多租户预留
    user_role: str
    session_id: str = ""                  # 已通过 conversation_owner 属主校验
    permissions: List[str] = Field(default_factory=list)     # 功能级（RBAC）
    allowed_elders: Optional[List[str]] = None               # 对象级（admin=None 不限；[] 全拒）
    allowed_tools: List[str] = Field(default_factory=list)   # 工具白名单（三档规则过滤后）
    perm_version: int = 0
    issued_at: str = ""
    expires_at: str = ""

    def can(self, resource_type: str, resource_id: Optional[str], action: str = "read") -> bool:
        """统一授权判断入口（所有执行点都调这里）。

        fail-closed 三连：未注册资源类型=False；身份过期=False；
        对象级列表不含目标=False（admin 对 elder 类豁免，其余按权限码）。
        """
        if resource_type not in RESOURCE_REGISTRY:
            return False
        if self._expired():
            return False
        if action == "read" and resource_type == "elder":
            return self.allowed_elders is None or resource_id in (self.allowed_elders or [])
        if resource_type == "memory":
            return resource_id in (None, self.user_id) and "agent:memory_read" in self.permissions
        if resource_type in ("audit", "admin", "tool", "knowledge"):
            need = {"audit": "audit:read", "admin": "system:diag",
                    "tool": TOOL_PERM_MAP.get(resource_id or "", "agent:chat"),
                    "knowledge": "agent:chat"}.get(resource_type, "")
            return need in self.permissions
        return False

    def _expired(self) -> bool:
        try:
            return datetime.fromisoformat(self.expires_at) < datetime.now()
        except Exception:
            return True


# ============================================================
# RBAC：role_permission 表（缺失时回退硬编码）
# ============================================================
_SYNC_ENGINE = None


def _sync_engine():
    """独立同步引擎（pymysql）：身份模块在路由与 agent 线程池都会用到，
    与请求级 AsyncSession 解耦（backend_tools/data_gateway 同款手法）。"""
    global _SYNC_ENGINE
    if _SYNC_ENGINE is None:
        import os

        from sqlalchemy import create_engine

        url = (f"mysql+pymysql://{os.getenv('MYSQL_USER')}:{os.getenv('MYSQL_PASSWORD')}"
               f"@{os.getenv('MYSQL_HOST')}:{os.getenv('MYSQL_PORT', 3306)}/{os.getenv('MYSQL_DATABASE')}"
               "?charset=utf8mb4")
        _SYNC_ENGINE = create_engine(url, pool_pre_ping=True, pool_size=2)
    return _SYNC_ENGINE


def load_role_permissions(role: str) -> Set[str]:
    """读角色的权限码集合：DB 表优先，表不存在/查失败回退 ROLE_DEFAULT_PERMS。"""
    try:
        from sqlalchemy import text

        with _sync_engine().begin() as conn:
            rows = conn.execute(
                text("SELECT perm_code FROM role_permission WHERE role=:r AND active=1"),
                {"r": role},
            ).scalars().all()
        if rows:
            return set(rows)
    except Exception:
        pass  # 表还没建/DB 不可用 → 回退
    return set(ROLE_DEFAULT_PERMS.get(role, ROLE_DEFAULT_PERMS["common"]))


def tools_for(perms: Set[str]) -> List[str]:
    """工具白名单 = TOOL_PERM_MAP 中权限码被角色持有的工具。"""
    return [t for t, p in TOOL_PERM_MAP.items() if p in perms]


# ============================================================
# Redis 缓存：identity:{id} + perm_version 撤销
# ============================================================
_TTL_SECONDS = 900
_KEY_ID = "identity:{}"
_KEY_VER = "user:permver:{}"


def _redis():
    from core.redis_client import get_redis_sync

    return get_redis_sync()


def get_perm_version(user_id: str) -> int:
    try:
        from core.redis_client import redis_available

        if not redis_available():
            return 0
        v = _redis().get(_KEY_VER.format(user_id))
        return int(v) if v else 0
    except Exception:
        return 0


def bump_perm_version(user_id: str) -> int:
    """角色/绑定/账户状态变更时调用：旧身份全部即时失效。"""
    try:
        return int(_redis().incr(_KEY_VER.format(user_id)))
    except Exception:
        return get_perm_version(user_id) + 1


def build_identity(user_id: str, user_role: str, session_id: str,
                   bound_elder_ids: Optional[List[str]] = None) -> TrustedContext:
    """服务端装配可信身份（客户端传入的任何 allowed_* 一律不采信）。"""
    perms = load_role_permissions(user_role)
    now = datetime.now()
    return TrustedContext(
        identity_id=f"itd-{uuid.uuid4().hex[:12]}",
        user_id=user_id,
        user_role=user_role,
        session_id=session_id,
        permissions=sorted(perms),
        allowed_elders=None if user_role == "admin" else list(bound_elder_ids or []),
        allowed_tools=tools_for(perms),
        perm_version=get_perm_version(user_id),
        issued_at=now.isoformat(timespec="seconds"),
        expires_at=(now + timedelta(seconds=_TTL_SECONDS)).isoformat(timespec="seconds"),
    )


def put_identity(ctx: TrustedContext) -> bool:
    """写入 Redis（JSON，TTL 900s）。失败返回 False（调用方仍可内存使用本请求副本）。"""
    try:
        from core.redis_client import redis_available

        if not redis_available():
            return False
        return bool(_redis().setex(
            _KEY_ID.format(ctx.identity_id), _TTL_SECONDS,
            ctx.model_dump_json(),
        ))
    except Exception as exc:
        logger.warning("[identity] Redis 写入失败（本请求用内存副本，后续请求回源重建）：%s", exc)
        return False


def get_identity(identity_id: str, expected_user_id: Optional[str] = None) -> Optional[TrustedContext]:
    """读取并双重校验：TTL 内 + perm_version 未变 +（可选）属主匹配。

    版本不符/过期/不存在 → None（调用方回源重建或拒绝）。
    """
    try:
        from core.redis_client import redis_available

        if not redis_available():
            return None
        raw = _redis().get(_KEY_ID.format(identity_id))
        if not raw:
            return None
        ctx = TrustedContext.model_validate_json(raw)
        if ctx._expired():
            return None
        if expected_user_id and ctx.user_id != expected_user_id:
            return None  # 属主不匹配：拿别人的 identity_id 也用不了
        if ctx.perm_version != get_perm_version(ctx.user_id):
            logger.info("[identity] 身份已因权限变更失效：%s", identity_id)
            return None
        return ctx
    except Exception as exc:
        logger.warning("[identity] Redis 读取失败（按无缓存处理）：%s", exc)
        return None


def revoke_user(user_id: str) -> int:
    """撤销某用户的全部在线身份（登出/降权/锁定）：版本+1 即全失效。"""
    return bump_perm_version(user_id)
