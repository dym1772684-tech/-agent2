"""Agent 数据访问与敏感信息中间件（middleware 层）。

【安全模型】
1. DBAccessMiddleware —— 唯一允许访问数据库的组件（DB 网关）。
   - 智能体（LangGraph）的工具调用一律经过本网关：鉴权 → 查询 → 敏感扫描 → JSON 返回；
   - 数据库工具函数本身不再持有任何 DB engine/连接（见 backend_tools.py 的薄壳改造），
     真正执行查询的代码只存在于本网关内部 → “智能体不能自己调用数据库，只让 middleware 调”。
   - 每次 DB 调用都做鉴权（工具白名单 / 老人数据越权 / 角色），默认“拒绝”兜底（fail closed）。

2. SensitiveGuard —— 入口级敏感阻断。
   - 在智能体链路启动前（LLM 进入之前）扫描用户消息，命中敏感信息请求且无权限
     → 直接阻断，不进入智能体链路。

3. 鉴权上下文通过 contextvars 传递:
   - run_agent / HTTP router 入口调用 set_access_context() 注入当前用户权限;
   - 网关执行时读取;无上下文 = 按 common + 不允许敏感处理(安全默认)。
   【修复】原用 threading.local:enter_context 在事件循环线程设置,
   run_in_threadpool 在 worker 线程执行,读不到上下文 → 网关鉴权整体失效。
   改用 contextvars:asyncio 与 anyio 线程池都会正确复制/传播上下文。

返回给智能体服务层的标准 JSON 格式：
   正常  ： {"code": 0,   "data": {"tool": "<工具名>", "result": "<查询结果>"}}
   无权限： {"code": 403, "msg": "<拒绝原因>"}
   敏感阻断：{"code": 451, "msg": "<阻断原因>"}
"""
from __future__ import annotations

import json
import os
import re
import threading
from typing import Any, Dict, List, Optional, Tuple

from sqlalchemy import create_engine, desc, select
from sqlalchemy.orm import Session

from core.audit import write_audit  # 【审计】网关每次允许/拒绝都落审计
from core.cache import (  # 【缓存】Cache-Aside 同步版(网关跑在线程池里)
    NULL_SENTINEL,
    cache_get_json_sync,
    cache_set_json_sync,
    elder_cache_key,
)
from models.elder import Elder
from models.health import HealthRecord


# ============================================================
# 1. 鉴权上下文 —— 【MCP 改造】实现迁移到 core/agent_context.py
#    (contextvars 存储 + HMAC 令牌编解码都在那里;该模块被 fastap2 与
#     mcp_servers 两个进程共用:fastap2 侧 set+签名,MCP 侧验签+重建)
#    这里保留原名导出,路由层 import 路径不变。
# ============================================================
from core.agent_context import (  # noqa: E402,F401
    clear_access_context,
    get_access_context,
    set_access_context,
)


# ============================================================
# 2. 敏感信息规则（PII 正则 / 字段名 / 意图词）
# ============================================================
# PII 正则：命中即脱敏（无权限时）
PII_PATTERNS: List[Tuple[re.Pattern, str]] = [
    (re.compile(r"(?<!\d)\d{17}[\dXx](?!\d)"), "身份证号"),
    (re.compile(r"(?<!\d)1[3-9]\d{9}(?!\d)"), "手机号"),
    (re.compile(r"\b0\d{2,3}-?\d{7,8}\b"), "座机电话"),
    (re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+"), "邮箱"),
]

# 强敏感字段名：结果中出现这些 → 无权限时整条数据阻断（不给 model 原文）
STRONG_SENSITIVE_FIELDS = ("身份证", "id_card", "完整病史", "病历", "医保")

# 一般敏感字段名：无权限时脱敏处理（弱敏感）
WEAK_SENSITIVE_FIELDS = ("联系电话", "phone", "家庭住址", "住址", "address", "手机号")

# 意图级敏感词：用户消息请求这些内容 → 入口阻断
SENSITIVE_INTENT_KEYWORDS = (
    "身份证",
    "手机号",
    "联系电话",
    "家庭住址",
    "住址",
    "医保",
    "完整病史",
    "病历明细",
    "隐私",
)


def _mask_pii(text: str) -> str:
    """对文本中的 PII 做脱敏（保持格式可读但不泄露原文）。"""
    def mask_phone(m: re.Match) -> str:
        s = m.group(0)
        return s[:3] + "****" + s[7:] if len(s) >= 11 else s[:3] + "****"

    def mask_idcard(m: re.Match) -> str:
        s = m.group(0)
        return s[:6] + "********" + s[-4:] if len(s) >= 15 else "********"

    def mask_landline(m: re.Match) -> str:
        s = m.group(0)
        return s[:3] + "****" + s[-2:]

    def mask_email(m: re.Match) -> str:
        s = m.group(0)
        at = s.find("@")
        return s[:3] + "***" + s[at:] if at > 3 else "***" + s[at:]

    text = PII_PATTERNS[0][0].sub(mask_idcard, text)
    text = PII_PATTERNS[1][0].sub(mask_phone, text)
    text = PII_PATTERNS[2][0].sub(mask_landline, text)
    text = PII_PATTERNS[3][0].sub(mask_email, text)
    return text


# ============================================================
# 3. DBAccessMiddleware —— 唯一允许访问数据库的组件
# ============================================================
class DBAccessMiddleware:
    """数据库访问网关（单例）。

    智能体每次调用数据库工具，最终都会执行到这里：
    authorize（鉴权）→ _query_*（只有这里查库）→ 敏感扫描 → JSON 返回。
    """

    _instance: "DBAccessMiddleware | None" = None
    _engine: Any = None
    _engine_lock = threading.Lock()

    DB_TOOLS = ("query_elder_profile", "query_health_records")

    @classmethod
    def get(cls) -> "DBAccessMiddleware":
        if cls._instance is None:
            cls._instance = DBAccessMiddleware()
        return cls._instance

    # ---- 统一入口：每次 DB 调用都经过这里 ----
    def execute(self, tool_name: str, args: dict) -> str:
        """【鉴权】网关统一执行入口，返回 JSON 字符串（智能体服务层标准格式）。"""
        ctx = get_access_context()

        # 1) 鉴权（每次调用都执行；无权限直接拒绝，不触碰数据库）
        err = self._authorize(tool_name, args, ctx)
        if err:
            # 【审计】拒绝必须留痕:谁(user_id)、试了什么工具、为什么被拒 —— 原实现静默返回,
            # 被人试探越权完全无感知。审计失败不影响拒绝本身的返回(write_audit 内部吞异常)。
            write_audit(ctx.user_id, f"gateway:{tool_name}",
                        resource=f"elder:{(args or {}).get('elder_id', '-')}",
                        decision=False, reason=err)
            return json.dumps({"code": 403, "msg": err}, ensure_ascii=False)

        # 2) 查询（只有网关内部持有 DB engine）
        try:
            if tool_name == "query_elder_profile":
                raw = self._query_profile(args.get("elder_id"))
            elif tool_name == "query_health_records":
                raw = self._query_health(args.get("elder_id"), args.get("metric_type"))
            else:
                return json.dumps(
                    {"code": 400, "msg": f"网关不支持的数据库工具: {tool_name}"},
                    ensure_ascii=False,
                )
        except Exception as exc:  # 异常不外泄细节给 model
            return json.dumps(
                {"code": 500, "msg": "数据库查询失败（详情已脱敏，请联系管理员查看日志）"},
                ensure_ascii=False,
            )

        # 3) 敏感扫描：强敏感且无权限 → 阻断；否则脱敏
        blocked, safe_text = self._scan_sensitive(raw, ctx)
        if blocked:
            return json.dumps(
                {"code": 451, "msg": "检测到敏感信息（身份证/完整病史等），已阻断，不进入智能体链路"},
                ensure_ascii=False,
            )

        # 4) JSON 化返回给智能体服务层
        # 【审计】放行也留痕(医疗场景:谁在何时看了哪个老人的数据必须可追溯)
        write_audit(ctx.user_id, f"gateway:{tool_name}",
                    resource=f"elder:{(args or {}).get('elder_id', '-')}",
                    decision=True, reason="ok")
        return json.dumps(
            {"code": 0, "data": {"tool": tool_name, "result": safe_text}},
            ensure_ascii=False,
        )

    # ---- 鉴权 ----
    def _authorize(self, tool_name: str, args: dict, ctx: _AccessContext) -> Optional[str]:
        """鉴权失败返回错误文案；通过返回 None。"""
        # 【安全】未注入鉴权上下文（未登录/非正规入口）→ 一律拒绝数据库访问
        if not ctx.authenticated:
            return "未注入鉴权上下文，拒绝访问数据库（请先登录）"

        # 工具白名单（默认四工具；外部可注入 allowed_tools 收紧）
        allowed_tools = ctx.permissions.get("allowed_tools")
        if allowed_tools is not None and tool_name not in allowed_tools:
            return f"无权调用工具 {tool_name}（不在白名单内）"

        # 老人数据类工具：必须有 elder_id，且不得越权访问其他老人
        if tool_name in ("query_elder_profile", "query_health_records"):
            elder_id = (args or {}).get("elder_id")
            if not elder_id:
                return "缺少老人编号 elder_id，无法查询"
            allowed_ids = ctx.permissions.get("allowed_elder_ids")
            if allowed_ids is not None and elder_id not in allowed_ids:
                return f"无权访问老人 {elder_id} 的数据（超出授权范围）"
        return None

    # ---- 查询（仅网关内部调用；DB 逻辑从 backend_tools 迁移至此） ----
    def _get_engine(self) -> Any:
        if DBAccessMiddleware._engine is None:
            with DBAccessMiddleware._engine_lock:  # 双检锁，单例连接池
                if DBAccessMiddleware._engine is None:
                    user = os.getenv("MYSQL_USER", "root")
                    pwd = os.getenv("MYSQL_PASSWORD", "")
                    host = os.getenv("MYSQL_HOST", "127.0.0.1")
                    port = os.getenv("MYSQL_PORT", "3306")
                    db = os.getenv("MYSQL_DATABASE", "")
                    url = f"mysql+pymysql://{user}:{pwd}@{host}:{port}/{db}?charset=utf8mb4"
                    DBAccessMiddleware._engine = create_engine(url, pool_pre_ping=True)
        return DBAccessMiddleware._engine

    def _query_profile(self, elder_id: Optional[str]) -> str:
        """【缓存·Cache-Aside 完整示例】老人档案查询的三步走:

        1) 读缓存:命中(含空值哨兵)直接返回,不碰 MySQL
        2) 未命中 → 查库
        3) 结果写回缓存:有数据 TTL=300s(策略1 被动过期);
           无数据写 NULL_SENTINEL TTL=60s(【防穿透】:查不存在的 id 也不会每次打到库)
        一致性由【策略2 写后删】保证:routers/elder.py 的增/改/删会调 invalidate_elder_cache。
        """
        key = elder_cache_key(elder_id)
        cached = cache_get_json_sync(key)
        if cached == NULL_SENTINEL:            # 缓存了"确实没有这条记录"
            return "档案里没有该老人的记录"
        if cached is not None:                 # 缓存了档案文本
            return cached

        with Session(self._get_engine()) as session:
            elder = session.execute(
                select(Elder).where(Elder.elder_id == elder_id)
            ).scalar_one_or_none()
            if elder is None:
                cache_set_json_sync(key, NULL_SENTINEL, ttl=60)   # 空结果短缓存,防穿透
                return "档案里没有该老人的记录"
            text = (
                f"姓名：{elder.name}；年龄：{elder.age}；性别：{elder.gender}；"
                f"自理状态：{elder.health_tag}；慢性病：{elder.chronic_disease}；联系电话：{elder.phone}"
            )
            cache_set_json_sync(key, text, ttl=300)               # 正常结果 5 分钟
            return text

    @staticmethod
    def _format_record(r) -> str:
        # 【健壮性】部分历史数据可能没有 record_time，显式兜底为“时间未知”
        t = r.record_time if r.record_time is not None else "时间未知"
        return f"{t} {r.metric_type}={r.value}"

    def _query_health(self, elder_id: Optional[str], metric_type: Optional[str] = None) -> str:
        with Session(self._get_engine()) as session:
            stmt = select(HealthRecord).where(HealthRecord.elder_id == elder_id)
            if metric_type:
                stmt = stmt.where(HealthRecord.metric_type == metric_type)
            stmt = stmt.order_by(desc(HealthRecord.record_time)).limit(10)
            rows = session.execute(stmt).scalars().all()
            if not rows:
                return "档案里没有该老人的健康监测记录"
            return "；".join(self._format_record(r) for r in rows)

    # ---- 敏感扫描 ----
    def _scan_sensitive(self, text: str, ctx: _AccessContext) -> Tuple[bool, str]:
        """返回 (是否阻断, 处理后的文本)。无权限时强敏感字段 → 阻断，其余脱敏。"""
        if not text:
            return False, text
        # 强敏感字段名命中且无权限 → 阻断（不给 model 原文）
        if not ctx.allow_sensitive:
            for f in STRONG_SENSITIVE_FIELDS:
                if f in text:
                    return True, text
        # 弱敏感：无权限 → 脱敏；有权限 → 原样返回
        if ctx.allow_sensitive:
            return False, text
        masked = _mask_pii(text)
        return False, masked


# ============================================================
# 4. SensitiveGuard —— 入口级敏感阻断（LLM 链路之前）
# ============================================================
class SensitiveGuard:
    """入口守卫：智能体链路启动前阻断敏感请求，避免 model 识别到敏感信息。"""

    @staticmethod
    def check(message: str, permissions: Optional[dict]) -> Optional[dict]:
        """返回 None = 放行；返回 dict = 阻断响应（不进入智能体链路）。"""
        if not message:
            return None
        allow_sensitive = bool((permissions or {}).get("allow_sensitive"))
        if allow_sensitive:
            return None
        hits = [kw for kw in SENSITIVE_INTENT_KEYWORDS if kw in message]
        if hits:
            return {
                "code": 403,
                "msg": f"检测到敏感信息请求（{', '.join(hits)}），已阻断，不进入智能体链路",
            }
        return None


# ============================================================
# 5. AgentSecurityService —— 安全服务门面（对外唯一调用入口）
# ============================================================
class AgentSecurityService:
    """统一安全服务门面：外部（HTTP 路由 / 智能体服务层）只调用本类的方法，
    所有鉴权 / 敏感阻断 / DB 网关逻辑均收敛在 middleware.py 内部。
    """

    @staticmethod
    def build_permissions(
        current_user: Any,
        allowed_elder_ids: Optional[list] = None,
        allowed_tools: Optional[list] = None,
    ) -> dict:
        """【鉴权】由登录用户 + 请求约束生成权限上下文。

        - admin：allow_sensitive=True（可看敏感字段、可访问全部范围）
        - common：allow_sensitive=False（敏感信息一律脱敏/阻断）
        - allowed_elder_ids / allowed_tools 可进一步收紧范围
        """
        user_role = getattr(current_user, "user_role", "common")
        permissions: dict = {
            "user_role": user_role,
            "user_id": str(getattr(current_user, "user_id", "") or ""),  # 【审计】随权限上下文携带操作者
            "allow_sensitive": user_role == "admin",
        }
        if allowed_elder_ids is not None:
            permissions["allowed_elder_ids"] = allowed_elder_ids
        if allowed_tools is not None:
            permissions["allowed_tools"] = allowed_tools
        return permissions

    @staticmethod
    def guard_message(message: str, permissions: dict) -> Optional[dict]:
        """【安全】入口敏感阻断：返回 None=放行；dict=阻断（不进入智能体链路）。"""
        return SensitiveGuard.check(message, permissions)

    # ------------------------------------------------------------
    # 【零信任·第②步】请求校验与安全清洗（网关层，瞬时零 token）
    # ------------------------------------------------------------
    # Prompt 注入倾向：规则级匹配（角色覆盖/泄露系统提示/越权指令）。
    # 不用 LLM 判定——网关必须毫秒级；语义级防御交给链路内中间件纵深。
    _INJECTION_PATTERNS = (
        r"忽略(之前|以上|前面).{0,12}(指令|要求|设定|提示词?)",
        r"(无视|不管)(之前|以上|前面).{0,12}(设定|规则|提示词?)",
        r"disregard.{0,20}(previous|above|instructions)",
        r"(泄露|打印|显示|告诉我).{0,10}(系统提示|system\s*prompt|你的设定|内部指令)",
        r"你(现在)?(是|扮演)(管理员|root|admin|开发者模式|developer\s*mode)",
        r"(关闭|跳过|绕过|无视)(安全|权限|审查|鉴权)(限制|规则|检查)?",
        r"(jailbreak|DAN\s*mode)",
    )

    @staticmethod
    def scan_injection(message: str) -> Optional[str]:
        """注入倾向扫描：命中返回违规说明（403 零 token 中断），None=放行。"""
        import re

        text = message or ""
        for pat in AgentSecurityService._INJECTION_PATTERNS:
            if re.search(pat, text, re.IGNORECASE):
                return f"消息包含疑似提示注入内容(规则:{pat[:20]}…)"
        return None

    # 老人编号格式白名单（防探测：参数化防执行，格式校验防枚举扫描）
    _ELDER_ID_RE = re.compile(r"^[Ee]\d{1,6}$")

    @staticmethod
    def validate_identifiers(elder_id: Optional[str]) -> Optional[str]:
        """标识符格式校验：不合法返回错误说明（422），None=放行。"""
        if not elder_id:
            return None
        if not AgentSecurityService._ELDER_ID_RE.match(str(elder_id).strip()):
            return "老人编号格式不合法(应为 E+数字,如 E001)"
        return None

    @staticmethod
    def build_permissions_from_identity(ctx) -> dict:
        """TrustedContext → MCP 网关兼容的 permissions dict（claims 对齐，网关零改动）。"""
        return {
            "user_role": ctx.user_role,
            "user_id": ctx.user_id,
            "allow_sensitive": ctx.user_role == "admin",
            "allowed_elder_ids": ctx.allowed_elders,
            "allowed_tools": ctx.allowed_tools,
            "identity_id": ctx.identity_id,
        }

    @staticmethod
    def enter_context(permissions: dict) -> None:
        """【鉴权】注入 contextvars 鉴权上下文（供 DBAccessMiddleware 每次 DB 调用读取）。"""
        user_role = (permissions or {}).get("user_role") or "common"
        allow_sensitive = bool((permissions or {}).get("allow_sensitive", False))
        user_id = str((permissions or {}).get("user_id") or "")
        set_access_context(permissions, user_role, allow_sensitive, user_id)

    @staticmethod
    def exit_context() -> None:
        """【鉴权】清理线程级上下文，防止线程复用导致权限串号。"""
        clear_access_context()

    @staticmethod
    def query_elder_profile_safe(elder_id: str) -> str:
        """【服务】查询老人档案（经网关：鉴权 → 查询 → 敏感扫描 → JSON 返回）。"""
        return DBAccessMiddleware.get().execute("query_elder_profile", {"elder_id": elder_id})

    @staticmethod
    def query_health_records_safe(elder_id: str, metric_type: Optional[str] = None) -> str:
        """【服务】查询老人健康记录（经网关：鉴权 → 查询 → 敏感扫描 → JSON 返回）。"""
        return DBAccessMiddleware.get().execute(
            "query_health_records", {"elder_id": elder_id, "metric_type": metric_type}
        )
