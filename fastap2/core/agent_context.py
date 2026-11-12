"""跨进程权限上下文:contextvars 存储 + HMAC 签名令牌(编解码)。

这个模块被两个进程共用(fastap2 主服务 与 mcp_servers 数据服务),所以放在 core:
- fastap2 侧:set_access_context 注入请求权限 → make_context_token 每次工具调用签名
- MCP 侧:verify_context_token 验签 → 重建权限上下文 → ACL 校验

【安全模型】agent_context 是一个 60 秒时效的"能力令牌"(capability token):
- 无令牌 / 签名不符 / 过期 → MCP 端 fail-closed 拒绝
- HMAC-SHA256 防伪造;compare_digest 防时序攻击
- 重放窗口 60 秒 + 仅内网;生产可再加 nonce 去重(Redis)
"""
import base64
import contextvars
import hashlib
import hmac
import json
import os
import time
from typing import Any, Dict, Optional

TOKEN_TTL_SECONDS = 60


# ============================================================
# 进程内权限上下文(contextvars 版,替代 threading.local)
# ============================================================
class _AccessContext:
    __slots__ = ("permissions", "user_role", "allow_sensitive", "user_id", "authenticated")

    def __init__(
        self,
        permissions: Optional[dict] = None,
        user_role: str = "common",
        allow_sensitive: bool = False,
        user_id: str = "",
        authenticated: bool = True,
    ) -> None:
        self.permissions: Dict[str, Any] = permissions or {}
        self.user_role: str = user_role or "common"
        self.allow_sensitive: bool = bool(allow_sensitive)
        self.user_id: str = user_id or ""
        # authenticated=False 表示"没有注入鉴权上下文"(未登录/非正规入口),一律拒绝
        self.authenticated: bool = authenticated


_ACCESS_CTX: contextvars.ContextVar = contextvars.ContextVar("access_ctx", default=None)


def set_access_context(
    permissions: Optional[dict] = None,
    user_role: str = "common",
    allow_sensitive: bool = False,
    user_id: str = "",
) -> None:
    _ACCESS_CTX.set(_AccessContext(permissions, user_role, allow_sensitive, user_id))


def clear_access_context() -> None:
    _ACCESS_CTX.set(None)


def get_access_context() -> _AccessContext:
    ctx = _ACCESS_CTX.get()
    if ctx is None:
        return _AccessContext(None, "common", False, "", authenticated=False)
    return ctx


# ============================================================
# HMAC 签名令牌
# ============================================================
def _secret() -> bytes:
    s = os.getenv("AGENT_CONTEXT_SECRET") or os.getenv("SECRET_KEY") or ""
    if not s:
        raise RuntimeError("AGENT_CONTEXT_SECRET / SECRET_KEY 未配置,无法签名权限上下文")
    return s.encode()


def _b64e(b: bytes) -> str:
    return base64.urlsafe_b64encode(b).decode().rstrip("=")


def _b64d(s: str) -> bytes:
    return base64.urlsafe_b64decode(s + "=" * (-len(s) % 4))


def make_context_token(ctx: _AccessContext, ttl: int = TOKEN_TTL_SECONDS) -> str:
    """把当前请求的权限上下文签成短时效令牌(MCP 调用时随参数携带)。"""
    now = int(time.time())
    payload = {
        "user_id": str(ctx.user_id or ""),
        "user_role": ctx.user_role or "common",
        "allow_sensitive": bool(ctx.allow_sensitive),
        # None=不限定(admin);list(可能为空)=白名单;空列表=全禁
        "allowed_elder_ids": (ctx.permissions or {}).get("allowed_elder_ids"),
        "allowed_tools": (ctx.permissions or {}).get("allowed_tools"),
        "iat": now,
        "exp": now + ttl,
    }
    body = _b64e(json.dumps(payload, separators=(",", ":")).encode())
    sig = _b64e(hmac.new(_secret(), body.encode(), hashlib.sha256).digest())
    return f"{body}.{sig}"


def verify_context_token(token: str) -> Optional[dict]:
    """验签并返回 payload;无效/过期返回 None(调用方必须 fail-closed)。"""
    if not token or "." not in token:
        return None
    try:
        body, sig = token.rsplit(".", 1)
        expect = hmac.new(_secret(), body.encode(), hashlib.sha256).digest()
        if not hmac.compare_digest(_b64d(sig), expect):
            return None
        payload = json.loads(_b64d(body))
        if int(payload.get("exp", 0)) < time.time():
            return None
        return payload
    except Exception:
        return None
