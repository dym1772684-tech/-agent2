"""全自动测试套件 · 公共工具库。

职责：
- sys.path 装配（项目根 + fastap2），所有 t*.py 只需 `import _lib`
- 统一的用例登记 / PASS / FAIL / SKIP 判定与报告输出
- 主应用 TestClient 进程内启动（自动触发 lifespan → 连 MCP/Redis/MySQL）
- 登录测试账号拿 JWT；给 data_gateway 造 HMAC 权限令牌

每个测试模块暴露 `run() -> list[tuple[name, ok, detail]]`，
可单独执行（python t04_auth_crud.py）也可被 run_all.py 聚合。
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# ---- 路径装配：tests/ 在项目根，fastap2/allagent2 都要可导入 ----
ROOT = Path(__file__).resolve().parent.parent
FASTAP2 = ROOT / "fastap2"
for p in (str(ROOT), str(FASTAP2)):
    if p not in sys.path:
        sys.path.insert(0, p)

os.chdir(FASTAP2)  # bootstrap 里的相对路径(logs/models)以 fastap2 为工作目录


class Skip(Exception):
    """模块级依赖不可用时抛出，runner 将整组用例标记 SKIP。"""


def load_env():
    """加载 key.env（不经过 main 导入而需要 env 的模块调用，幂等）。"""
    from agent_bridge.bootstrap import _load_envs
    _load_envs()


def probe_http(url: str, timeout: float = 2.0) -> bool:
    """探测 HTTP 服务是否可达（vLLM / MCP / 主服务）。

    有任何 HTTP 响应（含 4xx/5xx）都算"服务在"；
    只有连接拒绝/超时才算不可用。
    """
    import urllib.error
    import urllib.request
    try:
        with urllib.request.urlopen(url, timeout=timeout):
            return True
    except urllib.error.HTTPError:
        return True   # 服务在，只是路径/方法不对
    except Exception:
        return False


# ============================================================
# 用例登记
# ============================================================
_results: list[tuple[str, bool, str]] = []


def check(name: str, ok, detail: str = ""):
    """登记一条用例结果：ok=True/False/None(SKIP)。"""
    _results.append((name, ok if ok is None else bool(ok), detail))
    tag = "SKIP" if ok is None else ("PASS" if ok else "FAIL")
    print(f"  [{tag}] {name}" + (f"  {detail}" if detail else ""))
    return ok


def results() -> list[tuple[str, bool, str]]:
    return list(_results)


# ============================================================
# 主应用（TestClient 进程内，lifespan 存活）
# ============================================================
_client = None


def open_app():
    """打开主应用（触发 lifespan：连 MCP / Redis / 编译 team 图）。

    run_all.py 在 API 阶段开始时调用一次，结束时 close_app()。
    """
    global _client
    if _client is not None:
        return _client
    import main as app_module  # noqa: 导入即完成数据库/设置装配
    from fastapi.testclient import TestClient

    _client = TestClient(app_module.app).__enter__()
    return _client


def close_app():
    global _client
    if _client is not None:
        try:
            _client.__exit__(None, None, None)
        except Exception:
            pass
        _client = None


def client():
    """拿打开状态的主应用客户端（必须先 open_app）。"""
    if _client is None:
        open_app()
    return _client


# ============================================================
# 登录与权限令牌
# ============================================================
TEST_USER = ("testuser01", "test123456")


def login(c, username: str = TEST_USER[0], password: str = TEST_USER[1]):
    """登录拿 (token, user_id)。user_id 从 JWT payload 解出（响应体不含它）。"""
    r = c.post("/api/v1/login", json={"username": username, "password": password})
    if r.status_code != 200:
        return None, None
    body = r.json()
    token = body.get("access_token") or (body.get("data") or {}).get("access_token")
    if not token:
        return None, None
    uid = (body.get("data") or {}).get("user_id") if isinstance(body.get("data"), dict) else None
    if not uid:
        import base64
        import json as _json
        try:
            payload_b64 = token.split(".")[1]
            payload_b64 += "=" * (-len(payload_b64) % 4)  # 补齐 padding
            uid = _json.loads(base64.urlsafe_b64decode(payload_b64)).get("user_id")
        except Exception:
            uid = None
    return token, uid


def auth_header(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"}


# ============================================================
# admin 账号（ACL/审计组用）：探测现有 → 没有就自建临时号
# ============================================================
_ADMIN_CACHE: list = []
_ADMIN_CANDIDATES = [("admin", "admin123"), ("admin", "admin@123"), ("admin", "Admin@123")]


def ensure_admin(c):
    """拿一个可用 admin token：先探测现有账号，失败则临时创建 suite_admin。

    临时账号跑完用 cleanup_admin() 删除，不留后门。
    """
    if _ADMIN_CACHE:
        return _ADMIN_CACHE[0]
    for u, p in _ADMIN_CANDIDATES:
        tok, _ = login(c, u, p)
        if tok:
            _ADMIN_CACHE.append((tok, u, True))  # True=已有账号，不删
            return tok
    # ---- 自建临时 admin（同步 pymysql 引擎，避开 aiomysql 的 loop 绑定）----
    import uuid

    from mcp_servers.data_gateway import _get_engine
    from schemas.user import User as UserSchema
    from core.settings import hash_password
    from sqlalchemy import text as _text

    name, pwd = "suite_admin_tmp", "Suite#Test2026"
    uid = str(uuid.uuid4())
    hashed = hash_password(UserSchema(username=name, password=pwd))
    with _get_engine().begin() as conn:
        conn.execute(_text(
            "INSERT INTO User(username, password, user_id, user_role) "
            "VALUES (:u, :p, :uid, 'admin')"),
            {"u": name, "p": hashed, "uid": uid})
    tok, _ = login(c, name, pwd)
    if tok:
        _ADMIN_CACHE.append((tok, name, False))  # False=临时号，可清理
        return tok
    return None


def cleanup_admin(c):
    """删除自建的临时 admin（探测到的既有账号不动）。"""
    if not _ADMIN_CACHE:
        return
    tok, name, preexisting = _ADMIN_CACHE[0]
    if preexisting:
        return
    from mcp_servers.data_gateway import _get_engine
    from sqlalchemy import text as _text

    try:
        with _get_engine().begin() as conn:
            conn.execute(_text("DELETE FROM User WHERE username = :u"), {"u": name})
    except Exception:
        pass
    _ADMIN_CACHE.clear()


def make_agent_token(user_id: str, elder_ids: list[str],
                     tools: list[str] | None = None, allow_sensitive: bool = False) -> str:
    """给 data_gateway 测试造 HMAC 权限上下文令牌（走项目自己的签名实现）。"""
    from core.agent_context import _AccessContext, make_context_token

    ctx = _AccessContext(
        permissions={"allowed_elder_ids": elder_ids, "allowed_tools": tools},
        user_role="admin" if allow_sensitive else "common",
        allow_sensitive=allow_sensitive,
        user_id=user_id,
        authenticated=True,
    )
    return make_context_token(ctx)
