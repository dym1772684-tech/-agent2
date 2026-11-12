"""系统体检：一条命令告诉你「这个项目现在能跑到第几层」。

用法（在 fastap2 目录下）：
    python verify_system.py

分层与判读（对应「一条消息怎么走」的 12 站）：
    L1 代码层  模块能不能导、守卫函数齐不齐、路由挂了几条
    L2 配置层  必需 env 是否齐全（只打印名字，绝不打印值）
    L3 依赖层  Redis / MySQL / MCP / LLM 端点是否可达
    L4 装配层  bootstrap() 能不能把图编译出来（tools_count / topology / persistent）
    L5 链路层  日志与账本落点是否可写

退出码：0 = 全部通过；1 = 有失败项（警告不算失败）。
"""
from __future__ import annotations

import os
import socket
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))
sys.path.insert(0, str(BASE.parent))

OK, WARN, FAIL = "OK", "WARN", "FAIL"
_rows: list[tuple[str, str, str]] = []


def rec(level: str, item: str, detail: str = "") -> None:
    _rows.append((level, item, detail))
    tag = {OK: "[ OK ]", WARN: "[WARN]", FAIL: "[FAIL]"}[level]
    print(f"{tag} {item:44s} {detail}")


def section(title: str) -> None:
    print(f"\n--- {title} " + "-" * max(0, 56 - len(title)))


def port_open(host: str, port: int, timeout: float = 1.0) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except Exception:
        return False


# ============================================================
# L1 代码层
# ============================================================
section("L1 代码层")

try:
    import main  # noqa: F401

    routes = {getattr(r, "path", "") for r in main.app.routes}
    rec(OK, "import main", f"{len(routes)} routes")
except Exception as exc:
    rec(FAIL, "import main", f"{type(exc).__name__}: {exc}")
    routes = set()

EXPECTED_ROUTES = (
    "/api/v1/agent/chat",
    "/api/v1/agent/chat/stream",
    "/api/v1/agent/interrupt/{thread_id}",
    "/api/v1/agent/resume",
    "/api/v1/agent/status",
    "/api/v1/agent/trace/{thread_id}",
    "/api/v1/agent/files",
    "/api/v1/skills",
    "/api/v1/health/records/{elder_id}",
    "/api/v1/health/stats/{elder_id}",
    "/api/v1/health/risk/{elder_id}",
    "/metrics",
)
missing = [p for p in EXPECTED_ROUTES if p not in routes]
if not missing:
    rec(OK, "关键路由齐全", f"{len(EXPECTED_ROUTES)} 条")
else:
    rec(FAIL, "关键路由缺失", ", ".join(missing))

try:
    from allagent2.agent import middleware as mw

    symbols = ("build_prompt", "check_emergency", "check_identity",
               "check_slot", "check_tool_permission", "mask_pii")
    absent = [s for s in symbols if not hasattr(mw, s)]
    if absent:
        rec(FAIL, "守卫纯函数", f"缺 {absent}（react 引擎会 ImportError）")
    else:
        rec(OK, "守卫纯函数", "6/6")
except Exception as exc:
    rec(FAIL, "导入 allagent2.agent.middleware", f"{type(exc).__name__}: {exc}")

try:
    import agent_bridge

    need = ("run_agent", "run_agent_stream", "resume_agent", "set_abort", "get_status")
    absent = [n for n in need if not hasattr(agent_bridge, n)]
    rec(OK if not absent else FAIL, "agent_bridge 导出",
        "齐全" if not absent else f"缺 {absent}")
except Exception as exc:
    rec(FAIL, "import agent_bridge", f"{type(exc).__name__}: {exc}")

# ============================================================
# L2 配置层
# ============================================================
section("L2 配置层")

REQUIRED = ("OPENAI_API_KEY", "OPENAI_MODEL", "OPENAI_BASE_URL",
            "SECRET_KEY", "MYSQL_USER", "MYSQL_PASSWORD", "MYSQL_DATABASE",
            "REDIS_HOST")
try:
    from agent_bridge.bootstrap import _load_envs

    _load_envs()
    rec(OK, "加载 env（3 份）")
except Exception as exc:
    rec(WARN, "加载 env", f"{type(exc).__name__}: {exc}")

for name in REQUIRED:
    value = os.getenv(name)
    if value:
        rec(OK, f"env {name}")
    else:
        rec(FAIL, f"env {name}", "未配置")

topology = (os.getenv("AGENT_TOPOLOGY") or "single").strip().lower()
engine = (os.getenv("AGENT_ENGINE") or "react").strip().lower()
rec(OK if topology in ("single", "team") else WARN,
    "拓扑", f"{topology} / engine={engine}")

# ============================================================
# L3 依赖层
# ============================================================
section("L3 依赖层")

redis_host, redis_port = os.getenv("REDIS_HOST", "127.0.0.1"), int(os.getenv("REDIS_PORT", "6379"))
if port_open(redis_host, redis_port):
    rec(OK, "Redis 端口", f"{redis_host}:{redis_port}")
else:
    rec(FAIL, "Redis 不可达", f"{redis_host}:{redis_port} → 会话不持久、幂等与限流降级")

mysql_host, mysql_port = os.getenv("MYSQL_HOST", "127.0.0.1"), int(os.getenv("MYSQL_PORT", "3306"))
if port_open(mysql_host, mysql_port):
    rec(OK, "MySQL 端口", f"{mysql_host}:{mysql_port}")
else:
    rec(FAIL, "MySQL 不可达", f"{mysql_host}:{mysql_port} → 登录/权限/审计全部不可用")

for name, default in (("MCP_ELDER_URL", 8001), ("MCP_LIFE_URL", 8002),
                      ("MCP_KNOWLEDGE_URL", 8003)):
    url = os.getenv(name)
    if not url:
        rec(WARN, f"{name}", "未配置")
        continue
    try:
        host = url.split("//", 1)[1].split("/", 1)[0]
        h, p = host.rsplit(":", 1)
        reachable = port_open(h, int(p))
    except Exception:
        reachable, h, p = False, "?", str(default)
    rec(OK if reachable else WARN, f"MCP {name}",
        f"{h}:{p}" + ("" if reachable else " 不可达（该服务工具会缺失，链路仍可跑）"))

llm_base = os.getenv("OPENAI_BASE_URL") or ""
if llm_base:
    rec(OK, "LLM base_url", llm_base)
else:
    rec(WARN, "LLM base_url", "未配置（将走 SDK 默认端点）")

# ============================================================
# L4 装配层
# ============================================================
section("L4 装配层")

if not (os.getenv("MYSQL_USER") and os.getenv("SECRET_KEY")):
    rec(WARN, "bootstrap 跳过", "配置不全，启动必然失败")
else:
    try:
        import agent_bridge

        status = agent_bridge.get_status()
        if status.get("ready"):
            rec(OK, "bootstrap", f"ready tools={status.get('tools_count')} "
                                 f"topology={status.get('topology')} "
                                 f"persistent={status.get('persistent')}")
        elif status.get("error"):
            rec(FAIL, "bootstrap 失败", str(status.get("error"))[:120])
        else:
            rec(WARN, "bootstrap 未执行", "首次 /chat 或手动 bootstrap() 时装配")
    except Exception as exc:
        rec(FAIL, "bootstrap 异常", f"{type(exc).__name__}: {str(exc)[:120]}")

# ============================================================
# L5 链路层
# ============================================================
section("L5 链路层")

agent_dir = BASE.parent / "allagent2"
logs = agent_dir / "guance" / "logs"
if logs.exists():
    files = {p.name: p.stat().st_size for p in logs.glob("*.jsonl")}
    rec(OK, "日志目录", f"{logs}  jsonl={list(files)}" if files else f"{logs}（暂无 jsonl）")
    for name in ("agent.log", "token_usage.jsonl", "trace_team.jsonl"):
        p = logs / name
        if p.exists():
            rec(OK, f"落点 {name}", f"{p.stat().st_size} bytes")
        else:
            rec(WARN, f"落点 {name}", "尚未生成（跑一轮对话后出现）")
else:
    rec(WARN, "日志目录不存在", str(logs))

# ============================================================
# 汇总
# ============================================================
n_fail = sum(1 for lv, _, _ in _rows if lv == FAIL)
n_warn = sum(1 for lv, _, _ in _rows if lv == WARN)
n_ok = sum(1 for lv, _, _ in _rows if lv == OK)
print("\n" + "=" * 64)
print(f"OK {n_ok} | WARN {n_warn} | FAIL {n_fail}")
print("=" * 64)
if n_fail:
    print("失败项就是当前链路断掉的地方：从最上面一条开始修。")
sys.exit(1 if n_fail else 0)
