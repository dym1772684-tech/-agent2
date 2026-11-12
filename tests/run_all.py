"""全自动化测试总入口 —— 一条命令跑完项目全部功能。

用法（在 tests/ 目录）:
    E:/python/python.exe run_all.py            # 全量（含 vLLM 真链路，约 2~4 分钟）
    E:/python/python.exe run_all.py --quick    # 跳过 LLM 真链路组（约 30~60 秒）

自动化内容:
    1. 自动拉起 MCP 双服务（8001/8002，已监听则复用）
    2. 主应用 TestClient 进程内启动（不用手动起 uvicorn）
    3. 分组执行:
       t01 零依赖单元   t02 基础设施   t03 安全网关   t04 认证+CRUD
       t05 ACL/审计     t06 智能体链路  t07 SSE 流式   t08 RAG   t09 记忆系统
       + allagent2/tests/test_memory.py（35 项零 LLM 单测）
    4. 依赖不可用的分组自动 SKIP（Redis/MySQL/vLLM/MCP/admin 账号）
    5. 汇总报告打印 + 写入 tests/report_*.txt，有 FAIL 时退出码 1

前置条件（各分组按需，缺什么 SKIP 什么）:
    Redis / MySQL 已启动；vLLM 跑真链路才需要；admin 账号用 create_admin.py 建。
"""
from __future__ import annotations

import importlib
import io
import socket
import subprocess
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _lib import Skip  # noqa: E402  （导入即完成 sys.path 装配 + chdir fastap2）
import _lib  # noqa: E402

QUICK = "--quick" in sys.argv
LOG_DIR = Path(__file__).resolve().parent


# ============================================================
# MCP 双服务拉起
# ============================================================
def _port_open(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        return s.connect_ex(("127.0.0.1", port)) == 0


def ensure_mcp() -> list[tuple[str, bool]]:
    """确保 MCP elder(8001)/life(8002) 在跑。返回 [(name, started_here)]。"""
    out = []
    for name, port, module in (("elder", 8001, "mcp_servers.elder_server"),
                               ("life", 8002, "mcp_servers.life_server")):
        if _port_open(port):
            out.append((f"mcp-{name}", False))
            continue
        subprocess.Popen(
            [sys.executable, "-m", module],
            cwd=str(_lib.FASTAP2), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        out.append((f"mcp-{name}", True))
    # 等端口就绪
    for _ in range(30):
        if _port_open(8001) and _port_open(8002):
            break
        time.sleep(0.5)
    return out


# ============================================================
# runner
# ============================================================
GROUPS = [
    ("t01 零依赖单元", "t01_unit"),
    ("t02 基础设施", "t02_infra"),
    ("t03 安全网关", "t03_gateway"),
    ("t04 认证+CRUD", "t04_auth_crud"),
    ("t05 ACL/审计", "t05_acl"),
    ("t09 记忆系统", "t09_memory"),
    ("t08 RAG检索", "t08_rag"),
]
LLM_GROUPS = [  # 需要 vLLM；--quick 时跳过
    ("t06 智能体链路(含真问答)", "t06_agent"),
    ("t07 SSE流式", "t07_stream"),
]


def run_module(title: str, mod_name: str, out: io.StringIO):
    """执行单个测试模块，返回 [(name, ok, detail)]；异常转为 FAIL 行。

    兼容两种模块接口：
    - 有 run() -> list[(name, ok, detail)]  的项目套件模块
    - unittest 结构（如 allagent2/tests/test_memory.py）→ TestLoader 全量执行
    """
    print(f"\n■ {title}")
    try:
        mod = importlib.import_module(mod_name)
        if hasattr(mod, "run"):
            importlib.reload(mod)  # 支持单独重跑
            return mod.run()
        # ---- unittest 兜底 ----
        import unittest

        suite = unittest.defaultTestLoader.loadTestsFromModule(mod)
        runner = unittest.TextTestRunner(verbosity=0, stream=io.StringIO())
        result = runner.run(suite)
        return [(str(t), not failure, (failure[1] if failure else "").splitlines()[-1][:70]
                 if failure else "") for t, failure in
                zip(result.tests, [f for _, f in result.failures + result.errors]
                    + [""] * (len(result.tests) - len(result.failures + result.errors)))]
    except Skip as e:
        print(f"  [SKIP] 整组跳过: {e}")
        return [(f"{mod_name}(整组)", None, str(e))]
    except Exception as e:
        tb = traceback.format_exc(limit=3)
        out.write(f"\n### {title} 崩溃\n{tb}\n")
        print(f"  [FAIL] 整组崩溃: {type(e).__name__}: {e}")
        return [(f"{mod_name}(整组)", False, f"{type(e).__name__}: {str(e)[:60]}")]


def main() -> int:
    started = time.time()
    print("=" * 64)
    print("养老智能体 · 全功能自动化测试套件")
    print(f"时间: {datetime.now():%Y-%m-%d %H:%M:%S}   模式: {'QUICK' if QUICK else 'FULL'}")
    print("=" * 64)

    crash_log = io.StringIO()

    # 1) MCP 拉起
    mcp_started = ensure_mcp()
    print(f"\n[MCP] elder/life 状态: {mcp_started}")

    # 2) 零依赖组（不需要主应用）
    all_results: list[tuple[str, str, bool, str]] = []  # (group, name, ok, detail)

    def record(group, rs):
        for name, ok, detail in rs:
            all_results.append((group, name, ok, detail))

    for title, mod in GROUPS:
        record(title, run_module(title, mod, crash_log))

    # 3) 老单测套件（35 项零 LLM）
    record("unit: 记忆系统35项", run_module("unit: allagent2/tests/test_memory.py",
                                          "allagent2.tests.test_memory", crash_log))

    # 4) 需要真链路的组（client() 会自动 open_app，与上面 API 组共用同一实例）
    for title, mod in (LLM_GROUPS if not QUICK else []):
        record(title, run_module(title, mod, crash_log))

    # 5) 汇总
    n_pass = sum(1 for *_, ok, _ in all_results if ok is True)
    n_fail = sum(1 for *_, ok, _ in all_results if ok is False)
    n_skip = sum(1 for *_, ok, _ in all_results if ok is None)
    total = time.time() - started

    print("\n" + "=" * 64)
    print(f"汇总: PASS {n_pass} | FAIL {n_fail} | SKIP {n_skip}   耗时 {total:.0f}s")
    print("=" * 64)
    cur_group = None
    for group, name, ok, detail in all_results:
        if group != cur_group:
            cur_group = group
            print(f"\n— {group}")
        tag = "PASS" if ok is True else "FAIL" if ok is False else "SKIP"
        print(f"  [{tag}] {name}" + (f"  {detail}" if detail else ""))

    # 6) 报告落盘
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report = LOG_DIR / f"report_{stamp}.txt"
    with io.open(report, "w", encoding="utf-8") as f:
        f.write(f"全功能自动化测试报告  {datetime.now():%Y-%m-%d %H:%M:%S}  "
                f"mode={'QUICK' if QUICK else 'FULL'}\n")
        f.write(f"PASS {n_pass} | FAIL {n_fail} | SKIP {n_skip} | {total:.0f}s\n\n")
        for group, name, ok, detail in all_results:
            tag = "PASS" if ok is True else "FAIL" if ok is False else "SKIP"
            f.write(f"[{tag}] [{group}] {name}" + (f"  {detail}" if detail else "") + "\n")
        f.write(crash_log.getvalue())
    print(f"\n报告已写入: {report}")

    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
