"""MCP 链路端到端验证（不调 LLM，零 token 成本）。

用法:
    1. 先启动两个 MCP Server（各自独立终端或后台）:
       cd fastap2 && E:/python/python.exe -m mcp_servers.elder_server
       cd fastap2 && E:/python/python.exe -m mcp_servers.life_server
    2. cd fastap2 && E:/python/python.exe e2e_mcp.py

验证项:
    [1] bootstrap 接线成功，工具确实来自 MCP（calculator/get_real_weather/
        query_elder_profile/query_health_records 由远端提供）
    [2] 生活工具直调: calculator 1+2=3（无鉴权要求）
    [3] 鉴权链路: 注入合法上下文 → elder 工具拿到令牌（MySQL 未启动时返回
        网关 500 脱敏信息，也证明「验签→ACL→查库」整条链走通）
    [4] fail-closed: 清空上下文直调 elder 工具 → 403
    [5] 越权: 上下文只允许 E999 却查 E001 → 403
    [6] 跨线程: 在子线程里调用（模拟 langgraph executor）→ 令牌仍正确注入
"""
import sys
import threading
import traceback

results = []


def check(name, ok, detail=""):
    results.append((name, ok, detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}")


def main():
    import agent_bridge
    st = agent_bridge.get_status()
    check("1.bootstrap 接线", st.get("ready") is True, f"status={st}")

    from agent_bridge.mcp_runtime import get_mcp_tools
    tools = {t.name: t for t in get_mcp_tools()}
    expect_mcp = {"calculator", "get_real_weather", "query_elder_profile", "query_health_records"}
    got = expect_mcp & set(tools)
    check("2.MCP 工具注册", len(got) == len(expect_mcp),
          f"远端工具={sorted(got)} 共{len(tools)}个")

    from core.agent_context import clear_access_context, set_access_context

    # [3] 生活工具直调（life_server 无鉴权要求）
    try:
        out = tools["calculator"].invoke({"fuhao": "+", "number1": 1, "number2": 2})
        check("3.calculator 直调", out.strip() in ("3", "3.0"), f"返回={out!r}")
    except Exception as e:
        check("3.calculator 直调", False, repr(e))

    # [4] 鉴权链路: 合法上下文 → elder 工具拿到令牌
    set_access_context(
        permissions={"allowed_elder_ids": ["E001"]},
        user_role="common",
        allow_sensitive=False,
        user_id="e2e-probe",
    )
    try:
        out = tools["query_elder_profile"].invoke({"elder_id": "E001"})
        # MySQL 在线 → code 0 真数据；离线 → code 500 脱敏信息；两者都证明令牌通过验签
        ok = ("403" not in out) and ("权限上下文无效" not in out)
        check("4.elder 工具鉴权链路", ok, f"返回={out[:120]!r}")
    except Exception as e:
        check("4.elder 工具鉴权链路", False, repr(e))

    # [5] fail-closed: 清空上下文 → 403
    clear_access_context()
    try:
        out = tools["query_elder_profile"].invoke({"elder_id": "E001"})
        check("5.未登录 fail-closed", "403" in out or "权限上下文无效" in out, f"返回={out[:100]!r}")
    except Exception as e:
        check("5.未登录 fail-closed", False, repr(e))

    # [6] 越权: 只授权 E999 却查 E001 → 403
    set_access_context(
        permissions={"allowed_elder_ids": ["E999"]},
        user_role="common", allow_sensitive=False, user_id="e2e-probe",
    )
    try:
        out = tools["query_elder_profile"].invoke({"elder_id": "E001"})
        check("6.越权拦截", "403" in out and "超出授权范围" in out, f"返回={out[:100]!r}")
    except Exception as e:
        check("6.越权拦截", False, repr(e))

    # [7] 跨线程令牌传播: 模拟 langgraph executor 的传播方式（copy_context().run）
    holder = {}

    def _thread_call():
        try:
            holder["out"] = tools["query_elder_profile"].invoke({"elder_id": "E001"})
        except Exception as e:
            holder["out"] = repr(e)

    import contextvars
    thread_ctx = contextvars.copy_context()  # 与 langgraph/_executor.py 相同的传播方式
    t = threading.Thread(target=lambda: thread_ctx.run(_thread_call))
    t.start()
    t.join()
    out = holder.get("out", "")
    # 子线程应继承主线程 set 的 E999 上下文 → 越权 403，而不是"令牌无效"
    check("7.跨线程令牌传播", "权限上下文无效" not in out, f"返回={out[:100]!r}")

    failed = [r for r in results if not r[1]]
    print(f"\n==== {len(results) - len(failed)}/{len(results)} 通过 ====")
    if failed:
        for name, _, detail in failed:
            print(f"  失败项: {name} -> {detail}")
        sys.exit(1)


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        traceback.print_exc()
        sys.exit(1)
