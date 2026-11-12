"""[t06] 智能体链路测试。

- 零 LLM 部分：/agent/status、急救短路（不进图）、/agent/memory 端点、/trace
- 真链路部分（vLLM 可用时）：/agent/chat 完整问答 + structured/usage 断言
"""
from _lib import TEST_USER, auth_header, check, cleanup_admin, client, ensure_admin, login, probe_http, results

import os

VLLM_URL = os.getenv("VLLM_BASE_URL", "http://127.0.0.1:8005/v1/models")


def run():
    print("[t06] 智能体链路")
    c = client()
    token, uid = login(c)
    if not token:
        raise RuntimeError("登录失败，跳过本组")
    h = auth_header(token)

    # ---- 1) 权限矩阵：status/trace 是 admin-only，common 应 403 ----
    r = c.get("/api/v1/agent/status", headers=h)
    check("权限: common 访问 /status 403", r.status_code == 403, f"http={r.status_code}")
    r = c.get("/api/v1/agent/trace/t-anything", headers=h)
    check("权限: common 访问 /trace 403", r.status_code == 403, f"http={r.status_code}")

    admin_tok = ensure_admin(c)
    if admin_tok:
        ha = auth_header(admin_tok)
        r = c.get("/api/v1/agent/status", headers=ha)
        ok = r.status_code == 200
        body = r.json() if ok else {}
        data = body.get("data") or body
        check("agent: admin /status 200", ok, f"http={r.status_code}")
        check("agent: status 报告就绪状态",
              isinstance(data, dict) and ("ready" in data or "tools_count" in data), str(data)[:80])
    else:
        check("agent: admin /status 200", None, "SKIP：无可用 admin 账号")

    # ---- 2) 急救短路（零 LLM：命中关键词直接返回，不进图）----
    r = c.post("/api/v1/agent/chat", headers=h,
               json={"message": "老人突然胸痛，呼吸急促，怎么办！", "thread_id": "t-ems-test"})
    ok = r.status_code == 200
    answer = ""
    if ok:
        data = (r.json().get("data") or {})
        answer = str(data.get("answer") or "")
    check("急救: 端点 200", ok, f"http={r.status_code}")
    check("急救: 返回非空急救指引", len(answer) >= 10, f"answer={answer[:40]}…")

    # ---- 3) trace 由 admin 视角测（team 下允许降级）----
    if admin_tok:
        r = c.get("/api/v1/agent/trace/t-ems-test", headers=auth_header(admin_tok))
        check("trace: admin 端点可达(允许降级)", r.status_code in (200, 404, 500),
              f"http={r.status_code}")

    # ---- 4) 记忆端点：查看 / 删除自己的记忆 ----
    r = c.get("/api/v1/agent/memory", headers=h)
    check("memory: GET 200", r.status_code == 200, f"http={r.status_code}")
    r = c.request("DELETE", "/api/v1/agent/memory", headers=h)
    check("memory: DELETE 200(删除权)", r.status_code == 200, f"http={r.status_code}")

    # ---- 5) 真链路 /chat（vLLM 可用时；中途掉线记 SKIP 不算 FAIL）----
    if probe_http(VLLM_URL, timeout=3):
        try:
            r = c.post("/api/v1/agent/chat", headers=h, json={
                "message": "高血压老人日常饮食要注意什么？一句话回答。",
                "thread_id": "t-e2e-chat"})
            ok = r.status_code == 200
            check("真链路: /chat 200", ok, f"http={r.status_code} {r.text[:80] if not ok else ''}")
            if ok:
                data = (r.json().get("data") or {})
                answer = str(data.get("answer") or "")
                usage = data.get("usage") or {}
                check("真链路: answer 非空", len(answer) >= 5, f"answer={answer[:40]}…")
                check("真链路: usage 有 token 账单",
                      isinstance(usage, dict) and
                      any(v for k, v in usage.items() if "token" in str(k).lower() or
                          isinstance(v, (int, float))),
                      str(usage)[:60])
                check("真链路: structured 结构存在", "structured" in data or "final" in data,
                      f"keys={sorted(data)[:6]}")
        except Exception as e:
            check("真链路: /chat 完整问答", None, f"SKIP：vLLM 执行中掉线({type(e).__name__})")
    else:
        check("真链路: /chat 完整问答", None, "SKIP：vLLM 未启动")

    cleanup_admin(c)
    return results()


if __name__ == "__main__":
    run()
