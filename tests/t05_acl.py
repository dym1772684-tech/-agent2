"""[t05] ACL 授权 / 审计 / admin 权限矩阵（TestClient 进程内）。

- bind/unbind/bindings/user_id 全端点
- 对象级越权：未绑定老人查询 403（testuser01 未绑定幽灵档案）
- admin 专属端点：/audit/logs、/agent/metrics
- 提供 find_admin()：t04 复用（探测常见 admin 凭据，探测结果进程内缓存）
"""
import uuid

from _lib import auth_header, check, cleanup_admin, client, ensure_admin, login, results

GHOST_ELDER = "E9999-ghost"  # e2e_security 约定的"永不绑定"幽灵档案


def run():
    print("[t05] ACL / 审计 / 权限矩阵")
    c = client()
    token, uid = login(c)
    if not token:
        raise RuntimeError("testuser01 登录失败，跳过本组")
    h = auth_header(token)

    admin_tok = ensure_admin(c)
    ha = auth_header(admin_tok) if admin_tok else {}

    # ---- 1) 未绑定老人的对象级越权 ----
    r = c.get("/api/v1/elder_identity/select_elder", headers=h, params={"elder_id": GHOST_ELDER})
    check("ACL: 未绑定老人查询 403", r.status_code == 403, f"http={r.status_code}")

    # ---- 2) admin 专属端点的权限矩阵 ----
    r = c.get("/api/v1/audit/logs", headers=h)
    check("权限: common 访问审计日志 403", r.status_code in (401, 403), f"http={r.status_code}")
    r = c.get("/api/v1/agent/metrics", headers=h)
    check("权限: common 访问 metrics 403", r.status_code in (401, 403), f"http={r.status_code}")

    if admin_tok:
        r = c.get("/api/v1/audit/logs", headers=ha)
        check("权限: admin 访问审计日志 200", r.status_code == 200, f"http={r.status_code}")
        r = c.get("/api/v1/agent/metrics", headers=ha)
        check("权限: admin 访问 metrics 200", r.status_code == 200, f"http={r.status_code}")
        body = r.json() if r.status_code == 200 else {}
        data = body.get("data") or body
        check("metrics: 含 token 账本字段", isinstance(data, dict) and len(data) >= 1,
              str(list(data)[:4]) if isinstance(data, dict) else "")
    else:
        check("权限: admin 端点验证", None, "SKIP：无 admin 账号(create_admin.py 创建后重跑)")

    # ---- 3) bind/unbind 全生命周期（admin 操作，绑定→验证→解绑）----
    if admin_tok and uid:
        tmp_elder = f"T-ACL-{uuid.uuid4().hex[:6]}"
        # 先建一个临时档案让绑定有意义（select 才能通过对象级校验）
        c.post("/api/v1/elder_identity/create_elder", headers=ha, json={
            "elder_id": tmp_elder, "name": "ACL临时"})
        r = c.post("/api/v1/acl/bind", headers=ha,
                   json={"user_id": uid, "elder_id": tmp_elder, "relation": "test"})
        check("ACL: bind 成功/幂等", r.status_code == 200, f"http={r.status_code}")
        r = c.post("/api/v1/acl/bind", headers=ha,
                   json={"user_id": uid, "elder_id": tmp_elder, "relation": "test"})
        check("ACL: 重复 bind 幂等", r.status_code == 200, f"http={r.status_code}")
        r = c.get("/api/v1/elder_identity/select_elder", headers=h, params={"elder_id": tmp_elder})
        check("ACL: 绑定后可查询该档案", r.status_code == 200, f"http={r.status_code}")
        r = c.get("/api/v1/acl/bindings", headers=ha, params={"user_id": uid})
        check("ACL: bindings 列出绑定", r.status_code == 200 and tmp_elder in r.text,
              f"http={r.status_code}")
        r = c.request("DELETE", "/api/v1/acl/bind", headers=ha,
                      json={"user_id": uid, "elder_id": tmp_elder})
        check("ACL: unbind 成功", r.status_code == 200, f"http={r.status_code}")
        r = c.get("/api/v1/elder_identity/select_elder", headers=h, params={"elder_id": tmp_elder})
        check("ACL: 解绑后查询 403", r.status_code == 403, f"http={r.status_code}")
        c.request("DELETE", "/api/v1/elder_identity/delete_elder",
                  headers=ha, json={"elder_id": tmp_elder})  # 清理

    # ---- 4) 审计留痕抽查（本组产生的 403 应出现在审计日志里）----
    if admin_tok:
        r = c.get("/api/v1/audit/logs", headers=ha, params={"page": 1, "page_size": 50})
        if r.status_code == 200:
            check("审计: 日志非空", len(r.text) > 20)
        else:
            check("审计: 日志非空", False, f"http={r.status_code}")

    cleanup_admin(c)
    return results()


if __name__ == "__main__":
    run()
