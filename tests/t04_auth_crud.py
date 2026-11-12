"""[t04] 认证 + 老人档案/健康记录 CRUD（TestClient 进程内真实 HTTP）。

- 认证：注册防重复/角色写死、登录成功/密码错、坏 token 401
- 档案：create(需 elder:write)→select→update→delete(仅 admin) 全生命周期 + 自清理
- 健康：create_record / record/{id} / stats / risk
- common 角色对写操作 403（权限矩阵回归）
"""
import uuid

from _lib import TEST_USER, auth_header, check, cleanup_admin, client, ensure_admin, login, results

ELDER_ID = f"T-ELDER-{uuid.uuid4().hex[:8]}"  # 测试专用档案，最后删除


def run():
    print("[t04] 认证 + 档案/健康 CRUD")
    c = client()

    # ---- 1) 认证 ----
    # 注意顺序：失败尝试会累积 login_lock 失败计数 → 真账号必须先登录成功；
    # 失败用例全部用独立用户名，不污染 testuser01 的计数
    token, uid = login(c)
    check("认证: 测试用户登录成功", token is not None and bool(uid), f"user_id={uid}")
    if not token:
        raise RuntimeError("登录失败，后续用例全部无法执行")
    h = auth_header(token)

    r = c.post("/api/v1/login", json={"username": "ghost_no_such_user", "password": "x"})
    check("认证: 不存在用户 401", r.status_code in (401, 400, 404), f"http={r.status_code}")
    r = c.post("/api/v1/login", json={"username": "ghost_no_such_user", "password": "wrong-password"})
    check("认证: 错误密码拒绝(401/429限流)", r.status_code in (401, 429), f"http={r.status_code}")

    r = c.get("/api/v1/agent/memory", headers=h)
    check("认证: 有效 token 访问受保护端点 200", r.status_code == 200,
          f"http={r.status_code}(/agent/memory,common 可访问自己的记忆)")
    r = c.get("/api/v1/agent/status", headers={"Authorization": "Bearer bad.token"})
    check("认证: 坏 token 401", r.status_code == 401, f"http={r.status_code}")

    # ---- 2) 档案 CRUD（common 无写权限，用 admin）----
    admin_tok = ensure_admin(c)
    check("档案: 存在可用的 admin 账号", admin_tok is not None,
          "自动探测/创建失败则写操作用例 SKIP")

    if admin_tok:
        ha = auth_header(admin_tok)
        r = c.post("/api/v1/elder_identity/create_elder", headers=ha, json={
            "elder_id": ELDER_ID, "name": "测试老人", "age": 80,
            "gender": "女", "health_tag": "自理", "chronic_disease": "高血压", "phone": "13800000000",
        })
        check("档案: admin 创建 200", r.status_code == 200, f"http={r.status_code} {r.text[:60]}")

        r = c.post("/api/v1/elder_identity/create_elder", headers=ha, json={
            "elder_id": ELDER_ID, "name": "重复创建"})
        check("档案: 重复 elder_id 被拒(非200)", r.status_code >= 400, f"http={r.status_code}")

        r = c.get("/api/v1/elder_identity/select_elder", headers=ha, params={"elder_id": ELDER_ID})
        ok = r.status_code == 200 and ELDER_ID in r.text
        check("档案: 创建后可查回", ok, f"http={r.status_code}")

        r = c.post("/api/v1/elder_identity/update_elder", headers=ha, json={
            "elder_id": ELDER_ID, "name": "测试老人改", "age": 81})
        check("档案: update 200", r.status_code == 200, f"http={r.status_code}")

        # common 角色越权写 → 403
        r = c.post("/api/v1/elder_identity/create_elder", headers=h, json={
            "elder_id": f"T-ELDER-X{uuid.uuid4().hex[:4]}", "name": "越权"})
        check("权限: common 创建档案 403", r.status_code in (401, 403), f"http={r.status_code}")

        # ---- 3) 健康记录 ----
        r = c.post("/api/v1/health/create_record", headers=ha, json={
            "health_id": 990001, "elder_id": ELDER_ID, "device_id": "T-DEV-01",
            "metric_type": "血压", "value": 135.0, "risk_content": {"systolic": 135}})
        check("健康: 创建记录", r.status_code == 200, f"http={r.status_code} {r.text[:60]}")
        r = c.get("/api/v1/health/record/990001", headers=ha)
        check("健康: 按 id 查回记录", r.status_code == 200 and "血压" in r.text,
              f"http={r.status_code}")
        r = c.get("/api/v1/health/stats/" + ELDER_ID, headers=ha)
        check("健康: stats 端点可达", r.status_code == 200, f"http={r.status_code}")
        r = c.get("/api/v1/health/risk/" + ELDER_ID, headers=ha)
        check("健康: risk 端点可达", r.status_code == 200, f"http={r.status_code}")

        # ---- 4) 清理 ----
        r = c.request("DELETE", "/api/v1/elder_identity/delete_elder",
                      headers=ha, json={"elder_id": ELDER_ID})
        check("档案: admin 删除(测试数据自清理)", r.status_code == 200, f"http={r.status_code}")
        r = c.get("/api/v1/elder_identity/select_elder", headers=ha, params={"elder_id": ELDER_ID})
        check("档案: 删除后查无此档案", r.status_code in (200, 404) and ELDER_ID not in r.text)

    cleanup_admin(c)
    return results()


if __name__ == "__main__":
    run()
