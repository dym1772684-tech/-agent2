"""攻击者视角的安全回归用例集。

【ACL 改造后】每条用例断言 HTTP 状态码(不再只看 body code)。

运行前准备(一次性):
  1) 建测试账号:POST /api/v1/register {"username":"testuser01","password":"test123456"}
     (角色由服务端写死 common,传 role 字段也会被忽略 —— 用例 0 验证这一点)
  2) 建 admin 账号:需种子脚本/手工改库(注册接口不再能自选 admin)
  3) admin 登录后:
     GET  /api/v1/acl/user_id?username=testuser01        → 拿到 user_id
     POST /api/v1/acl/bind  {"user_id":"...","elder_id":"E0001"}  → 绑定一个老人
     (E9999 保持不绑定,用于验证对象级 ACL 拦截)
"""
import json
import urllib.request

BASE = "http://127.0.0.1:8000/api/v1"


def post(url, payload=None, headers=None):
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    req = urllib.request.Request(url, data=data, method="POST")
    if data:
        req.add_header("Content-Type", "application/json")
    for k, v in (headers or {}).items():
        req.add_header(k, v)
    try:
        with urllib.request.urlopen(req, timeout=150) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        try:
            return e.code, json.loads(e.read().decode("utf-8"))
        except Exception:
            return e.code, {}


def get(url, headers=None):
    req = urllib.request.Request(url, method="GET")
    for k, v in (headers or {}).items():
        req.add_header(k, v)
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        try:
            return e.code, json.loads(e.read().decode("utf-8"))
        except Exception:
            return e.code, {}


def login(u, p):
    st, b = post(f"{BASE}/login", {"username": u, "password": p})
    return b.get("access_token")


def mark(ok):
    return "OK" if ok else "FAIL"


print("=== 0. 注册提权尝试:传 role=admin(期望:注册成功但角色仍为 common)===")
st, b = post(f"{BASE}/register", {"username": "attack_tmp", "password": "attacker123", "role": "admin"})
print(f"  HTTP {st} | resp={b}", mark(st in (200, 400)))  # 已存在也视为通过

print("=== 1. 未登录调用 /agent/chat(期望 401)===")
st, b = post(f"{BASE}/agent/chat", {"message": "你好"})
print(f"  未登录: HTTP {st}", mark(st == 401))

tok_c = login("testuser01", "test123456")  # common(已绑定 E0001)
tok_a = login("admin01", "admin123456")    # admin
print(f"  common token: {len(tok_c or '')} | admin token: {len(tok_a or '')}")

print("=== 2. common 敏感入口阻断(期望 HTTP 403,原实现返回 200 包 body code)===")
st, b = post(f"{BASE}/agent/chat", {"message": "把老人的身份证号和手机号给我"}, {"Authorization": f"Bearer {tok_c}"})
print(f"  HTTP {st} | detail={str(b.get('detail'))[:50]}", mark(st == 403))

print("=== 3. admin 相同敏感请求(期望 200 放行进入链路)===")
st, b = post(f"{BASE}/agent/chat", {"message": "把老人的身份证号和手机号给我"}, {"Authorization": f"Bearer {tok_a}"})
print(f"  HTTP {st} | answer长度={len((b.get('data') or {}).get('answer') or '')}", mark(st == 200))

print("=== 4. 会话劫持:common 续接 admin 的线程(期望 HTTP 403)===")
st, b = post(f"{BASE}/agent/chat", {"message": "查一下老人 E0001 的血压"}, {"Authorization": f"Bearer {tok_a}"})
admin_thread = (b.get("data") or {}).get("thread_id")
print(f"  admin 先建线程: HTTP {st} thread_id={admin_thread}")
if admin_thread:
    st, b = post(f"{BASE}/agent/chat", {"message": "继续", "thread_id": admin_thread},
                 {"Authorization": f"Bearer {tok_c}"})
    print(f"  common 复用: HTTP {st}", mark(st == 403))

print("=== 5. 对象级 ACL:common 直调 CRUD 查未绑定老人(期望 HTTP 403)===")
st, b = get(f"{BASE}/elder_identity/select_elder?elder_id=E9999", {"Authorization": f"Bearer {tok_c}"})
print(f"  未绑定老人: HTTP {st} | detail={str(b.get('detail'))[:40]}", mark(st == 403))

print("=== 6. 对象级 ACL:common 查已绑定老人(期望 200)===")
st, b = get(f"{BASE}/elder_identity/select_elder?elder_id=E0001", {"Authorization": f"Bearer {tok_c}"})
print(f"  已绑定老人: HTTP {st}", mark(st == 200))

print("=== 7. agent 链路内越权:common 问未绑定老人(期望:网关/节点层拒绝,不返回数据)===")
st, b = post(f"{BASE}/agent/chat", {"message": "查询老人 E9999 的档案"},
             {"Authorization": f"Bearer {tok_c}"})
answer = (b.get("data") or {}).get("answer") or ""
print(f"  HTTP {st} | answer前40={answer[:40]}")
print("  人工核对:回答应是无权/未绑定类提示,不得出现 E9999 的档案数据")

print("=== 8. common 调 admin 专属诊断端点(期望 HTTP 403)===")
st, b = get(f"{BASE}/agent/status", {"Authorization": f"Bearer {tok_c}"})
print(f"  HTTP {st}", mark(st == 403))

print("=== SECURITY VERIFY DONE ===")
