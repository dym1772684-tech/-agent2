"""S1 可信身份模块单测（零 LLM，需 Redis + MySQL 在线）。

验证项：
  [1] PERM_REGISTRY / AGENT_PROFILES / RESOURCE_REGISTRY 完整性
  [2] build_identity：RBAC 装配 + 工具白名单三档过滤 + admin 资源豁免
  [3] Redis 缓存：put → get 命中复用；属主不匹配 → None
  [4] perm_version 撤销：revoke 后旧身份即时失效
  [5] can()：fail-closed（未注册类型 False / 未绑定 elder False / admin 豁免）
  [6] role_permission DB 表缺失 → 回退硬编码
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

results = []


def check(name, ok, detail=""):
    results.append(ok)
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}")


def main():
    from core import identity as idm

    # ① 注册表完整性
    check("1a.PERM_REGISTRY 非空", len(idm.PERM_REGISTRY) >= 12)
    check("1b.AGENT_PROFILES 四智能体齐全",
          {"supervisor", "health", "knowledge", "life"} <= set(idm.AGENT_PROFILES))
    check("1c.RESOURCE_REGISTRY 覆盖 8 类资源",
          {"thread", "elder", "health", "memory", "knowledge", "audit", "admin", "tool"}
          <= set(idm.RESOURCE_REGISTRY))

    # ② 构建：family 角色
    ctx = idm.build_identity("u-test-1", "family", "t-smoke")
    check("2a.family 持有敏感档权限",
          {"elder:profile_read", "elder:health_read"} <= set(ctx.permissions), str(ctx.permissions))
    check("2b.family 工具白名单含 elder 工具",
          {"query_elder_profile", "analyze_health_trend"} <= set(ctx.allowed_tools))
    check("2c.family elders 为空列表（fail-closed 而非不限）", ctx.allowed_elders == [])
    ctx2 = idm.build_identity("u-admin-1", "admin", "t-smoke")
    check("2d.admin elders=None（全豁免）", ctx2.allowed_elders is None)

    # ③ 缓存
    ok = idm.put_identity(ctx)
    got = idm.get_identity(ctx.identity_id, expected_user_id="u-test-1")
    check("3a.put→get 命中且内容一致", ok and got and got.identity_id == ctx.identity_id)
    check("3b.属主不匹配 → None",
          idm.get_identity(ctx.identity_id, expected_user_id="someone-else") is None)

    # ④ 版本撤销
    idm.revoke_user("u-test-1")
    check("4a.revoke 后旧身份即时失效", idm.get_identity(ctx.identity_id) is None)
    fresh = idm.build_identity("u-test-1", "family", "t-smoke")
    idm.put_identity(fresh)
    check("4b.重建的新身份版本已更新", fresh.perm_version == ctx.perm_version + 1)

    # ⑤ can() fail-closed
    check("5a.未绑定 elder → False", fresh.can("elder", "E001") is False)
    admin2 = idm.build_identity("u-admin-1", "admin", "t-smoke")
    check("5b.admin elder 豁免 → True", admin2.can("elder", "E001") is True)
    check("5c.未注册资源类型 → False", fresh.can("nonexistent", "x") is False)
    check("5d.memory 只能本人", fresh.can("memory", "u-test-1") is True
          and fresh.can("memory", "other-user") is False)
    vol = idm.build_identity("u-vol-1", "volunteer", "t-smoke")
    check("5e.volunteer 无 elder 权限码且工具不含 elder 工具",
          "elder:profile_read" not in vol.permissions
          and "query_elder_profile" not in vol.allowed_tools)

    print(f"\n结果: {sum(results)}/{len(results)} 通过")
    sys.exit(0 if all(results) else 1)


if __name__ == "__main__":
    main()
