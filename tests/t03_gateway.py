"""[t03] 数据面网关安全测试（data_gateway，进程内，不依赖 MCP 进程）。

核心安全模型：fail-closed。
- 坏令牌 / 无令牌 → 一律 403，绝不触碰数据库
- 工具白名单 → 白名单外 403
- 对象级 ACL → 未绑定老人 403
- 敏感字段 → 无敏感权限时脱敏
"""
import json

from _lib import check, load_env, make_agent_token, results


def _exec(tool, args, token):
    from mcp_servers.data_gateway import execute
    return execute(tool, args, token)


def run():
    print("[t03] data_gateway 安全网关")
    load_env()  # 签名密钥在 key.env 里，必须先装配

    # ---- 1) fail-closed：令牌缺失/伪造 ----
    out = _exec("query_elder_profile", {"elder_id": "E0001"}, "")
    check("网关: 空令牌拒绝", json.loads(out)["code"] == 403)
    out = _exec("query_elder_profile", {"elder_id": "E0001"}, "fake.token.sig")
    check("网关: 伪造令牌拒绝", json.loads(out)["code"] == 403)

    # ---- 2) 合法令牌 + 工具白名单 ----
    tok_full = make_agent_token("u-gw-test", ["E0001"])  # 不带 allowed_tools → 全放行
    tok_limited = make_agent_token("u-gw-test", ["E0001"], tools=["query_elder_profile"])
    out = _exec("nonexistent_tool", {"elder_id": "E0001"}, tok_full)
    body = json.loads(out)
    check("网关: 未知工具名被拒(400)", body.get("code") == 400, str(body)[:60])
    out = _exec("query_health_records", {"elder_id": "E0001"}, tok_limited)
    check("网关: 白名单外工具 403", json.loads(out)["code"] == 403)

    # ---- 3) 对象级 ACL：绑定 vs 未绑定 ----
    out_ok = _exec("query_elder_profile", {"elder_id": "E0001"}, tok_full)
    out_no = _exec("query_elder_profile", {"elder_id": "E9999-ghost"}, tok_full)
    code_ok = json.loads(out_ok).get("code")
    code_no = json.loads(out_no).get("code")
    check("网关: 绑定老人查询放行(code=0)", code_ok == 0, f"code={code_ok}")
    check("网关: 未绑定老人 403", code_no == 403, f"code={code_no}")

    # ---- 4) 令牌 TTL：过期令牌拒绝 ----
    from core.agent_context import _AccessContext, make_context_token
    ctx = _AccessContext(permissions={"allowed_elder_ids": ["E0001"]},
                         user_role="common", user_id="u-gw-test", authenticated=True)
    expired = make_context_token(ctx, ttl=-1)  # 立即过期
    out = _exec("query_elder_profile", {"elder_id": "E0001"}, expired)
    check("网关: 过期令牌 403", json.loads(out)["code"] == 403)

    # ---- 5) 白名单空列表=全禁（注释约定的语义）----
    tok_none = make_agent_token("u-gw-test", ["E0001"], tools=[])
    out = _exec("query_elder_profile", {"elder_id": "E0001"}, tok_none)
    check("网关: 空工具白名单全禁", json.loads(out)["code"] == 403)

    return results()


if __name__ == "__main__":
    run()
