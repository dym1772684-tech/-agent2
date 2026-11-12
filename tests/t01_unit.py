"""[t01] 零依赖单元测试：PII 脱敏 / 记忆评分 / 时间衰减 / 路由 gate。

不碰任何外部服务，秒级完成。失败说明核心纯逻辑坏了，后面的都别看。
"""
from _lib import check, results


def _rec(**kw):
    """构造 MemoryRecord（评分函数的输入载体）。"""
    from allagent2.session.schema import MemoryRecord
    base = dict(user_id="u-test", layer="semantic", category="知识", content="x",
                importance=0.5, confidence=0.5, specificity=0.5, relevance=0.5,
                hit_count=0)
    base.update(kw)
    return MemoryRecord(**base)


def run():
    print("[t01] 零依赖单元测试")

    # ---- 1) data_gateway 的 PII 正则 ----
    from mcp_servers.data_gateway import PII_PATTERNS, _mask_pii

    masked = _mask_pii("联系人张三，手机13812345678，身份证110101199001011234，邮箱 a@b.com")
    # 脱敏策略 keep 3+4：138****5678 —— 断言"完整号码不再出现"
    check("PII: 手机号脱敏", "13812345678" not in masked and "138****5678" in masked)
    check("PII: 身份证脱敏", "110101199001011234" not in masked and "110****" in masked)
    check("PII: 邮箱脱敏", "a@b.com" not in masked)
    check("PII: 四类规则齐备", len(PII_PATTERNS) == 4, f"patterns={len(PII_PATTERNS)}")

    # ---- 2) 记忆评分：乘法门控 + 分层权重（真实接口 compute/passes_gate）----
    from datetime import datetime, timedelta

    from allagent2.session.scoring import compute, frequency_factor, passes_gate, time_factor

    now = datetime.now()
    fresh = _rec(relevance=0.9, updated_at=now - timedelta(hours=1))
    stale = _rec(relevance=0.9, updated_at=now - timedelta(days=365))
    s_fresh, s_stale = compute(fresh, now=now), compute(stale, now=now)
    check("评分: 新记忆分数>旧记忆(时间衰减)", s_fresh > s_stale, f"{s_fresh:.3f}>{s_stale:.3f}")

    irr = _rec(relevance=0.02, updated_at=now)  # 低相关非 profile → 门控淘汰
    check("门控: 低相关记忆得分为0(乘法门控)", compute(irr, now=now) == 0.0)
    check("门控: profile 层必带不走门控", passes_gate(_rec(layer="profile", relevance=0.0)) is True)
    check("门控: 高相关 semantic 放行", passes_gate(_rec(relevance=0.9)) is True)

    check("频率因子: 0次=0 / 封顶=1",
          frequency_factor(0) == 0.0 and frequency_factor(100) == 1.0)
    check("时间因子: 高频老记忆保分",
          time_factor("episodic", now - timedelta(days=365), now=now, hit_count=50)
          > time_factor("episodic", now - timedelta(days=365), now=now, hit_count=0))

    # ---- 3) 路由 gate（曾被"逐字符迭代正则"bug 打穿，回归保护）----
    from allagent2.session.recall import route_layers

    l_weather = route_layers("今天天气如何")
    l_health = route_layers("血压监测怎么做")
    check("路由: 返回非空层列表", bool(l_weather) and bool(l_health), f"weather={l_weather}")
    check("路由: 不同意图路由结果有区分度", l_weather != l_health,
          f"weather={l_weather} health={l_health}")

    # ---- 4) token 估算（预算装配用）----
    from allagent2.guance.token_accounting import count_tokens

    n = count_tokens("血压" * 100)
    check("token估算: 重复词长文本计数>100", n > 100, f"n={n}")
    check("token估算: 空串为0", count_tokens("") == 0)

    return results()


if __name__ == "__main__":
    run()
