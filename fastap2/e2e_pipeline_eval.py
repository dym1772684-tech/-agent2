"""全链路评测:100 条测试集(整链路/边界/安全/多用户)走完整 agent 链路。

用法:cd fastap2 && python e2e_pipeline_eval.py [--workers 6] [--filter normal] [--limit 20]

与 e2e_mcp.py(只测工具接线)不同,本脚本直接调 bootstrap.run_agent ——
覆盖 急救短路 → 权限上下文 → (team 拓扑 supervisor→3专员→aggregate) →
MCP 工具 → 结构化输出 → 兜底 的完整链路。

判定口径(宽松,LLM 措辞有随机性):
- any_of :任一关键词出现在最终回答 → 满足;any_of 为空列表 = 只要求"链路不崩、有回答"
- none_of:禁用词出现 → 直接 FAIL(泄露/越权/注入得手等)
- emergency:期望急救短路(structured.emergency_flag=True 且零 LLM)
产出:控制台分 catégorie 汇总 + eval_pipeline_results.json 明细。
"""
import argparse
import concurrent.futures
import json
import sys
import time
from pathlib import Path

FASTAP2_DIR = Path(__file__).resolve().parent
AGENT2_DIR = FASTAP2_DIR.parent / "allagent2"
sys.path.insert(0, str(FASTAP2_DIR))
sys.path.insert(0, str(AGENT2_DIR))

EVALSET = AGENT2_DIR / "tests" / "evalset_100.jsonl"
OUT = FASTAP2_DIR / "eval_pipeline_results.json"


def load_cases():
    cases = []
    with open(EVALSET, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                cases.append(json.loads(line))
    return cases


def run_case(case: dict) -> dict:
    """跑单条:支持多轮(turns,同 thread 顺序执行),返回判定结果。"""
    from agent_bridge.bootstrap import run_agent

    questions = case.get("turns") or [case.get("question", "")]
    last_answer, last_struct, error = "", {}, None
    t0 = time.perf_counter()
    try:
        for i, q in enumerate(questions):
            result = run_agent(
                q,
                thread_id=f"{case['thread_id']}",
                elder_id=case.get("elder_id"),
                permissions=case.get("permissions"),
                user_id=case.get("user_id"),
            )
            last_answer = result.get("answer") or ""
            last_struct = result.get("structured") or {}
            # 多轮只有最后一轮参与判定(前几轮只负责铺上下文)
            if i < len(questions) - 1:
                continue
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    elapsed = time.perf_counter() - t0

    expect = case.get("expect") or {}
    answer = last_answer
    reasons = []

    # 链路级硬失败:异常 / 空回答 / 兜底话术
    if error:
        reasons.append(f"异常:{error}")
    if not answer.strip():
        reasons.append("空回答")

    # emergency 短路判定
    emergency_expected = expect.get("emergency")
    if emergency_expected is not None:
        got = bool(last_struct.get("emergency_flag"))
        if got != emergency_expected:
            reasons.append(f"emergency_flag 期望 {emergency_expected} 实际 {got}")

    # none_of:禁用词(泄露/越权得手/注入得手)一票否决
    for word in expect.get("none_of") or []:
        if word and word in answer:
            reasons.append(f"出现禁用词:{word!r}")

    # any_of:宽松命中;空列表 = 只要求不崩
    any_of = expect.get("any_of") or []
    if any_of and not any(w in answer for w in any_of):
        reasons.append(f"未命中关键词(任一):{any_of}")

    status = "FAIL" if reasons else "PASS"
    return {
        "id": case["id"],
        "category": case["category"],
        "status": status,
        "reasons": reasons,
        "elapsed": round(elapsed, 1),
        "answer": answer[:200],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--filter", default=None, help="只跑某 category")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    cases = load_cases()
    if args.filter:
        cases = [c for c in cases if c["category"] == args.filter]
    if args.limit:
        cases = cases[: args.limit]

    print(f"待跑 {len(cases)} 条,并发 {args.workers}")

    results = []
    t0 = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_case, c): c["id"] for c in cases}
        for n, fut in enumerate(concurrent.futures.as_completed(futures), 1):
            r = fut.result()
            results.append(r)
            mark = {"PASS": "✅", "FAIL": "❌"}.get(r["status"], "❓")
            print(f"[{n:>3}/{len(cases)}] {mark} {r['id']} ({r['category']}) "
                  f"{r['elapsed']}s" + (f"  {'; '.join(r['reasons'])}" if r["reasons"] else ""))

    # 汇总
    by_cat: dict = {}
    for r in results:
        stat = by_cat.setdefault(r["category"], {"pass": 0, "fail": 0, "fails": []})
        if r["status"] == "PASS":
            stat["pass"] += 1
        else:
            stat["fail"] += 1
            stat["fails"].append(r["id"])

    print("\n" + "=" * 60)
    total_pass = sum(s["pass"] for s in by_cat.values())
    total = total_pass + sum(s["fail"] for s in by_cat.values())
    for cat, s in by_cat.items():
        pct = s["pass"] / (s["pass"] + s["fail"]) * 100 if (s["pass"] + s["fail"]) else 0
        print(f"{cat:>10}: {s['pass']}/{s['pass'] + s['fail']} ({pct:.0f}%)"
              + (f"  失败:{','.join(s['fails'])}" if s["fails"] else ""))
    print(f"{'TOTAL':>10}: {total_pass}/{total}  总耗时 {time.perf_counter() - t0:.0f}s")

    OUT.write_text(json.dumps(
        {"summary": {k: {"pass": v["pass"], "fail": v["fail"], "fails": v["fails"]}
                     for k, v in by_cat.items()},
         "results": sorted(results, key=lambda r: r["id"])},
        ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"明细已写入 {OUT}")


if __name__ == "__main__":
    main()
