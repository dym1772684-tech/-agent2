"""eval CLI 入口。

用法(cd fastap2/):
  python eval/run_eval.py --suite security --gate            # 安全门禁(必须100%)
  python eval/run_eval.py --suite qa --mode offline --workers 4
  python eval/run_eval.py --suite qa --mode e2e --judge      # 带参考答案的条目走裁判
  python eval/run_eval.py --suite qa --limit 20              # 抽样
  python eval/run_eval.py --suite rag                        # 只跑检索(不碰LLM)
  python eval/run_eval.py --suite all --mode offline         # rag+qa+security
  python eval/run_eval.py --suite qa --save-baseline         # 全绿后存基线

注意:
  offline 模式须在 fastap2/ 目录运行(bootstrap 的相对路径依赖);
  e2e 模式须先起服务(8000);并发 worker>4 时注意 API 限流(混元默认5并发)。
"""
from __future__ import annotations

import argparse
import concurrent.futures
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))  # 支持 python eval/run_eval.py 直跑

from runner import load_suite, load_references, run_case, run_rag_suite  # noqa: E402
import report  # noqa: E402


def run_suite(suite: str, mode: str, workers: int, limit: int,
              use_judge: bool, gate: bool, save_baseline: bool) -> int:
    # rag 是独立口径(无 LLM、无并发 agent)
    if suite == "rag":
        payload = run_rag_suite()
        payload["summary"] = {"pass_rate": 1.0 if payload["R@4"] >= 0.9 else 0.0}
        print(f"R@4={payload['R@4']}  H@1={payload['H@1']}  MRR@10={payload['MRR@10']}")
        path = report.save_result("rag", payload)
        print(f"明细: {path}")
        if save_baseline:
            print(f"基线已存: {report.save_baseline('rag', payload)}")
        problems = report.gate_check("rag", payload, False) if gate else []
        for p in problems:
            print(f"[GATE] ❌ {p}")
        return 1 if problems else 0

    cases = load_suite(suite)
    if limit:
        cases = cases[:limit]
    refs = load_references() if use_judge else {}
    if use_judge and not refs:
        print("提示: qa_reference.jsonl 为空,judge 无可评条目")
    print(f"[{suite}] 待跑 {len(cases)} 条,mode={mode},workers={workers},judge={'on' if refs else 'off'}")

    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(run_case, c, mode, refs.get(c["id"], "")): c["id"]
                   for c in cases}
        for n, fut in enumerate(concurrent.futures.as_completed(futures), 1):
            r = fut.result()
            results.append(r)
            mark = {"PASS": "✅", "FAIL": "❌"}.get(r["status"], "❓")
            print(f"[{n:>3}/{len(cases)}] {mark} {r['id']} ({r['category']}) "
                  f"{r['elapsed']}s" + (f"  {'; '.join(r['reasons'])}" if r["reasons"] else ""))

    summary = report.aggregate(results)
    payload = {"suite": suite, "mode": mode, "summary": summary, "results": results}
    for line in report.compare(payload, report.load_baseline(suite)):
        print(line)
    for cat, stat in summary["by_category"].items():
        n = stat["pass"] + stat["fail"]
        print(f"  {cat:>10}: {stat['pass']}/{n}"
              + (f"  失败:{','.join(stat['fails'])}" if stat["fails"] else ""))
    if summary.get("judge_mean"):
        print(f"  judge_mean: {summary['judge_mean']} (errors={summary['judge_errors']})")
    print(f"  通过率: {summary['pass_rate']*100:.1f}%")

    path = report.save_result(suite, payload)
    print(f"明细: {path}")
    if save_baseline:
        print(f"基线已存: {report.save_baseline(suite, payload)}")

    problems = report.gate_check(suite, payload, bool(refs)) if gate else []
    for p in problems:
        print(f"[GATE] ❌ {p}")
    if gate and not problems:
        print("[GATE] ✅ 通过")
    return 1 if problems else 0


def main() -> int:
    ap = argparse.ArgumentParser(description="统一评测框架")
    ap.add_argument("--suite", required=True,
                    choices=["rag", "qa", "security", "multi_turn", "all"])
    ap.add_argument("--mode", default="offline", choices=["offline", "e2e"])
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--judge", action="store_true", help="启用 LLM 裁判(需 JUDGE_API_KEY)")
    ap.add_argument("--gate", action="store_true", help="门禁模式:不达标 exit 1")
    ap.add_argument("--save-baseline", action="store_true", help="结果写入 baseline")
    args = ap.parse_args()

    rc = 0
    suites = ["rag", "qa", "security"] if args.suite == "all" else [args.suite]
    for s in suites:
        rc |= run_suite(s, args.mode, args.workers, args.limit,
                        args.judge, args.gate, args.save_baseline)
    return rc


if __name__ == "__main__":
    sys.exit(main())
