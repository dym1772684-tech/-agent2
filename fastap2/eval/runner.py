"""评测执行器:三 suite × 两模式的单条执行逻辑。

模式:
  offline  直调 agent_bridge.bootstrap.run_agent(快,开发迭代用;须在 fastap2/ 目录运行)
  e2e      真实 HTTP 链路(登录→Bearer→POST /agent/chat;测的是全链路,上线前用)

suite:
  rag       检索质量(不碰 LLM:只跑 RAG search,本地 embedding 模型)
  qa        端到端回答(evalset_100 全量;带参考答案的条目可选 LLM 裁判)
  security  安全(evalset_100 security 类目 + datasets/security_extra.jsonl 合并)

延迟导入:bootstrap/run_agent 只在 offline 执行时导入 —— 框架自身 import 零副作用,
单测和 --help 不拉起任何重依赖。
"""
from __future__ import annotations

import json
import os
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional

FASTAP2_DIR = Path(__file__).resolve().parents[1]
AGENT2_DIR = FASTAP2_DIR.parent / "allagent2"
EVAL_DIR = Path(__file__).resolve().parent

QA_EVALSET = AGENT2_DIR / "tests" / "evalset_100.jsonl"       # 单一事实源,不复制
RAG_EVALSET = AGENT2_DIR / "rag" / "evalset.jsonl"
SECURITY_EXTRA = EVAL_DIR / "datasets" / "security_extra.jsonl"
QA_REFERENCE = EVAL_DIR / "datasets" / "qa_reference.jsonl"
MULTI_TURN = EVAL_DIR / "datasets" / "multi_turn.jsonl"        # 2~5 轮对话专用集
RESUME_CASES = EVAL_DIR / "datasets" / "resume_cases.jsonl"    # 中断/续跑场景(由 e2e_resume.py 消费)

E2E_BASE = os.getenv("EVAL_E2E_BASE", "http://127.0.0.1:8000/api/v1")
E2E_USER = os.getenv("EVAL_E2E_USER", "testuser01")
E2E_PASS = os.getenv("EVAL_E2E_PASS", "test123456")


def _sys_path_ready() -> None:
    for p in (str(FASTAP2_DIR), str(AGENT2_DIR)):
        if p not in sys.path:
            sys.path.insert(0, p)


def load_jsonl(path: Path) -> List[dict]:
    out = []
    if not path.exists():
        return out
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def load_suite(suite: str) -> List[dict]:
    """按 suite 装载用例(qa=100 全量;security=32+补充;rag=56)。"""
    if suite == "rag":
        return load_jsonl(RAG_EVALSET)
    if suite == "qa":
        return load_jsonl(QA_EVALSET)
    if suite == "security":
        base = [c for c in load_jsonl(QA_EVALSET) if c.get("category") == "security"]
        return base + load_jsonl(SECURITY_EXTRA)
    if suite == "multi_turn":
        # 只取多轮条目:evalset_100 里已有的 7 条(turns 字段) + datasets/multi_turn.jsonl
        base = [c for c in load_jsonl(QA_EVALSET) if c.get("turns")]
        return base + load_jsonl(MULTI_TURN)
    raise ValueError(f"未知 suite: {suite}(可选 rag|qa|security|multi_turn)")


def load_references() -> Dict[str, str]:
    """参考答案映射 {case_id: reference_answer}(judge 只对有参考答案的条目启用)。"""
    return {r["id"]: r["reference_answer"]
            for r in load_jsonl(QA_REFERENCE) if r.get("id") and r.get("reference_answer")}


# ------------------------------------------------------------
# e2e 模式:HTTP 客户端(token 进程级缓存)
# ------------------------------------------------------------
_token: Optional[str] = None


def _e2e_login() -> str:
    global _token
    if _token:
        return _token
    req = urllib.request.Request(
        f"{E2E_BASE}/login",
        data=json.dumps({"username": E2E_USER, "password": E2E_PASS}).encode("utf-8"),
        headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=30) as resp:
        body = json.loads(resp.read().decode("utf-8"))
    _token = body["access_token"]
    return _token


def _e2e_chat(message: str) -> Dict[str, Any]:
    """单次 HTTP chat;返回 {answer, structured}(与 offline 口径对齐)。"""
    req = urllib.request.Request(
        f"{E2E_BASE}/agent/chat",
        data=json.dumps({"message": message, "debug": False}).encode("utf-8"),
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {_e2e_login()}"}, method="POST")
    with urllib.request.urlopen(req, timeout=180) as resp:
        body = json.loads(resp.read().decode("utf-8"))
    data = body.get("data") or {}
    return {"answer": data.get("answer") or "",
            "structured": data.get("structured") or {}}


def _run_turns(case: dict, mode: str) -> Dict[str, Any]:
    """按 turns(或单 question)执行;多轮只有最后一轮参与判定。"""
    questions = case.get("turns") or [case.get("question", "")]
    answer, structured, error = "", {}, None
    try:
        if mode == "e2e":
            for q in questions:
                r = _e2e_chat(q)
                answer, structured = r["answer"], r["structured"]
        else:
            _sys_path_ready()
            from agent_bridge.bootstrap import run_agent

            for q in questions:
                r = run_agent(
                    q,
                    thread_id=f"{case['thread_id']}",
                    elder_id=case.get("elder_id"),
                    permissions=case.get("permissions"),
                    user_id=case.get("user_id"),
                )
                answer = r.get("answer") or ""
                structured = r.get("structured") or {}
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    return {"answer": answer, "structured": structured, "error": error}


def run_case(case: dict, mode: str, reference_answer: str = "") -> dict:
    """单条 qa/security 用例:执行 → 规则断言 → (可选)LLM 裁判。"""
    try:
        from .assertions import evaluate
        from . import judge
    except ImportError:  # python eval/run_eval.py 直跑时无父包
        from assertions import evaluate
        import judge

    t0 = time.perf_counter()
    turn = _run_turns(case, mode)
    reasons = []
    if turn["error"]:
        reasons.append(f"异常:{turn['error']}")
    reasons += evaluate(turn["answer"], turn["structured"], case.get("expect") or {})

    result = {
        "id": case["id"],
        "category": case.get("category", "?"),
        "status": "FAIL" if reasons else "PASS",
        "reasons": reasons,
        "elapsed": round(time.perf_counter() - t0, 1),
        "answer": (turn["answer"] or "")[:300],
    }
    # LLM 裁判:仅 qa 条目 + 有参考答案 + 规则断言已过(断言挂了裁判分无意义)
    if reference_answer and not reasons and case.get("category") != "security":
        verdict = judge.judge_answer(case.get("question") or turn["answer"],
                                     turn["answer"], reference_answer)
        result["judge"] = verdict
    return result


# ------------------------------------------------------------
# rag suite:检索质量(无 LLM)
# ------------------------------------------------------------
def run_rag_suite(top_k: int = 10) -> dict:
    """56 条检索评测:R@4 / H@1 / MRR@10(gold 命中 heading_path 判定)。

    对齐 e2e_rag_eval.py 可信基线口径(R@4=0.929):top_k=10 取回,
    R@4 只看前 4,MRR 取首个命中名次的倒数。
    """
    _sys_path_ready()
    from allagent2.rag.rag import search

    cases = load_suite("rag")
    hits4 = hits1 = 0
    mrr_sum = 0.0
    details = []
    for c in cases:
        query, gold = c["query"], c["gold"]
        try:
            docs = search(query, top_k=top_k).get("docs") or []
        except Exception as exc:
            details.append({"query": query, "gold": gold, "error": str(exc)[:120]})
            continue
        rank = next((i + 1 for i, d in enumerate(docs)
                     if gold in (d.get("heading_path") or "")), 0)
        hit4 = 0 < rank <= 4
        hits4 += hit4
        hits1 += rank == 1
        mrr_sum += 1.0 / rank if rank else 0.0
        details.append({"query": query, "gold": gold, "rank": rank,
                        "top1_path": (docs[0].get("heading_path") if docs else "")})
    n = len(cases) or 1
    return {
        "suite": "rag",
        "n": len(cases),
        "R@4": round(hits4 / n, 3),
        "H@1": round(hits1 / n, 3),
        "MRR@10": round(mrr_sum / n, 3),
        "details": details,
    }
