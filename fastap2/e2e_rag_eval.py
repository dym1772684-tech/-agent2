"""RAG 参数网格搜索：切分参数 × 检索参数 两级搜索，指标 Recall@4 / MRR@10 / Hit@1。

============================================================
为什么是两级（这是省时间的关键）
------------------------------------------------------------
第 1 级扫**切分参数**（CHILD_TOKENS × PARENT_RATIO = 3×3 = 9 组）：
    换切分就要重建向量索引（重新 embedding）——这是最贵的步骤。

第 2 级扫**检索参数**（K1 × B × 权重配比 = 3×3×3 = 27 组）：
    关键洞察：向量路结果只跟切分参数有关，跟 K1/B/权重无关！
    所以每个切分组合下，把 56 条 query 的向量召回**算一次缓存**，
    27 组检索参数只是在缓存的命中列表上换融合公式——纯 CPU、瞬间完成。
    9 个索引 × 56 query 的 embedding（约 500 次，1 分钟内）+ 免费的融合扫描。

CrossEncoder 在搜索阶段关闭（太慢，56 条要 1.5 分钟），只对最终 top-3
组合开启验证——这就是为什么 config 里有 ENABLE_RERANK 开关。

指标口径：
    Recall@4  gold 条目出现在 top-4 父块的 source 里（能不能召回）
    MRR@10    首个 gold 命中的排名倒数（排得靠前分高）
    Hit@1     top-1 就是 gold（严格正确率）

产出：全量排名表 + top-3 重排验证 + 最优参数打印（人工回写 config）。
用法：
    E:/python/python.exe e2e_rag_eval.py            # 全量搜索
    E:/python/python.exe e2e_rag_eval.py --quick    # 只跑 128tok 默认切分
============================================================
"""
import argparse
import itertools
import json
import os
import sys
import time
from pathlib import Path

# ---- 接线：把 allagent2 注册成可导入包（与 bootstrap 同款手法，但只读 RAG）----
FASTAP2_DIR = Path(__file__).resolve().parent
AGENT2_DIR = FASTAP2_DIR.parent / "allagent2"
sys.path.insert(0, str(FASTAP2_DIR))
sys.path.insert(0, str(AGENT2_DIR))
import types as _types

_pkg = _types.ModuleType("allagent2")
_pkg.__path__ = [str(AGENT2_DIR)]
sys.modules.setdefault("allagent2", _pkg)

from allagent2.rag.config import RAGConfig, rag_config  # noqa: E402
from allagent2.rag.fusion import fuse  # noqa: E402
from allagent2.rag.lexical import LexicalIndex  # noqa: E402
from allagent2.rag.rag import _embed, _load_embedding_model, build_index  # noqa: E402
from allagent2.rag.splitter import split_document  # noqa: E402

EVALSET = AGENT2_DIR / "rag" / "evalset.jsonl"

# 网格（初值来自经验区间；跑完全量后按 top 结果收窄再扫）
GRID_SPLIT_CHILD = [96, 128, 160]
GRID_SPLIT_RATIO = [1.5, 1.8, 2.0]
GRID_K1 = [0.9, 1.2, 1.5]
GRID_B = [0.6, 0.75, 0.9]
GRID_WEIGHTS = [(0.3, 0.7), (0.4, 0.6), (0.5, 0.5)]  # (W_LEXICAL, W_VECTOR)

TOP_K_EVAL = 4
MRR_DEPTH = 10


def load_evalset():
    cases = []
    with open(EVALSET, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                cases.append(json.loads(line))
    return cases


def metrics_at(ranked_sources, gold, k=TOP_K_EVAL):
    """ranked_sources: 按分值降序的 source 列表；gold: 目标条目名（source 包含即算命中）。"""
    top = ranked_sources[:k]
    hit_k = any(gold in s for s in top)
    rr = 0.0
    for i, s in enumerate(ranked_sources[:MRR_DEPTH], 1):
        if gold in s:
            rr = 1.0 / i
            break
    return (1.0 if (ranked_sources and gold in ranked_sources[0]) else 0.0), hit_k, rr


def eval_combo(cases, idx_obj, cfg, vec_cache):
    """在给定索引与配置下评测全部 query。vec_cache: {query: vec_hits}（重算融合零成本）。"""
    recall = mrr = hit1 = 0.0
    for case in cases:
        q, gold = case["query"], case["gold"]
        lex_hits = [h for h in idx_obj.lexical.score(q, top_k=cfg.LEXICAL_TOP_K) if h["score"] > 0]
        vec_hits = vec_cache[q]
        fused = fuse(lex_hits, vec_hits, cfg)
        # 子块 → 父块去重（与在线 search 同逻辑，但不重排）
        seen_parents, sources = set(), []
        for c in fused:
            pid = c.get("parent_id") or ""
            if pid and pid not in seen_parents:
                seen_parents.add(pid)
                sources.append(c.get("doc_source", ""))
            if len(sources) >= TOP_K_EVAL * 2:
                break
        h1, hk, rr = metrics_at(sources, gold)
        recall += hk
        mrr += rr
        hit1 += h1
    n = len(cases)
    return recall / n, mrr / n, hit1 / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="只跑默认切分（128/1.8），扫检索参数")
    ap.add_argument("--rerank-check", action="store_true", help="对 top-3 组合开启重排验证")
    args = ap.parse_args()

    cases = load_evalset()
    print(f"评测集：{len(cases)} 条（14 条目 × 4 口语 query）")

    # 加载 embedding 模型（进程内单例：__main__ 已预加载，这里直接复用）
    global _shared_model
    if _shared_model is None:
        _shared_model = _load_embedding_model(rag_config)
    model = _shared_model
    print("embedding 模型就绪")

    split_grid = [(128, 1.8)] if args.quick else list(
        itertools.product(GRID_SPLIT_CHILD, GRID_SPLIT_RATIO)
    )
    results = []
    t_all = time.perf_counter()

    for child, ratio in split_grid:
        t0 = time.perf_counter()
        cfg_split = rag_config.with_overrides(CHILD_TOKENS=child, PARENT_RATIO=ratio)
        split = split_document(child_tokens=child, parent_ratio=ratio)
        lex = LexicalIndex(split.children, cfg_split)
        # 向量召回缓存：只跟切分参数有关，27 组检索参数直接复用
        vec_cache = {}
        for case in cases:
            try:
                q_vec = _embed([case["query"]], model)[0]
                vec_cache[case["query"]] = lex_store_query(split, q_vec, cfg_split)
            except Exception as exc:
                print(f"  向量召回失败（{case['query'][:12]}…）→ 该 query 仅词法路：{exc}")
                vec_cache[case["query"]] = []
        print(f"\n[切分 child={child} ratio={ratio}] {len(split.children)}子块/{len(split.parents)}父块 "
              f"建索引 {time.perf_counter()-t0:.1f}s")

        for k1, b, (wl, wv) in itertools.product(GRID_K1, GRID_B, GRID_WEIGHTS):
            cfg = cfg_split.with_overrides(
                K1=k1, B=b, W_LEXICAL=wl, W_VECTOR=wv, ENABLE_RERANK=False
            )
            r, m, h = eval_combo(cases, _IndexView(split, lex), cfg, vec_cache)
            results.append({
                "child": child, "ratio": ratio, "k1": k1, "b": b,
                "w_lex": wl, "w_vec": wv,
                "recall@4": round(r, 4), "mrr@10": round(m, 4), "hit@1": round(h, 4),
            })

    results.sort(key=lambda x: (x["recall@4"], x["mrr@10"], x["hit@1"]), reverse=True)

    print(f"\n{'='*88}\n全量搜索完成（{len(results)} 组合，总耗时 {time.perf_counter()-t_all:.0f}s），TOP 10：")
    header = f"{'child':>5} {'ratio':>5} {'k1':>4} {'b':>5} {'w_lex':>5} {'w_vec':>5}  {'R@4':>6} {'MRR@10':>7} {'H@1':>6}"
    print(header)
    for r in results[:10]:
        print(f"{r['child']:>5} {r['ratio']:>5} {r['k1']:>4} {r['b']:>5} "
              f"{r['w_lex']:>5} {r['w_vec']:>5}  {r['recall@4']:>6.3f} {r['mrr@10']:>7.3f} {r['hit@1']:>6.3f}")

    # 保存全量结果，便于后续收窄网格
    out = FASTAP2_DIR / "rag_eval_results.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=1)
    print(f"\n全量结果已存 {out.name}")

    # 可选：top-3 组合开启 CrossEncoder 重排验证
    if args.rerank_check:
        print("\n[top-3 重排验证]（CrossEncoder 逐 query 精排，约需数分钟）")
        for r in results[:3]:
            cfg = rag_config.with_overrides(
                CHILD_TOKENS=r["child"], PARENT_RATIO=r["ratio"], K1=r["k1"], B=r["b"],
                W_LEXICAL=r["w_lex"], W_VECTOR=r["w_vec"], ENABLE_RERANK=True,
            )
            idx = build_index(cfg)  # 完整索引（含向量库与重排配置）
            recall = mrr = hit1 = 0.0
            for case in cases:
                res = _search_with_rerank(case["query"], idx, cfg)
                h1, hk, rr = metrics_at([d["source"] for d in res], case["gold"])
                recall += hk; mrr += rr; hit1 += h1
            n = len(cases)
            print(f"  child={r['child']} ratio={r['ratio']} k1={r['k1']} b={r['b']} "
                  f"w=({r['w_lex']},{r['w_vec']}) → R@4={recall/n:.3f} MRR={mrr/n:.3f} H@1={hit1/n:.3f}")

    best = results[0]
    print("\n" + "=" * 88)
    print("建议回写 config.py 的默认值：")
    print(f"  CHILD_TOKENS = {best['child']}")
    print(f"  PARENT_RATIO = {best['ratio']}")
    print(f"  K1 = {best['k1']}    B = {best['b']}")
    print(f"  W_LEXICAL = {best['w_lex']}    W_VECTOR = {best['w_vec']}")


def lex_store_query(split, q_vec, cfg):
    """向量路召回（ChromaStore 建一次查一次，随切分索引走）。"""
    from allagent2.rag.vectorstore import ChromaStore

    store = ChromaStore(cfg)
    store.add(split.children, _embed(split.child_index_texts, _shared_model))
    return store.query(q_vec, top_k=cfg.VECTOR_TOP_K)


_shared_model = None


class _IndexView:
    """评测用轻量索引视图（避免为每组参数重复建 Chroma）。"""

    def __init__(self, split, lex):
        self.children = split.children
        self.lexical = lex


def _search_with_rerank(query, idx, cfg):
    """带重排的完整检索。idx 是 build_index 产出的完整 RAGIndex（向量库复用，不重建）。"""
    from allagent2.rag.rerank import rerank as _rerank

    lex_hits = [h for h in idx.lexical.score(query, top_k=cfg.LEXICAL_TOP_K) if h["score"] > 0]
    q_vec = _embed([query], _shared_model)[0]
    vec_hits = idx.store.query(q_vec, top_k=cfg.VECTOR_TOP_K)
    fused = fuse(lex_hits, vec_hits, cfg)
    ranked = _rerank(query, fused, cfg, top_k=cfg.TOP_K)
    seen, docs = set(), []
    for c in ranked:
        pid = c.get("parent_id") or ""
        if pid and pid not in seen:
            seen.add(pid)
            docs.append({"source": c.get("doc_source", ""), "score": c.get("rerank_score", 0)})
    return docs[: cfg.TOP_K]


if __name__ == "__main__":
    _shared_model = _load_embedding_model(rag_config)
    main()
