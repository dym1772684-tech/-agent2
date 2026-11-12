"""融合层：rrf / weighted_rrf / relative_score 三种可切换策略。

============================================================
为什么主推 relative（归一化线性加权）
------------------------------------------------------------
你的核心诉求是"模型越强，给向量检索更高权重"。三种策略对这条诉求
的支持度完全不同：

  rrf（原实现）       1/(k+rank) 等权相加 —— 两路永远同权，
                     想给向量加权无从下手，且分数不可解释。
  weighted_rrf        RRF 基础上加权 —— 能调权，但 1/(k+rank) 的
                     分数量纲仍是"排名的函数"，权重数值没有直观含义。
  relative ★默认      两路各自归一化到 [0,1]，再 w_lex·s_lex +
                     w_vec·s_vec 线性组合 —— 权重就是贡献占比，
                     W_VECTOR 从 0.6 提到 0.7 的效果一目了然。

归一化方式：min-max（该路内最高分→1，最低→0）。若该路全零分
（比如 BM25 完全没命中），整路置零，不给融合注入噪声。
============================================================
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from .config import RAGConfig


def _minmax_normalize(hits: List[Dict[str, Any]], key: str = "score") -> List[Dict[str, Any]]:
    """路内 min-max 归一化；全零路整体置零（不给融合注噪声）。

    【修复】span=0 的边界：该路只有一条命中（lo=hi>0）时，
    min-max 会把它归一化成 0 —— 明明是唯一的强命中却拿零分。
    修正：分数全相同且 > 0 时统一给 1.0。
    """
    scores = [h.get(key, 0.0) for h in hits]
    lo, hi = (min(scores), max(scores)) if scores else (0.0, 0.0)
    span = hi - lo
    for h, s in zip(hits, scores):
        if span > 0:
            h[f"norm_{key}"] = (s - lo) / span
        else:
            h[f"norm_{key}"] = 1.0 if hi > 0 else 0.0
    return hits


def relative_fusion(
    lexical_hits: List[Dict[str, Any]],
    vector_hits: List[Dict[str, Any]],
    config: RAGConfig,
    hyde_hits: Optional[List[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    """归一化线性加权：w_lex·s_lex + w_vec·s_vec（+ w_hyde·s_hyde）。

    HyDE 路为空时**不参与权重归一**——"没配 key"与"配了 key 但没命中"的结果
    必须与两路完全一致（回归保障，见 test_hyde.test_empty_hyde_same_as_none）。
    """
    _minmax_normalize(lexical_hits)
    _minmax_normalize(vector_hits)
    hyde_hits = hyde_hits or []
    _minmax_normalize(hyde_hits)

    use_hyde = bool(hyde_hits)
    total_w = config.W_LEXICAL + config.W_VECTOR + (config.W_HYDE if use_hyde else 0.0)
    total_w = total_w or 1.0
    wl = config.W_LEXICAL / total_w
    wv = config.W_VECTOR / total_w
    wh = (config.W_HYDE / total_w) if use_hyde else 0.0

    pool: Dict[str, Dict[str, Any]] = {}

    def _absorb(hits: List[Dict[str, Any]], side: str, weight: float) -> None:
        for h in hits:
            cid = h["id"]
            if cid not in pool:
                pool[cid] = {
                    "id": cid,
                    "text": h.get("text", ""),
                    "heading_path": h.get("heading_path", ""),
                    "parent_id": h.get("parent_id", ""),
                    "doc_source": h.get("doc_source", ""),
                    "score": 0.0,
                    "lex_score": 0.0,
                    "vec_score": 0.0,
                    "hyde_score": 0.0,
                    "matched_words": [],
                }
            s = h.get("norm_score", 0.0)
            pool[cid][f"{side}_score"] = s
            pool[cid]["score"] += weight * s
            if side == "lex" and h.get("matched_words"):
                pool[cid]["matched_words"] = h["matched_words"]

    _absorb(lexical_hits, "lex", wl)
    _absorb(vector_hits, "vec", wv)
    if use_hyde:
        _absorb(hyde_hits, "hyde", wh)

    return sorted(pool.values(), key=lambda x: x["score"], reverse=True)


def rrf_fusion(
    lexical_hits: List[Dict[str, Any]],
    vector_hits: List[Dict[str, Any]],
    config: RAGConfig,
    hyde_hits: Optional[List[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    """经典 RRF（等权）。保留作基线对照。"""
    return _rrf(lexical_hits, vector_hits, hyde_hits, config.RRF_K, 1.0, 1.0, 1.0)


def weighted_rrf_fusion(
    lexical_hits: List[Dict[str, Any]],
    vector_hits: List[Dict[str, Any]],
    config: RAGConfig,
    hyde_hits: Optional[List[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    """加权 RRF。权重作用于 1/(k+rank)，数值含义不如 relative 直观。"""
    return _rrf(lexical_hits, vector_hits, hyde_hits, config.RRF_K,
                config.W_LEXICAL, config.W_VECTOR, config.W_HYDE)


def _rrf(
    lexical_hits: List[Dict[str, Any]],
    vector_hits: List[Dict[str, Any]],
    hyde_hits: Optional[List[Dict[str, Any]]],
    k: int,
    wl: float,
    wv: float,
    wh: float = 1.0,
) -> List[Dict[str, Any]]:
    pool: Dict[str, Dict[str, Any]] = {}

    def _absorb(hits: List[Dict[str, Any]], weight: float) -> None:
        for rank, h in enumerate(hits):
            cid = h["id"]
            if cid not in pool:
                pool[cid] = {
                    "id": cid,
                    "text": h.get("text", ""),
                    "heading_path": h.get("heading_path", ""),
                    "parent_id": h.get("parent_id", ""),
                    "doc_source": h.get("doc_source", ""),
                    "score": 0.0,
                    "lex_score": 0.0,
                    "vec_score": 0.0,
                    "hyde_score": 0.0,
                    "matched_words": h.get("matched_words", []),
                }
            pool[cid]["score"] += weight / (k + rank + 1)

    _absorb(lexical_hits, wl)
    _absorb(vector_hits, wv)
    if hyde_hits:
        _absorb(hyde_hits, wh)
    return sorted(pool.values(), key=lambda x: x["score"], reverse=True)


_FUSION_IMPLS = {
    "relative": relative_fusion,
    "rrf": rrf_fusion,
    "weighted_rrf": weighted_rrf_fusion,
}


def fuse(
    lexical_hits: List[Dict[str, Any]],
    vector_hits: List[Dict[str, Any]],
    config: RAGConfig,
    hyde_hits: Optional[List[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    """按 config.FUSION_MODE 分发。未知名回落 relative（config 已白名单校验）。"""
    impl = _FUSION_IMPLS.get(config.FUSION_MODE, relative_fusion)
    return impl(lexical_hits, vector_hits, config, hyde_hits)
