"""重排层：CrossEncoder 精排 + 低分阈值过滤。

================================================------------
bi-encoder vs cross-encoder 的分工（你笔记里也写了）
------------------------------------------------------------
  双编码器（embedding）：query 和文档**分别**编码，向量可离线预计算，
      快，但两段文本在编码时互相看不见 → 语义交互浅。
  交叉编码器（rerank）：把 (query, doc) **拼接后一起**进模型，
      注意力能跨段落交互 → 直接输出相关性分，质量高但慢。

所以工业界的标准做法就是本模块的位置：召回（快、粗）→ 融合 →
CrossEncoder 精排（慢、准，只处理 top N 条）。

改的意义（对照原 rerank）：
  1. 新增低分阈值 RERANK_MIN_SCORE：低于阈值的候选直接丢弃。
     融合分高不代表真相关（比如两路都弱命中），宁可少给 prompt
     喂一条噪声——喂错上下文比不喂更伤答案质量。
  2. 重排对象从"融合 top_k"改为"融合 top RERANK_TOP_N"：
     原来只对融合前 top_k 重排，融合排序误差没有矫正空间；
     现在从 20 条里挑最好的 4 条，CrossEncoder 才有发挥空间。
  3. 模型名/开关进 config；评测网格搜索阶段可关闭（太慢），
     只对最终 top-3 参数组合开启验证。
============================================================
"""
from __future__ import annotations

import threading
from typing import Any, Dict, List

from allagent2.guance.logging_helper import logger
from .config import RAGConfig

_cross_encoder = None
_ce_lock = threading.Lock()


def _get_cross_encoder(model_name: str):
    """进程级单例（延迟加载，import 时不拉起 torch）。"""
    global _cross_encoder
    if _cross_encoder is None:
        with _ce_lock:
            if _cross_encoder is None:
                from sentence_transformers import CrossEncoder  # 延迟导入

                logger.info("[rerank] 加载 CrossEncoder：%s", model_name)
                _cross_encoder = CrossEncoder(model_name)
    return _cross_encoder


def rerank(
    query: str,
    candidates: List[Dict[str, Any]],
    config: RAGConfig,
    top_k: int = 4,
) -> List[Dict[str, Any]]:
    """CrossEncoder 精排。

    Args:
        candidates: 融合后的候选（内部取前 RERANK_TOP_N 条进入精排）
        top_k:      最终保留数量
    返回:
        按重排分降序、且 ≥ RERANK_MIN_SCORE 的候选，带 rerank_score 字段
    """
    pool = candidates[: config.RERANK_TOP_N]
    if not pool:
        return []

    ce = _get_cross_encoder(config.RERANKER_MODEL)
    pairs = [(query or "", c.get("text", "")) for c in pool]
    scores = ce.predict(pairs)

    ranked: List[Dict[str, Any]] = []
    for c, s in zip(pool, scores):
        c = dict(c)
        c["rerank_score"] = float(s)
        ranked.append(c)
    ranked.sort(key=lambda x: x["rerank_score"], reverse=True)

    # 低分过滤：不相关的候选宁缺毋滥（防噪声进 prompt）
    kept = [c for c in ranked if c["rerank_score"] >= config.RERANK_MIN_SCORE]
    if len(kept) < len(ranked[:top_k]):
        logger.info(
            "[rerank] 低分过滤：%d 条候选中丢弃 %d 条（阈值 %.2f）",
            len(ranked),
            len(ranked) - len(kept),
            config.RERANK_MIN_SCORE,
        )
    return kept[:top_k]
