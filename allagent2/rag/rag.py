"""RAG 编排层：子块双路召回 → 融合 → 重排 → 父块输出。

============================================================
本文件只做编排，算法全部下沉到兄弟模块
------------------------------------------------------------
    splitter.py    结构切分 → 符号递归 → 父子配对 → 元数据
    lexical.py     chunk 级 TF × DF/IDF（BM25，系数全暴露）
    vectorstore.py VectorStore 协议 + ChromaStore(HNSW) / MilvusStore
    fusion.py      rrf / weighted_rrf / relative 三种融合
    rerank.py      CrossEncoder 精排 + 低分过滤
    config.py      全部可调参数（env 可覆盖，评测可回写）

一次检索的数据流：
    query ──┬─ 词法路：LexicalIndex.score() ──┐
            │                                  ├→ fuse() → rerank()
            └─ 向量路：embedding → store.query ─┘        │
                                                     子块 top-k
                                                            │
                                    parent_id 映射 → 父块输出（上下文完整）

为什么"检索用子块、返回父块"：
    子块粒度细（≤128 tok），TF 信号和向量语义都不被稀释；
    但直接把子块喂给 LLM 会缺上下文（比如"限盐 5 克"脱离血压条目），
    所以命中后映射回父块（完整条目，≤230 tok）再输出。

安全约束（沿用既有改造）：
    重依赖（torch/chromadb）全部延迟加载——顶层 import 拉起 torch
    曾导致与 database 原生扩展同进程段错误（退出码 139）。
============================================================
"""
from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from allagent2.guance.logging_helper import logger
from allagent2.guance.token_accounting import count_tokens

from .config import RAGConfig, rag_config
from .fusion import fuse
from .lexical import LexicalIndex
from .rerank import rerank
from .splitter import ChildChunk, ParentChunk, split_document
from .vectorstore import ChromaStore

DEFAULT_DOC_FILE = "elder_care.md"


# ------------------------------------------------------------
# 索引对象：一次构建、多次查询的全部状态
# ------------------------------------------------------------
@dataclass
class RAGIndex:
    children: List[ChildChunk]
    parents: List[ParentChunk]
    lexical: LexicalIndex
    store: Any                 # VectorStore 协议实现
    embedding_model: Any
    config: RAGConfig

    def __post_init__(self) -> None:
        self.child_by_id = {c.id: c for c in self.children}
        self.parent_by_id = {p.id: p for p in self.parents}

    @property
    def degraded(self) -> bool:
        """向量路是否处于降级状态（兜底零向量模型）。"""
        return not hasattr(self.embedding_model, "encode") or \
            type(self.embedding_model).__name__ == "_FallbackEmbeddingModel"


class _FallbackEmbeddingModel:
    """兜底 embedding：模型加载失败时保证链路不崩（但会打 ERROR 告警）。"""

    def encode(self, texts, batch_size=32, normalize_embeddings=True, show_progress_bar=False):
        # 兼容单文本与批量入参：返回与输入等长的向量列表
        if isinstance(texts, str):
            return [0.0 for _ in range(384)]
        return [[0.0 for _ in range(384)] for _ in texts]


# 【性能】进程级单例表：按模型名缓存，整个进程只加载一次。
#   原实现每次 build_index() 都 new 一个 SentenceTransformer，
#   网格搜索里建 N 个索引就加载 N 次模型（每次几十秒），这是测评慢的根源。
#   与 rerank.py 的 CrossEncoder 单例同款设计。
_embedding_models: Dict[str, Any] = {}
_embed_lock = threading.Lock()


def _note_degraded(component: str, exc: BaseException) -> None:
    """【降级计数】RAG 侧的降级埋点（走 metrics_hook，不反向依赖 fastap2）。"""
    try:
        from allagent2.guance.metrics_hook import meter

        meter("agent_degraded_total", 1, component=component, reason=type(exc).__name__)
    except Exception:
        pass


def _load_embedding_model(config: RAGConfig):
    """加载 embedding 模型（进程级单例，只初始化一次；失败走兜底并告警）。"""
    name = config.EMBEDDING_MODEL
    if name in _embedding_models:
        return _embedding_models[name]
    with _embed_lock:
        if name in _embedding_models:          # 双检锁：并发下也只加载一次
            return _embedding_models[name]
        try:
            # 【修复】HF_ENDPOINT 必须在 import sentence_transformers 之前设置：
            #   直连 huggingface.co 会 SSL 校验失败（原 rag.py 顶部有此设置，
            #   重写时一度丢失）；走镜像站，模型已有本地缓存时可直接加载。
            os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
            # 【修复】优先解析本地缓存快照：零网络请求。
            #   chromadb 比 embedding 先 import 时，huggingface_hub 的 HF_ENDPOINT
            #   常量已被冻结成 huggingface.co → HEAD 走直连 → SSL 失败 + 重试后
            #   "client has been closed"（多线程下必现）。本地快照路径完全绕开网络。
            model_source = name
            try:
                from huggingface_hub import snapshot_download

                model_source = snapshot_download(name, local_files_only=True)
            except Exception:
                pass  # 本地无缓存 → 保持模型名走在线拉取（镜像）
            from sentence_transformers import SentenceTransformer  # 延迟导入

            logger.info("[rag] 加载 embedding 模型：%s（进程内只加载一次）", name)
            _embedding_models[name] = SentenceTransformer(model_source)
        except Exception as exc:
            # 【安全】显式告警而非静默降级：向量路废了必须有人知道
            _note_degraded("embedding", exc)
            logger.error("embedding 模型加载失败，向量召回降级为零向量（仅剩词法路）：%s", exc)
            _embedding_models[name] = _FallbackEmbeddingModel()
        return _embedding_models[name]


def _embed(texts: List[str], model, batch_size: int = 32) -> List[List[float]]:
    vecs = model.encode(list(texts), batch_size=batch_size, normalize_embeddings=True, show_progress_bar=False)
    # ndarray 行向量 / 兜底 list 都统一成纯 float 列表
    return [list(map(float, v)) for v in vecs]


# ------------------------------------------------------------
# 索引构建（在线缓存版 + 评测直建版）
# ------------------------------------------------------------
def _index_fingerprint(cfg: RAGConfig, children: List) -> str:
    """索引指纹：文档内容(子块文本) + 切分参数 + embedding 模型 + HNSW 参数。

    持久化复用的判据——指纹一致说明磁盘上的向量与当前配置完全对应，
    可以跳过重嵌入；任何一项变了都自动触发重建。
    """
    h = hashlib.sha1()
    h.update(cfg.EMBEDDING_MODEL.encode("utf-8"))
    for key in ("CHILD_TOKENS", "PARENT_RATIO", "HNSW_SPACE", "HNSW_M",
                "HNSW_CONSTRUCTION_EF"):
        h.update(str(getattr(cfg, key)).encode("utf-8"))
    for c in children:
        h.update(c.index_text.encode("utf-8"))
    return h.hexdigest()


def build_index(config: Optional[RAGConfig] = None, doc_file: str = "",
                persist: bool = False, force_rebuild: bool = False) -> RAGIndex:
    """构建完整索引。

    persist=False（默认）：内存版索引（EphemeralClient），评测网格搜索用。
    persist=True：落盘版（PersistentClient）。磁盘上已有指纹匹配的索引时
        直接复用向量（跳过 embedding 模型加载与全部重嵌入），embedding_model
        置 None、由 search() 在首次查询时延迟加载——服务冷启动最快。
    """
    cfg = config or rag_config
    split = split_document(doc_file, child_tokens=cfg.CHILD_TOKENS, parent_ratio=cfg.PARENT_RATIO)
    if not split.children:
        raise ValueError(f"知识库切分后为空：{doc_file or DEFAULT_DOC_FILE}")

    lexical = LexicalIndex(split.children, cfg)
    fingerprint = _index_fingerprint(cfg, split.children)
    store = ChromaStore(cfg, persistent=persist,
                        fingerprint=fingerprint, force_rebuild=force_rebuild)

    if store.reused:
        # 【B3】命中持久化索引：不加载 embedding 模型、不重嵌入（延迟到首次查询）
        model = None
        logger.info(
            "[rag] 命中持久化索引（fp=%s…）：%d子块/%d父块，跳过重嵌入",
            fingerprint[:8], len(split.children), len(split.parents),
        )
    else:
        # 【改造】embedding 用带标题的 index_text（语义携带主题信号）
        model = _load_embedding_model(cfg)
        store.add(split.children, _embed(split.child_index_texts, model))
        logger.info(
            "[rag] 索引构建完成并落盘=%s：%d子块/%d父块，词法停用词%d个，降级=%s",
            persist, len(split.children), len(split.parents),
            len(lexical.stopwords), _is_fallback(model),
        )

    return RAGIndex(
        children=split.children, parents=split.parents,
        lexical=lexical, store=store, embedding_model=model, config=cfg,
    )


def _is_fallback(model) -> bool:
    return type(model).__name__ == "_FallbackEmbeddingModel"


_index_cache: Dict[tuple, RAGIndex] = {}
_index_lock = threading.Lock()


def get_rag_index(doc_file: str = "") -> RAGIndex:
    """在线检索入口：按 (doc_file, 切分参数) 缓存，整个进程只建一次。

    【B3】走持久化路径：磁盘上有指纹匹配的索引就复用（不重嵌入），
    embedding 模型延迟到首次查询才加载。离线预构建：rag/build_index.py。
    """
    key = (doc_file or DEFAULT_DOC_FILE, rag_config.CHILD_TOKENS, rag_config.PARENT_RATIO)
    if key not in _index_cache:
        with _index_lock:
            if key not in _index_cache:
                logger.info("[rag] 首次加载知识库索引：%s（只执行一次）", key[0])
                _index_cache[key] = build_index(rag_config, doc_file, persist=True)
    return _index_cache[key]


# ------------------------------------------------------------
# 检索编排
# ------------------------------------------------------------
def search(
    query: str,
    index: Optional[RAGIndex] = None,
    config: Optional[RAGConfig] = None,
    top_k: int = 0,
) -> Dict[str, Any]:
    """完整检索链路，返回 {docs, metrics}（docs 与 metrics 分离，便于评测）。"""
    cfg = config or (index.config if index else rag_config)
    idx = index or get_rag_index()
    # 【B3】持久化命中时 embedding_model 为 None → 此刻才加载（进程内单例，只一次）
    if idx.embedding_model is None:
        idx.embedding_model = _load_embedding_model(idx.config)
    k = top_k or cfg.TOP_K
    t0 = time.perf_counter()

    # ① 双路召回（都在子块粒度）
    lex_hits = idx.lexical.score(query or "", top_k=cfg.LEXICAL_TOP_K)
    lex_hits = [h for h in lex_hits if h["score"] > 0]

    vec_hits: List[Dict[str, Any]] = []
    if not idx.degraded and (query or "").strip():
        try:
            q_vec = _embed([query], idx.embedding_model)[0]
            vec_hits = idx.store.query(q_vec, top_k=cfg.VECTOR_TOP_K)
        except Exception as exc:
            # 【安全】单路失败显式告警 + 降级到另一路，不静默吞掉
            _note_degraded("rag_vector", exc)
            logger.error("[rag] 向量召回失败，本轮仅词法路：%s", exc)

    # ①.5 HyDE 第三路（可选）：先生成假设回答，再用它的向量召回。
    # 任何失败（无 key / 超时 / 模型报错）都退化为双路，不影响主链路。
    hyde_hits: List[Dict[str, Any]] = []
    if getattr(cfg, "HYDE_ENABLED", False) and not idx.degraded and (query or "").strip():
        try:
            from .hyde import generate_hypothesis

            hypothesis = generate_hypothesis(query, cfg)
            if hypothesis:
                h_vec = _embed([hypothesis], idx.embedding_model)[0]
                hyde_hits = idx.store.query(h_vec, top_k=getattr(cfg, "HYDE_TOP_K", 20))
        except Exception as exc:
            _note_degraded("rag_hyde", exc)
            logger.warning("[rag] HyDE 路失败，本轮退化为双路：%s", exc)

    # ② 融合（子块粒度，可选三路）
    fused = fuse(lex_hits, vec_hits, cfg, hyde_hits)

    # ③ 重排（可选；评测扫参阶段关闭）
    if cfg.ENABLE_RERANK and fused:
        ranked = rerank(query or "", fused, cfg, top_k=k)
    else:
        ranked = fused[:k]

    # ④ 子块 → 父块映射（去重：多个子块命中同一父块只输出一次，保留最高分）
    best_by_parent: Dict[str, Dict[str, Any]] = {}
    for c in ranked:
        pid = c.get("parent_id") or ""
        if pid and pid in idx.parent_by_id:
            if pid not in best_by_parent or c.get("rerank_score", c.get("score", 0)) > \
                    best_by_parent[pid].get("_rank_score", 0):
                entry = dict(c)
                entry["_rank_score"] = c.get("rerank_score", c.get("score", 0))
                best_by_parent[pid] = entry

    docs: List[Dict[str, Any]] = []
    for pid, entry in best_by_parent.items():
        parent = idx.parent_by_id[pid]
        docs.append({
            "text": parent.text,                     # 父块全文（上下文完整）
            "source": entry.get("doc_source") or parent.doc_source,  # 元数据来源，不再正则抠
            "heading_path": parent.heading_path,
            "child_id": entry["id"],
            "score": entry.get("rerank_score", entry.get("score", 0.0)),
        })
    docs.sort(key=lambda d: d["score"], reverse=True)
    docs = docs[:k]

    # ⑤ 可观测：RAG 专属指标（回答"检索吃了多少上下文、两路谁在干活"）
    lex_ids = {h["id"] for h in lex_hits}
    vec_ids = {h["id"] for h in vec_hits}
    hit_children = {d["child_id"] for d in docs}
    both = len(hit_children & lex_ids & vec_ids)
    metrics = {
        "query": (query or "")[:64],
        "latency_ms": round((time.perf_counter() - t0) * 1000, 1),
        "lex_hits": len(lex_hits),
        "vec_hits": len(vec_hits),
        "hyde_hits": len(hyde_hits),
        "hyde_used": bool(hyde_hits),
        "fused": len(fused),
        "parents_returned": len(docs),
        "injected_tokens": sum(count_tokens(d["text"]) for d in docs),
        "contrib_both": both,
        "contrib_lex_only": len(hit_children & lex_ids - vec_ids),
        "contrib_vec_only": len(hit_children & vec_ids - lex_ids),
        "vector_degraded": idx.degraded,
        "rerank_used": bool(cfg.ENABLE_RERANK and fused),
    }
    return {"docs": docs, "metrics": metrics}


# ------------------------------------------------------------
# 对外兼容入口：knowledge_tool.search_knowledge_base 依赖此签名
# ------------------------------------------------------------
def search_knowledge(query: str, top_k: int = 4) -> list:
    """检索养老养生知识库，返回 [{"text", "source"}, ...]（父块粒度）。

    【签名不变】工具层零感知；内部已换成父子切分 + 双路融合 +
    重排 + 元数据来源。指标落盘见 rag_metrics.jsonl。
    """
    result = search(query, top_k=top_k)
    _append_metrics(result["metrics"])
    return [{"text": d["text"], "source": d["source"]} for d in result["docs"]]


# ------------------------------------------------------------
# 指标落盘（与 token_usage.jsonl 同目录，供成本对账）
# ------------------------------------------------------------
_METRICS_LOG = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "guance", "logs", "rag_metrics.jsonl"
)


def _append_metrics(metrics: Dict[str, Any]) -> None:
    """一行一次检索。RAG 指标与计费指标分开存，事后可按时间关联对账。

    【修复】Windows 上常驻服务进程/OneDrive 同步会瞬时锁住日志文件
    （Errno 13 Permission denied）——加一次短重试，仍失败只告警不影响主链路。
    """
    record = {"ts": time.strftime("%Y-%m-%d %H:%M:%S"), **metrics}
    line = json.dumps(record, ensure_ascii=False) + "\n"
    try:
        os.makedirs(os.path.dirname(_METRICS_LOG), exist_ok=True)
        for attempt in range(2):
            try:
                with open(_METRICS_LOG, "a", encoding="utf-8") as f:
                    f.write(line)
                return
            except PermissionError:
                if attempt == 0:
                    time.sleep(0.2)  # 瞬时锁：等一下再试一次
                else:
                    raise
    except Exception as exc:
        logger.warning("[rag] 指标落盘失败：%s", exc)


def read_metrics_log(limit: int = 100) -> List[Dict[str, Any]]:
    """读最近 limit 条检索指标（管理端点用）。"""
    if not os.path.exists(_METRICS_LOG):
        return []
    try:
        with open(_METRICS_LOG, "r", encoding="utf-8") as f:
            lines = f.readlines()
    except Exception:
        return []
    out = []
    for line in lines[-limit:]:
        try:
            out.append(json.loads(line))
        except Exception:
            continue
    return out
