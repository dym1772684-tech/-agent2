"""向量层：VectorStore 协议 + ChromaStore(HNSW) + MilvusStore 桩。

============================================================
设计意图
------------------------------------------------------------
为什么抽协议：你提到接下来想换 pymilvus。现在把向量库收敛到一个
最小接口（add / query），上层（rag.py 编排）只认协议不认实现——
将来接 Milvus 时新增一个类，编排层一行不改。

HNSW 参数全部从 rag_config 透出（Chroma 通过 collection metadata 下发）：
  hnsw:M                 每节点连接数，越大召回越高、内存越大
  hnsw:construction_ef   建图搜索深度
  hnsw:search_ef         查询搜索深度
改的意义：原来 EphemeralClient + 默认参数，HNSW 不可调，
评测时无法回答"ef_search 提到 128 能换回多少召回"。

注意：EphemeralClient（网格搜索路径）每次重启重建索引；在线路径已改用
PersistentClient 落盘（persistent=True，见 B3），指纹匹配即复用磁盘向量。
HNSW 参数在两种模式下同样生效。
============================================================
"""
from __future__ import annotations

from typing import Any, Dict, List, Protocol, runtime_checkable

from .config import RAGConfig
from .splitter import ChildChunk


@runtime_checkable
class VectorStore(Protocol):
    """向量库最小协议：上层编排只依赖这两个方法。"""

    def add(self, children: List[ChildChunk], embeddings: List[List[float]]) -> None: ...
    def query(self, embedding: List[float], top_k: int) -> List[Dict[str, Any]]: ...


class ChromaStore:
    """Chroma 实现（当前默认）。HNSW 参数经 collection metadata 下发。

    两种模式（B3 持久化改造）：
      persistent=False（默认）→ EphemeralClient + 每次删重建，
          网格搜索用：为不同切分参数反复建索引，进程内共享单例要求先删后建。
      persistent=True → PersistentClient(path=cfg.PERSIST_DIR) 落盘，
          在线路径用：指纹（文档内容+切分参数+embedding 模型）匹配且条数一致
          → 直接复用磁盘上的向量（reused=True，调用方跳过重嵌入）；
          指纹不匹配 / 强制重建 → 删旧集合重建。
    """

    _FP_KEY = "index_fingerprint"  # 指纹存进 collection metadata，与 hnsw:* 并列

    def __init__(self, config: RAGConfig, collection_name: str = "care_children",
                 persistent: bool = False, fingerprint: str = "",
                 force_rebuild: bool = False):
        import chromadb  # 延迟导入（内部带 onnxruntime）

        self.cfg = config
        self.reused = False
        metadata = {
            "hnsw:space": config.HNSW_SPACE,
            "hnsw:M": config.HNSW_M,
            "hnsw:construction_ef": config.HNSW_CONSTRUCTION_EF,
            "hnsw:search_ef": config.HNSW_SEARCH_EF,
            self._FP_KEY: fingerprint,
        }
        if persistent:
            self._client = chromadb.PersistentClient(path=config.PERSIST_DIR)
            if not force_rebuild:
                try:
                    col = self._client.get_collection(collection_name)
                    fp = (col.metadata or {}).get(self._FP_KEY, "")
                    if fp == fingerprint and fingerprint and col.count() > 0:
                        # 指纹匹配且非空 → 命中，直接复用磁盘向量
                        self._collection = col
                        self.reused = True
                        return
                except Exception:
                    pass  # 集合不存在 → 走下面的重建
        else:
            # 注意：chromadb.EphemeralClient() 在同进程内是共享单例——
            # 网格搜索要为不同切分参数建多个索引，必须先删旧集合再建，
            # 否则第二次 create_collection 直接报 already exists。
            self._client = chromadb.EphemeralClient()
        try:
            self._client.delete_collection(collection_name)
        except Exception:
            pass  # 首次构建时本就不存在
        self._collection = self._client.create_collection(
            name=collection_name,
            metadata=metadata,
        )

    def count(self) -> int:
        return self._collection.count()

    def add(self, children: List[ChildChunk], embeddings: List[List[float]]) -> None:
        """子块入库：id / 文本 / embedding / 元数据 一起走。

        文本用 index_text（标题末段+正文）：embedding 对孤立片段的语义
        表征是漂的，拼上标题让向量携带主题信号（见 ChildChunk.index_text）。
        """
        if not children:
            return
        self._collection.add(
            ids=[c.id for c in children],
            documents=[c.index_text for c in children],
            embeddings=embeddings,
            metadatas=[c.metadata() for c in children],
        )

    def query(self, embedding: List[float], top_k: int) -> List[Dict[str, Any]]:
        res = self._collection.query(query_embeddings=[embedding], n_results=top_k)
        out: List[Dict[str, Any]] = []
        ids = res.get("ids") or [[]]
        docs = res.get("documents") or [[]]
        dists = res.get("distances") or [[]]
        metas = res.get("metadatas") or [[]]
        for i in range(len(ids[0])):
            d = dists[0][i] if i < len(dists[0]) else None
            out.append({
                "id": ids[0][i],
                "text": docs[0][i],
                # cosine 距离 → 相似度（fusion 的 relative 模式要求分数越大越好）
                "score": (1.0 - d) if d is not None else 0.0,
                "distance": d,
                **(metas[0][i] if i < len(metas[0]) and metas[0][i] else {}),
            })
        return out


class MilvusStore:
    """pymilvus 桩（未安装依赖，接口占位）。

    将来接入时把 NotImplementedError 换成 pymilvus 实现，
    并在 rag.py 里把 store 换成本类——上层零改动。
    """

    def __init__(self, config: RAGConfig, collection_name: str = "care_children"):
        self.cfg = config
        self.collection_name = collection_name
        self._records: Dict[str, Dict[str, Any]] = {}  # 桩：内存兜底，保证接口可用

    def add(self, children: List[ChildChunk], embeddings: List[List[float]]) -> None:
        for c, e in zip(children, embeddings):
            self._records[c.id] = {
                "id": c.id, "text": c.text, "embedding": e, **c.metadata(),
            }

    def query(self, embedding: List[float], top_k: int) -> List[Dict[str, Any]]:
        """桩实现：暴力余弦（无 HNSW，仅供联调，性能勿参考）。"""
        import math

        def cos(a, b):
            dot = sum(x * y for x, y in zip(a, b))
            na = math.sqrt(sum(x * x for x in a)) or 1e-9
            nb = math.sqrt(sum(x * x for x in b)) or 1e-9
            return dot / (na * nb)

        scored = [
            {k: v for k, v in r.items() if k != "embedding"} | {"score": cos(embedding, r["embedding"])}
            for r in self._records.values()
        ]
        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:top_k]
