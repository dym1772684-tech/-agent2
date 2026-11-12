"""文档检索库:Chroma 入库(acl 元数据 + 文档指纹 + 增量删改)。

设计沿用 rag/vectorstore 与 session/store 的成熟模式:
  - MySQL 记文档账(document_record,fastap2 models 建),Chroma 纯向量索引;
  - 文档级指纹(sha256+parser_version):重复上传同文件直接跳过解析重嵌入;
  - 增量更新:同 doc_id 重传 → 先按 doc_id 精准 delete 旧向量再写,不碰别人;
  - ACL 元数据进 Chroma where(检索层过滤是硬约束,不是拿到结果再筛):
      acl_scope: personal | kb | global
      owner:     personal 必填(用户隔离)
      kb_id:     企业库必填(知识库隔离)
  - embedding 复用 rag.rag 的进程级单例(不重复加载模型)。
"""
from __future__ import annotations

import logging
import threading
from typing import Dict, List, Optional

from .chunker import ChunkResult

logger = logging.getLogger("agent")

_COLLECTION = "rag_docs"
_chroma = None
_lock = threading.Lock()


def _collection():
    global _chroma
    if _chroma is None:
        import chromadb  # 延迟导入(内部带 onnxruntime,顶层导入有段错误前科)

        from .config import DOCS_PERSIST_DIR

        _chroma = chromadb.PersistentClient(path=DOCS_PERSIST_DIR)
    return _chroma.get_or_create_collection(_COLLECTION)


def embed(texts: List[str]) -> List[List[float]]:
    from allagent2.rag.config import rag_config
    from allagent2.rag.rag import _embed, _load_embedding_model

    model = _load_embedding_model(rag_config)
    return _embed(texts, model)


# ============================================================
# 写路径
# ============================================================
def upsert_document(doc_id: str, chunked: ChunkResult, acl: Dict) -> int:
    """整篇文档入库:先删旧(同 doc_id 增量)再写。返回写入子块数。

    acl = {"acl_scope","owner","kb_id","tenant_id",...},逐 child 展开进 metadata。
    """
    if not chunked.children:
        return 0
    delete_document(doc_id)
    col = _collection()
    ids = [f"{doc_id}:{c.id}" for c in chunked.children]
    metadatas = []
    for c in chunked.children:
        md = {
            "doc_id": doc_id, "child_id": c.id, "parent_id": c.parent_id,
            "page": c.page, "chunk_type": c.chunk_type,
            "heading_path": c.heading_path[:300],
            **acl,
        }
        metadatas.append({k: (str(v) if v is not None else "") for k, v in md.items()})
    try:
        col.upsert(ids=ids, documents=[c.text for c in chunked.children],
                   embeddings=embed([c.index_text for c in chunked.children]),
                   metadatas=metadatas)
        return len(ids)
    except Exception as exc:
        logger.error("[ingest] 文档 %s 向量入库失败:%s", doc_id, exc)
        raise


def delete_document(doc_id: str) -> int:
    """按 doc_id 精准删除(增量更新/删除权共用)。"""
    try:
        col = _collection()
        got = col.get(where={"doc_id": doc_id})
        ids = (got or {}).get("ids") or []
        if ids:
            col.delete(ids=ids)
        return len(ids)
    except Exception as exc:
        logger.warning("[ingest] 删除文档向量失败 doc_id=%s(可重建):%s", doc_id, exc)
        return 0


# ============================================================
# 读路径(检索侧;权限 where 由调用方组装后传入,本模块不做授权判断)
# ============================================================
def query(query_text: str, acl_where: Dict, top_k: int = 8) -> List[Dict]:
    """带 ACL where 的向量召回。acl_where 结构必须来自 identity 层组装。

    返回 [{id,text,page,chunk_type,heading_path,doc_id,score}],score 越大越相似。
    """
    try:
        col = _collection()
        res = col.query(query_embeddings=embed([query_text])[0:1],
                        n_results=top_k, where=acl_where)
    except Exception as exc:
        logger.warning("[ingest] 文档向量检索失败(该路降级):%s", exc)
        return []
    out = []
    ids = (res.get("ids") or [[]])[0]
    docs = (res.get("documents") or [[]])[0]
    dists = (res.get("distances") or [[]])[0]
    metas = (res.get("metadatas") or [[]])[0]
    for i in range(len(ids)):
        d = dists[i] if i < len(dists) else 1.0
        out.append({
            "id": ids[i], "text": docs[i], "score": 1.0 - d,
            **(metas[i] if i < len(metas) and metas[i] else {}),
        })
    out.sort(key=lambda x: x["score"], reverse=True)
    return out


# ============================================================
# 文档指纹:由 pipeline 层结合文档账本(MySQL document_record)实现,
# 本模块只管向量索引,不跨项目依赖。
# ============================================================
