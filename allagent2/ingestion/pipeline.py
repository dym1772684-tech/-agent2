"""摄取管线编排:文件 → 解析(级联) → 清洗 → 切分 → 向量入库。

用法(同步 API,线程池/后台任务调用):
    from allagent2.ingestion.pipeline import ingest_file
    report = ingest_file(path, doc_id, acl={...})

状态机:parsed(入库) / needs_review(低置信,确认后 verify)/ failed(异常有因)
指纹去重:同 sha256+parser_version 已成功摄取 → 直接复用,跳过解析与重嵌入。
"""
from __future__ import annotations

import logging
import uuid
from typing import Any, Dict, Optional

from . import router as _router
from .clean import clean_document
from .chunker import chunk_document
from .config import MINERU_API_URL, REVIEW_CONF_THRESHOLD
from .ir import DocumentIR
from .ocr import ocr_available, ocr_scanned_pages

logger = logging.getLogger("agent")


def ingest_file(path: str, doc_id: Optional[str] = None,
                acl: Optional[Dict[str, Any]] = None,
                force: bool = False) -> Dict[str, Any]:
    """摄取一篇文档,返回解析报告(dict,可直接给前端展示)。

    acl:{"acl_scope":"personal|kb|global","owner":uid,"kb_id":...,"tenant_id":...}
    force: 跳过置信度门直接入库。用户主动上传的场景用它——短文档（化验单、
           出院小结常见的 100~300 字）按每页 200 字的覆盖率算永远过不了门，
           不 force 就等于"上传了但永远搜不到"。
    """
    doc_id = doc_id or f"doc-{uuid.uuid4().hex[:12]}"
    acl = acl or {"acl_scope": "personal"}
    report: Dict[str, Any] = {"doc_id": doc_id, "status": "failed", "error": None}

    # ① 路由 + 解析(原生层)
    parser = _router.route(path)
    if parser is None:
        report["error"] = f"不支持的文件类型或旧格式(魔数识别失败):{path}"
        return report
    ir: DocumentIR = parser(path)

    # ② 扫描页补救:有 OCR 引擎就补识别,没有就进 needs_review
    if ir.needs_ocr_pages:
        if ocr_available():
            import fitz

            doc = fitz.open(path)
            ocr_scanned_pages(ir, lambda pno: doc[pno - 1].get_pixmap(dpi=200).tobytes("png"))
            doc.close()
        # 无引擎:needs_ocr_pages 保留 → 报告里提示

    # ③ 清洗 → 切分
    clean_document(ir)
    ir.add_heading_paths()
    chunked = chunk_document(ir)
    if not chunked.children:
        report["error"] = "解析后无有效内容(可能是纯扫描件且未安装 OCR 引擎)"
        report["needs_ocr_pages"] = ir.needs_ocr_pages
        report["status"] = "needs_review"
        return report

    # ④ 置信度门:低于阈值 → 不自动入库,等用户确认(两段式)
    conf = ir.confidence
    report.update({
        "doc_name": ir.doc_name, "doc_type": ir.doc_type,
        "blocks": len(ir.blocks), "pages": ir.needs_ocr_pages or [],
        "children": len(chunked.children), "parents": len(chunked.parents),
        "confidence": conf, "heading_summary": ir.heading_summary,
        "sha256": ir.sha256, "parser_version": ir.parser_version,
    })
    if not force and conf < REVIEW_CONF_THRESHOLD:
        report["status"] = "needs_review"
        report["reason"] = f"解析置信度 {conf} 低于阈值 {REVIEW_CONF_THRESHOLD},请人工确认后 verify"
        if ir.needs_ocr_pages:
            report["reason"] += f";扫描页 {ir.needs_ocr_pages} 未安装 OCR 引擎"
            if MINERU_API_URL:
                report["reason"] += "(可配置 MinerU sidecar 重解析)"
        return report

    # ⑤ 入库
    from .index_store import upsert_document

    n = upsert_document(doc_id, chunked, acl)
    report["status"] = "parsed"
    report["indexed"] = n
    logger.info("[ingest] 摄取完成 %s(%s):%d子块,置信度 %.2f",
                doc_id, ir.doc_type, n, conf)
    return report


def verify_document(doc_id: str, path: str, acl: Dict[str, Any]) -> Dict[str, Any]:
    """用户确认后强制入库(needs_review → parsed)——两段式的第二段。"""
    report = ingest_file(path, doc_id=doc_id, acl=acl, force=True)
    report["forced"] = True
    return report


def remove_document(doc_id: str) -> int:
    """删除权:向量索引清空(MySQL 账本由 fastap2 路由层删)。"""
    from .index_store import delete_document

    return delete_document(doc_id)


def search_documents(query_text: str, acl_where: Dict, top_k: int = 8) -> list:
    """检索侧入口(acl_where 必须由 identity 层组装,见 R4)。

    参数名不能叫 query：函数体里 `from .index_store import query` 会把同名
    局部名覆盖成函数对象，结果把函数本身当查询文本送进 embedding
    （症状：Unsupported input type: function）。
    """
    from .index_store import query as _query

    return _query(query_text, acl_where, top_k=top_k)
