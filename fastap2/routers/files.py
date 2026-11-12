"""文件上传端点：解析成文本 → 落盘 → 返回 file_id 供 /agent/chat 引用。

安全边界：
1. 类型白名单 + 10MB 上限（解析器里校验）
2. 落盘用 uuid 命名，绝不用用户文件名拼路径
3. 按 user_id 分目录，读取时校验归属（拿到别人的 file_id 也读不到）
"""
from __future__ import annotations

import logging
import os

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool

from core.acl import PERM_AGENT_CHAT, require_permission
from core.file_parser import MAX_BYTES, ParseError, parse_bytes
from core.file_store import save, uploads_dir
from core.responses import ApiResponse

logger = logging.getLogger("agent.files")

files_router = APIRouter(prefix="/agent", tags=["文件上传"])


@files_router.post("/files")
async def upload_file(
    file: UploadFile = File(..., description="PDF/DOCX/XLSX/TXT/MD/CSV（≤10MB）"),
    current=Depends(require_permission(PERM_AGENT_CHAT)),
):
    """上传并解析一个文件，返回 file_id（原文件不保存，只留解析后的文本）。"""
    data = await file.read()
    if len(data) > MAX_BYTES:
        raise HTTPException(413, f"文件超过 {MAX_BYTES // 1024 // 1024}MB")

    try:
        text = parse_bytes(file.filename or "", data).strip()
    except ParseError as exc:
        raise HTTPException(400, str(exc))
    if not text:
        raise HTTPException(400, "解析后没有可读文本（可能是扫描件，需要 OCR）")

    meta = save(str(current.user_id), file.filename or "未命名", text)
    logger.info(
        "[file] 上传成功 user=%s name=%s chars=%d fid=%s",
        current.user_id, meta["name"], meta["chars"], meta["file_id"],
    )

    ingest = await _ingest_original(current.user_id, meta["file_id"], file.filename, data)
    return ApiResponse.success(data={
        "file_id": meta["file_id"],
        "name": meta["name"],
        "chars": meta["chars"],
        "preview": text[:200],
        "ingested": bool(ingest and ingest.get("status") == "parsed"),
        "ingest": {
            "status": (ingest or {}).get("status", "skipped"),
            "children": (ingest or {}).get("children"),
            "confidence": (ingest or {}).get("confidence"),
            "reason": (ingest or {}).get("reason") or (ingest or {}).get("error"),
        },
    })


async def _ingest_original(user_id, file_id: str, filename: str | None, data: bytes):
    """把原文落临时文件 → 摄取（切分/ACL/向量） → 删临时文件。

    ingest_file 是同步阻塞的（解析 + 切分 + embedding），必须丢线程池，
    否则会把事件循环卡住（项目里踩过一次的坑）。
    摄取失败只降级：文件照样可用 file_id 引用，只是进不了检索。
    """
    suffix = os.path.splitext(filename or "")[1] or ".bin"
    tmp_path = os.path.join(uploads_dir(str(user_id)), f"{file_id}.orig{suffix}")
    try:
        with open(tmp_path, "wb") as f:
            f.write(data)
        from allagent2.ingestion.pipeline import ingest_file

        # force=True：用户主动上传即视为确认入库，否则短文档会被置信度门挡住
        report = await run_in_threadpool(
            ingest_file, tmp_path, file_id,
            {"acl_scope": "personal", "owner": str(user_id)}, True,
        )
        logger.info("[file] 摄取 fid=%s status=%s children=%s",
                    file_id, report.get("status"), report.get("children"))
        return report
    except Exception as exc:
        logger.warning("[file] 摄取失败（不影响本次上传）：%s", exc)
        return {"status": "failed", "error": str(exc)[:200]}
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass
