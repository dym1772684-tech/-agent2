"""上传文件的文本解析：把 PDF / Word / Excel / 纯文本统一抽成字符串。

只负责"抽文本"，不做切块与向量化（那是 rag 的事）。
扫描件 PDF 抽不出字时**明确报错**，让前端提示需要 OCR，而不是静默返回空串。
"""
from __future__ import annotations

import io
from pathlib import Path

ALLOWED_EXTS = {".pdf", ".docx", ".xlsx", ".txt", ".md", ".csv"}
MAX_BYTES = 10 * 1024 * 1024          # 单文件 10MB


class ParseError(ValueError):
    """解析失败（类型不支持 / 内容为空 / 文件损坏）。"""


def ext_of(filename: str) -> str:
    return Path(filename or "").suffix.lower()


def parse_bytes(filename: str, data: bytes) -> str:
    """按扩展名把字节流解析成纯文本。失败一律抛 ParseError。"""
    ext = ext_of(filename)
    if ext not in ALLOWED_EXTS:
        raise ParseError(
            f"不支持的文件类型：{ext or '(无扩展名)'}；仅支持 "
            + "、".join(sorted(ALLOWED_EXTS))
        )
    if not data:
        raise ParseError("文件是空的")
    if len(data) > MAX_BYTES:
        raise ParseError(f"文件超过 {MAX_BYTES // 1024 // 1024}MB")

    if ext in (".txt", ".md", ".csv"):
        return data.decode("utf-8", errors="ignore")

    if ext == ".pdf":
        from pypdf import PdfReader

        try:
            reader = PdfReader(io.BytesIO(data))
            pages = [(page.extract_text() or "") for page in reader.pages]
        except Exception as exc:  # 损坏 / 加密
            raise ParseError(f"PDF 解析失败：{exc}") from exc
        text = "\n".join(pages).strip()
        if not text:
            raise ParseError("PDF 里没有可提取的文字（可能是扫描件，需要 OCR）")
        return text

    if ext == ".docx":
        import docx

        try:
            doc = docx.Document(io.BytesIO(data))
        except Exception as exc:
            raise ParseError(f"Word 解析失败：{exc}") from exc
        parts = [p.text for p in doc.paragraphs]
        for table in doc.tables:
            for row in table.rows:
                parts.append("\t".join(cell.text for cell in row.cells))
        return "\n".join(p for p in parts if p and p.strip())

    if ext == ".xlsx":
        import openpyxl

        try:
            wb = openpyxl.load_workbook(io.BytesIO(data), read_only=True, data_only=True)
        except Exception as exc:
            raise ParseError(f"Excel 解析失败：{exc}") from exc
        lines = []
        for ws in wb.worksheets:
            lines.append(f"## {ws.title}")
            for row in ws.iter_rows(values_only=True):
                if row is None:
                    continue
                cells = ["" if c is None else str(c) for c in row]
                if any(c.strip() for c in cells):
                    lines.append("\t".join(cells))
        return "\n".join(lines)

    raise ParseError("无法解析该文件")
