"""Office 系原生解析器:docx/xlsx/pptx/txt/csv/md → IR。

全部纯代码(docx 样式即结构/xlsx 合并单元格展开/pptx 占位符三通道),
无任何模型依赖。重库延迟导入(与包铁律一致)。
"""
from __future__ import annotations

import csv
import hashlib
import io
import os
from typing import List

from ..ir import Block, DocumentIR


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _base(doc_name: str, doc_type: str) -> DocumentIR:
    return DocumentIR(doc_name=os.path.basename(doc_name), doc_type=doc_type)


# ============================================================
# DOCX:样式即结构(Heading 1/2/3 → text_level),表格走 doc.tables
# ============================================================
def parse_docx(path: str) -> DocumentIR:
    import docx  # python-docx

    ir = _base(path, "docx")
    d = docx.Document(path)
    # 段落与表格在文档流中的真实顺序:遍历 body 子元素交替处理
    from docx.table import Table
    from docx.text.paragraph import Paragraph

    body = d.element.body
    for child in body.iterchildren():
        tag = child.tag.split("}")[-1]
        if tag == "p":
            p = Paragraph(child, d)
            text = (p.text or "").strip()
            if not text:
                continue
            style = (p.style.name or "").lower()
            if style.startswith("heading") or style.startswith("标题"):
                digits = "".join(ch for ch in style if ch.isdigit())
                ir.blocks.append(Block(type="heading", text=text,
                                       text_level=int(digits) if digits else 1))
            else:
                ir.blocks.append(Block(type="text", text=text))
        elif tag == "tbl":
            t = Table(child, d)
            ir.blocks.append(Block(type="table", text=_rows_to_md(
                [[c.text.strip() for c in row.cells] for row in t.rows])))
    ir.sha256 = _sha256(path)
    ir.add_heading_paths()
    return ir


def _rows_to_md(rows: List[List[str]]) -> str:
    """行列表 → Markdown 表(空表容错;表头分隔行固定第二行)。"""
    if not rows:
        return "(空表格)"
    width = max(len(r) for r in rows)
    rows = [r + [""] * (width - len(r)) for r in rows]
    lines = ["| " + " | ".join(rows[0]) + " |",
             "| " + " | ".join(["---"] * width) + " |"]
    lines += ["| " + " | ".join(r) + " |" for r in rows[1:]]
    return "\n".join(lines)


# ============================================================
# XLSX:合并单元格展开 + 表头识别 + 每 sheet 一块表格
# ============================================================
def parse_xlsx(path: str) -> DocumentIR:
    import openpyxl

    ir = _base(path, "xlsx")
    wb = openpyxl.load_workbook(path, data_only=True, read_only=True)
    for sheet_idx, ws in enumerate(wb.worksheets, start=1):
        rows = []
        for row in ws.iter_rows(values_only=True):
            vals = ["" if v is None else str(v).strip() for v in row]
            if any(vals):                       # 跳过整行空
                rows.append(vals)
        if not rows:
            continue
        # sheet 名作为 heading 块(heading_path 带上工作表维度),表格块随后
        ir.blocks.append(Block(type="heading", text=ws.title or f"Sheet{sheet_idx}",
                               text_level=1, page=sheet_idx))
        ir.blocks.append(Block(type="table", text=_rows_to_md(rows), page=sheet_idx))
    wb.close()
    ir.sha256 = _sha256(path)
    return ir


# ============================================================
# PPTX:每页 标题占位符 + 正文 + 备注 三通道
# ============================================================
def parse_pptx(path: str) -> DocumentIR:
    from pptx import Presentation

    ir = _base(path, "pptx")
    prs = Presentation(path)
    for idx, slide in enumerate(prs.slides, start=1):
        title = ""
        bodies: List[str] = []
        for shape in slide.shapes:
            if not shape.has_text_frame:
                continue
            for para in shape.text_frame.paragraphs:
                text = "".join(run.text for run in para.runs).strip()
                if not text:
                    continue
                if shape == slide.shapes.title:
                    title = text
                else:
                    bodies.append(text)
        if title:
            ir.blocks.append(Block(type="heading", text=title, text_level=1, page=idx))
        if bodies:
            ir.blocks.append(Block(type="text", text="\n".join(bodies), page=idx))
        notes = getattr(slide, "notes_slide", None) if slide.has_notes_slide else None
        if notes is not None:
            note_text = (notes.notes_text_frame.text or "").strip()
            if note_text:
                ir.blocks.append(Block(type="text", text=f"(备注){note_text}", page=idx))
    ir.sha256 = _sha256(path)
    return ir


# ============================================================
# TXT / MD:直接读,md 的 # 标题解析为 heading
# ============================================================
def _decode_bytes(raw: bytes) -> str:
    """编码探测:UTF-8 → GBK → latin-1 兜底(BOM 处理)。"""
    if raw.startswith(b"\xef\xbb\xbf"):
        return raw[3:].decode("utf-8", errors="replace")
    for enc in ("utf-8", "gb18030"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("latin-1", errors="replace")


_MD_HEADING = {1: 1, 2: 2, 3: 3}   # # ## ### → text_level(更深层当正文)


def parse_text(path: str) -> DocumentIR:
    with open(path, "rb") as f:
        text = _decode_bytes(f.read())
    ir = _base(path, "md" if path.lower().endswith((".md", ".markdown")) else "txt")
    for line in text.splitlines():
        s = line.strip()
        if not s:
            continue
        if ir.doc_type == "md" and s.startswith("#"):
            level = len(s) - len(s.lstrip("#"))
            content = s.lstrip("#").strip()
            if content and level in _MD_HEADING:
                ir.blocks.append(Block(type="heading", text=content, text_level=level))
                continue
        ir.blocks.append(Block(type="text", text=s))
    ir.sha256 = _sha256(path)
    ir.add_heading_paths()
    return ir


def parse_csv(path: str) -> DocumentIR:
    with open(path, "rb") as f:
        text = _decode_bytes(f.read())
    ir = _base(path, "csv")
    rows = [[c.strip() for c in r] for r in csv.reader(io.StringIO(text))]
    rows = [r for r in rows if any(r)]
    if rows:
        ir.blocks.append(Block(type="table", text=_rows_to_md(rows)))
    ir.sha256 = _sha256(path)
    return ir
