"""魔数识别 + 解析器级联路由。

不信扩展名(改名绕过白名单是常见绕法),读文件头魔数判真实类型;
级联策略:原生层(纯代码) → OCR 层(未装引擎→needs_review) →
MinerU(配置了才启用)。路由器返回解析函数,不直接执行。
"""
from __future__ import annotations

import os
from typing import Callable, Optional

from .ir import DocumentIR

# 魔数签名(前 8 字节足够区分本系统支持的格式)
_MAGIC = [
    (b"%PDF-", "pdf"),
    (b"PK\x03\x04", "ooxml"),   # docx/xlsx/pptx 共用 zip 头,再看内部结构
    (b"\xd0\xcf\x11\xe0", "ole"),  # 旧版 .doc/.xls/.ppt(提示转存新格式)
    (b"\x89PNG", "image"),
    (b"\xff\xd8\xff", "image"),
    (b"GIF8", "image"),
    (b"BM", "image"),
]


def sniff_type(path: str) -> str:
    """魔数识别真实类型。返回标准类型名;无法识别返回 'unknown'。"""
    try:
        with open(path, "rb") as f:
            head = f.read(8)
    except OSError:
        return "unknown"
    for magic, kind in _MAGIC:
        if head.startswith(magic):
            if kind == "ooxml":
                return _ooxml_subtype(path)
            return kind
    # 纯文本类:魔数判不了,检查可打印率
    if _looks_text(path):
        return _text_subtype(path)
    return "unknown"


def _ooxml_subtype(path: str) -> str:
    """zip 容器内看目录结构区分 docx/xlsx/pptx。"""
    import zipfile

    try:
        with zipfile.ZipFile(path) as z:
            names = set(z.namelist())
        if "word/document.xml" in names:
            return "docx"
        if "xl/workbook.xml" in names:
            return "xlsx"
        if any(n.startswith("ppt/") for n in names):
            return "pptx"
    except Exception:
        pass
    return "unknown"


def _looks_text(path: str) -> bool:
    """可打印字符占比 >90% 视为文本文件。"""
    try:
        with open(path, "rb") as f:
            sample = f.read(4096)
        if not sample:
            return True
        printable = sum(1 for b in sample if 32 <= b < 127 or b in (9, 10, 13) or b >= 0x80)
        return printable / len(sample) > 0.9
    except OSError:
        return False


def _text_subtype(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".csv":
        return "csv"
    if ext in (".md", ".markdown"):
        return "md"
    return "txt"


def route(path: str) -> Optional[Callable[[str], DocumentIR]]:
    """级联路由:返回该类型的原生解析函数。

    返回 None = 无原生路径(image 未装 OCR 引擎时在 pipeline 层走
    needs_review;ole 旧格式提示转存;unknown 拒绝)。
    """
    kind = sniff_type(path)
    table = {
        "pdf": "allagent2.ingestion.parsers.pdf:parse_pdf",
        "docx": "allagent2.ingestion.parsers.office:parse_docx",
        "xlsx": "allagent2.ingestion.parsers.office:parse_xlsx",
        "pptx": "allagent2.ingestion.parsers.office:parse_pptx",
        "txt": "allagent2.ingestion.parsers.office:parse_text",
        "md": "allagent2.ingestion.parsers.office:parse_text",
        "csv": "allagent2.ingestion.parsers.office:parse_csv",
    }
    entry = table.get(kind)
    if entry is None:
        return None
    mod_name, func_name = entry.split(":")
    import importlib

    return getattr(importlib.import_module(mod_name), func_name)
