"""PDF 原生解析器:零模型版面重建。

四项启发式(全部纯代码,不依赖任何 ML 模型):
  ① 标题识别:字号聚类——众数字号=正文,≥1.3 倍且行长短的行=标题
  ② 阅读顺序:y 中心排序;x 中心双峰(gap test) → 双栏,先左后右
  ③ 页眉页脚:跨页重复模式——顶部/底部固定带内 ≥50% 页面重复前缀 → 剔除
  ④ 表格提取:pdfplumber 几何检测(矢量线条/矩形聚类找单元格,非 ML)
  ⑤ 扫描页判定:每页字符 < SCAN_MIN_CHARS → needs_ocr(整篇进 OCR 分支
     或 needs_review,由 pipeline 按引擎可用性决定)

设计取舍:启发式在"规范文书"(医院病历/出院小结/化验单——格式高度规范)
上很稳;疑难件(手写/印章遮挡)将来走 MinerU adapter。
"""
from __future__ import annotations

import logging
import re
from collections import Counter
from typing import Dict, List, Optional, Tuple

from ..ir import Block, DocumentIR
from .office import _sha256

logger = logging.getLogger("agent")

SCAN_MIN_CHARS = 50          # 每页低于此字符数 → 疑似扫描页
HEADER_FOOTER_BAND = 0.08    # 页顶/页底 8% 高度带,重复模式检测范围
HEADING_SIZE_RATIO = 1.3     # 字号 ≥ 正文众数的 1.3 倍 → 候选标题
HEADING_MAX_LEN = 60         # 标题行长上限(超过更像正文大段)


def parse_pdf(path: str) -> DocumentIR:
    try:
        import fitz  # pymupdf
    except ImportError:
        # pymupdf 提供字号/坐标，是版面启发式的前提；装不上就退化成纯文本，
        # 宁可在解析报告里降级，也不要让整条上传链路上抛 ImportError。
        logger.warning("[ingest] pymupdf 未安装，PDF 退化为纯文本解析（无版面/表格）")
        return _parse_pdf_text_only(path)

    ir = DocumentIR(doc_name=path.replace("\\", "/").split("/")[-1], doc_type="pdf")
    doc = fitz.open(path)

    lines_by_page = [_extract_lines(page) for page in doc]
    body_font = _dominant_font_size(lines_by_page)
    header_texts, footer_texts = _repeated_header_footer(lines_by_page, doc)

    table_bboxes = _pdfplumber_tables(path)     # {page_idx: [bbox,...]}

    for pno, lines in enumerate(lines_by_page, start=1):
        lines = [ln for ln in lines
                 if not _match_repeated(ln, header_texts, footer_texts)]
        # 扫描页的判据是"这一页有没有文字层"，不是"字够不够多"：
        # 封面页、短报告只有二三十个字，但它们确实有文本层，不是扫描件。
        # get_text 拿不到任何 span 才是真的没字（需要 OCR）。
        if not lines or not any(ln["text"].strip() for ln in lines):
            ir.needs_ocr_pages.append(pno)
            continue
        two_col = _is_two_column(lines)
        for ln in _reading_order(lines, two_col):
            # 落在表格 bbox 内的行交给表格块,不重复出文本块
            if _in_any_bbox(ln["bbox"], table_bboxes.get(pno - 1, [])):
                continue
            size = ln["size"]
            if size >= body_font * HEADING_SIZE_RATIO and len(ln["text"]) <= HEADING_MAX_LEN:
                level = _heading_level(size, body_font)
                ir.blocks.append(Block(type="heading", text=ln["text"], page=pno,
                                       text_level=level, bbox=ln["bbox"], source="geometric"))
            else:
                ir.blocks.append(Block(type="text", text=ln["text"], page=pno,
                                       bbox=ln["bbox"], source="geometric"))

    # 表格块(pdfplumber 结果 → Markdown,每表一块原子块)
    from .table import tables_to_blocks
    ir.blocks.extend(tables_to_blocks(path, table_bboxes))

    doc.close()
    ir.sha256 = _sha256(path)
    ir.add_heading_paths()
    return ir


def _parse_pdf_text_only(path: str) -> DocumentIR:
    """pypdf 兜底解析：只取每页文本，按空行切段。

    拿不到字号与坐标，所以没有标题识别、双栏排序、页眉页脚剔除；
    表格也会退化成普通文本行。用于"没有 pymupdf 但要把内容进索引"的场景。
    """
    from pypdf import PdfReader

    ir = DocumentIR(doc_name=path.replace("\\", "/").split("/")[-1], doc_type="pdf")
    try:
        reader = PdfReader(path)
    except Exception as exc:
        logger.warning("[ingest] pypdf 打开 PDF 失败：%s", exc)
        return ir

    for pno, page in enumerate(reader.pages, start=1):
        try:
            raw = page.extract_text() or ""
        except Exception as exc:
            logger.warning("[ingest] PDF 第 %d 页取文本失败：%s", pno, exc)
            raw = ""
        text = raw.strip()
        if len(text) < SCAN_MIN_CHARS:
            ir.needs_ocr_pages.append(pno)
            continue
        for para in re.split(r"\n\s*\n", text):
            line = " ".join(para.split())
            if line:
                ir.blocks.append(Block(type="text", text=line, page=pno, source="native"))

    ir.sha256 = _sha256(path)
    ir.add_heading_paths()
    logger.info("[ingest] pypdf 兜底解析完成：%d 块，扫描页 %s", len(ir.blocks),
                ir.needs_ocr_pages)
    return ir


# ------------------------------------------------------------
# 行提取:span 聚成视觉行(同 baseline 容差),带字号/坐标
# ------------------------------------------------------------
def _extract_lines(page) -> List[Dict]:
    raw = page.get_text("dict")
    lines = []
    for block in raw.get("blocks", []):
        if block.get("type") != 0:      # 0=文本块(1=图片)
            continue
        for line in block.get("lines", []):
            text = "".join(s["text"] for s in line.get("spans", [])).strip()
            if not text:
                continue
            sizes = [s["size"] for s in line["spans"] if s["text"].strip()]
            x0, y0, x1, y1 = line["bbox"]
            lines.append({
                "text": text,
                "size": max(sizes) if sizes else 10.0,
                "bbox": [x0, y0, x1, y1],
                "y_mid": (y0 + y1) / 2,
                "x_mid": (x0 + x1) / 2,
            })
    return lines


# ------------------------------------------------------------
# ① 正文基准字号:全文档字号众数
# ------------------------------------------------------------
def _dominant_font_size(pages_lines: List[List[Dict]]) -> float:
    counter = Counter()
    for lines in pages_lines:
        for ln in lines:
            counter[round(ln["size"], 1)] += len(ln["text"])   # 按字符量加权
    return counter.most_common(1)[0][0] if counter else 10.0


def _heading_level(size: float, body: float) -> int:
    """字号越大层级越高:≥2 倍=1级,≥1.6 倍=2级,其余=3级。"""
    r = size / max(body, 1e-6)
    if r >= 2.0:
        return 1
    if r >= 1.6:
        return 2
    return 3


# ------------------------------------------------------------
# ③ 页眉页脚:页顶/页底带内跨页重复文本
# ------------------------------------------------------------
def _repeated_header_footer(pages_lines, doc) -> Tuple[set, set]:
    page_h = [page.rect.height for page in doc]
    headers, footers = Counter(), Counter()
    n = len(pages_lines)
    if n < 3:
        return set(), set()
    for pno, lines in enumerate(pages_lines):
        h = page_h[pno]
        for ln in lines:
            _, y0, _, y1 = ln["bbox"]
            if y1 < h * HEADER_FOOTER_BAND:
                headers[ln["text"][:30]] += 1
            elif y0 > h * (1 - HEADER_FOOTER_BAND):
                footers[ln["text"][:30]] += 1
    # 出现在 ≥50% 页面 → 页眉/页脚(纯页码正则在 _match_repeated 里再兜一层)
    thresh = max(2, int(n * 0.5))
    return ({t for t, c in headers.items() if c >= thresh},
            {t for t, c in footers.items() if c >= thresh})


def _match_repeated(ln: Dict, headers: set, footers: set) -> bool:
    t = ln["text"]
    prefix = t[:30]
    if prefix in headers or prefix in footers:
        return True
    import re
    return bool(re.fullmatch(r"(第?\s*\d+\s*页?(/共\s*\d+\s*页?)?)", t.strip()))


# ------------------------------------------------------------
# ② 双栏检测与阅读顺序
# ------------------------------------------------------------
def _is_two_column(lines: List[Dict]) -> bool:
    """x 中心双峰:以页面中线为界,两侧都有足量行且跨中线行少 → 双栏。"""
    if len(lines) < 8:
        return False
    x_mids = sorted(ln["x_mid"] for ln in lines)
    median = x_mids[len(x_mids) // 2]
    left = [ln for ln in lines if ln["x_mid"] < median * 0.9]
    right = [ln for ln in lines if ln["x_mid"] > median * 1.1]
    spanning = len(lines) - len(left) - len(right)
    return len(left) >= len(lines) * 0.3 and len(right) >= len(lines) * 0.3 \
        and spanning <= len(lines) * 0.15


def _reading_order(lines: List[Dict], two_col: bool) -> List[Dict]:
    if not two_col:
        return sorted(lines, key=lambda ln: (round(ln["y_mid"], 1), ln["bbox"][0]))
    # 双栏:左栏(x_mid 小)整体先于右栏,栏内按 y 排
    mid = sum(ln["x_mid"] for ln in lines) / len(lines)
    left = sorted((ln for ln in lines if ln["x_mid"] <= mid),
                  key=lambda ln: (round(ln["y_mid"], 1), ln["bbox"][0]))
    right = sorted((ln for ln in lines if ln["x_mid"] > mid),
                   key=lambda ln: (round(ln["y_mid"], 1), ln["bbox"][0]))
    return left + right


# ------------------------------------------------------------
# ④ pdfplumber 几何表格(实现挪到 table.py,便于独立测试)
# ------------------------------------------------------------
def _pdfplumber_tables(path: str) -> Dict[int, List[List[float]]]:
    from .table import extract_table_bboxes
    return extract_table_bboxes(path)


def _in_any_bbox(bbox: List[float], bboxes: List[List[float]], pad: float = 2.0) -> bool:
    x0, y0, x1, y1 = bbox
    for bx0, by0, bx1, by1 in bboxes:
        if x0 >= bx0 - pad and y0 >= by0 - pad and x1 <= bx1 + pad and y1 <= by1 + pad:
            return True
    return False
