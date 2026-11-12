"""PDF 几何表格提取(pdfplumber)与 Markdown 转换。

pdfplumber 通过矢量线条/矩形填充聚类找单元格——纯几何,无模型。
电子版 PDF(医院文书)的表格线是矢量对象,行列还原精度很高;
无线表格(仅空格对齐)退化为文本行,交给上游普通文本块(不强行成表)。
"""
from __future__ import annotations

from typing import Dict, List

from ..ir import Block

_ROW_GROUP = 15   # 超长表格按行分组,每组重复表头


def extract_table_bboxes(path: str) -> Dict[int, List[List[float]]]:
    """返回 {page_idx(0起): [表格bbox,...]}。pdfplumber 失败时静默空结果。

    【可观测】缺包/解析失败以前是纯静默 pass —— 现象是"PDF 表格变成一堆文本行"，
    排查时完全看不出是没装依赖。现在计数（component=pdf_table）后再降级。
    """
    out: Dict[int, List[List[float]]] = {}
    try:
        import pdfplumber

        with pdfplumber.open(path) as pdf:
            for idx, page in enumerate(pdf.pages):
                tables = page.find_tables()
                if tables:
                    out[idx] = [list(t.bbox) for t in tables]
    except ImportError as exc:
        _note_degraded("pdf_table", "pdfplumber_missing", exc)
    except Exception as exc:
        _note_degraded("pdf_table", type(exc).__name__, exc)
    return out


def _note_degraded(component: str, reason: str, exc: BaseException) -> None:
    """降级计数（走 metrics_hook，不反向依赖 fastap2）。"""
    try:
        from allagent2.guance.metrics_hook import meter

        meter("agent_degraded_total", 1, component=component, reason=reason)
    except Exception:
        pass


def tables_to_blocks(path: str, bboxes_by_page: Dict[int, List[List[float]]]) -> List[Block]:
    """把 pdfplumber 检出的表格抽成 Markdown 表格块(原子块,不参与切分)。"""
    blocks: List[Block] = []
    if not bboxes_by_page:
        return blocks
    try:
        import pdfplumber

        with pdfplumber.open(path) as pdf:
            for idx, bboxes in bboxes_by_page.items():
                page = pdf.pages[idx]
                tables = page.find_tables()
                for t in tables:
                    rows = t.extract() or []
                    rows = [[("" if c is None else str(c).strip()) for c in r] for r in rows]
                    rows = [r for r in rows if any(r)]
                    if not rows:
                        continue
                    blocks.extend(_long_table_blocks(rows, page=idx + 1))
    except Exception:
        pass
    return blocks


def _long_table_blocks(rows: List[List[str]], page: int) -> List[Block]:
    """超长表格按行分组(每组重复表头),普通表格整块返回。"""
    width = max(len(r) for r in rows)
    rows = [r + [""] * (width - len(r)) for r in rows]
    if len(rows) <= _ROW_GROUP + 1:
        return [Block(type="table", text=_md(rows), page=page, source="geometric")]
    out = []
    header = rows[0]
    for start in range(1, len(rows), _ROW_GROUP):
        group = [header] + rows[start:start + _ROW_GROUP]
        out.append(Block(type="table", text=_md(group), page=page, source="geometric"))
    return out


def _md(rows: List[List[str]]) -> str:
    lines = ["| " + " | ".join(rows[0]) + " |",
             "| " + " | ".join(["---"] * len(rows[0])) + " |"]
    lines += ["| " + " | ".join(r) + " |" for r in rows[1:]]
    return "\n".join(lines)
