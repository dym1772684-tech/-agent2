"""摄取切分:IR → 子块(child,检索粒度)+父块(parent,返回粒度)。

与 rag/splitter 的关系:算法内核一致(符号递归+父子token),但输入不同——
rag/splitter 吃整篇 Markdown(全局知识库),本模块吃结构化 IR(用户文档),
因此多三件事:
  ① heading 分段:标题块开启新 section,heading_path 逐块携带
  ② 表格原子块:表格自成父块,超长按行组(表头重复),不参与符号递归
  ③ overlap:正文子块间滑窗重叠(切在符号边界),消除边界语义断裂
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .ir import Block, DocumentIR

# 与 rag.config 对齐的默认值(不 import RAGConfig,避免摄取层依赖检索层配置)
CHILD_TOKENS = 128
PARENT_RATIO = 1.8
OVERLAP_TOKENS = 20
TABLE_ROW_GROUP = 15
_SEPARATORS = ("\n\n", "\n", "。", "；", "，", "、")


@dataclass
class ChildChunk:
    id: str
    text: str               # 子块原文
    index_text: str         # 检索文本 = 标题路径 + 子块正文(带主题信号)
    parent_id: str
    page: int
    chunk_type: str         # text | table
    heading_path: str
    metadata: Dict = field(default_factory=dict)


@dataclass
class ParentChunk:
    id: str
    text: str
    page: int
    chunk_type: str
    heading_path: str
    doc_name: str


@dataclass
class ChunkResult:
    children: List[ChildChunk] = field(default_factory=list)
    parents: List[ParentChunk] = field(default_factory=list)


def _tokens(text: str) -> int:
    """粗略 token 估算:中文≈1字1token,英文按4字符1token。"""
    if not text:
        return 0
    cjk = sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff")
    return cjk + (len(text) - cjk) // 4 + 1


# ============================================================
# 主入口
# ============================================================
def chunk_document(ir: DocumentIR) -> ChunkResult:
    result = ChunkResult()
    _split_into_sections(ir, result)
    return result


def _split_into_sections(ir: DocumentIR, out: ChunkResult) -> None:
    """按标题把块分组为 section;无标题文档整体一个 section。"""
    sections: List[Tuple[str, List[Block]]] = []
    path: List[str] = []
    current: List[Block] = []
    current_path = ""

    def flush():
        if current:
            sections.append((current_path, list(current)))

    for b in ir.blocks:
        if b.type == "heading":
            flush()
            current = []
            # 维护层级栈:1级清空栈,2级保留1级,依此类推
            while len(path) >= b.text_level:
                path.pop()
            path.append(b.text)
            current_path = " > ".join(path)
        else:
            current.append(b)
    flush()

    if not sections:
        return
    prefix = ir.doc_name
    for si, (hpath, blocks) in enumerate(sections):
        hp = f"{prefix} > {hpath}" if hpath else prefix
        for b in blocks:
            if b.type == "table":
                _emit_table(b, hp, out)
            elif b.type in ("text", "heading"):
                _emit_text(b, hp, out)


# ============================================================
# 表格:原子父块(超长按行组,表头重复)
# ============================================================
def _emit_table(b: Block, hpath: str, out: ChunkResult) -> None:
    lines = [ln for ln in b.text.splitlines() if ln.strip()]
    pid = f"p-table-{b.page}-{abs(hash(b.text[:60])) % 100000}"
    groups = ([lines] if len(lines) <= TABLE_ROW_GROUP + 1
              else [[lines[0]] + lines[i:i + TABLE_ROW_GROUP]
                    for i in range(1, len(lines), TABLE_ROW_GROUP)])
    for gi, g in enumerate(groups):
        p = ParentChunk(id=f"{pid}-{gi}", text="\n".join(g), page=b.page,
                        chunk_type="table", heading_path=hpath, doc_name="")
        # child 的 index_text = 标题路径 + 表头 + 首行数据(让向量"闻到"表内容)
        header = next((ln for ln in g if ln.startswith("|") and "---" not in ln), "")
        first_data = next((ln for ln in g if ln.startswith("|") and "---" not in ln
                           and ln is not header), "")
        idx_text = f"{hpath}\n{header}\n{first_data}".strip()
        out.parents.append(p)
        out.children.append(ChildChunk(
            id=p.id.replace("p-", "c-"), text=p.text, index_text=idx_text[:800],
            parent_id=p.id, page=b.page, chunk_type="table", heading_path=hpath,
            metadata={"table_rows": len(g)}))


# ============================================================
# 正文:符号递归子块 + 父块聚合 + overlap
# ============================================================
def _split_by_separators(text: str, budget: int) -> List[str]:
    """符号递归切分(粗到细),返回 ≤budget token 的片段。"""
    if _tokens(text) <= budget:
        return [text] if text.strip() else []
    for sep in _SEPARATORS:
        if sep in text:
            parts = [p for p in text.split(sep) if p.strip()]
            if len(parts) > 1:
                out: List[str] = []
                buf = ""
                for p in parts:
                    piece = (p + ("" if sep in ("\n",) else sep)).strip()
                    if _tokens(piece) > budget:      # 递归细化
                        out.extend(_split_by_separators(piece, budget))
                    elif _tokens(buf) + _tokens(piece) > budget:
                        if buf:
                            out.append(buf.strip())
                        buf = piece
                    else:
                        buf = f"{buf}{sep}{piece}" if buf else piece
                if buf.strip():
                    out.append(buf.strip())
                return out
    # 无分隔符可切(极端) → 硬切
    step = budget * 2
    return [text[i:i + step] for i in range(0, len(text), step)] if text.strip() else []


def _add_overlap(pieces: List[str], overlap_tokens: int) -> List[str]:
    """相邻子块带重叠:把前块尾部 overlap_tokens 追加到后块头部(切在符号边界)。"""
    if len(pieces) <= 1 or overlap_tokens <= 0:
        return pieces
    out = [pieces[0]]
    for prev, cur in zip(pieces, pieces[1:]):
        tail = _tail_slice(prev, overlap_tokens)
        out.append(f"{tail}{cur}" if tail else cur)
    return out


def _tail_slice(text: str, n_tokens: int) -> str:
    """取 text 尾部约 n_tokens 的内容,对齐到符号起点(不切断词)。"""
    chars = n_tokens * 2
    tail = text[-chars:] if len(text) > chars else text
    m = re.search(r"[。；，\n]", tail)
    return tail[m.start():] if m else tail


def _emit_text(b: Block, hpath: str, out: ChunkResult) -> None:
    pieces = _split_by_separators(b.text, CHILD_TOKENS)
    pieces = _add_overlap(pieces, OVERLAP_TOKENS)
    if not pieces:
        return
    pid = f"p-{b.page}-{abs(hash(b.text[:60])) % 100000}"
    # 父块:把同 section 的相邻正文聚合(简化:本块全部子块合并即父块)
    parent_text = "\n".join(pieces)
    out.parents.append(ParentChunk(
        id=pid, text=parent_text, page=b.page, chunk_type="text",
        heading_path=hpath, doc_name=""))
    for ci, piece in enumerate(pieces):
        out.children.append(ChildChunk(
            id=f"{pid}-c{ci}", text=piece,
            index_text=f"{hpath}\n{piece}"[:800], parent_id=pid,
            page=b.page, chunk_type="text", heading_path=hpath,
            metadata={"overlap": int(ci > 0)}))
