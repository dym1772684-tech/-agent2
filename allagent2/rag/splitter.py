"""父子切分器：结构切分 → 符号递归 → 父子配对 → 元数据串行。

============================================================
怎么切的（你给的四步思路的落地）
------------------------------------------------------------
1. 【结构切分】按 markdown 标题层级（# ## ###）先切成 section，
   同时捕获 heading_path（如 "养老养生知识库 > 血压管理"）。
   文档性质不同标准不同：markdown 吃标题，普通文章跳过这步直接进 2。

2. 【符号递归】section 内按 ["\\n\\n", "\\n", "。", "，"] 逐级尝试：
   先按段落切，块还超限就降级到句子，再超降到逗号。
   每一级都保证：切出来的块 ≤ CHILD_TOKENS（token 用 count_tokens 计）。

3. 【父子配对】父块 = 相邻子块滑动合并，目标 token 量 =
   CHILD_TOKENS × PARENT_RATIO（你定的 1.5~2.0 倍区间，默认 1.8）。
   **检索用子块（粒度细、TF/向量信号强），返回父块（上下文完整）**。

4. 【元数据串行】每个块挂 {doc_source, heading_path, section_idx,
   child_idx, parent_id, tokens}，元数据是串起前三个思路的线索：
   检索命中子块 → 通过 parent_id 找到父块 → 通过 heading_path 标注来源。

改的意义（对照原来的 content.split("\\n\\n")）：
  - 原切分对普通长文章会切出巨大块，embedding 语义被稀释；
  - 无父子结构：要么块小缺上下文，要么块大检索弱，两头不能兼得；
  - 无元数据：来源标注靠正则从正文里抠，脆弱且不可靠。
============================================================
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from allagent2.guance.logging_helper import logger
from allagent2.guance.token_accounting import count_tokens
from .config import rag_config

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)

# 知识库条目标题里的来源标注格式：## 血压管理（来源：养老养生知识库·血压管理）
_SOURCE_IN_HEADING = re.compile(r"[（(]来源[：:](.+?)[）)]")


@dataclass
class ChildChunk:
    """子块：检索的最小单元（粒度细，TF/向量信号强）。"""

    id: str                      # 如 "s02c1"（第2节第1子块）
    text: str
    tokens: int
    parent_id: str               # 所属父块
    doc_source: str              # 文档名
    heading_path: str            # "一级标题 > 二级标题"
    section_idx: int
    child_idx: int

    def metadata(self) -> Dict[str, Any]:
        return {
            "doc_source": self.doc_source,
            "heading_path": self.heading_path,
            "section_idx": self.section_idx,
            "child_idx": self.child_idx,
            "parent_id": self.parent_id,
            "tokens": self.tokens,
        }

    @property
    def index_text(self) -> str:
        """进索引（词法分词 + 向量 embedding）用的文本：标题末段 + 正文。

        为什么要拼标题：中文分词错位的经典坑——用户问『血压高』，jieba
        切成『血压高』一个词，正文里却是『高血压』一个词，互相匹配不上；
        而标题『血压管理』能切出『血压』。把标题拼进索引文本，等于给每个
        子块补上了一条强主题信号（对向量路同样有效：孤立的子块片段
        『情绪波动、便秘用力…』不带主题词，embedding 语义是漂的）。
        """
        head = self.heading_path.split(" > ")[-1] if self.heading_path else ""
        return (head + "\n" + self.text) if head else self.text


@dataclass
class ParentChunk:
    """父块：返回给 LLM 的单元（上下文完整）。"""

    id: str
    text: str
    tokens: int
    heading_path: str
    doc_source: str
    child_ids: List[str] = field(default_factory=list)


@dataclass
class SplitResult:
    children: List[ChildChunk]
    parents: List[ParentChunk]

    @property
    def child_texts(self) -> List[str]:
        return [c.text for c in self.children]

    @property
    def child_index_texts(self) -> List[str]:
        """进索引的文本（带标题），见 ChildChunk.index_text。"""
        return [c.index_text for c in self.children]

    @property
    def parent_texts(self) -> List[str]:
        return [p.text for p in self.parents]

    def stats(self) -> Dict[str, Any]:
        """块大小分布（打样验证用：看粒度是否合理）。"""
        toks = [c.tokens for c in self.children]
        ptoks = [p.tokens for p in self.parents]
        return {
            "sections": len({c.section_idx for c in self.children}),
            "children": len(self.children),
            "parents": len(self.parents),
            "child_tokens_min": min(toks) if toks else 0,
            "child_tokens_max": max(toks) if toks else 0,
            "child_tokens_avg": round(sum(toks) / len(toks), 1) if toks else 0,
            "parent_tokens_min": min(ptoks) if ptoks else 0,
            "parent_tokens_max": max(ptoks) if ptoks else 0,
            "parent_tokens_avg": round(sum(ptoks) / len(ptoks), 1) if ptoks else 0,
        }


# ------------------------------------------------------------
# 第 1 步：结构切分（markdown 标题 → section）
# ------------------------------------------------------------
def _split_sections(md_text: str) -> List[Dict[str, Any]]:
    """按标题层级切 section，返回 [{heading_path, source, body, idx}]。

    无标题的文档（普通文章）整个作为 1 个 section，
    heading_path 为空串，后续步骤完全一致。
    """
    sections: List[Dict[str, Any]] = []
    matches = list(_HEADING_RE.finditer(md_text))
    if not matches:
        return [{
            "heading_path": "",
            "source": "",
            "body": md_text,
            "idx": 0,
        }]

    # 文档开头、第一个标题之前的内容（如果有）单独成节
    if matches[0].start() > 0:
        head = md_text[: matches[0].start()].strip()
        if head:
            sections.append({"heading_path": "", "source": "", "body": head, "idx": len(sections)})

    # 标题层级栈：维护 "一级 > 二级 > 三级" 路径
    stack: List[tuple[int, str]] = []
    for i, m in enumerate(matches):
        level = len(m.group(1))
        title = m.group(2).strip()
        while stack and stack[-1][0] >= level:
            stack.pop()
        stack.append((level, title))
        heading_path = " > ".join(t for _, t in stack)

        body = md_text[m.end(): matches[i + 1].start() if i + 1 < len(matches) else len(md_text)]
        src_m = _SOURCE_IN_HEADING.search(title)
        sections.append({
            "heading_path": heading_path,
            "source": src_m.group(1).strip() if src_m else title,
            "body": body.strip(),
            "idx": len(sections),
        })
    return [s for s in sections if s["body"]]


# ------------------------------------------------------------
# 第 2 步：符号递归切分（\n\n → \n → 。 → ，）
# ------------------------------------------------------------
def _split_by_separator(text: str, sep: str) -> List[str]:
    parts = text.split(sep) if sep == "\n\n" else text.split(sep)
    return [p.strip() for p in parts if p.strip()]


def _recursive_split(
    text: str,
    max_tokens: int,
    separators: tuple,
    depth: int = 0,
) -> List[str]:
    """递归符号切分：逐级降低切分符号，直到每块 ≤ max_tokens。

    到最后一级还超限（整段无符号超长）就硬切，保证不出现失控巨块。
    """
    if count_tokens(text) <= max_tokens or depth >= len(separators):
        # 硬兜底：无符号可切时按字符数近似硬切
        if count_tokens(text) <= max_tokens:
            return [text]
        return _hard_split(text, max_tokens)

    parts = _split_by_separator(text, separators[depth])
    out: List[str] = []
    buffer = ""
    for p in parts:
        # 单个 part 自身超限 → 递归降级到下一级符号
        if count_tokens(p) > max_tokens:
            if buffer:
                out.append(buffer)
                buffer = ""
            out.extend(_recursive_split(p, max_tokens, separators, depth + 1))
            continue
        # 贪心合并：能装下就并进当前块（避免切太碎）
        if count_tokens((buffer + separators[depth] + p) if buffer else p) <= max_tokens:
            buffer = (buffer + separators[depth] + p) if buffer else p
        else:
            if buffer:
                out.append(buffer)
            buffer = p
    if buffer:
        out.append(buffer)
    return [o for o in out if o.strip()]


def _hard_split(text: str, max_tokens: int) -> List[str]:
    """无符号超长文本硬切（按 token 比例折算字符数）。"""
    total = count_tokens(text)
    n_parts = (total // max_tokens) + 1
    step = max(len(text) // n_parts, 1)
    return [text[i: i + step].strip() for i in range(0, len(text), step) if text[i: i + step].strip()]


# ------------------------------------------------------------
# 第 3 步：父子配对（相邻子块合并，1.5~2 倍量）
# ------------------------------------------------------------
def _pair_parents(
    children: List[ChildChunk],
    parent_target_tokens: int,
) -> List[ParentChunk]:
    """相邻子块滑动合并成父块，目标 token ≤ parent_target_tokens。

    合并只发生在同一个 section 内（heading_path 相同），
    跨主题拼块会稀释父块上下文的价值。
    """
    parents: List[ParentChunk] = []
    by_section: Dict[tuple, List[ChildChunk]] = {}
    for c in children:
        by_section.setdefault((c.section_idx, c.heading_path), []).append(c)

    for (sec_idx, heading), secs in by_section.items():
        pid = f"s{sec_idx:02d}p"
        buf: List[ChildChunk] = []
        buf_tokens = 0
        for c in secs:
            if buf and buf_tokens + c.tokens > parent_target_tokens:
                parents.append(_make_parent(pid, buf, heading))
                buf, buf_tokens = [], 0
            buf.append(c)
            buf_tokens += c.tokens
        if buf:
            parents.append(_make_parent(pid, buf, heading))
    return parents


def _make_parent(pid: str, buf: List[ChildChunk], heading: str) -> ParentChunk:
    text = "\n".join(c.text for c in buf)
    # 子块归属父块（元数据串行：child → parent_id）
    for c in buf:
        c.parent_id = pid
    return ParentChunk(
        id=pid,
        text=text,
        tokens=count_tokens(text),
        heading_path=heading,
        doc_source=buf[0].doc_source,
        child_ids=[c.id for c in buf],
    )


# ------------------------------------------------------------
# 入口
# ------------------------------------------------------------
def split_document(
    doc_file: str = "",
    *,
    child_tokens: Optional[int] = None,
    parent_ratio: Optional[float] = None,
    separators: Optional[tuple] = None,
) -> SplitResult:
    """完整四步切分。参数不传时用全局 rag_config（网格搜索时显式传覆盖值）。"""
    child_max = child_tokens if child_tokens is not None else rag_config.CHILD_TOKENS
    ratio = parent_ratio if parent_ratio is not None else rag_config.PARENT_RATIO
    seps = separators if separators is not None else rag_config.SPLIT_SEPARATORS

    name = doc_file or "elder_care.md"
    path = os.path.join(os.path.dirname(__file__), name)
    if not os.path.exists(path):
        logger.warning("切分器：知识库 %s 不存在，回退 elder_care.md", name)
        path = os.path.join(os.path.dirname(__file__), "elder_care.md")
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    doc_source = os.path.basename(path)
    sections = _split_sections(content)
    parent_target = int(child_max * ratio)

    # 第 1+2+4 步联动：结构切 section → 符号递归切子块 → 顺带挂元数据
    children: List[ChildChunk] = []
    for sec in sections:
        pieces = _recursive_split(sec["body"], child_max, seps)
        sec_heading = sec["heading_path"]
        # heading_path 末段就是条目名（如 "血压管理"），来源优先用正文标注
        source = sec["source"] or (sec_heading.split(" > ")[-1] if sec_heading else doc_source)
        for ci, piece in enumerate(pieces):
            children.append(ChildChunk(
                id=f"s{sec['idx']:02d}c{ci}",
                text=piece,
                tokens=count_tokens(piece),
                parent_id="",  # 第 3 步回填
                doc_source=source or doc_source,
                heading_path=sec_heading,
                section_idx=sec["idx"],
                child_idx=ci,
            ))

    parents = _pair_parents(children, parent_target)
    result = SplitResult(children=children, parents=parents)
    logger.info(
        "[split] %s → %d节 / %d子块 / %d父块（child≤%dtok, parent≤%dtok）",
        doc_source,
        len({c.section_idx for c in children}),
        len(children),
        len(parents),
        child_max,
        parent_target,
    )
    return result
