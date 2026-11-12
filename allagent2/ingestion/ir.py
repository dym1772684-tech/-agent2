"""统一中间表示(IR)——摄取层的唯一契约。

所有解析器只产 IR,下游(清洗/切分/入库)只消费 IR:
将来接 MinerU/OCR 引擎,只是多一个 adapter,下游零改动。

Block 是最小内容块(标题/正文/表格/图片占位);DocumentIR 是一篇文档的
完整解析结果,带溯源元数据与启发式置信度。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

MIN_TEXT_CHARS = 20   # 整篇低于这个字符数 → 视为空壳解析，覆盖率记 0


@dataclass
class Block:
    """最小内容块。type 决定下游处理方式(表格原子块/标题喂 heading_path)。"""
    type: str                  # heading | text | table | image
    text: str                  # 表格已转 Markdown;图片为空(占位)
    page: int = 0              # 溯源:页码/sheet号/slide号(1 起)
    text_level: int = 0        # 标题层级:1/2/3;非标题=0
    conf: float = 1.0          # 块级置信度(OCR/启发式给出)
    source: str = "native"     # native | geometric | ocr | mineru
    bbox: Optional[List[float]] = None   # 几何坐标(PDF 用,排序/双栏检测)


@dataclass
class DocumentIR:
    """一篇文档的完整解析结果。"""
    doc_name: str                          # 原始文件名(溯源展示)
    doc_type: str = "unknown"              # pdf | docx | xlsx | pptx | txt | csv | md | image
    blocks: List[Block] = field(default_factory=list)
    needs_ocr_pages: List[int] = field(default_factory=list)   # 扫描页(未装OCR引擎时入needs_review)
    sha256: str = ""                       # 源文件指纹(重复上传跳过解析)
    parser_version: int = 1                # 解析器版本(升级后可识别旧文档需重解析)

    # ---- 启发式置信度(不靠模型的自评,低于阈值整篇 needs_review) ----
    @property
    def confidence(self) -> float:
        if not self.blocks:
            return 0.0
        # 表格也是内容：xlsx 的正文基本都在 table 块里，只统计 text/heading 会把
        # 纯表格文档判成"空壳解析"
        text_chars = sum(len(b.text) for b in self.blocks
                         if b.type in ("text", "heading", "table"))
        n_pages = max(max((b.page for b in self.blocks), default=1),
                      len(self.needs_ocr_pages), 1)

        # coverage 用"结构完整性"：有内容的页 / 总页数。
        # 不用"字符密度"——那会把"文档短"误判成"解析差"：一份 90 字的出院小结
        # 和一份 900 字的报告，解析质量可以完全一样。
        pages_with_content = {b.page for b in self.blocks if b.text.strip()}
        coverage = len(pages_with_content) / n_pages if n_pages else 0.0
        # 整体字符量下限：块都解析出来了但没有字，说明是空壳
        if text_chars < MIN_TEXT_CHARS:
            coverage = 0.0

        bad = sum(b.conf < 0.6 for b in self.blocks) / len(self.blocks)
        scan_penalty = len(self.needs_ocr_pages) / (n_pages + len(self.needs_ocr_pages))
        return round(max(0.0, min(1.0, 0.6 * coverage + 0.3 * (1 - bad) - 0.4 * scan_penalty)), 3)

    @property
    def heading_summary(self) -> str:
        """标题路径摘要(解析报告展示用)。"""
        return " > ".join(b.text[:20] for b in self.blocks if b.type == "heading")[:200]

    def add_heading_paths(self) -> None:
        """单遍维护 heading_path:遍历时压栈/弹栈,写进每个块的隐性上下文。

        实现为把层级路径前缀写回 text 之外的推导函数——切分层调用
        chunker 时按 heading 块分段,天然获得路径;此处仅保证 heading 块
        自身的层级合法(不超过上一层级+1,防止孤立三级标题)。
        """
        prev = 0
        for b in self.blocks:
            if b.type == "heading":
                if b.text_level > prev + 1:
                    b.text_level = prev + 1   # 修复跳级:三级不能凭空出现
                prev = b.text_level
