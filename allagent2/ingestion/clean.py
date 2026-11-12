"""摄取清洗层:解析产物(IR)入检索库前的三道闸。

① PII 脱敏:手机号/身份证打码(复用 session.store.mask_pii 同款正则,
   检索库不能成为脱敏漏点);临床信息(诊断/指标)不脱——那是文档价值所在。
② 注入清洗:上传的文档是外部内容进 LLM 上下文的经典入口,
   剔除"忽略之前指令"类提示注入内容,替换为安全占位。
③ 乱码治理:替换字符/私有区占比超阈值的块降置信度(触发 needs_review)。
"""
from __future__ import annotations

import re
from typing import List

from .ir import Block, DocumentIR

# ① PII(与 session.store.mask_pii 保持同款口径)
_PHONE_RE = re.compile(r"1[3-9]\d{9}")
_IDCARD_RE = re.compile(r"\d{17}[\dXx]")

# ② 注入模式(与 fastap2 网关 scan_injection 同思路;文档场景放宽——
#   命中只剔除该句并降置信度,不整篇拒绝)
_INJECTION_RE = re.compile(
    r"[^\n。]*(?:忽略(?:之前|以上|前面)[^\n。]{0,15}(?:指令|要求|设定|提示)"
    r"|disregard[^\n。]{0,20}(?:previous|above|instructions)"
    r"|(?:泄露|打印|显示)[^\n。]{0,8}(?:系统提示|system\s*prompt)[^\n。]*)", re.IGNORECASE)

# ③ 乱码特征:U+FFFD 替换字符 / 私有区 / 控制区
_GARBAGE_RE = re.compile(r"[\ufffd\ue000-\uf8ff\x00-\x08\x0b\x0c\x0e-\x1f]")


def clean_document(ir: DocumentIR) -> DocumentIR:
    """三道闸依次过,原地修改并返回(块级处理,保留页码/类型/层级)。"""
    for b in ir.blocks:
        b.text = _mask_pii(b.text)
        b.text, hits = _strip_injection(b.text)
        garbage_ratio = _garbage_ratio(b.text)
        if hits or garbage_ratio > 0.05:
            b.conf = min(b.conf, 0.5)   # 降置信度 → 汇总到 needs_review
        # 连续空白压缩(保留表格 | 结构)
        if b.type != "table":
            b.text = re.sub(r"[ \t]{3,}", " ", b.text)
    ir.blocks = [b for b in ir.blocks if b.text.strip() or b.type == "image"]
    return ir


def _mask_pii(text: str) -> str:
    text = _IDCARD_RE.sub(lambda m: m.group(0)[:3] + "***********" + m.group(0)[-2:], text)
    # 与 agent/middleware.mask_pii 同口径：保留后 4 位，人能核对、又不泄露
    return _PHONE_RE.sub(lambda m: m.group(0)[:3] + "****" + m.group(0)[-4:], text)


def _strip_injection(text: str) -> tuple:
    """剔除注入句;返回 (清洗后文本, 命中数)。"""
    cleaned, n = _INJECTION_RE.subn(" ", text)
    return cleaned.strip(), n


def _garbage_ratio(text: str) -> float:
    if not text:
        return 0.0
    return len(_GARBAGE_RE.findall(text)) / len(text)
