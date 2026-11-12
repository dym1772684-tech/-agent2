"""HyDE：假设回答生成（Hypothetical Document Embeddings）。

============================================================
是什么
------------------------------------------------------------
用户问题进来后，先用 API 大模型对问题生成"最可能的回答"（假设文档），
再拿这段回答的 embedding 去召回——假设回答是自然语言陈述句，与知识库
文档在同一语义空间，补足了"问题 → 文档"跨空间匹配的语义损失。
这就是检索链路的第三路（词法 BM25 / 向量 / HyDE）。

为什么独立 HYDE_* 环境变量
------------------------------------------------------------
主链路 OPENAI_* 指向本地 vLLM 3B（key.env → 127.0.0.1:8005），
小模型生成的中文假设回答质量不足；HyDE 单独走云端 API 大模型
（默认 DeepSeek，OpenAI 兼容协议）。HYDE_API_KEY 未配置时
整路自动禁用（fail-open，检索自动回退原有双路融合）。
============================================================
"""
from __future__ import annotations

import os
import threading
from typing import Any, Optional

from allagent2.guance.logging_helper import logger

_env_loaded = False
_env_lock = threading.Lock()

# 客户端缓存：按 (key, base_url) —— env 变了自动重建
_client_cache: dict = {}
_client_lock = threading.Lock()
_warned_no_key = False

_HYDE_PROMPT = (
    "你是一位养老健康知识助手。针对下面的问题，直接写出一段最可能的答案"
    "（80~150字，陈述句，包含具体做法或结论）。即使不确定也要写出最合理的"
    "推测；不要拒绝回答，不要解释，不要反问。\n\n问题：{query}"
)


def _ensure_env() -> None:
    """幂等加载 key.env（bootstrap 已加载则 no-op；单独跑评测脚本时兜底）。"""
    global _env_loaded
    if _env_loaded:
        return
    with _env_lock:
        if _env_loaded:
            return
        try:
            from pathlib import Path

            from dotenv import load_dotenv

            load_dotenv(
                Path(__file__).resolve().parents[1] / "core" / "key.env",
                override=False,
            )
        except Exception:
            pass
        _env_loaded = True


def _get_client() -> Optional[Any]:
    """按 (HYDE_API_KEY, HYDE_BASE_URL) 缓存 OpenAI 兼容客户端；无 key 返回 None。"""
    key = (os.getenv("HYDE_API_KEY") or "").strip()
    if not key:
        return None
    base = (os.getenv("HYDE_BASE_URL") or "https://api.deepseek.com/v1").strip()
    with _client_lock:
        if (key, base) not in _client_cache:
            from openai import OpenAI  # 延迟导入

            _client_cache[(key, base)] = OpenAI(api_key=key, base_url=base)
        return _client_cache[(key, base)]


def generate_hypothesis(query: str, cfg=None) -> str:
    """生成假设回答；未配置/失败/超时一律返回空串（调用方据此降级，不炸链路）。"""
    from .config import rag_config

    cfg = cfg or rag_config
    q = (query or "").strip()
    if not cfg.HYDE_ENABLED or not q:
        return ""
    _ensure_env()
    client = _get_client()
    global _warned_no_key
    if client is None:
        if not _warned_no_key:
            _warned_no_key = True
            logger.warning("[hyde] HYDE_API_KEY 未配置，HyDE 路禁用（回退双路融合）")
        return ""
    model = (os.getenv("HYDE_MODEL") or "deepseek-chat").strip()
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": _HYDE_PROMPT.format(query=q)}],
            temperature=0.5,
            max_tokens=cfg.HYDE_MAX_TOKENS,
            timeout=cfg.HYDE_TIMEOUT,
        )
        return (resp.choices[0].message.content or "").strip()
    except Exception as exc:
        logger.warning("[hyde] 假设回答生成失败（本轮跳过）：%s", exc)
        return ""
