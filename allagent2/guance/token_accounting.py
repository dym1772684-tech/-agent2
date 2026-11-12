"""Token 计量与成本可观测性。

============================================================
先回答一个关键问题：算 token 到底要不要"模型"？
------------------------------------------------------------
不要。算 token 用的是 **tokenizer**（分词器），不是大模型。它不推理、
不生成、不需要 GPU，只是把文本切成模型认识的子词单元并数个数。

DeepSeek 官方给过三条路（api-docs.deepseek.com/quick_start/token_usage）：
  1) 离线精确：官方 deepseek_tokenizer.zip（或 HF 上的 tokenizer 权重）
  2) 粗略估算：1 个中文字符 ≈ 0.6 token，1 个英文字符 ≈ 0.3 token
  3) 权威计费：每次请求响应里的 usage 字段 —— **官方明确说"以模型返回为准"**

本模块的选择：
  - 计费/观测 → 只用第 3 条（usage），这是账单，唯一可信来源；
  - 预算守卫 → 用第 2 条（零依赖、不下载任何东西），够用来拦截超长输入；
  - 第 1 条不内置：为了计数去下载一个 tokenizer 文件不划算，
    真需要时把 estimate_tokens() 换成 tokenizer 实现即可（接口不变）。

⚠️ 不要用 tiktoken 的 cl100k_base 去估 DeepSeek。网上有文章这么说，
   但 DeepSeek 的词表与 OpenAI 不同，官方自己都说"换算比例存在差异"。
============================================================

怎么接入的：
  1. middleWare 的 @after_model：每次模型调用后实时打一行 token 日志；
  2. bootstrap.run_agent：一轮对话结束后，把本轮新增 AI 消息的 usage
     汇总 → 算成本 → 写 JSONL → 返回给调用方。

改的意义：
  原来这个项目对 LLM 花了多少钱完全没感知——middleware 里虽然有
  ToolCallLimit / ModelCallLimit 做"次数"限制，但次数不等于成本：
  一次带 8000 token 上下文的调用和一次 50 token 的调用，成本差两个数量级。
  有了计量才能回答"哪一轮最贵""缓存命中率多少""摘要省了多少钱"。
"""
from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import dataclass, asdict, field
from typing import Any, Dict, Iterable, List, Optional

from .logging_helper import logger

# ============================================================
# 单价表（元 / 百万 token）
# ------------------------------------------------------------
# 怎么定的：DeepSeek 的定价会调整，写死必然过期。这里只给"量级正确"的
# 默认值，全部支持环境变量覆盖，并在日志里提醒以官网为准。
#
#   DEEPSEEK_PRICE_INPUT  输入（缓存未命中）
#   DEEPSEEK_PRICE_CACHE  输入（缓存命中，DeepSeek 便宜一个数量级）
#   DEEPSEEK_PRICE_OUTPUT 输出
#
# 缓存命中是 DeepSeek 的一个省钱大头：你的 middleware 每轮都注入同一段
# 养老人设前缀，这段前缀命中缓存后单价极低。不把 hit/miss 分开统计，
# 就永远看不出这层优化到底省了多少。
# ============================================================
def _price(env_key: str, default: float) -> float:
    try:
        return float(os.getenv(env_key, default))
    except (TypeError, ValueError):
        return default


PRICE_INPUT = _price("DEEPSEEK_PRICE_INPUT", 1.0)     # 元 / 百万 input（未命中缓存）
PRICE_CACHE = _price("DEEPSEEK_PRICE_CACHE", 0.1)     # 元 / 百万 input（命中缓存）
PRICE_OUTPUT = _price("DEEPSEEK_PRICE_OUTPUT", 2.0)   # 元 / 百万 output


# ============================================================
# 一、离线估算（零依赖，用于预算守卫）
# ============================================================
# 官方比例：1 中文字符 ≈ 0.6 token，1 英文字符 ≈ 0.3 token。
# 按字符逐个判断，中文（含中日韩统一表意文字）走 0.6，其余走 0.3。
def estimate_tokens(text: str) -> int:
    """粗略估算文本 token 数。只用于"要不要拦截"的判断，不用于计费。"""
    if not text:
        return 0
    total = 0.0
    for ch in text:
        if "\u4e00" <= ch <= "\u9fff" or "\u3400" <= ch <= "\u4dbf":
            total += 0.6
        elif ch.isspace():
            total += 0.1
        else:
            total += 0.3
    return int(total) + 1########逐个比对字符



# ChatCompletion 的 messages 有包装开销（role、分隔符），
# OpenAI 格式约每条 4 token。这里按官方常见口径补齐。
_MESSAGE_WRAPPER_TOKENS = 4


# ============================================================
# 一·补、切块侧计数器（与计费侧严格分离）
# ------------------------------------------------------------
# 同一个"token 计数"在两个位置要求完全相反：
#   切块：请求前就要知道，本地离线，相对准即可（块大点小点都能用）
#   计费：请求后才能拿到，必须权威（这是钱，只用上面 extract_usage）
# 所以这里单独提供 count_tokens()：给 RAG 切分器控制块大小用。
#
# 三档精度（TOKENIZER_MODE 环境变量切换，默认 char）：
#   char     中文 0.6/字 + 其他 0.3/字符（官方比例，零依赖）★默认
#   jieba    词数 × 1.2 修正（1 个中文词 ≈ 1.2 token，比词数更贴近真实）
#   official 官方 tokenizer 文件（设 DEEPSEEK_TOKENIZER_PATH 指向解压目录，
#            未配置时自动回退 char，不报错——评测发现块大小失控时才上这档）
# ============================================================
TOKENIZER_MODE = os.getenv("RAG_TOKENIZER_MODE", "char").lower()
_JIEBA_CORRECTION = 1.2  # jieba 词数 → token 的经验修正系数


def _get_official_tokenizer():
    """按需加载官方 tokenizer（只在 DEEPSEEK_TOKENIZER_PATH 配置了才尝试）。"""
    global _OFFICIAL_TOKENIZER
    if _OFFICIAL_TOKENIZER is _NOT_LOADED:
        path = os.getenv("DEEPSEEK_TOKENIZER_PATH", "")
        if path and os.path.isdir(path):
            try:
                from transformers import AutoTokenizer

                _OFFICIAL_TOKENIZER = AutoTokenizer.from_pretrained(
                    path, trust_remote_code=True
                )
            except Exception as exc:
                logger.warning("[token] 官方 tokenizer 加载失败，回退 char 档：%s", exc)
                _OFFICIAL_TOKENIZER = None
        else:
            _OFFICIAL_TOKENIZER = None
    return _OFFICIAL_TOKENIZER


_NOT_LOADED = object()
_OFFICIAL_TOKENIZER = _NOT_LOADED


def count_tokens(text: str, mode: str = "") -> int:
    """切块侧 token 计数（预算守卫同款，不用于计费）。

    切分器用它校验块大小（CHILD_TOKENS / PARENT_RATIO 都以此为尺度）。
    """
    if not text:
        return 0
    m = (mode or TOKENIZER_MODE).lower()
    if m == "jieba":
        try:
            import jieba

            return int(len(jieba.lcut(text)) * _JIEBA_CORRECTION) + 1
        except Exception:
            return estimate_tokens(text)
    if m == "official":
        tok = _get_official_tokenizer()
        if tok is not None:
            try:
                return len(tok.encode(text))
            except Exception:
                pass
    return estimate_tokens(text)


def estimate_messages_tokens(messages: Iterable[Any]) -> int:
    """估算一组消息进入 API 后的 token 数（含包装开销）。"""
    total = 0
    for m in messages or []:
        content = getattr(m, "content", None)
        if not isinstance(content, str):
            content = json.dumps(content, ensure_ascii=False) if content else ""
        total += estimate_tokens(content) + _MESSAGE_WRAPPER_TOKENS

        # 工具调用的定义与参数也会计入输入 token
        for call in getattr(m, "tool_calls", None) or []:
            total += estimate_tokens(json.dumps(call, ensure_ascii=False))
    return total


# ============================================================
# 二、usage 提取（权威数据来源）
# ============================================================
# LangChain 把各家厂商的 usage 统一成 usage_metadata：
#   input_tokens / output_tokens / total_tokens
#   input_token_details  = {cache_read, cache_creation, audio}
#   output_token_details = {reasoning, audio}
# DeepSeek 的 prompt_cache_hit_tokens 会被映射成 input_token_details.cache_read，
# reasoning_tokens 映射成 output_token_details.reasoning。
def extract_usage(msg: Any) -> Optional[Dict[str, int]]:
    """从一条 AI 消息里提取 token 用量；取不到返回 None。"""
    um = getattr(msg, "usage_metadata", None)
    if not um:
        # 兜底：部分兼容层只把原始 usage 放在 response_metadata
        resp = getattr(msg, "response_metadata", None) or {}
        raw = resp.get("token_usage") or resp.get("usage") or {}
        if not raw:
            return None
        input_tokens = int(raw.get("prompt_tokens", 0) or 0)
        output_tokens = int(raw.get("completion_tokens", 0) or 0)
        cache_read = int(raw.get("prompt_cache_hit_tokens", 0) or 0)
        reasoning = int((raw.get("completion_tokens_details") or {}).get("reasoning_tokens", 0) or 0)
    else:
        ind = um.get("input_token_details") or {}
        outd = um.get("output_token_details") or {}
        input_tokens = int(um.get("input_tokens", 0) or 0)
        output_tokens = int(um.get("output_tokens", 0) or 0)
        cache_read = int(ind.get("cache_read", 0) or 0)
        reasoning = int(outd.get("reasoning", 0) or 0)

    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cache_read": cache_read,
        # 缓存未命中 = 输入总量 - 命中量（DeepSeek: prompt_tokens = hit + miss）
        "cache_miss": max(input_tokens - cache_read, 0),
        "reasoning_tokens": reasoning,
        "total_tokens": input_tokens + output_tokens,
    }


def cost_of(usage: Dict[str, int]) -> float:
    """按单价表把用量换算成金额（元）。"""
    return (
        usage.get("cache_miss", 0) / 1_000_000 * PRICE_INPUT
        + usage.get("cache_read", 0) / 1_000_000 * PRICE_CACHE
        + usage.get("output_tokens", 0) / 1_000_000 * PRICE_OUTPUT
    )


# ============================================================
# 三、累加器
# ============================================================
@dataclass
class TokenAccountant:
    """线程安全的 token 累加器。

    calls        模型调用次数
    input/output 输入输出总量
    cache_read   命中缓存的输入量（省钱部分）
    cache_miss   未命中的输入量（全价部分）
    reasoning    思维链消耗（deepseek-reasoner 才有）
    cost         累计金额（元）
    """

    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read: int = 0
    cache_miss: int = 0
    reasoning_tokens: int = 0
    total_tokens: int = 0
    cost: float = 0.0
    _lock: Any = field(default_factory=threading.Lock, repr=False, compare=False)

    def add(self, usage: Dict[str, int]) -> float:
        """累加一条用量，返回本次产生的金额。"""
        if not usage:
            return 0.0
        c = cost_of(usage)
        with self._lock:
            self.calls += 1
            self.input_tokens += usage.get("input_tokens", 0)
            self.output_tokens += usage.get("output_tokens", 0)
            self.cache_read += usage.get("cache_read", 0)
            self.cache_miss += usage.get("cache_miss", 0)
            self.reasoning_tokens += usage.get("reasoning_tokens", 0)
            self.total_tokens += usage.get("total_tokens", 0)
            self.cost += c
        return c

    def merge(self, other: "TokenAccountant") -> None:
        with self._lock:
            self.calls += other.calls
            self.input_tokens += other.input_tokens
            self.output_tokens += other.output_tokens
            self.cache_read += other.cache_read
            self.cache_miss += other.cache_miss
            self.reasoning_tokens += other.reasoning_tokens
            self.total_tokens += other.total_tokens
            self.cost += other.cost

    @property
    def cache_hit_rate(self) -> float:
        total_in = self.cache_read + self.cache_miss
        return (self.cache_read / total_in) if total_in else 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "calls": self.calls,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cache_read": self.cache_read,
            "cache_miss": self.cache_miss,
            "cache_hit_rate": round(self.cache_hit_rate, 4),
            "reasoning_tokens": self.reasoning_tokens,
            "total_tokens": self.total_tokens,
            "cost_yuan": round(self.cost, 6),
        }


# ============================================================
# 四、去重 + 全局单例
# ============================================================
# 为什么要去重：多轮会话里，invoke 返回的 messages 包含**历史全部消息**。
# 如果不去重，第 5 轮会把第 1~4 轮的用量再算一遍，账单直接翻倍。
# 这里按 AIMessage.id（LangChain 生成的唯一 uuid）记录已计费的消息。
_COUNTED_IDS: set = set()
_IDS_LOCK = threading.Lock()
_IDS_MAX = 5000  # 防止无限增长；超出后清空（最坏情况是老消息被重复计一次）

_GLOBAL = TokenAccountant()


def get_global_accountant() -> TokenAccountant:
    return _GLOBAL


def reset_global_accountant() -> None:
    with _GLOBAL._lock:
        _GLOBAL.calls = 0
        _GLOBAL.input_tokens = 0
        _GLOBAL.output_tokens = 0
        _GLOBAL.cache_read = 0
        _GLOBAL.cache_miss = 0
        _GLOBAL.reasoning_tokens = 0
        _GLOBAL.total_tokens = 0
        _GLOBAL.cost = 0.0
    with _IDS_LOCK:
        _COUNTED_IDS.clear()


def collect_usage(
    messages: Iterable[Any],
    accountant: Optional[TokenAccountant] = None,
) -> TokenAccountant:
    """遍历消息，累加其中尚未计费过的 AI 消息用量。

    Args:
        messages:   invoke 返回的全部消息（含历史）
        accountant: 传入则累加到它上面；不传则新建一个（用于单轮统计）
    """
    acc = accountant if accountant is not None else TokenAccountant()
    for m in messages or []:
        if getattr(m, "type", "") != "ai":
            continue
        mid = getattr(m, "id", None)
        if mid:
            with _IDS_LOCK:
                if mid in _COUNTED_IDS:
                    continue
                if len(_COUNTED_IDS) >= _IDS_MAX:
                    _COUNTED_IDS.clear()
                _COUNTED_IDS.add(mid)
        usage = extract_usage(m)
        if usage:
            acc.add(usage)
    return acc


# ============================================================
# 五、落盘（JSONL，便于后续统计）
# ============================================================
_USAGE_LOG = os.path.join(os.path.dirname(__file__), "logs", "token_usage.jsonl")


def append_usage_log(accountant: TokenAccountant, *, thread_id: str = "", tag: str = "") -> None:
    """把一轮对话的用量追加写入 JSONL，一行一次对话。

    为什么用 JSONL 而不是只打日志：日志是用来"看"的，JSONL 是用来"算"的。
    攒够数据后一条 awk / pandas 就能算出"哪个 thread 最贵""日均成本"。
    """
    record = {
        "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
        "thread_id": thread_id,
        "tag": tag,
        **accountant.to_dict(),
    }
    try:
        os.makedirs(os.path.dirname(_USAGE_LOG), exist_ok=True)
        with open(_USAGE_LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception as exc:  # 落盘失败绝不能影响主链路
        logger.warning("[token] 用量落盘失败：%s", exc)


def read_usage_log(limit: int = 100) -> List[Dict[str, Any]]:
    """读取最近 limit 条用量记录（供管理端点展示）。"""
    if not os.path.exists(_USAGE_LOG):
        return []
    try:
        with open(_USAGE_LOG, "r", encoding="utf-8") as f:
            lines = f.readlines()
    except Exception:
        return []
    out = []
    for line in lines[-limit:]:
        try:
            out.append(json.loads(line))
        except Exception:
            continue
    return out


# ============================================================
# 六、可选的实时回调（流式 / 非 LangGraph 场景用）
# ============================================================
# 本项目的主计量路径是 bootstrap 里的 collect_usage()，因为 LangGraph 的
# invoke 返回后就拿到了全部消息，最简单也最准。
# 下面这个回调给需要"调用过程中就要看到用量"的场景（比如 SSE 流式）。
try:
    from langchain_core.callbacks import BaseCallbackHandler
except Exception:  # 环境异常时不影响本模块其它能力
    BaseCallbackHandler = None  # type: ignore


if BaseCallbackHandler is not None:

    class TokenAccountingCallback(BaseCallbackHandler):
        """每次 LLM 调用结束时累加用量（on_llm_end）。"""

        def __init__(self, accountant: Optional[TokenAccountant] = None):
            self.accountant = accountant or get_global_accountant()

        def on_llm_end(self, response, **kwargs) -> None:
            try:
                llm_output = getattr(response, "llm_output", None) or {}
                raw = llm_output.get("token_usage") or llm_output.get("usage") or {}
                if raw:
                    input_tokens = int(raw.get("prompt_tokens", 0) or 0)
                    output_tokens = int(raw.get("completion_tokens", 0) or 0)
                    cache_read = int(raw.get("prompt_cache_hit_tokens", 0) or 0)
                    self.accountant.add(
                        {
                            "input_tokens": input_tokens,
                            "output_tokens": output_tokens,
                            "cache_read": cache_read,
                            "cache_miss": max(input_tokens - cache_read, 0),
                            "reasoning_tokens": int(
                                (raw.get("completion_tokens_details") or {}).get(
                                    "reasoning_tokens", 0
                                )
                                or 0
                            ),
                            "total_tokens": input_tokens + output_tokens,
                        }
                    )
                    return
                # 没有 llm_output 时退到 generation 上的 usage_metadata
                for gens in getattr(response, "generations", []) or []:
                    for g in gens:
                        um = getattr(g, "message", None) and getattr(
                            g.message, "usage_metadata", None
                        )
                        if um:
                            self.accountant.add(
                                {
                                    "input_tokens": um.get("input_tokens", 0),
                                    "output_tokens": um.get("output_tokens", 0),
                                    "cache_read": (um.get("input_token_details") or {}).get(
                                        "cache_read", 0
                                    ),
                                    "cache_miss": max(
                                        um.get("input_tokens", 0)
                                        - (um.get("input_token_details") or {}).get(
                                            "cache_read", 0
                                        ),
                                        0,
                                    ),
                                    "reasoning_tokens": (
                                        um.get("output_token_details") or {}
                                    ).get("reasoning", 0),
                                    "total_tokens": um.get("total_tokens", 0),
                                }
                            )
            except Exception as exc:
                logger.warning("[token] 回调计量失败：%s", exc)

else:  # pragma: no cover
    TokenAccountingCallback = None  # type: ignore
