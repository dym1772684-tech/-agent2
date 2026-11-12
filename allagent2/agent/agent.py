from __future__ import annotations

import os
from pathlib import Path

from dotenv import find_dotenv, load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import BaseMessage

from allagent2.agent.state import ParsedOutput
from allagent2.guance.logging_helper import logger
from allagent2.tools.tool import tools

BASE_DIR = Path(__file__).resolve().parent.parent
ENV_PATH = BASE_DIR / "core" / "key.env"
load_dotenv(find_dotenv(str(ENV_PATH)))


class _FallbackModel:
    """当 LLM / SDK / API 不可用时，返回安全空结构，保证服务能启动和接口返回。"""

    def __init__(self, *, fallback_content: str = ""):
        self.fallback_content = fallback_content

    def invoke(self, messages, *args, **kwargs):
        last = messages[-1].content if messages else self.fallback_content
        return {"content": last, "parsed": {}}

    def bind_tools(self, tools_list):
        return self

    def with_structured_output(self, schema, include_raw=True):
        return self



def _llm_params() -> tuple[float, int, float]:
    """从 allagent2.config 读模型参数；配置不可用时回落到安全默认值。

    之前这里是硬编码 temperature=1 / max_tokens=1024。1024 会把 team 拓扑
    聚合阶段的结构化 JSON 截断（answer 为空 → 每轮都走降级文案），
    而配置系统里声明的是 2048 —— 声明了却没人读，等于没有配置。
    """
    try:
        from allagent2.config import get_settings

        cfg = get_settings().settings.llm
        return float(cfg.temperature), int(cfg.max_tokens), float(cfg.timeout)
    except Exception as exc:
        logger.warning("读取模型配置失败，使用默认参数：%s", exc)
        return 1.0, 2048, 30.0



def create_default_chat_model():
    """按环境变量创建"默认 chat 模型"，给记忆反思/画像裁决这类内部调用用。

    【修复】`session/profile.py` 的 `_adjudicate()` 一直在 import 这个名字，
    但本文件从未实现过 → 每次画像裁决都抛 ImportError，被 except 吞掉后
    保守返回 "coexist"（结果：画像只增不覆盖，永远学不会"老人改口了"）。
    这类"导入了不存在的名字"不会在启动时报，只在跑到那一步时静默降级。

    api_key 缺失时 create_chat_model 仍会构造客户端（首次调用才失败），
    所以这里额外做一次显式检查，把问题提前到调用点、日志里可见。
    """
    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY 未配置，无法创建默认 chat model")
    return create_chat_model(api_key, base_url)


def create_parser_model(api_key: str, base_url: str):
    """构造结构化解析模型。

    【改造】模型名从环境变量 OPENAI_MODEL 读取（默认 deepseek-chat），并显式声明
    model_provider="openai"：本地 vLLM / 任何 OpenAI 兼容端点都走同一套协议。
    DeepSeek 专属的 thinking 关闭参数只在模型名含 deepseek 时才发送，
    避免把私有字段塞给 vLLM 导致请求被拒。
    """
    model_name = os.getenv("OPENAI_MODEL", "deepseek-chat")
    logger.info("创建 parser model，model=%s, api_key present=%s, base_url=%s",
                model_name, api_key is not None, base_url)
    try:
        kwargs = dict(
            model=model_name,
            model_provider="openai",
            api_key=api_key,
            base_url=base_url,
            temperature=1,
            max_tokens=4096,
            timeout=30,
            max_retries=3,
        )
        if "deepseek" in model_name.lower():
            kwargs["extra_body"] = {"thinking": {"type": "disabled"}}
        model = init_chat_model(**kwargs)
        model = model.with_structured_output(ParsedOutput, include_raw=True)
        logger.info("parser model 创建完成")
        return model
    except Exception as exc:
        logger.warning("parser model 创建失败，使用兜底模型：%s", exc)
        return _FallbackModel(fallback_content="parser fallback")


def create_chat_model(api_key: str, base_url: str):
    """【改造】为 `create_agent` 提供「未绑定工具」的原始模型实例。

    怎么改的：
      原来只有 create_tool_model（bind_tools 后的模型），而 LangChain 的
      create_agent 会自己绑定工具并管理工具循环。把已绑定工具的模型再交进去
      会导致工具被绑定两次、且中间件拿不到干净的模型实例。这里新增一个
      「裸模型」构造函数供 create_agent 使用。

    改的意义：
      工具绑定权交回框架，模型的 tool_calls 才由 create_agent 统一调度，
      @wrap_tool_call 这类中间件才能真正拦得住每一次工具调用。

    【改造】模型名改从环境变量 OPENAI_MODEL 读取（默认 deepseek-chat），
      并显式声明 model_provider="openai"，使本地 vLLM 端点同样可用；
      DeepSeek 专属的 thinking 关闭参数仅在模型名含 deepseek 时发送。
    """
    model_name = os.getenv("OPENAI_MODEL", "deepseek-chat")
    logger.info("创建 chat model，model=%s, api_key present=%s, base_url=%s",
                model_name, api_key is not None, base_url)
    _temperature, _max_tokens, _timeout = _llm_params()
    kwargs = dict(
        model=model_name,
        model_provider="openai",
        api_key=api_key,
        base_url=base_url,
        temperature=_temperature,
        max_tokens=_max_tokens,
        timeout=_timeout,
        max_retries=3,
    )
    if "deepseek" in model_name.lower():
        kwargs["extra_body"] = {"thinking": {"type": "disabled"}}
    return init_chat_model(**kwargs)


def create_tool_model(api_key: str, base_url: str):
    """（遗留入口，当前链路未使用）同样改为读 OPENAI_MODEL + OpenAI 协议，保持一致性。"""
    model_name = os.getenv("OPENAI_MODEL", "deepseek-chat")
    logger.info("创建 tool model，model=%s, api_key present=%s, base_url=%s",
                model_name, api_key is not None, base_url)
    try:
        kwargs = dict(
            model=model_name,
            model_provider="openai",
            api_key=api_key,
            base_url=base_url,
            temperature=1,
            max_tokens=4096,
            timeout=30,
            max_retries=3,
        )
        if "deepseek" in model_name.lower():
            kwargs["extra_body"] = {"thinking": {"type": "disabled"}}
        model = init_chat_model(**kwargs)
        model = model.bind_tools(tools)
        logger.info("tool model 创建完成")
        return model
    except Exception as exc:
        logger.warning("tool model 创建失败，使用兜底模型：%s", exc)
        return _FallbackModel(fallback_content="tool fallback")


def agent_invoke(model, state):
    logger.info("agent_invoke 开始，消息数量=%d", len(state.get("messages", [])))
    try:
        result = model.invoke(state["messages"])
        logger.info(
            "agent_invoke 完成，type=%s, content=%s, tool_calls=%s",
            type(result),
            getattr(result, "content", None),
            getattr(result, "tool_calls", None),
        )
        return result
    except Exception:
        logger.exception("agent_invoke 失败")
        raise
