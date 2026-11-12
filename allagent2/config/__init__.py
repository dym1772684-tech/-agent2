"""配置体系（模仿 pi-main 的分层配置 + schema 校验）。

用法：
    from allagent2.config import get_settings, reload, describe

    cfg = get_settings()
    if not cfg.llm_ready:                 # 缺 key → 业务侧降级，不抛异常
        ...
    cfg.settings.agent.max_rounds         # 已校验、已转类型的值
    cfg.sources["llm.base_url"]           # 这个值从哪来的：default/env_file/environ/runtime
    describe()                            # 安全展示（密钥打码）

三层结构：
    resolve.py  模板解析   $VAR / ${VAR} / $$ 转义（不做命令替换）
    schema.py   pydantic  字段表 + 校验模型（SecretStr 保护密钥）
    loader.py   分层加载  default → env_file → environ → runtime，逐字段回退
"""
from allagent2.config.loader import (
    ConfigError,
    LoadedSettings,
    describe,
    get_settings,
    load,
    reload,
    set_settings,
)
from allagent2.config.schema import Settings

__all__ = [
    "ConfigError",
    "LoadedSettings",
    "Settings",
    "describe",
    "get_settings",
    "load",
    "reload",
    "set_settings",
]
