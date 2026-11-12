"""配置 schema（pydantic 校验 + 声明式字段表）。

============================================================
两层设计（对应 pi 的 typebox schema + modelOverrides）
------------------------------------------------------------
① 字段表 `FIELDS`：声明"哪个配置项、从哪个环境变量来、默认值、类型、
   是否敏感、是否必需"。loader 按表逐项解析，因此**每一项都能追踪来源**。
② pydantic 模型：作为最终校验层（取值范围、类型、枚举），
   对应 pi 用 typebox 编译 schema 校验 models.json 的做法。

为什么要"先解析成 dict、再 validate"，而不是直接
`Settings(**os.environ)`：后者一旦某项非法就整体失败，
服务起不来；而本项目的现实是缺 key 也要能降级跑（HyDE/JUDGE 都没有 key）。
所以 loader 的语义是**逐字段回退**：谁非法谁回默认，其余照常生效。

============================================================
敏感项
------------------------------------------------------------
密钥一律用 pydantic 的 SecretStr：`repr()` / `str()` / 日志里永远是
`**********`，只有显式 `.get_secret_value()` 才拿得到真值。
这样 `describe()` 直接把 Settings 打出来也不会泄露。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, SecretStr, model_validator


# ============================================================
# ① 字段表
# ============================================================
@dataclass(frozen=True)
class EnvField:
    """一条配置项与它对应的环境变量。"""

    path: str                 # 在 Settings 里的路径，如 "llm.api_key"
    env: str                  # 环境变量名
    default: Any
    cast: type = str
    sensitive: bool = False
    required: bool = False
    doc: str = ""


FIELDS: tuple[EnvField, ...] = (
    # ---- 主模型 ----
    EnvField("llm.api_key", "OPENAI_API_KEY", "", sensitive=True,
             doc="主对话模型密钥；空=无法调用模型"),
    EnvField("llm.base_url", "OPENAI_BASE_URL", "",
             doc="OpenAI 兼容端点；指向本地 vLLM 时形如 http://127.0.0.1:8005/v1"),
    EnvField("llm.model", "OPENAI_MODEL", "",
             doc="模型 ID"),
    EnvField("llm.temperature", "AGENT_TEMPERATURE", 0.3, float),
    EnvField("llm.max_tokens", "AGENT_MAX_TOKENS", 2048, int),
    EnvField("llm.timeout", "AGENT_LLM_TIMEOUT", 60.0, float),
    # ---- 天气 ----
    EnvField("weather.api_key", "WEATHER_API_KEY", "", sensitive=True),
    # ---- HyDE（缺失即自动降级为双路检索）----
    EnvField("hyde.api_key", "HYDE_API_KEY", "", sensitive=True,
             doc="HyDE 用的独立模型；空=跳过 HyDE，检索回退双路融合"),
    EnvField("hyde.base_url", "HYDE_BASE_URL", ""),
    EnvField("hyde.model", "HYDE_MODEL", "deepseek-chat"),
    EnvField("hyde.timeout", "HYDE_TIMEOUT", 5.0, float),
    # ---- Judge（缺失即跳过 LLM 评分，只跑规则断言）----
    EnvField("judge.api_key", "JUDGE_API_KEY", "", sensitive=True,
             doc="LLM-as-Judge 用的模型；空=评测只跑规则断言"),
    EnvField("judge.base_url", "JUDGE_BASE_URL", ""),
    EnvField("judge.model", "JUDGE_MODEL", "hunyuan-turbo"),
    # ---- 智能体 ----
    EnvField("agent.engine", "AGENT_ENGINE", "react",
             doc="react=手写思考循环（默认）| create_agent=官方封装（回退用）"),
    EnvField("agent.topology", "AGENT_TOPOLOGY", "single",
             doc="single=单 agent | team=黑板多智能体"),
    EnvField("agent.max_rounds", "AGENT_MAX_ROUNDS", 6, int),
    EnvField("agent.compress_trigger", "AGENT_COMPRESS_TRIGGER", 30, int,
             doc="消息条数超过该值时压缩历史"),
    EnvField("agent.compress_keep", "AGENT_COMPRESS_KEEP", 16, int),
    EnvField("agent.tool_result_limit", "AGENT_TOOL_RESULT_LIMIT", 4000, int),
    EnvField("agent.tool_retries", "AGENT_TOOL_RETRIES", 2, int),
    EnvField("agent.interrupt_ttl", "AGENT_INTERRUPT_TTL", 600, int,
             doc="中断标记在 Redis 的存活秒数"),
    # ---- RAG ----
    EnvField("rag.enabled", "RAG_ENABLED", True, bool),
    EnvField("rag.top_k", "RAG_TOP_K", 4, int),
    EnvField("rag.hyde_enabled", "HYDE_ENABLED", True, bool),
    EnvField("rag.hyde_top_k", "HYDE_TOP_K", 20, int),
    EnvField("rag.weight_hyde", "W_HYDE", 0.3, float),
    EnvField("rag.rerank_top_n", "RERANK_TOP_N", 20, int),
    # ---- 记忆 ----
    EnvField("memory.enabled", "MEMORY_ENABLED", True, bool),
    EnvField("memory.recall_top_k", "MEMORY_RECALL_TOP_K", 8, int),
    EnvField("memory.reflect_threshold", "MEMORY_REFLECT_THRESHOLD", 3, int,
             doc="累计几轮对话后触发一次异步反思沉淀"),
    # ---- 可观测 ----
    EnvField("obs.metrics_enabled", "METRICS_ENABLED", True, bool),
    EnvField("obs.metrics_token", "METRICS_TOKEN", "", sensitive=True,
             doc="为空=裸挂 /metrics（仅建议 127.0.0.1 绑定）；非空=要求 Bearer 校验"),
    EnvField("obs.trace_max_events", "TRACE_MAX_EVENTS", 400, int,
             doc="单会话 trace 事件上限，超出后只保留最近 N 条"),
    EnvField("obs.request_id_header", "REQUEST_ID_HEADER", "X-Request-Id"),
)


# ============================================================
# ② pydantic 校验模型
# ============================================================
class LLMConfig(BaseModel):
    api_key: SecretStr = SecretStr("")
    base_url: str = ""
    model: str = ""
    temperature: float = Field(0.3, ge=0.0, le=2.0)
    max_tokens: int = Field(2048, gt=0)
    timeout: float = Field(60.0, gt=0)


class WeatherConfig(BaseModel):
    api_key: SecretStr = SecretStr("")


class HydeConfig(BaseModel):
    api_key: SecretStr = SecretStr("")
    base_url: str = ""
    model: str = "deepseek-chat"
    timeout: float = Field(5.0, gt=0, le=120)

    @property
    def usable(self) -> bool:
        """没有 key 就是不可用 —— 调用方据此决定是否降级，而不是抛异常。"""
        return bool(self.api_key.get_secret_value())


class JudgeConfig(BaseModel):
    api_key: SecretStr = SecretStr("")
    base_url: str = ""
    model: str = "hunyuan-turbo"

    @property
    def usable(self) -> bool:
        return bool(self.api_key.get_secret_value())


class AgentConfig(BaseModel):
    engine: Literal["react", "create_agent"] = "react"
    topology: Literal["single", "team"] = "single"
    max_rounds: int = Field(6, ge=1, le=30)
    compress_trigger: int = Field(30, ge=4)
    compress_keep: int = Field(16, ge=2)
    tool_result_limit: int = Field(4000, ge=200)
    tool_retries: int = Field(2, ge=0, le=5)
    interrupt_ttl: int = Field(600, ge=30)


class RagConfig(BaseModel):
    """RAG 视图（**只读镜像**，真源是 allagent2/rag/config.py 的 rag_config）。

    【修复"改这里不生效"】以前这几个字段是独立的 pydantic 值：改
    `get_settings().settings.rag.top_k`、或者给 loader 传 runtime 覆盖，
    都不会影响 rag.py —— 它只读 rag_config。两套值看起来一样但各走各的。
    现在改成 after-validator：构造完立刻用 rag_config 的实际值覆盖，
    所以这个对象永远反映"线上真正在用的参数"，快照不会再骗人。
    """
    enabled: bool = True
    top_k: int = Field(4, ge=1, le=50)
    hyde_enabled: bool = True
    hyde_top_k: int = Field(20, ge=1, le=100)
    weight_hyde: float = Field(0.3, ge=0.0, le=1.0)
    rerank_top_n: int = Field(20, ge=0, le=200)

    @model_validator(mode="after")
    def _sync_from_rag_config(self):
        """以 rag_config 为准回填（导入失败时保留 env 值，不影响启动）。"""
        try:
            from allagent2.rag.config import rag_config as _rc

            self.top_k = int(_rc.TOP_K)
            self.hyde_enabled = bool(_rc.HYDE_ENABLED)
            self.hyde_top_k = int(_rc.HYDE_TOP_K)
            self.weight_hyde = float(_rc.W_HYDE)
            self.rerank_top_n = int(_rc.RERANK_TOP_N)
        except Exception:
            pass
        return self


class MemoryConfig(BaseModel):
    enabled: bool = True
    recall_top_k: int = Field(8, ge=1, le=50)
    reflect_threshold: int = Field(3, ge=1)


class ObservabilityConfig(BaseModel):
    metrics_enabled: bool = True
    metrics_token: SecretStr = SecretStr("")
    trace_max_events: int = Field(400, ge=20)
    request_id_header: str = "X-Request-Id"


class Settings(BaseModel):
    """全部配置的不可变快照（对应 pi 的 "Immutable, credential-blind snapshot"）。"""

    llm: LLMConfig = Field(default_factory=LLMConfig)
    weather: WeatherConfig = Field(default_factory=WeatherConfig)
    hyde: HydeConfig = Field(default_factory=HydeConfig)
    judge: JudgeConfig = Field(default_factory=JudgeConfig)
    agent: AgentConfig = Field(default_factory=AgentConfig)
    rag: RagConfig = Field(default_factory=RagConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    obs: ObservabilityConfig = Field(default_factory=ObservabilityConfig)


def default_value(path: str) -> Any:
    """按路径取字段表里的默认值（loader 逐字段回退时用）。"""
    for f in FIELDS:
        if f.path == path:
            return f.default
    return None


def field_by_path(path: str) -> Optional[EnvField]:
    for f in FIELDS:
        if f.path == path:
            return f
    return None
