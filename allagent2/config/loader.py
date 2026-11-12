"""配置加载：分层覆盖 + 逐字段回退 + 来源追踪。

============================================================
优先级（后面的覆盖前面的）
------------------------------------------------------------
    default        字段表里的默认值
    env_file       key.env / mysql.env 等文件（按传入顺序，后者覆盖前者）
    environ        进程环境变量（容器/CI 通常走这里）
    runtime        调用方显式传入的覆盖（测试、灰度、按用户切模型）

每一层命中的值都会记进 `sources[path]`，所以线上排查时能直接回答
"这个 base_url 到底是从哪读来的"——这是散落 os.getenv 时永远答不上来的问题。

============================================================
"各种状况"的处理策略
------------------------------------------------------------
  缺失必需项        不抛异常；标记 source="missing"，需要时调 require()
  缺失可选项        用默认值；业务侧用 xxx.usable 判断是否降级
  类型非法          **只回退这一项**到默认 + warning（不整体失败）
  引用变量不存在    整值视为未配置（不做半截替换）+ warning
  敏感项            一律 SecretStr；describe() 打码，永不回传真值
  配置改过了        reload() 递增 version，版本号进 trace 便于归因
  取值非法（越界）  pydantic 校验失败 → 定位到字段 → 该字段回默认 → 再校验

这与 pi-main 的分层思路一致（builtin → provider config → modelOverrides
只允许覆盖部分字段），但把"服务端不能起不来"作为第一优先级：
任何单项错误都不允许拖垮整个配置快照。
"""
from __future__ import annotations

import hashlib
import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from pydantic import ValidationError

from allagent2.config.resolve import env_names, resolve
from allagent2.config.schema import FIELDS, Settings, default_value, field_by_path


class ConfigError(RuntimeError):
    """必需配置缺失/非法时抛出（只在显式 require() 时出现）。"""


_TRUE = {"1", "true", "yes", "on", "y", "t"}
_FALSE = {"0", "false", "no", "off", "n", "f"}


def _coerce(raw: Any, cast: type, path: str, warnings: List[str]) -> Any:
    """类型转换；失败返回 None 让调用方回退默认值。"""
    if raw is None:
        return None
    try:
        if cast is bool:
            if isinstance(raw, bool):
                return raw
            text = str(raw).strip().lower()
            if text in _TRUE:
                return True
            if text in _FALSE:
                return False
            raise ValueError("不是布尔值")
        if cast is int:
            return int(float(str(raw).strip()))
        if cast is float:
            return float(str(raw).strip())
        return str(raw)
    except Exception:
        warnings.append(f"{path} 的值无法转成 {cast.__name__}，已回退默认值")
        return None


def read_env_file(path: str | os.PathLike[str]) -> Dict[str, str]:
    """读 .env 文件（支持 # 注释、export 前缀、引号包裹、行内注释）。

    不用 python-dotenv 是为了拿到"这个键确实出现在这个文件里"的确定性，
    排障时来源追踪比便利性重要。
    """
    out: Dict[str, str] = {}
    p = Path(path)
    if not p.exists():
        return out
    try:
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("export "):
                line = line[7:].strip()
            if "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip()
            if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
                value = value[1:-1]
            elif " #" in value:  # 行内注释
                value = value.split(" #", 1)[0].rstrip()
            if key:
                out[key] = value
    except Exception:
        pass
    return out


@dataclass
class LoadedSettings:
    """配置快照（对应 pi 的 immutable, credential-blind snapshot）。"""

    settings: Settings
    sources: Dict[str, str] = field(default_factory=dict)   # path -> default/env_file/environ/runtime/missing
    warnings: List[str] = field(default_factory=list)
    version: str = ""
    loaded_at: float = 0.0
    files: List[str] = field(default_factory=list)

    # ---- 取值 ----
    def get(self, path: str, fallback: Any = None) -> Any:
        node: Any = self.settings
        for part in path.split("."):
            node = getattr(node, part, None)
            if node is None:
                return fallback
        return node

    def require(self, path: str) -> Any:
        value = self.get(path)
        if value in (None, "") or self.sources.get(path) == "missing":
            raise ConfigError(f"必需配置 {path} 缺失（来源：{self.sources.get(path, 'unknown')}）")
        return value

    def warning_count(self) -> int:
        return len(self.warnings)

    # ---- 展示 ----
    def describe(self, reveal_prefix: int = 6) -> Dict[str, Any]:
        """安全展示：密钥只给前 N 位 + 长度，永不回传真值。"""
        rows: Dict[str, Any] = {}
        for f in FIELDS:
            raw = self.get(f.path)
            if f.sensitive:
                secret = (
                    raw.get_secret_value()
                    if hasattr(raw, "get_secret_value")
                    else str(raw or "")
                )
                if not secret:
                    shown = "(empty)"
                else:
                    shown = f"{secret[:reveal_prefix]}…(len={len(secret)})"
            elif hasattr(raw, "model_dump"):
                shown = raw.model_dump(mode="json")
            else:
                shown = raw
            rows[f.path] = {
                "value": shown,
                "source": self.sources.get(f.path, "unknown"),
                "env": f.env,
                "required": f.required,
                "doc": f.doc,
            }
        return rows

    def to_dict(self) -> Dict[str, Any]:
        """完整快照（含密钥明文）—— 只给进程内部用，绝不能直接回给 HTTP。"""
        return self.settings.model_dump()

    # ---- 派生判断（业务侧据此降级，而不是捕获异常）----
    @property
    def llm_ready(self) -> bool:
        return bool(self.settings.llm.api_key.get_secret_value())

    @property
    def hyde_usable(self) -> bool:
        return self.settings.rag.hyde_enabled and self.settings.hyde.usable

    @property
    def judge_usable(self) -> bool:
        return self.settings.judge.usable


# ============================================================
# 加载
# ============================================================
def _set_path(data: Dict[str, Any], path: str, value: Any) -> None:
    node = data
    parts = path.split(".")
    for part in parts[:-1]:
        node = node.setdefault(part, {})
    node[parts[-1]] = value


def _flatten_validation_errors(exc: ValidationError) -> List[str]:
    """把 pydantic 的 loc ('agent', 'max_rounds') 还原成 'agent.max_rounds'。"""
    paths: List[str] = []
    for err in exc.errors():
        loc = [str(p) for p in err.get("loc", ()) if p not in ("__root__",)]
        if loc:
            paths.append(".".join(loc))
    return paths


def load(
    env_files: Tuple[str, ...] = (),
    runtime: Optional[Dict[str, Any]] = None,
) -> LoadedSettings:
    """按 default → env_file → environ → runtime 的优先级组装配置快照。"""
    warnings: List[str] = []
    values: Dict[str, Any] = {}
    sources: Dict[str, str] = {}

    # ① 收集各层原始值（后层覆盖前层，同时记录来源）
    file_values: Dict[str, str] = {}
    used_files: List[str] = []
    for path in env_files:
        got = read_env_file(path)
        if got:
            file_values.update(got)
            used_files.append(str(path))

    runtime = runtime or {}
    for f in FIELDS:
        raw: Any = None
        source = "missing"

        # runtime 层用 path 取值，其余层用 env 名
        if f.path in runtime:
            raw, source = runtime[f.path], "runtime"
            # runtime 允许直接给已转换的值
            if isinstance(raw, (bool, int, float)) and f.cast is not bool:
                pass
        elif f.env in os.environ:
            raw, source = os.environ[f.env], "environ"
        elif f.env in file_values:
            raw, source = file_values[f.env], "env_file"

        # ② 模板解析（配置值里可以写 ${OTHER_KEY}）
        if isinstance(raw, str) and "$" in raw:
            merged_env = {**file_values, **os.environ}
            resolved = resolve(raw, merged_env)
            if resolved is None:
                warnings.append(
                    f"{f.path} 引用了不存在的变量 {env_names(raw)}，整值按未配置处理"
                )
                raw = None
                source = "missing"
            else:
                raw = resolved

        # ③ 类型转换（失败回默认）
        if raw is None:
            values_path_value = f.default
            if f.default in (None, "", []) and not f.required:
                sources[f.path] = "missing"
            else:
                sources[f.path] = "default"
        else:
            casted = _coerce(raw, f.cast, f.path, warnings)
            if casted is None:
                values_path_value = f.default
                sources[f.path] = "default"
            else:
                values_path_value = casted
                sources[f.path] = source

        _set_path(values, f.path, values_path_value)

    # ④ 整体校验 + 逐字段回退
    for attempt in range(3):
        try:
            settings = Settings.model_validate(values)
            break
        except ValidationError as exc:
            bad = _flatten_validation_errors(exc)
            if not bad:
                warnings.append("配置整体校验失败且无法定位字段，全部回默认")
                values = {}
                for f in FIELDS:
                    _set_path(values, f.path, f.default)
                    sources[f.path] = "default"
                continue
            for path in bad:
                fb = default_value(path)
                warnings.append(f"{path} 校验未通过，已回退默认值 {fb!r}")
                _set_path(values, path, fb)
                sources[path] = "default"
    else:  # 三次都失败 → 用纯默认兜底（服务必须能起来）
        values = {}
        for f in FIELDS:
            _set_path(values, f.path, f.default)
            sources[f.path] = "default"
        warnings.append("校验连续失败，已使用全默认配置启动")
        settings = Settings.model_validate(values)

    # 版本号：内容指纹，配置一改就变，可写进 trace 做归因
    fingerprint = hashlib.sha1(
        repr(settings.model_dump(mode="json")).encode("utf-8")
    ).hexdigest()[:12]

    return LoadedSettings(
        settings=settings,
        sources=sources,
        warnings=warnings,
        version=f"cfg-{fingerprint}",
        loaded_at=round(time.time(), 3),
        files=used_files,
    )


# ============================================================
# 进程单例
# ============================================================
_LOCK = threading.Lock()
_CURRENT: Optional[LoadedSettings] = None


def get_settings() -> LoadedSettings:
    """取当前配置快照；首次调用时加载（幂等、线程安全）。"""
    global _CURRENT
    if _CURRENT is not None:
        return _CURRENT
    with _LOCK:
        if _CURRENT is None:
            from allagent2.config.paths import DEFAULT_ENV_FILES

            _CURRENT = load(DEFAULT_ENV_FILES)
    return _CURRENT


def set_settings(loaded: LoadedSettings) -> LoadedSettings:
    """直接替换快照（测试用）。"""
    global _CURRENT
    with _LOCK:
        _CURRENT = loaded
    return loaded


def reload(runtime: Optional[Dict[str, Any]] = None) -> LoadedSettings:
    """重新加载（热更新）。返回新快照，version 会变化。"""
    from allagent2.config.paths import DEFAULT_ENV_FILES

    return set_settings(load(DEFAULT_ENV_FILES, runtime))


def describe() -> Dict[str, Any]:
    loaded = get_settings()
    return {
        "version": loaded.version,
        "loaded_at": loaded.loaded_at,
        "files": loaded.files,
        "warnings": loaded.warnings,
        "values": loaded.describe(),
    }
