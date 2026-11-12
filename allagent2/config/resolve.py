"""配置值引用解析（模仿 pi-main `core/resolve-config-value.ts`）。

============================================================
pi 的做法（三种形态）
------------------------------------------------------------
  字面量       "sk-abc123"
  $VAR 模板    "${OPENAI_API_KEY}" / "$OPENAI_BASE_URL/v1"
  命令替换     "!pass show openai"        ← 本次**不实现**，见下

解析出来的模板是 TemplatePart 列表（literal / env），任何变量缺失 →
整个值视为未配置，**不做部分替换**（否则会拼出 "sk-/v1" 这种半截密钥，
比直接没有还难排查）。

============================================================
为什么本项目不做 `!command`
------------------------------------------------------------
pi 是本地 CLI，用户自己写配置文件、自己信任自己。
本项目是服务端：配置文件一旦能被"读成命令"，就等于给了一个任意代码执行
入口（谁改了 key.env 谁就能在服务进程里跑命令，而且完全绕过审计）。
所以这里明确只支持字面量与环境变量，且在日志里标注"命令替换未启用"。

============================================================
转义规则（与 pi 对齐）
------------------------------------------------------------
  $$        →  $
  $!        →  !
  $NAME     →  环境变量 NAME
  ${NAME}   →  环境变量 NAME（推荐写法，能接非字母数字边界）
  ${1abc}   →  原样保留 "${1abc}"（非法变量名不解析）
"""
from __future__ import annotations

import os
import re
from typing import Dict, List, Optional, Tuple

_ENV_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_ENV_PREFIX = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*")

Part = Tuple[str, str]  # ("literal", text) | ("env", NAME)


def parse_template(text: str) -> List[Part]:
    """把配置字符串切成 literal / env 片段。纯解析，不读环境变量。"""
    parts: List[Part] = []
    buf: List[str] = []

    def flush() -> None:
        if buf:
            parts.append(("literal", "".join(buf)))
            buf.clear()

    i = 0
    n = len(text)
    while i < n:
        ch = text[i]
        if ch != "$":
            buf.append(ch)
            i += 1
            continue

        nxt = text[i + 1] if i + 1 < n else ""
        # $$ → $   /   $! → !
        if nxt in ("$", "!"):
            buf.append(nxt)
            i += 2
            continue

        # ${NAME}
        if nxt == "{":
            end = text.find("}", i + 2)
            if end < 0:
                buf.append("$")
                i += 1
                continue
            name = text[i + 2:end]
            if _ENV_NAME.match(name):
                flush()
                parts.append(("env", name))
            else:
                buf.append(text[i:end + 1])  # 非法名 → 原样保留
            i = end + 1
            continue

        # $NAME
        m = _ENV_PREFIX.match(text[i + 1:])
        if m:
            flush()
            parts.append(("env", m.group(0)))
            i += 1 + len(m.group(0))
            continue

        buf.append("$")
        i += 1

    flush()
    return parts


def env_names(text: str) -> List[str]:
    """这个配置值引用了哪些环境变量（用于 /config 端点展示依赖）。"""
    seen: List[str] = []
    for kind, value in parse_template(text or ""):
        if kind == "env" and value not in seen:
            seen.append(value)
    return seen


def is_reference(text: str) -> bool:
    """是否引用了环境变量（纯字面量返回 False）。"""
    return any(kind == "env" for kind, _ in parse_template(text or ""))


def resolve(text: str, env: Optional[Dict[str, str]] = None) -> Optional[str]:
    """解析配置值。任一变量缺失 → 返回 None（整值视为未配置）。

    env 参数优先于进程环境，便于测试与"运行时覆盖"。
    """
    if text is None:
        return None
    source = os.environ if env is None else {**os.environ, **env}
    out: List[str] = []
    for kind, value in parse_template(text):
        if kind == "literal":
            out.append(value)
            continue
        got = source.get(value)
        if got is None:
            return None
        out.append(got)
    return "".join(out)


def describe_reference(text: str) -> str:
    """把配置值描述成可安全展示的形式：字面量原样，引用只显示变量名。"""
    if not text:
        return ""
    if not is_reference(text):
        return text
    rendered = []
    for kind, value in parse_template(text):
        rendered.append(f"${{{value}}}" if kind == "env" else value)
    return "".join(rendered)
