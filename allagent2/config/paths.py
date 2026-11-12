"""配置文件位置解析。

两个项目平级：练手agent/allagent2 与 练手agent/fastap2。
allagent2 不反向依赖 fastap2，所以这里只按**目录约定**找文件，
找不到就跳过（缺文件不等于配置错误，容器里通常全靠环境变量）。
"""
from __future__ import annotations

import os
from pathlib import Path

AGENT_DIR = Path(__file__).resolve().parents[1]          # .../allagent2（config 的上一层）
WORKSPACE_DIR = AGENT_DIR.parent                          # .../练手agent


def _env_files() -> tuple[str, ...]:
    candidates = [
        AGENT_DIR / "core" / "key.env",                   # 主模型 / HyDE / Judge
        WORKSPACE_DIR / "fastap2" / "core" / "key.env",    # 后端 JWT 等
        WORKSPACE_DIR / "fastap2" / "database" / "mysql.env",
        AGENT_DIR / "core" / "local.env",                 # 本地覆盖（不进版本库）
    ]
    # 额外文件用环境变量追加（容器里挂 ConfigMap 时用）
    extra = os.getenv("AGENT_ENV_FILES", "")
    for item in extra.split(os.pathsep):
        if item.strip():
            candidates.append(Path(item.strip()))
    return tuple(str(p) for p in candidates)


DEFAULT_ENV_FILES: tuple[str, ...] = _env_files()
