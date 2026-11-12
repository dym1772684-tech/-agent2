#!/usr/bin/env bash
# ============================================================
# 养老智能体 - conda 环境一键搭建脚本 (Git Bash 下运行)
# 用法:  cd fastap2 && bash setup_env.sh
# 说明:  base 环境是 Python 3.14,太新且不能装包 -> 必须新建 3.12 环境
# ============================================================
set -e

CONDA_ROOT="/d/miniconda"      # Miniconda 实际安装位置
ENV_NAME="eldercare"           # 本项目专用环境名
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "[1/5] 加载 conda ..."
source "$CONDA_ROOT/etc/profile.d/conda.sh"

echo "[2/5] 创建环境 $ENV_NAME (Python 3.12) ..."
if conda env list | grep -q "^${ENV_NAME} "; then
  echo "      环境已存在,跳过创建"
else
  conda create -n "$ENV_NAME" python=3.12 -y
fi

echo "[3/5] 激活环境 ..."
conda activate "$ENV_NAME"
python --version

echo "[4/5] 安装依赖(锁定版本,耗时数分钟) ..."
python -m pip install --upgrade pip
python -m pip install -r "$SCRIPT_DIR/requirements.lock.txt"

echo "[5/5] 验证 ..."
python - <<'PY'
import importlib, sys
print("python:", sys.version.split()[0])
mods = ["fastapi", "langchain", "langgraph", "torch", "chromadb",
        "sentence_transformers", "jieba", "mcp", "sqlalchemy", "redis"]
for m in mods:
    importlib.import_module(m)
    print("  OK", m)
import langgraph, langchain, torch
print("langgraph:", langgraph.__version__ if hasattr(langgraph,'__version__') else '-',
      "| langchain:", langchain.__version__, "| torch:", torch.__version__)
PY

echo "=== 完成 ==="
echo "以后每次开发先执行:  source /d/miniconda/etc/profile.d/conda.sh && conda activate eldercare"
