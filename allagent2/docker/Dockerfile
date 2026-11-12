# syntax=docker/dockerfile:1
# 构建上下文：练手agent/（仓库根），内含 fastap2 与 allagent2
FROM python:3.13-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

# 编译部分 Python 包（bcrypt / cryptography 等）需要 gcc；
# redis / mysql 客户端为纯 Python，无需额外 dev 库
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 先拷依赖清单，利用层缓存（requirements 不变则无需重装）
COPY fastap2/requirements.txt /app/requirements.txt
RUN pip install --upgrade pip && pip install -r /app/requirements.txt

# 拷源码（.dockerignore 已排除 key.env / mysql.env / .env / rag/models 等）
# 大模型权重（embedding / reranker）运行时从 HF 镜像源下载，不进镜像
COPY fastap2 /app/fastap2
COPY allagent2 /app/allagent2

# 多进程管理 + 依赖等待脚本
RUN pip install supervisor
COPY supervisord.conf /etc/supervisor/conf.d/supervisord.conf
COPY wait_for.py /app/wait_for.py
COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

# 默认环境变量（可被 compose / docker run -e 覆盖）
ENV PYTHONPATH=/app/fastap2:/app/allagent2 \
    ALLAGENT_DIR=/app/allagent2 \
    RAG_PERSIST_DIR=/data/chroma \
    HF_ENDPOINT=https://hf-mirror.com \
    HF_HOME=/root/.cache/huggingface

EXPOSE 8000 8001 8002
WORKDIR /app/fastap2
ENTRYPOINT ["/app/entrypoint.sh"]
