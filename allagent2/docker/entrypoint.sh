#!/bin/sh
# 容器启动入口：先等 MySQL / Redis 健康，再起 supervisord（uvicorn + 2 个 MCP）
set -e

echo "[entrypoint] 等待 MySQL (${MYSQL_HOST}:${MYSQL_PORT}) ..."
until python /app/wait_for.py mysql "${MYSQL_HOST}" "${MYSQL_PORT}"; do
  echo "[entrypoint] MySQL 未就绪，2s 后重试"
  sleep 2
done

echo "[entrypoint] 等待 Redis (${REDIS_HOST}:${REDIS_PORT}) ..."
until python /app/wait_for.py redis "${REDIS_HOST}" "${REDIS_PORT}"; do
  echo "[entrypoint] Redis 未就绪，2s 后重试"
  sleep 2
done

echo "[entrypoint] 依赖就绪，启动 supervisord"
exec supervisord -c /etc/supervisor/conf.d/supervisord.conf
