"""[t02] 基础设施连通性探测：Redis / MySQL / embedding / vLLM / MCP。

不可用的服务 → 标明影响范围；探测结果缓存供 run_all 判断后续分组是否 SKIP。
"""
from _lib import check, probe_http, results

infra: dict = {}


def run():
    print("[t02] 基础设施连通性")

    # ---- 先加载 key.env（本组在 main 导入之前运行，env 还没装配）----
    from agent_bridge.bootstrap import _load_envs
    _load_envs()
    import os

    # ---- Redis ----
    try:
        import redis
        r = redis.Redis(
            host=os.getenv("REDIS_HOST", "127.0.0.1"),
            port=int(os.getenv("REDIS_PORT", "6379")),
            password=os.getenv("REDIS_PASSWORD") or None,
            db=int(os.getenv("REDIS_DB", "0")),
            socket_connect_timeout=3,
        )
        r.ping()
        infra["redis"] = True
        check("Redis: PING", True)
    except Exception as e:
        infra["redis"] = False
        check("Redis: PING", False, f"不可用(记忆/限流/checkpointer 将降级): {type(e).__name__}")

    # ---- MySQL ----
    try:
        from sqlalchemy import create_engine, text
        eng = create_engine(
            f"mysql+pymysql://{os.getenv('MYSQL_USER')}:{os.getenv('MYSQL_PASSWORD')}"
            f"@{os.getenv('MYSQL_HOST', '127.0.0.1')}:{os.getenv('MYSQL_PORT', '3306')}"
            f"/{os.getenv('MYSQL_DATABASE')}?charset=utf8mb4",
            pool_pre_ping=True,
        )
        with eng.connect() as conn:
            conn.execute(text("SELECT 1"))
        infra["mysql"] = True
        check("MySQL: SELECT 1", True)
    except Exception as e:
        infra["mysql"] = False
        check("MySQL: SELECT 1", False, f"不可用: {type(e).__name__}")

    # ---- MCP 双服务 ----
    infra["mcp_elder"] = probe_http("http://127.0.0.1:8001/mcp", timeout=3)
    infra["mcp_life"] = probe_http("http://127.0.0.1:8002/mcp", timeout=3)
    check("MCP: elder(8001) 可达", infra["mcp_elder"])
    check("MCP: life(8002) 可达", infra["mcp_life"])

    # ---- vLLM（可选服务）----
    ok = probe_http(os.getenv("VLLM_BASE_URL", "http://127.0.0.1:8005/v1/models"), timeout=3)
    infra["vllm"] = ok
    check("vLLM: 本地推理服务", ok, "可用→跑真链路用例" if ok else "不可用→LLM 用例将 SKIP")

    # ---- RAG 模块可导入（重依赖延迟导入，不在此时加载模型）----
    try:
        from allagent2.rag.rag import search_knowledge  # noqa: F401
        infra["embedding"] = True
        check("RAG: 模块可导入", True)
    except Exception as e:
        infra["embedding"] = False
        check("RAG: 模块可导入", False, str(e)[:80])

    return results()


if __name__ == "__main__":
    run()
