"""养老数据 MCP Server(端口 8001,streamable-HTTP)。

启动:cd fastap2 && python -m mcp_servers.elder_server

【安全要点】
- 每个工具多一个 agent_context 参数:由 fastap2 的 mcp_runtime 在每次调用时
  自动注入(HMAC 签名的权限上下文令牌),LLM 不需要也不应该填它;
- 服务端验签 → 工具白名单 → 老人越权校验 → 查库 → 脱敏 → 审计,全在本进程内
  的 data_gateway 完成(fail-closed:令牌无效一律 403)。
"""
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP

# MCP Server 是独立进程,自己加载 fastap2 的 env(AGENT_CONTEXT_SECRET/MYSQL_*)
load_dotenv(Path(__file__).resolve().parent.parent / "core" / "key.env")
load_dotenv(Path(__file__).resolve().parent.parent / "database" / "mysql.env")

from mcp_servers.data_gateway import execute  # noqa: E402

mcp = FastMCP("elder-data", host="127.0.0.1", port=8001)


@mcp.tool()
def query_elder_profile(elder_id: str, agent_context: str = "") -> str:
    """根据老人编号查询老人档案:姓名、年龄、性别、自理状态、慢性病、联系电话;查不到会提示无记录"""
    return execute("query_elder_profile", {"elder_id": elder_id}, agent_context)


@mcp.tool()
def query_health_records(elder_id: str, metric_type: Optional[str] = None, agent_context: str = "") -> str:
    """根据老人编号查询其最新健康监测记录(指标类型、数值、记录时间),最多返回10条;metric_type 如 血压/血糖/心率,不填返回全部最新"""
    return execute(
        "query_health_records",
        {"elder_id": elder_id, "metric_type": metric_type},
        agent_context,
    )


if __name__ == "__main__":
    mcp.run(transport="streamable-http")
