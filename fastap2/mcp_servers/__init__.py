"""mcp_servers:养老智能体的工具服务化层(MCP 协议)。

- elder_server(8001):养老数据服务 —— 唯一数据出口,内置 DB 网关+ACL+审计
- life_server(8002):生活工具服务 —— 天气/计算器,无敏感数据

启动(两个终端,或后台):
    cd fastap2
    python -m mcp_servers.elder_server
    python -m mcp_servers.life_server
"""
