# ============================================================
# 养生养老 Agent 项目一键启动脚本
# 启动顺序：elder MCP(8001) → life MCP(8002) → FastAPI 主服务(8000)
# knowledge(8003) 不启动：知识检索走进程内工具（allagent2/tools/knowledge_tool.py）
# 已存在的服务会自动跳过（幂等），可反复执行。
# 用法：powershell -ExecutionPolicy Bypass -File .\start_all.ps1
# ============================================================
$ErrorActionPreference = "Stop"
$PY = "E:\python\python.exe"                              # 【按需改】你的 Python 解释器
$ROOT = "C:\Users\17726\Desktop\练手agent\fastap2"        # 【按需改】fastap2 目录
Set-Location $ROOT

function Test-Port($port) {
    return [bool](Get-NetTCPConnection -State Listen -LocalPort $port -ErrorAction SilentlyContinue)
}

function Start-Background($name, $args, $port) {
    if (Test-Port $port) {
        Write-Host "[SKIP] $name 已在运行 (端口 $port)" -ForegroundColor DarkGray
        return
    }
    Write-Host "[START] $name (端口 $port) …" -ForegroundColor Green
    Start-Process -FilePath $PY -ArgumentList $args -WorkingDirectory $ROOT `
        -RedirectStandardOutput "$ROOT\server_$name.out.log" `
        -RedirectStandardError  "$ROOT\server_$name.err.log" `
        -WindowStyle Hidden
}

# 1) 养老数据 MCP Server（依赖 MySQL + AGENT_CONTEXT_SECRET）
Start-Background "elder_mcp"  @("-m", "mcp_servers.elder_server") 8001
# 2) 生活工具 MCP Server（天气 + 计算器）
Start-Background "life_mcp"   @("-m", "mcp_servers.life_server") 8002
# 3) FastAPI 主服务（依赖前两者 + Redis + DeepSeek key）
Start-Background "main"       @("main.py") 8000

Write-Host ""
Start-Sleep -Seconds 3
Write-Host "当前服务状态：" -ForegroundColor Cyan
foreach ($p in 8000, 8001, 8002) {
    # 注意：$PID 是 PowerShell 只读自动变量，不能赋值（变量名不区分大小写）
    $ownerPid = (Get-NetTCPConnection -State Listen -LocalPort $p -ErrorAction SilentlyContinue).OwningProcess
    if ($ownerPid) { Write-Host "  ✓ 端口 $p 运行中 (PID $ownerPid)" -ForegroundColor Green }
    else      { Write-Host "  ✗ 端口 $p 未启动（看 server_*.err.log）" -ForegroundColor Red }
}
Write-Host ""
Write-Host "打开浏览器访问：http://127.0.0.1:8000/ （普通用户）/ http://127.0.0.1:8000/admin （管理员）" -ForegroundColor Yellow
