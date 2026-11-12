我刚逐个探测了端口，先说当前真实状态：

组件	端口	现状
MySQL	3306	✅ 已在跑（PID 6092）
Redis	6379	❌ 容器 redis 已退出（2 分钟前）
vLLM	8005	❌ 没有监听（WSL 侧也没有）
MCP 三个 + 主服务	8000-8003	❌ 都没有启动
所以要先补三个前置组件，再启动那 4 个 Python 进程。

第 1 步：启动 Redis（PowerShell）
powershell

docker start redis
docker exec redis redis-cli -a 3820023130dym ping     # 返回 PONG 就对了
容器里已经带了 --requirepass 和 --appendonly yes，执行 start 就行，不用再传参数。需要注意它设置了 RestartPolicy=no，因此开机不会自动启动 —— 以后每次重启电脑都得手动执行一次。

这一步必须先做：Redis 连不上时，checkpointer 会自动降级为 InMemorySaver，导致所有会话中断点全部丢失，而且不报错。

第 2 步：启动 vLLM（WSL 终端里执行）
你现在 key.env 里已经切成 qwen2.5-3b-awq。启动不了它，/chat 就会直接返回 503。请先打开一个 WSL 终端：

bash

pkill -f "vllm.entrypoints" ; sleep 2
sudo ss -lptn 'sport = :8005'          # 应该没输出

source /home/dym/venv-vllm/bin/activate
vllm serve /home/dym/models/qwen2.5-3b-awq \
  --served-model-name qwen2.5-3b-awq \
  --host 0.0.0.0 --port 8005 \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.8 \
  --enable-auto-tool-choice \
  --tool-call-parser hermes
等到出现 Uvicorn running on http://0.0.0.0:8005 后，在 Windows 侧验证：

powershell

curl http://127.0.0.1:8005/v1/models
如果返回的 JSON 里 id 是 qwen2.5-3b-awq，就说明已经通了。这个终端的窗口不要关闭。

第 3 步：启动 4 个 Python 进程（PowerShell，另开 4 个窗口）
方便查看日志，也可以直接跳到下面的“一条命令版本”。

powershell

cd C:\Users\17726\Desktop\练手agent\fastap2

# 窗口 1
E:\python\python.exe -m mcp_servers.elder_server       # 8001

# 窗口 2
E:\python\python.exe -m mcp_servers.life_server        # 8002

# 窗口 3
E:\python\python.exe -m mcp_servers.knowledge_server   # 8003
8003 这个进程最慢 —— 它需要加载 embedding 模型，可能等 30-60 秒。要等到它打印出监听 8003 后再继续，否则主服务启动时抓不到 search_knowledge_base，整个 RAG 能力就等于没有。

powershell

# 窗口 4（等上面三个都起来后）
E:\python\python.exe main.py                            # 8000
一条命令替代第 3 步（在后台启动，已在运行的会自动跳过）：

powershell

powershell -ExecutionPolicy Bypass -File C:\Users\17726\Desktop\练手agent\fastap2\start_all.ps1
第 4 步：验证
powershell

# 装配状态：看 ready / tools_count / topology / checkpointer
curl http://127.0.0.1:8000/api/v1/agent/status
期望看到：ready: true、tools_count 为 8（起齐三个 MCP 时）、checkpointer 持久化标记为 true。如果 tools_count 只有 5，说明 8003 没起或还没就绪。

然后浏览器打开 http://127.0.0.1:8000/dashboard，登录 testuser01 / test123456。

顺序为什么不能乱