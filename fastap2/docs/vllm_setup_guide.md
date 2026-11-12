# 从 Miniconda 到 vLLM：全流程命令详解（RTX 4060 Laptop 8GB）

> 适用：Windows 11 + WSL2 + Qwen2.5-3B-Instruct-AWQ + vLLM 0.31 + FastAPI 项目接入  
> 每条命令都包含：做什么 / 为什么 / 关键参数 / 踩过的坑

---

## 0. 整体架构（先看懂再动手）

```
┌─ Windows ─────────────────────────────────────────────┐
│  conda(eldercare) ── FastAPI 后端(:8000)               │
│       │                │ 连接                          │
│       │          MCP elder(:8001) / life(:8002)        │
│  NVIDIA 驱动(CUDA 13.4) ── GPU 硬件唯一入口             │
│       │ WSL2 GPU 直通(不装 Linux 驱动)                  │
├─ WSL2 (Ubuntu) ───────────────────────────────────────┤
│  venv-vllm ── vLLM 服务(:8005, OpenAI 兼容)            │
│  ~/models/qwen2.5-3b-awq (权重 ~2GB, ext4 本地盘)      │
└────────────────────────────────────────────────────────┘
```

数据流：FastAPI → `OPENAI_BASE_URL=http://127.0.0.1:8005/v1` → vLLM → GPU。  
WSL2 的 127.0.0.1 与 Windows 互通（localhost 转发）。

---

## 1. Windows 侧：Miniconda 与项目环境

### 1.1 安装 Miniconda

```powershell
winget install -e --id Anaconda.Miniconda3
```

- **做什么**：装一个独立的 Python 发行版管理器，把"项目环境"和"系统 Python"隔离。
- **为什么**：系统 Python(E:\python) 已堆 332 个包，版本互相污染；重装/迁移都困难。
- **坑**：base 环境自带 Python 3.14——太新，主流 ML 库轮子不齐，**永远不要直接用 base 跑项目**。

### 1.2 创建项目环境（Python 3.12）

```bash
conda create -n eldercare python=3.12 -y
conda activate eldercare
```

- **为什么 3.12**：langchain/langgraph/torch 生态的轮子最全；3.13/3.14 部分包没有预编译版本。
- **-y**：跳过确认提示。
- Git Bash 里 conda 不在 PATH 时先执行：`source /d/miniconda/etc/profile.d/conda.sh`

### 1.3 按锁定清单装依赖

```bash
python -m pip install -r requirements.lock.txt
```

- **为什么锁版本**：如 `langgraph-checkpoint-redis==0.5.2`——它把 checkpoint 存 RedisJSON，  
  换版本序列化行为就变，多轮对话直接 500。锁文件=可复现环境。
- **验证**：装完跑 `python e2e_stream.py`。

---

## 2. WSL2 安装与配置

### 2.1 安装（管理员 PowerShell）

```powershell
wsl --install -d Ubuntu
wsl --update
wsl --shutdown
```

- `--install`：装内核+默认发行版；`--update`：升级到 WSL2；`--shutdown`：改配置后必须重启 WSL 才生效。

### 2.2 GPU 直通配置（关键认知）

- **WSL 内不装任何 NVIDIA 驱动**，`nvidia-smi` 用的是 Windows 侧驱动直通。
- **禁止**：`sudo apt install nvidia-utils-*`（会装 Linux 驱动，覆盖直通库，GPU 直接消失）。
- Windows 侧配置文件是 `C:\Users\<用户>\.wslconfig`（没有就新建）：
  ```ini
  [wsl2]
  memory=8GB
  processors=8
  # guiApplications=false   # 关 WSLg 图形支持可省 ~1GB 显存，按需打开
  ```
- **血的教训**：`[wsl2]` 段属于 Windows 侧的 `.wslconfig`；误写进 Linux 内的  
  `/etc/wsl.conf` 会报 "Unknown key"，且 `/usr/lib/wsl/lib` 不再挂载 → `nvidia-smi` 消失。  
  修复：`sudo rm /etc/wsl.conf && wsl --shutdown`。
- **驱动版本对齐**：Windows 驱动太旧（CUDA 12.7）会与 vLLM 自带的 torch cu130 不匹配，  
  `torch.cuda.is_available()` 返回 False。升级 GeForce 驱动后验证：
  ```bash
  nvidia-smi            # WSL 内能显示显卡即直通成功
  python -c "import torch; print(torch.cuda.is_available())"
  ```

---

## 3. WSL 内：vLLM 运行环境

### 3.1 系统依赖（一次性）

```bash
sudo apt update && sudo apt install -y build-essential
```

- **为什么**：vLLM 的 Triton 启动时要用 **C 编译器**现编 CUDA 辅助模块。  
  没装会报：`RuntimeError: Failed to find C compiler. Please specify via CC ...`

### 3.2 虚拟环境（venv 就够，不必 conda）

```bash
python3 -m venv ~/venv-vllm
source ~/venv-vllm/bin/activate
pip install vllm
```

- **为什么 venv**：vLLM 是单一用途的推理服务，venv 隔离足够且更轻。
- 安装会自动带上匹配版本的 torch（cu12x/13x）。
- **注意权限**：始终用普通用户（dym），**不要 su root**——`~` 会解析到 /root，  
  模型路径、缓存属主全乱。

---

## 4. 模型下载（国内源）

### 4.1 hf-mirror 镜像

```bash
source ~/venv-vllm/bin/activate
export HF_ENDPOINT=https://hf-mirror.com     # HuggingFace 国内镜像
export HF_HUB_DISABLE_XET=1                  # 必须禁 Xet（见下）
hf download Qwen/Qwen2.5-3B-Instruct-AWQ --local-dir ~/models/qwen2.5-3b-awq
```

- **模型名怎么选**：`Qwen/Qwen2.5-3B-Instruct-AWQ` = 官方组织/模型/用途/AWQ 4bit 量化。  
  仓库名不存在会报 `Model not found`（先 curl API 确认再下）。
- **为什么禁 Xet**：hf-mirror 不代理 Xet 传输协议，直连 `cas-server.xethub.hf.co`  
  会 401。判据：下载进度条出现 `Reconstructing` 字样 = Xet 还在跑，没禁干净。  
  彻底解法：`pip uninstall -y hf-xet`。
- **备选（纯国内 CDN）**：
  ```bash
  pip install -U modelscope
  modelscope download --model Qwen/Qwen2.5-3B-Instruct-AWQ --local_dir ~/models/qwen2.5-3b-awq
  ```

### 4.2 下载的两个隐藏知识

- **权重必须放 WSL 的 ext4**（~/models），不要放 /mnt/d 再让 vLLM 读——9P 跨文件系统  
  加载 safetensors 慢一个数量级。
- **缓存是双份的**：`hf download` 同时在 `~/.cache/huggingface/hub/` 留一份 blob 副本。  
  删模型要两处一起删：
  ```bash
  rm -rf ~/models/qwen2.5-7b-awq
  rm -rf ~/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct-AWQ
  rm -rf ~/.cache/huggingface/xet
  ```

---

## 5. 启动 vLLM 服务（核心命令逐参数讲解）

### 5.1 最终可用版本

```bash
source ~/venv-vllm/bin/activate
export HF_HUB_OFFLINE=1                  # 权重已本地化，彻底断网，杜绝再走 HF
export CC=/usr/bin/gcc                   # 显式给 Triton 编译器路径
export VLLM_USE_FLASHINFER_SAMPLER=0     # 不用 flashinfer 采样（需 nvcc，WSL 没装）
nohup python -m vllm.entrypoints.openai.api_server \
  --model ~/models/qwen2.5-3b-awq \
  --served-model-name qwen2.5-3b-awq \
  --host 0.0.0.0 --port 8005 \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.80 \
  --enable-auto-tool-choice --tool-call-parser hermes \
  > ~/vllm.log 2>&1 &

tail -f ~/vllm.log     # Ctrl+C 退出观察，不影响后台服务
```

| 参数                              | 作用         | 为什么是这个值                                                           |
| ------------------------------- | ---------- | ----------------------------------------------------------------- |
| `--model`                       | 权重目录       | 放 ext4 本地盘                                                        |
| `--served-model-name`           | API 里的模型别名 | **必须与项目 `OPENAI_MODEL` 一致**，客户端按名字路由                              |
| `--host 0.0.0.0`                | 监听所有网卡     | 默认 127.0.0.1 时 Windows 侧可能连不进 WSL                                 |
| `--max-model-len 8192`          | 最大上下文      | `prompt + max_tokens ≤ 此值`；系统提示 3000+，4096 不够用                    |
| `--gpu-memory-utilization 0.80` | 显存预算上限     | 8GB×0.80=6.4GB；减去权重 2GB、CUDA graph 0.4GB、CUDA context，剩 ~3GB 给 KV |
| `--enable-auto-tool-choice`     | 允许模型发起工具调用 | agent 项目必需                                                        |
| `--tool-call-parser hermes`     | 工具调用格式解析器  | Qwen2.5 使用 Hermes 格式；不配则工具调用返回裸文本                                 |

### 5.2 环境变量三件套

- `HF_HUB_OFFLINE=1`：启动完全离线。权重已就位后它是最稳的保险。
- `CC=/usr/bin/gcc`：Triton 编译需要；配合 `apt install build-essential`。
- `VLLM_USE_FLASHINFER_SAMPLER=0`：flashinfer 采样算子要 **nvcc JIT 编译**，  
  WSL 没装 CUDA toolkit 会报  
  `Could not find nvcc and default cuda_home='/usr/local/cuda' doesn't exist`。  
  改用 PyTorch 原生采样，3B 模型性能差异可忽略。（要彻底解决可 `apt install nvidia-cuda-toolkit`）

### 5.3 判定成功的三行日志

```bash
grep -E "Available KV cache memory|Model loading took|Application startup complete" ~/vllm.log
```

- `Model loading took 2.0x GiB` —— 权重装进显存
- `Available KV cache memory: ~3 GiB` —— **正数**才放行（负数=显存不够，见 5.4）
- `Application startup complete.` —— HTTP 服务就绪

### 5.4 本次踩过的五个坑（完整对照表）

| 报错                                                            | 根因                          | 解法                                            |
| ------------------------------------------------------------- | --------------------------- | --------------------------------------------- |
| `Failed to find C compiler`                                   | WSL 无 gcc                   | `apt install build-essential`                 |
| `Engine core initialization failed` + KV cache 为 **-1.17GiB** | 显存预算 < 权重+graph+上下文         | 提 `--gpu-memory-utilization` / 关 WSLg / 换更小模型 |
| `unrecognized arguments: --swap-space`                        | vLLM 0.31 已移除该参数            | 删掉（V1 引擎不用 CPU swap）                          |
| `unrecognized arguments: --no-torch-compile`                  | 参数不存在                       | 用 `--enforce-eager`                           |
| `Could not find nvcc`                                         | flashinfer 采样需 CUDA toolkit | `VLLM_USE_FLASHINFER_SAMPLER=0`               |
| `maximum context length is 4096 ... 4097 tokens`              | prompt+max_tokens 超上限       | `--max-model-len 8192`（vLLM 严格执行，云端 API 从不报）  |

显存账本公式（能背下来，面试能讲）：

```
KV bytes = 2 × 层数 × KV头数 × head_dim × dtype字节 × token数
Qwen2.5-3B: 2×36×2×128×2 ≈ 36 KB/token → 3GB KV ≈ 8 万 token 容量
总需求 ≈ 权重 + KV + CUDA graph(~0.4G) + CUDA context(~0.5G) + 激活值
```

### 5.5 日常操作

```bash
# 停止
pkill -f vllm
# 看实时日志
tail -f ~/vllm.log
# 冒烟测试
curl http://127.0.0.1:8005/v1/models
curl http://127.0.0.1:8005/v1/chat/completions -H "Content-Type: application/json" \
  -d '{"model":"qwen2.5-3b-awq","messages":[{"role":"user","content":"你好"}],"max_tokens":100}'
```

---

## 6. 项目接入（FastAPI ← vLLM）

### 6.1 key.env（`allagent2/core/key.env`）

```ini
OPENAI_API_KEY="local"                       # vLLM 不校验，非空即可
OPENAI_BASE_URL="http://127.0.0.1:8005/v1"   # 指向本地推理服务
OPENAI_MODEL="qwen2.5-3b-awq"                # = --served-model-name
```

切回云端 DeepSeek：注释这三行，放开 DeepSeek 那组，重启后端。**一行切换，双路并存。**

### 6.2 agent.py 三个要点（模型构造层）

1. `model=os.getenv("OPENAI_MODEL", "deepseek-chat")` —— 模型名可配置；
2. `model_provider="openai"` —— **必须显式**：langchain 的 `init_chat_model` 会按模型名  
   猜 provider，猜成 deepseek 就走 DeepSeek 私有协议；
3. `extra_body={"thinking": ...}` 只在模型名含 deepseek 时发送 —— 私有字段发给 vLLM  
   属于脏参数。

### 6.3 启动顺序（三个 PowerShell 窗口）

```powershell
python -m mcp_servers.elder_server    # :8001 养老数据 MCP
python -m mcp_servers.life_server     # :8002 生活工具 MCP
python -m uvicorn main:app --port 8000
```

- **必须 `-m` 模块方式启动**：直接 `python mcp_servers\elder_server.py` 会把包目录  
  当脚本根目录，`from mcp_servers.data_gateway import ...` 直接 ModuleNotFoundError。
- 验证：`python e2e_stream.py`（登录→SSE 流式→phase 时序→final→structured→多轮）。

---

## 7. 速查卡（贴墙用）

```bash
# ===== 每次开机（WSL）=====
source ~/venv-vllm/bin/activate
export HF_HUB_OFFLINE=1 CC=/usr/bin/gcc VLLM_USE_FLASHINFER_SAMPLER=0
nohup python -m vllm.entrypoints.openai.api_server \
  --model ~/models/qwen2.5-3b-awq --served-model-name qwen2.5-3b-awq \
  --host 0.0.0.0 --port 8005 --max-model-len 8192 \
  --gpu-memory-utilization 0.80 \
  --enable-auto-tool-choice --tool-call-parser hermes > ~/vllm.log 2>&1 &

# ===== 确认 =====
grep -E "KV cache|startup complete" ~/vllm.log
```

## 8. 进阶方向（把它变成真正的 infra 能力）

1. **压测出指标**：TTFT / TPOT / 吞吐，做"本地 3B vs DeepSeek API"对比表；
2. **QLoRA 微调 3B**（LLaMA-Factory，8GB 卡可跑）→ 合并 → 转 AWQ → 部署回 vLLM，  
   形成「训练→量化→部署→应用」完整闭环；
3. **监控**：记录 tokens/s 与显存水位，观察 continuous batching 下的并发表现。
