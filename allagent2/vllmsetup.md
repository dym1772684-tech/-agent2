🚀 第一步：进入环境 & 设置防坑变量
bash
# 1. 激活 vLLM 虚拟环境
source ~/venv-vllm/bin/activate

# 2. 设置三件套环境变量（离线模式 + 编译器路径 + 禁用 FlashInfer 防报错）
export HF_HUB_OFFLINE=1 CC=/usr/bin/gcc VLLM_USE_FLASHINFER_SAMPLER=0
🏃‍♂️ 第二步：后台启动服务
bash
# 3. 启动 vLLM 服务并挂载到后台，日志写入 ~/vllm.log
nohup python -m vllm.entrypoints.openai.api_server \
  --model ~/models/qwen2.5-3b-awq \
  --served-model-name qwen2.5-3b-awq \
  --host 0.0.0.0 --port 8005 \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.80 \
  --enable-auto-tool-choice --tool-call-parser hermes \
  > ~/vllm.log 2>&1 &
🔍 第三步：验证是否启动成功
bash
# 4. 实时查看日志（看到 "Application startup complete" 即成功，按 Ctrl+C 退出查看）
tail -f ~/vllm.log

# 5. 快速测试 API 是否响应（新开一个终端窗口运行）
curl http://127.0.0.1:8005/v1/models
🛑 第四步：如何优雅关闭服务
bash
# 如果下次需要重启或关闭服务，直接杀掉 vllm 进程
pkill -f vllm
🎁 进阶：写成一键启动脚本（推荐）
如果你觉得每次复制这么多很麻烦，可以直接在 WSL 里创建一个脚本。

在 WSL 终端执行：nano ~/start_vllm.sh

把下面这段完整代码粘贴进去：

bash
#!/bin/bash
# 启动 vLLM 的一键脚本

echo "正在启动 vLLM 服务..."

# 激活虚拟环境
source /home/dym/venv-vllm/bin/activate

# 设置环境变量
export HF_HUB_OFFLINE=1 
export CC=/usr/bin/gcc 
export VLLM_USE_FLASHINFER_SAMPLER=0

# 启动服务
nohup python -m vllm.entrypoints.openai.api_server \
  --model /home/dym/models/qwen2.5-3b-awq \
  --served-model-name qwen2.5-3b-awq \
  --host 0.0.0.0 --port 8005 \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.80 \
  --enable-auto-tool-choice --tool-call-parser hermes \
  > /home/dym/vllm.log 2>&1 &

echo "服务已在后台启动！"
echo "查看日志请输入：tail -f ~/vllm.log"
保存退出（Ctrl+O 回车保存，Ctrl+X 退出）。

赋予执行权限：chmod +x ~/start_vllm.sh

以后每次开机，你只需要输入：

bash
~/start_vllm.sh
就全部搞定了！