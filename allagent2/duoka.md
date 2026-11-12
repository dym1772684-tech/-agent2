请将以下内容保存为 01_setup_wsl.ps1，并以管理员身份运行。

powershell
<#
.SYNOPSIS
  一键配置 WSL2 + Ubuntu 24.04 环境
.DESCRIPTION
  自动启用 WSL/虚拟机平台，更新 WSL 内核，安装 Ubuntu 24.04，并配置显存优化参数。
  请以管理员身份运行此脚本。
#>

Write-Host "=== 开始配置 WSL2 + Ubuntu 24.04 ===" -ForegroundColor Green

# 1. 检查管理员权限
$isAdmin = ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole] "Administrator")
if (-not $isAdmin) {
    Write-Host "错误：请以管理员身份运行 PowerShell！" -ForegroundColor Red
    Pause
    exit 1
}

# 2. 启用 WSL 和虚拟机平台功能
Write-Host ">>> 正在启用 WSL 和虚拟机平台功能..." -ForegroundColor Yellow
dism.exe /online /enable-feature /featurename:Microsoft-Windows-Subsystem-Linux /all /norestart
dism.exe /online /enable-feature /featurename:VirtualMachinePlatform /all /norestart

Write-Host "功能启用完成，需要重启电脑才能生效。" -ForegroundColor Yellow
$restart = Read-Host "是否现在重启？(Y/N，重启后请再次以管理员身份运行此脚本)"
if ($restart -eq 'Y') {
    Restart-Computer -Force
    exit
} else {
    Write-Host "请手动重启后再次运行此脚本。" -ForegroundColor Yellow
    Pause
    exit
}

# ---------- 以下步骤需在重启后再次运行 ----------
Write-Host "重启后继续..." -ForegroundColor Green

# 3. 更新 WSL 内核
Write-Host ">>> 正在更新 WSL 内核..." -ForegroundColor Yellow
wsl --update
wsl --set-default-version 2

# 4. 安装 Ubuntu 24.04
Write-Host ">>> 正在安装 Ubuntu 24.04 (可能需要几分钟，请耐心等待)..." -ForegroundColor Yellow
# 静默安装，不启动发行版
wsl --install -d Ubuntu-24.04 --no-launch

# 5. 配置 .wslconfig 优化显存与资源
$wslConfigPath = "$env:USERPROFILE\.wslconfig"
Write-Host ">>> 正在写入 .wslconfig (限制 WSL 内存，防止挤占宿主机)..." -ForegroundColor Yellow

$wslConfigContent = @"
[wsl2]
# 限制 WSL 最大使用内存 (根据你的物理内存调整，8GB 是个安全值)
memory=8GB
# 限制处理器核心数，防止 WSL 占满所有 CPU
processors=4
# 关闭 WSLg 可节省约 1GB 显存 (若不需要图形界面)
# guiApplications=false
"@

if (-not (Test-Path $wslConfigPath)) {
    Set-Content -Path $wslConfigPath -Value $wslConfigContent -Encoding UTF8
    Write-Host ".wslconfig 已创建。" -ForegroundColor Green
} else {
    Write-Host ".wslconfig 已存在，请手动检查是否需要追加内容：" -ForegroundColor Yellow
    Write-Host $wslConfigContent
}

Write-Host "`n=== Windows 侧配置完成！ ===" -ForegroundColor Green
Write-Host "后续操作：" -ForegroundColor Cyan
Write-Host "1. 在开始菜单中打开 'Ubuntu 24.04'，完成初始用户名/密码设置。"
Write-Host "2. 进入 Ubuntu 终端后，运行第二部分脚本 (02_setup_vllm.sh)。"
Pause
关键点说明：

dism.exe：这是 Windows 底层的部署映像服务管理工具，比图形界面更可靠地启用 WSL 所需的两个核心功能。

wsl --install -d Ubuntu-24.04 --no-launch：静默安装指定发行版，不自动启动，方便你后续统一操作。

.wslconfig：这是 WSL2 的全局配置文件，必须放在 Windows 用户目录下，而不是 Linux 内部。memory=8GB 和 processors=4 是保守值，可根据学校电脑配置调整。

🐧 第二部分：WSL Ubuntu 内部脚本 (02_setup_vllm.sh)
打开 Ubuntu 终端，将以下内容保存为 02_setup_vllm.sh，然后 chmod +x 并运行。

bash
#!/bin/bash
# ==========================================
# vLLM + Qwen2.5-3B-AWQ 一键部署脚本 (WSL2 Ubuntu)
# ==========================================

set -e

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${GREEN}=== 开始配置 vLLM 环境 ===${NC}"

# 1. 更新系统并安装基础依赖
echo -e "${YELLOW}>>> 更新系统并安装依赖 (gcc, python3-venv 等)...${NC}"
sudo apt-get update
sudo apt-get install -y build-essential python3-venv python3-pip git curl

# 2. 验证 GPU 直通 (关键检查)
echo -e "${YELLOW}>>> 检查 GPU 直通状态...${NC}"
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi
    echo -e "${GREEN}GPU 直通正常。${NC}"
else
    echo -e "警告：nvidia-smi 命令未找到。请确认 Windows 侧已安装 NVIDIA 驱动，且 WSL2 已正确配置 GPU 直通。"
fi

# 3. 创建虚拟环境
VENV_DIR="$HOME/venv-vllm"
if [ ! -d "$VENV_DIR" ]; then
    echo -e "${YELLOW}>>> 创建 Python 虚拟环境...${NC}"
    python3 -m venv "$VENV_DIR"
fi
source "$VENV_DIR/bin/activate"

# 4. 安装 vLLM (使用国内镜像源)
echo -e "${YELLOW}>>> 安装 vLLM (清华源)...${NC}"
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
pip install --upgrade pip
pip install vllm
pip uninstall -y hf-xet  # 防止下载模型时报 401

# 5. 准备模型目录
MODEL_DIR="$HOME/models/qwen2.5-3b-awq"
mkdir -p "$(dirname "$MODEL_DIR")"

echo -e "${YELLOW}>>> 模型准备：${NC}"
echo "如果你已从 U 盘拷贝了模型，请将其放入: $MODEL_DIR"
echo "否则，脚本将从 hf-mirror 下载 (约 2GB)..."
read -p "是否继续下载？(Y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]] || [[ -z $REPLY ]]; then
    export HF_ENDPOINT=https://hf-mirror.com
    if [ ! -d "$MODEL_DIR" ] || [ -z "$(ls -A "$MODEL_DIR" 2>/dev/null)" ]; then
        hf download Qwen/Qwen2.5-3B-Instruct-AWQ --local-dir "$MODEL_DIR"
    else
        echo "模型目录已存在且非空，跳过下载。"
    fi
fi

# 6. 生成启动脚本
START_SCRIPT="$HOME/start_vllm.sh"
cat << 'EOF' > "$START_SCRIPT"
#!/bin/bash
source "$HOME/venv-vllm/bin/activate"
export HF_HUB_OFFLINE=1
export CC=/usr/bin/gcc
export VLLM_USE_FLASHINFER_SAMPLER=0

PORT=8005
if lsof -i :$PORT | grep LISTEN > /dev/null; then
    echo "端口 $PORT 被占用，正在清理旧进程..."
    pkill -f "vllm serve" || true
    sleep 2
fi

echo "正在启动 vLLM 服务 (端口: $PORT)..."
echo "日志: $HOME/vllm.log"
nohup python -m vllm.entrypoints.openai.api_server \
  --model "$HOME/models/qwen2.5-3b-awq" \
  --served-model-name qwen2.5-3b-awq \
  --host 0.0.0.0 --port $PORT \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.80 \
  --enable-auto-tool-choice \
  --tool-call-parser hermes \
  > "$HOME/vllm.log" 2>&1 &

tail -f "$HOME/vllm.log"
EOF

chmod +x "$START_SCRIPT"

echo -e "${GREEN}=========================================="
echo -e "🎉 环境配置完成！"
echo -e "启动命令: ${YELLOW}$HOME/start_vllm.sh${NC}"
echo -e "==========================================${NC}"
🚀 整体使用流程
Windows 侧：将 01_setup_wsl.ps1 拷到学校电脑，以管理员身份运行 PowerShell，执行该脚本。脚本会启用功能、提示重启、然后安装 Ubuntu。

重启后：再次以管理员身份运行 01_setup_wsl.ps1，完成 WSL 更新、Ubuntu 安装和 .wslconfig 配置。

进入 Ubuntu：从开始菜单打开 “Ubuntu 24.04”，设置用户名和密码。

Ubuntu 内部：将 02_setup_vllm.sh 拷入（可通过 \\wsl$\Ubuntu-24.04\home\<用户名> 路径访问），运行它完成 vLLM 部署。

启动服务：执行 ~/start_vllm.sh 即可。

