#!/usr/bin/env bash
# 登录远程机器后 source 一次，恢复 setup_remote.sh 建立的环境。
#
#   source deploy/env.sh
#
# 为什么需要它：`setup_remote.sh` 把 conda 环境、pip 缓存、HF 缓存、ALFWorld 数据
# 都放到了 WORK_DIR（AutoDL 上通常是 /root/autodl-tmp），而不是默认的 $HOME。
# 这些是**环境变量**，不会随登录自动恢复。不 source 的话 conda 找不到环境、
# HuggingFace 会往系统盘里重新下一份。

WORK_DIR="${WORK_DIR:-/root/autodl-tmp}"
ENV_NAME="${ENV_NAME:-verl-agent}"

export CONDA_ENVS_PATH="$WORK_DIR/conda-envs"
export PIP_CACHE_DIR="$WORK_DIR/pip-cache"
export HF_HOME="$WORK_DIR/hf"
export ALFWORLD_DATA="$WORK_DIR/alfworld"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

# AutoDL 学术加速（访问 GitHub / HuggingFace）
if [[ -f /etc/network_turbo ]]; then
    # shellcheck disable=SC1091
    source /etc/network_turbo 2>/dev/null || true
fi

if [[ -f /root/miniconda3/etc/profile.d/conda.sh ]]; then
    # shellcheck disable=SC1091
    source /root/miniconda3/etc/profile.d/conda.sh
fi

if [[ -d "$CONDA_ENVS_PATH/$ENV_NAME" ]]; then
    conda activate "$CONDA_ENVS_PATH/$ENV_NAME" && echo "✅ 已激活 $ENV_NAME（$CONDA_PREFIX）"
else
    echo "⚠️  环境不存在：$CONDA_ENVS_PATH/$ENV_NAME"
    echo "    请先跑 bash deploy/setup_remote.sh"
fi

echo "WORK_DIR=$WORK_DIR"
[[ -n "${HF_HOME:-}" ]] && echo "HF_HOME=$HF_HOME"
