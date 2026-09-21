#!/usr/bin/env bash
# verl-agent 远程 CUDA 环境一键安装。
#
# ⚠️ 本脚本**只在远程 CUDA 机器上执行**。本机是 macOS arm64，vllm/flash-attn 均无
#    wheel，跑本脚本必然失败（这是预期行为，不是 bug）。
#
# 用法：
#   bash deploy/setup_remote.sh                 # 完整安装
#   bash deploy/setup_remote.sh --base          # 只装框架（跳过 ALFWorld）
#   bash deploy/setup_remote.sh --env-name myenv
#   WORK_DIR=/path bash deploy/setup_remote.sh  # 指定大容量盘（见下）
#
# 磁盘布局（重要）
# ----------------
# 所有体积大的东西（conda 环境 / pip 缓存 / HF 缓存 / ALFWorld 数据）都放在
# `WORK_DIR` 下，默认 `$HOME`。**在 AutoDL/SeetaCloud 这类系统盘很小的平台上
# 必须显式指定**，否则会把系统盘撑满：
#     WORK_DIR=/root/autodl-tmp bash deploy/setup_remote.sh
# 脚本会在 AutoDL 上自动探测并采用该路径（见下面 detect_work_dir）。
#
# 版本依据：上游 README 的 Installation 章节 + repo/setup.py。
#   注意 repo/requirements.txt 与 setup.py 的 tensordict 约束**交集为空**，
#   两套装不到一起；`pip install -e .` 实际生效的是 setup.py，故本脚本以 setup.py 为准。

set -euo pipefail

ENV_NAME="verl-agent"
PY_VERSION="3.12"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/repo"
WITH_ALFWorld=1
WORK_DIR="${WORK_DIR:-}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --base)      WITH_ALFWorld=0; shift ;;
        --env-name)  ENV_NAME="$2"; shift 2 ;;
        --work-dir)  WORK_DIR="$2"; shift 2 ;;
        -h|--help)   sed -n '2,22p' "$0"; exit 0 ;;
        *)           echo "未知参数：$1" >&2; exit 2 ;;
    esac
done

log() { printf '\n\033[1;34m==> %s\033[0m\n' "$*"; }
die() { printf '\n\033[1;31m!! %s\033[0m\n' "$*" >&2; exit 1; }
warn() { printf '\033[1;33m⚠️  %s\033[0m\n' "$*"; }

# ---------------------------------------------------------------- 磁盘布局
# AutoDL/SeetaCloud：系统盘 / 通常只有 30G，数据盘挂在 /root/autodl-tmp。
# 这里优先用它，避免 conda 环境 + 模型权重把系统盘撑爆。
detect_work_dir() {
    if [[ -n "$WORK_DIR" ]]; then echo "$WORK_DIR"; return; fi
    if [[ -d /root/autodl-tmp && -w /root/autodl-tmp ]]; then echo /root/autodl-tmp; return; fi
    echo "$HOME"
}
WORK_DIR="$(detect_work_dir)"
WORK_DIR="${WORK_DIR%/}"

export CONDA_ENVS_PATH="$WORK_DIR/conda-envs"
export PIP_CACHE_DIR="$WORK_DIR/pip-cache"
export HF_HOME="$WORK_DIR/hf"
export ALFWORLD_DATA="$WORK_DIR/alfworld"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"   # 国内镜像，可覆盖

# ⚠️ pip 源必须显式指定。平台上常见的 /etc/pip.conf 指向阿里云，但**本机实测
#    aliyun 只有 25 KB/s**（vllm 的 438MB wheel 要下 5 小时），而清华 6633 KB/s
#    （快 260 倍，同一文件约 70 秒）。实测方法见本文件末尾注释。
export PIP_INDEX_URL="${PIP_INDEX_URL:-https://pypi.tuna.tsinghua.edu.cn/simple}"
export PIP_TRUSTED_HOST="${PIP_TRUSTED_HOST:-pypi.tuna.tsinghua.edu.cn}"

mkdir -p "$CONDA_ENVS_PATH" "$PIP_CACHE_DIR" "$HF_HOME" "$ALFWORLD_DATA"

# ---------------------------------------------------------------- 前置检查
log "前置检查"

# conda 常常没进非交互式 shell 的 PATH（AutoDL 等平台就是这样）。
# 这里主动找一下，免得脚本在第一步就误报"未安装 conda"。
if ! command -v conda >/dev/null 2>&1; then
    for c in /root/miniconda3 /root/anaconda3 /opt/conda "$HOME/miniconda3" "$HOME/anaconda3"; do
        if [[ -f "$c/etc/profile.d/conda.sh" ]]; then
            # shellcheck disable=SC1091
            source "$c/etc/profile.d/conda.sh"
            echo "已从 $c 载入 conda"
            break
        fi
    done
fi
command -v conda >/dev/null || die "未找到 conda。请先安装 miniconda/miniforge。"
[[ -f "$REPO_DIR/setup.py" ]] || die "未找到 $REPO_DIR/setup.py。请确认 repo/ 已克隆。"

# ⚠️ 无 GPU **不是**致命错误。AutoDL 的「无卡模式」正好适合装环境（便宜一个数量级），
#    而安装全程不需要 GPU —— 只有训练/推理才需要。
#    所以这里只警告，不 die。装完之后 GPU 一就位即可直接用。
# ⚠️ 判据必须看**输出是否为空**，不能只看退出码：
#    AutoDL 无卡模式下 `nvidia-smi -L` 和 `--query-gpu` 都**返回 0 但无输出**，
#    只看退出码会误判为"有 GPU"（实测踩过），进而漏设 TORCH_CUDA_ARCH_LIST。
HAS_GPU=0
GPU_LIST=""
if command -v nvidia-smi >/dev/null 2>&1; then
    GPU_LIST=$(nvidia-smi --query-gpu=name,memory.total,driver_version,compute_cap \
               --format=csv,noheader 2>/dev/null | sed '/^[[:space:]]*$/d')
fi
if [[ -n "$GPU_LIST" ]]; then
    HAS_GPU=1
    echo "$GPU_LIST"
else
    warn "未检测到可用 GPU（可能是无卡模式）。环境照常安装 —— 安装不需要 GPU。"
    warn "装完后 check_env.py 会报 CUDA 不可用，那是**预期结果**，不是失败。"
fi

# 无 GPU 时 flash-attn 无法自动探测计算能力，必须显式指定，否则编译会失败。
# 4090 / 4090D 是 Ada，compute capability 8.9。
if [[ "$HAS_GPU" -eq 0 && -z "${TORCH_CUDA_ARCH_LIST:-}" ]]; then
    export TORCH_CUDA_ARCH_LIST="8.9"
    warn "已设 TORCH_CUDA_ARCH_LIST=8.9（4090/4090D 是 Ada）以支持无卡编译 flash-attn。"
    warn "换其他卡请自行覆盖该变量（A100=8.0, H100=9.0, 3090=8.6）。"
fi

echo "工作目录 WORK_DIR = $WORK_DIR"
df -h "$WORK_DIR" | tail -1
AVAIL_G=$(df -BG --output=avail "$WORK_DIR" 2>/dev/null | tail -1 | tr -dc '0-9')
if [[ -n "${AVAIL_G:-}" && "$AVAIL_G" -lt 25 ]]; then
    warn "可用空间仅 ${AVAIL_G}G。vllm+torch 环境约需 15-20G，加上模型与数据可能不够。"
fi

GPU_COUNT=0
if [[ "$HAS_GPU" -eq 1 ]]; then
    GPU_COUNT=$(printf '%s\n' "$GPU_LIST" | wc -l | tr -d ' ')
fi
if [[ "$HAS_GPU" -eq 1 && "$GPU_COUNT" -lt 2 ]]; then
    warn "检测到 $GPU_COUNT 张 GPU。官方 run_alfworld.sh 用 tensor_model_parallel_size=2 且"
    warn "n_gpus_per_node=2，需 >=2 张。单卡必须改配置（tp=1 + 降 batch + 开 offload），"
    warn "见 deploy/README.md 的「单卡配置」一节。"
fi

# ---------------------------------------------------------------- 网络加速
# ⚠️ **不要全局启用 /etc/network_turbo**。AutoDL 自己警告：
#     "开启加速后对访问其他资源如 pip 源等会*更慢*"
# 而本脚本绝大部分流量走 PyPI/conda 镜像，全局加速反而拖慢。
# 只在确实需要访问 Google Drive（ALFWorld 数据）时才临时启用，见下面 ALFWorld 段。

# ---------------------------------------------------------------- 创建环境
log "创建 conda 环境：$ENV_NAME (python $PY_VERSION)"
log "  位置：$CONDA_ENVS_PATH/$ENV_NAME"
if [[ -d "$CONDA_ENVS_PATH/$ENV_NAME" ]]; then
    echo "环境已存在，跳过创建。"
else
    # --override-channels 是必须的：默认 channel 里的 `defaults` 指向
    # repo.anaconda.com，在国内会长时间 read timeout（实测卡在
    # "ReadTimeoutError ... repo.anaconda.com ... /pkgs/r/linux-64/repodata.json.zst"）。
    conda create -p "$CONDA_ENVS_PATH/$ENV_NAME" "python==$PY_VERSION" -y \
        --override-channels \
        -c "${CONDA_MIRROR:-https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main}"
fi

# shellcheck disable=SC1091
eval "$(conda shell.bash hook)"
conda activate "$CONDA_ENVS_PATH/$ENV_NAME" || die "无法激活环境"
echo "已激活：$CONDA_PREFIX"
echo "python: $(python -V 2>&1)"

pip install --upgrade pip

# ---------------------------------------------------------------- 框架
log "安装 vllm==0.11.0（体积较大，耐心等待）"
pip install "vllm==0.11.0"

log "安装 flash-attn"
# ---------------------------------------------------------------------------
# ⚠️ 不要盲目源码编译 —— 在受限容器里这条路基本走不通。实测教训：
#
#   上游 pip install flash-attn==2.7.4.post1 会触发**源码编译**，而 flash-attn
#   2.7.4 的构建脚本会为 sm80/sm90/compute_100/compute_120 等**5 个架构**编译
#   700+ 个 CUDA 文件（即使设了 TORCH_CUDA_ARCH_LIST 也不生效）。
#   并行编译时每个 cc1plus 吃 1-2GB 内存，而 AutoDL 这类容器**内存被 cgroup 限到 2G**
#   （`cat /sys/fs/cgroup/memory.max` 可查；`free` 显示的是宿主机内存，会误导），
#   于是 gcc 直接被 OOM kill：
#       gcc: fatal error: Killed signal terminated program cc1plus
#
# 正确做法：**优先装预编译 wheel**。flash-attention 的 GitHub release 提供了
# 覆盖各 torch 版本 × cxx11abi 变体的 wheel，必须三者都对上：
#     torch 次版本(如 2.8) × python tag(cp312) × cxx11abi(TRUE/FALSE)
# 其中 cxx11abi 要与本机 torch 一致（`torch._C._GLIBCXX_USE_CXX11_ABI`）。
#
# 注意：上游钉的是 2.7.4.post1，而该版本**没有 torch2.8 的 wheel**（只到 torch2.7）。
#       所以 torch 2.8 环境必须用更新的 flash-attn（如 2.8.3）。
#       **这是一处与上游的版本偏离，对照实验时须记录。**
# ---------------------------------------------------------------------------
FA_FALLBACK_VERSIONS="${FA_FALLBACK_VERSIONS:-2.8.3 2.8.1}"

if python -c "import flash_attn" 2>/dev/null; then
    echo "flash-attn 已安装（$(python -c 'import flash_attn;print(flash_attn.__version__)' 2>/dev/null)），跳过。"
else
    FA_INFO=$(python - <<'PY'
import torch, sys
tv = torch.__version__.split("+")[0]                 # 2.8.0
minor = ".".join(tv.split(".")[:2])                  # 2.8
abi = "TRUE" if torch._C._GLIBCXX_USE_CXX11_ABI else "FALSE"
print(minor, f"cp{sys.version_info.major}{sys.version_info.minor}", abi)
PY
)
    read -r TORCH_MINOR PY_TAG ABI_TAG <<<"$FA_INFO"
    echo "本机组合：torch ${TORCH_MINOR} / ${PY_TAG} / cxx11abi${ABI_TAG}"

    installed=0
    for v in $FA_FALLBACK_VERSIONS; do
        whl="flash_attn-${v}%2Bcu12torch${TORCH_MINOR}cxx11abi${ABI_TAG}-${PY_TAG}-${PY_TAG}-linux_x86_64.whl"
        url="https://github.com/Dao-AILab/flash-attention/releases/download/v${v}/${whl}"
        out="$WORK_DIR/${whl//%2B/+}"
        echo "尝试预编译 wheel：flash-attn ${v}"
        if curl -sS -L --max-time 900 -o "$out" -f "$url"; then
            # ⚠️ 文件名**必须保持原样**：里面的 python/abi/platform 标签是 pip
            #    判断兼容性的依据，改名会得到 "not a supported wheel on this platform"。
            if pip install --no-deps "$out" 2>&1 | tail -2; then
                installed=1
                echo "✅ 已从 wheel 安装 flash-attn ${v}"
                break
            fi
        else
            echo "   （该版本无匹配 wheel）"
        fi
    done

    if [[ "$installed" -eq 0 ]]; then
        warn "未找到匹配的预编译 wheel，回退到源码编译。"
        warn "已设 MAX_JOBS=${MAX_JOBS:-2} 以规避容器内存限额导致的 OOM。"
        warn "⚠️ 源码编译在 2G 内存容器里**非常慢**（可能数小时），请确认后再继续。"
        MAX_JOBS="${MAX_JOBS:-2}" TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.9}" \
            pip install "flash-attn==2.7.4.post1" --no-build-isolation --no-cache-dir
    fi

    python -c "import flash_attn;print('flash-attn', flash_attn.__version__)" \
        || warn "flash-attn 最终不可用。注意：verl 对它是 try/except 降级（见
             verl/utils/torch_functional.py:31-36），训练仍可能跑起来，只是更慢更耗显存。"
fi

log "以 editable 模式安装 verl-agent"
# 从 repo/ 目录执行，让 pip install -e . 使用 repo/setup.py。
( cd "$REPO_DIR" && pip install -e . )

# ---------------------------------------------------------------- ALFWorld
if [[ "$WITH_ALFWorld" -eq 1 ]]; then
    log "安装 ALFWorld 依赖"
    # ⚠️ gymnasium/stable-baselines3 会拉入自己的 torch/numpy，可能与上面 vllm 的
    #    版本冲突。上游 README 明确建议**每个环境独立 conda env**。若此处引发
    #    torch 被降级，请改用独立环境并把此处注释掉。
    pip install "gymnasium==0.29.1"
    pip install "stable-baselines3==2.6.0"
    pip install alfworld

    log "下载 ALFWorld 资源到 $ALFWORLD_DATA"
    # alfworld 从环境变量 ALFWORLD_DATA 取路径（alfworld/info.py），上面已导出到 WORK_DIR。
    alfworld-download -f \
        || warn "下载失败。若因 gdown 限流，见 deploy/README.md 的排障章节。"
else
    log "已跳过 ALFWorld（--base）"
fi

# ---------------------------------------------------------------- 数据
log "准备训练数据 → $WORK_DIR/data"
# prepare.py 用 hiyouga/geometry3k 仅作**模态与数据量的占位符**，真实任务数据来自
# 环境本身。但仍需 HuggingFace 连通性来下载该占位数据集。
# 默认 local_dir 是 ~/data/verl-agent，这里显式改到 WORK_DIR，避免占用系统盘。
( cd "$REPO_DIR" && python -m examples.data_preprocess.prepare --mode text \
    --local_dir "$WORK_DIR/data/verl-agent" \
    --train_data_size 16 --val_data_size 128 ) \
    || warn "数据准备失败——检查 HuggingFace 连通性（当前 HF_ENDPOINT=$HF_ENDPOINT）。"

# ---------------------------------------------------------------- 自检
log "环境自检"
python "$(dirname "${BASH_SOURCE[0]}")/check_env.py" --repo "$REPO_DIR" || true

log "完成"
REPO_DIR="$REPO_DIR" WORK_DIR="$WORK_DIR" python - <<'PYEOF'
import os
repo, work = os.environ["REPO_DIR"], os.environ["WORK_DIR"]
print(f"""
后续步骤（WORK_DIR={work}）：

0. 本环境的所有路径都已指向 WORK_DIR，**下次登录务必先导出同样的变量**，
   否则 conda / HF / ALFWorld 会找不到东西：
     export CONDA_ENVS_PATH={work}/conda-envs
     export HF_HOME={work}/hf
     export PIP_CACHE_DIR={work}/pip-cache
     export ALFWORLD_DATA={work}/alfworld
     source /root/miniconda3/etc/profile.d/conda.sh
     conda activate {work}/conda-envs/verl-agent
   也可直接 source {repo}/../deploy/env.sh（如果存在）。

1. 拉取模型权重（约 3GB）：
     huggingface-cli download Qwen/Qwen2.5-1.5B-Instruct

2. 跑 baseline：
     单卡必须先改配置（tp=1 + 降 batch + 开 offload），详见 deploy/README.md。
     cd {repo} && bash examples/gigpo_trainer/run_alfworld.sh
""")
PYEOF
cat <<'EOF'

3. 日志里留意 "Avg size of step-level group"（gigpo/core_gigpo.py:330 无条件打印）
   —— 这是 GiGPO 步级分组的平均组大小，直接关系到 step 级信号是否有效。
   P0 的 E0.3 估计它是 1.17（合成数据），**真实值正好可以在这次训练里实测出来**。
EOF
