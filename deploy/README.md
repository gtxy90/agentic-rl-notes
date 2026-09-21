# 远程部署说明

本目录的资产**只在远程 CUDA 机器上执行**。本机（Apple M5 / 无 CUDA）跑不通——
这是预期行为，见下。

## 为什么本地装不了

verl-agent 的两个硬依赖在 macOS arm64 上都没有 wheel：

| 依赖 | 上游要求 | 本地情况 |
|---|---|---|
| `vllm` | `==0.11.0` | 无 macOS arm64 wheel |
| `flash-attn` | `==2.7.4.post1` | 无 macOS wheel，源码编译也需 CUDA toolkit |

所以本地只能建**轻量 CPU 环境跑算法单测**（`../tests/`），训练必须在远程。

## 快速开始

```bash
# 在远程 CUDA 机器上，从项目根目录执行
bash deploy/setup_remote.sh              # 完整安装（含 ALFWorld）
bash deploy/setup_remote.sh --base       # 只装框架，跳过环境包
bash deploy/setup_remote.sh --env-name myenv
```

脚本会依次完成：前置检查 → 建 conda 环境 → 装 vllm/flash-attn → `pip install -e repo/`
→ 装 ALFWorld 并下载资源 → 准备数据 → 跑自检。

**版本依据**：上游 README 的 Installation 章节 + `repo/setup.py`。

⚠️ **不要用 `repo/requirements.txt`**。它与 `setup.py` 的 `tensordict` 约束**交集为空**
（`requirements.txt:17` 写 `<=0.6.2`，`setup.py:40` 写 `>=0.8.0,<=0.10.0,!=0.9.0`），
两套装不到一起。`pip install -e .` 实际生效的是 `setup.py`，以它为准。

## 自检

```bash
python deploy/check_env.py            # 远程：检查 CUDA/依赖/数据
python deploy/check_env.py --local    # 本地：预期报无 CUDA，用于验证脚本不误报
```

逐项检查 Python 版本、CUDA 与 GPU 数量、各依赖的安装与**实际导入**、repo 可导入性、
数据集与 ALFWorld 资源，最后给出可执行结论。

特别地，它会显式暴露 flash-attn 的降级状态——上游 `verl/utils/torch_functional.py:31-36`
把 flash-attn 的导入包在 `try/except` 里，缺失时**静默**降级为普通实现（训练能跑但更慢更耗显存）。

## 单卡配置

官方 `run_alfworld.sh` 写死了 2 卡（`tensor_model_parallel_size=2` + `n_gpus_per_node=2`），
**单卡直接跑会失败**。单卡用：

```bash
bash deploy/run_single_gpu.sh --smoke    # 先冒烟：极小规模验证管线
bash deploy/run_single_gpu.sh            # 正式档
```

改动与理由都写在脚本末尾的注释里。要点：

| 项 | 2 卡 → 1 卡 | 为什么 |
|---|---|---|
| `tensor_model_parallel_size` | 2 → **1** | tp=2 在单卡上是非法配置 |
| `free_cache_engine` | False → **True** | 训练阶段释放 vLLM 的 KV cache，单卡省显存最有效的一项 |
| `gpu_memory_utilization` | 0.6 → **0.35** | 上游按 80G 卡留的余量 |
| micro_batch（3 处） | 32 → **8** | 激活值显存的主要来源 |
| `param_offload` / `optimizer_offload` | False → **True** | Adam 状态是大头（1.5B ≈ 12G） |
| `enable_activation_offload` | — → **True** | 激活值也 offload |

**24G 卡上的可行性**：靠 offload 换显存，属于"勉强能跑"。内存要够
（AutoDL 那台 503G，很宽裕）。若 OOM，依次调小 `gpu_memory_utilization` →
调小 micro_batch → 换 LoRA。

⚠️ **本配置未经实测**（写于无卡模式期间）。首次务必用 `--smoke` 并盯住显存。
实测结果应记入 `exp/report.md`。

**一个有利的副作用**：单卡下 `size_divisor` 从 64 降到 8，而 `adjust_batch` 的复制行数
上限 = `size_divisor - 1`，所以 R1（复制行污染分组统计）的影响反而更小。见 `exp/report.md` 的 E0.2。

## 算力需求

| 项 | 要求 |
|---|---|
| GPU 数 | **≥2**（官方 `run_alfworld.sh` 用 `tensor_model_parallel_size=2`） |
| 模型 | Qwen2.5-1.5B-Instruct（baseline）；7B 需更多显存 |
| 磁盘 | 模型权重 + ALFWorld 资源 + 数据集 |

若只能拿到单卡：把 `tensor_model_parallel_size` 降为 1 并调小 batch，
属额外工作量，且需重新验证 baseline。

## 排障

### ALFWorld 依赖与 vllm 冲突

`gymnasium==0.29.1` / `stable-baselines3==2.6.0` 会拉入自己的 torch/numpy，
可能与 vllm 锁定的版本冲突。上游 README 明确建议**每个环境独立 conda env**。

若 `setup_remote.sh` 在 ALFWorld 步骤导致 torch 被降级，改用独立环境：

```bash
bash deploy/setup_remote.sh --base          # 主环境只装框架
conda create -n verl-agent-alfworld python=3.12 -y
conda activate verl-agent-alfworld
pip install gymnasium==0.29.1 stable-baselines3==2.6.0 alfworld
```

### `alfworld-download` 失败（gdown 限流）

资源托管在 Google Drive，可能限流。按上游提示：访问 `https://drive.google.com/`，
取 cookie 写入 `~/.cache/gdown/cookies.txt`，再重跑；或手动下载。

### HuggingFace 连不上

`examples/data_preprocess/prepare.py` 需要下载 `hiyouga/geometry3k`
（它**只是模态与数据量的占位符**，真实任务数据来自环境）。国内机器可设镜像：

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

### WebShop / Search 需要独立环境

- **WebShop** 要求 Python ≤3.10，必须单独建环境（上游 README 有完整步骤）。
- **Search** 需要另一个 `retriever` 环境（faiss-gpu），且检索服务每张 GPU 占约 6GB 显存。

两者都不在 `setup_remote.sh` 的默认流程里。

## Docker 备选

见 `docker-notes.md`。适合无 root、必须用镜像的集群。
