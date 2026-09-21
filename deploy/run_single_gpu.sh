#!/usr/bin/env bash
# 单卡（24G，如 RTX 4090 / 4090D）版 ALFWorld + GiGPO 启动脚本。
#
# 与上游 `examples/gigpo_trainer/run_alfworld.sh` 的**每一处差异**都标了理由（见下）。
# 上游脚本写死了 2 卡（tp=2 + n_gpus_per_node=2），单卡直接跑会失败。
#
# 用法：
#   bash deploy/run_single_gpu.sh              # 正式档（150 epochs）
#   bash deploy/run_single_gpu.sh --smoke      # 冒烟档：极小规模，验证管线
#
# 注意：**单卡 + 全参微调在 24G 上是勉强能跑**，靠 offload 换显存。若 OOM，
#      依次尝试：调小 gpu_memory_utilization → 调小 micro_batch → 换 LoRA。
#
# ⚠️ 数据边界：本配置**未经实测**（写于无卡模式期间，尚无 GPU 验证）。
#    首次运行请用 --smoke，并盯住显存。实测结果记入 exp/report.md。

set -x

export VLLM_ATTENTION_BACKEND=XFORMERS

# ---- 路径（AutoDL 上系统盘只有 30G，数据与环境都放 WORK_DIR）----
WORK_DIR="${WORK_DIR:-/root/autodl-tmp}"
DATA_DIR="${DATA_DIR:-$WORK_DIR/data/verl-agent}"
REPO_DIR="${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../repo" && pwd)}"

# ---- 参数解析 ----
# 自己的开关（--smoke）与引擎名要挑出来，**不能混进 hydra override**，
# 否则 hydra 会报未知参数；也不能让 --smoke 顶替 ENGINE。
ENGINE=""
SMOKE=0
EXTRA_ARGS=()
for a in "$@"; do
    case "$a" in
        --smoke)    SMOKE=1 ;;
        vllm|sglang) ENGINE="$a" ;;
        *)          EXTRA_ARGS+=("$a") ;;
    esac
done
ENGINE="${ENGINE:-vllm}"

if [[ "$SMOKE" -eq 1 ]]; then
    # 冒烟档：只验证"环境→rollout→优势→训练"全链路能跑通，不看效果。
    # 轨迹短、数据少、步数少 —— 几分钟内应能跑完。
    train_data_size=4
    val_data_size=8
    group_size=4
    total_epochs=2
    test_freq=1
    save_freq=1
    exp_name='smoke_single_gpu'
else
    train_data_size=16
    val_data_size=128
    group_size=8
    total_epochs=150
    test_freq=5
    save_freq=-1
    exp_name='gigpo_qwen2.5_1.5b_single_gpu'
fi

mode="mean_std_norm"

# `python -m examples...` / `python -m verl.trainer...` 都要求从仓库根运行。
cd "$REPO_DIR" || { echo "未找到 repo 目录：$REPO_DIR" >&2; exit 1; }

# 数据准备（占位数据集，仅指示模态与规模）。--local_dir 必须与 DATA_DIR 一致。
python3 -m examples.data_preprocess.prepare \
    --mode 'text' \
    --local_dir "$DATA_DIR" \
    --train_data_size $train_data_size \
    --val_data_size $val_data_size

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=gigpo \
    data.train_files=$DATA_DIR/text/train.parquet \
    data.val_files=$DATA_DIR/text/test.parquet \
    data.train_batch_size=$train_data_size \
    data.val_batch_size=$val_data_size \
    data.max_prompt_length=2048 \
    data.max_response_length=512 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path=Qwen/Qwen2.5-1.5B-Instruct \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=256 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.model.enable_activation_offload=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=$ENGINE \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.35 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.4 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.use_invalid_action_penalty=True \
    actor_rollout_ref.actor.invalid_action_penalty_coef=0.1 \
    algorithm.use_kl_in_reward=False \
    algorithm.gamma=0.95 \
    algorithm.gigpo.step_advantage_w=1.0 \
    algorithm.gigpo.mode=$mode \
    env.env_name=alfworld/AlfredTWEnv \
    env.seed=0 \
    env.max_steps=50 \
    env.rollout.n=$group_size \
    env.resources_per_worker.num_cpus=0.1 \
    trainer.critic_warmup=0 \
    trainer.logger=['console'] \
    trainer.project_name='verl_agent_alfworld' \
    trainer.experiment_name=$exp_name \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    trainer.save_freq=$save_freq \
    trainer.test_freq=$test_freq \
    trainer.total_epochs=$total_epochs \
    trainer.val_before_train=True ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}

# ============================================================================
# 与上游 run_alfworld.sh 的差异汇总（每一项都为单卡 24G 服务）
# ----------------------------------------------------------------------------
#  trainer.n_gpus_per_node            2 → 1    单卡
#  rollout.tensor_model_parallel_size 2 → 1    tp=2 在单卡上是非法配置，必须改
#  rollout.free_cache_engine       False → True 上游关了它（yaml 默认是 True）。
#                                              开着才能在训练阶段释放 vLLM 的 KV cache，
#                                              这是单卡省显存最有效的一项。
#  rollout.gpu_memory_utilization   0.6 → 0.35 上游按 80G 卡留的余量；24G 上必须压下来，
#                                              否则 vLLM 会把训练侧的显存吃掉。
#  ppo_micro_batch_size_per_gpu      32 → 8    激活值显存的主要来源
#  log_prob_micro_batch_size_per_gpu 32 → 8    同上（rollout 与 ref 各一处）
#  actor.fsdp_config.param_offload       False → True   参数offload到CPU内存
#  actor.fsdp_config.optimizer_offload   False → True   Adam 状态是大头（1.5B ≈ 12G），
#                                                       offload 掉是单卡能跑起来的关键。
#                                                       AutoDL 这台有 503G 内存，很宽裕。
#  model.enable_activation_offload       新增 True      激活值也 offload
#  trainer.logger              ['console','wandb'] → ['console']
#                                              冒烟阶段不配 wandb，避免登录阻塞
# ----------------------------------------------------------------------------
# 一个副作用：size_divisor 从
#     lcm(32*2, 32*2, 32*2) = 64   变成   lcm(8*1, 8*1, 8*1) = 8
# 而 adjust_batch 的复制行数上限 = size_divisor - 1，所以单卡配置下
# R1（复制行污染分组统计）的影响反而更小。见 exp/report.md 的 E0.2。
# ============================================================================
