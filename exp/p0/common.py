"""P0 实验的公共构件：构造贴近真实 ALFWorld + GiGPO 的合成 batch。

⚠️ **数据来源声明**：ALFWorld 真实数据需 `ALFWORLD_DATA`（数 GB，Google Drive），
本机没有。所以下面构造的是**按真实结构建模的合成数据**，不是实测数据。
结构参数（批大小、组大小、轨迹长度分布、成功率）都来自仓库里的真实配置与代码，
并全部暴露为可调参数，拿到真实数据后可直接替换 `build_batch` 重跑。

真实配置依据（`examples/gigpo_trainer/run_alfworld.sh`）：
    data.train_batch_size=16        每轮 16 个 task
    env.rollout.n=8                 每个 task 8 条 rollout（GRPO/GiGPO 的组大小）
    env.max_steps=50                单条轨迹上限
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=32
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=32
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=32
    trainer.n_gpus_per_node=2, nnodes=1
⇒ world_size = 2 ⇒ size_divisor = lcm(64, 64, 64) = 64
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# 来自 run_alfworld.sh 的真实取值
N_PROMPTS = 16
N_ROLLOUTS = 8
MAX_STEPS = 50
WORLD_SIZE = 2
MICRO_BATCH = 32
SIZE_DIVISOR = 64  # lcm(32*2, 32*2, 32*2)


@dataclass
class Batch:
    """扁平的 step-level batch —— 一行 = 一个 agent step 的一次 LLM 调用。

    这是 GiGPO 的数据布局（见 notes/01-credit-assignment.md §1.1）。
    """

    prompt_id: np.ndarray      # (bs,) 同一 task 的 n 条 rollout 共享 → GiGPO 的 episode 组
    traj_id: np.ndarray        # (bs,) 每条 rollout 一个 id
    step_idx: np.ndarray       # (bs,) 该步在轨迹内的序号
    outcome: np.ndarray        # (bs,) 该轨迹的最终结果奖励（稀疏，仅末步非零）
    anchor_obs: np.ndarray     # (bs,) 该步开始时的原始观测 → GiGPO 的 step 分组键
    traj_len: np.ndarray       # (bs,) 所属轨迹的总步数

    def __len__(self) -> int:
        return len(self.outcome)

    def take(self, idx: np.ndarray) -> "Batch":
        """按索引取子集（用于对比复制前后）。"""
        return Batch(**{f: getattr(self, f)[idx] for f in self.__dataclass_fields__})


def _room_obs(seed_rng: np.random.Generator, room: int, step: int) -> str:
    """生成一条贴近 ALFWorld 真实格式的观测文本。

    ALFWorld 的 anchor 观测是环境返回的**原始**房间描述（未经 prompt 模板包装），
    形如："You are in the middle of a room. Looking quickly around you, you see ..."
    """
    rooms = ["kitchen", "bedroom", "bathroom", "living room", "garden"]
    items = ["bed", "desk", "drawer", "garbage can", "shelf", "sofa",
             "countertop", "fridge", "microwave", "sink", "toilet", "cabinet"]
    k = seed_rng.integers(3, 7)
    picks = seed_rng.choice(items, size=k, replace=False)
    listing = ", ".join(f"a {p} {i + 1}" for i, p in enumerate(picks))
    return (f"You are in the middle of a {rooms[room % len(rooms)]}. "
            f"Looking quickly around you, you see {listing}.")


def build_batch(
    n_prompts: int = N_PROMPTS,
    n_rollouts: int = N_ROLLOUTS,
    max_steps: int = MAX_STEPS,
    success_rate: float = 0.5,
    len_success: tuple[int, int] = (5, 30),
    len_fail: tuple[int, int] = (40, MAX_STEPS),
    share_prob: float = 0.15,
    seed: int = 0,
) -> Batch:
    """构造一轮 rollout 后的扁平 step-level batch。

    轨迹长度建模依据：ALFWorld 里成功的 episode 会提前 `done`（短），
    失败的会一直撞到 `max_steps`（长）——这是长轨迹信用分配问题的来源。

    `share_prob` 控制**状态共享程度**，直接决定 GiGPO 的 step 分组有没有东西可分：
        同一 task 的 n 条 rollout，在第 s 步落在"别的 rollout 已访问过的状态"上的概率。
        - s=0 恒为 1（所有 rollout 从同一初始观测出发 —— ALFWorld 的真实行为）
        - s>0 时，动作选择不同 → 很快分道扬镳。真实取值需在 ALFWorld 上实测，
          这里作为可调参数暴露出来做敏感性分析。

    ⚠️ 默认值 0.15 是**估计值**，不是实测值。E0.3 会给出它对结论的影响曲线。
    """
    rng = np.random.default_rng(seed)

    prompt_id, traj_id, step_idx, outcome, anchor, traj_len = [], [], [], [], [], []

    for p in range(n_prompts):
        init_room = _room_obs(rng, p, 0)
        # 每个 task 维护"该步已出现过的状态"，用于模拟路径合并
        seen_states: dict[int, list[str]] = {}

        for r in range(n_rollouts):
            tid = f"t{p}_{r}"
            won = rng.random() < success_rate
            lo, hi = len_success if won else len_fail
            L = int(rng.integers(lo, hi + 1))

            for s in range(L):
                prompt_id.append(f"p{p}")
                traj_id.append(tid)
                step_idx.append(s)
                # 稀疏结果奖励：只有最后一步非零（ALFWorld 真实行为）
                outcome.append(1.0 if (won and s == L - 1) else 0.0)

                pool = seen_states.setdefault(s, [])
                if s == 0:
                    obs = init_room                      # 所有 rollout 共享初始状态
                    if not pool:
                        pool.append(obs)
                elif pool and rng.random() < share_prob:
                    obs = pool[int(rng.integers(0, len(pool)))]   # 与别的 rollout 撞上同一状态
                else:
                    obs = _room_obs(rng, p * 7 + r, s)
                    pool.append(obs)
                anchor.append(obs)
                traj_len.append(L)

    return Batch(
        prompt_id=np.array(prompt_id, dtype=object),
        traj_id=np.array(traj_id, dtype=object),
        step_idx=np.array(step_idx),
        outcome=np.array(outcome, dtype=np.float32),
        anchor_obs=np.array(anchor, dtype=object),
        traj_len=np.array(traj_len),
    )


def to_dataproto(batch: Batch):
    """转成 GiGPO 的 `compute_step_discounted_returns` 需要的 DataProto。

    该函数读 non_tensor_batch 的 rewards / traj_uid / active_masks，
    以及 batch['input_ids']（仅用于取 device）。
    """
    import torch
    from tensordict import TensorDict
    from verl import DataProto

    bs = len(batch)
    return DataProto(
        batch=TensorDict(
            {"input_ids": torch.zeros((bs, 4), dtype=torch.long)}, batch_size=[bs]
        ),
        non_tensor_batch={
            # 该函数期望的 'rewards' 是**环境每步**奖励，即这里的 outcome 列
            "rewards": batch.outcome.astype(np.float32),
            "traj_uid": batch.traj_id,
            "active_masks": np.ones(bs, dtype=np.float32),
        },
    )


def response_mask(bs: int, resp_len: int = 4):
    import torch
    return torch.ones((bs, resp_len), dtype=torch.float32)


def summarize(name: str, values: np.ndarray, fmt: str = "{:.4f}") -> None:
    v = np.asarray(values, dtype=float)
    print(f"  {name:<34} min={fmt.format(v.min())}  "
          f"中位={fmt.format(np.median(v))}  max={fmt.format(v.max())}  "
          f"均值={fmt.format(v.mean())}")
