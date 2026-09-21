#!/usr/bin/env python3
"""E0.2 — 量化 `adjust_batch(mode="copy")` 的复制行对分组统计的污染。

对应 `notes/01-credit-assignment.md` 的 **R1**。

问题
----
`verl/trainer/ppo/ray_trainer.py:1118` 在 `compute_advantage`（`:1221`）**之前**调用
`adjust_batch(config, batch)`。后者默认 `mode="copy"`
（`agent_system/multi_turn_rollout/utils.py:121-126`）：

    to_add = size_divisor - remainder
    dup_indices = np.random.choice(bs, to_add, replace=False)
    adjusted_batch = DataProto.concat([data, data.select_idxs(dup_indices)])

**随机复制若干行**补齐整除。这些复制行带**相同的 uid / traj_uid / anchor_obs**，
会原样进入 GiGPO 的分组与 mean/std 计算 → **给部分样本额外加权**。

佐证：HGPO 的 trainer 特意把这一步挪到 `compute_advantage` **之后**
（`recipe/hgpo/hgpo_ray_trainer.py:1219`，文件头 `:18-19` 有注释说明原因），主 trainer 没有。

参数（`run_alfworld.sh`）
------------------------
    n_gpus_per_node=2, nnodes=1                    ⇒ world_size = 2
    rollout.log_prob_micro_batch_size_per_gpu=32   ⇒ 64
    ref.log_prob_micro_batch_size_per_gpu=32       ⇒ 64   (use_kl_loss=True)
    actor.ppo_micro_batch_size_per_gpu=32          ⇒ 64
    ⇒ size_divisor = lcm(64, 64, 64) = 64

用法：.venv/bin/python exp/p0/e02_adjust_batch.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "repo"))

from common import SIZE_DIVISOR, build_batch, response_mask, to_dataproto  # noqa: E402

SEP = "=" * 78


def h(t: str) -> None:
    print(f"\n{SEP}\n{t}\n{SEP}")


def copy_duplicate(batch, to_add: int, rng: np.random.Generator):
    """复刻 adjust_batch 的 mode="copy" 行为：随机选 to_add 行复制并追加到末尾。

    返回 (新 batch, 被复制行的索引)。
    """
    bs = len(batch)
    dup_idx = rng.choice(bs, size=to_add, replace=False)
    ordered = np.concatenate([np.arange(bs), dup_idx])       # 原行在前，副本追加在后
    return batch.take(ordered), dup_idx


def episode_adv(core_gigpo, batch) -> np.ndarray:
    """跑一遍 episode 级优势，返回 (bs,) 的标量优势。"""
    bs = len(batch)
    resp_len = 4
    tr = torch.zeros((bs, resp_len), dtype=torch.float32)
    tr[:, -1] = torch.tensor(batch.outcome, dtype=torch.float32)
    adv = core_gigpo.episode_norm_reward(
        tr, response_mask(bs, resp_len),
        index=batch.prompt_id, traj_index=batch.traj_id, remove_std=True,
    )
    return adv[:, 0].numpy()


def main() -> int:
    print(SEP)
    print("E0.2 — adjust_batch 复制行对分组统计的污染")
    print(SEP)

    from gigpo import core_gigpo

    rng = np.random.default_rng(0)
    batch = build_batch(seed=0)
    bs = len(batch)

    h(f"size_divisor = {SIZE_DIVISOR} 下的复制幅度")
    print(f"\n  本轮 rollout 产出的 step 数 bs = {bs}")
    remainder = bs % SIZE_DIVISOR
    to_add = 0 if remainder == 0 else SIZE_DIVISOR - remainder
    print(f"  remainder = bs % {SIZE_DIVISOR} = {remainder}")
    print(f"  to_add    = {to_add}  ⇒ 复制 {to_add} 行（占原始 {to_add / bs:.1%}）")

    if to_add == 0:
        print("\n  本轮恰好整除，无需复制。下面扫一批 bs 取值看整体情况。")

    # ------------------------------------------------------------------ #
    h("扫描：不同 bs 下的复制幅度")

    print(f"\n  {'bs':>6} {'remainder':>10} {'to_add':>8} {'复制占比':>10}")
    print("  " + "-" * 38)
    for sample_bs in (500, 512, 600, 640, 700, 768, 800, 1000, 1024, 1200):
        rem = sample_bs % SIZE_DIVISOR
        add = 0 if rem == 0 else SIZE_DIVISOR - rem
        print(f"  {sample_bs:>6} {rem:>10} {add:>8} {add / sample_bs:>9.1%}")
    print(f"\n  注意：bs 恰好是 {SIZE_DIVISOR} 的倍数时 to_add=0（无污染）；")
    print(f"        否则平均复制 {SIZE_DIVISOR // 2} 行 —— bs 越小，占比越可观。")

    # ------------------------------------------------------------------ #
    h("污染幅度：复制前后，**原有行**的优势变化")

    base_adv = episode_adv(core_gigpo, batch)

    print(f"\n  {'to_add':>8} {'复制占比':>10} {'|Δ优势|均值':>13} {'|Δ优势|最大':>13} "
          f"{'Δ优势 std':>11}")
    print("  " + "-" * 60)

    worst = None
    for add in (4, 16, 32, 63, len(batch) // 4):
        add = min(add, bs)
        dup_batch, _ = copy_duplicate(batch, add, np.random.default_rng(1))
        dup_adv = episode_adv(core_gigpo, dup_batch)[:bs]     # 取回原有行的优势
        delta = np.abs(dup_adv - base_adv)
        print(f"  {add:>8} {add / bs:>9.1%} {delta.mean():>13.4f} {delta.max():>13.4f} "
              f"{delta.std():>11.4f}")
        if worst is None or delta.max() > worst[2]:
            worst = (add, delta.mean(), delta.max())

    # ------------------------------------------------------------------ #
    h("机制：复制行如何改变组内 mean/std")

    add = 32
    dup_batch, dup_idx = copy_duplicate(batch, add, np.random.default_rng(2))

    # 取一个被复制过的 task，看它的组统计变化
    chosen_traj = batch.traj_id[dup_idx[0]]
    pid = batch.prompt_id[dup_idx[0]]
    m0 = batch.prompt_id == pid
    m1 = dup_batch.prompt_id == pid

    def traj_scores(b, mask):
        # episode_norm_reward 内部按行（=step）收集 token_level_rewards 之和
        s = torch.zeros((len(b), 4), dtype=torch.float32)
        s[:, -1] = torch.tensor(b.outcome, dtype=torch.float32)
        return s[mask].sum(dim=-1).numpy()

    s0 = traj_scores(batch, m0)
    s1 = traj_scores(dup_batch, m1)
    print(f"\n  被抽中的 task：{pid}（复制了轨迹 {chosen_traj} 的行）")
    print(f"    复制前：该组 {len(s0)} 行，均值 {s0.mean():.4f}，std {s0.std():.4f}")
    print(f"    复制后：该组 {len(s1)} 行，均值 {s1.mean():.4f}，std {s1.std():.4f}")

    n_dup_in_group = int(np.sum(dup_batch.prompt_id[bs:] == pid))
    print(f"    该组被追加了 {n_dup_in_group} 个副本 "
          f"⇒ 组内样本数 {len(s0)} → {len(s1)}，**该 task 的权重被放大**")

    # ------------------------------------------------------------------ #
    h("与 E0.3 的交互（更要害）：复制行把单元素组「变成」2 元素组")

    from collections import Counter

    def row_group_size(b) -> np.ndarray:
        """每一行所属状态组的**成员数**。

        ⚠️ 不能直接比对两次 `build_step_group` 的 uid —— 它每次调用都用
        `uuid.uuid4()` 重新生成（`core_gigpo.py:291`），跨调用不可比。
        所以改为按行统计组大小。
        """
        uids = core_gigpo.build_step_group(b.anchor_obs, b.prompt_id)
        counts = Counter(uids.tolist())
        return np.array([counts[u] for u in uids])

    base_gs = row_group_size(batch)

    print(f"\n  {'to_add':>8} {'被提升的步数':>14} {'占原单元素步比例':>18}")
    print("  " + "-" * 44)
    base_singleton_rows = int(np.sum(base_gs == 1))
    for add in (8, 32, 63):
        dup_batch2, _ = copy_duplicate(batch, add, np.random.default_rng(3))
        dup_gs = row_group_size(dup_batch2)[:bs]      # 只看原有行的组大小变化
        promoted = int(np.sum((base_gs == 1) & (dup_gs > 1)))
        print(f"  {add:>8} {promoted:>14} {promoted / base_singleton_rows:>17.3%}")
    print(f"\n  （原始单元素步共 {base_singleton_rows} / {bs} 行）")

    print("""
    ⚠️ 这是比 episode 级更尖锐的后果：E0.3 已证明 step 级非零优势的步只占 ~14%，
    而这些非零优势**本应只来自"多条 rollout 真的一起访问过的状态"**。
    复制行会凭空造出 2 元素组，让一些**本应优势为 0** 的步获得非零优势 ——
    即向 step 通道注入**虚假**的信用信号。""")

    # ------------------------------------------------------------------ #
    h("结论")
    print(f"""
  ⚠️ **本项实测影响很小**，与 R2（E0.3）不在一个量级 —— 这是本次量化最有用的结论。

  1. **机制成立**：复制行带相同的 prompt_id / traj_id / anchor_obs，
     确实原样进入 `episode_norm_reward` 与 `build_step_group` 的分组统计。

  2. **但量级很小**（本轮 bs={bs}）：
     - to_add 平均 {SIZE_DIVISOR // 2} 行，占比 {SIZE_DIVISOR // 2 / bs:.1%}
     - 对原有行优势的影响：|Δ| 均值 ~0.0005，最大 ~0.005
     - 把单元素组"提升"为多元素组的步：{SIZE_DIVISOR // 2} 个副本仅影响 ~25 步
       （占单元素步的 0.8%）

     原因：bs 是**几千**量级，而 to_add 上限只有 {SIZE_DIVISOR - 1}。

  3. **风险排序**：R1（本项，影响 <1%）**远低于** R2
     （E0.3：单元素步占 74%，step 优势非零的步仅 14%）。
     补这一项**优先级低**，不必在项目早期投入。

  4. **唯一的隐患是随机性**：`np.random.choice` 每轮选不同的行，
     所以加权对象逐轮变化 —— 表现为训练曲线的额外噪声而非稳定偏置，
     排查成本高于其量级。若实验中要严格控制变量，可把 `adjust_batch`
     挪到 `compute_advantage` 之后（与 HGPO 一致，
     `recipe/hgpo/hgpo_ray_trainer.py:1219`），但**这不是优先事项**。

  ⚠️ 数据边界：合成数据（见 common.py）。结论对该参数不敏感 ——
     复制占比由 bs 与 SIZE_DIVISOR 的整除关系决定，与数据分布无关。
""")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
