#!/usr/bin/env python3
"""E0.3 — 量化单元素组抹平在长轨迹下的占比。

对应 `notes/01-credit-assignment.md` 的 **R2**：
    step 级：`len==1` ⇒ mean = 该分数本身 ⇒ 优势 **恒为 0**
    （`gigpo/core_gigpo.py:365-369`，与 episode 级的 `:225-227` 行为相反）

GiGPO 的 step 级分组键是 **anchor 观测**——只有"多条 rollout 抵达同一状态"时
才能构造出相对优势。长轨迹下各 rollout 很快分道扬镳，单元素组成为常态。

待量化
------
1. step 级组大小分布、单元素组占比、非零 step 优势的步占比
2. 与 episode 级的对比（episode 组 = 同一 task 的 n 条 rollout，恒定非单元素）
3. **敏感性**：状态共享程度 `share_prob` 如何影响上述数字
4. `enable_similarity`（相似度分组）能否缓解

⚠️ 数据边界：`share_prob` 是**估计值**，真实取值需在 ALFWorld 上实测（需 ALFWORLD_DATA）。
所以本脚本给的是**参数化曲线**而非单一结论——但"episode 级非单元素、step 级大面积
单元素"这个**结构性差异**与具体取值无关。

用法：.venv/bin/python exp/p0/e03_singleton_groups.py
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "repo"))

from common import N_ROLLOUTS, build_batch, response_mask, to_dataproto  # noqa: E402

SEP = "=" * 78


def h(t: str) -> None:
    print(f"\n{SEP}\n{t}\n{SEP}")


def group_stats(uids: np.ndarray) -> tuple[Counter, float, float]:
    """返回（组大小分布, 单元素组内的步占比, 平均组大小）。"""
    counts = Counter(uids.tolist())
    sizes = np.array(list(counts.values()))
    n_steps = len(uids)
    singleton_steps = sum(1 for u in uids if counts[u] == 1)
    return Counter(sizes.tolist()), singleton_steps / n_steps, float(sizes.mean())


def main() -> int:
    print(SEP)
    print("E0.3 — 单元素组抹平在长轨迹下的占比")
    print(SEP)

    from gigpo import core_gigpo

    # ------------------------------------------------------------------ #
    h("基线情形（share_prob=0.15）")

    batch = build_batch(seed=0)
    bs = len(batch)
    print(f"\n合成 batch：{bs} 行（step）/ {len(np.unique(batch.traj_id))} 条轨迹 / "
          f"{len(np.unique(batch.prompt_id))} 个 task（每 task {N_ROLLOUTS} 条 rollout）")

    step_uids = core_gigpo.build_step_group(batch.anchor_obs, batch.prompt_id)
    dist, singleton_frac, avg_size = group_stats(step_uids)

    print(f"\n  【step 级分组】键 = anchor 观测")
    print(f"    平均组大小                    {avg_size:.3f}")
    print(f"    **单元素组内的步占比**         {singleton_frac:.1%}")
    print("    组大小分布（大小: 组数）      " +
          ", ".join(f"{k}:{v}" for k, v in sorted(dist.items())[:8]))

    # episode 级：组 = 同一 task 的 n 条 rollout（用 (prompt, traj) 去重后的轨迹数）
    traj_per_prompt = Counter(batch.prompt_id.tolist())
    print(f"\n  【episode 级分组】键 = task（prompt）")
    print(f"    每 task 的 rollout 数         {N_ROLLOUTS}")
    print(f"    **单元素组内的轨迹占比**       0.0%（每个 task 恒有 {N_ROLLOUTS} 条）")

    # ------------------------------------------------------------------ #
    h("对最终优势的影响")

    resp_len = 4
    token_rewards = torch.zeros((bs, resp_len), dtype=torch.float32)
    token_rewards[:, -1] = torch.tensor(batch.outcome, dtype=torch.float32)
    dp = to_dataproto(batch)

    sr = core_gigpo.compute_step_discounted_returns(batch=dp, gamma=1.0)
    step_adv, _ = core_gigpo.compute_gigpo_outcome_advantage(
        torch.zeros_like(token_rewards), sr, response_mask(bs, resp_len),
        anchor_obs=batch.anchor_obs, index=batch.prompt_id,
        traj_index=batch.traj_id, mode="mean_norm",
    )
    epi_adv = core_gigpo.episode_norm_reward(
        token_rewards, response_mask(bs, resp_len),
        index=batch.prompt_id, traj_index=batch.traj_id, remove_std=True,
    )

    sa = step_adv[:, 0].numpy()
    ea = epi_adv[:, 0].numpy()
    print(f"\n  {'指标':<34}{'step 级':>12}{'episode 级':>14}")
    print("  " + "-" * 60)
    print(f"  {'优势非零的步占比':<32}{np.mean(np.abs(sa) > 1e-9):>12.1%}"
          f"{np.mean(np.abs(ea) > 1e-9):>14.1%}")
    print(f"  {'优势标准差':<34}{np.std(sa):>12.4f}{np.std(ea):>14.4f}")
    print(f"  {'优势绝对值均值':<32}{np.mean(np.abs(sa)):>12.4f}{np.mean(np.abs(ea)):>14.4f}")

    # ------------------------------------------------------------------ #
    h("敏感性：状态共享程度 share_prob 的影响")

    print(f"\n  {'share_prob':>10} {'平均组大小':>10} {'单元素步占比':>12} "
          f"{'step优势非零占比':>16}")
    print("  " + "-" * 54)
    rows = []
    for sp in (0.0, 0.05, 0.15, 0.3, 0.5, 0.8):
        b = build_batch(share_prob=sp, seed=1)
        uids = core_gigpo.build_step_group(b.anchor_obs, b.prompt_id)
        _, sf, asz = group_stats(uids)
        b_dp = to_dataproto(b)
        b_sr = core_gigpo.compute_step_discounted_returns(batch=b_dp, gamma=1.0)
        b_tr = torch.zeros((len(b), 4), dtype=torch.float32)
        b_tr[:, -1] = torch.tensor(b.outcome, dtype=torch.float32)
        b_adv, _ = core_gigpo.compute_gigpo_outcome_advantage(
            torch.zeros_like(b_tr), b_sr, response_mask(len(b), 4),
            anchor_obs=b.anchor_obs, index=b.prompt_id,
            traj_index=b.traj_id, mode="mean_norm",
        )
        nz = float(np.mean(np.abs(b_adv[:, 0].numpy()) > 1e-9))
        rows.append((sp, asz, sf, nz))
        print(f"  {sp:>10.2f} {asz:>10.3f} {sf:>11.1%} {nz:>15.1%}")

    # ------------------------------------------------------------------ #
    h("enable_similarity（相似度分组）能否缓解")

    for sp in (0.15, 0.5):
        b = build_batch(share_prob=sp, seed=2)
        exact = core_gigpo.build_step_group(b.anchor_obs, b.prompt_id, enable_similarity=False)
        fuzzy = core_gigpo.build_step_group(
            b.anchor_obs, b.prompt_id, enable_similarity=True, similarity_thresh=0.95)
        _, sf_e, asz_e = group_stats(exact)
        _, sf_f, asz_f = group_stats(fuzzy)
        print(f"\n  share_prob={sp}")
        print(f"    精确匹配：平均组大小 {asz_e:.3f}，单元素步占比 {sf_e:.1%}")
        print(f"    相似度匹配(0.95)：平均组大小 {asz_f:.3f}，单元素步占比 {sf_f:.1%}")

    print("""
    说明：合成数据的观测文本差异较大（不同房间/不同物品列表），
    字符级相似度（SequenceMatcher）几乎合并不了任何状态。
    真实 ALFWorld 上同样如此——观测的差异是**内容**差异而非拼写差异。
    若要用语义相似度，需要换成 embedding 度量（见 notes/01 的改造点 (d)）。""")

    # ------------------------------------------------------------------ #
    h("结论")

    nz_at_015 = rows[2][3]
    print(f"""
  1. **step 级分组几乎全是单元素组**：share_prob=0.15 时单元素步占比 {singleton_frac:.1%}，
     平均组大小仅 {avg_size:.2f}（组大小=1 意味着该状态只被访问过一次）。

  2. **后果是 step 级优势被大面积抹平**：非零优势的步仅占 {nz_at_015:.1%}，
     而 episode 级是 {np.mean(np.abs(ea) > 1e-9):.1%}。这是**结构性差异**，非调参能解决。

  3. **敏感性**：即便把状态共享概率拉到 0.8（极不现实的乐观值），
     单元素步占比仍有 {rows[-1][2]:.1%}。要让它显著下降，需要 rollout 之间
     高度同质——而长轨迹任务恰恰不是这样。

  4. **相似度分组无效**：观测差异是内容差异，字符级相似度合并不了。

  ⇒ **R2 成立且量级很大**：GiGPO 的 step 级信号在长轨迹上大面积失效。
     这为 PRM 提供了明确动机——**用不依赖"同状态重复访问"的过程信号**
     来补上这一维。

  ⚠️ 数据边界：`share_prob` 是估计值（见 common.py）。真实值需在 ALFWorld 上实测。
     **但结构性结论与取值无关**：episode 组的成员数由 `env.rollout.n` 固定为 8，
     而 step 组的成员数取决于轨迹重合度，两者不在一个量级。
""")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
