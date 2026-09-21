#!/usr/bin/env python3
"""E0.4 — 量化 `gamma` 对 step 级信号信息量的影响。

对应 `exp/design.md` §2 的立论：**`gamma=1.0` 下 GiGPO 的 "step-level" 是形式上的。**

背景
----
`gigpo/core_gigpo.py:87-132` 的 `compute_step_discounted_returns` 做的是：
    running_return = r[t] + gamma * running_return        （沿轨迹逆序）
而各环境的每步奖励几乎全是 0、只有终止步非零（已在 notes/03-prm.md §2.1 逐个环境核实）。

于是：
    gamma = 1.0  ⇒  step_rewards[t] ≡ R（轨迹最终结果），与 t 无关
    gamma < 1.0  ⇒  step_rewards[t] = R * gamma^(L-1-t)，编码"距终点还有多远"

**待验证的推论**：gamma=1.0 时 step 级信号不含任何位置信息，step 级优势只是
"把 outcome 在访问过同一状态的轨迹子集里重新归一化"。

判据（三条，逐条可证伪）
------------------------
1. 轨迹内 `step_rewards` 的标准差：gamma=1.0 应为 **0**，gamma=0.95 应 > 0
2. `step_rewards` 与"距终点步数"的相关性：gamma=1.0 应为 **未定义/0**（无变化）
3. step 级优势与 episode 级优势的关系：gamma=1.0 时应高度重合（因为二者都只是
   outcome 的归一化，只是分组不同）

用法：.venv/bin/python exp/p0/e04_gamma_information.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "repo"))

from common import build_batch, response_mask, summarize, to_dataproto  # noqa: E402

SEP = "=" * 78


def h(t: str) -> None:
    print(f"\n{SEP}\n{t}\n{SEP}")


def within_traj_std(step_rewards: np.ndarray, traj_id: np.ndarray) -> np.ndarray:
    """每条轨迹内部 step_rewards 的标准差。这是"是否含位置信息"的直接度量。"""
    out = []
    for t in np.unique(traj_id):
        v = step_rewards[traj_id == t]
        out.append(float(np.std(v)))
    return np.array(out)


def main() -> int:
    print(SEP)
    print("E0.4 — gamma 对 step 级信号信息量的影响")
    print(SEP)

    from gigpo import core_gigpo

    batch = build_batch(seed=0)
    bs = len(batch)
    n_traj = len(np.unique(batch.traj_id))
    print(f"\n合成 batch：{bs} 行（step）/ {n_traj} 条轨迹 / "
          f"{len(np.unique(batch.prompt_id))} 个 task")
    print(f"轨迹长度：min={batch.traj_len.min()} max={batch.traj_len.max()}  "
          f"均值={batch.traj_len.mean():.1f}")
    print(f"成功率（outcome 非零的轨迹占比）："
          f"{np.mean([batch.outcome[batch.traj_id == t].sum() > 0 for t in np.unique(batch.traj_id)]):.1%}")

    dp = to_dataproto(batch)

    # ------------------------------------------------------------------ #
    h("判据 1：轨迹内 step_rewards 的标准差")
    results = {}
    for gamma in (1.0, 0.95, 0.9):
        sr = core_gigpo.compute_step_discounted_returns(batch=dp, gamma=gamma).numpy()
        stds = within_traj_std(sr, batch.traj_id)
        frac_zero = float(np.mean(stds < 1e-9))
        results[gamma] = sr
        print(f"\n  gamma={gamma}")
        summarize("轨迹内 step_rewards 标准差", stds)
        print(f"  {'标准差为 0 的轨迹占比':<34} {frac_zero:.1%}")
        if frac_zero > 0.999:
            print("  ⇒ 所有轨迹内 step_rewards 恒定，**不含任何位置信息** ✅ 立论成立")
        else:
            print("  ⇒ step_rewards 随步变化，编码了位置/时间信息")

    # ------------------------------------------------------------------ #
    h("判据 2：位置信息对 step_rewards 的**额外**解释力（R² 分解）")

    # 不能用跨轨迹的相关系数——那会被不同轨迹的 outcome 差异主导。
    # 正确问法是：在已知 outcome 的前提下，再加"距终点步数"能否多解释一点方差？
    #   模型 A:  step_rewards ~ outcome
    #   模型 B:  step_rewards ~ outcome + 距终点步数
    # gamma=1.0 时 A 已完美拟合，B 的增量 R² 应为 0。
    remaining = (batch.traj_len - 1 - batch.step_idx).astype(np.float64)

    # ⚠️ 自变量必须是**轨迹级** outcome（整条轨迹的最终结果，对每个 step 都一样），
    # 而不是 batch.outcome —— 后者是**每步**奖励，只有末步非零，两者不是一回事。
    traj_outcome_map = {
        t: float(batch.outcome[batch.traj_id == t].sum()) for t in np.unique(batch.traj_id)
    }
    outcome_col = np.array([traj_outcome_map[t] for t in batch.traj_id], dtype=np.float64)

    def r2(y: np.ndarray, *cols: np.ndarray) -> float:
        X = np.column_stack([np.ones_like(y), *cols])
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        resid = y - X @ beta
        ss_tot = np.sum((y - y.mean()) ** 2)
        return 1.0 - np.sum(resid**2) / ss_tot if ss_tot > 1e-12 else float("nan")

    for gamma in (1.0, 0.95):
        sr = results[gamma].astype(np.float64)
        r2_a = r2(sr, outcome_col)
        r2_b = r2(sr, outcome_col, remaining)
        print(f"\n  gamma={gamma}")
        print(f"    R²(step_rewards ~ outcome)                = {r2_a:.6f}")
        print(f"    R²(step_rewards ~ outcome + 距终点步数)     = {r2_b:.6f}")
        print(f"    位置带来的**增量** R²                       = {r2_b - r2_a:+.6f}")
        if r2_b - r2_a < 1e-9:
            print("    ⇒ 位置信息**零增量**：step_rewards 完全由 outcome 决定 ✅ 立论成立")
        else:
            print("    ⇒ 位置确实带来额外信息（「离终点还有多远」），但那是位置而非过程质量")

    # ------------------------------------------------------------------ #
    h("判据 3：step 级优势 vs episode 级优势（信息是否重合）")

    # episode 级优势走 episode_norm_reward（需要 token 级奖励，末尾 token 非零）
    resp_len = 4
    token_rewards = torch.zeros((bs, resp_len), dtype=torch.float32)
    token_rewards[:, -1] = torch.tensor(batch.outcome, dtype=torch.float32)

    for gamma in (1.0, 0.95):
        sr = torch.tensor(results[gamma], dtype=torch.float32)
        scores, _ = core_gigpo.compute_gigpo_outcome_advantage(
            token_rewards, sr, response_mask(bs, resp_len),
            anchor_obs=batch.anchor_obs, index=batch.prompt_id,
            traj_index=batch.traj_id, mode="mean_norm",
        )
        epi = core_gigpo.episode_norm_reward(
            token_rewards, response_mask(bs, resp_len),
            index=batch.prompt_id, traj_index=batch.traj_id, remove_std=True,
        )
        total = scores[:, 0].numpy()
        e = epi[:, 0].numpy()
        corr = float(np.corrcoef(total, e)[0, 1])

        # step 项自身：与 outcome 的关系
        step_only, _ = core_gigpo.compute_gigpo_outcome_advantage(
            torch.zeros_like(token_rewards), sr, response_mask(bs, resp_len),
            anchor_obs=batch.anchor_obs, index=batch.prompt_id,
            traj_index=batch.traj_id, mode="mean_norm",
        )
        so = step_only[:, 0].numpy()
        print(f"\n  gamma={gamma}")
        print(f"    corr(总优势, episode 优势)        = {corr:+.4f}")
        print(f"    step 优势非零的步占比             = {np.mean(np.abs(so) > 1e-9):.1%}")
        print(f"    step 优势的标准差                 = {np.std(so):.4f}")

        if gamma == 1.0:
            # 按轨迹看：step 优势是否只是 outcome 的函数
            per_traj_step, per_traj_out = [], []
            for t in np.unique(batch.traj_id):
                m = batch.traj_id == t
                per_traj_step.append(so[m][0])
                per_traj_out.append(float(batch.outcome[m].sum()))
            r2 = float(np.corrcoef(per_traj_step, per_traj_out)[0, 1])
            print(f"    corr(该轨迹的 step 优势, 该轨迹的 outcome) = {r2:+.4f}")
            print("    ⇒ 若接近 ±1，说明 gamma=1.0 时 step 项**只是 outcome 的函数**，")
            print("      不含超出结果的信息 —— step 级与 episode 级的区别仅在于**分组方式**。")

    # ------------------------------------------------------------------ #
    h("结论")
    frac = float(np.mean(within_traj_std(results[1.0], batch.traj_id) < 1e-9))
    print(f"""
  gamma=1.0：{frac:.0%} 的轨迹内 step_rewards 恒定（标准差为 0）
            ⇒ **不含任何位置/过程信息**，step 级优势退化为
              "outcome 在「访问过同一 anchor 状态」的轨迹子集内的重新归一化"
  gamma=0.95：step_rewards 随离终点距离指数衰减，额外编码了"还有多远"
            ⇒ 但这是**位置的函数**，不是**过程质量的函数** ——
              它只知道"离结束还有几步"，不知道"这一步做得好不好"

  ⇒ 立论成立：GiGPO 的 step 级信号在没有外部过程奖励时，**无法提供过程信用**。
     这正是 PRM 要补的位置。

  ⚠️ 数据边界：本脚本用合成数据（见 common.py 的声明）。真实 ALFWorld 上
     轨迹长度分布不同会让具体数值变化，但 gamma=1.0 ⇒ 常数这一条是**代数必然**，
     与数据无关。
""")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
