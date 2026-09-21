"""`gigpo/core_gigpo.py` 的 CPU 单元测试。

对应实验计划 `exp/design.md` 的 **P0 阶段（E0.1 / E0.3 / E0.4）**：

- E0.1 建立可回归的夹具 —— 本文件整体
- E0.3 量化单元素组抹平 —— `TestStepNormReward::test_singleton_group_is_zeroed`
- E0.4 量化 gamma 对 step 信号信息量的影响 —— `TestDiscountedReturns::test_gamma_one_makes_step_rewards_constant`

这些结论直接影响 `exp/design.md` §2 的立论，改动上游算法后重跑本文件即可回归。
"""

import numpy as np
import pytest
import torch


# --------------------------------------------------------------------------- #
# to_hashable —— 分组键的归一化
# --------------------------------------------------------------------------- #
class TestToHashable:
    def test_scalars_pass_through(self, core_gigpo):
        assert core_gigpo.to_hashable(1) == 1
        assert core_gigpo.to_hashable(1.5) == 1.5
        assert core_gigpo.to_hashable("a") == "a"
        assert core_gigpo.to_hashable(True) is True

    def test_numpy_scalars_become_python(self, core_gigpo):
        out = core_gigpo.to_hashable(np.float32(1.5))
        assert isinstance(out, float) and out == pytest.approx(1.5)

    def test_ndarray_becomes_tuple(self, core_gigpo):
        out = core_gigpo.to_hashable(np.array([[1, 2], [3, 4]]))
        assert out == (1, 2, 3, 4)
        assert hash(out)  # 必须可哈希，否则分组会崩

    def test_nested_structures_are_recursive(self, core_gigpo):
        out = core_gigpo.to_hashable({"b": [1, np.array([2, 3])], "a": (4, 5)})
        # dict 按 key 排序，保证同内容不同插入顺序得到同一 hash
        assert out[0] == ("a", (4, 5))            # 标量元组原样保留
        assert out[1][0] == "b"
        assert out[1][1][0] == 1                  # list 的第一个元素
        assert tuple(int(v) for v in out[1][1][1]) == (2, 3)   # ndarray 展平后嵌在里面
        assert hash(out)          # 必须可哈希，否则 build_step_group 会崩

    def test_ndarray_elements_stay_numpy_scalars(self, core_gigpo):
        """注意：ndarray 分支只做 `tuple(x.flatten())`，**不转成 Python 标量**。

        留在元组里的是 np.int64 等 numpy 标量。这依赖 numpy 标量与 Python 标量
        的 `==` / `hash` 一致性（`hash(np.int64(2)) == hash(2)`），换 numpy 大版本
        时值得留意。这里把这个既有行为固定下来，避免上游改动被静默吞掉。
        """
        out = core_gigpo.to_hashable(np.array([2, 3]))
        assert isinstance(out[0], np.integer)
        assert out == (2, 3)          # 逐元素比较仍然成立
        assert hash(out) == hash((2, 3))

    def test_dict_key_order_does_not_matter(self, core_gigpo):
        a = core_gigpo.to_hashable({"x": 1, "y": 2})
        b = core_gigpo.to_hashable({"y": 2, "x": 1})
        assert a == b

    def test_unsupported_type_raises(self, core_gigpo):
        with pytest.raises(TypeError):
            core_gigpo.to_hashable(object())


# --------------------------------------------------------------------------- #
# are_similar —— 相似度分组的判定
# --------------------------------------------------------------------------- #
class TestAreSimilar:
    def test_identical_is_similar(self, core_gigpo):
        assert core_gigpo.are_similar("hello world", "hello world", 0.95) is True

    def test_threshold_boundary(self, core_gigpo):
        # "abc" vs "abd" ratio = 2*2/(3+3) = 0.666...
        assert core_gigpo.are_similar("abc", "abd", 0.95) is False
        assert core_gigpo.are_similar("abc", "abd", 0.6) is True

    def test_non_string_raises(self, core_gigpo):
        """只支持文本——这限制了 enable_similarity 只能用于纯文本任务。"""
        with pytest.raises(ValueError):
            core_gigpo.are_similar(("a", 1), ("a", 1), 0.95)


# --------------------------------------------------------------------------- #
# build_step_group —— anchor 状态分组（GiGPO 的核心机制）
# --------------------------------------------------------------------------- #
class TestBuildStepGroup:
    def test_identical_obs_share_group(self, core_gigpo):
        anchor = np.array(["s0", "s1", "s0", "s2"], dtype=object)
        index = np.array(["t", "t", "t", "t"])
        uids = core_gigpo.build_step_group(anchor, index)

        assert uids[0] == uids[2]          # 同状态 → 同组
        assert uids[1] != uids[0]
        assert uids[3] != uids[0]
        assert len(set(uids)) == 3

    def test_grouping_is_scoped_within_index(self, core_gigpo):
        """同样的观测在不同 task 下不能分到一组——状态等价性只在同一任务内有意义。"""
        anchor = np.array(["s0", "s0"], dtype=object)
        index = np.array(["t1", "t2"])
        uids = core_gigpo.build_step_group(anchor, index)
        assert uids[0] != uids[1]

    def test_similarity_mode_clusters_near_duplicates(self, core_gigpo):
        anchor = np.array(["open the fridge", "open the  fridge", "go to bed"], dtype=object)
        index = np.array(["t", "t", "t"])

        exact = core_gigpo.build_step_group(anchor, index, enable_similarity=False)
        assert exact[0] != exact[1]        # 精确匹配下是两步

        fuzzy = core_gigpo.build_step_group(anchor, index, enable_similarity=True, similarity_thresh=0.9)
        assert fuzzy[0] == fuzzy[1]        # 相似度下合并
        assert fuzzy[2] != fuzzy[0]

    def test_similarity_mode_rejects_out_of_range_threshold(self, core_gigpo):
        anchor = np.array(["a", "b"], dtype=object)
        index = np.array(["t", "t"])
        with pytest.raises(AssertionError):
            core_gigpo.build_step_group(anchor, index, enable_similarity=True, similarity_thresh=1.0)


# --------------------------------------------------------------------------- #
# episode_norm_reward —— 结果级优势
# --------------------------------------------------------------------------- #
def _mask(bs, length):
    return torch.ones((bs, length), dtype=torch.float32)


class TestEpisodeNormReward:
    def test_mean_norm_subtracts_mean_only(self, core_gigpo):
        # 两条轨迹同属一个 prompt 组，token 级奖励之和分别为 1 和 3
        rewards = torch.tensor([[1.0, 0.0], [3.0, 0.0]])
        adv = core_gigpo.episode_norm_reward(
            rewards, _mask(2, 2),
            index=np.array(["p", "p"]), traj_index=np.array(["A", "B"]),
            remove_std=True,
        )
        # 均值 2 → [-1, +1]，广播到所有 token
        assert adv[0, 0].item() == pytest.approx(-1.0)
        assert adv[1, 0].item() == pytest.approx(1.0)
        assert adv[0, 1].item() == pytest.approx(-1.0)   # 同轨迹内所有 token 共享

    def test_mean_std_norm_divides_by_unbiased_std(self, core_gigpo):
        rewards = torch.tensor([[1.0, 0.0], [3.0, 0.0]])
        adv = core_gigpo.episode_norm_reward(
            rewards, _mask(2, 2),
            index=np.array(["p", "p"]), traj_index=np.array(["A", "B"]),
            remove_std=False,
        )
        # 无偏标准差 std([1,3]) = sqrt(2) ≈ 1.41421
        expected = 1.0 / np.sqrt(2.0)
        assert adv[0, 0].item() == pytest.approx(-expected)
        assert adv[1, 0].item() == pytest.approx(expected)

    def test_singleton_group_keeps_raw_score(self, core_gigpo):
        """单元素组：mean=0/std=1 ⇒ 优势 = 原始分数（不是 0）。

        与 step 级的同名分支行为相反——见 TestStepNormReward。
        """
        rewards = torch.tensor([[5.0, 0.0]])
        adv = core_gigpo.episode_norm_reward(
            rewards, _mask(1, 2),
            index=np.array(["p"]), traj_index=np.array(["A"]),
            remove_std=True,
        )
        assert adv[0, 0].item() == pytest.approx(5.0)


# --------------------------------------------------------------------------- #
# step_norm_reward —— 步级优势
# --------------------------------------------------------------------------- #
class TestStepNormReward:
    def test_explicitly_grouped_rewards_are_normalized(self, core_gigpo):
        step_rewards = torch.tensor([1.0, 3.0])
        groups = np.array(["g", "g"])
        adv = core_gigpo.step_norm_reward(step_rewards, _mask(2, 2), groups, remove_std=True)
        assert adv[0, 0].item() == pytest.approx(-1.0)
        assert adv[1, 0].item() == pytest.approx(1.0)

    def test_singleton_group_is_zeroed(self, core_gigpo):
        """【E0.3 核心】单元素状态组 → 优势恒为 0。

        实现上是 `mean = 该分数本身`，于是 `score - mean == 0`。
        语义上说得通（只访问过一次的状态无法判断好坏），但**长轨迹里单元素组是
        常态**，意味着 GiGPO 的 step 级信号在大面积失效——这正是 PRM 的动机。
        """
        step_rewards = torch.tensor([5.0])
        groups = np.array(["only_visit"])
        adv = core_gigpo.step_norm_reward(step_rewards, _mask(1, 2), groups, remove_std=True)
        assert adv[0, 0].item() == pytest.approx(0.0)
        assert adv[0, 1].item() == pytest.approx(0.0)

    def test_adv_is_masked_outside_response(self, core_gigpo):
        """被 mask 掉的 token 位置优势必须为 0，否则会污染 loss。"""
        step_rewards = torch.tensor([1.0, 3.0])
        mask = torch.tensor([[1.0, 0.0], [1.0, 1.0]])
        adv = core_gigpo.step_norm_reward(step_rewards, mask, np.array(["g", "g"]), remove_std=True)
        assert adv[0, 1].item() == pytest.approx(0.0)
        assert adv[1, 1].item() == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
# compute_step_discounted_returns —— step 信号的来源
# --------------------------------------------------------------------------- #
class TestDiscountedReturns:
    def test_gamma_one_makes_step_rewards_constant(self, core_gigpo, make_dataproto):
        """【E0.4 核心】gamma=1.0 + 稀疏奖励 ⇒ step_rewards 恒等于最终结果。

        这是 `exp/design.md` §2 立论的直接证据：`ppo_trainer.yaml:235` 默认
        gamma=1.0，而各环境的每步奖励几乎全是 0、只有终止步非零。此时
        `step_rewards[t]` 对任意 t 都等于轨迹最终回报 R——**不含任何中间信息**，
        所谓的 "step-level" 只是把 outcome 在"访问过同一状态"的轨迹子集里重新归一化。

        ⇒ GiGPO 解决的是"跨轨迹的相对信用"，不是"过程信用"。PRM 补的是后者。
        """
        batch = make_dataproto(
            rewards=[0.0, 0.0, 10.0],
            traj_uid=["A", "A", "A"],
            active_masks=[1, 1, 1],
        )
        out = core_gigpo.compute_step_discounted_returns(batch=batch, gamma=1.0)
        assert out.tolist() == pytest.approx([10.0, 10.0, 10.0])

    def test_gamma_below_one_encodes_time_to_completion(self, core_gigpo, make_dataproto):
        """gamma<1 时 step_rewards 随离终点的距离衰减，才带上了一点"过程"信息。

        官方脚本 `run_alfworld.sh` 用的是 gamma=0.95，与 yaml 默认的 1.0 不同。
        """
        batch = make_dataproto(
            rewards=[0.0, 0.0, 10.0],
            traj_uid=["A", "A", "A"],
            active_masks=[1, 1, 1],
        )
        out = core_gigpo.compute_step_discounted_returns(batch=batch, gamma=0.95)
        assert out[2].item() == pytest.approx(10.0)
        assert out[1].item() == pytest.approx(9.5)
        assert out[0].item() == pytest.approx(9.025)

    def test_trajectories_are_discounted_independently(self, core_gigpo, make_dataproto):
        """两条轨迹交错时，折扣累计不能跨轨迹串味。"""
        batch = make_dataproto(
            rewards=[0.0, 0.0, 0.0, 10.0],      # A 失败，B 成功
            traj_uid=["A", "B", "A", "B"],
            active_masks=[1, 1, 1, 1],
        )
        out = core_gigpo.compute_step_discounted_returns(batch=batch, gamma=1.0)
        assert out[0].item() == pytest.approx(0.0)    # A 的第一步
        assert out[2].item() == pytest.approx(0.0)    # A 的第二步
        assert out[1].item() == pytest.approx(10.0)   # B 的第一步
        assert out[3].item() == pytest.approx(10.0)   # B 的第二步

    def test_requires_all_steps_active(self, core_gigpo, make_dataproto):
        """`core_gigpo.py:110` 的硬断言：done 之后的步必须在进 batch 前丢掉。"""
        batch = make_dataproto(
            rewards=[0.0, 10.0],
            traj_uid=["A", "A"],
            active_masks=[1, 0],
        )
        with pytest.raises(AssertionError):
            core_gigpo.compute_step_discounted_returns(batch=batch, gamma=1.0)


# --------------------------------------------------------------------------- #
# compute_gigpo_outcome_advantage —— 端到端融合
# --------------------------------------------------------------------------- #
class TestGigpoOutcomeAdvantage:
    def _fixture(self):
        """4 行 = 2 条轨迹 × 2 步，同一个 prompt 组。

        结果：轨迹 A 成功（末步 +10），轨迹 B 失败。
        状态：A 的第一步与 B 的第一步是同一状态 s0（共享初始状态），其余各自独立。
        """
        token_level_rewards = torch.tensor([
            [0.0, 0.0],   # A step0
            [0.0, 10.0],  # A step1（成功，末步拿分）
            [0.0, 0.0],   # B step0
            [0.0, 0.0],   # B step1
        ])
        step_rewards = torch.tensor([10.0, 10.0, 0.0, 0.0])
        response_mask = _mask(4, 2)
        anchor_obs = np.array(["s0", "s1", "s0", "s2"], dtype=object)
        index = np.array(["p", "p", "p", "p"])
        traj_index = np.array(["A", "A", "B", "B"])
        return token_level_rewards, step_rewards, response_mask, anchor_obs, index, traj_index

    def test_episode_term_matches_manual_normalization(self, core_gigpo):
        args = self._fixture()
        scores, returns = core_gigpo.compute_gigpo_outcome_advantage(*args, step_advantage_w=0.0)
        # episode 组内分数 = [0, 10, 0, 0] → 均值 2.5
        # mean_norm(remove_std=True) ⇒ [−2.5, 7.5, −2.5, −2.5]
        assert scores[0, 0].item() == pytest.approx(-2.5)
        assert scores[1, 0].item() == pytest.approx(7.5)
        assert scores[2, 0].item() == pytest.approx(-2.5)
        assert scores[3, 0].item() == pytest.approx(-2.5)
        assert torch.equal(scores, returns)   # GiGPO 无 critic，advantage 即 return

    def test_step_term_only_fires_on_shared_states(self, core_gigpo):
        """共享状态 s0（第 0、2 行）才产生非零 step 优势；单元素状态组被抹平。"""
        args = self._fixture()
        scores, _ = core_gigpo.compute_gigpo_outcome_advantage(*args, step_advantage_w=1.0)
        # step 组 s0 = [10, 0] → 均值 5 ⇒ [+5, −5]；s1/s2 是单元素组 ⇒ 0
        assert scores[0, 0].item() == pytest.approx(-2.5 + 5.0)
        assert scores[1, 0].item() == pytest.approx(7.5 + 0.0)
        assert scores[2, 0].item() == pytest.approx(-2.5 - 5.0)
        assert scores[3, 0].item() == pytest.approx(-2.5 + 0.0)

    def test_step_advantage_w_scales_only_the_step_term(self, core_gigpo):
        """权重只乘 step 项；episode 项系数硬编码为 1.0（GraphGPO 才加了 episode_advantage_w）。"""
        args = self._fixture()
        s0, _ = core_gigpo.compute_gigpo_outcome_advantage(*args, step_advantage_w=0.0)
        s2, _ = core_gigpo.compute_gigpo_outcome_advantage(*args, step_advantage_w=2.0)
        # 第 0 行的 step 优势为 +5，权重 2 ⇒ 额外贡献 +10
        assert s2[0, 0].item() == pytest.approx(s0[0, 0].item() + 10.0)

    def test_rejects_unknown_mode(self, core_gigpo):
        args = self._fixture()
        with pytest.raises(ValueError, match="Unknown mode"):
            core_gigpo.compute_gigpo_outcome_advantage(*args, mode="nope")
