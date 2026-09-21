# 长轨迹信用分配 — 代码落点

> 证据分级：✅ 已核实（读代码确认）｜🔍 已定位待验证（有依据但未逐行确认）｜⬜ 待办
> 基线 commit：`20bd331`（master, 2026-06-09）
> 文中行号均指 `repo/` 下的相对路径。

## 1. 现状：仓库里已经有什么

信用分配的**全部逻辑只有 7 个函数**，集中在 `gigpo/core_gigpo.py`（384 行）。这是 GiGPO 论文的官方实现，NeurIPS 2025。

### 1.1 张量布局（理解全文的前提）✅

batch 是**扁平的 token-level 批**：**一行 = 一个 agent step 的一次 LLM 调用**。
`bs` = 总步数（不是轨迹数），`response_length` = 单步回复的 token 数。

这一点极易误读——`rewards`、`step_rewards` 都是 `(bs,)`，即"每步一个标量"，不是"每轨迹一个标量"。

### 1.2 两级优势 + 融合

入口是唯一编排点 `compute_gigpo_outcome_advantage`（`gigpo/core_gigpo.py:138-171`）✅：

```python
# :161  结果级优势
episode_advantages = episode_norm_reward(token_level_rewards, response_mask, index, traj_index, epsilon, remove_std)
# :164  anchor 状态分组
step_group_uids = build_step_group(anchor_obs, index, enable_similarity, similarity_thresh)
# :167  步级优势（注意第三参传的是 step_group_uids，不是 index）
step_advantages = step_norm_reward(step_rewards, response_mask, step_group_uids, epsilon, remove_std)
# :170  融合（Eq.8）
scores = episode_advantages + step_advantage_w * step_advantages
return scores, scores     # advantage 与 return 相同，GiGPO 不用 critic
```

**episode 级**（`episode_norm_reward`, `:174-240`）：每步一个标量 `token_level_rewards.sum(-1)`（`:209`），按 `index`（同一 prompt 的 n 条轨迹为一组）做归一，再广播到该步全部 token 并乘 `response_mask`（`:238`）。
→ **同一条轨迹内所有步共享同一个 episode 优势**。

**step 级**（`step_norm_reward`, `:334-384`）：输入 `step_rewards` 形状 `(bs,)`，按 `step_group_uids` 分组归一，tile 到 `response_length`（`:382`）。
→ **同一个 response 内所有 token 共享同一个 step 优势**。

**归一化模式**（`:153-158`）：`mode="mean_norm"` → `remove_std=True`，只减均值；`"mean_std_norm"` → 减均值除标准差。

### 1.3 anchor state 分组（GiGPO 的核心机制）

`build_step_group`（`:243-331`）：先按 `index`（task）分桶，**桶内**再聚类。跨 task 不共享状态。

- `enable_similarity=False`（默认）：用 `to_hashable(obs)`（`:34-47`）做**精确 hash** 分簇。
- `enable_similarity=True`：**贪心增量聚类**——新 obs 只与簇代表（簇内第一个 obs）比，用 `are_similar`（`:72-85`，`difflib.SequenceMatcher.ratio() >= threshold`）。⚠️ 不与簇内其他成员比，**相似度不具传递性**。
- 每簇分配 `uuid4()` 作为 `step_group_uid`。

`to_hashable` 递归处理 ndarray/list/tuple/dict，其他类型 `raise TypeError`。

**`anchor_obs` 的来源** ✅ —— 按环境管理器不同，anchor 的语义**不一样**：

| 环境 | 位置 | anchor 是什么 |
|---|---|---|
| ALFWorld | `env_manager.py:164` | `text_obs` —— 环境返回的**原始观测**（非模板化 prompt） |
| Search | `env_manager.py:78` | `next_obs.copy()` —— 原始观测拷贝 |

对照：同一步喂给模型的是 `build_text_obs(...)` 产出的模板化文本（ALFWorld 在 `:180`），而 anchor 用的是**模板化之前**的原始观测。这个区分很重要：GiGPO 的状态等价性判断基于原始环境状态，不受 prompt 模板变化干扰。

写入路径：`env_manager.py:164` → `rollout_loop.py:180` 的 `non_tensor_batch['anchor_obs']`。

### 1.4 `step_rewards` 从哪来

`compute_step_discounted_returns`（`:87-132`）：用 `non_tensor_batch['rewards']`（**环境每步 reward**，非 token reward）沿轨迹**逆序折扣累加**：

```python
running_return = traj_rewards[t] + gamma * running_return   # :117-119
```

所以 `step_rewards[t]` 的语义是 **"从第 t 步开始的折扣回报"（return-to-go）**。

⚠️ `:110` 有硬断言 `assert traj_active_masks.all()`——done 之后的步必须在进 batch 前被丢掉。丢弃发生在 `rollout_loop.py:266`（`if data['active_masks']`）。

### 1.5 数据流的完整链路 ✅

| key | 写入位置 | 内容 |
|---|---|---|
| `uid` | `rollout_loop.py:358` | 同一 prompt 的 env 组（GRPO/GiGPO 的 group） |
| `traj_uid` | `rollout_loop.py:325`、`:359` | 每个 env 一条轨迹的 uuid |
| `anchor_obs` | `rollout_loop.py:180` | 原始观测 |
| `rewards` | `rollout_loop.py:387` | **环境每步 reward** |
| `active_masks` | `rollout_loop.py:333`、`:388` | `np.logical_not(is_done)` |
| `is_action_valid` | `rollout_loop.py:374-377` | 动作是否合法 |
| `step_rewards` | **`ray_trainer.py:1116`** | 由 trainer 算出后塞进 `batch.batch` |
| `token_level_rewards` | `ray_trainer.py:1215` | `= token_level_scores` |

trainer 侧编排（均在 `verl/trainer/ppo/ray_trainer.py`）：

```
:1082  traj_collector.multi_turn_loop(...)           # 跑环境交互，产出上面各 key
:1111  if adv_estimator == GiGPO:
:1112      step_rewards = compute_step_discounted_returns(batch, gamma)
:1116      batch.batch['step_rewards'] = step_rewards
:1118  batch = adjust_batch(config, batch)            # ⚠️ 见风险 R1
:1205  apply_invalid_action_penalty(...)              # ⚠️ 见风险 R4
:1215  batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]
:1221  compute_advantage(...)                         # :1232-1235 读 algorithm.gigpo.*
```

分发器 `compute_advantage` 在 `:244`，GiGPO 分支在 `:345-359`。

### 1.6 可调超参（全部 5 个）✅

`verl/trainer/config/ppo_trainer.yaml`：

| 参数 | 行 | 默认 | 备注 |
|---|---|---|---|
| `algorithm.gamma` | `:235` | **1.0** | GiGPO 示例脚本用 `0.95` |
| `algorithm.gigpo.step_advantage_w` | `:251` | 1.0 | 仅作用于 step 项 |
| `algorithm.gigpo.mode` | `:252` | `"mean_norm"` | 示例脚本用 `"mean_std_norm"` |
| `algorithm.gigpo.enable_similarity` | `:253` | False | 仅文本任务 |
| `algorithm.gigpo.similarity_thresh` | `:254` | 0.95 | |
| `env.max_steps` | `:295` | 50 | |
| `env.rollout.n`（组大小） | `:301` | 1 | 示例脚本用 8 |

⚠️ **`episode_advantage_w` 不存在**——episode 项系数在 `:170` 硬编码为 1.0。那是 GraphGPO 后加的（`recipe/GraphGPO/core_graph.py:850`）。

### 1.7 两个可对比的近亲

**GraphGPO**（ICML 2026）：**换 step return 的来源**——从"沿轨迹逆序折扣 env reward"换成"状态转移图上到目标的最短路径距离"（`recipe/GraphGPO/core_graph.py:774-838`，反向 Dijkstra 在 `:809`）。分组与归一化机制**原样复用 GiGPO**（`core_graph.py:9` 直接 `from gigpo.core_gigpo import episode_norm_reward, build_step_group, step_norm_reward`）。

**HGPO**（ICLR 2026）：**换分组方式**——把"当前状态"扩展成"最近 k 步的历史观测序列"（`recipe/hgpo/core_hgpo.py:243-249`，key 是 `(k, 最近k步序列)`），一个 step 会落入多个层级组，再用 `w ∝ L^alpha` 加权聚合（`:200-202`）。

> 一句话对比：**GiGPO 换的是"怎么分组"，GraphGPO 换的是"step 信号从哪来"，HGPO 换的是"分组键的上下文长度"。**

## 2. 缺口：要做事还缺什么

✅ **PRM 没有**（见 `03-prm.md`）。
✅ **无 per-token 的过程信号**：step 优势 tile 到整段 response（`:382`），粒度是"步"不是"token"。
🔍 **状态等价性判断很弱**：默认精确字符串匹配；开相似度也只有 SequenceMatcher（字符级），且不具传递性。对长轨迹里"语义相同但表述不同"的状态，会退化成几乎每个状态都是单元素组。
✅ **单元素组被抹平**（见 R2）——这在长轨迹里恰恰是常态。

## 3. 改造点（按侵入性排序）

| 方案 | 改动位置 | 适用 |
|---|---|---|
| **(a) 只换 step 信号** | `ray_trainer.py:1111-1116` | 接 PRM。`step_rewards` 只是一个 `(bs,)` 标量张量，换掉来源即可，下游分组/归一/融合完全不动。**最干净** |
| **(b) 换融合公式** | `core_gigpo.py:170` 一行 | 非对称权重、加 episode 权重、换非线性融合 |
| **(c) 换 anchor 定义** | `rollout_loop.py:180` 或 `build_step_group` 簇 key（`:286`/`:308`） | PRM 通常要 (state, action) 对而非纯 state。⚠️ `are_similar` 只接受 str（`:83-84`），改元组会直接 raise |
| **(d) 换相似度度量** | `core_gigpo.py:72-85` | 换成 embedding 距离 / 学习到的状态抽象 |
| **(e) 改 per-token 粒度** | `core_gigpo.py:382` 与 `:238` | per-token 过程奖励必须改这两处 |
| **(f) 加全新 estimator** | 照抄 `recipe/GraphGPO/` 模式 | 仓库自己推荐的扩展姿势：新建 `recipe/<name>/core_<name>.py` 复用三个基础函数 + 加 `AdvantageEstimator` 枚举 + 新 trainer 的 `elif` 分支 |

## 4. 风险

### R1 —— 主 trainer 的 `adjust_batch` 会污染分组统计 ✅（已量化：**影响很小，优先级低**）

`ray_trainer.py:1118` 调用 `adjust_batch(config, batch)`，默认 `mode="copy"`（`agent_system/multi_turn_rollout/utils.py:121-126`）——batch size 不整除时会**随机复制若干行**。

这些复制行带**相同的 `uid` / `traj_uid` / `anchor_obs`**，会原样进入 GiGPO 的分组与 mean/std 计算，**等于给部分样本额外加权**。位置还在 `compute_advantage`（`:1221`）**之前**。

佐证：HGPO 的 trainer 特意把这一步挪到 `compute_advantage` **之后**（`recipe/hgpo/hgpo_ray_trainer.py:1219`，文件头 `:18-19` 有注释说明原因），**主 trainer 没有这个处理**。

**✅ 已量化（E0.2，见 `../report.md`）——影响很小**：

| 指标 | 实测（bs=4064, `size_divisor=64`） |
|---|---|
| `to_add` | 平均 32 行（**0.8%**） |
| 对原有行优势的 \|Δ\| | 均值 ~0.0005，最大 ~0.005 |
| 被"提升"为多元素组的步 | 25 个（占单元素步 **0.83%**） |

原因：bs 是**几千**量级，而 `to_add` 上限只有 63。

→ **风险排序远低于 R2**（74%）。若要严格控制变量可把它挪到 `compute_advantage` 之后，
但**不必在项目早期投入**。唯一隐患是逐轮随机选行 → 额外噪声而非稳定偏置。

### R2 —— 单元素组在 episode 与 step 两级行为不一致 ✅（已量化：**影响很大，是核心问题**）

| | `len==1` 时 | 结果 |
|---|---|---|
| episode（`:225-227`） | `mean=0.0, std=1.0` | 优势 = **原始分数** |
| step（`:365-369`） | `mean=torch.mean(该分数)` | 优势 = **恒为 0** |

即**只被访问过一次的状态组会被完全抹平成零优势**。语义上说得通（没见过第二次 → 无法判断好坏），但**长轨迹里单元素组是常态**。

**✅ 已量化（E0.3，见 `../report.md`）——这是核心问题，不是边角料**：

| 指标 | step 级 | episode 级 |
|---|---|---|
| 单元素组内的步占比 | **74.3%** | 0%（每组恒 8 条） |
| 优势非零的步占比 | **12.1%** | **100.0%** |

即便把状态共享概率拉到极不现实的 0.8，单元素步占比仍有 10.7%。
`enable_similarity` 也救不了（75.4% → 71.2%，因为观测差异是**内容**差异不是拼写差异）。

→ **GiGPO 的 step 级信号在长轨迹上大面积失效**，且这是**结构性**的、非调参能解决。
这正是 PRM 的量化动机：用**不依赖"同状态重复访问"**的过程信号补上这一维。
跨 task 看指标时极易误判。

### R3 —— 标准差实现的三个隐患 🔍

`torch.std(torch.tensor([id2score[idx]]))`（`:230`、`:372`）多包了一层 list，变成 2D `(1,N)`，靠 `torch.std` 对全部元素归约才**碰巧**正确。且用默认 `correction=1`（unbiased）：

- `N=2` 时标准差被放大 **√2 倍** → `mean_std_norm` 模式下优势被系统性缩小。
- `N=1` 时 unbiased std = NaN —— 这正是需要 `len==1` 特判的原因。

### R4 —— invalid action penalty 减在折扣回报上 🔍

`ray_trainer.py:1205` → `:202-222` 的 `apply_invalid_action_penalty` 执行 `step_rewards[i] -= coef * action_invalids`。此时 `step_rewards` 已经是**折扣回报**（`:1112-1116` 先算），对一个"到轨迹结束的折扣和"减常数，语义上不如减在原始 per-step reward 上干净。**接入 PRM 时要注意这个顺序**。

### R5 —— 依赖声明自相矛盾 ✅

| | `requirements.txt` | `setup.py` |
|---|---|---|
| transformers | `==4.51.1`（`:19`） | `<=4.57.3`（`:41`） |
| tensordict | `<=0.6.2`（`:17`） | `>=0.8.0,<=0.10.0,!=0.9.0`（`:40`） |

**tensordict 的约束交集为空**，两套装不到一起。`requirements.txt` 还把 vllm 注释掉了（`:20` `# vllm==0.8.4`）。

→ **以 `setup.py` 为准**（它是 `pip install -e .` 实际生效的那个）。本地 CPU 环境用的是 setup.py 的区间。

### R6 —— 其他 🔍

- `compute_mean_std_cross_steps` 默认 `True`，而 `seen_pairs.add(...)` 只在 `if not compute_mean_std_cross_steps` 时执行（`:221-222`）→ `seen_pairs` 永远是空集（`:218` 的去重永不触发）。名字叫 "cross_steps"，实现上 `True` 才是"不去重"。
- `compute_step_discounted_returns` 的 docstring 说 "Eq.5"，但用的是 env reward 而非 token reward，返回值是 `(bs,)` 而非 per-token（`:87-98`）。
- `gamma` 字段被三处复用但**语义完全不同**：GRPO 的折扣、GiGPO 的步间折扣（0.95）、GraphGPO 的图距离衰减（0.10）。
- 不要用主 trainer 跑 GraphGPO：`core_graph.py:789-791` 需要 `non_tensor_batch['next_obs']`，而主 `rollout_loop.py` **从不写 `next_obs`**（只写 `anchor_obs`）。必须走 `recipe.GraphGPO.main_graphgpo`。
- `compute_advantage` 收了 `use_pf_ppo` / `pf_ppo_reweight_method` / `pf_ppo_weight_pow` 但函数体完全不用（靠 `**kwargs` 吞掉），config 里 `algorithm.pf_ppo`（yaml `:246-249`）是死配置。
- `apply_kl_penalty` 漏传 `multi_turn`（`ray_trainer.py:1212`）：签名有 `multi_turn=False`（`:152`），`:174-179` 用它决定 mask 取 `loss_mask` 还是 `attention_mask`，但调用点没传。默认 `use_kl_in_reward=False` 所以平时不触发，**一旦打开就是隐藏 bug**。

## 5. 结论

- **有强基线可直接改造**，且改造点非常集中（核心只有 `core_gigpo.py` 一个文件 + trainer 两处）。
- **R1 与 R2 的实测影响相差两个数量级**（<1% vs 74%）：**精力应放在 R2，不是 R1**。
  这是 P0 量化后对原判断的修正 —— 原先把两者并列。
- **R2 是项目的核心问题**：74% 的步优势被单元素组抹平，且 `enable_similarity` 无效、
  调参也无效（结构性）。它直接回答"长轨迹下 GiGPO 的 step 级信号是否还有效"。
- R2 与 PRM 天然咬合：既然 step 信号因"同状态重复访问"稀缺而大面积失效，
  **用 PRM 提供一个不依赖重复访问的过程信号**就是有明确动机、有量化支撑的路线。
  配合 E0.4 的结论（`gamma=1.0` 时 step 项位置信息零增量），立论链完整：
  **当前 step 通道既无过程信息、又大面积失效 → 必须外部注入过程信号。**
