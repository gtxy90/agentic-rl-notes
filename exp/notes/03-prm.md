# PRM（过程奖励模型）— 代码落点

> 证据分级：✅ 已核实｜🔍 已定位待验证｜⬜ 待办
> 基线 commit：`20bd331`（master, 2026-06-09）

## 0. 结论先行

**PRM 在仓库里完全不存在，但注入通道已经存在——而且比想象的更现成。**

GiGPO 已经有一个 `step_rewards` 输入张量，驱动着步级优势。但在主路径里，它**不是独立信号**，而是从稀疏结果奖励**折扣折算**出来的（return-to-go），**不含超出结果的新信息**。

→ 所以 PRM 的工作可以精确地定义为：**把 `ray_trainer.py:1116` 那一行写入的值，换成一个不依赖"同状态重复访问"的过程信号。** 下游的分组、归一化、融合全部不用动。

**但有一个硬缺口**：文本版 ALFWorld（默认 baseline）**没有现成的稠密过程标签**（§2.1）。这是本方向最需要先解决的问题。

## 1. 现状：已有什么

### 1.1 PRM / 过程奖励：完全不存在 ✅

- 全仓无 `prm` / `process_reward` 相关实现。
- `verl/workers/reward_model/`（`base.py`、`megatron/reward_model.py`）是**结果奖励模型**（outcome RM），且是 Megatron 后端的通用组件，未被本仓库的 agent 训练路径使用。
- `step_rewards` 只在 `recipe/GraphGPO/` 和 `recipe/hgpo/` 里被另行覆盖成别的形式（图路径回报），主路径没有。

### 1.2 注入通道：`step_rewards` ✅

主路径的产出点（`verl/trainer/ppo/ray_trainer.py:1111-1116`）：

```python
if self.config.algorithm.adv_estimator == AdvantageEstimator.GiGPO:
    step_rewards_tensor = core_gigpo.compute_step_discounted_returns(
        batch=batch,
        gamma=self.config.algorithm.gamma
    )
    batch.batch['step_rewards'] = step_rewards_tensor
```

而 `compute_step_discounted_returns`（`gigpo/core_gigpo.py:87-132`）做的是：

```python
running_return = traj_rewards[t] + gamma * running_return    # :117-119，沿轨迹逆序
```

其中 `traj_rewards` 来自 `non_tensor_batch['rewards']` —— **环境每步 reward，不是 token reward**。

**⚠️ 这里有一个必须讲清的关键点（也是本方向最重要的发现）**：

由于各环境的每步奖励**几乎全是 0、只有终止步非零**（逐个环境核实，见下表），`step_rewards` 的实际内容取决于 `gamma`：

| `gamma` | `step_rewards[t]` | 含义 |
|---|---|---|
| **1.0**（yaml `:235` 默认） | 恒等于轨迹最终结果 `R` | **完全不含中间信息**。step 级优势退化成"按 anchor 状态分组后的 outcome 归一" |
| **0.95**（`run_alfworld.sh` 用） | `R * 0.95^(剩余步数)` | 多了一个"距成功还有多远"的时间信号 |

**即：`gamma=1.0` 下 GiGPO 的 "step-level" 是形式上的——它没有用到任何真正的中间信号，只是把 outcome 在"访问过同一状态"的轨迹子集里重新归一化。**

这仍然是 GiGPO 的核心贡献（状态分组的相对优势），但**它补不了"过程"这一维**。这正是 PRM 的确切空白位，也是项目里必须讲清的 baseline 说明——否则容易被质疑"GiGPO 已经有 step reward 了"。

各环境的奖励稀疏性（已核实）：

| 环境 | 位置 | 奖励语义 |
|---|---|---|
| ALFWorld | `env_package/alfworld/envs.py:48-53` | 仅终止步 `10.0 * won` |
| WebShop | `env_package/webshop/envs.py:47-53` | 仅终止步，成功给 10.0 |
| Search | `skyRL_gym/envs/search/env.py:55-57` | 源码注释直接写 "No reward for intermediate steps for Search tasks" |
| gym-cards | `env_package/gym_cards/envs/{ezpoints,blackjack}.py` | 仅终止步 |
| **Sokoban** | `env_package/sokoban/sokoban/base.py:137-150` | **半稠密**：每把一个箱子推上目标 +1 |

**Sokoban 里还有一个仓库自带的先例**：`sokoban/base.py:129-150` 有一段**被注释掉的 `thinking_reward` 钩子**——作者曾尝试在步级奖励里插过程信号。这是本仓库"往 step 通道塞过程信号"的现成代码范式，值得作为起点参考。

### 1.3 每步 reward 是怎么来的 ✅

写入点：`agent_system/multi_turn_rollout/rollout_loop.py:387`

```python
assert len(rewards) == batch_size, f"env should return rewards for all environments, ..."
batch.non_tensor_batch['rewards'] = torch_to_numpy(rewards, is_object=True)
```

`rewards` 由环境管理器返回。以 ALFWorld 为例（`agent_system/environments/env_manager.py:153`）：

```python
text_obs, image_obs, rewards, dones, infos = self.envs.step(actions)
```

而真正的奖励函数在 `agent_system/environments/env_package/alfworld/envs.py:48-53`：

```python
def compute_reward(info, multi_modal=False):
    if multi_modal:
        reward = 10.0 * float(info['won']) + float(info['goal_condition_success_rate'])
    else:
        reward = 10.0 * float(info['won'])
    return reward
```

`envs.py:137` 对**每一步**都调用它。而 `multi_modal` 的定义在 `envs.py:97`：

```python
self.multi_modal = (env_type == 'AlfredThorEnv')
```

### 1.4 可复用的中间信号 ✅

`infos` 里可用的字段（ALFWorld）：

| 字段 | 可用性 | 性质 |
|---|---|---|
| `won` | **每步都有** | bool，但只有最后一步才为 True |
| `admissible_commands` | 每步都有 | 合法动作集合（动作空间约束，非进度信号） |
| `extra.gamefile` | 每步都有 | 任务标识（含任务类型名，见 `env_manager.py:229-243`） |
| `is_action_valid` | 每步都有 | `env_manager.py:162` 写入 |
| `goal_condition_success_rate` | **仅视觉版** | 稠密子目标完成率——**文本版拿不到**，见 §2.1 |

另外 `agent_system/memory/` 模块在 `memory.store()` 里已经存了 `(text_obs, action)` 对（ALFWorld 在 `env_manager.py:153`，Search/WebShop 在 `:70` / `:409`），这是 PRM 通常需要的 (state, action) 结构，**已有现成来源**。

## 2. 缺口

### 2.1 ✅ 稠密过程标签**存在**——只是没被请求（原判断已修正）

> **修正记录（2026-09-20）**：本节初稿写的是"文本版拿不到稠密标签，是方向性缺口"。
> 做最小实验后发现 **TextWorld 本身一直提供这些信号**，`AlfredTWEnv` 只是没请求。
> 缺口从"方向性阻塞"降级为"两行代码的改动"。

**证据（本地实测 + TextWorld 源码 `textworld/core.py` 的 `EnvInfos.__slots__`）**：

| 字段 | 实测行为 |
|---|---|
| `facts` | 世界当前全部事实（`Proposition` 列表），逐步变化 |
| `win_facts` | 目标条件，嵌套结构（见下），静态 |
| `intermediate_reward` | ⚠️ **0/1 二值，非分级** |
| `score` / `max_score` | ⚠️ **按 quest 完成跳变**；单 quest 多步链式任务中全程为 0 |

**⚠️ 关键修正**：`score`/`max_score` **不是**目标条件满足率——它统计的是 quest 完成数。
实测一个多步链式任务（objective："First off, take a trip east. Once you manage that,
recover the knife..."），`score` 在中间步**始终为 0**，直到最终完成才跳变。

**真正分级的信号**：

```
goal_rate = |facts ∩ 目标命题| / |目标命题|
```

实测（两个随机游戏）：`2/3 → 3/3`、`3/4 → 4/4`，**单调且在终局前变化**。
这是 THOR 版 `goal_condition_success_rate`（`alfred_thor_env.py:178` 的
`pcs[0] / float(pcs[1])`）在文本环境下的等价物。

嵌套结构（**踩过的坑**，`bs=1`）：

```
facts[0]     = list[Proposition]          扁平
win_facts[0] = list[quest] → quest = list[condition_group] → group = list[Proposition]
```

漏掉 batch 维会得到恒为 0 的结果。

现状只是没请求（`alfred_tw_env.py:254`）：

```python
request_infos = textworld.EnvInfos(won=True, admissible_commands=True, extras=["gamefile"])
```

**落地改动（共 2 处）**：

1. `alfred_tw_env.py:254` 的 `request_infos` 加入 `facts=True, win_facts=True`
   （`intermediate_reward` 可选，但实测是 0/1，价值低于命题满足率）
2. `envs.py:48` 的 `compute_reward` 里按上面的公式算出 rate，参考**同文件已有的
   `multi_modal` 分支写法**（`:50` 那行就是现成范式）

⚠️ 落地时注意 R2b：**不要就地改 `rewards`**，会同时污染 outcome 奖励，必须新开数组。

**已知稀释效应**：目标命题里有些**静态为真**（如 `at(drawer, washroom)`），reset 时
就已满足，所以基线不是 0 而是 0.67 / 0.75。这不破坏信号（增量仍在），但**不能把 rate
直接当"完成百分比"**，用增量或剔除静态命题更稳妥。

**验证状态**：
- ✅ **本地实测完成**——见 `../probe-prm-label/`（TextWorld 1.7.0，可复现）
- ⬜ **唯一剩下的验证**：在真实 ALFWorld 游戏上复现。ALFWorld 的游戏由 `json_2.1.1/*`
  的 PDDL 问题实例生成（不在仓库里，需 `alfworld-download`），**其目标的实际命题数未实测**。
  若只有 1 个命题，满足率退化成 0/1 与 `won` 等价。
  （ALFWorld 的任务类型 3/4/5 是 `pick_clean_then_place` 这类两段式、类型 6 是
  `pick_two_obj_and_place`，名字本身就暗示 ≥2 条件，但仍需实测坐实。）

### 2.2 其他缺口

- **无 PRM 训练/推理基础设施**：没有 step-level 标签的存储格式、没有 PRM 的加载/推理入口。
- **粒度是"步"不是"token"**：step 优势 tile 到整段 response（`gigpo/core_gigpo.py:382`），PRM 若想给 per-token 信号需要额外改造。
- **`step_rewards` 与 `gamma` 的语义耦合**（见 R3）。

## 3. 改造点

### 方案 1（推荐，侵入性最小）—— 替换 `step_rewards` 的来源

**改 `verl/trainer/ppo/ray_trainer.py:1111-1116` 一处。**

`step_rewards` 只是一个 shape `(bs,)` 的标量张量。把 `compute_step_discounted_returns` 换成 PRM 打分：

```python
# 现状
step_rewards_tensor = core_gigpo.compute_step_discounted_returns(batch=batch, gamma=...)

# 改成（示意）
step_rewards_tensor = compute_prm_scores(batch, prm_model, ...)
```

**下游完全不用动**：`build_step_group` 的分组、`step_norm_reward` 的归一化、`core_gigpo.py:170` 的融合，全部照常工作。

代价：需要把 PRM 打分所需的信息（anchor_obs、action、观测历史）组织成 `(bs,)` 张量。这些信息在 batch 里都有（`anchor_obs` 在 `rollout_loop.py:180`，action 可从 response 解码）。

### 方案 2 —— 与折扣回报相加/加权融合

在 `:1205` 之后追加 `batch.batch['step_rewards'] += w * prm_scores`。

⚠️ **不推荐作为首选**：此时 `step_rewards` 已经是**折扣回报**，直接相加语义不干净（见 R2）。若要用，应先调整顺序，让 PRM 与原始 per-step reward 在同一层级融合。

### 方案 3 —— 改融合公式

**改 `gigpo/core_gigpo.py:170` 一行**：

```python
scores = episode_advantages + step_advantage_w * step_advantages
```

适合做"PRM 信号与 GiGPO 步级信号如何配比"的消融。注意 episode 项系数当前**硬编码为 1.0**，GraphGPO 才补上了 `episode_advantage_w`（`recipe/GraphGPO/core_graph.py:850`）。

### 方案 4 —— 加独立 estimator（照抄 GraphGPO）

新建 `recipe/<name>/core_<name>.py`，`from gigpo.core_gigpo import episode_norm_reward, build_step_group, step_norm_reward` 复用三个基础函数，只替换 step 信号来源。这是**仓库自己推荐的扩展姿势**，GraphGPO 和 HGPO 都走这条路。适合 PRM 方案定型后固化。

## 4. 风险

### R1 —— 文本 ALFWorld 的标签缺口（见 §2.1）

这是**方向性的风险**：如果 TextWorld 内部状态取不到、且跑 Monte Carlo 采样的算力预算不够，PRM 在这个环境上就没有可靠标签来源，整个方向需要换环境（Sokoban / Gym Cards 可能有更结构化的状态）。

→ **建议：动手前先用一个最小实验确认 TextWorld 能否导出 goal-condition 信息**，再决定是否投入。

### R2 —— `step_rewards` 与 invalid action penalty 的顺序问题 🔍

`ray_trainer.py:1205` 调用 `apply_invalid_action_penalty`（`:202-222`），执行：

```python
step_rewards[i] -= invalid_action_penalty_coef * action_invalids    # :223
```

此时 `step_rewards` 已经是折扣回报（`:1112-1116` 先算）。**对一个"到轨迹结束的折扣和"减常数**，语义上不如减在原始 per-step reward 上干净。

→ 引入 PRM 时要特别注意这个顺序：PRM 打的分是**每步的即时质量**，而 penalty 现在减在**累积回报**上，两者混在一起会互相污染。

### R2b —— 就地改 `rewards` 会同时污染 outcome 奖励（**必读**）✅

`rollout_loop.py:383` 和 `:387` 读的是**同一个局部变量 `rewards`**：

```python
:383   episode_rewards[active_masks] += rewards[active_masks]      # → outcome 奖励
:387   batch.non_tensor_batch['rewards'] = torch_to_numpy(rewards) # → GiGPO step 通道
```

`:383` 的 `episode_rewards` 经 `EpisodeRewardManager`（`agent_system/reward_manager/episode.py:29`）变成 `token_level_scores`，即 outcome 奖励。

→ **想让 PRM 只影响 step 通道、不污染 outcome，必须新开一个数组**（如 `non_tensor_batch['prm_rewards']`），**不能就地改 `rewards`**。就地改会让"过程奖励"混进结果奖励，整个实验失去意义。

### R2c —— 注入位置的执行顺序约束 ✅

`ray_trainer.py` 里的顺序是刚性的：

```
:1112  step_rewards = compute_step_discounted_returns(...)   ← 先算
:1116  batch.batch['step_rewards'] = step_rewards
:1139  compute_reward(batch, self.reward_fn)                 ← 后算
:1197  token_level_scores = reward_tensor
:1203  apply_invalid_action_penalty(...)                     ← 又改一次 step_rewards
:1221  compute_advantage(...)                                ← 最终消费
```

**推论：若 PRM 走 reward_fn 路径（`compute_reward`），它的分数在 `:1112` 时还拿不到。** 必须在 `:1116` 之后注入，或把 PRM 计算前移到 rollout 阶段。

顺带：`apply_invalid_action_penalty`（`:202-222`）本身就是现成的"给 `step_rewards` 加逐项修正"的代码范式——**照抄它的写法即可**，不必另起结构。

### R3 —— `gamma` 字段的语义三重载 🔍

`algorithm.gamma` 被三处复用，语义完全不同：
- GRPO 的折扣
- GiGPO 的**步间折扣**（示例用 0.95）
- GraphGPO 的**图距离衰减**（用 0.10）

PRM 若也引入自己的折扣/温度参数，**不要复用 `gamma`**，否则会和现有语义打架、让消融实验无法解释。

### R3b —— 同构代码有三份，改一处要同步三处 ✅

`compute_step_discounted_returns` 在仓库里有**三个副本**：

- `gigpo/core_gigpo.py:87`（主）
- `recipe/GraphGPO/core_graph.py:714` / `:785`
- `recipe/hgpo/core_hgpo.py:30`（复制粘贴副本）

→ 若在主路径改 step 信号相关逻辑（方案 A/B），**记得同步这两处**，否则三个 recipe 行为不一致，跨算法对比会失效。

### R4 —— 单元素组抹平会影响 PRM 的收益评估 ✅

`gigpo/core_gigpo.py:365-369`：step 级单元素组 → 优势**恒为 0**（详见 `01-credit-assignment.md` R2）。

这意味着：**PRM 的信号只有在"同一状态被多条轨迹访问"时才能通过 GiGPO 的步级通道生效。** 若长轨迹下单元素组是常态，PRM 即使在方案 1 上正确接入了，也可能**看不出效果**——而这未必是 PRM 的问题，是分组机制的问题。

→ **做 PRM 消融时必须同时记录 step-level 组大小分布**（`summarize_group_size` 在 `core_gigpo.py:49` 已有现成实现），否则会把"分组失效"误判成"PRM 无效"。

## 5. 对项目的含义

1. **PRM 方向与信用分配方向天然咬合，应该合并推进**，而不是当作两条独立线。PRM 提供的正是"不依赖同状态重复访问"的过程信号——这恰好补上了 `01` 文档 R2 指出的 GiGPO 在长轨迹上的弱点。
2. **第一优先级是确认标签来源**（§2.1），而不是先写 PRM 模型。没有标签，后面的工作全部悬空。
3. **接入点非常干净**（`ray_trainer.py:1116` 一行），这意味着验证成本低——可以先接一个**假的/规则的** step 信号（比如"动作是否合法"、"是否推进了子目标"）跑通全链路，确认管线无误后再换成真 PRM。这是低风险的第一步。
4. 若最终走方案 4（独立 estimator），可完全复用 GraphGPO 的代码模式，**工作量可控**。
