# 最小实验：文本版 ALFWorld 的稠密过程标签可行性

**日期**：2026-09-20　**状态**：✅ 已在本地实测完成（TextWorld 1.7.0 / macOS arm64）
**复现**：`cd exp/probe-prm-label && .venv/bin/python probe.py`

> ⚠️ **可复现性说明**：探针每次生成的是**随机游戏**（`GameOptions` 未固定种子），
> 所以具体数值每次不同。已观察到的 reset 基线有 `0.50 / 0.67 / 0.75` 等。
> **定性结论（满足率在终局前单调上升）在每次运行中都成立**，但不要拿本文件的
> 具体数字当基准值。要固定游戏需自行设置种子。

**结论**：✅ **可行**。TextWorld 提供全部所需原始量，且**目标命题满足率是真正分级的**
（实测 0.75 → 1.00，单调、在终局前变化）。缺口从"方向性阻塞"降级为**两行代码的改动**。

---

## 1. `EnvInfos` 提供什么（✅ 实测）

TextWorld 1.7.0 的 `EnvInfos` 共 26 个可用字段，其中候选稠密信号**全部存在**：

| 字段 | TextWorld 自带文档 | 实测行为 |
|---|---|---|
| `intermediate_reward` | "Reward (proxy) indicating if the player is making progress. Changes from one step to another." | **0/1 二值，非分级** ⚠️ |
| `score` / `max_score` | 当前分 / 可达满分 | **按 quest 完成跳变**，单 quest 多步链式任务中全程为 0 ⚠️ |
| `facts` | "All the facts that are currently true about the world." | 世界全部事实（Proposition 列表） |
| `win_facts` | "Mutually exclusive sets of winning facts for each quest." | 目标条件（嵌套结构，见 §3） |
| `moves` | 已走步数 | 逐步递增（但只是计数器，非进度） |

**⚠️ 关键修正**：我原先以为 `score`/`max_score` 就是目标条件满足率。
**实测不是**——`score` 统计的是 **quest 完成数**。对一个多步链式任务
（objective 原文："First off, take a trip east. **Once you manage that**, recover the knife
from the refrigerator..."），`score` 在中间步**始终为 0**，直到最终完成才跳变。
`intermediate_reward` 同样是 0/1 而非分级。

## 2. 真正分级的信号：目标命题满足率（✅ 实测）

```
goal_rate = |facts ∩ 目标命题| / |目标命题|
```

实测两次（不同随机游戏）：

| 游戏 | 目标命题数 | reset | 中间步 | 终局 | 单调性 |
|---|---|---|---|---|---|
| A（nq=2） | 3 | 2/3 = 0.67 | **3/3 = 1.00** | 3/3 | ✅ |
| B（nq=2, 4 rooms） | 4 | 3/4 = 0.75 | **4/4 = 1.00** | 4/4 | ✅ |

**在终局之前就发生了变化**，这正是 PRM 需要的过程标签。

对 ALFWorld 的含义：像 `pick_clean_then_place`（清洁 + 放置）这类任务，目标是
`isClean(X) ∧ inReceptacle(X, Y)` 形式的多谓词合取（见 `alfred/data/alfred.pddl`
里的 `isClean` / `isHot` / `isCool` / `inReceptacle`），满足率应当给出
**0 → 0.5 → 1.0** 的分级。

**注意一个已知的稀释效应**：目标命题里有些是**静态为真**的（如
`at(drawer, washroom)` 这类位置关系），它们在 reset 时就已经满足。
所以基线不是 0 而是 0.67 / 0.75。这不破坏信号（增量仍在），但意味着
**不能直接把 rate 当作"完成了百分之多少"**，用增量或去掉静态命题更稳妥。

## 3. 结构踩坑（照抄会出错）

实测的嵌套层级（`batch_size=1`）：

```
facts      : list[bs] → facts[0]     = list[Proposition]      世界当前全部事实（扁平）
win_facts  : list[bs] → win_facts[0] = list[quest]
                            quest    = list[condition_group]
                            group    = list[Proposition]      ← 合取，进度粒度在此
```

- **漏掉 batch 维**（直接对 `facts` 和 `win_facts` 求交）会得到恒为 0 的结果——
  我第一版就是这样，白跑一次。
- `facts` 里是 `Proposition` 对象（`textworld.logic`），**不是** list/dict。
- 某些字段（`score`/`max_score`/`intermediate_reward`/`won`）返回 **list**，
  需要取 `[0]`。

## 4. 本地安装（结论：能装，但慢）

> **修正记录**：我中途曾判断"jericho 无 macOS wheel，所以装不了"——**错误**。
> `jericho` 确实只有 sdist，但能正常构建，产出 `py3-none-any` **纯 Python wheel**。
> 教训：`pip install --only-binary=:all:` 失败 ≠ 无法安装。

**实际耗时**：`textworld` 1.7.0 同样只有 sdist，构建期要下载 **22.1 MB 的 Inform 7
发行包**（`textworld/thirdparty/inform7/`）。本机网络实测 **~42 KB/s，约 9 分钟**。
首次尝试因下载中断而失败，重跑即成功。

**建议**：远程机器上 textworld 是训练环境必备依赖（`alfworld` 依赖它），
装一次即可，本机不必重复。

## 5. 下一步

**远程（P1 之后）——唯一剩下的验证**：
在真实 ALFWorld 游戏上复现本探针，确认
1. `goal_rate` 确实分级（预期 0 → 0.5 → 1.0）
2. 静态命题占比有多大

**落地改动（共 2 处）**：
1. `alfred_tw_env.py:254` 的 `request_infos` 加入 `facts=True, win_facts=True`
   （`intermediate_reward` 可选，但实测它是 0/1，价值低于命题满足率）
2. `envs.py:48` 的 `compute_reward` 里按 §2 公式算出 rate，参考同文件 `multi_modal`
   分支（`:50`）的写法

⚠️ 注入时注意 `../notes/03-prm.md` R2b：**不要就地改 `rewards`**（会同时污染
outcome 奖励），必须新开数组如 `non_tensor_batch['prm_rewards']`。

## 6. 这个探针改变了什么

| | 探针前 | 探针后 |
|---|---|---|
| 标签来源 | 未知，可能需采样训 PRM 或人工标注 | **现成**：`facts` + `win_facts` 算出 |
| 方向风险 | 可能整体不成立 | 低——只剩"ALFWorld 上复现" |
| 工作量 | 可能需 Monte Carlo 采样（算力开销大） | 环境侧小改（2 处） |
| 信号选择 | 以为 `score` 就够 | **`score` 不够**，必须按命题算满足率 |
