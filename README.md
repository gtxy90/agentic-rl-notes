# agentic-rl-notes

对 [`verl-agent`](https://github.com/langfengQ/verl-agent)（veRL 的 LLM Agent RL 扩展，
GiGPO 论文官方实现）的**代码分析笔记 + 可复现实验脚手架**。

这个仓库不重新分发上游代码。它记录的是：读代码读出来的东西、以及能被复现的实验数字。

## 三个有实测数字支撑的发现

### 1. GiGPO 的 "step-level" 在没有外部过程奖励时不含过程信息

`ppo_trainer.yaml` 默认 `gamma=1.0`，而环境的每步奖励几乎全是 0、只有终止步非零。
于是 `compute_step_discounted_returns` 算出的 `step_rewards[t]` **恒等于轨迹最终结果 R**。

| 判据 | `gamma=1.0` | `gamma=0.95` |
|---|---|---|
| 轨迹内 `step_rewards` 恒定的轨迹占比 | **100.0%** | 49.2% |
| `R²(step_rewards ~ 轨迹级 outcome)` | **1.000000** | 0.8668 |
| 位置信息带来的**增量** R² | **+0.000000** | +0.0149 |

即：step 级优势只是"把 outcome 在访问过同一状态的轨迹子集里重新归一化"，
**不含超出结果的信息**。`gamma<1` 确实带来额外信息，但那是**位置**的函数
（"离结束还有几步"），不是**过程质量**的函数（"这一步做得好不好"）。

复现：`bash exp/p0/run_all.sh` → E0.4

### 2. step 级优势被单元素组大面积抹平

`step_norm_reward` 对单元素组的输出恒为 0（`core_gigpo.py:365-369`，与 episode 级的
`:225-227` 行为相反）。而 GiGPO 的 step 分组键是 anchor 观测——长轨迹下各 rollout
很快分道扬镳，单元素组成为常态：

| 指标 | step 级 | episode 级 |
|---|---|---|
| 单元素组内的步占比 | **74.3%** | 0%（每组恒 8 条） |
| 优势非零的步占比 | **12.1%** | **100.0%** |

即便把状态共享概率拉到极不现实的 0.8，单元素步占比仍有 10.7%。
`enable_similarity` 也救不了（75.4% → 71.2%）——观测差异是**内容**差异不是拼写差异，
字符级 `SequenceMatcher` 合并不了。

复现：`bash exp/p0/run_all.sh` → E0.3

### 3. 两处被并列的风险，实测影响相差两个数量级

| | 机制 | 实测影响 |
|---|---|---|
| **R1** | `ray_trainer.py:1118` 在 `compute_advantage` **之前**调用 `adjust_batch(mode="copy")`，随机复制行进入分组统计 | **<1%** |
| **R2** | 上述单元素组抹平 | **74%** |

R1 因为 bs 是几千量级而 `to_add` 上限只有 63，实际影响很小。
**量化之后优先级完全变了**——这正是做实验而不是读代码猜的价值。

复现：`bash exp/p0/run_all.sh` → E0.2

## 一个可用的过程标签来源

PRM（过程奖励模型）方向通常卡在"没有过程标签"。实测发现 TextWorld 的 `EnvInfos`
一直提供 `facts`（当前事实）与 `win_facts`（目标条件），

```
goal_rate = |facts ∩ 目标命题| / |目标命题|
```

是**真正分级**的过程信号（实测 `2/3 → 3/3`、`3/4 → 4/4`，单调且在终局前变化）。
文本版 ALFWorld 只是没请求这两个字段（`alfred_tw_env.py:254`）。

⚠️ 反直觉之处：`score` / `max_score` **不能**当进度用——它统计的是 quest 完成数，
单 quest 多步链式任务中全程为 0；`intermediate_reward` 是 0/1 而非分级。
**必须按命题算满足率。**

复现：`cd exp/probe-prm-label && python probe.py`（需 `pip install textworld`）

## 目录

```
exp/
  design.md          技术路线：方向分析、实验矩阵、陷阱清单
  report.md          实验报告：所有实测数字的唯一归档处
  notes/             代码落点分析，含 文件:行号 证据
    01-credit-assignment.md   信用分配（GiGPO / GraphGPO / HGPO 对比）
    02-async-rl.md            异步 RL 与 staleness
    03-prm.md                 过程奖励标签
  p0/                可本地复现的实验（CPU + 合成数据，无需 GPU）
    run_all.sh       一键跑完
    common.py        合成 batch 构造（含数据来源声明）
  probe-prm-label/   PRM 标签可行性实测（TextWorld）
deploy/              远程部署资产
  setup_remote.sh    conda + vllm + flash-attn + 环境包
  env.sh             登录后恢复环境变量
  check_env.py       环境自检（支持 --local 验证不误报）
  run_single_gpu.sh  单卡（24G）启动配置
tests/               本地 CPU 算法单测（28 项）
```

## 本地能跑什么

⚠️ **先克隆上游**。本仓库不含上游代码，但测试与实验都会 import 它，
需要把它放在 `repo/` 下：

```bash
git clone https://github.com/langfengQ/verl-agent.git repo
```

上游的 `vllm` / `flash-attn` 在 macOS arm64 上装不了，但**算法层不依赖 GPU**：
`verl/protocol.py` 只需要 numpy/pandas/ray/tensordict/torch，且 flash-attn 在上游是
`try/except` 保护的（`verl/utils/torch_functional.py:31-36`）。所以信用分配的数学
可以完全在 CPU 上验证。

```bash
python3 -m venv .venv
.venv/bin/pip install -r tests/requirements-local.txt
.venv/bin/python -m pytest tests/ -q     # 28 项
bash exp/p0/run_all.sh                   # P0 四项实验
```

PRM 探针额外需要一个 TextWorld 环境：

```bash
cd exp/probe-prm-label
python3 -m venv .venv && .venv/bin/pip install textworld
.venv/bin/python probe.py
```

单测验证的是**算法数学正确性**，不是训练效果。

## 数据边界

本机无 CUDA，所以仓库里的数字有两类来源，都已明确标注：

- **代数必然**（与数据无关）：如 E0.4 的 `gamma=1.0 ⇒ step_rewards 恒定`
- **合成数据**（结构按真实配置建模，见 `exp/p0/common.py` 的参数声明）：
  如 E0.3 的单元素组占比。真实取值需在真实 ALFWorld 数据上实测

`exp/report.md` 的每条记录都带「数据边界」段落。

## 归属

本仓库**不重新分发** `verl-agent` 的任何代码。但有两处衍生：

- `deploy/run_single_gpu.sh` 是基于上游 `examples/gigpo_trainer/run_alfworld.sh`
  修改的（58 行配置中 40 行相同，其余为单卡适配）——属 Apache 2.0 的衍生作品
- `exp/notes/` 中的代码片段为分析评论用途

上游版权声明见 [`NOTICE`](NOTICE)。本仓库同样以 **Apache License 2.0** 发布。

`verl-agent`、`GiGPO` 等名称归其各自所有者，本仓库与上游团队及任何机构均无隶属或背书关系。
