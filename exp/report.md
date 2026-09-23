# 实验报告 — verl-agent 项目

> **本文件是实验结果的唯一归档处。** 每做完一项实验就往这里追加一条，不散落在别处。
> 实验计划/设计在 `design.md`（**做什么**），本文件记录**做出了什么**。
>
> 证据分级：✅ 已核实（实测数字）｜🔍 已定位待验证（有依据未实测）｜⬜ 待办
> 数据边界：本机无 CUDA，所有结果来自 **CPU + 合成数据**或**引擎级实测**，
> 不代表训练效果。跨到远程 CUDA 的结论一律标注。
>
> 复现：`bash exp/p0/run_all.sh`（P0 四项）；探针见 `probe-prm-label/README.md`。

---

## 索引

| 编号 | 实验 | 日期 | 状态 | 一句话结论 |
|---|---|---|---|---|
| [EP-0](#ep-0) | PRM 标签可行性探针 | 2026-09-20 | ✅ | 分级过程信号存在，改 2 行即可拿到 |
| [E0.1](#e01) | `core_gigpo` 算法单测夹具 | 2026-09-20 | ✅ | 28 项全绿，可回归 |
| [E0.4](#e04) | `gamma` 对 step 信号信息量的影响 | 2026-09-20 | ✅ | **立论成立**：gamma=1.0 时位置信息零增量 |
| [E0.3](#e03) | 单元素组抹平占比 | 2026-09-20 | ✅ | **R2 量级很大**：74% 的步优势被抹平 |
| [E0.2](#e02) | `adjust_batch` 复制行污染 | 2026-09-20 | ✅ | **R1 影响很小**（<1%），优先级低 |
| [EP-1](#ep-1) | 远程环境部署（AutoDL） | 2026-09-21 | ✅ | 环境就绪；**踩了 5 个坑，都是同一个根因** |

---

<a id="ep-0"></a>
## EP-0 — PRM 标签可行性探针 ✅

**日期**：2026-09-20　**位置**：`probe-prm-label/`（TextWorld 1.7.0，本地实测）

**问题**：PRM 方向成立的前提是有过程标签。文本版 ALFWorld（默认 baseline）拿得到吗？

**结果**：

| 候选信号 | 实测行为 | 可用性 |
|---|---|---|
| `score` / `max_score` | 统计 **quest 完成数**；单 quest 多步链式任务中全程为 0 | ❌ |
| `intermediate_reward` | 0/1 二值 | ❌ |
| `\|facts ∩ 目标命题\| / \|目标命题\|` | **2/3→3/3、3/4→4/4，单调且在终局前变化** | ✅ |

**关键数字**：满足率在 reset 后即出现分级（基线 0.50–0.75，因目标含静态命题），
在终局前上升到 1.00。

**结论**：分级过程信号**存在且现成**（`facts` + `win_facts` 两个字段），
`AlfredTWEnv` 只是没请求（`alfred_tw_env.py:254`）。落地改动 2 处。
⚠️ 唯一剩下验证：ALFWorld 真实目标是否含多个命题（需 `ALFWORLD_DATA`，远程做）。

**踩坑记录**（避免重犯）：
- 我一度判断"jericho 无 macOS wheel → textworld 装不了"——**错**。它能装，
  只是构建期要下 22MB 的 Inform 7 包，本机 42 KB/s 耗时约 9 分钟。**是慢，不是不兼容。**
- `|facts ∩ win_facts|` 漏掉 batch 维会恒为空集，白跑一次。

---

<a id="e01"></a>
## E0.1 — `core_gigpo` 算法单测夹具 ✅

**日期**：2026-09-20　**位置**：`tests/test_core_gigpo.py`

**结果**：28 项断言，**全绿**（`.venv/bin/python -m pytest tests/ -q`）。
覆盖 `core_gigpo.py` 全部 7 个函数：

| 函数 | 覆盖点 |
|---|---|
| `to_hashable` | 各类型归一化；**ndarray 元素保留为 numpy 标量**（已固定为测试） |
| `are_similar` | 阈值语义、非字符串 raise |
| `build_step_group` | 精确 vs 相似度两条聚类路径、跨 task 不共享 |
| `episode_norm_reward` | `mean_norm`/`mean_std_norm`、单元素组保留原始分 |
| `step_norm_reward` | **单元素组优势恒为 0**（R2 的直接证据） |
| `compute_step_discounted_returns` | 折扣累计、`gamma=1.0` 常数性、跨轨迹不串味 |
| `compute_gigpo_outcome_advantage` | 端到端融合、`step_advantage_w` 只乘 step 项 |

**结论**：信用分配的数学已从框架剥离、可本地回归。后续改算法（接 PRM、改加权）
能立即验证，不必上远程等结果。

---

<a id="e04"></a>
## E0.4 — `gamma` 对 step 级信号信息量的影响 ✅

**日期**：2026-09-20　**位置**：`exp/p0/e04_gamma_information.py`　**对应**：`design.md` §2 立论

**问题**：`gamma=1.0` 下 GiGPO 的 "step-level" 是否只是形式上的？

**结果**：

| 判据 | `gamma=1.0` | `gamma=0.95` |
|---|---|---|
| 轨迹内 `step_rewards` 标准差为 0 的轨迹占比 | **100.0%** | 49.2% |
| `R²(step_rewards ~ 轨迹级 outcome)` | **1.000000** | 0.8668 |
| 位置（距终点步数）带来的**增量** R² | **+0.000000** | +0.0149 |
| `corr(该轨迹的 step 优势, 该轨迹的 outcome)` | **+0.9704** | — |

**结论**：✅ **立论成立**。`gamma=1.0` 时 `step_rewards[t] ≡ R`（与 t 无关），
位置信息**零增量**，step 级优势只是"把 outcome 在访问过同一状态的轨迹子集里
重新归一化"。

`gamma=0.95` 确实带来了额外信息，但那是**位置的函数**（"离结束还有几步"），
**不是过程质量的函数**（"这一步做得好不好"）。

⚠️ 数据边界：`gamma=1.0 ⇒ 常数` 是**代数必然**，与数据无关；其余数字来自合成数据。

**踩过的坑**：判据 2 最初用**每步** `outcome` 做自变量，而 `step_rewards` 是**轨迹级**
常数，两者不是一回事 → R²=0.04 的假象。改用轨迹级 outcome 后 R²=1.000000。

---

<a id="e03"></a>
## E0.3 — 单元素组抹平占比 ✅

**日期**：2026-09-20　**位置**：`exp/p0/e03_singleton_groups.py`　**对应**：`notes/01` R2

**问题**：`step_norm_reward` 对单元素组给出恒为 0 的优势（`core_gigpo.py:365-369`）。
长轨迹下这种情况占多大比例？

**结果**（`share_prob=0.15`）：

| 指标 | step 级 | episode 级 |
|---|---|---|
| **单元素组内的步占比** | **74.3%** | 0%（每组恒 8 条） |
| 平均组大小 | 1.17 | 8 |
| **优势非零的步占比** | **12.1%** | **100.0%** |

**敏感性**（状态共享概率 → 单元素步占比）：

| `share_prob` | 0.00 | 0.05 | 0.15 | 0.30 | 0.50 | 0.80 |
|---|---|---|---|---|---|---|
| 单元素步占比 | 96.4% | 88.0% | **74.3%** | 56.6% | 33.5% | 10.7% |
| step 优势非零占比 | 3.2% | 6.4% | 14.2% | 23.2% | 36.6% | 60.3% |

**`enable_similarity` 无效**：75.4% → 71.2%（`share_prob=0.15`）。
原因：ALFWorld 观测的差异是**内容**差异（不同房间/物品），
不是拼写差异，字符级 `SequenceMatcher` 合并不了。要用需换 embedding 度量。

**结论**：✅ **R2 成立且量级很大**。即便把状态共享概率拉到极不现实的 0.8，
单元素步占比仍有 10.7%。要显著下降需要 rollout 高度同质——而长轨迹任务恰恰不是。

⇒ 这为 PRM 提供了明确动机：**用不依赖"同状态重复访问"的过程信号补上这一维**。

⚠️ 数据边界：`share_prob` 是估计值，真实值需在 ALFWorld 上实测。
但**结构性差异与取值无关**：episode 组成员数由 `env.rollout.n` 固定为 8，
step 组成员数取决于轨迹重合度，两者不在一个量级。

---

<a id="e02"></a>
## E0.2 — `adjust_batch` 复制行对分组统计的污染 ✅

**日期**：2026-09-20　**位置**：`exp/p0/e02_adjust_batch.py`　**对应**：`notes/01` R1

**问题**：`ray_trainer.py:1118` 在 `compute_advantage`（`:1221`）**之前**调用
`adjust_batch(mode="copy")`，随机复制若干行补齐整除。复制行带相同的
`uid`/`traj_uid`/`anchor_obs`，会进入分组统计。

**结果**（`size_divisor=64`，bs=4064）：

| 指标 | 实测 |
|---|---|
| `to_add` | 平均 32 行（**0.8%**） |
| 对原有行优势的影响 \|Δ\| | 均值 ~0.0005，最大 ~0.005 |
| 被"提升"为多元素组的步 | 25 个（占单元素步的 **0.83%**） |

**结论**：⚠️ **机制成立但影响很小**，与 R2（74%）**不在一个量级**。

原因：bs 是**几千**量级，而 `to_add` 上限只有 63。

⇒ **风险排序：R1 远低于 R2。补这一项优先级低**，不必在项目早期投入。
唯一隐患是 `np.random.choice` 逐轮选不同行 → 额外噪声而非稳定偏置，排查成本高于量级。
若要严格控制变量，可把 `adjust_batch` 挪到 `compute_advantage` 之后
（与 HGPO 一致，`recipe/hgpo/hgpo_ray_trainer.py:1219`），但**不是优先事项**。

⚠️ 数据边界：结论对该参数不敏感——复制占比由 bs 与 `SIZE_DIVISOR` 的整除关系决定，
与数据分布无关。

---

## 阶段性结论（2026-09-20，P0 完成）

**P0 四项全部完成。** 对项目的直接含义：

1. **E0.4 把立论坐实了**：GiGPO 的 step 级信号在没有外部过程奖励时，
   确实**无法提供过程信用**（gamma=1.0 时位置信息零增量）。这条是项目的出发点，
   现在有精确实测数字支撑（R² 恰好 1.000000）。

2. **E0.3 给出了 PRM 的量化动机**：74% 的步优势被单元素组抹平，
   这不是调参能解决的**结构性**问题。PRM 要补的正是"不依赖同状态重复访问"的信号。

3. **风险优先级重排**：原先把 R1（`adjust_batch`）和 R2（单元素组）并列，
   实测后 **R1 影响 <1%、R2 影响 74%** —— 精力应放在 R2 与 PRM，不是 R1。

**下一步**：P1（远程跑通 baseline，阻塞：算力）。P0 的四项不依赖算力，已清空。

---

<a id="ep-1"></a>
## EP-1 — 远程环境部署（AutoDL / RTX 4090D）✅

**日期**：2026-09-21　**性质**：部署记录（非算法实验）
**结论**：✅ **环境就绪**。`check_env.py --no-gpu` 全绿（3 项警告均为预期或可选）。

### 环境规格（实测）

| 项 | 值 | 备注 |
|---|---|---|
| GPU | RTX 4090D, 24GB | 单卡 |
| **容器内存限额** | **2 GB**（cgroup） | ⚠️ **关键约束，见坑 4** |
| 宿主机内存 | 503 GB | `free` 显示的是这个，**会误导** |
| CPU 核数 | 128 | 与内存限额叠加成灾 |
| CUDA / gcc | 12.8 / 11.4.0 | |
| 系统盘 | 30 GB（overlay） | 只能放系统 |
| 数据盘 | 50 GB（`/root/autodl-tmp`） | **模型/环境/数据都放这里** |
| 安装模式 | **无卡模式** | 成本比 GPU 档低一个数量级，且安装不需要 GPU |

### 安装结果

| 组件 | 版本 | 方式 |
|---|---|---|
| torch | 2.8.0+cu128 | vllm 依赖 |
| vllm | 0.11.0 | aliyun→**清华**源 |
| **flash-attn** | **2.8.3** | **预编译 wheel**（非编译，见坑 4） |
| verl | editable | `pip install -e .` |
| textworld | 1.7.0 | 不带 `[pddl]` extra（见坑 5） |
| alfworld | 0.4.2 | `--no-deps`（见坑 5） |
| ALFWorld 数据 | **4027 个 game 文件**, 2.2G | `alfworld-download -f` 成功 |

**磁盘账单**（数据盘 50G）：环境 12G + ALFWorld 数据 2.2G + HF 缓存 115M
+ pip 缓存 5.1G（可清）= **20G / 50G**。

### 五个坑 —— **其中四个是同一个根因**

#### 坑 1：全局开学术加速会拖慢 pip

AutoDL 的 `/etc/network_turbo` 启用后自带的警告原文：

> 开启加速后对访问其他资源如 pip 源等会**更慢**

而安装的绝大部分流量走 PyPI/conda。
**处置**：不要全局 source，只在需要访问 Google Drive（ALFWorld 数据）时临时启用。

#### 坑 2：conda 默认 channel 在国内超时

默认 channel 里的 `defaults` 指向 `repo.anaconda.com`，实测长时间卡在
`ReadTimeoutError ... /pkgs/r/linux-64/repodata.json.zst`。
**处置**：`conda create --override-channels -c <清华源>`。

#### 坑 3：无卡模式下 `nvidia-smi` 返回 0 但输出为空

`nvidia-smi -L` 与 `--query-gpu` 在无卡模式下**退出码为 0、输出为空**。
只看退出码会误判成"有 GPU"，进而漏设 `TORCH_CUDA_ARCH_LIST`，
**要等到十几分钟后 flash-attn 编译时才炸**。
**处置**：判据改为**看输出是否为空**，不看退出码。

#### 坑 4：容器内存被 cgroup 限到 2G，而 `free` 显示 503G —— 编译必然 OOM

这是**最贵的一个坑**，flash-attn 与 ALFWorld 都栽在上面。

现象：
```
gcc: fatal error: Killed signal terminated program cc1plus
setup_remote.sh: line 256:  4712 Killed    pip install alfworld
```

根因是三个因素叠加：
1. `free` 读的是**宿主机**内存（503G），真实限额在 `/sys/fs/cgroup/memory.max` = **2G**
2. CPU 有 **128 核**，构建系统默认按 `nproc` 全开并行
3. 每个 `cc1plus` 吃 1–2GB → 瞬间爆掉

而且 flash-attn 2.7.4 的构建脚本会为 **5 个架构**（sm80/sm90/compute_100/compute_120）
编译 **700+ 个 CUDA 文件**，**设了 `TORCH_CUDA_ARCH_LIST=8.9` 也不生效**——
即便单线程也要数小时。限制 `MAX_JOBS=2` 对 flash-attn 和 fast-downward **都救不回来**。

**处置**：
- flash-attn → **装预编译 wheel**。关键是要三者对上：
  `torch 次版本 × python tag × cxx11abi`。注意：
  - 上游钉的 `2.7.4.post1` **没有 torch2.8 的 wheel**（只到 torch2.7），
    而 vllm 0.11.0 装的是 torch 2.8 ⇒ 必须用更新的 **2.8.3**（**与上游的版本偏离，须记录**）
  - `cxx11abi` 要与本机 torch 一致（`torch._C._GLIBCXX_USE_CXX11_ABI`）
  - ⚠️ **wheel 文件名不能改** —— 里面的 python/abi/platform 标签是 pip 判断兼容性的依据，
    改名会得到 `not a supported wheel on this platform`

#### 坑 5：`pip install alfworld` 会拉入一个装不上的 C++ 依赖

`alfworld → textworld[pddl] → fast_downward_textworld`（C++ PDDL 规划器，**只有 sdist**）
⇒ 与坑 4 同样的 OOM。

**但它根本不需要**（静态分析结论）：

| 检查 | 结果 |
|---|---|
| vendor 版 alfworld 里 `import fast_downward` | **0 处** |
| 同上，`import pddl` | **0 处** |
| `configs/config_tw.yaml` 的 `expert_type` | `handcoded`（非 `planner`） |

**处置**：拆开装 —— `pip install "textworld>=1.6.1"`（不带 extra）
+ `pip install --no-deps alfworld`（只为拿 `alfworld-download` CLI）。

#### 附带：pip 源选错会慢 260 倍

平台默认 `/etc/pip.conf` 指向阿里云，实测：

| 源 | 速度 |
|---|---|
| 阿里云（平台默认） | **25 KB/s** ← 438MB 的 vllm wheel 要下 5 小时 |
| 清华 | **6633 KB/s** |
| 中科大 / 腾讯云 | 取不到 URL |

换源后同样文件约 70 秒下完。**这个不换，整个安装会被卡死在一个包上。**

### ⚠️ 一处待实测的推断（重要）

跳过 `fast_downward` 是**基于静态分析的推断，不是实测**。上面的证据
（vendor 代码 0 处 import + 配置用 handcoded）很强，但**只有在 GPU 上真跑起来才能确认**。

⇒ **P1 首次跑 baseline 时，若 ALFWorld 报与 planner / PDDL 相关的错误，就是这里。**
届时回退方案：在有更大内存的实例上编译 fast_downward，或改用
`expert_type: handcoded` 之外的路径前先确认。

### 对后续实验的影响

1. **单卡配置**已写入 `deploy/run_single_gpu.sh`（含 `--smoke` 冒烟档）——
   但**尚未实测**，首次务必冒烟并盯显存。
2. `size_divisor` 从 64 降到 8（单卡），R1 的影响进一步减小（见 E0.2）。
3. 环境装在 `/root/autodl-tmp`，**每次登录须 `source deploy/env.sh` 恢复环境变量**，
   否则 conda 找不到环境、HF 会往系统盘重下。

