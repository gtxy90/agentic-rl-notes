# 异步 RL 稳定性 — 代码落点

> 证据分级：✅ 已核实｜🔍 已定位待验证｜⬜ 待办
> 基线 commit：`20bd331`（master, 2026-06-09）

## 0. 结论先行

**这个方向在本仓库里是未开采区，而且比表面看起来空得多。**

我最初判断"已合并最新 veRL 所以有 async rollout 基础设施"——**这个判断是错的**。事实是：

1. 仓库基于 **verl ~0.4.x**，没有新版 `verl/experimental/agent_loop/`、没有 `AgentLoop` / `AgentLoopWorker` / `AgentLoopManager` 那套结构。
2. **`master` 上 async 是死代码**：`ray_trainer.py:1074-1079` 的 async 分支被整段注释掉，实际永远走同步路径。
3. **staleness 处理完全没有**，是**结构性**的缺失（见 §2），不是"没调好"。
4. 作者的 `langfeng_async` 分支只做过**一次最小原型尝试**（单条 commit），随后精力就转向 README 和 memory manager 了。

→ 对本项目的含义：**这里没有现成 baseline 可比对，必须自己先立基线**，且立基线之前要先排除三个工程缺陷（§4），否则会把工程问题误判成算法问题。

## 1. 现状：仓库里到底有什么 ✅

### 1.1 真正的 agent loop 在哪

不是 `verl/experimental/agent_loop/`，而是 **`agent_system/multi_turn_rollout/rollout_loop.py`** 的 `TrajectoryCollector`。

关键：它跑在 **driver 进程里，是同步的环境交互循环**：

```python
# rollout_loop.py 内（vanilla_multi_turn_loop）
for _step in range(self.config.env.max_steps):          # :332
    ...
    batch_output = actor_rollout_wg.generate_sequences(batch_input_padded)   # :354
    ...
    next_obs, rewards, dones, infos = envs.step(text_actions)                # :365
```

### 1.2 async 三件套的角色

| 文件 | 层 | 角色 |
|---|---|---|
| `verl/workers/rollout/async_server.py` | 管理调度 | `AsyncLLMServerManager`（`:218`）起 N 个 ray actor；`ChatCompletionScheduler`（`:108`）做 HTTP 客户端 + 负载均衡 |
| `verl/workers/rollout/vllm_rollout/vllm_async_server.py` | vLLM 后端 | `AsyncvLLMServer`（`:106`），`ExternalRayDistributedExecutor`（`:39`）让 vLLM v1 借用 FSDP actor 当 worker |
| `verl/workers/rollout/sglang_rollout/async_sglang_server.py` | SGLang 后端 | `AsyncSglangServer`（`:29`），把请求透传给 SPMD workers |

### 1.3 为什么说它"名不副实" 🔍

chat scheduler 确实跑在**独立线程 + 独立 asyncio event loop** 里（`async_server.py:283-298`），但：

```python
# async_server.py:334
future = asyncio.run_coroutine_threadsafe(coro, loop)
... future.result()      # 同步阻塞 driver
```

**既不与 trainer 流水线重叠，也不与 `envs.step` 重叠。**

所以它的准确定位是：**"异步推理引擎 + 同步环境循环"**，不是异步 RL 基础设施。把它当成异步 RL 基线会严重高估其能力——它既没有 partial rollout，也没有跨 iteration 的轨迹续跑，更没有 generation 与环境步进的重叠。

### 1.4 开关与配置 🔍

`verl/trainer/config/ppo_trainer.yaml`：

| 行 | 配置 | 说明 |
|---|---|---|
| `:105` | `mode: sync # sync: LLM, async: AsyncLLM` | 总开关 |
| `:106` | `chat_scheduler: null` | **必填**，形如 `examples.ppo_trainer.naive_chat_scheduler.NaiveChatCompletionScheduler` |
| `:124` | `max_num_seqs: 1024` | 并发序列上限 |
| `:130` | `enable_chunked_prefill: True` | 与 `max_num_batched_tokens` 联动 |
| `:226` | `launch_reward_fn_async: False` | 仅 reward fn 的 CPU 异步，**与 rollout 无关** |

代码侧入口：`ray_trainer.py:903-909`（`if mode == "async": self.async_rollout_mode = True; self.async_rollout_manager = AsyncLLMServerManager(...)`）。

开启异步的完整配方（来自 `langfeng_async` 分支新增的 `run_webshop_async.sh`）：
- `export VLLM_USE_V1=1`
- `actor_rollout_ref.rollout.mode=async`
- `actor_rollout_ref.rollout.chat_scheduler=examples.gigpo_trainer.naive_chat_scheduler.NaiveChatCompletionScheduler`
- `actor_rollout_ref.rollout.free_cache_engine=False`（async 下必须）

**⚠️ 但见 §4-T3：`master` 上光设这些不够，async 分支是死的，必须打补丁。**

## 2. 缺口：staleness 是结构性不校正

### 2.1 关键词全空 ✅

我在全仓（`--include=*.py --include=*.yaml`）实测命中数：

| 关键词 | 命中 |
|---|---|
| `staleness` | **0** |
| `off_policy` | **0** |
| `partial_rollout` | **0** |
| `importance_ratio` | 5 —— **全部**在 `core_algos.py:541-548` 的 GSPO 里 |

那 5 处是 `log_seq_importance_ratio` 的 `torch.clamp(..., max=10.0)`，纯防溢出的数值稳定措施，**与 staleness 无关**。

### 2.2 根因：`rollout_log_probs` 是死数据 ✅

这是最关键的发现，也是理解"为什么 staleness 不校正"的钥匙。

**生产端**——注释直说了会抛弃它：

```python
# verl/workers/rollout/vllm_rollout/vllm_rollout_spmd.py:379
'rollout_log_probs': rollout_log_probs, # we will recompute old log prob with actor
# verl/workers/rollout/sglang_rollout/sglang_rollout.py:609 同
```

**消费端**——只拿去算三个 diagnostic metric 就丢掉（`ray_trainer.py:1153-1175`）：

```python
if "rollout_log_probs" in batch.batch.keys():
    rollout_old_log_probs = batch.batch["rollout_log_probs"]
    ...
    rollout_probs = torch.exp(rollout_old_log_probs)
    actor_probs   = torch.exp(actor_old_log_probs)
    rollout_probs_diff = torch.abs(rollout_probs - actor_probs)
    # → rollout_probs_diff_max / mean / std 三个 metric，完了
```

**而 PPO 真正用的 ratio 来自重算**（`ray_trainer.py:1141-1151`）：

```python
old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)   # 用「当前 actor」重算
batch = batch.union(old_log_prob)
```

`fsdp_workers.py:685-698` 的注释也印证：`# we should always recompute old_log_probs when it is HybridEngine`。

最终 `core_algos.py:473`：`ratio = torch.exp(negative_approx_kl)`，其中 `negative_approx_kl = log_prob - old_log_prob`。

**后果链**：用当前 actor 重算的 logprob 当 `old_log_prob` → **第一个内层 epoch 必然 `ratio ≡ 1`** → 产生轨迹的那个**陈旧策略的概率被彻底丢弃** → staleness 偏差**连 clip 都触发不了**。

`ray_trainer.py:1153-1175` 是**全仓唯一**同时握有 behavior-policy logprob 和 actor logprob 的地方——但它只做诊断。

### 2.3 缺口清单

- 没有 staleness 度量（没有记录"这条轨迹是第几个 policy version 产的"）。
- 没有 staleness 加权 / IS 校正。
- 没有 partial rollout。
- generation 与训练不重叠（`future.result()` 同步阻塞）。
- **同一 batch 内不同轨迹可能来自不同策略版本**——异步下这是必然，而当前代码对此零处理。

## 3. 改造点（按优先级）

| # | 位置 | 作用 |
|---|---|---|
| **(a)** | `ray_trainer.py:1141-1175` | **唯一咽喉**。behavior logprob 与 actor logprob 同时在此，现仅做诊断。把 `rollout_log_probs` 保留进 batch 并在此计算 staleness 权重 / IS ratio。所有训练路径（train + validate）都经过这里 |
| **(b)** | `core_algos.py:431`（GSPO 版 `:495`） | **加权生效点**。`ratio = torch.exp(negative_approx_kl)`（`:473`）就在函数体内，公式形态现成。truncated IS 自然落在这个 surrogate 上 |
| **(c)** | `dp_actor.py:377-406` | **配置透传点**。`clip_ratio{,_low,_high,_c}` 全部在此从 config 读出传给 `compute_policy_loss`。新增 `staleness_*` / `is_clip_*` 参数从这里灌入，与上游 PPO 惯例一致 |
| **(d)** | `rollout_loop.py`（`generate_sequences` 调用点，`:354` 附近） | **版本打标点**。这是唯一知道"生成时刻模型版本"的地方。要 per-token 粒度做 staleness 加权，必须把 policy version / rollout step id 写进 `batch.non_tensor_batch`（紧邻现有 `uid`(`:358`) / `traj_uid`(`:359`) 的写法）。异步下尤其关键 |

**最小可行组合**：`(a) + (d)` 做度量与打标，`(b) + (c)` 做加权。若只想验证"IS 能否救异步"，`(a)` + `(c)` 两个文件即可跑通。

## 4. 风险：三个必须先排除的陷阱

**做稳定性对比实验之前必须先处理这三项**，否则会把纯工程缺陷误判成"异步导致不稳定"。

### T1 —— 每个环境步清一次 prefix cache ✅（最严重）

`langfeng_async` 分支的 `wake_up()/sleep()` 被写在 `for _step in range(max_steps)` 的**循环体内**：

```python
# langfeng_async 分支，rollout_loop.py
if not self.async_rollout_mode:
    batch_output = actor_rollout_wg.generate_sequences(batch_input)
else:
    actor_rollout_wg.wake_up()
    batch_output = actor_rollout_wg.generate_sequences(batch_input)
    actor_rollout_wg.sleep()          # ← 在 for _step 循环体内
```

而 `sleep()` 会调 `engine.reset_prefix_cache()`（`vllm_async_server.py:248-251`）。

→ **每个环境步清空一次 prefix cache** → 巨大吞吐塌陷。若不排除此项，几乎必然把"缓存被清导致的性能劣化"误读成"异步训练不稳定"。

### T2 —— 超时与重试是关闭的 ✅

```python
# verl/workers/rollout/async_server.py:196
AsyncOpenAI(..., timeout=None, max_retries=0)
# :202
aiohttp.ClientTimeout(total=None)
```

**无限等待 + 零重试**。单次卡住的 vLLM 请求会**永久挂死整个 step**。

→ 这是异步稳定性研究的**第一道必修项**。任何 staleness 实验在补上超时/重试/熔断之前都无法可靠运行。这个前置工作本身即可写成方案里的贡献点（"现有异步管线缺乏故障隔离，我们补上并量化其对训练稳定性的影响"）。

### T3 —— `master` 上 async 是死代码 ✅

`ray_trainer.py:1074-1079` 整段注释掉：

```python
# if not self.async_rollout_mode:
#     gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)
# else:
#     self.async_rollout_manager.wake_up()
#     gen_batch_output = self.async_rollout_manager.generate_sequences(gen_batch)
#     self.async_rollout_manager.sleep()

################ agent-environment loop ###############
gen_batch_output = self.traj_collector.multi_turn_loop(
                        gen_batch=gen_batch,
                        actor_rollout_wg=self.actor_rollout_wg,   # ← 永远是同步的
                        envs=self.envs, is_train=True)
```

同理 `:749-754`（validate）也把 `actor_rollout_wg=self.actor_rollout_wg` 传进去。

而 async 模式下那是 `AsyncActorRolloutRefWorker`，其 `generate_sequences` 直接 `raise NotImplementedError`（`fsdp_workers.py:1454-1456`）。

→ **必须打上 `langfeng_async` 那 2 行补丁才能用**（`ray_trainer.py` 两处调用点改成 `actor_rollout_wg=self.async_rollout_manager if self.async_rollout_mode else self.actor_rollout_wg`）。

### T4 —— `NaiveChatCompletionScheduler` 是单轮退化版 🔍

`langfeng_async` 新增的 `examples/gigpo_trainer/naive_chat_scheduler.py`：
- **单轮** chat completion，不是多轮 agent 循环。
- 明确**没有 mask tool token**（源码留 `# TODO: mask out tools calling tokens?`）。
- 有 `# NOTE: we can call tools and resubmit chat completions here`（未实现）。
- 有 `# TODO: we may need to control max concurrent requests here, or it will harm prefix cache hit rate.`（未实现）。

→ 它**复现不出 GiGPO 的多轮轨迹结构**，基于它的 async 结果**与同步 GiGPO 不可比**。要用必须明确声明是"单轮退化版本"，不能拿来直接对比多轮 GiGPO 的指标。

## 5. `langfeng_async` 分支的实况 🔍

- `status: diverged`，**ahead 14 / behind 24**，共 14 个 commit，**只有 4 个文件不同**：

| status | 文件 | 改动量 |
|---|---|---|
| modified | `agent_system/multi_turn_rollout/rollout_loop.py` | +11 / -2 |
| added | `examples/gigpo_trainer/naive_chat_scheduler.py` | +120 |
| added | `examples/gigpo_trainer/run_webshop_async.sh` | +75 |
| modified | `verl/trainer/ppo/ray_trainer.py` | +2 / -2 |

- 14 条 commit 里**只有第一条 `2c54a7c9 async try` 真正做异步**，其余是 README / framework chart / memory manager / seed controller 等无关改动。

**判读**：作者只做过一次最小原型（替换推理后端 + 一个单轮 scheduler），**没有触碰 staleness、partial rollout、IS 校正**。这个方向在仓库里确实未被推进。

## 6. 对项目的含义

1. **没有现成 baseline**。要么自己实现一个最小异步基线，要么以"同步 + 人为注入 staleness"作为受控基线（后者更可控，也更容易讲清因果）。
2. **先补工程缺陷，再谈算法稳定性**。T2（超时/重试）和 T1（prefix cache）不解决，任何稳定性观测都不可信。
3. **T4 提醒**：如果项目要的是"多轮 agent 轨迹"的异步稳定性，那么现有的单轮 scheduler 无法直接用，需要先把它扩成多轮——这本身是工作量，应计入评估。
4. **(a) 号改造点是唯一的咽喉**，且改动量小、收益明确，适合作为第一个动手点。
