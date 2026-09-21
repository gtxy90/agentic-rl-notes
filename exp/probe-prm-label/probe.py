#!/usr/bin/env python3
"""最小实验：文本版 ALFWorld 能否拿到稠密的过程标签？

背景
----
`notes/03-prm.md` §2.1 指出：`AlfredTWEnv`（默认 baseline 用的文本环境）向 TextWorld
只请求了 `won` / `admissible_commands` / `gamefile`（`alfred_tw_env.py:254`），
所以 `goal_condition_success_rate` 拿不到——它只存在于视觉版 `AlfredThorEnv`。

PRM 方向成立的前提是有**过程标签**。本探针回答：
**TextWorld 本身是否暴露"目标完成进度"信号？**

源码层的答案（已确认，见 `TextWorld/textworld/core.py` 的 `EnvInfos.__slots__`）
------------------------------------------------------------------
TextWorld 提供三个候选稠密信号，其文档字符串原文：

  - `intermediate_reward`："Reward (proxy) indicating if the player is making
    progress. This information changes from one step to another."
  - `win_facts`："Mutually exclusive sets of winning facts for each quest."
    （**不**随步变化 = 目标条件全集，分母）
  - `facts`："All the facts that are currently true about the world."
    （随步变化 = 当前成立的事实）
  - `score` / `max_score`：当前分 / 可达满分

⇒ `|facts ∩ win_facts| / |win_facts|` 就是目标条件满足率，即 THOR 版
  `goal_condition_success_rate`（`alfred_thor_env.py:178` 的 `pcs[0]/pcs[1]`）
  在文本环境下的等价物。**信号一直都在，只是没被请求。**

本脚本做**实测确认**：跑一个真实 TextWorld 游戏，逐步记录这些信号，
验证它们确实在终局前就变化、且目标条件满足率单调递增。

用法：
    .venv/bin/python probe.py
"""

from __future__ import annotations

import inspect
import sys
import traceback
from pathlib import Path
from typing import Any

SEP = "=" * 78
REQUEST = dict(won=True, score=True, max_score=True, intermediate_reward=True,
               facts=True, win_facts=True, fail_facts=True, moves=True)


def h(title: str) -> None:
    print(f"\n{SEP}\n{title}\n{SEP}")


# --------------------------------------------------------------------------- #
def probe_env_infos() -> bool:
    h("探针 1：EnvInfos 支持的字段")
    import textworld

    print(f"textworld 版本：{getattr(textworld, '__version__', '未知')}")

    fields = list(getattr(textworld.EnvInfos, "__slots__", []))
    if not fields:
        print("⚠️  无法读取 __slots__，尝试签名反射")
        try:
            fields = [p for p in inspect.signature(textworld.EnvInfos.__init__).parameters if p != "self"]
        except (TypeError, ValueError):
            fields = []

    print(f"共 {len(fields)} 个字段：")
    for f in sorted(fields):
        mark = "  ← 候选稠密信号" if f in ("facts", "win_facts", "fail_facts",
                                          "intermediate_reward", "score", "max_score") else ""
        print(f"    - {f}{mark}")

    missing = [k for k in REQUEST if k not in fields]
    if missing:
        print(f"\n⚠️  本版本缺少：{missing}")
    return not missing


# --------------------------------------------------------------------------- #
def make_game() -> Any:
    """生成一个**多目标**游戏。

    `GameOptions.__init__` 无参，字段靠属性赋值（见 textworld/generator/game.py:1129）。
    `nb_parallel_quests=2` 生成两个不相交的目标 → 多组 win_facts，
    对应 ALFWorld 的多条件目标。

    注意 `textworld.make` 返回 `(game_file, game)` 元组。
    """
    import shutil

    import textworld
    from textworld import GameOptions

    # TextWorld 按文件名缓存游戏：同名但结构不同会直接 AssertionError。
    # 每次清空输出目录，避免上一次的残留干扰。
    out_dir = Path(__file__).parent / "tw_games"
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # rooms/objects 取大一些，才能生成"目标含多个命题"的游戏——
    # 目标命题数越多，满足率的分级越明显（实测 3 命题目标给出 2/3 → 3/3）。
    def opts(n_quests: int, tag: str, n_rooms: int = 4, n_objects: int = 12):
        o = GameOptions()
        o.nb_parallel_quests = n_quests
        o.nb_rooms = n_rooms
        o.nb_objects = n_objects
        o.path = str(out_dir / f"probe_{tag}.z8")   # 每次不同文件名
        return o

    for desc, n, tag in [("2 个并行目标", 2, "q2"), ("1 个目标（兜底）", 1, "q1")]:
        try:
            game_file, game = textworld.make(opts(n, tag))
            print(f"✅ 生成游戏成功（{desc}）：{game_file}")
            return game_file, game
        except Exception as e:  # noqa: BLE001
            print(f"   {desc} 失败：{type(e).__name__}: {e}")
    raise RuntimeError("游戏生成失败")


def get_walkthrough(game: Any) -> list[str]:
    h("探针 2：游戏结构与标准解")

    meta = getattr(game, "metadata", {}) or {}
    print(f"metadata 键：{sorted(meta.keys())}")

    # `walkthrough` 是 Game 的 property：先查 metadata，缺失时**自动从 quests 推导**
    # （见 textworld/generator/game.py:625-637）。比直接读 metadata 更可靠。
    wt: list[str] = []
    try:
        wt = list(game.walkthrough or [])
    except Exception as e:  # noqa: BLE001
        print(f"⚠️  game.walkthrough 取值失败：{type(e).__name__}: {e}")
        wt = list(meta.get("walkthrough") or [])

    if wt:
        print(f"\n✅ 得到 walkthrough，{len(wt)} 步：")
        for i, c in enumerate(wt):
            print(f"    {i:>2}. {c}")
    else:
        print("\n⚠️  无 walkthrough")

    # 顺带看 objective，确认目标是否多条件
    obj = getattr(game, "objective", None) or meta.get("objective")
    if obj:
        print(f"\n目标（objective）：{obj}")
    return wt


# --------------------------------------------------------------------------- #
def scalar(v: Any) -> Any:
    """TextWorld 某些字段返回 list（batch 维度），这里压成标量。"""
    if isinstance(v, (list, tuple)):
        if len(v) == 1:
            return scalar(v[0])
        nums = [x for x in v if isinstance(x, (int, float))]
        return sum(nums) if nums else None
    return v


def _key(x: Any) -> Any:
    """把 fact 归一成可哈希的形式。

    TextWorld 返回的 fact 是嵌套 list（如 `["in", "sponge", "parlor"]`），
    不可直接放进 set。
    """
    if isinstance(x, (list, tuple)):
        return tuple(_key(e) for e in x)
    if isinstance(x, dict):
        return tuple(sorted((k, _key(v)) for k, v in x.items()))
    if isinstance(x, set):
        return tuple(sorted(_key(e) for e in x))
    return x


def goal_rate(info: dict) -> tuple[int, int, float] | None:
    """目标命题满足率 = |facts ∩ 目标命题| / |目标命题|。

    ⚠️ 嵌套层级是踩过的坑，实测结构如下（TextWorld 1.7.0，batch_size=1）：

        facts      : list[bs] -> facts[0]      = list[Proposition]    世界当前全部事实
        win_facts  : list[bs] -> win_facts[0]  = list[quest]
                                  quest        = list[condition_group]
                                  group        = list[Proposition]   **合取**

    进度粒度是 **group 内的命题**——实测一个 3 命题目标给出 2/3 → 3/3 的分级变化。

    注意 `facts` 里是 `Proposition` 对象（`textworld.logic`），不是普通 list；
    `_key` 仅为兜底处理某些版本返回嵌套 list 的情况。
    """
    facts = info.get("facts")
    win = info.get("win_facts")
    if facts is None or win is None:
        return None

    # 去掉 batch 维度
    facts = facts[0] if isinstance(facts, (list, tuple)) and len(facts) == 1 else facts
    win = win[0] if isinstance(win, (list, tuple)) and len(win) == 1 else win

    facts = {_key(f) for f in facts}
    quests = win if isinstance(win, (list, tuple)) else [win]

    best: tuple[int, int, float] | None = None
    for quest in quests:
        groups = quest if isinstance(quest, (list, tuple)) else [quest]
        for g in groups:
            g = {_key(f) for f in g}
            if not g:
                continue
            hit = len(facts & g)
            rate = hit / len(g)
            if best is None or rate > best[2]:
                best = (hit, len(g), rate)
    return best


def run_episode(game_file: str, walkthrough: list[str]) -> list[dict]:
    h("探针 3：逐步执行，记录信号")

    import textworld
    import textworld.gym

    infos = textworld.EnvInfos(**REQUEST)
    # register_games 接受游戏文件路径（ALFWorld 的 alfred_tw_env.py:273 也是传路径）
    env_id = textworld.gym.register_games([game_file], infos, batch_size=1,
                                          asynchronous=False, max_episode_steps=200)
    env = textworld.gym.make(env_id)
    obs, info = env.reset()

    def unpack(x: Any) -> dict:
        return x[0] if isinstance(x, (list, tuple)) else x

    print(f"初始 info 键：{sorted(unpack(info).keys())}\n")
    print(f"{'步':>3} {'won':>5} {'score':>6} {'max':>5} {'moves':>6} "
          f"{'inter_r':>8} {'facts':>6} {'goal达成':>9}  action")
    print("-" * 78)

    trace: list[dict] = []

    def show(step: int, d: dict, action: str) -> None:
        trace.append(d)
        gr = goal_rate(d)
        grs = "     n/a" if gr is None else f"{gr[0]}/{gr[1]}={gr[2]:.2f}"
        ir = scalar(d.get("intermediate_reward"))
        irs = "     n/a" if ir is None else f"{ir:>8.3f}"
        nf = len(d.get("facts") or [])
        print(f"{step:>3} {str(scalar(d.get('won'))):>5} {str(scalar(d.get('score'))):>6} "
              f"{str(scalar(d.get('max_score'))):>5} {str(scalar(d.get('moves'))):>6} "
              f"{irs} {nf:>6} {grs:>9}  {action[:28]}")

    show(0, unpack(info), "(reset)")

    for i, cmd in enumerate(walkthrough, start=1):
        obs, _rew, done, info = env.step([cmd])
        d = unpack(info)
        show(i, d, cmd)
        if unpack(done):
            print("     → 终止")
            break

    env.close()
    return trace


# --------------------------------------------------------------------------- #
def analyze(trace: list[dict]) -> bool:
    h("分析：这些信号稠密吗？")

    if len(trace) < 2:
        print("❌ 轨迹太短，无法判断")
        return False

    print(f"{'信号':<22} {'不同值数':>8}  起 → 止      终局前变化?")
    print("-" * 78)

    dense: list[str] = []
    for k in ("score", "intermediate_reward", "moves"):
        s = [d.get(k) for d in trace]
        if any(v is None for v in s):
            continue
        distinct = len(set(map(str, s)))
        changed_early = any(str(v) != str(s[0]) for v in s[:-1])
        tag = "✅ 是" if changed_early else "  否"
        if changed_early:
            dense.append(k)
        print(f"{k:<22} {distinct:>8}  {str(s[0])[:9]:>9} → {str(s[-1])[:9]:<9} {tag}")

    rates = [(goal_rate(d) or (0, 0, 0.0))[2] for d in trace]
    distinct_r = len({round(x, 6) for x in rates})
    # 关键判据：在**终局之前**就出现过变化，而不是只看取值个数
    changed_early_r = any(abs(rates[i] - rates[0]) > 1e-9 for i in range(len(rates) - 1))
    rising = all(rates[i] <= rates[i + 1] + 1e-9 for i in range(len(rates) - 1))
    print(f"{'goal达成率(算得)':<20} {distinct_r:>8}  {rates[0]:>9.3f} → {rates[-1]:<9.3f} "
          f"{'✅ 是' if changed_early_r else '  否'}")

    print("\n" + "-" * 78)
    ok = bool(dense) or changed_early_r
    if ok:
        print("✅ 结论：TextWorld **确实**提供终局前变化的过程信号。")
        if changed_early_r:
            print(f"   其中目标命题满足率 {rates[0]:.2f} → {rates[-1]:.2f}"
                  f"（{distinct_r} 个不同值{'，单调不减' if rising else '，注意：非单调'}）"
                  "，粒度是目标内部的单个命题。")
        print("   ⇒ PRM 有现成标签来源，无需采样或人工标注。")
        print("   ⚠️ 但注意：实测 `score` 只在**整个 quest 完成**时跳变（单 quest 多步链式任务中")
        print("      它全程为 0），`intermediate_reward` 是 0/1 而非分级。**真正分级的只有")
        print("      按命题算的满足率**——这正是要在 ALFWorld 上复现验证的那一个。")
    else:
        print("❌ 结论：未发现终局前变化的信号。")
    return ok


# --------------------------------------------------------------------------- #
def main() -> int:
    print(SEP)
    print("最小实验：文本版 ALFWorld 的稠密过程标签可行性")
    print(SEP)

    try:
        fields_ok = probe_env_infos()
        if not fields_ok:
            print("\n⚠️  字段不全，后续结果可能不完整（继续尝试）")

        game_file, game = make_game()
        wt = get_walkthrough(game)
        if not wt:
            print("\n⚠️  无 walkthrough，无法自动求解。探针止步于此。")
            return 1

        trace = run_episode(game_file, wt)
        ok = analyze(trace)
    except Exception:  # noqa: BLE001
        print("\n❌ 探针执行失败：")
        traceback.print_exc()
        return 1

    h("结论边界（必读）")
    print("""
1. 本探针验证的是 **TextWorld 引擎机制**，不是 ALFWorld 的真实游戏。
   ALFWorld 的游戏由 `json_2.1.1/*` 的 PDDL 问题实例生成，不在本仓库里
   （`configs/config_tw.yaml` 指向 $ALFWORLD_DATA，需 `alfworld-download`）。
2. 可迁移的理由：ALFWorld 的文本游戏同样是 TextWorld PDDL 游戏，目标同样是
   多谓词合取（见 `alfworld/data/alfred.pddl`）。ALFWorld 论文本身也用
   goal-condition 作为评估指标，说明其目标确实是多条件的。
3. **迁移性未经实测**。坐实需要在有 ALFWORLD_DATA 的机器上对真实游戏复现本探针。

若要在 ALFWorld 上落地，改动很小（共 2 处）：
  a. `alfred_tw_env.py:254` 的 request_infos 加入
     `intermediate_reward=True, facts=True, win_facts=True`
  b. `envs.py:48` 的 compute_reward 里把它取出来，参考 `multi_modal` 分支写法
  然后按 `notes/03-prm.md` §3 方案 1 注入 `ray_trainer.py:1116`。

⚠️ 注意 `notes/03-prm.md` R2b：不要就地改 `rewards`，会同时污染 outcome 奖励，
   必须新开数组（如 `non_tensor_batch['prm_rewards']`）。
""")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
