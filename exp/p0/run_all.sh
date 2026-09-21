#!/usr/bin/env bash
# 跑完 P0 阶段可本地执行的三项实验（E0.1 是 tests/ 下的单测）。
#
# 用法：bash exp/p0/run_all.sh
# 依赖：项目根已建好 .venv（见 tests/requirements-local.txt）
#
# ⚠️ 全部在 CPU 上跑，用合成数据（见 common.py 的数据来源声明）。
#    这些实验验证的是**机制与量级**，不是训练效果。
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="$ROOT/.venv/bin/python"

[[ -x "$PY" ]] || { echo "未找到 $PY，请先建 .venv（见 tests/requirements-local.txt）" >&2; exit 1; }

echo "############ E0.1：core_gigpo 算法单测 ############"
"$PY" -m pytest "$ROOT/tests/" -q 2>&1 | tail -3

for e in e04_gamma_information e03_singleton_groups e02_adjust_batch; do
    echo
    echo "############ $e ############"
    # core_gigpo 会无条件打印 "Avg size of step-level group"（core_gigpo.py:330），滤掉降噪
    "$PY" "$ROOT/exp/p0/$e.py" 2>&1 | grep -v "^Avg size of step-level group"
done
