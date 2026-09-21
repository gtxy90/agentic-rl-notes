"""本地 CPU 测试的公共夹具。

设计要点：**不做 `pip install -e repo/`**。上游 setup.py 的 GPU 依赖组含
`flash-attn`，在 macOS arm64 上装不了；而 `verl/__init__.py` 的导入链
（→ `verl/protocol.py` → `verl/utils/torch_functional.py`）里 flash-attn
是 try/except 保护的，缺失时优雅降级。所以只把 `repo/` 注入 sys.path 即可。

边界：本目录只验证**算法数学正确性**，不验证训练效果。CPU 与 CUDA 的数值
不可互引，本地结论仅限"公式实现是否符合预期"。
"""

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent / "repo"

if not (REPO_ROOT / "gigpo" / "core_gigpo.py").exists():
    raise RuntimeError(
        f"未找到上游源码：{REPO_ROOT}/gigpo/core_gigpo.py\n"
        "请先克隆 verl-agent 到 repo/（见项目 README）。"
    )

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


@pytest.fixture(scope="session")
def core_gigpo():
    """GiGPO 信用分配核心模块。导入即验证了 CPU 依赖链是通的。"""
    from gigpo import core_gigpo as _core_gigpo

    return _core_gigpo


@pytest.fixture
def make_dataproto():
    """构造一个最小的 DataProto，只含 `compute_step_discounted_returns` 需要的字段。

    该函数读取 `non_tensor_batch` 的 rewards / traj_uid / active_masks，
    以及 `batch['input_ids']`（仅用于取 device）。
    """
    import numpy as np
    import torch
    from tensordict import TensorDict
    from verl import DataProto

    def _make(rewards, traj_uid, active_masks, response_len=3):
        rewards = np.asarray(rewards, dtype=np.float32)
        traj_uid = np.asarray(traj_uid, dtype=object)
        active_masks = np.asarray(active_masks, dtype=np.float32)

        bs = len(rewards)
        batch = TensorDict(
            {"input_ids": torch.zeros((bs, response_len), dtype=torch.long)},
            batch_size=[bs],
        )
        return DataProto(
            batch=batch,
            non_tensor_batch={
                "rewards": rewards,
                "traj_uid": traj_uid,
                "active_masks": active_masks,
            },
        )

    return _make
