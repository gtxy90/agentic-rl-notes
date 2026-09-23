#!/usr/bin/env python3
"""verl-agent 环境自检。

在远程 CUDA 机器上跑，逐项确认依赖是否齐备，并给出**可执行的结论**。
在本地（无 CUDA）跑应正确报出"不可训练"，而不是误报通过。

用法：
    python deploy/check_env.py              # 完整检查
    python deploy/check_env.py --local      # 本地模式：预期会报无 CUDA，用于验证脚本本身
    python deploy/check_env.py --repo PATH  # 指定 repo 路径（默认 ../repo）
"""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata as md
import os
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path

OK, WARN, FAIL = "✅", "⚠️ ", "❌"


@dataclass
class Report:
    results: list[tuple[str, str, str]] = field(default_factory=list)

    def add(self, status: str, item: str, detail: str = "") -> None:
        self.results.append((status, item, detail))

    def has_failures(self) -> bool:
        return any(s == FAIL for s, _, _ in self.results)

    def render(self) -> None:
        width = max(len(i) for _, i, _ in self.results) + 2
        print()
        for status, item, detail in self.results:
            print(f"  {status}  {item:<{width}}{detail}")
        print()


def _version(pkg: str) -> str | None:
    try:
        return md.version(pkg)
    except md.PackageNotFoundError:
        return None


def check_python(rep: Report) -> None:
    v = sys.version_info
    if v >= (3, 12):
        rep.add(OK, "Python 版本", f"{v.major}.{v.minor}.{v.micro}")
    elif v >= (3, 10):
        rep.add(WARN, "Python 版本", f"{v.major}.{v.minor} —— 上游推荐 3.12")
    else:
        rep.add(FAIL, "Python 版本", f"{v.major}.{v.minor} —— 需要 >=3.10")


def check_torch(rep: Report, expect_no_gpu: bool = False) -> None:
    try:
        import torch
    except ImportError:
        rep.add(FAIL, "torch", "未安装")
        return

    rep.add(OK, "torch", torch.__version__)

    if not torch.cuda.is_available():
        if expect_no_gpu:
            # 无卡模式（如 AutoDL 的无卡开机）下这是**预期状态** —— 装环境不需要 GPU，
            # 只有训练才需要。降级为 ⚠️，避免把整体结论判成"未就绪"。
            rep.add(WARN, "CUDA",
                    "不可用（--no-gpu 模式，预期如此）—— 环境可用于安装/自检，训练需 GPU")
        else:
            rep.add(FAIL, "CUDA", "不可用 —— 训练无法在本机进行")
        return

    n = torch.cuda.device_count()
    rep.add(OK if n >= 2 else WARN, "GPU 数量", f"{n} 张" + ("" if n >= 2 else " —— 默认配置需 >=2"))
    for i in range(n):
        p = torch.cuda.get_device_properties(i)
        rep.add(OK, f"  GPU {i}", f"{p.name}, {p.total_memory / 1024**3:.0f} GiB, cc{p.major}.{p.minor}")


def check_packages(rep: Report) -> None:
    # (包名, 期望约束, 是否必需)
    required = [
        ("vllm", "==0.11.0", True),
        ("flash-attn", "==2.7.4.post1", True),
        ("transformers", "<=4.57.3", True),
        ("tensordict", ">=0.8.0,<=0.10.0,!=0.9.0", True),
        ("ray", ">=2.41.0,<=2.50.0", True),
        ("torchdata", None, True),
        ("peft", None, True),
        ("wandb", None, False),
        ("sglang", "==0.5.5（可选后端）", False),
    ]
    for pkg, want, need in required:
        v = _version(pkg)
        if v is None:
            rep.add(FAIL if need else WARN, pkg, "未安装" + ("" if need else "（可选）"))
        else:
            rep.add(OK, pkg, f"{v}" + (f"  （期望 {want}）" if want else ""))


def check_imports(rep: Report) -> None:
    """真正 import 一遍——版本号对不代表能导入。

    flash-attn 的三方包名是 flash_attn；verl 的导入链里它是 try/except 保护的，
    所以缺失不会在这里报错，而是体现为 FLAH_ATTN_CROSS_ENTROPY_LOSS_AVAILABLE=False。
    """
    for mod, label in [
        ("flash_attn", "flash-attn"),
        ("vllm", "vllm"),
        ("tensordict", "tensordict"),
        ("ray", "ray"),
    ]:
        try:
            importlib.import_module(mod)
            rep.add(OK, f"import {label}", "")
        except Exception as e:  # noqa: BLE001 - 需要捕获任意导入期异常
            rep.add(FAIL, f"import {label}", f"{type(e).__name__}: {e}")


def check_repo(rep: Report, repo: Path) -> None:
    if not (repo / "gigpo" / "core_gigpo.py").exists():
        rep.add(FAIL, "repo 路径", f"{repo} 下没有 gigpo/core_gigpo.py")
        return
    rep.add(OK, "repo 路径", str(repo))

    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    try:
        from gigpo import core_gigpo  # noqa: F401
        rep.add(OK, "import gigpo.core_gigpo", "")
    except Exception as e:  # noqa: BLE001
        rep.add(FAIL, "import gigpo.core_gigpo", f"{type(e).__name__}: {e}")

    # flash-attn 缺失时上游会静默降级，这里显式暴露出来
    try:
        from verl.utils import torch_functional

        if torch_functional.FLAH_ATTN_CROSS_ENTROPY_LOSS_AVAILABLE:
            rep.add(OK, "flash-attn 交叉熵路径", "可用")
        else:
            rep.add(WARN, "flash-attn 交叉熵路径", "降级为普通实现（训练可跑，但更慢/更耗显存）")
    except Exception as e:  # noqa: BLE001
        rep.add(WARN, "verl.utils.torch_functional", f"{type(e).__name__}: {e}")


def _data_dir_candidates() -> list[Path]:
    """训练数据的候选位置。

    ⚠️ 不能只看 `~/data/verl-agent` —— `setup_remote.sh` 会把 `--local_dir`
    指到 `WORK_DIR`（AutoDL 上系统盘只有 30G，数据必须放数据盘）。
    不读环境变量会**报假阴性**，而假阴性比不检查更糟。
    """
    cands = []
    if os.environ.get("VERL_AGENT_DATA"):
        cands.append(Path(os.environ["VERL_AGENT_DATA"]))
    if os.environ.get("WORK_DIR"):
        cands.append(Path(os.environ["WORK_DIR"]) / "data" / "verl-agent")
    cands.append(Path.home() / "data" / "verl-agent")
    return cands


def _alfworld_dir_candidates() -> list[Path]:
    """ALFWorld 资源的候选位置（alfworld 从 ALFWORLD_DATA 取路径）。"""
    cands = []
    if os.environ.get("ALFWORLD_DATA"):
        cands.append(Path(os.environ["ALFWORLD_DATA"]))
    if os.environ.get("WORK_DIR"):
        cands.append(Path(os.environ["WORK_DIR"]) / "alfworld")
    cands.append(Path.home() / ".cache" / "alfworld")
    return cands


def check_data(rep: Report) -> None:
    # ---- 训练数据 ----
    for sub in ("text", "visual"):
        found: tuple[Path, list[Path]] | None = None
        for d in _data_dir_candidates():
            files = sorted((d / sub).glob("*.parquet")) if (d / sub).is_dir() else []
            if files:
                found = (d, files)
                break
        if found:
            d, files = found
            rep.add(OK, f"数据集 {sub}", f"{', '.join(f.name for f in files)}  ({d})")
        else:
            tried = " | ".join(str(d / sub) for d in _data_dir_candidates())
            rep.add(WARN, f"数据集 {sub}", f"缺失 —— 需跑 prepare.py；已找过：{tried}")

    # ---- ALFWorld 资源 ----
    alf: Path | None = None
    for d in _alfworld_dir_candidates():
        if d.is_dir() and any(d.iterdir()):
            alf = d
            break
    if alf:
        # 有 json_2.1.1 才算完整，否则只是空目录
        if (alf / "json_2.1.1").is_dir():
            n = sum(1 for _ in (alf / "json_2.1.1").rglob("game.tw-pddl"))
            rep.add(OK, "ALFWorld 资源", f"{alf}  （{n} 个 game 文件）")
        else:
            rep.add(WARN, "ALFWorld 资源", f"{alf} 存在但没有 json_2.1.1/，下载可能不完整")
    else:
        rep.add(WARN, "ALFWorld 资源",
                f"缺失 —— 需跑 `alfworld-download -f`；已找过："
                f"{' | '.join(str(d) for d in _alfworld_dir_candidates())}")
        rep.add(WARN, "  ALFWorld 包", "已安装" if _version("alfworld") else "未安装")


def check_system(rep: Report) -> None:
    for exe, need in [("conda", False), ("git", True), ("nvidia-smi", True)]:
        path = shutil.which(exe)
        if path:
            rep.add(OK, exe, path)
        else:
            rep.add(WARN if not need else FAIL, exe, "未找到")

    if shutil.which("nvidia-smi"):
        out = os.popen("nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null").read().strip()
        if out:
            rep.add(OK, "NVIDIA 驱动", out.splitlines()[0])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", default=str(Path(__file__).resolve().parent.parent / "repo"))
    ap.add_argument(
        "--local",
        action="store_true",
        help="本地模式：预期报出无 CUDA。用于验证本脚本在无 GPU 机器上不会误报通过。",
    )
    ap.add_argument(
        "--no-gpu",
        action="store_true",
        help="无卡模式：CUDA 不可用属预期，降级为警告而不是失败。"
             "用于在无 GPU 的实例上验证安装（装环境不需要 GPU）。",
    )
    args = ap.parse_args()
    expect_no_gpu = args.local or args.no_gpu

    rep = Report()
    print("=" * 72)
    tag = "（本地模式）" if args.local else ("（无卡模式）" if args.no_gpu else "")
    print(f"verl-agent 环境自检{tag}")
    print("=" * 72)

    check_python(rep)
    check_system(rep)
    check_torch(rep, expect_no_gpu=expect_no_gpu)
    check_packages(rep)
    check_imports(rep)
    check_repo(rep, Path(args.repo).resolve())
    check_data(rep)
    rep.render()

    if args.local:
        print("本地模式：本机无 CUDA 属**预期结果**，说明自检脚本判断正确。")
        print("（本机只做算法单测，正式训练在远程 CUDA 执行。）")
        return 0

    if rep.has_failures():
        print("结论：❌ 环境未就绪 —— 请先解决上面标 ❌ 的项。")
        return 1

    warn = sum(1 for s, _, _ in rep.results if s == WARN)
    print(f"结论：✅ 环境就绪" + (f"（{warn} 项警告，多为可选依赖或数据未就绪）" if warn else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
