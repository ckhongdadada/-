"""
消融实验: 创新AL + 基础SSL
============================
目的: 隔离AL创新的贡献（SSL用基础FixMatch，不加deficit阈值和类别加权）

实验矩阵位置: 创新AL列 + 基础SSL行
  AL策略: class_aware_entropy, gap_aware_entropy, adaptive_gap_entropy
  SSL方法: 基础FlexMatch（无deficit阈值，无类别加权损失）

与已有实验对比:
  - al_ssl (基线AL+基础SSL) → 对照组
  - innovative_al_ssl_basic (创新AL+基础SSL) → 隔离AL创新效果
  - innovative_al_ssl (创新AL+创新SSL) → 联合效果

运行方式:
    python experiments/run_innovative_al_basic_ssl.py
"""

import os
import sys
import subprocess
import json
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "output" / "innovative_al_ssl_basic"
EXPERIMENT_SCRIPT = PROJECT_ROOT / "experiments" / "v8_controlled_fast_al_ssl.py"

RHO_LIST = [1, 5, 10, 20, 50, 100]
SEEDS = [42, 123, 456]
INNOVATIVE_STRATEGIES = ["class_aware_entropy", "gap_aware_entropy", "adaptive_gap_entropy"]


def count_checkpoints(group, rho):
    ckpt_dir = OUTPUT_DIR / f"rho{rho}" / "checkpoints"
    if not ckpt_dir.exists():
        return 0
    return len(list(ckpt_dir.glob("*_seed*.json")))


def check_completed(rho):
    expected = len(INNOVATIVE_STRATEGIES) * len(SEEDS)
    return count_checkpoints("innovative_al_ssl_basic", rho) >= expected


def run_experiment(rho, seeds):
    """运行创新AL+基础SSL实验（不加deficit阈值和类别加权）"""
    output_dir = OUTPUT_DIR / f"rho{rho}"
    os.makedirs(output_dir, exist_ok=True)

    cmd = [
        sys.executable, str(EXPERIMENT_SCRIPT),
        "--dataset", "cifar10",
        "--budget-level", "ultra_low",
        "--model-type", "simplecnn",
        "--strategies", *INNOVATIVE_STRATEGIES,
        "--seeds", *[str(s) for s in seeds],
        "--imbalance-ratio", str(rho),
        "--output-dir", str(output_dir),
        "--use-ssl",
        "--ssl-method", "flexmatch",
        # 注意: 不加 --ssl-deficit-threshold 和 --ssl-class-weighted
    ]

    print(f"\n{'='*60}")
    print(f"[Innovative AL + Base SSL] CIFAR-10 rho={rho}")
    print(f"Strategies: {INNOVATIVE_STRATEGIES}")
    print(f"SSL: base FlexMatch (no deficit, no class weight)")
    print(f"Seeds: {seeds}")
    print(f"{'='*60}")

    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    return result.returncode


def main():
    print("=" * 60)
    print("Ablation: Innovative AL + Base SSL")
    print(f"Start: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    failed = []

    for rho in RHO_LIST:
        if check_completed(rho):
            n_ckpts = count_checkpoints("innovative_al_ssl_basic", rho)
            expected = len(INNOVATIVE_STRATEGIES) * len(SEEDS)
            print(f"  [OK] rho={rho} done ({n_ckpts}/{expected})")
        else:
            rc = run_experiment(rho, SEEDS)
            if rc != 0:
                failed.append(f"innovative_al_ssl_basic/rho{rho}")
                print(f"  [FAIL] rho={rho}")
            else:
                print(f"  [OK] rho={rho}")

    print(f"\n{'='*60}")
    print(f"End: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    if failed:
        print(f"Failed: {failed}")
    else:
        print("All done!")
    print("=" * 60)


if __name__ == "__main__":
    main()
