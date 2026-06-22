"""
消融实验: 基础AL + 创新SSL
============================
目的: 隔离SSL创新的贡献（AL用基线策略，SSL用deficit阈值+类别加权）

实验矩阵位置: 基线AL列 + 创新SSL行
  AL策略: random, entropy, margin, coreset, badge, qbc
  SSL方法: deficit阈值 + 类别加权一致性损失

与已有实验对比:
  - al_ssl (基线AL+基础SSL) → 对照组
  - al_ssl_innovative (基线AL+创新SSL) → 隔离SSL创新效果
  - innovative_al_ssl (创新AL+创新SSL) → 联合效果

运行方式:
    python experiments/run_al_ssl_innovative.py
"""

import os
import sys
import subprocess
import json
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "output" / "al_ssl_innovative"
EXPERIMENT_SCRIPT = PROJECT_ROOT / "experiments" / "v8_controlled_fast_al_ssl.py"

RHO_LIST = [1, 5, 10, 20, 50, 100]
SEEDS = [42, 123, 456]
BASE_STRATEGIES = ["random", "entropy", "margin", "coreset", "badge", "qbc"]


def count_checkpoints(group, rho):
    ckpt_dir = OUTPUT_DIR / f"rho{rho}" / "checkpoints"
    if not ckpt_dir.exists():
        return 0
    return len(list(ckpt_dir.glob("*_seed*.json")))


def check_completed(rho):
    expected = len(BASE_STRATEGIES) * len(SEEDS)
    return count_checkpoints("al_ssl_innovative", rho) >= expected


def run_experiment(rho, seeds):
    """运行基础AL+创新SSL实验（加deficit阈值和类别加权）"""
    output_dir = OUTPUT_DIR / f"rho{rho}"
    os.makedirs(output_dir, exist_ok=True)

    cmd = [
        sys.executable, str(EXPERIMENT_SCRIPT),
        "--dataset", "cifar10",
        "--budget-level", "ultra_low",
        "--model-type", "simplecnn",
        "--strategies", *BASE_STRATEGIES,
        "--seeds", *[str(s) for s in seeds],
        "--imbalance-ratio", str(rho),
        "--output-dir", str(output_dir),
        "--use-ssl",
        "--ssl-method", "flexmatch",
        "--ssl-deficit-threshold",
        "--ssl-deficit-alpha", "0.25",
        "--ssl-class-weighted",
    ]

    print(f"\n{'='*60}")
    print(f"[AL+Innovative SSL] CIFAR-10 rho={rho}")
    print(f"Strategies: {BASE_STRATEGIES}")
    print(f"SSL: deficit threshold + class weighted")
    print(f"Seeds: {seeds}")
    print(f"{'='*60}")

    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    return result.returncode


def main():
    print("=" * 60)
    print("Ablation: Base AL + Innovative SSL")
    print(f"Start: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    failed = []

    for rho in RHO_LIST:
        if check_completed(rho):
            n_ckpts = count_checkpoints("al_ssl_innovative", rho)
            expected = len(BASE_STRATEGIES) * len(SEEDS)
            print(f"  [OK] rho={rho} done ({n_ckpts}/{expected})")
        else:
            rc = run_experiment(rho, SEEDS)
            if rc != 0:
                failed.append(f"al_ssl_innovative/rho{rho}")
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
