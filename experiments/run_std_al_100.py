"""
标准AL策略实验 - 100/100统一配置
=================================
6策略 × 5种子 × 6个ρ值
策略: random, entropy, margin, coreset, badge, qbc
ρ值: 1, 5, 10, 20, 50, 100
配置: n_initial=100, n_query=100, n_rounds=10, 无SSL

运行方式:
    # 运行全部（约4-6小时）
    python experiments/run_std_al_100.py

    # 运行单个ρ值
    python experiments/run_std_al_100.py --rho 10

    # 运行单个策略
    python experiments/run_std_al_100.py --strategies random entropy

    # 快速测试（1种子1轮）
    python experiments/run_std_al_100.py --quick
"""

import os
import sys
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EXPERIMENT_SCRIPT = PROJECT_ROOT / "experiments" / "v8_controlled_fast_al_ssl.py"

# 实验配置
STRATEGIES = ["random", "entropy", "margin", "coreset", "badge", "qbc"]
SEEDS = [42, 123, 456, 789, 1024]
RHO_VALUES = [1, 5, 10, 20, 50, 100]
OUTPUT_BASE = PROJECT_ROOT / "output" / "std_al"


def run_experiment(rho, strategies=None, seeds=None, quick=False):
    """运行单个ρ值的标准AL实验"""
    if strategies is None:
        strategies = STRATEGIES
    if seeds is None:
        seeds = SEEDS

    output_dir = OUTPUT_BASE / f"rho{rho}"
    os.makedirs(output_dir, exist_ok=True)

    cmd = [
        sys.executable, str(EXPERIMENT_SCRIPT),
        "--dataset", "cifar10",
        "--budget-level", "ultra_low",
        "--model-type", "simplecnn",
        "--strategies", *strategies,
        "--seeds", *[str(s) for s in seeds],
        "--imbalance-ratio", str(rho),
        "--output-dir", str(output_dir),
        "--no-use-ssl",  # 标准AL不使用SSL
    ]

    if quick:
        cmd.extend(["--quick"])

    print(f"\n{'='*60}")
    print(f"Running: ρ={rho}, strategies={strategies}, seeds={seeds}")
    print(f"Output: {output_dir}")
    print(f"{'='*60}")

    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    if result.returncode != 0:
        print(f"ERROR: ρ={rho} failed with return code {result.returncode}")
    return result.returncode


def main():
    import argparse
    parser = argparse.ArgumentParser(description="标准AL策略实验(100/100配置)")
    parser.add_argument("--rho", type=int, nargs="+", default=None, help="只运行指定ρ值")
    parser.add_argument("--strategies", type=str, nargs="+", default=None, help="只运行指定策略")
    parser.add_argument("--seeds", type=int, nargs="+", default=None, help="只运行指定种子")
    parser.add_argument("--quick", action="store_true", help="快速测试模式")
    args = parser.parse_args()

    rho_values = args.rho if args.rho else RHO_VALUES
    strategies = args.strategies if args.strategies else STRATEGIES
    seeds = args.seeds if args.seeds else SEEDS

    print("=" * 60)
    print("标准AL策略实验 - 100/100统一配置")
    print(f"策略: {strategies}")
    print(f"种子: {seeds}")
    print(f"ρ值: {rho_values}")
    print(f"总运行数: {len(rho_values)} × {len(strategies)} × {len(seeds)} = {len(rho_values)*len(strategies)*len(seeds)}")
    print("=" * 60)

    failed = []
    for rho in rho_values:
        rc = run_experiment(rho, strategies, seeds, args.quick)
        if rc != 0:
            failed.append(rho)

    print("\n" + "=" * 60)
    if failed:
        print(f"FAILED ρ values: {failed}")
    else:
        print("ALL EXPERIMENTS COMPLETED SUCCESSFULLY")
    print(f"Results saved to: {OUTPUT_BASE}")
    print("=" * 60)


if __name__ == "__main__":
    main()
