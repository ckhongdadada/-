"""
CIFAR-100 DL实验调度脚本 - 交叉验证
=====================================
1种子(42) × 3ρ(1,10,50) × 4实验组
与CIFAR-10主实验完全对齐，验证结论跨数据集通用性

实验组:
  1. 基础AL (std_al): 6策略 × 3ρ × 1种子 = 18
  2. 基础AL+SSL (al_ssl): 6策略 × 3ρ × 1种子 = 18
  3. 创新AL (innovative_al): 3策略 × 3ρ × 1种子 = 9
  4. 创新AL+创新SSL (innovative_al_ssl): 3策略 × 3ρ × 1种子 = 9
  总计: 54个实验

运行方式:
    python experiments/run_cifar100_dl_scheduler.py
"""

import os
import sys
import subprocess
import json
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "output" / "cifar100"
EXPERIMENT_SCRIPT = PROJECT_ROOT / "experiments" / "v8_controlled_fast_al_ssl.py"

RHO_LIST = [1, 10, 50]
SEEDS = [42]
BASE_STRATEGIES = ["random", "entropy", "margin", "coreset", "badge", "qbc"]
INNOVATIVE_STRATEGIES = ["adaptive_gap_entropy", "class_aware_entropy", "gap_aware_entropy"]


def check_completed(group, rho, strategy=None, seed=None):
    """检查某个实验是否已完成"""
    ckpt_dir = OUTPUT_DIR / group / f"rho{rho}" / "checkpoints"
    if not ckpt_dir.exists():
        return False
    if strategy and seed:
        ckpt = ckpt_dir / f"{strategy}_seed{seed}.json"
        return ckpt.exists()
    return False


def count_checkpoints(group, rho):
    """统计已完成的checkpoint数"""
    ckpt_dir = OUTPUT_DIR / group / f"rho{rho}" / "checkpoints"
    if not ckpt_dir.exists():
        return 0
    return len(list(ckpt_dir.glob("*.json")))


def run_experiment(group, rho, strategies, seeds, use_ssl=False, innovative_ssl=False):
    """运行一组实验"""
    output_dir = OUTPUT_DIR / group / f"rho{rho}"
    os.makedirs(output_dir, exist_ok=True)

    cmd = [
        sys.executable, str(EXPERIMENT_SCRIPT),
        "--dataset", "cifar100",
        "--budget-level", "ultra_low",
        "--model-type", "simplecnn",
        "--strategies", *strategies,
        "--seeds", *[str(s) for s in seeds],
        "--imbalance-ratio", str(rho),
        "--output-dir", str(output_dir),
    ]

    if use_ssl:
        cmd.extend(["--use-ssl", "--ssl-method", "flexmatch"])

    if innovative_ssl:
        cmd.extend([
            "--ssl-deficit-threshold",
            "--ssl-deficit-alpha", "0.25",
            "--ssl-class-weighted",
        ])

    group_desc = group
    print(f"\n{'='*60}")
    print(f"[CIFAR-100 DL] {group_desc} ρ={rho}")
    print(f"策略: {strategies}, 种子: {seeds}")
    print(f"{'='*60}")

    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    return result.returncode


def main():
    print("=" * 60)
    print("CIFAR-100 DL实验调度脚本 - 交叉验证")
    print(f"开始时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"ρ值: {RHO_LIST}, 种子: {SEEDS}")
    print("=" * 60)

    failed = []

    # ===== 阶段1: 基础AL =====
    print("\n" + "=" * 60)
    print("阶段1: 基础AL (std_al)")
    print("=" * 60)
    for rho in RHO_LIST:
        n_ckpts = count_checkpoints("std_al", rho)
        expected = len(BASE_STRATEGIES) * len(SEEDS)
        if n_ckpts >= expected:
            print(f"  ✅ ρ={rho} 已完成 ({n_ckpts}/{expected})，跳过")
        else:
            print(f"  ❌ ρ={rho} 待运行 ({n_ckpts}/{expected})")
            rc = run_experiment("std_al", rho, BASE_STRATEGIES, SEEDS, use_ssl=False)
            if rc != 0:
                failed.append(f"std_al/rho{rho}")
                print(f"  ❌ 失败: std_al/rho{rho}")
            else:
                print(f"  ✅ 完成: std_al/rho{rho}")

    # ===== 阶段2: 基础AL+SSL =====
    print("\n" + "=" * 60)
    print("阶段2: 基础AL+SSL (al_ssl)")
    print("=" * 60)
    for rho in RHO_LIST:
        n_ckpts = count_checkpoints("al_ssl", rho)
        expected = len(BASE_STRATEGIES) * len(SEEDS)
        if n_ckpts >= expected:
            print(f"  ✅ ρ={rho} 已完成 ({n_ckpts}/{expected})，跳过")
        else:
            print(f"  ❌ ρ={rho} 待运行 ({n_ckpts}/{expected})")
            rc = run_experiment("al_ssl", rho, BASE_STRATEGIES, SEEDS, use_ssl=True)
            if rc != 0:
                failed.append(f"al_ssl/rho{rho}")
                print(f"  ❌ 失败: al_ssl/rho{rho}")
            else:
                print(f"  ✅ 完成: al_ssl/rho{rho}")

    # ===== 阶段3: 创新AL =====
    print("\n" + "=" * 60)
    print("阶段3: 创新AL (innovative_al)")
    print("=" * 60)
    for rho in RHO_LIST:
        n_ckpts = count_checkpoints("innovative_al", rho)
        expected = len(INNOVATIVE_STRATEGIES) * len(SEEDS)
        if n_ckpts >= expected:
            print(f"  ✅ ρ={rho} 已完成 ({n_ckpts}/{expected})，跳过")
        else:
            print(f"  ❌ ρ={rho} 待运行 ({n_ckpts}/{expected})")
            rc = run_experiment("innovative_al", rho, INNOVATIVE_STRATEGIES, SEEDS, use_ssl=False)
            if rc != 0:
                failed.append(f"innovative_al/rho{rho}")
                print(f"  ❌ 失败: innovative_al/rho{rho}")
            else:
                print(f"  ✅ 完成: innovative_al/rho{rho}")

    # ===== 阶段4: 创新AL+创新SSL =====
    print("\n" + "=" * 60)
    print("阶段4: 创新AL+创新SSL (innovative_al_ssl)")
    print("=" * 60)
    for rho in RHO_LIST:
        n_ckpts = count_checkpoints("innovative_al_ssl", rho)
        expected = len(INNOVATIVE_STRATEGIES) * len(SEEDS)
        if n_ckpts >= expected:
            print(f"  ✅ ρ={rho} 已完成 ({n_ckpts}/{expected})，跳过")
        else:
            print(f"  ❌ ρ={rho} 待运行 ({n_ckpts}/{expected})")
            rc = run_experiment("innovative_al_ssl", rho, INNOVATIVE_STRATEGIES, SEEDS,
                              use_ssl=True, innovative_ssl=True)
            if rc != 0:
                failed.append(f"innovative_al_ssl/rho{rho}")
                print(f"  ❌ 失败: innovative_al_ssl/rho{rho}")
            else:
                print(f"  ✅ 完成: innovative_al_ssl/rho{rho}")

    # ===== 最终汇总 =====
    print("\n" + "=" * 60)
    print(f"CIFAR-100 DL实验调度完成！结束时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    if failed:
        print(f"\n失败实验 ({len(failed)}):")
        for f in failed:
            print(f"  ❌ {f}")
    else:
        print("\n🎉 所有CIFAR-100 DL实验成功完成！")

    # 完成统计
    print("\n完成统计:")
    for rho in RHO_LIST:
        for group in ["std_al", "al_ssl", "innovative_al", "innovative_al_ssl"]:
            n = count_checkpoints(group, rho)
            expected_map = {
                "std_al": len(BASE_STRATEGIES) * len(SEEDS),
                "al_ssl": len(BASE_STRATEGIES) * len(SEEDS),
                "innovative_al": len(INNOVATIVE_STRATEGIES) * len(SEEDS),
                "innovative_al_ssl": len(INNOVATIVE_STRATEGIES) * len(SEEDS),
            }
            expected = expected_map[group]
            status = "✅" if n >= expected else "❌"
            print(f"  ρ={rho:>3d} {group:25s}: {n}/{expected} {status}")


if __name__ == "__main__":
    main()
