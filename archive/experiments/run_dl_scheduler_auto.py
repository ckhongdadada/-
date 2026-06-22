"""
DL实验调度脚本 - 自动运行所有剩余DL实验
==========================================
依次运行: 基础AL+SSL → 创新AL+创新SSL
自动跳过已完成的实验，确保可中断后继续

运行方式:
    python experiments/run_dl_scheduler_auto.py
"""

import os
import sys
import subprocess
import json
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "output"

RHO_LIST = [1, 5, 10, 20, 50, 100]
SEEDS = [42, 123, 456, 789, 1024]
BASE_STRATEGIES = ["random", "entropy", "margin", "coreset", "badge", "qbc"]
INNOVATIVE_STRATEGIES = ["adaptive_gap_entropy", "class_aware_entropy", "gap_aware_entropy"]


# ========== 检查函数 ==========

def check_al_ssl_completed(rho):
    """检查基础AL+SSL某个ρ是否完成（6策略×5种子=30个checkpoint）"""
    ckpt_dir = OUTPUT_DIR / "al_ssl" / f"rho{rho}" / "checkpoints"
    if not ckpt_dir.exists():
        return False
    ckpts = list(ckpt_dir.glob("*.json"))
    return len(ckpts) >= 30


def check_innovative_al_ssl_completed(rho):
    """检查创新AL+创新SSL某个ρ是否完成（3策略×5种子=15个checkpoint）"""
    ckpt_dir = OUTPUT_DIR / "innovative_al_ssl" / f"rho{rho}" / "checkpoints"
    if not ckpt_dir.exists():
        return False
    ckpts = list(ckpt_dir.glob("*.json"))
    return len(ckpts) >= 15


def check_std_al_completed(rho):
    """检查基础AL某个ρ是否完成"""
    ckpt_dir = OUTPUT_DIR / "std_al" / f"rho{rho}" / "checkpoints"
    if not ckpt_dir.exists():
        return False
    ckpts = list(ckpt_dir.glob("*.json"))
    return len(ckpts) >= 30


# ========== 运行函数 ==========

def run_al_ssl(rho):
    """运行基础AL+SSL实验"""
    script = PROJECT_ROOT / "experiments" / "run_al_ssl_100.py"
    cmd = [sys.executable, str(script), "--rho", str(rho)]
    print(f"\n{'='*60}")
    print(f"[DL] 运行 基础AL+SSL ρ={rho}")
    print(f"{'='*60}")
    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    return result.returncode


def run_innovative_al_ssl(rho):
    """运行创新AL+创新SSL实验"""
    script = PROJECT_ROOT / "experiments" / "run_innovative_al_ssl_100.py"
    cmd = [sys.executable, str(script), "--rho", str(rho)]
    print(f"\n{'='*60}")
    print(f"[DL] 运行 创新AL+创新SSL ρ={rho}")
    print(f"{'='*60}")
    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    return result.returncode


def run_std_al(rho):
    """运行基础AL实验"""
    script = PROJECT_ROOT / "experiments" / "run_std_al_100.py"
    cmd = [sys.executable, str(script), "--rho", str(rho)]
    print(f"\n{'='*60}")
    print(f"[DL] 运行 基础AL ρ={rho}")
    print(f"{'='*60}")
    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    return result.returncode


# ========== 主逻辑 ==========

def main():
    print("=" * 60)
    print("DL实验调度脚本 - 自动运行所有剩余DL实验")
    print(f"开始时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    failed = []

    # ===== 阶段1: 基础AL =====
    print("\n" + "=" * 60)
    print("阶段1: 基础AL实验 (std_al)")
    print("=" * 60)
    for rho in RHO_LIST:
        if check_std_al_completed(rho):
            print(f"  ✅ ρ={rho} 已完成，跳过")
        else:
            print(f"  ❌ ρ={rho} 待运行")
            rc = run_std_al(rho)
            if rc != 0:
                failed.append(f"std_al/rho{rho}")
                print(f"  ❌ 失败: std_al/rho{rho}")
            else:
                print(f"  ✅ 完成: std_al/rho{rho}")

    # ===== 阶段2: 基础AL+SSL =====
    print("\n" + "=" * 60)
    print("阶段2: 基础AL+SSL实验 (al_ssl)")
    print("=" * 60)
    for rho in RHO_LIST:
        if check_al_ssl_completed(rho):
            print(f"  ✅ ρ={rho} 已完成，跳过")
        else:
            print(f"  ❌ ρ={rho} 待运行")
            rc = run_al_ssl(rho)
            if rc != 0:
                failed.append(f"al_ssl/rho{rho}")
                print(f"  ❌ 失败: al_ssl/rho{rho}")
            else:
                print(f"  ✅ 完成: al_ssl/rho{rho}")

    # ===== 阶段3: 创新AL+创新SSL =====
    print("\n" + "=" * 60)
    print("阶段3: 创新AL+创新SSL实验 (innovative_al_ssl)")
    print("=" * 60)
    for rho in RHO_LIST:
        if check_innovative_al_ssl_completed(rho):
            print(f"  ✅ ρ={rho} 已完成，跳过")
        else:
            print(f"  ❌ ρ={rho} 待运行")
            rc = run_innovative_al_ssl(rho)
            if rc != 0:
                failed.append(f"innovative_al_ssl/rho{rho}")
                print(f"  ❌ 失败: innovative_al_ssl/rho{rho}")
            else:
                print(f"  ✅ 完成: innovative_al_ssl/rho{rho}")

    # ===== 最终汇总 =====
    print("\n" + "=" * 60)
    print(f"DL实验调度完成！结束时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    if failed:
        print(f"\n失败实验 ({len(failed)}):")
        for f in failed:
            print(f"  ❌ {f}")
    else:
        print("\n🎉 所有DL实验成功完成！")

    # 完成统计
    print("\n完成统计:")
    for rho in RHO_LIST:
        std_ok = "✅" if check_std_al_completed(rho) else "❌"
        al_ssl_ok = "✅" if check_al_ssl_completed(rho) else "❌"
        innov_ok = "✅" if check_innovative_al_ssl_completed(rho) else "❌"
        print(f"  ρ={rho:>3d}: 基础AL {std_ok}  基础AL+SSL {al_ssl_ok}  创新AL+SSL {innov_ok}")


if __name__ == "__main__":
    main()
