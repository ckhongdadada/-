"""
DL实验调度脚本 - 依次运行缺失的DL实验
==========================================
使用GPU运行，与TML实验（CPU）并行不冲突

缺失实验：
- al_ssl/rho10, rho50

注意：不运行创新AL+SSL（用户要求）

运行方式：
    python experiments/run_dl_scheduler.py
"""

import os
import sys
import subprocess
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
AL_SSL_SCRIPT = PROJECT_ROOT / "experiments" / "run_al_ssl_100.py"
OUTPUT_DIR = PROJECT_ROOT / "output" / "al_ssl"

# 缺失的DL实验配置（rho1正在运行，rho100已完成）
MISSING_DL_EXPERIMENTS = [
    {"experiment": "al_ssl", "rho": 10},
    {"experiment": "al_ssl", "rho": 50},
]


def check_al_ssl_result_exists(rho):
    """检查AL+SSL实验结果是否已存在"""
    result_file = OUTPUT_DIR / f"rho{rho}" / "raw_results.json"
    return result_file.exists()


def run_single_al_ssl_experiment(rho):
    """运行单个AL+SSL实验"""
    cmd = [
        sys.executable, str(AL_SSL_SCRIPT),
        "--rho", str(rho),
    ]

    print(f"\n{'='*60}")
    print(f"Running DL AL+SSL: rho={rho}")
    print(f"策略: random, entropy, margin, coreset, badge, qbc")
    print(f"种子: 42, 123, 456, 789, 1024")
    print(f"{'='*60}")

    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    return result.returncode


def main():
    print("=" * 60)
    print("DL实验调度脚本")
    print(f"总缺失实验数: {len(MISSING_DL_EXPERIMENTS)}")
    print("=" * 60)

    # 检查rho1是否正在运行（通过checkpoint判断）
    rho1_ckpt_dir = OUTPUT_DIR / "rho1" / "checkpoints"
    if rho1_ckpt_dir.exists():
        ckpt_count = len(list(rho1_ckpt_dir.glob("*.json")))
        if ckpt_count > 0 and ckpt_count < 30:
            print(f"⏳ rho=1 正在运行中（已有{ckpt_count}/30个checkpoint），跳过")
        elif ckpt_count >= 30:
            print(f"✓ rho=1 已完成")

    # 检查rho100是否已完成
    if check_al_ssl_result_exists(100):
        print(f"✓ rho=100 已完成")

    # 过滤已完成的实验
    pending_experiments = []
    for exp in MISSING_DL_EXPERIMENTS:
        rho = exp["rho"]
        if exp["experiment"] == "al_ssl":
            if not check_al_ssl_result_exists(rho):
                pending_experiments.append(exp)
            else:
                print(f"✓ 已完成: al_ssl/rho{rho}")

    print(f"\n待运行实验数: {len(pending_experiments)}")

    if len(pending_experiments) == 0:
        print("所有DL实验已完成！")
        return

    # 依次运行
    failed = []
    for i, exp in enumerate(pending_experiments, 1):
        print(f"\n[{i}/{len(pending_experiments)}] 开始运行...")
        if exp["experiment"] == "al_ssl":
            rc = run_single_al_ssl_experiment(exp["rho"])
        else:
            print(f"未知实验类型: {exp['experiment']}")
            continue

        if rc != 0:
            failed.append(exp)
            print(f"❌ 失败: {exp}")
        else:
            print(f"✅ 完成: {exp}")

    print("\n" + "=" * 60)
    if failed:
        print(f"失败实验数: {len(failed)}")
        for exp in failed:
            print(f"  - {exp}")
    else:
        print("所有DL实验成功完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()