"""
TML实验调度脚本 - 与DL实验完全对齐
==========================================
策略: random, entropy, margin, badge, adaptive_gap_entropy, coreset, qbc
模型: LR, RF
数据集: CIFAR-10, FashionMNIST
ρ值: 1, 5, 10, 20, 50, 100
模式: 纯AL, AL+SSL

运行方式:
    python experiments/run_tml_scheduler.py
"""

import os
import sys
import subprocess
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TML_SCRIPT = PROJECT_ROOT / "experiments" / "run_tml_validation.py"
OUTPUT_DIR = PROJECT_ROOT / "output" / "tml_validation"

RHO_LIST = [1, 5, 10, 20, 50, 100]
MODELS = ["lr", "rf"]
DATASETS = ["cifar10", "fashionmnist"]
SSL_MODES = [False, True]


def check_result_exists(model, dataset, rho, use_ssl):
    """检查实验结果是否已存在"""
    ssl_suffix = "_ssl" if use_ssl else ""
    result_file = OUTPUT_DIR / f"{model}_{dataset}{ssl_suffix}_rho{rho}_results.json"
    return result_file.exists()


def generate_all_experiments():
    """生成所有TML实验配置"""
    experiments = []
    for dataset in DATASETS:
        for model in MODELS:
            for rho in RHO_LIST:
                for use_ssl in SSL_MODES:
                    experiments.append({
                        "model": model,
                        "dataset": dataset,
                        "rho": rho,
                        "use_ssl": use_ssl,
                    })
    return experiments


def run_single_tml_experiment(model, dataset, rho, use_ssl):
    """运行单个TML实验"""
    cmd = [
        sys.executable, str(TML_SCRIPT),
        "--model", model,
        "--dataset", dataset,
        "--rho", str(rho),
    ]
    if use_ssl:
        cmd.append("--use-ssl")

    print(f"\n{'='*60}")
    print(f"Running TML: model={model}, dataset={dataset}, rho={rho}, ssl={use_ssl}")
    print(f"策略: random, entropy, margin, badge, adaptive_gap_entropy, coreset, qbc")
    print(f"{'='*60}")

    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    return result.returncode


def main():
    all_experiments = generate_all_experiments()
    total = len(all_experiments)

    print("=" * 60)
    print("TML实验调度脚本（与DL完全对齐）")
    print(f"总实验数: {total}")
    print("=" * 60)

    # 过滤已完成的实验
    pending_experiments = []
    completed = 0
    for exp in all_experiments:
        if check_result_exists(exp["model"], exp["dataset"], exp["rho"], exp["use_ssl"]):
            completed += 1
        else:
            pending_experiments.append(exp)

    print(f"已完成: {completed}/{total}")
    print(f"待运行: {len(pending_experiments)}")

    if len(pending_experiments) == 0:
        print("所有TML实验已完成！")
        return

    # 打印待运行列表
    print("\n待运行实验:")
    for exp in pending_experiments:
        ssl_tag = "+SSL" if exp["use_ssl"] else ""
        print(f"  - {exp['model']}/{exp['dataset']}/rho{exp['rho']}{ssl_tag}")

    # 依次运行
    failed = []
    for i, exp in enumerate(pending_experiments, 1):
        print(f"\n[{i}/{len(pending_experiments)}] 开始运行...")
        rc = run_single_tml_experiment(
            exp["model"], exp["dataset"], exp["rho"], exp["use_ssl"]
        )
        if rc != 0:
            failed.append(exp)
            print(f"❌ 失败: {exp}")
        else:
            print(f"✅ 完成: {exp}")

    print("\n" + "=" * 60)
    print(f"完成: {len(pending_experiments) - len(failed)}/{len(pending_experiments)}")
    if failed:
        print(f"失败实验数: {len(failed)}")
        for exp in failed:
            print(f"  - {exp}")
    else:
        print("所有TML实验成功完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
