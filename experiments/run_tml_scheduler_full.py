"""
完整TML实验调度脚本
====================
与DL实验完全对齐:
  - 数据集: CIFAR-10, FashionMNIST, CIFAR-100
  - 策略: 9个(6基础 + 3创新) = DL完全对齐
  - 模式: AL + AL+SSL
  - 种子: 3 (CIFAR-10/FashionMNIST), 1 (CIFAR-100)
  - ρ值: CIFAR-10/FashionMNIST=[1,5,10,20,50,100], CIFAR-100=[1,10,50]

自动跳过已完成的实验，确保可中断后继续

运行方式:
    python experiments/run_tml_scheduler_full.py
"""

import os
import sys
import subprocess
import json
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TML_SCRIPT = PROJECT_ROOT / "experiments" / "run_tml_validation.py"
OUTPUT_DIR = PROJECT_ROOT / "output" / "tml_validation"

# CIFAR-10和FashionMNIST: 完整配置
RHO_LIST_FULL = [1, 5, 10, 20, 50, 100]
# CIFAR-100: 精简配置（交叉验证）
RHO_LIST_CIFAR100 = [1, 10, 50]

MODELS = ["lr", "rf"]
DATASETS_FULL = ["cifar10", "fashionmnist"]
SEEDS_FULL = [42, 123, 456]
SEEDS_CIFAR100 = [42]

# 9个策略 = 6基础 + 3创新，与DL完全对齐
ALL_STRATEGIES = [
    "random", "entropy", "margin", "badge", "coreset", "qbc",  # 基础6个
    "adaptive_gap_entropy", "class_aware_entropy", "gap_aware_entropy",  # 创新3个
]
SSL_MODES = [False, True]


def check_result_exists(model, dataset, rho, use_ssl):
    """检查实验结果是否已存在"""
    ssl_suffix = "_ssl" if use_ssl else ""
    result_file = OUTPUT_DIR / f"{model}_{dataset}{ssl_suffix}_rho{rho}_results.json"
    if not result_file.exists():
        return False
    # 检查结果中是否包含所有9个策略
    try:
        with open(result_file, 'r') as f:
            data = json.load(f)
        for strategy in ALL_STRATEGIES:
            if strategy not in data:
                return False
        return True
    except (json.JSONDecodeError, IOError):
        return False


def generate_all_experiments():
    """生成所有TML实验配置"""
    experiments = []

    # CIFAR-10和FashionMNIST: 完整配置
    for dataset in DATASETS_FULL:
        for model in MODELS:
            for rho in RHO_LIST_FULL:
                for use_ssl in SSL_MODES:
                    experiments.append({
                        "model": model,
                        "dataset": dataset,
                        "rho": rho,
                        "use_ssl": use_ssl,
                        "seeds": SEEDS_FULL,
                    })

    # CIFAR-100: 交叉验证配置
    for model in MODELS:
        for rho in RHO_LIST_CIFAR100:
            for use_ssl in SSL_MODES:
                experiments.append({
                    "model": model,
                    "dataset": "cifar100",
                    "rho": rho,
                    "use_ssl": use_ssl,
                    "seeds": SEEDS_CIFAR100,
                })

    return experiments


def run_single_tml_experiment(model, dataset, rho, use_ssl, seeds):
    """运行单个TML实验"""
    cmd = [
        sys.executable, str(TML_SCRIPT),
        "--model", model,
        "--dataset", dataset,
        "--rho", str(rho),
        "--seeds", *[str(s) for s in seeds],
        "--strategies", *ALL_STRATEGIES,
    ]
    if use_ssl:
        cmd.append("--use-ssl")

    ssl_tag = "+SSL" if use_ssl else ""
    print(f"\n{'='*60}")
    print(f"[TML] {model}/{dataset}/ρ={rho}{ssl_tag} seeds={seeds}")
    print(f"策略: {ALL_STRATEGIES}")
    print(f"{'='*60}")

    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    return result.returncode


def main():
    print("=" * 60)
    print("完整TML实验调度脚本 - 与DL完全对齐")
    print(f"开始时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"策略: {ALL_STRATEGIES}")
    print("=" * 60)

    all_experiments = generate_all_experiments()
    total = len(all_experiments)

    # 过滤已完成的实验
    pending_experiments = []
    completed = 0
    for exp in all_experiments:
        if check_result_exists(exp["model"], exp["dataset"], exp["rho"], exp["use_ssl"]):
            completed += 1
        else:
            pending_experiments.append(exp)

    print(f"总实验数: {total}")
    print(f"已完成: {completed}")
    print(f"待运行: {len(pending_experiments)}")

    if len(pending_experiments) == 0:
        print("所有TML实验已完成！")
        return

    # 打印待运行列表
    print("\n待运行实验:")
    for exp in pending_experiments:
        ssl_tag = "+SSL" if exp["use_ssl"] else ""
        print(f"  - {exp['model']}/{exp['dataset']}/ρ={exp['rho']}{ssl_tag} seeds={exp['seeds']}")

    # 依次运行
    failed = []
    for i, exp in enumerate(pending_experiments, 1):
        ssl_tag = "+SSL" if exp["use_ssl"] else ""
        print(f"\n[{i}/{len(pending_experiments)}] {exp['model']}/{exp['dataset']}/ρ={exp['rho']}{ssl_tag}")
        rc = run_single_tml_experiment(
            exp["model"], exp["dataset"], exp["rho"], exp["use_ssl"], exp["seeds"]
        )
        if rc != 0:
            failed.append(exp)
            print(f"  ❌ 失败")
        else:
            print(f"  ✅ 完成")

    # ===== 最终汇总 =====
    print("\n" + "=" * 60)
    print(f"TML实验调度完成！结束时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    if failed:
        print(f"\n失败实验 ({len(failed)}):")
        for exp in failed:
            ssl_tag = "+SSL" if exp["use_ssl"] else ""
            print(f"  ❌ {exp['model']}/{exp['dataset']}/ρ={exp['rho']}{ssl_tag}")
    else:
        print("\n🎉 所有TML实验成功完成！")

    # 完成统计
    print("\n完成统计:")
    all_datasets = list(DATASETS_FULL) + ["cifar100"]
    for dataset in all_datasets:
        print(f"\n  [{dataset}]")
        rho_list = RHO_LIST_FULL if dataset in DATASETS_FULL else RHO_LIST_CIFAR100
        for model in MODELS:
            for rho in rho_list:
                al_ok = "✅" if check_result_exists(model, dataset, rho, False) else "❌"
                ssl_ok = "✅" if check_result_exists(model, dataset, rho, True) else "❌"
                print(f"    {model}/ρ={rho:>3d}: AL {al_ok}  AL+SSL {ssl_ok}")


if __name__ == "__main__":
    main()
