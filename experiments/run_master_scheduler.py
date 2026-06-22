"""
主调度脚本 - 自动依次运行所有剩余实验
==========================================
确保同一时间最多 1个DL(GPU) + 1个TML(CPU)

实验组:
1. 基础AL+SSL (ρ1,5,10,20,50,100) - 6策略×5种子×6ρ
2. 创新AL+创新SSL (ρ1,5,10,20,50,100) - 3策略×5种子×6ρ
3. TML验证 (2数据集×2模型×6ρ×2模式) - 7策略×5种子

运行方式:
    python experiments/run_master_scheduler.py
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

# ========== 检查函数 ==========

def check_al_ssl_completed(rho):
    """检查基础AL+SSL某个ρ是否完成"""
    rho_dir = OUTPUT_DIR / "al_ssl" / f"rho{rho}"
    if not rho_dir.exists():
        return False
    json_files = list(rho_dir.glob("*.json"))
    # 需要6策略×5种子=30个文件（每个种子一个json）
    # 或者检查是否有汇总文件
    expected = 6 * len(SEEDS)  # 30
    return len(json_files) >= expected


def check_innovative_al_ssl_completed(rho):
    """检查创新AL+创新SSL某个ρ是否完成"""
    rho_dir = OUTPUT_DIR / "innovative_al_ssl" / f"rho{rho}"
    if not rho_dir.exists():
        return False
    json_files = list(rho_dir.glob("*.json"))
    expected = 3 * len(SEEDS)  # 3策略×5种子=15
    return len(json_files) >= expected


def check_tml_completed(model, dataset, rho, use_ssl):
    """检查TML实验是否完成"""
    ssl_suffix = "_ssl" if use_ssl else ""
    result_file = OUTPUT_DIR / "tml_validation" / f"{model}_{dataset}{ssl_suffix}_rho{rho}_results.json"
    return result_file.exists()


def check_std_al_completed(rho):
    """检查基础AL某个ρ是否完成"""
    rho_dir = OUTPUT_DIR / "std_al" / f"rho{rho}"
    if not rho_dir.exists():
        return False
    json_files = list(rho_dir.glob("*.json"))
    expected = 6 * len(SEEDS)
    return len(json_files) >= expected


# ========== 运行函数 ==========

def run_command(cmd, cwd=None):
    """运行命令并等待完成"""
    print(f"\n  执行: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=cwd or str(PROJECT_ROOT))
    return result.returncode


def run_al_ssl(rho):
    """运行基础AL+SSL实验"""
    script = PROJECT_ROOT / "experiments" / "run_al_ssl_100.py"
    cmd = [sys.executable, str(script), "--rho", str(rho)]
    return run_command(cmd)


def run_innovative_al_ssl(rho):
    """运行创新AL+创新SSL实验"""
    script = PROJECT_ROOT / "experiments" / "run_innovative_al_ssl_100.py"
    cmd = [sys.executable, str(script), "--rho", str(rho)]
    return run_command(cmd)


def run_tml_experiment(model, dataset, rho, use_ssl):
    """运行单个TML实验"""
    script = PROJECT_ROOT / "experiments" / "run_tml_validation.py"
    cmd = [sys.executable, str(script), "--model", model, "--dataset", dataset, "--rho", str(rho)]
    if use_ssl:
        cmd.append("--use-ssl")
    return run_command(cmd)


# ========== 主调度逻辑 ==========

def main():
    print("=" * 70)
    print("主调度脚本 - 自动运行所有剩余实验")
    print("=" * 70)

    total_failed = []

    # ===== 阶段1: 基础AL+SSL =====
    print("\n" + "=" * 70)
    print("阶段1: 基础AL+SSL实验")
    print("=" * 70)

    al_ssl_pending = []
    for rho in RHO_LIST:
        if check_al_ssl_completed(rho):
            print(f"  ✅ ρ={rho} 已完成")
        else:
            al_ssl_pending.append(rho)
            print(f"  ❌ ρ={rho} 待运行")

    for rho in al_ssl_pending:
        print(f"\n--- 运行 基础AL+SSL ρ={rho} ---")
        rc = run_al_ssl(rho)
        if rc != 0:
            total_failed.append(f"al_ssl/rho{rho}")
            print(f"  ❌ 失败: al_ssl/rho{rho}")
        else:
            print(f"  ✅ 完成: al_ssl/rho{rho}")

    # ===== 阶段2: TML实验 =====
    print("\n" + "=" * 70)
    print("阶段2: TML验证实验")
    print("=" * 70)

    tml_pending = []
    for dataset in ["cifar10", "fashionmnist"]:
        for model in ["lr", "rf"]:
            for rho in RHO_LIST:
                for use_ssl in [False, True]:
                    if check_tml_completed(model, dataset, rho, use_ssl):
                        pass  # 已完成
                    else:
                        tml_pending.append({
                            "model": model, "dataset": dataset,
                            "rho": rho, "use_ssl": use_ssl
                        })

    print(f"  TML待运行: {len(tml_pending)} 个实验")

    for i, exp in enumerate(tml_pending, 1):
        ssl_tag = "+SSL" if exp["use_ssl"] else ""
        print(f"\n--- [{i}/{len(tml_pending)}] TML: {exp['model']}/{exp['dataset']}/ρ{exp['rho']}{ssl_tag} ---")
        rc = run_tml_experiment(exp["model"], exp["dataset"], exp["rho"], exp["use_ssl"])
        if rc != 0:
            total_failed.append(f"tml/{exp['model']}_{exp['dataset']}_rho{exp['rho']}{ssl_tag}")
            print(f"  ❌ 失败")
        else:
            print(f"  ✅ 完成")

    # ===== 阶段3: 创新AL+创新SSL =====
    print("\n" + "=" * 70)
    print("阶段3: 创新AL+创新SSL实验（方案3: 双向交互）")
    print("=" * 70)

    innovative_pending = []
    for rho in RHO_LIST:
        if check_innovative_al_ssl_completed(rho):
            print(f"  ✅ ρ={rho} 已完成")
        else:
            innovative_pending.append(rho)
            print(f"  ❌ ρ={rho} 待运行")

    for rho in innovative_pending:
        print(f"\n--- 运行 创新AL+创新SSL ρ={rho} ---")
        rc = run_innovative_al_ssl(rho)
        if rc != 0:
            total_failed.append(f"innovative_al_ssl/rho{rho}")
            print(f"  ❌ 失败: innovative_al_ssl/rho{rho}")
        else:
            print(f"  ✅ 完成: innovative_al_ssl/rho{rho}")

    # ===== 最终汇总 =====
    print("\n" + "=" * 70)
    print("实验调度完成！")
    print("=" * 70)

    if total_failed:
        print(f"\n失败实验 ({len(total_failed)}):")
        for f in total_failed:
            print(f"  ❌ {f}")
    else:
        print("\n🎉 所有实验成功完成！")

    # 打印完成统计
    print("\n完成统计:")
    for rho in RHO_LIST:
        al_ssl_ok = "✅" if check_al_ssl_completed(rho) else "❌"
        innov_ok = "✅" if check_innovative_al_ssl_completed(rho) else "❌"
        print(f"  ρ={rho:>3d}: 基础AL+SSL {al_ssl_ok}  创新AL+SSL {innov_ok}")


if __name__ == "__main__":
    main()
