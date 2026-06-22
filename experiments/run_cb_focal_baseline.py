"""
CB Loss / Focal Loss 基线实验调度脚本
======================================
对比长尾处理基线方法，验证创新方法的优越性

实验设计:
  - 数据集: CIFAR-10 (3种子, 6ρ) + CIFAR-100 (1种子, 3ρ)
  - 损失函数: CB Loss, Focal Loss (对比默认CE)
  - AL策略: entropy (代表性策略)
  - 模式: AL only + AL+SSL
  - 目的: 证明deficit+类别加权方法优于传统长尾损失

与主实验对比:
  - CE + Entropy AL → 基线 (已有)
  - CB + Entropy AL → 传统长尾损失基线
  - Focal + Entropy AL → 传统难样本聚焦基线
  - 创新AL+创新SSL → 我们的方法 (已有)

运行方式:
    python experiments/run_cb_focal_baseline.py
"""

import os
import sys
import subprocess
import json
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR_CB = PROJECT_ROOT / "output" / "cb_focal_baseline"
EXPERIMENT_SCRIPT = PROJECT_ROOT / "experiments" / "v8_controlled_fast_al_ssl.py"

# CIFAR-10: 3种子节省时间
RHO_LIST_CIFAR10 = [1, 5, 10, 20, 50, 100]
SEEDS_CIFAR10 = [42, 123, 456]

# CIFAR-100: 交叉验证配置
RHO_LIST_CIFAR100 = [1, 10, 50]
SEEDS_CIFAR100 = [42]

# 使用entropy作为代表性AL策略
STRATEGY = "entropy"

# 损失函数类型
LOSS_TYPES = ["cb", "focal"]

# 模式: AL only, AL+SSL
SSL_MODES = [False, True]


def check_completed(dataset, rho, loss_type, use_ssl, seed):
    """检查某个实验是否已完成"""
    ssl_tag = "_ssl" if use_ssl else ""
    ckpt_dir = OUTPUT_DIR_CB / dataset / f"rho{rho}" / loss_type / "checkpoints"
    if not ckpt_dir.exists():
        return False
    ckpt = ckpt_dir / f"{STRATEGY}_seed{seed}{ssl_tag}.json"
    return ckpt.exists()


def count_completed(dataset, rho, loss_type, use_ssl, seeds):
    """统计已完成的checkpoint数"""
    ckpt_dir = OUTPUT_DIR_CB / dataset / f"rho{rho}" / loss_type / "checkpoints"
    if not ckpt_dir.exists():
        return 0
    ssl_tag = "_ssl" if use_ssl else ""
    count = 0
    for seed in seeds:
        ckpt = ckpt_dir / f"{STRATEGY}_seed{seed}{ssl_tag}.json"
        if ckpt.exists():
            count += 1
    return count


def run_experiment(dataset, rho, loss_type, use_ssl, seeds):
    """运行一组实验"""
    output_dir = OUTPUT_DIR_CB / dataset / f"rho{rho}" / loss_type
    os.makedirs(output_dir, exist_ok=True)

    cmd = [
        sys.executable, str(EXPERIMENT_SCRIPT),
        "--dataset", dataset,
        "--budget-level", "ultra_low",
        "--model-type", "simplecnn",
        "--strategies", STRATEGY,
        "--seeds", *[str(s) for s in seeds],
        "--imbalance-ratio", str(rho),
        "--output-dir", str(output_dir),
        "--loss-type", loss_type,
    ]

    if use_ssl:
        cmd.extend(["--use-ssl", "--ssl-method", "flexmatch"])

    ssl_tag = "+SSL" if use_ssl else ""
    print(f"\n{'='*60}")
    print(f"[CB/Focal基线] {dataset}/ρ={rho}/{loss_type}{ssl_tag}")
    print(f"策略: {STRATEGY}, 种子: {seeds}")
    print(f"{'='*60}")

    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    return result.returncode


def main():
    print("=" * 60)
    print("CB Loss / Focal Loss 基线实验调度脚本")
    print(f"开始时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"策略: {STRATEGY}")
    print(f"损失函数: {LOSS_TYPES}")
    print("=" * 60)

    failed = []
    total = 0
    completed = 0

    # ===== CIFAR-10 实验 =====
    print("\n" + "=" * 60)
    print("CIFAR-10 基线实验")
    print("=" * 60)
    for loss_type in LOSS_TYPES:
        for rho in RHO_LIST_CIFAR10:
            for use_ssl in SSL_MODES:
                total += len(SEEDS_CIFAR10)
                n_done = count_completed("cifar10", rho, loss_type, use_ssl, SEEDS_CIFAR10)
                completed += n_done

                if n_done >= len(SEEDS_CIFAR10):
                    ssl_tag = "+SSL" if use_ssl else ""
                    print(f"  [OK] cifar10/ρ={rho}/{loss_type}{ssl_tag} 已完成 ({n_done}/{len(SEEDS_CIFAR10)})，跳过")
                else:
                    ssl_tag = "+SSL" if use_ssl else ""
                    print(f"  [FAIL] cifar10/ρ={rho}/{loss_type}{ssl_tag} 待运行 ({n_done}/{len(SEEDS_CIFAR10)})")
                    rc = run_experiment("cifar10", rho, loss_type, use_ssl, SEEDS_CIFAR10)
                    if rc != 0:
                        failed.append(f"cifar10/ρ={rho}/{loss_type}{'+SSL' if use_ssl else ''}")
                        print(f"  [FAIL] 失败")
                    else:
                        completed += len(SEEDS_CIFAR10) - n_done
                        print(f"  [OK] 完成")

    # ===== CIFAR-100 实验 =====
    print("\n" + "=" * 60)
    print("CIFAR-100 基线实验")
    print("=" * 60)
    for loss_type in LOSS_TYPES:
        for rho in RHO_LIST_CIFAR100:
            for use_ssl in SSL_MODES:
                total += len(SEEDS_CIFAR100)
                n_done = count_completed("cifar100", rho, loss_type, use_ssl, SEEDS_CIFAR100)
                completed += n_done

                if n_done >= len(SEEDS_CIFAR100):
                    ssl_tag = "+SSL" if use_ssl else ""
                    print(f"  [OK] cifar100/ρ={rho}/{loss_type}{ssl_tag} 已完成 ({n_done}/{len(SEEDS_CIFAR100)})，跳过")
                else:
                    ssl_tag = "+SSL" if use_ssl else ""
                    print(f"  [FAIL] cifar100/ρ={rho}/{loss_type}{ssl_tag} 待运行 ({n_done}/{len(SEEDS_CIFAR100)})")
                    rc = run_experiment("cifar100", rho, loss_type, use_ssl, SEEDS_CIFAR100)
                    if rc != 0:
                        failed.append(f"cifar100/ρ={rho}/{loss_type}{'+SSL' if use_ssl else ''}")
                        print(f"  [FAIL] 失败")
                    else:
                        completed += len(SEEDS_CIFAR100) - n_done
                        print(f"  [OK] 完成")

    # ===== 最终汇总 =====
    print("\n" + "=" * 60)
    print(f"CB/Focal基线实验调度完成！结束时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"完成: {completed}/{total}")
    print("=" * 60)

    if failed:
        print(f"\n失败实验 ({len(failed)}):")
        for f in failed:
            print(f"  [FAIL] {f}")
    else:
        print("\n🎉 所有CB/Focal基线实验成功完成！")

    # 完成统计
    print("\n完成统计:")
    for dataset, rho_list, seeds in [
        ("cifar10", RHO_LIST_CIFAR10, SEEDS_CIFAR10),
        ("cifar100", RHO_LIST_CIFAR100, SEEDS_CIFAR100),
    ]:
        print(f"\n  [{dataset}]")
        for loss_type in LOSS_TYPES:
            for rho in rho_list:
                al_n = count_completed(dataset, rho, loss_type, False, seeds)
                ssl_n = count_completed(dataset, rho, loss_type, True, seeds)
                expected = len(seeds)
                al_ok = "[OK]" if al_n >= expected else "[FAIL]"
                ssl_ok = "[OK]" if ssl_n >= expected else "[FAIL]"
                print(f"    {loss_type}/ρ={rho:>3d}: AL {al_n}/{expected} {al_ok}  AL+SSL {ssl_n}/{expected} {ssl_ok}")


if __name__ == "__main__":
    main()
