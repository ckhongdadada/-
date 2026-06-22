"""
统一串行调度脚本 - 过夜版
============================
按顺序运行所有待完成实验，同一时间只有一个实验在运行。
支持中断后继续（自动跳过已完成的实验）。

任务列表:
  1. al_ssl_innovative 消融实验 (剩余 rho100)
  2. ResNet-18 验证实验 (rho=10,50 创新策略)
  3. CB/Focal 基线实验
  4. TML 补充实验 (CIFAR-10 rho=50,100)
  5. ResNet-18 扩展实验 (全部策略, 全部rho)
  6. CIFAR-100 加种子实验
  7. TML FashionMNIST 完整实验

运行方式:
    python experiments/run_all_scheduler.py
    python experiments/run_all_scheduler.py --only ablation
    python experiments/run_all_scheduler.py --start-from 3  # 从任务3开始
"""

import os
import sys
import subprocess
import json
import time
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "output"
CIFAR100_DIR = OUTPUT_DIR / "cifar100"
EXPERIMENT_SCRIPT = PROJECT_ROOT / "experiments" / "v8_controlled_fast_al_ssl.py"

SEEDS = [42, 123, 456]
SEEDS_EXTENDED = [789, 1024]  # 额外种子
STD_STRATEGIES = ["random", "entropy", "margin", "coreset", "badge", "qbc"]
INNOVATIVE_STRATEGIES = ["class_aware_entropy", "gap_aware_entropy", "adaptive_gap_entropy"]
ALL_STRATEGIES = STD_STRATEGIES + INNOVATIVE_STRATEGIES


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def run_cmd(cmd, desc, timeout=7200):
    """运行命令并返回退出码，超时默认2小时"""
    log(f"START: {desc}")
    log(f"CMD: {' '.join(cmd[:8])}...")
    try:
        result = subprocess.run(cmd, cwd=str(PROJECT_ROOT), timeout=timeout)
        if result.returncode == 0:
            log(f"DONE: {desc}")
        else:
            log(f"FAIL: {desc} (exit code {result.returncode})")
        return result.returncode
    except subprocess.TimeoutExpired:
        log(f"TIMEOUT: {desc} ({timeout}s)")
        return -1
    except Exception as e:
        log(f"ERROR: {desc}: {e}")
        return -2


def count_checkpoints(group, rho):
    ckpt_dir = OUTPUT_DIR / group / f"rho{rho}" / "checkpoints"
    if not ckpt_dir.exists():
        return 0
    return len(list(ckpt_dir.glob("*.json")))


# ============================================================
# 任务1: al_ssl_innovative 消融实验
# ============================================================
def task_al_ssl_innovative():
    """运行未完成的 al_ssl_innovative 实验"""
    group = "al_ssl_innovative"
    rho_list = [1, 5, 10, 20, 50, 100]
    expected = len(STD_STRATEGIES) * len(SEEDS)  # 6*3=18

    for rho in rho_list:
        n = count_checkpoints(group, rho)
        if n >= expected:
            log(f"[SKIP] {group} rho={rho} ({n}/{expected})")
            continue

        output_dir = OUTPUT_DIR / group / f"rho{rho}"
        output_dir.mkdir(parents=True, exist_ok=True)

        cmd = [
            sys.executable, str(EXPERIMENT_SCRIPT),
            "--dataset", "cifar10",
            "--budget-level", "ultra_low",
            "--model-type", "simplecnn",
            "--strategies", *STD_STRATEGIES,
            "--seeds", *[str(s) for s in SEEDS],
            "--imbalance-ratio", str(rho),
            "--output-dir", str(output_dir),
            "--use-ssl", "--ssl-method", "flexmatch",
            "--ssl-deficit-threshold",
            "--ssl-deficit-alpha", "0.25",
            "--ssl-class-weighted",
        ]

        rc = run_cmd(cmd, f"{group} rho={rho} ({n}/{expected})")
        if rc != 0:
            log(f"WARNING: {group} rho={rho} failed, continuing...")


# ============================================================
# 任务2: ResNet-18 验证实验
# ============================================================
def task_resnet():
    """ResNet-18 验证: 最优策略 × 关键 rho × 3 seeds"""
    # 最优策略: class_aware_entropy (根据已有结果)
    # 关键 rho: 10, 50
    strategies = ["class_aware_entropy", "gap_aware_entropy", "adaptive_gap_entropy"]
    rho_list = [10, 50]
    group = "resnet18"

    for rho in rho_list:
        output_dir = OUTPUT_DIR / group / f"rho{rho}"
        output_dir.mkdir(parents=True, exist_ok=True)

        # 检查是否已完成
        ckpt_dir = output_dir / "checkpoints"
        if ckpt_dir.exists():
            n = len(list(ckpt_dir.glob("*.json")))
            expected = len(strategies) * len(SEEDS)
            if n >= expected:
                log(f"[SKIP] ResNet-18 rho={rho} ({n}/{expected})")
                continue

        cmd = [
            sys.executable, str(EXPERIMENT_SCRIPT),
            "--dataset", "cifar10",
            "--budget-level", "ultra_low",
            "--model-type", "resnet18",  # ResNet-18
            "--strategies", *strategies,
            "--seeds", *[str(s) for s in SEEDS],
            "--imbalance-ratio", str(rho),
            "--output-dir", str(output_dir),
        ]

        rc = run_cmd(cmd, f"ResNet-18 rho={rho}")
        if rc != 0:
            log(f"WARNING: ResNet-18 rho={rho} failed, continuing...")


# ============================================================
# 任务3: CB/Focal 基线实验
# ============================================================
def task_cb_focal():
    """CB Loss / Focal Loss 基线"""
    cb_script = PROJECT_ROOT / "experiments" / "run_cb_focal_baseline.py"
    cmd = [sys.executable, str(cb_script)]
    run_cmd(cmd, "CB/Focal Baseline")


# ============================================================
# 任务4: TML CIFAR-10 补充实验
# ============================================================
def task_tml_cifar10():
    """TML: 补齐 CIFAR-10 rho=50,100"""
    tml_script = PROJECT_ROOT / "experiments" / "run_tml_validation.py"
    rho_list = [50, 100]
    models = ["lr", "rf"]
    strategies = ["random", "entropy", "adaptive_gap_entropy", "class_aware_entropy", "gap_aware_entropy"]

    for model in models:
        for rho in rho_list:
            result_file = OUTPUT_DIR / "tml_validation" / f"{model}_cifar10_rho{rho}_results.json"
            if result_file.exists():
                with open(result_file) as f:
                    data = json.load(f)
                if len(data) >= len(strategies):
                    log(f"[SKIP] TML {model} rho={rho} ({len(data)} strategies)")
                    continue

            cmd = [
                sys.executable, str(tml_script),
                "--model", model, "--dataset", "cifar10",
                "--rho", str(rho),
                "--seeds", *[str(s) for s in SEEDS],
                "--strategies", *strategies,
            ]
            run_cmd(cmd, f"TML {model} CIFAR-10 rho={rho}")


# ============================================================
# 任务5: ResNet-18 扩展实验
# ============================================================
def task_resnet_extended():
    """ResNet-18: 全部策略 × 全部 rho"""
    rho_list = [1, 5, 10, 20, 50, 100]
    group = "resnet18_full"

    for rho in rho_list:
        output_dir = OUTPUT_DIR / group / f"rho{rho}"
        output_dir.mkdir(parents=True, exist_ok=True)
        ckpt_dir = output_dir / "checkpoints"
        expected = len(ALL_STRATEGIES) * len(SEEDS)
        if ckpt_dir.exists() and len(list(ckpt_dir.glob("*.json"))) >= expected:
            log(f"[SKIP] ResNet-18-full rho={rho}")
            continue

        cmd = [
            sys.executable, str(EXPERIMENT_SCRIPT),
            "--dataset", "cifar10", "--budget-level", "ultra_low",
            "--model-type", "resnet18",
            "--strategies", *ALL_STRATEGIES,
            "--seeds", *[str(s) for s in SEEDS],
            "--imbalance-ratio", str(rho),
            "--output-dir", str(output_dir),
        ]
        run_cmd(cmd, f"ResNet-18-full rho={rho}")


# ============================================================
# 任务6: CIFAR-100 加种子
# ============================================================
def task_cifar100_extra_seeds():
    """CIFAR-100: 为已有实验补跑额外种子"""
    groups = {
        "std_al": (STD_STRATEGIES, False),
        "al_ssl": (STD_STRATEGIES, True),
        "innovative_al_ssl": (INNOVATIVE_STRATEGIES, True),
    }
    rho_list = [1, 10, 50]

    for group, (strategies, use_ssl) in groups.items():
        for rho in rho_list:
            for seed in SEEDS_EXTENDED:
                # 检查是否已有该 seed 的 checkpoint
                ckpt_dir = CIFAR100_DIR / group / f"rho{rho}" / "checkpoints"
                exists = any(ckpt_dir.glob(f"*seed{seed}.json")) if ckpt_dir.exists() else False
                if exists:
                    log(f"[SKIP] CIFAR100 {group} rho={rho} seed={seed}")
                    continue

                output_dir = CIFAR100_DIR / group / f"rho{rho}"
                output_dir.mkdir(parents=True, exist_ok=True)

                cmd = [
                    sys.executable, str(EXPERIMENT_SCRIPT),
                    "--dataset", "cifar100", "--budget-level", "ultra_low",
                    "--model-type", "simplecnn",
                    "--strategies", *strategies,
                    "--seeds", str(seed),
                    "--imbalance-ratio", str(rho),
                    "--output-dir", str(output_dir),
                ]
                if use_ssl:
                    cmd.extend(["--use-ssl", "--ssl-method", "flexmatch"])
                    if group == "innovative_al_ssl":
                        cmd.extend(["--ssl-deficit-threshold", "--ssl-deficit-alpha", "0.25",
                                    "--ssl-class-weighted"])

                run_cmd(cmd, f"CIFAR100 {group} rho={rho} seed={seed}")


# ============================================================
# 任务7: TML FashionMNIST 完整实验
# ============================================================
def task_tml_fashionmnist():
    """TML: FashionMNIST 全策略"""
    tml_script = PROJECT_ROOT / "experiments" / "run_tml_validation.py"
    rho_list = [1, 5, 10, 20, 50, 100]
    models = ["lr", "rf"]
    strategies = ["random", "entropy", "adaptive_gap_entropy", "class_aware_entropy", "gap_aware_entropy"]

    for model in models:
        for rho in rho_list:
            result_file = OUTPUT_DIR / "tml_validation" / f"{model}_fashionmnist_rho{rho}_results.json"
            if result_file.exists():
                with open(result_file) as f:
                    data = json.load(f)
                if len(data) >= len(strategies):
                    log(f"[SKIP] TML {model} FashionMNIST rho={rho}")
                    continue

            cmd = [
                sys.executable, str(tml_script),
                "--model", model, "--dataset", "fashionmnist",
                "--rho", str(rho),
                "--seeds", *[str(s) for s in SEEDS],
                "--strategies", *strategies,
            ]
            run_cmd(cmd, f"TML {model} FashionMNIST rho={rho}")


# ============================================================
# 主函数
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="Unified experiment scheduler - overnight")
    parser.add_argument("--start-from", type=int, default=1, help="Start from task N (1-7)")
    parser.add_argument("--only", type=str,
                        choices=["ablation", "resnet", "cb", "tml", "resnet_full", "cifar100", "tml_fm"],
                        help="Run only one task")
    args = parser.parse_args()

    log("=" * 60)
    log("Unified Experiment Scheduler (Overnight)")
    log("=" * 60)

    start_time = time.time()

    tasks = [
        (1, "ablation", "al_ssl_innovative Ablation", task_al_ssl_innovative),
        (2, "resnet", "ResNet-18 Validation", task_resnet),
        (3, "cb", "CB/Focal Baseline", task_cb_focal),
        (4, "tml", "TML CIFAR-10 Supplement", task_tml_cifar10),
        (5, "resnet_full", "ResNet-18 Full Experiment", task_resnet_extended),
        (6, "cifar100", "CIFAR-100 Extra Seeds", task_cifar100_extra_seeds),
        (7, "tml_fm", "TML FashionMNIST Full", task_tml_fashionmnist),
    ]

    for num, key, desc, func in tasks:
        if args.only and args.only != key:
            continue
        if num < args.start_from:
            log(f"\n[SKIP] Task {num}: {desc}")
            continue

        log(f"\n{'='*60}")
        log(f"Task {num}/7: {desc}")
        log(f"{'='*60}")
        try:
            func()
        except Exception as e:
            log(f"ERROR in Task {num}: {e}")

    elapsed = time.time() - start_time
    log(f"\n{'='*60}")
    log(f"All tasks completed in {elapsed/60:.1f} minutes ({elapsed/3600:.1f} hours)")
    log("=" * 60)


if __name__ == "__main__":
    main()
