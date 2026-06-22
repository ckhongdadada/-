"""
最终实验调度器
==============
按顺序完成所有剩余实验:
  1. 渐进式SSL (补齐rho=50,100)
  2. CB/Focal基线 (补齐CB rho=20,50,100)
  3. TML实验 (补齐策略)

运行方式:
    python experiments/run_final_scheduler.py
"""

import os
import sys
import subprocess
import json
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EXPERIMENT_SCRIPT = PROJECT_ROOT / "experiments" / "v8_controlled_fast_al_ssl.py"
TML_SCRIPT = PROJECT_ROOT / "experiments" / "run_tml_validation.py"

SEEDS = [42, 123, 456]
STRATEGIES_4 = ["entropy", "class_aware_entropy", "gap_aware_entropy", "adaptive_gap_entropy"]
STRATEGIES_6 = ["random", "entropy", "margin", "coreset", "badge", "qbc"]
TML_STRATEGIES = ["random", "entropy", "adaptive_gap_entropy", "class_aware_entropy", "gap_aware_entropy"]


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def run_cmd(cmd, desc, timeout=1800):
    log(f"START: {desc}")
    try:
        result = subprocess.run(cmd, cwd=str(PROJECT_ROOT),
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=timeout)
        if result.returncode == 0:
            log(f"DONE: {desc}")
        else:
            log(f"FAIL: {desc} (rc={result.returncode})")
        return result.returncode
    except subprocess.TimeoutExpired:
        log(f"TIMEOUT: {desc}")
        return -1
    except Exception as e:
        log(f"ERROR: {desc}: {e}")
        return -2


def count_checkpoints(ckpt_dir, seed=None):
    """统计checkpoint数量。seed=None时统计全部"""
    if not ckpt_dir.exists():
        return 0
    if seed is None:
        return len(list(ckpt_dir.glob("*seed*.json")))
    return len(list(ckpt_dir.glob(f"*seed{seed}.json")))


# ============================================================
# Task 1: Progressive SSL (补齐)
# ============================================================
def task_progressive_ssl():
    log("\n" + "=" * 60)
    log("Task 1: Progressive SSL (remaining)")
    log("=" * 60)

    output_dir = PROJECT_ROOT / "output" / "progressive_ssl_full"
    rho_list = [50, 100]
    ssl_configs = [
        ("no_ssl",          False, False, False, 0),
        ("base_ssl",        True,  False, False, 0),
        ("innov_ssl",       True,  True,  True,  0),
        ("progressive_r3",  True,  True,  True,  3),
        ("progressive_r5",  True,  True,  True,  5),
        ("progressive_r7",  True,  True,  True,  7),
    ]

    for rho in rho_list:
        # Step 1: 确保 Full Supervised 已计算（用 no_ssl 第一个 seed）
        no_ssl_agg = output_dir / f"no_ssl_rho{rho}" / "aggregated_results.json"
        has_full_sup = False
        if no_ssl_agg.exists():
            try:
                d = json.load(open(no_ssl_agg))
                if d.get("full_supervision", {}).get("f1", 0) > 0:
                    has_full_sup = True
            except: pass

        if not has_full_sup:
            # 先跑一次 no_ssl seed=42 获取 Full Supervised
            out = output_dir / f"no_ssl_rho{rho}"
            cmd = [sys.executable, str(EXPERIMENT_SCRIPT),
                   "--dataset", "cifar10", "--budget-level", "ultra_low", "--model-type", "simplecnn",
                   "--strategies", *STRATEGIES_4, "--seeds", "42",
                   "--imbalance-ratio", str(rho), "--output-dir", str(out)]
            run_cmd(cmd, f"Progressive SSL rho={rho} FULL SUP baseline")

        # Step 2: 跑所有配置，全部跳过 Full Supervised
        for seed in SEEDS:
            for ssl_name, use_ssl, deficit, cw, start_rd in ssl_configs:
                ckpt_dir = output_dir / f"{ssl_name}_rho{rho}" / "checkpoints"
                if count_checkpoints(ckpt_dir, seed) >= 4:
                    continue

                out = output_dir / f"{ssl_name}_rho{rho}"
                cmd = [sys.executable, str(EXPERIMENT_SCRIPT),
                       "--dataset", "cifar10", "--budget-level", "ultra_low", "--model-type", "simplecnn",
                       "--strategies", *STRATEGIES_4, "--seeds", str(seed),
                       "--imbalance-ratio", str(rho), "--output-dir", str(out),
                       "--skip-full-sup"]
                if use_ssl:
                    cmd.extend(["--use-ssl", "--ssl-method", "flexmatch"])
                if deficit:
                    cmd.extend(["--ssl-deficit-threshold", "--ssl-deficit-alpha", "0.25"])
                if cw:
                    cmd.extend(["--ssl-class-weighted"])
                if start_rd > 0:
                    cmd.extend(["--ssl-deficit-start-round", str(start_rd)])

                run_cmd(cmd, f"Progressive SSL rho={rho} seed={seed} ssl={ssl_name}")


# ============================================================
# Task 2: CB/Focal Baseline (补齐)
# ============================================================
def task_cb_focal():
    log("\n" + "=" * 60)
    log("Task 2: CB/Focal Baseline (remaining)")
    log("=" * 60)

    output_dir = PROJECT_ROOT / "output" / "cb_focal_baseline"
    rho_list = [20, 50, 100]

    for rho in rho_list:
        for loss_type in ["cb", "focal"]:
            ckpt_dir = output_dir / "cifar10" / f"rho{rho}" / loss_type / "checkpoints"
            n = count_checkpoints(ckpt_dir)
            if n >= len(SEEDS):
                log(f"[SKIP] rho={rho} {loss_type} ({n} checkpoints)")
                continue

            out = output_dir / "cifar10" / f"rho{rho}" / loss_type
            cmd = [sys.executable, str(EXPERIMENT_SCRIPT),
                   "--dataset", "cifar10", "--budget-level", "ultra_low", "--model-type", "simplecnn",
                   "--strategies", "entropy", "--seeds", *map(str, SEEDS),
                   "--imbalance-ratio", str(rho), "--output-dir", str(out),
                   "--loss-type", loss_type]
            run_cmd(cmd, f"CB/Focal rho={rho} loss={loss_type}")


# ============================================================
# Task 3: TML Experiments (补齐)
# ============================================================
def task_tml():
    log("\n" + "=" * 60)
    log("Task 3: TML Experiments (remaining)")
    log("=" * 60)

    output_dir = PROJECT_ROOT / "output" / "tml_validation"
    rho_list = [10, 50, 100]
    models = ["lr", "rf"]

    for model in models:
        for rho in rho_list:
            result_file = output_dir / f"{model}_cifar10_rho{rho}_results.json"
            if result_file.exists():
                with open(result_file) as f:
                    data = json.load(f)
                if len(data) >= len(TML_STRATEGIES):
                    log(f"[SKIP] TML {model} rho={rho} ({len(data)} strategies)")
                    continue

            cmd = [sys.executable, str(TML_SCRIPT),
                   "--model", model, "--dataset", "cifar10",
                   "--rho", str(rho), "--seeds", *map(str, SEEDS),
                   "--strategies", *TML_STRATEGIES]
            run_cmd(cmd, f"TML {model} CIFAR-10 rho={rho}")


# ============================================================
# Main
# ============================================================
def main():
    log("=" * 60)
    log("Final Experiment Scheduler")
    log("=" * 60)

    start = time.time()

    task_progressive_ssl()
    task_cb_focal()
    task_tml()

    elapsed = (time.time() - start) / 60
    log(f"\n{'='*60}")
    log(f"All tasks completed in {elapsed:.0f} minutes")
    log("=" * 60)


if __name__ == "__main__":
    main()
