"""
V3 全部实验调度器
=================
按优先级顺序运行所有 V3 实验，同一时间只有一个 GPU 任务。

优先级1: V3 SSL 策略 (class_aware_entropy_ssl, gap_aware_entropy_ssl)
优先级2: V3 纯AL 策略 (class_aware_entropy, gap_aware_entropy, adaptive_gap_entropy)
优先级3: 扩展策略 (two_stage_entropy_balance, curriculum_penalty_entropy)
优先级4: CIFAR-100 跨数据集验证

运行方式:
    python experiments/run_v3_all.py
    python experiments/run_v3_all.py --start-from 2  # 从优先级2开始
"""

import os
import sys
import subprocess
import json
import time
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
V8_SCRIPT = PROJECT_ROOT / "experiments" / "v8_controlled_fast_al_ssl.py"

SEEDS = [42, 123, 456]
RHOS = [1, 10, 50, 100]


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


def count_checkpoints(base_dir, group, rho, seed):
    ckpt_dir = base_dir / group / f"rho{rho}" / "checkpoints"
    if not ckpt_dir.exists():
        return 0
    return len(list(ckpt_dir.glob(f"*seed{seed}.json")))


def count_all_checkpoints(base_dir, group, rho):
    ckpt_dir = base_dir / group / f"rho{rho}" / "checkpoints"
    if not ckpt_dir.exists():
        return 0
    return len(list(ckpt_dir.glob("*seed*.json")))


# ============================================================
# Priority 1: V3 SSL strategies
# ============================================================
def priority1_v3_ssl():
    log("\n" + "=" * 60)
    log("Priority 1: V3 SSL Strategies (class_aware_entropy_ssl, gap_aware_entropy_ssl)")
    log("=" * 60)

    output_dir = PROJECT_ROOT / "output" / "v3_al_ssl"
    strategies = ["class_aware_entropy_ssl", "gap_aware_entropy_ssl"]
    expected = len(strategies) * len(SEEDS)

    for rho in RHOS:
        out = output_dir / f"rho{rho}"
        out.mkdir(parents=True, exist_ok=True)

        for seed in SEEDS:
            ckpt_dir = out / "checkpoints"
            existing = sum(1 for s in strategies
                          for f in ckpt_dir.glob(f"{s}_seed{seed}.json")) if ckpt_dir.exists() else 0
            if existing >= len(strategies):
                continue

            cmd = [
                sys.executable, str(V8_SCRIPT),
                "--dataset", "cifar10", "--budget-level", "ultra_low",
                "--model-type", "simplecnn",
                "--strategies", *strategies,
                "--seeds", str(seed),
                "--imbalance-ratio", str(rho),
                "--output-dir", str(out),
                "--use-ssl", "--ssl-method", "flexmatch",
                "--class-aware-adaptive",
                "--class-aware-soft-weighting",
            ]
            # Skip full sup if already exists
            agg = out / "aggregated_results.json"
            if agg.exists():
                try:
                    d = json.load(open(agg))
                    if d.get("full_supervision", {}).get("f1", 0) > 0:
                        cmd.append("--skip-full-sup")
                except: pass

            run_cmd(cmd, f"V3-SSL rho={rho} seed={seed}")


# ============================================================
# Priority 2: V3 pure AL strategies
# ============================================================
def priority2_v3_al():
    log("\n" + "=" * 60)
    log("Priority 2: V3 Pure AL Strategies")
    log("=" * 60)

    output_dir = PROJECT_ROOT / "output" / "v3_al"
    strategies = ["class_aware_entropy", "gap_aware_entropy", "adaptive_gap_entropy"]
    expected = len(strategies) * len(SEEDS)

    for rho in RHOS:
        out = output_dir / f"rho{rho}"
        out.mkdir(parents=True, exist_ok=True)

        for seed in SEEDS:
            ckpt_dir = out / "checkpoints"
            existing = sum(1 for s in strategies
                          for f in ckpt_dir.glob(f"{s}_seed{seed}.json")) if ckpt_dir.exists() else 0
            if existing >= len(strategies):
                continue

            cmd = [
                sys.executable, str(V8_SCRIPT),
                "--dataset", "cifar10", "--budget-level", "ultra_low",
                "--model-type", "simplecnn",
                "--strategies", *strategies,
                "--seeds", str(seed),
                "--imbalance-ratio", str(rho),
                "--output-dir", str(out),
                "--class-aware-adaptive",
                "--class-aware-soft-weighting",
            ]
            agg = out / "aggregated_results.json"
            if agg.exists():
                try:
                    d = json.load(open(agg))
                    if d.get("full_supervision", {}).get("f1", 0) > 0:
                        cmd.append("--skip-full-sup")
                except: pass

            run_cmd(cmd, f"V3-AL rho={rho} seed={seed}")


# ============================================================
# Priority 3: Extended strategies
# ============================================================
def priority3_extended():
    log("\n" + "=" * 60)
    log("Priority 3: Extended Strategies (two_stage, curriculum)")
    log("=" * 60)

    output_dir = PROJECT_ROOT / "output" / "v3_extended"
    strategies = ["two_stage_entropy_balance", "curriculum_penalty_entropy"]

    for rho in [50, 100]:
        out = output_dir / f"rho{rho}"
        out.mkdir(parents=True, exist_ok=True)

        for seed in SEEDS:
            ckpt_dir = out / "checkpoints"
            existing = sum(1 for s in strategies
                          for f in ckpt_dir.glob(f"{s}_seed{seed}.json")) if ckpt_dir.exists() else 0
            if existing >= len(strategies):
                continue

            cmd = [
                sys.executable, str(V8_SCRIPT),
                "--dataset", "cifar10", "--budget-level", "ultra_low",
                "--model-type", "simplecnn",
                "--strategies", *strategies,
                "--seeds", str(seed),
                "--imbalance-ratio", str(rho),
                "--output-dir", str(out),
                "--use-ssl", "--ssl-method", "flexmatch",
            ]
            agg = out / "aggregated_results.json"
            if agg.exists():
                try:
                    d = json.load(open(agg))
                    if d.get("full_supervision", {}).get("f1", 0) > 0:
                        cmd.append("--skip-full-sup")
                except: pass

            run_cmd(cmd, f"V3-Extended rho={rho} seed={seed}")


# ============================================================
# Priority 4: CIFAR-100 cross-dataset validation
# ============================================================
def priority4_cifar100():
    log("\n" + "=" * 60)
    log("Priority 4: CIFAR-100 Cross-Dataset Validation")
    log("=" * 60)

    output_dir = PROJECT_ROOT / "output" / "v3_cifar100"
    strategies = ["class_aware_entropy_ssl", "gap_aware_entropy_ssl"]

    for rho in [10, 50]:
        out = output_dir / f"rho{rho}"
        out.mkdir(parents=True, exist_ok=True)

        for seed in SEEDS:
            ckpt_dir = out / "checkpoints"
            existing = sum(1 for s in strategies
                          for f in ckpt_dir.glob(f"{s}_seed{seed}.json")) if ckpt_dir.exists() else 0
            if existing >= len(strategies):
                continue

            cmd = [
                sys.executable, str(V8_SCRIPT),
                "--dataset", "cifar100", "--budget-level", "ultra_low",
                "--model-type", "simplecnn",
                "--strategies", *strategies,
                "--seeds", str(seed),
                "--imbalance-ratio", str(rho),
                "--output-dir", str(out),
                "--use-ssl", "--ssl-method", "flexmatch",
                "--class-aware-adaptive",
                "--class-aware-soft-weighting",
            ]
            agg = out / "aggregated_results.json"
            if agg.exists():
                try:
                    d = json.load(open(agg))
                    if d.get("full_supervision", {}).get("f1", 0) > 0:
                        cmd.append("--skip-full-sup")
                except: pass

            run_cmd(cmd, f"V3-CIFAR100 rho={rho} seed={seed}")


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="V3 All Experiments Scheduler")
    parser.add_argument("--start-from", type=int, default=1, help="Start from priority N (1-4)")
    args = parser.parse_args()

    log("=" * 60)
    log("V3 Experiment Scheduler")
    log(f"Seeds: {SEEDS}, Rhos: {RHOS}")
    log("=" * 60)

    start = time.time()
    tasks = [
        (1, "V3 SSL Strategies", priority1_v3_ssl),
        (2, "V3 Pure AL Strategies", priority2_v3_al),
        (3, "Extended Strategies", priority3_extended),
        (4, "CIFAR-100 Validation", priority4_cifar100),
    ]

    for num, name, func in tasks:
        if num < args.start_from:
            log(f"\n[SKIP] Priority {num}: {name}")
            continue
        try:
            func()
        except Exception as e:
            log(f"ERROR in Priority {num}: {e}")

    elapsed = (time.time() - start) / 60
    log(f"\n{'='*60}")
    log(f"All V3 experiments completed in {elapsed:.0f} minutes")
    log("=" * 60)


if __name__ == "__main__":
    main()
