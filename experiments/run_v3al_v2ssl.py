"""
V3 AL + V2 SSL 实验
====================
AL策略: V3版 class_aware_entropy / gap_aware_entropy / adaptive_gap_entropy
        (自适应lambda + 软概率加权)
SSL方法: V2版 deficit阈值 + 类别加权损失

组合效果: V3改进的AL采样 + V2已验证的SSL创新

运行方式:
    python experiments/run_v3al_v2ssl.py
"""

import os
import sys
import subprocess
import json
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
V8_SCRIPT = PROJECT_ROOT / "experiments" / "v8_controlled_fast_al_ssl.py"

SEEDS = [42, 123, 456]
RHOS = [1, 10, 50, 100]
STRATEGIES = ["class_aware_entropy", "gap_aware_entropy", "adaptive_gap_entropy"]


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


def main():
    log("=" * 60)
    log("V3 AL + V2 SSL Experiment")
    log(f"Strategies: {STRATEGIES}")
    log(f"Seeds: {SEEDS}, Rhos: {RHOS}")
    log("=" * 60)

    output_dir = PROJECT_ROOT / "output" / "v3al_v2ssl"

    for rho in RHOS:
        out = output_dir / f"rho{rho}"
        out.mkdir(parents=True, exist_ok=True)

        # Check if full sup exists
        agg = out / "aggregated_results.json"
        has_full_sup = False
        if agg.exists():
            try:
                d = json.load(open(agg))
                if d.get("full_supervision", {}).get("f1", 0) > 0:
                    has_full_sup = True
            except: pass

        for seed in SEEDS:
            # Check if already done
            ckpt_dir = out / "checkpoints"
            if ckpt_dir.exists():
                existing = sum(1 for s in STRATEGIES
                              for f in ckpt_dir.glob(f"{s}_seed{seed}.json"))
                if existing >= len(STRATEGIES):
                    continue

            cmd = [
                sys.executable, str(V8_SCRIPT),
                "--dataset", "cifar10", "--budget-level", "ultra_low",
                "--model-type", "simplecnn",
                "--strategies", *STRATEGIES,
                "--seeds", str(seed),
                "--imbalance-ratio", str(rho),
                "--output-dir", str(out),
                # V3 AL: adaptive lambda + soft weighting (default in v8)
                "--class-aware-adaptive",
                "--class-aware-soft-weighting",
                # V2 SSL: deficit threshold + class-weighted
                "--use-ssl", "--ssl-method", "flexmatch",
                "--ssl-deficit-threshold", "--ssl-deficit-alpha", "0.25",
                "--ssl-class-weighted",
            ]
            if has_full_sup:
                cmd.append("--skip-full-sup")

            run_cmd(cmd, f"V3AL+V2SSL rho={rho} seed={seed}")

    # Summary
    log(f"\n{'='*60}")
    log("RESULTS SUMMARY")
    log(f"{'='*60}")
    header = "rho   "
    for s in STRATEGIES:
        header += f"{s:>25}"
    print(header)
    print("-" * 80)
    for rho in RHOS:
        agg = output_dir / f"rho{rho}" / "aggregated_results.json"
        if agg.exists():
            d = json.load(open(agg))
            line = f"{rho:<6}"
            for s in STRATEGIES:
                f1 = d.get(s, {}).get('final_f1_mean', 0)
                std = d.get(s, {}).get('final_f1_std', 0)
                line += f"{f1:.4f}+/-{std:.4f}".rjust(25)
            print(line)

    log("Done!")


if __name__ == "__main__":
    main()
