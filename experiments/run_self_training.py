"""
Self-training (Pseudo-Label) SSL 对比实验
=========================================
对比: 无SSL vs Self-training vs FlexMatch
目的: 展示SSL方法的递进关系

运行方式:
    python experiments/run_self_training.py
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
STRATEGIES = ["entropy"]


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
    log("Self-training SSL Comparison Experiment")
    log("=" * 60)

    output_dir = PROJECT_ROOT / "output" / "ssl_comparison"

    # Configs: (name, use_ssl, ssl_method, extra_args)
    configs = [
        ("no_ssl", False, None, []),
        ("self_training", True, "pseudo_label", []),
        ("flexmatch", True, "flexmatch", []),
    ]

    for rho in RHOS:
        for config_name, use_ssl, ssl_method, extra_args in configs:
            out = output_dir / config_name / f"rho{rho}"
            out.mkdir(parents=True, exist_ok=True)

            for seed in SEEDS:
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
                ]
                if use_ssl:
                    cmd.extend(["--use-ssl", "--ssl-method", ssl_method])
                cmd.extend(extra_args)

                # Skip full sup if exists
                agg = out / "aggregated_results.json"
                if agg.exists():
                    try:
                        d = json.load(open(agg))
                        if d.get("full_supervision", {}).get("f1", 0) > 0:
                            cmd.append("--skip-full-sup")
                    except: pass

                run_cmd(cmd, f"{config_name} rho={rho} seed={seed}")

    # Summary
    log(f"\n{'='*60}")
    log("RESULTS SUMMARY")
    log(f"{'='*60}")
    print(f"{'Config':<20} {'rho=1':>8} {'rho=10':>8} {'rho=50':>8} {'rho=100':>8}")
    print("-" * 50)
    for config_name, *_ in configs:
        line = f"{config_name:<20}"
        for rho in RHOS:
            agg = output_dir / config_name / f"rho{rho}" / "aggregated_results.json"
            if agg.exists():
                d = json.load(open(agg))
                f1 = d.get("entropy", {}).get("final_f1_mean", 0)
                line += f"{f1:.4f}".rjust(9)
            else:
                line += "       -"
        print(line)

    log("Done!")


if __name__ == "__main__":
    main()
