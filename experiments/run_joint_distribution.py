"""
联合分布感知实验（简单版）
===========================
AL策略: class_aware_entropy_ssl / gap_aware_entropy_ssl
        使用 labeled_labels + pseudo_labels(argmax不过滤) 的联合分布
        固定λ=0.5，硬argmax，不调参

对比: 标准AL策略 (entropy) + V2创新AL (class_aware_entropy, 非SSL版)

运行方式:
    python experiments/run_joint_distribution.py
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
    log("Joint Distribution Sensing Experiment (Simple)")
    log("=" * 60)

    output_dir = PROJECT_ROOT / "output" / "joint_distribution"

    # Config: (name, strategies, use_ssl, extra_args)
    configs = [
        ("baseline_entropy", ["entropy"], True, []),
        ("v2_class_aware", ["class_aware_entropy"], True, []),
        ("v2_gap_aware", ["gap_aware_entropy"], True, []),
        ("joint_class_aware", ["class_aware_entropy_ssl"], True, []),
        ("joint_gap_aware", ["gap_aware_entropy_ssl"], True, []),
    ]

    for rho in RHOS:
        for config_name, strategies, use_ssl, extra_args in configs:
            out = output_dir / config_name / f"rho{rho}"
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
                # Check if done
                ckpt_dir = out / "checkpoints"
                if ckpt_dir.exists():
                    existing = sum(1 for s in strategies
                                  for f in ckpt_dir.glob(f"{s}_seed{seed}.json"))
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
                    # V2 defaults: no adaptive lambda, no soft weighting
                    "--no-class-aware-adaptive",
                    "--no-class-aware-soft-weighting",
                ]
                if use_ssl:
                    cmd.extend(["--use-ssl", "--ssl-method", "flexmatch"])
                if has_full_sup:
                    cmd.append("--skip-full-sup")
                cmd.extend(extra_args)

                run_cmd(cmd, f"{config_name} rho={rho} seed={seed}")

    # Summary
    log(f"\n{'='*70}")
    log("RESULTS SUMMARY")
    log(f"{'='*70}")
    header = f"{'rho':<6}"
    for config_name, *_ in configs:
        header += f"{config_name:>22}"
    print(header)
    print("-" * (6 + 22 * len(configs)))

    for rho in RHOS:
        line = f"{rho:<6}"
        for config_name, strategies, *_ in configs:
            agg = output_dir / config_name / f"rho{rho}" / "aggregated_results.json"
            if agg.exists():
                d = json.load(open(agg))
                s = strategies[0]
                f1 = d.get(s, {}).get("final_f1_mean", 0)
                line += f"{f1:.4f}".rjust(22)
            else:
                line += "N/A".rjust(22)
        print(line)

    log("Done!")


if __name__ == "__main__":
    main()
