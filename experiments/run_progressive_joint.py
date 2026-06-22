"""
渐进式联合分布感知实验
=======================
测试不同切换时机的联合分布感知效果。

配置:
  - labeled_only: 纯AL，不使用联合分布
  - joint_r0: 全程联合分布（始终使用伪标签）
  - joint_r3: 前3轮纯AL，之后联合分布
  - joint_r5: 前5轮纯AL，之后联合分布
  - joint_r7: 前7轮纯AL，之后联合分布

运行方式:
    python experiments/run_progressive_joint.py
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
RHOS = [10, 50, 100]
STRATEGIES = ["class_aware_entropy_ssl", "gap_aware_entropy_ssl"]


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
    log("Progressive Joint Distribution Experiment")
    log("=" * 60)

    output_dir = PROJECT_ROOT / "output" / "progressive_joint"

    # Configs: (name, joint_start_round, strategies)
    configs = [
        ("labeled_only", -1, ["class_aware_entropy", "gap_aware_entropy"]),  # never use joint
        ("joint_r0", 0, STRATEGIES),     # always joint
        ("joint_r3", 3, STRATEGIES),     # switch at round 3
        ("joint_r5", 5, STRATEGIES),     # switch at round 5
        ("joint_r7", 7, STRATEGIES),     # switch at round 7
    ]

    for rho in RHOS:
        # Check if full sup exists for this rho
        first_out = output_dir / configs[0][0] / f"rho{rho}"
        has_full_sup = False
        agg = first_out / "aggregated_results.json"
        if agg.exists():
            try:
                d = json.load(open(agg))
                if d.get("full_supervision", {}).get("f1", 0) > 0:
                    has_full_sup = True
            except: pass

        for config_name, joint_start, strats in configs:
            out = output_dir / config_name / f"rho{rho}"
            out.mkdir(parents=True, exist_ok=True)

            for seed in SEEDS:
                # Check if done
                ckpt_dir = out / "checkpoints"
                if ckpt_dir.exists():
                    existing = sum(1 for s in strats
                                  for f in ckpt_dir.glob(f"{s}_seed{seed}.json"))
                    if existing >= len(strats):
                        continue

                cmd = [
                    sys.executable, str(V8_SCRIPT),
                    "--dataset", "cifar10", "--budget-level", "ultra_low",
                    "--model-type", "simplecnn",
                    "--strategies", *strats,
                    "--seeds", str(seed),
                    "--imbalance-ratio", str(rho),
                    "--output-dir", str(out),
                    "--use-ssl", "--ssl-method", "flexmatch",
                ]
                if joint_start >= 0:
                    cmd.extend(["--joint-start-round", str(joint_start)])
                if has_full_sup:
                    cmd.append("--skip-full-sup")

                run_cmd(cmd, f"{config_name} rho={rho} seed={seed}")

    # Summary
    log(f"\n{'='*70}")
    log("RESULTS SUMMARY")
    log(f"{'='*70}")

    for rho in RHOS:
        print(f"\n--- rho={rho} ---")
        print(f"{'Config':<15}", end='')
        for s in ["class_aware", "gap_aware"]:
            print(f"{s:>15}", end='')
        print()
        print("-" * 45)
        for config_name, *_ in configs:
            agg = output_dir / config_name / f"rho{rho}" / "aggregated_results.json"
            if agg.exists():
                d = json.load(open(agg))
                line = f"{config_name:<15}"
                for s in STRATEGIES:
                    f1 = d.get(s, {}).get("final_f1_mean", 0)
                    line += f"{f1:.4f}".rjust(15)
                print(line)

    log("Done!")


if __name__ == "__main__":
    main()
