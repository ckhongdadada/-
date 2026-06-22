"""
渐进式SSL实验
=============
前N轮用Base SSL（固定阈值），之后切换到Innov SSL（deficit阈值+类别加权）。
验证"早期保证质量，后期引入创新"是否优于全程Innov SSL。

实验设计:
  - 1 seed (42), 1 rho (10)
  - 对比: Base SSL vs Innov SSL vs Progressive SSL (start_round=3,5,7)

运行方式:
    python experiments/run_progressive_ssl.py
"""

import os
import sys
import subprocess
import json
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "output" / "progressive_ssl"
EXPERIMENT_SCRIPT = PROJECT_ROOT / "experiments" / "v8_controlled_fast_al_ssl.py"

SEED = 42
RHO = 10
STRATEGIES = ["entropy", "class_aware_entropy"]


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def run_experiment(name, strategies, use_ssl=False, deficit_threshold=False,
                   class_weighted=False, deficit_start_round=0):
    """运行单个实验配置"""
    output_dir = OUTPUT_DIR / name
    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, str(EXPERIMENT_SCRIPT),
        "--dataset", "cifar10",
        "--budget-level", "ultra_low",
        "--model-type", "simplecnn",
        "--strategies", *strategies,
        "--seeds", str(SEED),
        "--imbalance-ratio", str(RHO),
        "--output-dir", str(output_dir),
    ]

    if use_ssl:
        cmd.extend(["--use-ssl", "--ssl-method", "flexmatch"])
    if deficit_threshold:
        cmd.extend(["--ssl-deficit-threshold", "--ssl-deficit-alpha", "0.25"])
    if class_weighted:
        cmd.extend(["--ssl-class-weighted"])
    if deficit_start_round > 0:
        cmd.extend(["--ssl-deficit-start-round", str(deficit_start_round)])

    log(f"START: {name}")
    log(f"CMD: {' '.join(cmd[:12])}...")
    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    if result.returncode == 0:
        log(f"DONE: {name}")
    else:
        log(f"FAIL: {name} (exit code {result.returncode})")
    return result.returncode


def load_results(name):
    """加载实验结果"""
    f = OUTPUT_DIR / name / "aggregated_results.json"
    if f.exists():
        with open(f) as fh:
            return json.load(fh)
    return None


def main():
    log("=" * 60)
    log("Progressive SSL Experiment")
    log(f"Seed={SEED}, Rho={RHO}")
    log("=" * 60)

    experiments = [
        # (name, strategies, use_ssl, deficit_threshold, class_weighted, deficit_start_round)
        ("01_no_ssl", STRATEGIES, False, False, False, 0),
        ("02_base_ssl", STRATEGIES, True, False, False, 0),
        ("03_innov_ssl", STRATEGIES, True, True, True, 0),
        ("04_progressive_r3", STRATEGIES, True, True, True, 3),
        ("05_progressive_r5", STRATEGIES, True, True, True, 5),
        ("06_progressive_r7", STRATEGIES, True, True, True, 7),
    ]

    for name, strats, use_ssl, deficit, cw, start_rd in experiments:
        # 检查是否已完成
        existing = load_results(name)
        if existing:
            log(f"[SKIP] {name} already done")
            continue

        rc = run_experiment(name, strats, use_ssl, deficit, cw, start_rd)
        if rc != 0:
            log(f"WARNING: {name} failed")

    # 汇总结果
    log("\n" + "=" * 60)
    log("RESULTS SUMMARY")
    log("=" * 60)

    print(f"\n{'Config':<25} {'Entropy F1':>12} {'ClassAware F1':>14}")
    print("-" * 55)

    for name, strats, *_ in experiments:
        data = load_results(name)
        if data is None:
            print(f"{name:<25} {'N/A':>12} {'N/A':>14}")
            continue

        ent_f1 = data.get("entropy", {}).get("final_f1_mean", 0)
        ca_f1 = data.get("class_aware_entropy", {}).get("final_f1_mean", 0)
        print(f"{name:<25} {ent_f1:>12.4f} {ca_f1:>14.4f}")

    log("\nDone!")


if __name__ == "__main__":
    main()
