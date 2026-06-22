"""
渐进式SSL完整实验
=================
补齐所有ρ值和种子的渐进式SSL实验。

实验矩阵:
  - ρ: 1, 5, 10, 20, 50, 100
  - seeds: 42, 123, 456
  - AL策略: entropy, class_aware_entropy, gap_aware_entropy, adaptive_gap_entropy
  - SSL配置: no_ssl, base_ssl, innov_ssl, progressive_r3, progressive_r5, progressive_r7
  - 总计: 6ρ × 3seeds × 4策略 × 6配置 = 432 实验

运行方式:
    python experiments/run_progressive_ssl_full.py
    python experiments/run_progressive_ssl_full.py --rho 10 50
"""

import os
import sys
import subprocess
import json
import time
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "output" / "progressive_ssl_full"
EXPERIMENT_SCRIPT = PROJECT_ROOT / "experiments" / "v8_controlled_fast_al_ssl.py"

RHO_LIST = [1, 5, 10, 20, 50, 100]
SEEDS = [42, 123, 456]
STRATEGIES = ["entropy", "class_aware_entropy", "gap_aware_entropy", "adaptive_gap_entropy"]

# SSL配置: (name, use_ssl, deficit_threshold, class_weighted, deficit_start_round)
SSL_CONFIGS = [
    ("no_ssl",          False, False, False, 0),
    ("base_ssl",        True,  False, False, 0),
    ("innov_ssl",       True,  True,  True,  0),
    ("progressive_r3",  True,  True,  True,  3),
    ("progressive_r5",  True,  True,  True,  5),
    ("progressive_r7",  True,  True,  True,  7),
]


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def count_checkpoints(ssl_name, rho, seed):
    """检查某个配置是否已完成"""
    ckpt_dir = OUTPUT_DIR / f"{ssl_name}_rho{rho}" / "checkpoints"
    if not ckpt_dir.exists():
        return 0
    return len(list(ckpt_dir.glob(f"*seed{seed}.json")))


def run_single(rho, seed, strategies, ssl_name, use_ssl, deficit, cw, start_rd, skip_full_sup=False):
    """运行单个实验"""
    group_dir = f"{ssl_name}_rho{rho}"
    out_dir = OUTPUT_DIR / group_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, str(EXPERIMENT_SCRIPT),
        "--dataset", "cifar10",
        "--budget-level", "ultra_low",
        "--model-type", "simplecnn",
        "--strategies", *strategies,
        "--seeds", str(seed),
        "--imbalance-ratio", str(rho),
        "--output-dir", str(out_dir),
    ]

    if use_ssl:
        cmd.extend(["--use-ssl", "--ssl-method", "flexmatch"])
    if deficit:
        cmd.extend(["--ssl-deficit-threshold", "--ssl-deficit-alpha", "0.25"])
    if cw:
        cmd.extend(["--ssl-class-weighted"])
    if start_rd > 0:
        cmd.extend(["--ssl-deficit-start-round", str(start_rd)])
    if skip_full_sup:
        cmd.extend(["--skip-full-sup"])

    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT),
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=1800)
    return result.returncode


def main():
    parser = argparse.ArgumentParser(description="Progressive SSL full experiment")
    parser.add_argument("--rho", type=int, nargs="+", default=RHO_LIST)
    args = parser.parse_args()

    log("=" * 60)
    log("Progressive SSL Full Experiment")
    log(f"rho: {args.rho}, seeds: {SEEDS}")
    log(f"strategies: {STRATEGIES}")
    log(f"ssl configs: {[c[0] for c in SSL_CONFIGS]}")
    log("=" * 60)

    total = len(args.rho) * len(SEEDS) * len(SSL_CONFIGS)
    done = 0
    failed = 0
    skipped = 0

    # Track which rho values have already run full supervised
    full_sup_done = set()

    for rho in args.rho:
        for seed in SEEDS:
            for ssl_name, use_ssl, deficit, cw, start_rd in SSL_CONFIGS:
                # 检查是否已完成
                existing = count_checkpoints(ssl_name, rho, seed)
                if existing >= len(STRATEGIES):
                    skipped += 1
                    # Mark full sup as done if this config has aggregated results
                    agg_path = OUTPUT_DIR / f"{ssl_name}_rho{rho}" / "aggregated_results.json"
                    if agg_path.exists():
                        full_sup_done.add(rho)
                    continue

                # Skip full supervised if already computed for this rho
                skip_fs = rho in full_sup_done

                log(f"[{done+failed+skipped+1}/{total}] rho={rho} seed={seed} ssl={ssl_name}" +
                    (" (skip_full_sup)" if skip_fs else ""))
                try:
                    rc = run_single(rho, seed, STRATEGIES, ssl_name, use_ssl, deficit, cw, start_rd,
                                    skip_full_sup=skip_fs)
                except subprocess.TimeoutExpired:
                    rc = -1
                    log(f"  TIMEOUT")

                if rc == 0:
                    done += 1
                    full_sup_done.add(rho)
                    log(f"  OK")
                else:
                    failed += 1
                    log(f"  FAIL (rc={rc})")

    log(f"\n{'='*60}")
    log(f"Done! total={total} completed={done} skipped={skipped} failed={failed}")
    log("=" * 60)


if __name__ == "__main__":
    main()
