"""
尾类感知策略实验 - 100/100统一配置
===================================
5策略 × 5种子 × 3个ρ值 (仅CIFAR-10)
策略: random, entropy, margin, adaptive_gap_entropy, tail_aware_entropy
ρ值: 10, 50, 100
配置: n_initial=100, n_query=100, n_rounds=10, 无SSL

与标准AL实验(100/100)保持一致，便于公平对比。
注意：旧版尾类感知实验使用500/500+SSL，本版本去除SSL以公平对比。

运行方式:
    # 运行全部（约3-4小时）
    python experiments/run_tail_aware_100.py

    # 运行单个ρ值
    python experiments/run_tail_aware_100.py --rho 100

    # 快速测试
    python experiments/run_tail_aware_100.py --quick
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import f1_score, accuracy_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"
sys.path.insert(0, str(SRC_DIR))
sys.path.insert(0, str(EXPERIMENTS_DIR))

import importlib.util

_pt_path = EXPERIMENTS_DIR / "v8_phase_transition.py"
_pt_spec = importlib.util.spec_from_file_location("v8_phase_transition", str(_pt_path))
pt = importlib.util.module_from_spec(_pt_spec)
_pt_spec.loader.exec_module(pt)

pt.patch_v8_for_covertype()

v8 = pt.v8
CFG = v8.CFG

from deep_query_utils import (
    select_random,
    select_uncertainty,
    select_margin,
    select_adaptive_gap_entropy,
)
from ssl_v7_utils import make_longtail_indices

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("tail_aware_100")

# ====== 100/100统一配置 ======
STRATEGIES = ["random", "entropy", "margin", "adaptive_gap_entropy", "tail_aware_entropy"]
SEEDS = [42, 123, 456, 789, 1024]
RHO_VALUES = [10, 50, 100]

CIFAR10_CONFIG = {
    "n_classes": 10,
    "n_initial": 100,
    "n_query": 100,
    "n_rounds": 10,
    "n_epochs": 5,
    "lr": 0.001,
    "batch_size": 64,
    "model_type": "simplecnn",
}


def select_tail_aware_entropy(
    probs, pool_idx, n_query, labeled_labels, n_classes,
    tail_boost=2.0,
):
    n_select = min(n_query, len(pool_idx))
    probs = np.clip(probs.astype(np.float32, copy=False), 1e-7, 1.0)

    entropy = -np.sum(probs * np.log(probs), axis=1)
    max_entropy = np.log(n_classes)
    entropy_norm = entropy / max_entropy if max_entropy > 0 else entropy

    class_counts = np.bincount(labeled_labels.astype(int), minlength=n_classes).astype(np.float32)
    labeled_freq = class_counts / (class_counts.sum() + 1e-10)
    mean_freq = 1.0 / n_classes

    is_tail_class = labeled_freq < mean_freq
    tail_deficit = np.maximum(mean_freq - labeled_freq, 0.0)
    tail_deficit_norm = tail_deficit / (tail_deficit.max() + 1e-10)

    pred_classes = np.argmax(probs, axis=1)
    max_probs = np.max(probs, axis=1)

    tail_mask = np.array([is_tail_class[c] for c in pred_classes], dtype=np.float32)
    tail_weight = np.array([tail_deficit_norm[c] for c in pred_classes], dtype=np.float32)
    uncertainty_weight = 1.0 - max_probs

    tail_score = tail_mask * tail_weight * uncertainty_weight * tail_boost
    score = entropy_norm + tail_score
    top_k = np.argsort(score)[-n_select:]
    return [pool_idx[i] for i in top_k]


def run_single(rho, strategy_name, seed, output_dir, quick=False):
    cfg = CIFAR10_CONFIG
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(seed)
    n_classes = cfg["n_classes"]

    n_rounds = 2 if quick else cfg["n_rounds"]
    n_epochs = 2 if quick else cfg["n_epochs"]

    logger.info(f"Starting: CIFAR-10 ρ={rho} {strategy_name} seed={seed}")

    # 配置v8框架
    CFG.dataset = "cifar10"
    CFG.use_ssl = False  # 不使用SSL，与标准AL公平对比
    CFG.imbalance_ratio = rho
    CFG.model_type = cfg["model_type"]
    CFG.n_epochs_base = n_epochs
    CFG.learning_rate = cfg["lr"]
    CFG.batch_size_train = cfg["batch_size"]
    CFG.batch_size_infer = 256
    CFG.use_amp = True
    CFG.use_scheduler = True
    CFG.pretrained_features = ""
    CFG.n_initial = cfg["n_initial"]
    CFG.n_query = cfg["n_query"]
    CFG.n_rounds = n_rounds
    CFG.fast_4060_preset = True

    train_dataset, test_dataset, infer_dataset, raw_dataset = v8.load_cifar10()

    train_idx_all = list(range(len(train_dataset)))
    if rho > 1:
        targets = pt.gather_targets(train_dataset)
        filtered = make_longtail_indices(
            targets, imbalance_ratio=rho, distribution="exp", seed=42)
        train_idx_all = [train_idx_all[i] for i in filtered]
        logger.info(f"Long-tail filter: {len(targets)} -> {len(train_idx_all)} (ρ={rho})")

    all_indices = list(train_idx_all)
    n_initial = min(cfg["n_initial"], len(all_indices))
    labeled_idx = rng.choice(all_indices, n_initial, replace=False).tolist()
    pool_idx = [i for i in all_indices if i not in set(labeled_idx)]

    result = {
        "dataset": "cifar10", "rho": rho, "strategy": strategy_name,
        "seed": seed, "use_ssl": False,
        "config": {
            "n_initial": cfg["n_initial"], "n_query": cfg["n_query"],
            "n_rounds": n_rounds, "n_epochs": n_epochs,
            "model_type": cfg["model_type"],
        },
        "f1_scores": [], "accuracies": [], "n_labeled": [],
        "per_class_f1_final": {},
    }

    for rd in range(n_rounds):
        model = v8.create_model("cifar10", device)
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)
        criterion = nn.CrossEntropyLoss()

        model.train()
        subset = Subset(train_dataset, labeled_idx)
        loader = DataLoader(subset, batch_size=cfg["batch_size"], shuffle=True, num_workers=0)

        for epoch in range(n_epochs):
            for batch_data in loader:
                batch_x, batch_y = batch_data
                if isinstance(batch_x, np.ndarray):
                    batch_x = torch.from_numpy(batch_x)
                if isinstance(batch_y, np.ndarray):
                    batch_y = torch.from_numpy(batch_y)
                batch_x, batch_y = batch_x.to(device), batch_y.to(device).long()
                optimizer.zero_grad()
                logits = model(batch_x)
                loss = criterion(logits, batch_y)
                loss.backward()
                optimizer.step()
            scheduler.step()

        model.eval()
        acc, f1 = v8.evaluate(model, test_dataset, list(range(len(test_dataset))), device)

        result["f1_scores"].append(float(f1))
        result["accuracies"].append(float(acc))
        result["n_labeled"].append(len(labeled_idx))

        logger.info(f"  R{rd+1}/{n_rounds} F1={f1:.4f} Acc={acc:.4f} labeled={len(labeled_idx)}")

        if rd == n_rounds - 1:
            # 最后一轮：计算per-class F1
            test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=0)
            all_preds, all_labels = [], []
            with torch.no_grad():
                for bx, by in test_loader:
                    if isinstance(bx, np.ndarray):
                        bx = torch.from_numpy(bx)
                    bx = bx.to(device)
                    logits = model(bx)
                    preds = logits.argmax(dim=1).cpu().numpy()
                    all_preds.extend(preds)
                    if isinstance(by, torch.Tensor):
                        all_labels.extend(by.numpy())
                    else:
                        all_labels.extend(np.asarray(by))
            for c in range(n_classes):
                mask = np.array(all_labels) == c
                if mask.sum() > 0:
                    c_pred = np.array(all_preds)[mask]
                    c_true = np.array(all_labels)[mask]
                    tp = ((c_pred == c) & (c_true == c)).sum()
                    fp = ((c_pred == c) & (c_true != c)).sum()
                    fn = ((c_pred != c) & (c_true == c)).sum()
                    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
                    rec = tp / (tp + fn) if (tp + fn) > 0 else 0
                    f1_c = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
                    result["per_class_f1_final"][str(c)] = float(f1_c)
            break

        # 查询下一批样本
        subpool_idx = pool_idx
        if len(subpool_idx) > 3000:
            sub_idx = rng.choice(len(subpool_idx), 3000, replace=False)
            subpool_idx = [pool_idx[i] for i in sub_idx]

        probs, features = v8.get_probs_and_features(model, infer_dataset, subpool_idx, device)
        n_query = min(cfg["n_query"], len(subpool_idx))

        if strategy_name == "random":
            selected = select_random(subpool_idx, n_query, rng)
        elif strategy_name == "entropy":
            selected = select_uncertainty(probs, subpool_idx, n_query)
        elif strategy_name == "margin":
            selected = select_margin(probs, subpool_idx, n_query)
        elif strategy_name == "adaptive_gap_entropy":
            labeled_labels_arr = np.array([train_dataset[i][1] for i in labeled_idx])
            if isinstance(labeled_labels_arr[0], torch.Tensor):
                labeled_labels_arr = np.array([l.item() for l in labeled_labels_arr])
            selected = select_adaptive_gap_entropy(
                probs, subpool_idx, n_query, labeled_labels_arr, n_classes, lam_max=1.0)
        elif strategy_name == "tail_aware_entropy":
            labeled_labels_arr = np.array([train_dataset[i][1] for i in labeled_idx])
            if isinstance(labeled_labels_arr[0], torch.Tensor):
                labeled_labels_arr = np.array([l.item() for l in labeled_labels_arr])
            selected = select_tail_aware_entropy(
                probs, subpool_idx, n_query, labeled_labels_arr, n_classes,
                tail_boost=2.0)

        labeled_idx.extend(selected)
        labeled_set = set(labeled_idx)
        pool_idx = [i for i in pool_idx if i not in labeled_set]

    fname = f"cifar10_rho{rho}_{strategy_name}_seed{seed}.json"
    result_path = os.path.join(output_dir, fname)
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved: {result_path}")
    return result


def aggregate_results(output_dir, rho_values=None, strategies=None, seeds=None):
    """汇总所有实验结果"""
    if rho_values is None:
        rho_values = RHO_VALUES
    if strategies is None:
        strategies = STRATEGIES
    if seeds is None:
        seeds = SEEDS

    summary = {}
    for rho in rho_values:
        summary[rho] = {}
        for strategy in strategies:
            f1s = []
            for seed in seeds:
                fname = f"cifar10_rho{rho}_{strategy}_seed{seed}.json"
                fpath = os.path.join(output_dir, fname)
                if os.path.exists(fpath):
                    with open(fpath, "r") as f:
                        r = json.load(f)
                    f1s.append(r["f1_scores"][-1])
            if f1s:
                summary[rho][strategy] = {
                    "mean_f1": float(np.mean(f1s)),
                    "std_f1": float(np.std(f1s)),
                    "n_seeds": len(f1s),
                }

    # 保存汇总
    summary_path = os.path.join(output_dir, "tail_aware_100_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # 打印表格
    print("\n" + "=" * 70)
    print("尾类感知策略实验结果 (100/100配置, 无SSL)")
    print("=" * 70)
    header = f"{'策略':<25}"
    for rho in rho_values:
        header += f"{'ρ='+str(rho):>15}"
    print(header)
    print("-" * 70)
    for strategy in strategies:
        row = f"{strategy:<25}"
        for rho in rho_values:
            if rho in summary and strategy in summary[rho]:
                s = summary[rho][strategy]
                row += f"{s['mean_f1']:.4f}±{s['std_f1']:.3f}"
            else:
                row += f"{'N/A':>15}"
        print(row)
    print("=" * 70)

    return summary


def main():
    parser = argparse.ArgumentParser(description="尾类感知策略实验(100/100配置)")
    parser.add_argument("--rho", type=float, nargs="+", default=None, help="只运行指定ρ值")
    parser.add_argument("--strategies", type=str, nargs="+", default=None, help="只运行指定策略")
    parser.add_argument("--seeds", type=int, nargs="+", default=None, help="只运行指定种子")
    parser.add_argument("--quick", action="store_true", help="快速测试模式(2轮2epoch)")
    parser.add_argument("--analyze-only", action="store_true", help="只汇总已有结果")
    parser.add_argument("--output-dir", type=str, default=None, help="输出目录")
    args = parser.parse_args()

    rho_values = args.rho if args.rho else RHO_VALUES
    strategies = args.strategies if args.strategies else STRATEGIES
    seeds = args.seeds if args.seeds else SEEDS

    output_dir = args.output_dir or str(PROJECT_ROOT / "output" / "tail_aware_100")
    os.makedirs(output_dir, exist_ok=True)

    if args.analyze_only:
        aggregate_results(output_dir, rho_values, strategies, seeds)
        return

    print("=" * 60)
    print("尾类感知策略实验 - 100/100统一配置")
    print(f"策略: {strategies}")
    print(f"种子: {seeds}")
    print(f"ρ值: {rho_values}")
    print(f"总运行数: {len(rho_values)} × {len(strategies)} × {len(seeds)} = {len(rho_values)*len(strategies)*len(seeds)}")
    print(f"输出: {output_dir}")
    print("=" * 60)

    failed = []
    for rho in rho_values:
        for strategy in strategies:
            for seed in seeds:
                fname = f"cifar10_rho{rho}_{strategy}_seed{seed}.json"
                if os.path.exists(os.path.join(output_dir, fname)):
                    logger.info(f"Skip existing: {fname}")
                    continue
                try:
                    run_single(rho, strategy, seed, output_dir, args.quick)
                except Exception as e:
                    logger.error(f"FAILED: ρ={rho} {strategy} seed={seed}: {e}")
                    failed.append((rho, strategy, seed))

    # 汇总
    aggregate_results(output_dir, rho_values, strategies, seeds)

    if failed:
        print(f"\nFailed runs: {len(failed)}")
        for rho, strat, seed in failed:
            print(f"  ρ={rho} {strat} seed={seed}")
    else:
        print("\nALL EXPERIMENTS COMPLETED SUCCESSFULLY")


if __name__ == "__main__":
    main()
