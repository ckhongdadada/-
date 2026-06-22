"""
方向4：长尾场景下的尾类感知不确定性策略
========================================
目标：在 vanilla Entropy 基础上设计轻量级尾类感知度量，在长尾场景下稳定优于 vanilla Entropy。

已有基础：
  adaptive_gap_entropy (λ=0.5) 在 CIFAR-10 ρ=100 下较 vanilla Entropy 提升 +3.1%
  但 class_aware_entropy 未显示一致优势

本实验扩展：
  1. 在更多数据集和 ρ 值上系统测试 adaptive_gap_entropy
  2. 设计新的 tail_aware_entropy 策略：仅对预测为尾类但置信度低的样本加权
  3. 验证策略在不同 ρ 下的优势区间

实验矩阵：
  数据集: CIFAR-10 (ρ ∈ {10, 50, 100}), ECG5000 (ρ=20), Dry Bean (ρ=27)
  策略: Random, Entropy, adaptive_gap_entropy, tail_aware_entropy
  SSL: FixMatch (CIFAR-10), TS Consistency (ECG5000), VAT (Dry Bean)
  种子: 5 seeds

运行方式：
  python innovation_d4_tail_aware/run_tail_aware.py --dataset cifar10 --rho 100
  python innovation_d4_tail_aware/run_tail_aware.py --all
  python innovation_d4_tail_aware/run_tail_aware.py --analyze-only
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
import torch.nn.functional as F
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
logger = logging.getLogger("d4_tail")


DATASET_CONFIGS = {
    "cifar10": {
        "n_classes": 10, "rho_values": [10, 50, 100],
        "n_initial": 500, "n_query": 500, "n_rounds": 10,
        "n_epochs": 5, "lr": 0.001, "batch_size": 64,
        "ssl_method": "fixmatch", "model_type": "simplecnn",
    },
    "ecg5000": {
        "n_classes": 5, "rho_values": [20],
        "n_initial": 50, "n_query": 30, "n_rounds": 10,
        "n_epochs": 8, "lr": 0.001, "batch_size": 128,
        "ssl_method": "ts_consistency", "model_type": "timeseriescnn",
    },
    "drybean": {
        "n_classes": 7, "rho_values": [27],
        "n_initial": 200, "n_query": 100, "n_rounds": 10,
        "n_epochs": 10, "lr": 0.001, "batch_size": 128,
        "ssl_method": "vat", "model_type": "simplecnn",
    },
}

STRATEGIES = ["random", "entropy", "margin", "adaptive_gap_entropy", "tail_aware_entropy"]


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


def load_dataset_for_tail_aware(dataset_name):
    if dataset_name == "cifar10":
        return v8.load_cifar10()
    elif dataset_name == "ecg5000":
        return v8.load_ucr_dataset("ecg5000")
    elif dataset_name == "drybean":
        train_set, test_set, infer_set, raw_set = pt.load_drybean()
        CFG.drybean_input_dim = train_set.features.shape[1]
        return train_set, test_set, infer_set, raw_set
    elif dataset_name == "covertype":
        train_set, test_set, infer_set, raw_set = pt.load_covertype()
        CFG.covertype_input_dim = train_set.features.shape[1]
        return train_set, test_set, infer_set, raw_set
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


def run_single_tail_aware_experiment(dataset_name, rho, strategy_name, seed, use_ssl, output_dir):
    cfg = DATASET_CONFIGS[dataset_name]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(seed)
    n_classes = cfg["n_classes"]

    logger.info(f"Starting: {dataset_name} ρ={rho} {strategy_name} seed={seed} ssl={use_ssl}")

    CFG.dataset = dataset_name
    CFG.use_ssl = False
    CFG.imbalance_ratio = rho
    CFG.model_type = cfg["model_type"]
    CFG.n_epochs_base = cfg["n_epochs"]
    CFG.learning_rate = cfg["lr"]
    CFG.batch_size_train = cfg["batch_size"]
    CFG.batch_size_infer = 256
    CFG.use_amp = False
    CFG.use_scheduler = False
    CFG.pretrained_features = ""
    CFG.ssl_method = "auto"
    CFG.n_initial = cfg["n_initial"]
    CFG.n_query = cfg["n_query"]
    CFG.n_rounds = cfg["n_rounds"]
    CFG.fast_4060_preset = False

    if dataset_name in v8.BUDGET_LEVELS.get("low", {}):
        preset = v8.BUDGET_LEVELS["low"][dataset_name]
        for k, v in preset.items():
            if not hasattr(CFG, k) or k in ("n_initial", "n_query", "n_rounds",
                                              "n_epochs_base", "learning_rate",
                                              "batch_size_train", "batch_size_infer"):
                setattr(CFG, k, v)

    train_dataset, test_dataset, infer_dataset, raw_dataset = load_dataset_for_tail_aware(dataset_name)

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
        "dataset": dataset_name, "rho": rho, "strategy": strategy_name,
        "seed": seed, "use_ssl": use_ssl,
        "f1_scores": [], "accuracies": [], "n_labeled": [],
        "per_class_f1_final": {},
    }

    for rd in range(cfg["n_rounds"]):
        model = v8.create_model(dataset_name, device)
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss()

        model.train()
        subset = Subset(train_dataset, labeled_idx)
        loader = DataLoader(subset, batch_size=cfg["batch_size"], shuffle=True, num_workers=0)

        for epoch in range(cfg["n_epochs"]):
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

        model.eval()
        acc, f1 = v8.evaluate(model, test_dataset, list(range(len(test_dataset))), device)

        result["f1_scores"].append(float(f1))
        result["accuracies"].append(float(acc))
        result["n_labeled"].append(len(labeled_idx))

        logger.info(f"  R{rd+1}/{cfg['n_rounds']} F1={f1:.4f}")

        if rd == cfg["n_rounds"] - 1:
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

        subpool_idx = pool_idx
        if len(subpool_idx) > 5000:
            sub_idx = rng.choice(len(subpool_idx), 5000, replace=False)
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

    fname = f"{dataset_name}_rho{rho}_{strategy_name}_ssl{int(use_ssl)}_seed{seed}.json"
    result_path = os.path.join(output_dir, fname)
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved: {result_path}")
    return result


def analyze_tail_aware_results(output_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    all_results = defaultdict(lambda: defaultdict(list))
    for fname in os.listdir(output_dir):
        if not fname.endswith(".json"):
            continue
        if fname.startswith("tail_aware_summary") or (fname.startswith("tail_aware_") and "_rho" not in fname):
            continue
        with open(os.path.join(output_dir, fname), "r") as f:
            r = json.load(f)
        if "dataset" not in r or "strategy" not in r or "f1_scores" not in r:
            continue
        key = (r["dataset"], r["rho"], r["strategy"])
        all_results[key]["final_f1"].append(r["f1_scores"][-1])
        all_results[key]["all_f1"].append(r["f1_scores"])
        all_results[key]["per_class_f1"].append(r.get("per_class_f1_final", {}))

    datasets_with_rho = set()
    for (ds, rho, strat) in all_results.keys():
        datasets_with_rho.add((ds, rho))

    for ds, rho in sorted(datasets_with_rho):
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        ax1 = axes[0]
        for strat in STRATEGIES:
            key = (ds, rho, strat)
            if key not in all_results:
                continue
            f1s = all_results[key]["all_f1"]
            mean_f1 = np.mean(f1s, axis=0)
            std_f1 = np.std(f1s, axis=0)
            rounds = list(range(len(mean_f1)))
            ax1.plot(rounds, mean_f1, marker="o", markersize=3, label=strat)
            ax1.fill_between(rounds, mean_f1 - std_f1, mean_f1 + std_f1, alpha=0.1)
        ax1.set_xlabel("AL Round")
        ax1.set_ylabel("Macro-F1")
        ax1.set_title(f"Learning Curves: {ds} ρ={rho}")
        ax1.legend(fontsize=8)
        ax1.grid(True, alpha=0.3)

        ax2 = axes[1]
        strat_names = []
        strat_means = []
        strat_stds = []
        for strat in STRATEGIES:
            key = (ds, rho, strat)
            if key not in all_results:
                continue
            f1s = all_results[key]["final_f1"]
            strat_names.append(strat.replace("adaptive_gap_entropy", "ada_gap").replace("tail_aware_entropy", "tail_aware"))
            strat_means.append(np.mean(f1s))
            strat_stds.append(np.std(f1s))
        if strat_names:
            x_pos = np.arange(len(strat_names))
            ax2.bar(x_pos, strat_means, yerr=strat_stds, capsize=3, alpha=0.7,
                    color=["gray", "steelblue", "coral", "seagreen", "mediumpurple"][:len(strat_names)])
            ax2.set_xticks(x_pos)
            ax2.set_xticklabels(strat_names, rotation=30, ha="right", fontsize=8)
            ax2.set_ylabel("Final Macro-F1")
            ax2.set_title(f"Final F1 Comparison: {ds} ρ={rho}")
            ax2.grid(True, alpha=0.3, axis="y")

        ax3 = axes[2]
        for strat in ["entropy", "adaptive_gap_entropy", "tail_aware_entropy"]:
            key = (ds, rho, strat)
            if key not in all_results:
                continue
            per_class_list = all_results[key]["per_class_f1"]
            if not per_class_list or not per_class_list[0]:
                continue
            n_cls = len(per_class_list[0])
            mean_per_class = {}
            for c in range(n_cls):
                vals = [pc.get(str(c), pc.get(c, 0)) for pc in per_class_list if pc]
                mean_per_class[c] = np.mean([v for v in vals if v is not None]) if vals else 0
            classes = sorted(mean_per_class.keys())
            ax3.plot([f"C{c}" for c in classes], [mean_per_class[c] for c in classes],
                     marker="s", markersize=3, label=strat.replace("adaptive_gap_entropy", "ada_gap"))
        ax3.set_xlabel("Class")
        ax3.set_ylabel("Per-class F1")
        ax3.set_title(f"Per-class F1: {ds} ρ={rho}")
        ax3.legend(fontsize=8)
        ax3.grid(True, alpha=0.3)

        plt.suptitle(f"Tail-Aware Strategy Analysis: {ds} ρ={rho}", fontsize=13)
        plt.tight_layout()
        fig_path = os.path.join(output_dir, f"tail_aware_{ds}_rho{rho}.png")
        plt.savefig(fig_path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"Figure saved: {fig_path}")

    summary = {"results": {}}
    for (ds, rho, strat), data in all_results.items():
        f1s = data["final_f1"]
        rnd_key = (ds, rho, "random")
        rnd_f1s = all_results[rnd_key]["final_f1"] if rnd_key in all_results else []
        delta_vs_random = float(np.mean(f1s) - np.mean(rnd_f1s)) if rnd_f1s else None
        summary["results"][f"{ds}_rho{rho}_{strat}"] = {
            "mean_f1": float(np.mean(f1s)),
            "std_f1": float(np.std(f1s)),
            "delta_vs_random": delta_vs_random,
        }
    summary_path = os.path.join(output_dir, "tail_aware_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    logger.info(f"Summary saved: {summary_path}")
    return summary


def main():
    parser = argparse.ArgumentParser(description="D4: Tail-Aware Strategy Experiment")
    parser.add_argument("--dataset", type=str, choices=list(DATASET_CONFIGS.keys()), default=None)
    parser.add_argument("--rho", type=int, default=None)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--analyze-only", action="store_true")
    args = parser.parse_args()

    output_base = os.path.join(os.path.dirname(__file__), "output")
    os.makedirs(output_base, exist_ok=True)

    if args.analyze_only:
        analyze_tail_aware_results(output_base)
        return

    experiments = []
    if args.all:
        for ds_name, cfg in DATASET_CONFIGS.items():
            for rho in cfg["rho_values"]:
                for strat in STRATEGIES:
                    for seed in args.seeds:
                        experiments.append((ds_name, rho, strat, seed, True))
                        experiments.append((ds_name, rho, strat, seed, False))
    elif args.dataset and args.rho:
        for strat in STRATEGIES:
            for seed in args.seeds:
                experiments.append((args.dataset, args.rho, strat, seed, True))
                experiments.append((args.dataset, args.rho, strat, seed, False))
    else:
        logger.error("Specify --dataset --rho or --all")
        return

    logger.info(f"Total experiments: {len(experiments)}")
    for ds, rho, strat, seed, ssl in experiments:
        try:
            run_single_tail_aware_experiment(ds, rho, strat, seed, ssl, output_base)
        except Exception as e:
            logger.error(f"Failed: {ds} ρ={rho} {strat} seed={seed} ssl={ssl}: {e}")
            import traceback
            traceback.print_exc()

    logger.info("All experiments done. Running analysis...")
    analyze_tail_aware_results(output_base)


if __name__ == "__main__":
    main()
