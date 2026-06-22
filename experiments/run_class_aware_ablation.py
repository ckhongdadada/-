"""
Class-Aware Entropy 消融实验
============================
验证两个改进的效果：
1. 自适应 lambda (adaptive_lambda)
2. 软概率加权 (soft_weighting)

2×2 消融 + Entropy 基线 = 5 个配置

运行方式:
    python experiments/run_class_aware_ablation.py
"""

import os
import sys
import json
import time
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = str(PROJECT_ROOT / "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

# Need to set up v8 imports
sys.path.insert(0, str(PROJECT_ROOT / "experiments"))

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, TensorDataset, Subset
from sklearn.metrics import f1_score

from models import SimpleCNN
from deep_query_utils import select_class_aware_entropy, select_random


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def make_longtail_indices(targets, rho, seed=42):
    targets = np.array(targets)
    classes = np.unique(targets)
    n_classes = len(classes)
    rng = np.random.RandomState(seed)
    class_indices = {c: np.where(targets == c)[0] for c in classes}
    max_count = max(len(v) for v in class_indices.values())
    selected = []
    for c_idx, c in enumerate(classes):
        n_keep = max(1, int(max_count * (rho ** (-c_idx / (n_classes - 1)))))
        indices = class_indices[c]
        chosen = rng.choice(indices, min(n_keep, len(indices)), replace=False) if len(indices) > n_keep else indices
        selected.extend(chosen)
    return selected


def train_and_evaluate(train_x, train_y, test_x, test_y, n_classes, labeled_idx,
                       n_rounds=10, n_query=100, strategy="random",
                       adaptive_lambda=True, soft_weighting=True, seed=42):
    """Run a single active learning experiment."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = np.random.RandomState(seed)
    torch.manual_seed(seed)

    all_idx = list(range(len(train_y)))
    pool_idx = [i for i in all_idx if i not in set(labeled_idx)]
    test_loader = DataLoader(TensorDataset(test_x, test_y), batch_size=256, shuffle=False)

    f1_scores = []
    for rd in range(n_rounds):
        # Train
        model = SimpleCNN(num_classes=n_classes, in_channels=3).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5)

        subset = Subset(train_x[train_x.new_empty(0)] if False else train_x, labeled_idx)
        # Build dataset
        lx = train_x[labeled_idx]
        ly = train_y[labeled_idx]
        loader = DataLoader(TensorDataset(lx, ly), batch_size=64, shuffle=True, drop_last=False)

        for epoch in range(5):
            model.train()
            for bx, by in loader:
                bx, by = bx.to(device), by.to(device)
                optimizer.zero_grad()
                out = model(bx)
                loss = F.cross_entropy(out, by)
                loss.backward()
                optimizer.step()
            scheduler.step()

        # Evaluate
        model.eval()
        preds, labels = [], []
        with torch.no_grad():
            for bx, by in test_loader:
                out = model(bx.to(device))
                preds.extend(out.argmax(1).cpu().numpy())
                labels.extend(by.numpy())
        f1 = f1_score(labels, preds, average='macro', zero_division=0)
        f1_scores.append(f1)

        # Query
        if rd < n_rounds - 1:
            model.eval()
            all_probs = []
            with torch.no_grad():
                for i in range(0, len(train_x), 256):
                    bx = train_x[i:i+256].to(device)
                    probs = F.softmax(model(bx), dim=1)
                    all_probs.append(probs.cpu().numpy())
            all_probs = np.concatenate(all_probs, axis=0)
            pool_probs = all_probs[pool_idx]

            n_select = min(n_query, len(pool_idx))
            labeled_labels_np = train_y[labeled_idx].numpy()

            if strategy == "random":
                selected_local = rng.choice(len(pool_idx), n_select, replace=False)
            elif strategy == "class_aware":
                selected = select_class_aware_entropy(
                    pool_probs, pool_idx, n_select, labeled_labels_np, n_classes,
                    lam=0.5, adaptive_lambda=adaptive_lambda, soft_weighting=soft_weighting)
                selected_local = [pool_idx.index(s) for s in selected]
            else:
                selected_local = rng.choice(len(pool_idx), n_select, replace=False)

            new_idx = [pool_idx[i] for i in selected_local]
            labeled_idx = list(labeled_idx) + new_idx
            pool_idx = [i for i in pool_idx if i not in set(new_idx)]

    return f1_scores


def main():
    log("=" * 60)
    log("Class-Aware Entropy Ablation Experiment")
    log("=" * 60)

    # Load CIFAR-10
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])
    data_dir = str(PROJECT_ROOT / "data")
    train_dataset = torchvision.datasets.CIFAR10(data_dir, train=True, download=True, transform=transform)
    test_dataset = torchvision.datasets.CIFAR10(data_dir, train=False, download=True, transform=transform)

    train_x = torch.stack([train_dataset[i][0] for i in range(len(train_dataset))])
    train_y = torch.tensor([train_dataset[i][1] for i in range(len(train_dataset))])
    test_x = torch.stack([test_dataset[i][0] for i in range(len(test_dataset))])
    test_y = torch.tensor([test_dataset[i][1] for i in range(len(test_dataset))])

    n_classes = 10
    rhos = [1, 10, 50]
    seeds = [42, 123, 456]
    n_init = 100
    n_query = 100
    n_rounds = 10

    # 5 configurations
    configs = [
        ("entropy",            {"strategy": "random", "adaptive_lambda": False, "soft_weighting": False}),
        ("fixed_lam+hard",     {"strategy": "class_aware", "adaptive_lambda": False, "soft_weighting": False}),
        ("adaptive_lam+hard",  {"strategy": "class_aware", "adaptive_lambda": True,  "soft_weighting": False}),
        ("fixed_lam+soft",     {"strategy": "class_aware", "adaptive_lambda": False, "soft_weighting": True}),
        ("adaptive_lam+soft",  {"strategy": "class_aware", "adaptive_lambda": True,  "soft_weighting": True}),
    ]

    output_dir = PROJECT_ROOT / "output" / "class_aware_ablation"
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {}

    for rho in rhos:
        log(f"\n{'='*60}")
        log(f"rho = {rho}")
        log(f"{'='*60}")

        lt_idx = make_longtail_indices(train_y.numpy(), rho, seed=42)
        lt_x, lt_y = train_x[lt_idx], train_y[lt_idx]

        for config_name, config_params in configs:
            all_f1s = []
            for seed in seeds:
                # Initial labeled set
                rng = np.random.RandomState(seed)
                labeled_idx = list(rng.choice(len(lt_y), n_init, replace=False))

                f1s = train_and_evaluate(
                    lt_x, lt_y, test_x, test_y, n_classes, labeled_idx,
                    n_rounds=n_rounds, n_query=n_query, seed=seed, **config_params)
                all_f1s.append(f1s[-1])  # final F1

            mean_f1 = np.mean(all_f1s)
            std_f1 = np.std(all_f1s)
            key = f"rho{rho}_{config_name}"
            results[key] = {"mean_f1": mean_f1, "std_f1": std_f1, "f1s": all_f1s}
            log(f"  {config_name:<25} F1={mean_f1:.4f} ± {std_f1:.4f}")

    # Save results
    out_file = output_dir / "ablation_results.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)

    # Print summary table
    log(f"\n{'='*70}")
    log("ABLATION RESULTS SUMMARY")
    log(f"{'='*70}")
    print(f"\n{'Config':<25}", end='')
    for rho in rhos:
        print(f"{'rho='+str(rho):>12}", end='')
    print()
    print("-" * 65)
    for config_name, _ in configs:
        print(f"{config_name:<25}", end='')
        for rho in rhos:
            key = f"rho{rho}_{config_name}"
            f1 = results[key]["mean_f1"]
            print(f"{f1:>12.4f}", end='')
        print()

    # Compare with existing data
    log(f"\n{'='*70}")
    log("COMPARISON WITH EXISTING RESULTS")
    log(f"{'='*70}")
    for rho in rhos:
        # Get std_al entropy baseline
        std_al_file = PROJECT_ROOT / "output" / "std_al" / f"rho{rho}" / "aggregated_results.json"
        if std_al_file.exists():
            d = json.load(open(std_al_file))
            std_entropy = d.get("entropy", {}).get("final_f1_mean", 0)
            new_entropy = results.get(f"rho{rho}_entropy", {}).get("mean_f1", 0)
            new_full = results.get(f"rho{rho}_adaptive_lam+soft", {}).get("mean_f1", 0)
            log(f"  rho={rho}: std_al entropy={std_entropy:.4f}, "
                f"ablation entropy={new_entropy:.4f}, "
                f"new class_aware={new_full:.4f}")

    log(f"\nSaved to {out_file}")


if __name__ == "__main__":
    main()
