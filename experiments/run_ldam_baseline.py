#!/usr/bin/env python
"""
LDAM-DRW 全监督基线实验
对比"损失函数修正"(LDAM) vs "标注策略优化"(AL)
仅使用全标注长尾数据，不参与AL流程
"""
# 修复说明 (2026-06-13):
# 1. 仅保留全监督实验，移除AL实验（LDAM不适合少样本AL场景）
# 2. s=30 → s=10（SimpleCNN无BatchNorm，logit尺度不同于ResNet）
# 3. epochs=30 → epochs=50（与其他实验full_supervised_epochs对齐）
# 4. 添加CosineAnnealingLR学习率调度
# 5. 添加梯度裁剪（max_norm=5.0）
# 6. 添加CIFAR-10归一化（与v8引擎一致）
# 7. SGD → Adam lr=0.001（与v8引擎一致）
import os, sys, json, time, argparse, logging
import numpy as np
from pathlib import Path
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, TensorDataset, Dataset, Subset
from sklearn.metrics import f1_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = str(PROJECT_ROOT / "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from models import SimpleCNN

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("ldam")

SEEDS = [42, 123, 456]


# ─── Long-tail indices ───────────────────────────────────────────────
def make_longtail_indices(targets, imbalance_ratio, seed=42):
    """Exponential decay long-tail sampling"""
    targets = np.array(targets)
    classes = np.unique(targets)
    n_classes = len(classes)
    rng = np.random.RandomState(seed)

    # Count per class
    class_indices = {c: np.where(targets == c)[0] for c in classes}
    max_count = max(len(v) for v in class_indices.values())

    selected = []
    for c_idx, c in enumerate(classes):
        n_keep = max(1, int(max_count * (imbalance_ratio ** (-c_idx / (n_classes - 1)))))
        indices = class_indices[c]
        if len(indices) > n_keep:
            chosen = rng.choice(indices, n_keep, replace=False)
        else:
            chosen = indices
        selected.extend(chosen)

    return selected


# ─── LDAM Loss ───────────────────────────────────────────────────────
class LDAMLoss(nn.Module):
    """Label-Distribution-Aware Margin Loss (Cao et al., ICLR 2020)"""
    def __init__(self, cls_num_list, max_m=0.5, s=10):
        super().__init__()
        cls_num_list = np.array(cls_num_list, dtype=np.float64)
        per_cls_weights = 1.0 / (cls_num_list + 1e-6)
        per_cls_weights = per_cls_weights / per_cls_weights.sum() * len(cls_num_list)
        self.per_cls_weights = torch.tensor(per_cls_weights, dtype=torch.float32)

        m_list = 1.0 / np.sqrt(np.sqrt(cls_num_list + 1e-6))
        m_list = m_list * (max_m / (m_list.max() + 1e-10))
        self.m_list = torch.tensor(m_list, dtype=torch.float32)
        self.s = s

    def forward(self, x, target):
        if self.m_list.device != x.device:
            self.m_list = self.m_list.to(x.device)
            self.per_cls_weights = self.per_cls_weights.to(x.device)
        index = torch.zeros_like(x, dtype=torch.bool)
        index.scatter_(1, target.data.view(-1, 1), True)
        x_m = x.clone()
        x_m[index] -= self.m_list[target]
        x_m = x_m * self.s
        return F.cross_entropy(x_m, target, weight=self.per_cls_weights)


# ─── Training ────────────────────────────────────────────────────────
def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    n = 0
    for bx, by in loader:
        bx, by = bx.to(device), by.to(device)
        optimizer.zero_grad()
        out = model(bx)
        loss = criterion(out, by)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        total_loss += loss.item() * len(by)
        n += len(by)
    return total_loss / max(n, 1)


def evaluate(model, loader, n_classes, device):
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for bx, by in loader:
            out = model(bx.to(device))
            preds.extend(out.argmax(1).cpu().numpy())
            labels.extend(by.numpy())
    f1 = f1_score(labels, preds, average='macro', zero_division=0)
    acc = float(np.mean(np.array(preds) == np.array(labels)))
    return f1, acc


# ─── Load CIFAR-10 ──────────────────────────────────────────────────
def load_cifar10_tensors():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])
    train = torchvision.datasets.CIFAR10(str(PROJECT_ROOT / "data"), train=True, download=True, transform=transform)
    test = torchvision.datasets.CIFAR10(str(PROJECT_ROOT / "data"), train=False, download=True, transform=transform)

    train_x = torch.stack([train[i][0] for i in range(len(train))])
    train_y = torch.tensor([train[i][1] for i in range(len(train))])
    test_x = torch.stack([test[i][0] for i in range(len(test))])
    test_y = torch.tensor([test[i][1] for i in range(len(test))])
    return train_x, train_y, test_x, test_y, 10


# ─── Experiments ─────────────────────────────────────────────────────
def run_full_supervised(train_x, train_y, test_x, test_y, n_classes,
                        use_ldam=False, seed=42, n_epochs=50, device="cpu"):
    """Full supervision with CE or LDAM-DRW"""
    rng = np.random.RandomState(seed)
    torch.manual_seed(seed)

    cls_counts = [int((train_y == c).sum()) for c in range(n_classes)]
    loader = DataLoader(TensorDataset(train_x, train_y), batch_size=128, shuffle=True)
    test_loader = DataLoader(TensorDataset(test_x, test_y), batch_size=256, shuffle=False)

    model = SimpleCNN(num_classes=n_classes, in_channels=3).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)
    ce_crit = nn.CrossEntropyLoss()
    ldam_crit = LDAMLoss(cls_counts, max_m=0.5, s=10) if use_ldam else None

    f1_scores = []
    for ep in range(n_epochs):
        criterion = (ce_crit if not use_ldam or ep < n_epochs // 2 else ldam_crit)
        train_epoch(model, loader, criterion, optimizer, device)
        scheduler.step()
        f1, acc = evaluate(model, test_loader, n_classes, device)
        f1_scores.append(f1)

    return {"f1_scores": f1_scores, "final_f1": f1_scores[-1], "best_f1": max(f1_scores)}


# ─── Main ────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="LDAM-DRW 全监督基线实验")
    parser.add_argument("--rho", type=int, nargs="+", default=[10, 50, 100])
    parser.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    parser.add_argument("--output-dir", type=str, default="output/ldam_baseline")
    parser.add_argument("--epochs", type=int, default=50)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # Load CIFAR-10 once
    logger.info("Loading CIFAR-10...")
    train_x, train_y, test_x, test_y, n_classes = load_cifar10_tensors()
    logger.info(f"Full train: {len(train_x)}, Test: {len(test_x)}")

    all_results = {}

    for rho in args.rho:
        logger.info(f"\n{'='*60}")
        logger.info(f"=== ρ = {rho} ===")
        logger.info(f"{'='*60}")

        # Create long-tail
        lt_idx = make_longtail_indices(train_y.numpy(), rho, seed=42)
        lt_x, lt_y = train_x[lt_idx], train_y[lt_idx]
        lt_counts = [int((lt_y == c).sum()) for c in range(n_classes)]
        logger.info(f"Long-tail train: {len(lt_x)}, Class counts: {lt_counts}")

        # 1. Full supervision: CE
        ce_results = []
        for seed in args.seeds:
            r = run_full_supervised(lt_x, lt_y, test_x, test_y, n_classes,
                                   use_ldam=False, seed=seed, n_epochs=args.epochs, device=device)
            ce_results.append(r)
        all_results[f"ce_full_rho{rho}"] = {
            "mean_f1": float(np.mean([r["final_f1"] for r in ce_results])),
            "std_f1": float(np.std([r["final_f1"] for r in ce_results])),
            "best_f1_mean": float(np.mean([r["best_f1"] for r in ce_results])),
        }
        logger.info(f"  CE Full ρ={rho}: {all_results[f'ce_full_rho{rho}']['mean_f1']:.4f} "
                     f"± {all_results[f'ce_full_rho{rho}']['std_f1']:.4f}")

        # 2. Full supervision: LDAM-DRW
        ldam_results = []
        for seed in args.seeds:
            r = run_full_supervised(lt_x, lt_y, test_x, test_y, n_classes,
                                   use_ldam=True, seed=seed, n_epochs=args.epochs, device=device)
            ldam_results.append(r)
        all_results[f"ldam_full_rho{rho}"] = {
            "mean_f1": float(np.mean([r["final_f1"] for r in ldam_results])),
            "std_f1": float(np.std([r["final_f1"] for r in ldam_results])),
            "best_f1_mean": float(np.mean([r["best_f1"] for r in ldam_results])),
        }
        logger.info(f"  LDAM Full ρ={rho}: {all_results[f'ldam_full_rho{rho}']['mean_f1']:.4f} "
                     f"± {all_results[f'ldam_full_rho{rho}']['std_f1']:.4f}")

    # Save
    output_file = output_dir / "ldam_results.json"
    with open(output_file, "w") as f:
        json.dump(all_results, f, indent=2)

    # Print summary
    logger.info(f"\n{'='*70}")
    logger.info("LDAM-DRW 全监督基线结果汇总")
    logger.info(f"{'='*70}")
    logger.info(f"{'Key':45s} {'F1':>8s} {'±std':>8s}")
    logger.info("-" * 70)
    for key, data in all_results.items():
        logger.info(f"{key:45s} {data['mean_f1']:.4f}   ±{data['std_f1']:.4f}")

    logger.info(f"\nSaved to {output_file}")


if __name__ == "__main__":
    main()