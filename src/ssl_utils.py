"""
公共半监督学习工具函数
====================
提取自 v2 实验脚本的 SSL 逻辑
"""

from typing import List, Dict, Any, Optional, Tuple, Set
from collections import Counter

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset


class PseudoDataset(Dataset):
    def __init__(self, base_dataset, pseudo_labels: Dict[int, int]):
        self.base = base_dataset
        self.pseudo_labels = pseudo_labels

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        img, label = self.base[idx]
        if idx in self.pseudo_labels:
            label = self.pseudo_labels[idx]
        return img, label


def select_high_confidence_samples(
    probs: np.ndarray,
    pool_idx: List[int],
    threshold: float,
    max_samples: int,
    labeled_set: Set[int],
    true_labels: Optional[List[int]] = None,
    oracle_mode: bool = False,
    head_only: bool = False,
    head_class_threshold: int = 20000,
    class_counts: Optional[Dict[int, int]] = None,
) -> Tuple[List[int], List[int], Dict[str, Any]]:
    max_confs = np.max(probs, axis=1)
    pred_labels = np.argmax(probs, axis=1)

    quality: Dict[str, Any] = {
        "n_candidates": 0,
        "n_added": 0,
        "avg_confidence": 0.0,
        "pseudo_accuracy": None,
        "pseudo_class_dist": {},
    }

    hc_mask = max_confs >= threshold
    hc_indices = np.where(hc_mask)[0]

    if head_only and class_counts:
        head_classes = {c for c, cnt in class_counts.items() if cnt >= head_class_threshold}
        head_mask = np.array([pred_labels[i] in head_classes for i in range(len(pred_labels))])
        hc_indices = hc_indices[head_mask[hc_indices]]

    candidate_pairs = [
        (local_idx, pool_idx[local_idx])
        for local_idx in hc_indices
        if pool_idx[local_idx] not in labeled_set
    ]
    candidate_global_indices = [global_idx for _, global_idx in candidate_pairs]

    quality["n_candidates"] = len(candidate_global_indices)

    if len(candidate_global_indices) == 0:
        return [], [], quality

    if oracle_mode and true_labels is not None:
        selected = candidate_global_indices[:max_samples]
        pseudo_labels = [true_labels[idx] for idx in selected]
        selected_local = [local_idx for local_idx, _ in candidate_pairs[:max_samples]]
    else:
        candidate_local_indices = np.array([local_idx for local_idx, _ in candidate_pairs])
        sorted_indices = np.argsort(-max_confs[candidate_local_indices])
        selected = []
        pseudo_labels = []
        selected_local = []
        for i in sorted_indices[:max_samples]:
            local_idx = int(candidate_local_indices[i])
            global_idx = pool_idx[local_idx]
            selected.append(global_idx)
            pseudo_labels.append(int(pred_labels[local_idx]))
            selected_local.append(local_idx)

    quality["n_added"] = len(selected)
    if selected:
        quality["avg_confidence"] = float(np.mean(max_confs[selected_local]))
        if true_labels is not None:
            correct = sum(1 for i, idx in enumerate(selected) if true_labels[idx] == pseudo_labels[i])
            quality["pseudo_accuracy"] = correct / len(selected)
        quality["pseudo_class_dist"] = dict(Counter(pseudo_labels))

    return selected, pseudo_labels, quality


def apply_pseudo_labels(
    pseudo_labels_dict: Dict[int, int],
    new_indices: List[int],
    new_labels: List[int],
) -> Dict[int, int]:
    for idx, label in zip(new_indices, new_labels):
        pseudo_labels_dict[idx] = label
    return pseudo_labels_dict
