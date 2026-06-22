"""
V7 Semi-Supervised Learning Utilities
=====================================
Contains FixMatch and FlexMatch components for deep AL+SSL integration.

Key components:
  - FlexMatchTracker: Per-batch EMA-based dynamic threshold adjustment
  - UnlabeledDataset: Dual-augmentation wrapper for FixMatch consistency
  - get_strong_transforms: RandAugment-based strong perturbation pipelines
  - make_longtail_indices: Standard exponential long-tail benchmark builder
  - eval_pseudo_labels: Per-class pseudo-label quality diagnostics
"""

import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from PIL import Image
from collections import Counter


class FlexMatchTracker:
    """
    Dynamically adjusts the confidence threshold for pseudo-labels based on
    the learning status of each class (FlexMatch approach).

    Uses per-batch EMA updates (matching the original FlexMatch paper) instead
    of coarse per-epoch updates. An EMA momentum parameter smooths the
    per-class accuracy estimates to prevent threshold oscillation from
    single-batch statistical noise.
    """
    def __init__(
        self,
        n_classes: int,
        threshold: float = 0.95,
        ema_momentum: float = 0.9,
        min_threshold: float = 0.70,
    ):
        self.n_classes = n_classes
        self.base_threshold = threshold
        self.ema_momentum = ema_momentum
        self.min_threshold = min(min_threshold, threshold)
        # EMA of per-class "learning effect" (fraction exceeding base threshold)
        self.classwise_acc = torch.zeros(n_classes)
        self.current_thresholds = np.full(n_classes, threshold)
        # Whether each class has ever been observed (for warm start)
        self._class_seen = torch.zeros(n_classes, dtype=torch.bool)

    def update_per_batch(self, weak_probs: torch.Tensor):
        """
        Update per-class learning effect using weak-augmentation predictions
        from the current batch. Should be called once per FixMatch batch.

        Args:
            weak_probs: [B, C] softmax probabilities from weak-augmented images
        """
        max_probs, pseudo_labels = weak_probs.max(dim=1)
        batch_exceeds = (max_probs >= self.base_threshold).float()

        for c in range(self.n_classes):
            mask = pseudo_labels == c
            if mask.sum() > 0:
                class_effect = batch_exceeds[mask].mean()
                if self._class_seen[c]:
                    # EMA update
                    self.classwise_acc[c] = (
                        self.ema_momentum * self.classwise_acc[c]
                        + (1 - self.ema_momentum) * class_effect
                    )
                else:
                    # First observation: initialize directly
                    self.classwise_acc[c] = class_effect
                    self._class_seen[c] = True

        self._recompute_thresholds()

    def _recompute_thresholds(self):
        """Recompute per-class thresholds from current EMA state."""
        max_effect = self.classwise_acc.max().item()
        if max_effect < 1e-8:
            self.current_thresholds = np.full(self.n_classes, self.base_threshold)
            return

        thresholds = np.full(self.n_classes, self.base_threshold)
        for c in range(self.n_classes):
            if self._class_seen[c]:
                beta = self.classwise_acc[c].item() / max_effect
                # Keep FlexMatch adaptive, but do not allow early noisy classes
                # to collapse to a near-zero acceptance threshold.
                thresholds[c] = max(self.min_threshold, self.base_threshold * beta)
        self.current_thresholds = thresholds

    def update_and_get_thresholds(self, probs: np.ndarray) -> np.ndarray:
        """
        Bulk update from numpy probabilities (backward-compatible API).
        Used in the AL query phase for bulk pool evaluation.
        """
        if len(probs) == 0:
            return self.current_thresholds
        self.update_per_batch(torch.from_numpy(probs).float())
        return self.current_thresholds


class DeficitAwareFlexMatchTracker(FlexMatchTracker):
    """
    创新SSL阈值：在FlexMatch EMA基础上，叠加基于deficit的阈值调整。

    τ_c = max(τ_flexmatch(c) - α · deficit_norm(c), τ_min)

    其中:
      - τ_flexmatch(c) 是FlexMatch EMA计算的阈值
      - deficit_norm(c) = deficit(c) / max_c deficit(c)
      - deficit(c) = max(1/C - f_c^joint, 0)
      - f_c^joint = (n_c^labeled + n_c^pseudo) / N^joint

    效果：
      - 头类：deficit≈0 → τ_c≈τ_flexmatch（保持FlexMatch的质量控制）
      - 尾类：deficit≈1 → τ_c降低 → 更多尾类伪标签被接受
    """
    def __init__(
        self,
        n_classes: int,
        threshold: float = 0.95,
        ema_momentum: float = 0.9,
        min_threshold: float = 0.50,
        deficit_alpha: float = 0.25,
    ):
        super().__init__(n_classes, threshold, ema_momentum, min_threshold)
        self.deficit_alpha = deficit_alpha
        self._labeled_labels = None

    def set_labeled_distribution(self, labeled_labels: np.ndarray):
        """更新当前标注集的类别分布，用于计算deficit"""
        self._labeled_labels = labeled_labels

    def _recompute_thresholds(self):
        """先计算FlexMatch EMA阈值，再叠加deficit调整"""
        # Step 1: FlexMatch EMA阈值（与父类相同）
        max_effect = self.classwise_acc.max().item()
        if max_effect < 1e-8:
            flex_thresholds = np.full(self.n_classes, self.base_threshold)
        else:
            flex_thresholds = np.full(self.n_classes, self.base_threshold)
            for c in range(self.n_classes):
                if self._class_seen[c]:
                    beta = self.classwise_acc[c].item() / max_effect
                    flex_thresholds[c] = max(self.min_threshold, self.base_threshold * beta)

        # Step 2: 计算deficit
        if self._labeled_labels is not None and len(self._labeled_labels) > 0:
            labeled_counts = np.bincount(
                self._labeled_labels.astype(int), minlength=self.n_classes
            ).astype(np.float64)
            joint_freq = labeled_counts / (labeled_counts.sum() + 1e-10)
            mean_freq = 1.0 / self.n_classes
            deficit = np.maximum(mean_freq - joint_freq, 0.0)
            deficit_max = deficit.max()
            if deficit_max > 1e-8:
                deficit_norm = deficit / deficit_max
            else:
                deficit_norm = np.zeros(self.n_classes)
        else:
            deficit_norm = np.zeros(self.n_classes)

        # Step 3: 叠加deficit调整
        self.current_thresholds = np.maximum(
            flex_thresholds - self.deficit_alpha * deficit_norm,
            self.min_threshold
        )


class UnlabeledDataset(Dataset):
    """
    Dataset wrapper for FixMatch.
    Returns two versions of the same image: weakly augmented and strongly augmented.

    Expects a raw_dataset that returns (PIL_Image, label) without any transforms,
    or a dataset whose `.data` attribute provides raw numpy/tensor images.
    """
    def __init__(self, dataset, indices, transform_weak, transform_strong):
        self.dataset = dataset
        self.indices = indices
        self.transform_weak = transform_weak
        self.transform_strong = transform_strong

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        item = self.dataset[real_idx]

        # Unpack based on dataset return signature
        if isinstance(item, tuple) and len(item) == 2:
            img, label = item
        else:
            img = item
            label = -1

        # If the dataset has a `.data` attribute (torchvision CIFAR/MNIST),
        # access it directly to bypass pre-applied transforms.
        if hasattr(self.dataset, 'data'):
            img = self.dataset.data[real_idx]

        if isinstance(img, torch.Tensor):
            img = img.detach().cpu()
            if img.ndim in (2, 3):
                img = transforms.ToPILImage()(img)
            else:
                img = img.numpy()

        if isinstance(img, np.ndarray):
            if img.ndim == 2:
                img = Image.fromarray(img, mode='L')
            else:
                img = Image.fromarray(img)

        # At this point `img` should be a PIL Image
        img_w = self.transform_weak(img)
        img_s = self.transform_strong(img)

        return img_w, img_s, real_idx


def get_strong_transforms(dataset_name: str, base_norm: tuple):
    """
    Returns RandAugment-based strong transforms for FixMatch.
    """
    mean, std = base_norm

    if dataset_name in ["cifar10", "bloodmnist"]:
        size = 32 if dataset_name == "cifar10" else 28
        return transforms.Compose([
            transforms.RandomCrop(size, padding=int(size*0.125)),
            transforms.RandomHorizontalFlip(),
            transforms.RandAugment(num_ops=2, magnitude=10),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    elif dataset_name == "fashion_mnist":
        return transforms.Compose([
            transforms.RandomCrop(28, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandAugment(num_ops=2, magnitude=10),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    else:
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])


def make_longtail_indices(targets, imbalance_ratio, distribution="exp", seed=42):
    """
    Standard long-tail benchmark implementation (following LDAM-DRW setup).

    Creates an imbalanced subset by keeping different fractions of samples per class,
    following either an exponential decay or step function distribution.
    Only modifies sampling indices — the original dataset is never mutated.

    Args:
        targets: List/array of integer class labels for all training samples.
        imbalance_ratio: max_class_count / min_class_count ratio.
                         E.g. 100 means the largest class is 100x the smallest.
        distribution: "exp" for exponential decay (standard), "step" for step function.
        seed: Random seed for reproducibility.

    Returns:
        List of selected sample indices.
    """
    rng = np.random.default_rng(seed)
    targets_arr = np.array(targets)
    class_counts = Counter(targets_arr.tolist())
    n_classes = len(class_counts)

    # Sort classes by count descending
    sorted_classes = sorted(class_counts, key=class_counts.get, reverse=True)
    all_indices = np.arange(len(targets_arr))

    selected_indices = []
    for rank, cls in enumerate(sorted_classes):
        cls_indices = all_indices[targets_arr == cls]

        if distribution == "exp":
            # Exponential decay: n_i = n_max * (1/rho)^(i/(C-1))
            ratio = (1.0 / imbalance_ratio) ** (rank / max(1, n_classes - 1))
        else:  # step
            ratio = 1.0 if rank < n_classes // 2 else (1.0 / imbalance_ratio)

        keep_n = max(1, int(len(cls_indices) * ratio))
        chosen = rng.choice(cls_indices, keep_n, replace=False)
        selected_indices.extend(chosen.tolist())

    return selected_indices


def eval_pseudo_labels(pseudo_labels_tensor, mask_tensor, true_labels_tensor, n_classes):
    """
    Compute detailed per-class pseudo-label quality statistics.

    Args:
        pseudo_labels_tensor: [B] predicted pseudo labels
        mask_tensor: [B] binary mask (1 = pseudo-label accepted)
        true_labels_tensor: [B] ground-truth labels
        n_classes: number of classes

    Returns:
        dict with overall and per-class accuracy/coverage metrics.
    """
    stats = {
        "n_total": int(mask_tensor.numel()),
        "n_pseudo": int(mask_tensor.sum().item()),
        "overall_acc": 0.0,
        "per_class_acc": {},
        "per_class_coverage": {},
        "per_pred_class_precision": {},
        "confirmation_bias": 0.0,
    }

    if stats["n_pseudo"] == 0:
        return stats

    # Overall accuracy of accepted pseudo-labels
    accepted = mask_tensor.bool()
    correct = (pseudo_labels_tensor[accepted] == true_labels_tensor[accepted])
    stats["overall_acc"] = correct.float().mean().item()

    # Per-class breakdown by true class. per_class_acc is recall-style
    # pseudo-label correctness among accepted samples whose true label is c.
    per_class_accs = []
    for c in range(n_classes):
        total_in_class = (true_labels_tensor == c).sum().item()
        true_class_mask = true_labels_tensor == c
        selected_true_class = true_class_mask & accepted
        n_selected = selected_true_class.sum().item()

        if n_selected > 0:
            class_correct = (pseudo_labels_tensor[selected_true_class] == c).float().mean().item()
            stats["per_class_acc"][str(c)] = round(class_correct, 4)
            per_class_accs.append(class_correct)
        else:
            stats["per_class_acc"][str(c)] = None

        if total_in_class > 0:
            stats["per_class_coverage"][str(c)] = round(n_selected / total_in_class, 4)
        else:
            stats["per_class_coverage"][str(c)] = 0.0

        pred_class_mask = (pseudo_labels_tensor == c) & accepted
        n_pred = pred_class_mask.sum().item()
        if n_pred > 0:
            pred_precision = (true_labels_tensor[pred_class_mask] == c).float().mean().item()
            stats["per_pred_class_precision"][str(c)] = round(pred_precision, 4)
        else:
            stats["per_pred_class_precision"][str(c)] = None

    # Confirmation bias = 1 - mean(per_class_acc) for true classes that have accepted pseudo-labels.
    if per_class_accs:
        stats["confirmation_bias"] = round(1.0 - np.mean(per_class_accs), 4)

    return stats
