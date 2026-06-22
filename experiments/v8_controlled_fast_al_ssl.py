"""
V8 Controlled-Fast Active Learning + Semi-Supervised Learning Experiment
======================================================================
Builds on the v6 architecture with deep FixMatch + FlexMatch integration:
  - FixMatch consistency regularization (weak/strong augmentation)
  - FlexMatch per-batch EMA dynamic thresholding (per original paper)
  - SSL lambda_u warmup scheduling to suppress early-epoch noise
  - Long-tail exponential imbalance benchmarking (LDAM-DRW standard)
  - Per-class pseudo-label quality diagnostics (confirmation bias tracking)
  - ResNet-18 model support for CIFAR-10
  - Early stopping for full supervised baseline
  - Statistical significance tests (paired t-test, Cohen's d)
  - Computation cost tracking (wall-clock time, GPU memory)
  - Grad-CAM interpretability for uncertainty-sampled images
  - t-SNE feature evolution visualization + GIF animation
  - Checkpoint/resume functionality
  - YAML config file support

Strategies:
  1. Random Sampling (baseline)
  2. Least Confidence
  3. Margin Sampling
  4. Entropy Sampling
  5. CoreSet
  6. BADGE
  7. QBC (Query by Committee, heterogeneous)
  8. BALD (MC-Dropout)
  9. Learning Loss
"""

import os
import sys
import json
import time
import argparse
import logging
import traceback
import urllib.request
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple, Set
from collections import Counter, defaultdict
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
import torchvision
import torchvision.transforms as transforms
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix as sk_confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from deep_query_utils import (
    compute_grad_embeddings,
    select_badge,
    select_bald,
    select_coreset,
    select_dacs,
    select_random,
    select_uncertainty,
    select_least_confidence,
    select_margin,
    select_qbc,
    select_class_aware_entropy,
    select_gap_aware_entropy,
    select_adaptive_gap_entropy,
    select_two_stage_entropy_balance,
    select_curriculum_penalty_entropy,
    select_class_aware_entropy_ssl,
    select_gap_aware_entropy_ssl,
)
from visualize_utils import generate_and_save_gradcam
from models import SimpleCNN, TextMLPClassifier, TimeSeriesCNN, LossPredictionModule
from ssl_v7_utils import FlexMatchTracker, DeficitAwareFlexMatchTracker, UnlabeledDataset, get_strong_transforms, make_longtail_indices, eval_pseudo_labels

# Direction 1 & 2: Dynamic strategy switching and calibrated uncertainty sampling
from dynamic_strategy import (
    select_dynamic_switch,
    select_typiclust,
    select_uncertainty_coverage,
    get_default_switch_point,
)
from calibrated_query import (
    select_calibrated_entropy,
    select_calibrated_margin,
    compute_ece,
)

# Direction 5: Tabular model compatibility
from tabular_models import (
    create_tabular_model,
    train_tabular_model,
    evaluate_tabular_model,
    get_tabular_probs_and_features,
    TabularMLP,
    XGBoostTabularWrapper,
    LightGBMWrapper,
)

# Direction 3: Confirmation Bias suppression
from confirmation_bias import (
    inject_label_noise,
    compute_confirmation_bias_metrics,
    select_noise_aware_query,
    compute_pseudo_label_quality_evolution,
    detect_confirmation_bias_spike,
)

# Direction 4: Boundary-aware AL for imbalanced data
from boundary_aware import (
    select_boundary_aware_entropy,
    select_direct_style_query,
    compute_class_separation_quality,
    compute_boundary_distance,
)

logger = logging.getLogger(__name__)

ALL_STRATEGIES = [
    "random", "least_confidence", "margin", "entropy",
    "coreset", "badge", "qbc", "bald", "learning_loss",
    "class_aware_entropy", "gap_aware_entropy", "adaptive_gap_entropy",
    "two_stage_entropy_balance", "curriculum_penalty_entropy",
    "class_aware_entropy_ssl", "gap_aware_entropy_ssl",
    # Direction 1: Dynamic strategy switching
    "dynamic_typiclust_margin", "dynamic_typiclust_entropy",
    "dynamic_coreset_margin", "dynamic_coverage_uncertainty",
    # Direction 2: Calibrated uncertainty sampling
    "calibrated_entropy", "calibrated_margin",
    # Direction 3: Confirmation Bias robustness
    "noise_aware_entropy",
    # Direction 4: Boundary-aware AL
    "boundary_aware_entropy", "direct_style",
]

BUDGET_LEVELS = {
    "ultra_low": {
        "fashion_mnist": {"n_initial": 100, "n_query": 100, "n_rounds": 10, "n_epochs_base": 5, "learning_rate": 0.001, "batch_size_train": 64, "batch_size_infer": 128, "max_pool_subsample": 3000},
        "cifar10": {"n_initial": 100, "n_query": 100, "n_rounds": 10, "n_epochs_base": 5, "learning_rate": 0.001, "batch_size_train": 64, "batch_size_infer": 128, "max_pool_subsample": 3000},
        "cifar100": {"n_initial": 100, "n_query": 100, "n_rounds": 10, "n_epochs_base": 5, "learning_rate": 0.001, "batch_size_train": 64, "batch_size_infer": 128, "max_pool_subsample": 3000},
        "agnews": {"n_initial": 50, "n_query": 50, "n_rounds": 10, "n_epochs_base": 10, "learning_rate": 0.001, "batch_size_train": 128, "batch_size_infer": 256, "max_pool_subsample": 2000},
        "bloodmnist": {"n_initial": 100, "n_query": 100, "n_rounds": 10, "n_epochs_base": 5, "learning_rate": 0.001, "batch_size_train": 64, "batch_size_infer": 128, "max_pool_subsample": 3000},
        "adult": {"n_initial": 100, "n_query": 100, "n_rounds": 10, "n_epochs_base": 10, "learning_rate": 0.001, "batch_size_train": 128, "batch_size_infer": 256, "max_pool_subsample": 3000},
        "forda": {"n_initial": 50, "n_query": 50, "n_rounds": 10, "n_epochs_base": 8, "learning_rate": 0.001, "batch_size_train": 128, "batch_size_infer": 256, "max_pool_subsample": 2000},
        "ecg5000": {"n_initial": 50, "n_query": 30, "n_rounds": 10, "n_epochs_base": 8, "learning_rate": 0.001, "batch_size_train": 128, "batch_size_infer": 256, "max_pool_subsample": 3000},
        "spoken_arabic": {"n_initial": 200, "n_query": 100, "n_rounds": 10, "n_epochs_base": 8, "learning_rate": 0.001, "batch_size_train": 128, "batch_size_infer": 256, "max_pool_subsample": 5000},
        "character_traj": {"n_initial": 100, "n_query": 50, "n_rounds": 10, "n_epochs_base": 8, "learning_rate": 0.001, "batch_size_train": 128, "batch_size_infer": 256, "max_pool_subsample": 3000},
    },
    "low": {
        "fashion_mnist": {"n_initial": 500, "n_query": 500, "n_rounds": 10, "n_epochs_base": 5, "learning_rate": 0.001, "batch_size_train": 64, "batch_size_infer": 128, "max_pool_subsample": 5000},
        "cifar10": {"n_initial": 500, "n_query": 500, "n_rounds": 10, "n_epochs_base": 5, "learning_rate": 0.001, "batch_size_train": 64, "batch_size_infer": 128, "max_pool_subsample": 5000},
        "cifar100": {"n_initial": 500, "n_query": 500, "n_rounds": 10, "n_epochs_base": 5, "learning_rate": 0.001, "batch_size_train": 64, "batch_size_infer": 128, "max_pool_subsample": 5000},
        "agnews": {"n_initial": 200, "n_query": 100, "n_rounds": 10, "n_epochs_base": 10, "learning_rate": 0.001, "batch_size_train": 128, "batch_size_infer": 256, "max_pool_subsample": 3000},
        "bloodmnist": {"n_initial": 200, "n_query": 150, "n_rounds": 10, "n_epochs_base": 5, "learning_rate": 0.001, "batch_size_train": 64, "batch_size_infer": 128, "max_pool_subsample": 5000},
        "adult": {"n_initial": 400, "n_query": 200, "n_rounds": 10, "n_epochs_base": 10, "learning_rate": 0.001, "batch_size_train": 128, "batch_size_infer": 256, "max_pool_subsample": 5000},
        "forda": {"n_initial": 200, "n_query": 150, "n_rounds": 10, "n_epochs_base": 8, "learning_rate": 0.001, "batch_size_train": 128, "batch_size_infer": 256, "max_pool_subsample": 3000},
        "ecg5000": {"n_initial": 50, "n_query": 30, "n_rounds": 10, "n_epochs_base": 10, "learning_rate": 0.001, "batch_size_train": 128, "batch_size_infer": 256, "max_pool_subsample": 4000},
        "spoken_arabic": {"n_initial": 500, "n_query": 200, "n_rounds": 10, "n_epochs_base": 10, "learning_rate": 0.001, "batch_size_train": 128, "batch_size_infer": 256, "max_pool_subsample": 5000},
        "character_traj": {"n_initial": 200, "n_query": 100, "n_rounds": 10, "n_epochs_base": 10, "learning_rate": 0.001, "batch_size_train": 128, "batch_size_infer": 256, "max_pool_subsample": 4000},
    },
    "medium": {
        "fashion_mnist": {"n_initial": 1500, "n_query": 500, "n_rounds": 10, "n_epochs_base": 7, "learning_rate": 0.001, "batch_size_train": 64, "batch_size_infer": 128, "max_pool_subsample": 8000},
        "cifar10": {"n_initial": 1500, "n_query": 500, "n_rounds": 10, "n_epochs_base": 7, "learning_rate": 0.001, "batch_size_train": 64, "batch_size_infer": 128, "max_pool_subsample": 8000},
        "cifar100": {"n_initial": 1500, "n_query": 500, "n_rounds": 10, "n_epochs_base": 7, "learning_rate": 0.001, "batch_size_train": 64, "batch_size_infer": 128, "max_pool_subsample": 8000},
        "agnews": {"n_initial": 500, "n_query": 200, "n_rounds": 10, "n_epochs_base": 12, "learning_rate": 0.001, "batch_size_train": 128, "batch_size_infer": 256, "max_pool_subsample": 5000},
        "bloodmnist": {"n_initial": 500, "n_query": 300, "n_rounds": 10, "n_epochs_base": 7, "learning_rate": 0.001, "batch_size_train": 64, "batch_size_infer": 128, "max_pool_subsample": 8000},
        "adult": {"n_initial": 1000, "n_query": 300, "n_rounds": 10, "n_epochs_base": 12, "learning_rate": 0.001, "batch_size_train": 128, "batch_size_infer": 256, "max_pool_subsample": 6000},
        "forda": {"n_initial": 500, "n_query": 250, "n_rounds": 10, "n_epochs_base": 10, "learning_rate": 0.001, "batch_size_train": 128, "batch_size_infer": 256, "max_pool_subsample": 3000},
        "ecg5000": {"n_initial": 200, "n_query": 100, "n_rounds": 10, "n_epochs_base": 12, "learning_rate": 0.001, "batch_size_train": 128, "batch_size_infer": 256, "max_pool_subsample": 4000},
        "spoken_arabic": {"n_initial": 1000, "n_query": 300, "n_rounds": 10, "n_epochs_base": 12, "learning_rate": 0.001, "batch_size_train": 128, "batch_size_infer": 256, "max_pool_subsample": 5000},
        "character_traj": {"n_initial": 400, "n_query": 150, "n_rounds": 10, "n_epochs_base": 12, "learning_rate": 0.001, "batch_size_train": 128, "batch_size_infer": 256, "max_pool_subsample": 4000},
    },
    "high": {
        "fashion_mnist": {"n_initial": 3000, "n_query": 1000, "n_rounds": 10, "n_epochs_base": 8, "learning_rate": 0.001, "batch_size_train": 64, "batch_size_infer": 128, "max_pool_subsample": 10000},
        "cifar10": {"n_initial": 3000, "n_query": 1000, "n_rounds": 10, "n_epochs_base": 8, "learning_rate": 0.001, "batch_size_train": 64, "batch_size_infer": 128, "max_pool_subsample": 10000},
        "cifar100": {"n_initial": 3000, "n_query": 1000, "n_rounds": 10, "n_epochs_base": 8, "learning_rate": 0.001, "batch_size_train": 64, "batch_size_infer": 128, "max_pool_subsample": 10000},
        "agnews": {"n_initial": 1000, "n_query": 300, "n_rounds": 10, "n_epochs_base": 12, "learning_rate": 0.001, "batch_size_train": 128, "batch_size_infer": 256, "max_pool_subsample": 8000},
        "bloodmnist": {"n_initial": 500, "n_query": 300, "n_rounds": 10, "n_epochs_base": 8, "learning_rate": 0.001, "batch_size_train": 64, "batch_size_infer": 128, "max_pool_subsample": 5000},
        "adult": {"n_initial": 2000, "n_query": 500, "n_rounds": 10, "n_epochs_base": 15, "learning_rate": 0.001, "batch_size_train": 128, "batch_size_infer": 256, "max_pool_subsample": 8000},
        "forda": {"n_initial": 1000, "n_query": 300, "n_rounds": 10, "n_epochs_base": 12, "learning_rate": 0.001, "batch_size_train": 128, "batch_size_infer": 256, "max_pool_subsample": 3000},
        "ecg5000": {"n_initial": 300, "n_query": 150, "n_rounds": 10, "n_epochs_base": 15, "learning_rate": 0.001, "batch_size_train": 128, "batch_size_infer": 256, "max_pool_subsample": 4000},
        "spoken_arabic": {"n_initial": 2000, "n_query": 500, "n_rounds": 10, "n_epochs_base": 15, "learning_rate": 0.001, "batch_size_train": 128, "batch_size_infer": 256, "max_pool_subsample": 5000},
        "character_traj": {"n_initial": 800, "n_query": 200, "n_rounds": 10, "n_epochs_base": 15, "learning_rate": 0.001, "batch_size_train": 128, "batch_size_infer": 256, "max_pool_subsample": 4000},
    },
}
BLOODMNIST_CLASS_NAMES = [
    "basophil", "eosinophil", "erythroblast", "ig", "lymphocyte",
    "monocyte", "neutrophil", "platelet"
]


class ResNet18(nn.Module):
    def __init__(self, num_classes: int = 10, in_channels: int = 3):
        super().__init__()
        from torchvision.models import resnet18, ResNet18_Weights
        weights = ResNet18_Weights.DEFAULT if in_channels == 3 else None
        self.backbone = resnet18(weights=weights)
        if in_channels != 3:
            self.backbone.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_classes)
        self._feature_dim = self.backbone.fc.in_features

    def forward(self, x):
        return self.backbone(x)

    def get_features(self, x):
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)
        x = self.backbone.avgpool(x)
        return x.flatten(1)


class SmallCNN(nn.Module):
    def __init__(self, num_classes: int = 10, in_channels: int = 1, hidden_dim: int = 32):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, 3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(hidden_dim, hidden_dim * 2, 3, padding=1),
            nn.BatchNorm2d(hidden_dim * 2),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(7),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(hidden_dim * 2 * 7 * 7, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )
        self._feature_dim = hidden_dim * 2 * 7 * 7

    def forward(self, x):
        return self.classifier(self.features(x))

    def get_features(self, x):
        return self.features(x).flatten(1)

CONTROLLED_FAST_STRATEGIES = ["random", "margin", "entropy", "coreset", "badge", "qbc"]


@dataclass
class Config:
    output_dir: str = ""
    dataset: str = "fashion_mnist"
    budget_level: str = "low"
    model_type: str = "simplecnn"

    n_initial: int = 500
    n_query: int = 500
    n_rounds: int = 10
    strategies: List[str] = field(default_factory=lambda: CONTROLLED_FAST_STRATEGIES[:])

    batch_size_train: int = 64
    batch_size_infer: int = 128
    learning_rate: float = 0.001
    n_epochs_base: int = 5
    use_scheduler: bool = True

    max_pool_subsample: int = 5000
    seeds: List[int] = field(default_factory=lambda: [42, 123, 456])

    n_committee: int = 3
    committee_epochs: int = 3
    bald_mc_samples: int = 10
    qbc_reuse_committee: bool = True
    qbc_incremental_epochs: int = 1

    tfidf_dim: int = 5000
    tfidf_hidden: int = 256

    adult_input_dim: int = 100
    adult_hidden: int = 256

    full_supervised_epochs: int = 50
    early_stopping_patience: int = 5

    use_ssl: bool = False
    ssl_method: str = "auto"
    ssl_threshold: float = 0.95
    ssl_use_flexmatch: bool = True
    ssl_max_per_round: int = 500
    ssl_lambda_u: float = 1.0
    ssl_lambda_u_warmup_epochs: int = 10  # linearly ramp from 0 -> ssl_lambda_u across early AL rounds
    ssl_class_weighted: bool = False       # 创新点3: 类别加权一致性损失
    loss_type: str = "ce"                  # 损失函数: ce, cb (Class-Balanced), focal
    ssl_deficit_threshold: bool = False    # 创新点2: deficit-based自适应阈值
    ssl_deficit_alpha: float = 0.25        # deficit调整系数
    ssl_deficit_start_round: int = 0       # 渐进式SSL: 前N轮用Base SSL，之后切换Innov SSL (0=全程Innov)
    ssl_joint_distribution: bool = False   # 联合分布SSL: deficit和类权重基于labeled+pseudo计算
    ts_jitter_std: float = 0.03
    ts_scaling_std: float = 0.10
    ts_time_warp_sigma: float = 0.05
    ts_use_time_warp: bool = True
    dropout_consistency_passes: int = 2
    vat_epsilon: float = 1.0
    use_amp: bool = True
    fast_4060_preset: bool = True

    enable_tsne: bool = False
    enable_cost_tracking: bool = True

    checkpoint_dir: str = ""
    resume: bool = False

    # Experiment variants
    imbalance_ratio: float = 0.0  # 0 = balanced, >1 = max_class_count / min_class_count
    cold_start: bool = False  # ultra-small initial set mode
    ablation: str = ""  # ablation dimension: pool_size, epochs, initial_size, committee_size

    # Direction 1: Class-aware AL
    class_aware_lambda: float = 0.5
    class_aware_adaptive: bool = False  # True = V3 adaptive lambda (scales by skewness)
    class_aware_soft_weighting: bool = False  # True = V3 soft probability weighting (replaces hard argmax)
    curriculum_warmup_rounds: int = 5
    joint_start_round: int = 0  # 0 = always use joint distribution; N = use labeled-only for first N rounds

    # Direction 2: Pseudo-label dynamic refresh
    pseudo_refresh_freq: int = 0  # 0 = no refresh (default), K = refresh pseudo-labels every K AL rounds

    # Direction 4: Staged AL-SSL decoupling
    ssl_warmup_rounds: int = 0  # 0 = full coupling (default), N = pure supervised for first N rounds
    ssl_adaptive: bool = False  # True = enable SSL only when pseudo-label accuracy > threshold
    ssl_adaptive_threshold: float = 0.80  # pseudo-label accuracy threshold for adaptive coupling

    # Direction 3: Pretrained features
    pretrained_features: str = ""  # "", "clip", "sbert"

    # ECG5000 extended mode: merge train+test then re-split 80/20
    merge_train_test: bool = False

    # Direction 1: Dynamic strategy switching
    dynamic_strategy: str = "typiclust_to_margin"  # "typiclust_to_margin" | "typiclust_to_entropy" | "coreset_to_margin" | "coverage_to_uncertainty"
    dynamic_switch_point: Optional[float] = None  # None = auto, float = manual switch point
    dynamic_competence_method: str = "budget_ratio"  # "budget_ratio" | "slope" | "absolute"

    # Direction 2: Calibrated uncertainty sampling
    calibrated_n_bins: int = 15  # ECE bins for calibration evaluation

    # Direction 3: Confirmation Bias robustness
    label_noise_ratio: float = 0.0  # 0.0 = no noise, 0.05 = 5%, 0.10 = 10%
    label_noise_type: str = "uniform"  # "uniform" | "pairflip" | "asymmetric"

    # Direction 4: Boundary-aware AL
    boundary_aware_beta: float = 0.5  # boundary score weight
    boundary_aware_gamma: float = 2.0  # minority class bonus weight

    # Direction 5: Tabular model compatibility
    tabular_model_type: str = "mlp"  # "mlp" | "xgboost" | "lightgbm"
    tabular_n_estimators: int = 100
    tabular_max_depth: int = 6

CFG = Config()

IMAGE_SSL_DATASETS = {"fashion_mnist", "cifar10", "cifar100", "bloodmnist"}
PSEUDO_LABEL_SSL_DATASETS = {"agnews", "adult", "forda", "ecg5000", "spoken_arabic", "character_traj"}
SSL_SUPPORTED_DATASETS = IMAGE_SSL_DATASETS | PSEUDO_LABEL_SSL_DATASETS


def get_num_classes(dataset_name: str) -> int:
    if dataset_name == "agnews":
        return 4
    if dataset_name == "bloodmnist":
        return 8
    if dataset_name == "adult":
        return 2
    if dataset_name == "forda":
        return 2
    if dataset_name == "ecg5000":
        return 5
    if dataset_name == "spoken_arabic":
        return 10
    if dataset_name == "character_traj":
        return 20
    if dataset_name == "cifar100":
        return 100
    return 10


def apply_fast_4060_preset(explicit_bald=False, explicit_pool=False):
    """Fast non-equivalent defaults for RTX 4060 Laptop 8GB runs."""
    if CFG.dataset in ["fashion_mnist", "cifar10", "cifar100", "bloodmnist", "forda", "ecg5000", "spoken_arabic", "character_traj"]:
        CFG.batch_size_train = max(CFG.batch_size_train, 256)
        CFG.batch_size_infer = max(CFG.batch_size_infer, 512)
    else:
        CFG.batch_size_train = max(CFG.batch_size_train, 512)
        CFG.batch_size_infer = max(CFG.batch_size_infer, 1024)

    CFG.qbc_reuse_committee = True
    CFG.qbc_incremental_epochs = max(1, CFG.qbc_incremental_epochs)


def resolve_ssl_method(dataset_name: str, requested: str = "auto") -> str:
    if requested != "auto":
        return requested
    if dataset_name in IMAGE_SSL_DATASETS:
        return "flexmatch" if CFG.ssl_use_flexmatch else "fixmatch"
    if dataset_name == "agnews":
        return "dropout_consistency"
    if dataset_name == "adult":
        return "vat"
    if dataset_name == "forda":
        return "ts_consistency"
    if dataset_name == "ecg5000":
        return "ts_consistency"
    if dataset_name == "spoken_arabic":
        return "ts_consistency"
    if dataset_name == "character_traj":
        return "ts_consistency"
    return "pseudo_label"


def parse_args():
    parser = argparse.ArgumentParser(description="V8 controlled-fast AL+SSL experiment")
    parser.add_argument("--dataset", type=str, default=None, choices=["fashion_mnist", "cifar10", "cifar100", "agnews", "bloodmnist", "adult", "forda", "ecg5000", "spoken_arabic", "character_traj"])
    parser.add_argument("--budget-level", type=str, default=None, choices=list(BUDGET_LEVELS.keys()))
    parser.add_argument("--model-type", type=str, default=None, choices=["simplecnn", "resnet18", "timeseriescnn"])
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--skip-full-sup", action="store_true", help="Skip full supervised baseline (save time)")
    # skip_full_sup handled directly below
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--rounds", type=int, default=None)
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    parser.add_argument("--strategies", type=str, nargs="+", default=None, choices=ALL_STRATEGIES)
    parser.add_argument("--use-ssl", dest="use_ssl", action="store_true", default=None, help="Enable AL+SSL joint experiment")
    parser.add_argument("--no-use-ssl", dest="use_ssl", action="store_false", help="Disable AL+SSL joint experiment")
    parser.add_argument(
        "--ssl-method",
        type=str,
        default=None,
        choices=["auto", "pseudo_label", "fixmatch", "flexmatch", "dropout_consistency", "vat", "ts_consistency"],
        help="SSL method family. 'auto' maps each modality to the final experiment matrix.",
    )
    parser.add_argument("--ssl-threshold", type=float, default=None)
    parser.add_argument("--ssl-use-flexmatch", dest="ssl_use_flexmatch", action="store_true", default=None)
    parser.add_argument("--no-ssl-use-flexmatch", dest="ssl_use_flexmatch", action="store_false", help="Disable FlexMatch and use a fixed FixMatch threshold")
    parser.add_argument("--ts-jitter-std", type=float, default=None, help="Std of Gaussian jitter for time-series SSL strong view")
    parser.add_argument("--ts-scaling-std", type=float, default=None, help="Std of multiplicative scaling for time-series SSL strong view")
    parser.add_argument("--ts-time-warp-sigma", type=float, default=None, help="Relative time-warp displacement scale for time-series SSL strong view")
    parser.add_argument("--no-ts-time-warp", action="store_true", help="Disable time-warp augmentation for time-series SSL")
    parser.add_argument("--dropout-consistency-passes", type=int, default=None, help="Number of stochastic passes for dropout consistency")
    parser.add_argument("--vat-epsilon", type=float, default=None, help="Adversarial perturbation norm for tabular VAT")
    parser.add_argument("--no-amp", action="store_true", help="Disable CUDA automatic mixed precision")
    parser.add_argument("--enable-tsne", dest="enable_tsne", action="store_true", default=None, help="Enable t-SNE visualization")
    parser.add_argument("--no-enable-tsne", dest="enable_tsne", action="store_false", help="Disable t-SNE visualization")
    parser.add_argument("--resume", dest="resume", action="store_true", default=None, help="Resume from checkpoint")
    parser.add_argument("--no-resume", dest="resume", action="store_false", help="Start without checkpoint resume")
    parser.add_argument("--no-scheduler", action="store_true", help="Disable cosine LR scheduler")
    parser.add_argument("--bald-mc-samples", type=int, default=None, help="MC-Dropout forward passes for BALD")
    parser.add_argument("--fast-bald-mc-samples", type=int, default=None, help="Reduce BALD MC-Dropout samples for faster, non-equivalent runs")
    parser.add_argument("--max-pool-subsample", type=int, default=None, help="Override query/SSL pool subsample size; changes candidate-set semantics")
    parser.add_argument("--qbc-reuse-committee", dest="qbc_reuse_committee", action="store_true", default=None, help="Reuse QBC committee across rounds and fine-tune it; changes QBC semantics")
    parser.add_argument("--no-qbc-reuse-committee", dest="qbc_reuse_committee", action="store_false", help="Train a fresh QBC committee each round")
    parser.add_argument("--qbc-incremental-epochs", type=int, default=None, help="Fine-tuning epochs when QBC committee reuse is enabled")
    parser.add_argument("--no-fast-4060-preset", action="store_true", help="Disable RTX 4060 8GB fast defaults")
    parser.add_argument("--imbalance-ratio", type=float, default=None, help="Long-tail imbalance ratio max_class_count/min_class_count (0=balanced; e.g. 100)")
    parser.add_argument("--class-aware-lambda", type=float, default=None, help="λ for class-aware entropy strategy (default 0.5)")
    parser.add_argument("--no-class-aware-adaptive", dest="class_aware_adaptive", action="store_false", default=None, help="Disable adaptive lambda (use fixed lambda)")
    parser.add_argument("--class-aware-adaptive", dest="class_aware_adaptive", action="store_true", help="Enable V3 adaptive lambda (scales by skewness)")
    parser.add_argument("--no-class-aware-soft-weighting", dest="class_aware_soft_weighting", action="store_false", default=None, help="Disable soft probability weighting (use hard argmax)")
    parser.add_argument("--class-aware-soft-weighting", dest="class_aware_soft_weighting", action="store_true", help="Enable V3 soft probability weighting (replaces hard argmax)")
    parser.add_argument("--ssl-class-weighted", dest="ssl_class_weighted", action="store_true", default=None, help="Enable class-weighted SSL consistency loss (innovation 3)")
    parser.add_argument("--loss-type", type=str, default=None, choices=["ce", "cb", "focal"], help="Loss function: ce (CrossEntropy), cb (Class-Balanced Loss), focal (Focal Loss)")
    parser.add_argument("--ssl-deficit-threshold", dest="ssl_deficit_threshold", action="store_true", default=None, help="Enable deficit-based adaptive SSL threshold (innovation 2)")
    parser.add_argument("--ssl-joint-distribution", action="store_true", default=False, help="Use joint distribution (labeled + pseudo-labels) for SSL threshold and class-weighted loss")
    parser.add_argument("--ssl-deficit-alpha", type=float, default=None, help="Alpha for deficit threshold adjustment (default 0.25)")
    parser.add_argument("--ssl-deficit-start-round", type=int, default=None, help="Progressive SSL: use Base SSL for first N rounds, then switch to Innov SSL (0=full Innov)")
    parser.add_argument("--curriculum-warmup-rounds", type=int, default=None, help="Warmup rounds for curriculum penalty (default 5)")
    parser.add_argument("--joint-start-round", type=int, default=None, help="Use labeled-only distribution for first N rounds, then switch to joint (0=always joint)")
    parser.add_argument("--pseudo-refresh-freq", type=int, default=None, help="Refresh pseudo-labels every K AL rounds (0=no refresh)")
    parser.add_argument("--ssl-warmup-rounds", type=int, default=None, help="Pure supervised for first N rounds, then enable SSL (0=full coupling)")
    parser.add_argument("--ssl-adaptive", dest="ssl_adaptive", action="store_true", default=None, help="Enable SSL only when pseudo-label accuracy exceeds threshold")
    parser.add_argument("--no-ssl-adaptive", dest="ssl_adaptive", action="store_false", help="Disable adaptive SSL coupling")
    parser.add_argument("--ssl-adaptive-threshold", type=float, default=None, help="Pseudo-label accuracy threshold for adaptive SSL (default 0.80)")
    parser.add_argument("--pretrained-features", type=str, default=None, choices=["", "clip", "sbert"], help="Use frozen pretrained features: clip (CLIP ViT-B/32 for images), sbert (Sentence-BERT for text)")
    parser.add_argument("--cold-start", dest="cold_start", action="store_true", default=None, help="Ultra-small initial set experiment")
    parser.add_argument("--no-cold-start", dest="cold_start", action="store_false", help="Disable ultra-small initial set experiment")
    parser.add_argument("--ablation", type=str, default=None, choices=["", "pool_size", "epochs", "initial_size", "committee_size"], help="Ablation study dimension")
    parser.add_argument("--merge-train-test", action="store_true", default=False, help="Merge train+test then re-split 80/20 (for ECG5000 extended mode)")
    parser.add_argument("--config", type=str, default=None, help="YAML config file path")
    return parser.parse_args()


def apply_runtime_args(args):
    # Load YAML config first (CLI args override YAML)
    yaml_cfg = {}
    if args.config:
        try:
            import yaml
            with open(args.config, 'r', encoding='utf-8') as f:
                yaml_cfg = yaml.safe_load(f) or {}
            for key, val in yaml_cfg.items():
                if hasattr(CFG, key):
                    setattr(CFG, key, val)
            logger.info(f"Loaded config from {args.config}")
        except ImportError:
            logger.warning("PyYAML not installed, skipping config file. Install with: pip install pyyaml")
        except Exception as e:
            logger.warning(f"Failed to load config file: {e}")

    if args.dataset is not None:
        CFG.dataset = args.dataset
    if args.budget_level is not None:
        CFG.budget_level = args.budget_level
    if args.model_type is not None:
        CFG.model_type = args.model_type
    if args.use_ssl is not None:
        CFG.use_ssl = args.use_ssl
    if args.ssl_method is not None:
        CFG.ssl_method = args.ssl_method
    if args.ssl_threshold is not None:
        CFG.ssl_threshold = args.ssl_threshold
    if args.ssl_use_flexmatch is not None:
        CFG.ssl_use_flexmatch = args.ssl_use_flexmatch
    if args.ts_jitter_std is not None:
        CFG.ts_jitter_std = max(0.0, args.ts_jitter_std)
    if args.ts_scaling_std is not None:
        CFG.ts_scaling_std = max(0.0, args.ts_scaling_std)
    if args.ts_time_warp_sigma is not None:
        CFG.ts_time_warp_sigma = max(0.0, args.ts_time_warp_sigma)
    if args.no_ts_time_warp:
        CFG.ts_use_time_warp = False
    if args.dropout_consistency_passes is not None:
        CFG.dropout_consistency_passes = max(2, args.dropout_consistency_passes)
    if args.vat_epsilon is not None:
        CFG.vat_epsilon = max(0.0, args.vat_epsilon)
    if args.no_amp:
        CFG.use_amp = False
    if args.no_fast_4060_preset:
        CFG.fast_4060_preset = False
    if args.enable_tsne is not None:
        CFG.enable_tsne = args.enable_tsne
    if args.resume is not None:
        CFG.resume = args.resume
    if args.no_scheduler:
        CFG.use_scheduler = False
    if args.bald_mc_samples is not None:
        CFG.bald_mc_samples = args.bald_mc_samples
    if args.fast_bald_mc_samples is not None:
        CFG.bald_mc_samples = max(1, args.fast_bald_mc_samples)
    if args.qbc_reuse_committee is not None:
        CFG.qbc_reuse_committee = args.qbc_reuse_committee
    if args.qbc_incremental_epochs is not None:
        CFG.qbc_incremental_epochs = max(1, args.qbc_incremental_epochs)
    if args.imbalance_ratio is not None:
        CFG.imbalance_ratio = args.imbalance_ratio
    if args.class_aware_lambda is not None:
        CFG.class_aware_lambda = args.class_aware_lambda
    if args.class_aware_adaptive is not None:
        CFG.class_aware_adaptive = args.class_aware_adaptive
    if args.class_aware_soft_weighting is not None:
        CFG.class_aware_soft_weighting = args.class_aware_soft_weighting
    if args.ssl_class_weighted is not None:
        CFG.ssl_class_weighted = args.ssl_class_weighted
    if args.loss_type is not None:
        CFG.loss_type = args.loss_type
    if args.ssl_deficit_threshold is not None:
        CFG.ssl_deficit_threshold = args.ssl_deficit_threshold
    if args.ssl_deficit_alpha is not None:
        CFG.ssl_deficit_alpha = args.ssl_deficit_alpha
    if args.ssl_deficit_start_round is not None:
        CFG.ssl_deficit_start_round = args.ssl_deficit_start_round
    if args.ssl_joint_distribution:
        CFG.ssl_joint_distribution = True
    if args.curriculum_warmup_rounds is not None:
        CFG.curriculum_warmup_rounds = args.curriculum_warmup_rounds
    if args.joint_start_round is not None:
        CFG.joint_start_round = args.joint_start_round
    if args.pseudo_refresh_freq is not None:
        CFG.pseudo_refresh_freq = args.pseudo_refresh_freq
    if args.ssl_warmup_rounds is not None:
        CFG.ssl_warmup_rounds = args.ssl_warmup_rounds
    if args.ssl_adaptive is not None:
        CFG.ssl_adaptive = args.ssl_adaptive
    if args.ssl_adaptive_threshold is not None:
        CFG.ssl_adaptive_threshold = args.ssl_adaptive_threshold
    if args.pretrained_features is not None:
        CFG.pretrained_features = args.pretrained_features
    if args.cold_start is not None:
        CFG.cold_start = args.cold_start
    if args.ablation is not None:
        CFG.ablation = args.ablation
    if args.merge_train_test:
        CFG.merge_train_test = True
    if 0 < CFG.imbalance_ratio < 1:
        raise ValueError("--imbalance-ratio must be 0 for balanced data or >= 1 for max/min class count ratio")

    dcfg = BUDGET_LEVELS[CFG.budget_level][CFG.dataset]
    CFG.n_initial = dcfg["n_initial"]
    CFG.n_query = dcfg["n_query"]
    CFG.n_rounds = dcfg["n_rounds"]
    CFG.n_epochs_base = dcfg["n_epochs_base"]
    CFG.learning_rate = dcfg["learning_rate"]
    CFG.batch_size_train = dcfg["batch_size_train"]
    CFG.batch_size_infer = dcfg["batch_size_infer"]
    CFG.max_pool_subsample = dcfg["max_pool_subsample"]
    for key in (
        "n_initial", "n_query", "n_rounds", "n_epochs_base",
        "learning_rate", "batch_size_train", "batch_size_infer",
        "max_pool_subsample",
    ):
        if key in yaml_cfg:
            setattr(CFG, key, yaml_cfg[key])
    if args.max_pool_subsample is not None:
        CFG.max_pool_subsample = max(1, args.max_pool_subsample)
    if "ssl_lambda_u_warmup_epochs" not in yaml_cfg:
        CFG.ssl_lambda_u_warmup_epochs = 2 * CFG.n_epochs_base

    if CFG.fast_4060_preset:
        apply_fast_4060_preset(
            explicit_bald=args.fast_bald_mc_samples is not None,
            explicit_pool=args.max_pool_subsample is not None or "max_pool_subsample" in yaml_cfg,
        )
    if CFG.dataset == "forda" and CFG.model_type == "simplecnn":
        CFG.model_type = "timeseriescnn"
    if CFG.dataset == "ecg5000" and CFG.model_type == "simplecnn":
        CFG.model_type = "timeseriescnn"
    if CFG.dataset == "spoken_arabic" and CFG.model_type == "simplecnn":
        CFG.model_type = "timeseriescnn"
    if CFG.dataset == "character_traj" and CFG.model_type == "simplecnn":
        CFG.model_type = "timeseriescnn"

    # Cold-start override: ultra-small initial set, n_query adapts to n_classes
    if CFG.cold_start:
        n_classes_map = {"fashion_mnist": 10, "cifar10": 10, "cifar100": 100, "agnews": 4, "bloodmnist": 8, "adult": 2, "forda": 2, "ecg5000": 5, "spoken_arabic": 10, "character_traj": 20}
        n_cls = n_classes_map.get(CFG.dataset, 10)
        CFG.n_initial = max(n_cls * 2, 20)
        CFG.n_query = max(50, n_cls * 3)

    # Balanced output directory: key dimensions in path, rest in config.json
    from datetime import datetime
    run_ts = datetime.now().strftime("%m%d_%H%M")
    ssl_flag = f"_ssl{int(CFG.use_ssl)}"
    imb_flag = f"_imb{int(CFG.imbalance_ratio)}" if CFG.imbalance_ratio > 0 else ""
    cold_flag = "_cold" if CFG.cold_start else ""
    CFG.output_dir = yaml_cfg.get(
        "output_dir",
        str(PROJECT_ROOT / "output" / f"{CFG.dataset}_v8_{CFG.budget_level}_{CFG.model_type}{ssl_flag}{imb_flag}{cold_flag}")
    )
    CFG.checkpoint_dir = os.path.join(CFG.output_dir, "checkpoints")

    if args.quick:
        CFG.output_dir = str(PROJECT_ROOT / "output" / f"{CFG.dataset}_v8_quick")
        CFG.checkpoint_dir = os.path.join(CFG.output_dir, "checkpoints")
        CFG.n_initial = min(200, CFG.n_initial)
        CFG.n_query = min(100, CFG.n_query)
        CFG.n_rounds = 5
        CFG.seeds = [42]
        CFG.strategies = ["random", "entropy", "badge"]
        CFG.max_pool_subsample = 2000
        CFG.n_epochs_base = 3
        CFG.committee_epochs = 2
        CFG.full_supervised_epochs = 10
        CFG.enable_tsne = False
        if args.max_pool_subsample is not None:
            CFG.max_pool_subsample = max(1, args.max_pool_subsample)

    if args.output_dir:
        CFG.output_dir = args.output_dir
        CFG.checkpoint_dir = os.path.join(CFG.output_dir, "checkpoints")
    if args.rounds is not None:
        CFG.n_rounds = args.rounds
    if args.seeds:
        CFG.seeds = args.seeds
    if args.strategies:
        CFG.strategies = args.strategies


def load_fashion_mnist():
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(28, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.2860,), (0.3530,)),
    ])
    transform_infer = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.2860,), (0.3530,)),
    ])
    train_set = torchvision.datasets.FashionMNIST(root=str(PROJECT_ROOT / "data"), train=True, download=True, transform=transform_train)
    test_set = torchvision.datasets.FashionMNIST(root=str(PROJECT_ROOT / "data"), train=False, download=True, transform=transform_infer)
    # Inference-only copy of train set (no augmentation) for consistent queries
    infer_set = torchvision.datasets.FashionMNIST(root=str(PROJECT_ROOT / "data"), train=True, download=False, transform=transform_infer)
    # Raw dataset (no transform at all) for FixMatch UnlabeledDataset
    raw_set = torchvision.datasets.FashionMNIST(root=str(PROJECT_ROOT / "data"), train=True, download=False, transform=None)
    return train_set, test_set, infer_set, raw_set


def load_cifar10():
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])
    transform_infer = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])
    train_set = torchvision.datasets.CIFAR10(root=str(PROJECT_ROOT / "data"), train=True, download=True, transform=transform_train)
    test_set = torchvision.datasets.CIFAR10(root=str(PROJECT_ROOT / "data"), train=False, download=True, transform=transform_infer)
    infer_set = torchvision.datasets.CIFAR10(root=str(PROJECT_ROOT / "data"), train=True, download=False, transform=transform_infer)
    raw_set = torchvision.datasets.CIFAR10(root=str(PROJECT_ROOT / "data"), train=True, download=False, transform=None)
    return train_set, test_set, infer_set, raw_set


def load_cifar100():
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])
    transform_infer = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])
    train_set = torchvision.datasets.CIFAR100(root=str(PROJECT_ROOT / "data"), train=True, download=True, transform=transform_train)
    test_set = torchvision.datasets.CIFAR100(root=str(PROJECT_ROOT / "data"), train=False, download=True, transform=transform_infer)
    infer_set = torchvision.datasets.CIFAR100(root=str(PROJECT_ROOT / "data"), train=True, download=False, transform=transform_infer)
    raw_set = torchvision.datasets.CIFAR100(root=str(PROJECT_ROOT / "data"), train=True, download=False, transform=None)
    return train_set, test_set, infer_set, raw_set


class AGNewsDataset(Dataset):
    def __init__(self, texts, labels, tfidf_vectorizer=None, tfidf_dim=5000, fit=True):
        self.texts = texts
        self.labels = labels
        if tfidf_vectorizer is None and fit:
            self.vectorizer = TfidfVectorizer(max_features=tfidf_dim, stop_words='english')
            self.tfidf = self.vectorizer.fit_transform(texts).toarray().astype(np.float32)
        elif tfidf_vectorizer is not None:
            self.vectorizer = tfidf_vectorizer
            self.tfidf = tfidf_vectorizer.transform(texts).toarray().astype(np.float32)
        else:
            self.vectorizer = None
            self.tfidf = np.zeros((len(texts), tfidf_dim), dtype=np.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.tfidf[idx], self.labels[idx]


def load_agnews():
    import datasets as hf_datasets
    cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "datasets")
    ag_cache = os.path.join(cache_dir, "ag_news")
    use_local = os.path.isdir(ag_cache) and any(
        f.endswith(".arrow") for _, _, files in os.walk(ag_cache) for f in files
    )
    if use_local:
        print("Loading AG News from local cache (offline mode)...")
        os.environ["HF_DATASETS_OFFLINE"] = "1"
        ds = hf_datasets.load_dataset("ag_news")
    else:
        print("Loading AG News from HuggingFace Hub...")
        ds = hf_datasets.load_dataset("ag_news", download_mode="reuse_cache_if_exists")
    train_texts = [item['text'] for item in ds['train']]
    train_labels = [item['label'] for item in ds['train']]
    test_texts = [item['text'] for item in ds['test']]
    test_labels = [item['label'] for item in ds['test']]

    np.random.seed(42)
    if len(train_texts) > 30000:
        idx = np.random.choice(len(train_texts), 30000, replace=False)
        train_texts = [train_texts[i] for i in idx]
        train_labels = [train_labels[i] for i in idx]
    if len(test_texts) > 2000:
        idx = np.random.choice(len(test_texts), 2000, replace=False)
        test_texts = [test_texts[i] for i in idx]
        test_labels = [test_labels[i] for i in idx]

    train_set = AGNewsDataset(train_texts, train_labels, tfidf_dim=CFG.tfidf_dim, fit=True)
    test_set = AGNewsDataset(test_texts, test_labels, tfidf_vectorizer=train_set.vectorizer, fit=False)
    # Text (TF-IDF) has no stochastic augmentation, so infer_set == train_set
    return train_set, test_set, train_set, None  # raw_set=None for text


class BloodMNISTWrapper(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        from PIL import Image
        img = self.images[idx]
        if isinstance(img, np.ndarray):
            img = Image.fromarray(img)
        label = self.labels[idx]
        if self.transform:
            img = self.transform(img)
        return img, label


def load_bloodmnist():
    import medmnist
    from medmnist import BloodMNIST

    print("Loading BloodMNIST (medical domain - blood cell microscopy)...")
    data_root = str(PROJECT_ROOT / "data" / "medmnist")

    train_dataset = BloodMNIST(root=data_root, split="train", download=True, transform=None)
    test_dataset = BloodMNIST(root=data_root, split="test", download=True, transform=None)

    train_images = train_dataset.imgs
    train_labels_arr = train_dataset.labels.squeeze()
    test_images = test_dataset.imgs
    test_labels_arr = test_dataset.labels.squeeze()

    if isinstance(train_labels_arr, np.ndarray):
        train_labels = train_labels_arr.tolist()
    else:
        train_labels = train_labels_arr.numpy().tolist()
    if isinstance(test_labels_arr, np.ndarray):
        test_labels = test_labels_arr.tolist()
    else:
        test_labels = test_labels_arr.numpy().tolist()

    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize((0.8162, 0.6685, 0.6955), (0.2152, 0.2415, 0.1175)),
    ])
    transform_infer = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.8162, 0.6685, 0.6955), (0.2152, 0.2415, 0.1175)),
    ])

    train_set = BloodMNISTWrapper(train_images, train_labels, transform=transform_train)
    test_set = BloodMNISTWrapper(test_images, test_labels, transform=transform_infer)
    infer_set = BloodMNISTWrapper(train_images, train_labels, transform=transform_infer)

    raw_set = BloodMNISTWrapper(train_images, train_labels, transform=None)

    logger.info(f"  BloodMNIST: train={len(train_set)}, test={len(test_set)}, classes=8")
    return train_set, test_set, infer_set, raw_set


ADULT_FEATURE_NAMES = [
    "age", "workclass", "fnlwgt", "education", "education-num",
    "marital-status", "occupation", "relationship", "race", "sex",
    "capital-gain", "capital-loss", "hours-per-week", "native-country"
]


class AdultDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


def load_adult():
    print("Loading Adult/Census Income (tabular domain - income prediction)...")
    data_root = str(PROJECT_ROOT / "data" / "adult")
    os.makedirs(data_root, exist_ok=True)

    cache_path = os.path.join(data_root, "adult_cache.npz")
    if os.path.exists(cache_path):
        print("  Loading cached Adult dataset...")
        cache = np.load(cache_path)
        train_features = cache["train_features"]
        train_labels = cache["train_labels"]
        test_features = cache["test_features"]
        test_labels = cache["test_labels"]
    else:
        try:
            from sklearn.datasets import fetch_openml
            print("  Downloading Adult dataset from OpenML...")
            adult = fetch_openml(name="adult", version=2, as_frame=True, parser="auto")
            df = adult.frame
            target_col = "class"
            y = (df[target_col] == ">50K").astype(int).values
            X = df.drop(columns=[target_col])
            cat_cols = X.select_dtypes(include=["category", "object"]).columns.tolist()
            num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
            from sklearn.preprocessing import StandardScaler, OneHotEncoder
            from sklearn.compose import ColumnTransformer
            preprocessor = ColumnTransformer(
                transformers=[
                    ("num", StandardScaler(), num_cols),
                    ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols)
                ]
            )
            X_processed = preprocessor.fit_transform(X)
            X_processed = X_processed.astype(np.float32)
            np.random.seed(42)
            perm = np.random.permutation(len(y))
            X_processed = X_processed[perm]
            y = y[perm]
            n_test = min(10000, len(y) // 5)
            test_features = X_processed[:n_test]
            test_labels = y[:n_test]
            train_features = X_processed[n_test:]
            train_labels = y[n_test:]
            np.savez(cache_path,
                     train_features=train_features, train_labels=train_labels,
                     test_features=test_features, test_labels=test_labels)
        except Exception as e:
            print(f"  Cannot download Adult dataset ({e}). Generating synthetic tabular data...")
            n_train, n_test = 40000, 10000
            n_features = 100
            np.random.seed(42)
            train_features = np.random.randn(n_train, n_features).astype(np.float32)
            train_labels = (train_features[:, 0] + train_features[:, 1] > 0).astype(np.int64)
            test_features = np.random.randn(n_test, n_features).astype(np.float32)
            test_labels = (test_features[:, 0] + test_features[:, 1] > 0).astype(np.int64)
            np.savez(cache_path,
                     train_features=train_features, train_labels=train_labels,
                     test_features=test_features, test_labels=test_labels)

    train_set = AdultDataset(train_features, train_labels)
    test_set = AdultDataset(test_features, test_labels)
    infer_set = AdultDataset(train_features, train_labels)

    logger.info(f"  Adult: train={len(train_set)}, test={len(test_set)}, classes=2, features={train_features.shape[1]}")
    return train_set, test_set, infer_set, None  # raw_set=None for tabular


class PrecomputedFeatureDataset(Dataset):
    """Dataset wrapping precomputed frozen features (e.g. from CLIP / SBERT)."""

    def __init__(self, features: np.ndarray, labels: np.ndarray):
        self.features = features.astype(np.float32)
        self.labels = labels.astype(np.int64)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


def extract_pretrained_features(dataset, indices, model_name: str, dataset_name: str, device):
    """Extract frozen features from a pretrained model.

    Args:
        dataset: source dataset (images or text).
        indices: subset indices to extract.
        model_name: "clip" or "sbert".
        dataset_name: "cifar10", "agnews", etc.
        device: torch device.

    Returns:
        features: np.ndarray of shape (len(indices), feature_dim).
        labels: np.ndarray of shape (len(indices),).
    """
    from torch.utils.data import DataLoader, Subset

    subset = Subset(dataset, indices)
    loader = DataLoader(subset, batch_size=256, shuffle=False, num_workers=0)

    if model_name == "clip":
        import clip as clip_pkg
        clip_model, _ = clip_pkg.load("ViT-B/32", device=device)
        clip_model.eval()

        all_features = []
        all_labels = []
        with torch.no_grad():
            for batch in loader:
                images, labels = batch
                images = images.to(device)
                # CLIP expects 224x224 RGB input; our images may be smaller.
                # Resize if needed.
                if images.shape[-1] != 224 or images.shape[-2] != 224:
                    images = torch.nn.functional.interpolate(
                        images, size=224, mode="bilinear", align_corners=False)
                # Ensure 3 channels (CLIP expects RGB)
                if images.shape[1] == 1:
                    images = images.repeat(1, 3, 1, 1)
                features = clip_model.encode_image(images).float()
                features = features / features.norm(dim=-1, keepdim=True)  # L2 normalize
                all_features.append(features.cpu().numpy())
                all_labels.append(labels.numpy())

        features = np.concatenate(all_features, axis=0)
        labels = np.concatenate(all_labels, axis=0)
        del clip_model

    elif model_name == "sbert":
        from sentence_transformers import SentenceTransformer
        sbert_model = SentenceTransformer("all-MiniLM-L6-v2", device=str(device))

        # For text datasets, we need the raw text. The dataset stores TF-IDF
        # vectors, not raw text. We need to re-create the text from the
        # dataset's internal representation.
        # For AG News: the AGNewsDataset stores raw texts internally.
        if hasattr(dataset, "texts"):
            raw_texts = [dataset.texts[i] for i in indices]
        else:
            # Fallback: convert TF-IDF vectors back (lossy, not recommended)
            raise ValueError(
                f"Dataset '{dataset_name}' does not store raw text. "
                "SBERT requires raw text input. Use CLIP for image datasets."
            )

        features = sbert_model.encode(
            raw_texts, batch_size=256, show_progress_bar=True,
            convert_to_numpy=True, normalize_embeddings=True
        )
        labels = np.array([dataset[i][1] for i in indices])
        del sbert_model

    else:
        raise ValueError(f"Unknown pretrained model: {model_name}")

    logger.info(f"  Extracted {model_name} features: {features.shape}, labels: {labels.shape}")
    return features, labels


UCR_DATASETS = {
    "forda": {
        "archive_name": "FordA",
        "train_url": "https://zenodo.org/records/11191164/files/FordA_TRAIN.ts?download=1",
        "test_url": "https://zenodo.org/records/11191164/files/FordA_TEST.ts?download=1",
    },
    "ecg5000": {
        "archive_name": "ECG5000",
        "train_url": "",
        "test_url": "",
    },
    "spoken_arabic": {
        "archive_name": "SpokenArabicDigits",
        "train_url": "",
        "test_url": "",
    },
    "character_traj": {
        "archive_name": "CharacterTrajectories",
        "train_url": "",
        "test_url": "",
    }
}


class UCRTimeSeriesDataset(Dataset):
    def __init__(self, series, labels):
        self.series = torch.as_tensor(series, dtype=torch.float32)
        self.labels = torch.as_tensor(labels, dtype=torch.long)

    def __len__(self):
        return int(self.labels.shape[0])

    def __getitem__(self, idx):
        return self.series[idx], self.labels[idx]


def jitter_time_series(x, std):
    if std <= 0:
        return x
    return x + torch.randn_like(x) * std


def scale_time_series(x, std):
    if std <= 0:
        return x
    scale = torch.randn(x.size(0), 1, dtype=x.dtype, device=x.device) * std + 1.0
    return x * scale


def warp_time_series(x, sigma, n_knots=6):
    if sigma <= 0 or x.size(-1) < 4:
        return x
    length = x.size(-1)
    knots = max(2, min(n_knots, length // 2))
    offsets = torch.randn(1, 1, knots, dtype=x.dtype, device=x.device)
    offsets = F.interpolate(offsets, size=length, mode="linear", align_corners=True)[0, 0]
    offsets = offsets * (sigma * length)
    source_pos = torch.arange(length, dtype=x.dtype, device=x.device) + offsets
    source_pos = source_pos.clamp(0, length - 1)

    left = source_pos.floor().long()
    right = (left + 1).clamp(max=length - 1)
    weight = (source_pos - left.to(source_pos.dtype)).unsqueeze(0)
    return x[:, left] * (1.0 - weight) + x[:, right] * weight


def strong_time_series_augment(x):
    augmented = x.clone()
    augmented = jitter_time_series(augmented, CFG.ts_jitter_std)
    augmented = scale_time_series(augmented, CFG.ts_scaling_std)
    if CFG.ts_use_time_warp:
        augmented = warp_time_series(augmented, CFG.ts_time_warp_sigma)
    return augmented


def tfidf_dropout(x, drop_prob=0.15):
    if drop_prob <= 0:
        return x
    keep = torch.rand_like(x).ge(drop_prob).to(x.dtype)
    return x * keep / max(1e-6, 1.0 - drop_prob)


def vat_perturb(model, x, target_probs, epsilon):
    if epsilon <= 0:
        return x
    with torch.enable_grad():
        noise = torch.randn_like(x)
        flat = noise.view(noise.size(0), -1)
        norm = flat.norm(p=2, dim=1, keepdim=True).clamp_min(1e-6)
        noise = (flat / norm).view_as(noise).detach()
        noise.requires_grad_(True)

        logits_adv = model(x.detach() + 1e-3 * noise)
        log_probs_adv = F.log_softmax(logits_adv, dim=-1)
        vat_loss = F.kl_div(log_probs_adv, target_probs.detach(), reduction="batchmean")
        grad = torch.autograd.grad(vat_loss, noise, retain_graph=False, create_graph=False)[0]
        grad_flat = grad.view(grad.size(0), -1)
        grad_norm = grad_flat.norm(p=2, dim=1, keepdim=True).clamp_min(1e-6)
        direction = (grad_flat / grad_norm).view_as(grad)
    return (x + epsilon * direction).detach()


class TimeSeriesConsistencyDataset(Dataset):
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = list(indices)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        series, label = self.dataset[real_idx]
        weak = series.clone()
        strong = strong_time_series_augment(series)
        return weak, strong, label


def _download_ucr_file(url, target_path):
    os.makedirs(os.path.dirname(target_path), exist_ok=True)
    try:
        urllib.request.urlretrieve(url, target_path)
    except Exception as exc:
        raise RuntimeError(
            f"UCR file is missing and automatic download failed: {target_path}. "
            f"Download it manually from {url} and rerun the experiment."
        ) from exc


def _parse_ucr_ts_file(path):
    series_rows = []
    raw_labels = []
    in_data = False
    with open(path, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if line.lower().startswith("@data"):
                in_data = True
                continue
            if line.startswith("@"):
                continue
            if not in_data:
                continue

            parts = line.split(":")
            if len(parts) < 2:
                raise ValueError(f"Invalid UCR .ts row in {path}: {line[:80]}")
            dims = []
            for dim_text in parts[:-1]:
                values = [
                    np.nan if val.strip() in {"?", "NaN", "nan"} else float(val)
                    for val in dim_text.split(",")
                    if val.strip() != ""
                ]
                dims.append(values)
            if not dims:
                raise ValueError(f"UCR .ts row has no series values in {path}: {line[:80]}")
            lengths = {len(dim) for dim in dims}
            if len(lengths) != 1:
                raise ValueError(f"Unequal dimension lengths in {path}: {line[:80]}")
            series_rows.append(dims)
            raw_labels.append(parts[-1].strip())

    if not series_rows:
        raise ValueError(f"No data rows were parsed from {path}")
    try:
        data = np.asarray(series_rows, dtype=np.float32)
    except ValueError:
        max_len = max(max(len(d) for d in row) for row in series_rows)
        n_dims = len(series_rows[0])
        padded = np.full((len(series_rows), n_dims, max_len), np.nan, dtype=np.float32)
        for i, row in enumerate(series_rows):
            for d, dim_vals in enumerate(row):
                padded[i, d, :len(dim_vals)] = dim_vals
        data = padded
    return data, raw_labels


def _load_cached_or_raw_ucr(cache_path, train_path, test_path):
    if os.path.exists(cache_path):
        cache = np.load(cache_path)
        return (
            cache["train_x"],
            cache["train_y"],
            cache["test_x"],
            cache["test_y"],
            cache["classes"].tolist(),
        )

    train_x, train_labels_raw = _parse_ucr_ts_file(train_path)
    test_x, test_labels_raw = _parse_ucr_ts_file(test_path)
    if train_x.ndim == 3 and test_x.ndim == 3 and train_x.shape[2] != test_x.shape[2]:
        max_len = max(train_x.shape[2], test_x.shape[2])
        if train_x.shape[2] < max_len:
            pad = np.full((train_x.shape[0], train_x.shape[1], max_len - train_x.shape[2]), np.nan, dtype=np.float32)
            train_x = np.concatenate([train_x, pad], axis=2)
        if test_x.shape[2] < max_len:
            pad = np.full((test_x.shape[0], test_x.shape[1], max_len - test_x.shape[2]), np.nan, dtype=np.float32)
            test_x = np.concatenate([test_x, pad], axis=2)
    classes = sorted(set(train_labels_raw + test_labels_raw))
    label_map = {label: idx for idx, label in enumerate(classes)}
    train_y = np.asarray([label_map[label] for label in train_labels_raw], dtype=np.int64)
    test_y = np.asarray([label_map[label] for label in test_labels_raw], dtype=np.int64)

    mean = np.nanmean(train_x, keepdims=True)
    std = np.nanstd(train_x, keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    train_x = np.nan_to_num((train_x - mean) / std).astype(np.float32)
    test_x = np.nan_to_num((test_x - mean) / std).astype(np.float32)

    np.savez(
        cache_path,
        train_x=train_x,
        train_y=train_y,
        test_x=test_x,
        test_y=test_y,
        classes=np.asarray(classes),
    )
    return train_x, train_y, test_x, test_y, classes


def load_ucr_dataset(name):
    meta = UCR_DATASETS[name]
    archive_name = meta["archive_name"]
    data_root = PROJECT_ROOT / "data" / "ucr" / archive_name
    os.makedirs(data_root, exist_ok=True)

    train_path = data_root / f"{archive_name}_TRAIN.ts"
    test_path = data_root / f"{archive_name}_TEST.ts"
    if not train_path.exists():
        if meta.get("train_url"):
            print(f"  Downloading {archive_name} train split from UCR archive mirror...")
            _download_ucr_file(meta["train_url"], str(train_path))
        else:
            raise FileNotFoundError(f"UCR train file not found and no download URL configured: {train_path}")
    if not test_path.exists():
        if meta.get("test_url"):
            print(f"  Downloading {archive_name} test split from UCR archive mirror...")
            _download_ucr_file(meta["test_url"], str(test_path))
        else:
            raise FileNotFoundError(f"UCR test file not found and no download URL configured: {test_path}")

    cache_path = data_root / f"{archive_name}_cache.npz"
    train_x, train_y, test_x, test_y, classes = _load_cached_or_raw_ucr(
        str(cache_path), str(train_path), str(test_path)
    )

    train_set = UCRTimeSeriesDataset(train_x, train_y)
    test_set = UCRTimeSeriesDataset(test_x, test_y)
    infer_set = UCRTimeSeriesDataset(train_x, train_y)

    if CFG.merge_train_test:
        from sklearn.model_selection import train_test_split
        all_x = np.concatenate([train_x, test_x], axis=0)
        all_y = np.concatenate([train_y, test_y], axis=0)
        train_x_new, test_x_new, train_y_new, test_y_new = train_test_split(
            all_x, all_y, test_size=0.2, random_state=42, stratify=all_y
        )
        train_set = UCRTimeSeriesDataset(train_x_new, train_y_new)
        test_set = UCRTimeSeriesDataset(test_x_new, test_y_new)
        infer_set = UCRTimeSeriesDataset(train_x_new, train_y_new)
        logger.info(
            f"  UCR/{archive_name} (merged+resplit): train={len(train_set)}, test={len(test_set)}, "
            f"classes={len(classes)}, length={train_x.shape[-1]}"
        )
    else:
        logger.info(
            f"  UCR/{archive_name}: train={len(train_set)}, test={len(test_set)}, "
            f"classes={len(classes)}, length={train_x.shape[-1]}"
        )
    return train_set, test_set, infer_set, None


def create_model(dataset, device, model_type=None):
    if model_type is None:
        model_type = CFG.model_type

    # Direction 3: pretrained features use a simple MLP classifier
    if CFG.pretrained_features:
        n_classes = get_num_classes(dataset)
        return TextMLPClassifier(
            input_dim=CFG.pretrained_input_dim,
            num_classes=n_classes,
            hidden_dim=256,
            dropout=0.2,
        ).to(device)

    if dataset == "fashion_mnist":
        if model_type == "resnet18":
            model = ResNet18(num_classes=10, in_channels=1).to(device)
        else:
            model = SimpleCNN(num_classes=10, in_channels=1).to(device)
    elif dataset == "cifar10":
        if model_type == "resnet18":
            model = ResNet18(num_classes=10, in_channels=3).to(device)
        else:
            model = SimpleCNN(num_classes=10, in_channels=3).to(device)
    elif dataset == "cifar100":
        if model_type == "resnet18":
            model = ResNet18(num_classes=100, in_channels=3).to(device)
        else:
            model = SimpleCNN(num_classes=100, in_channels=3).to(device)
    elif dataset == "agnews":
        model = TextMLPClassifier(input_dim=CFG.tfidf_dim, num_classes=4, hidden_dim=CFG.tfidf_hidden, dropout=0.2).to(device)
    elif dataset == "bloodmnist":
        if model_type == "resnet18":
            model = ResNet18(num_classes=8, in_channels=3).to(device)
        else:
            model = SimpleCNN(num_classes=8, in_channels=3).to(device)
    elif dataset == "adult":
        model = TextMLPClassifier(input_dim=CFG.adult_input_dim, num_classes=2, hidden_dim=CFG.adult_hidden, dropout=0.2).to(device)
    elif dataset == "forda":
        model = TimeSeriesCNN(num_classes=2, in_channels=1, hidden_dim=64, dropout=0.2).to(device)
    elif dataset == "ecg5000":
        model = TimeSeriesCNN(num_classes=5, in_channels=1, hidden_dim=64, dropout=0.2).to(device)
    elif dataset == "spoken_arabic":
        model = TimeSeriesCNN(num_classes=10, in_channels=13, hidden_dim=64, dropout=0.2).to(device)
    elif dataset == "character_traj":
        model = TimeSeriesCNN(num_classes=20, in_channels=3, hidden_dim=64, dropout=0.2).to(device)
    return model


def train_model(
    model,
    dataset,
    train_idx,
    device,
    lr=None,
    n_epochs=None,
    early_stopping=False,
    val_dataset=None,
    val_idx=None,
    loss_pred_module=None,
    pool_idx=None,
    flex_tracker=None,
    raw_dataset=None,
    warmup_offset_epochs=0,
):
    model.train()
    if lr is None:
        lr = CFG.learning_rate
    if n_epochs is None:
        n_epochs = CFG.n_epochs_base

    ssl_method = resolve_ssl_method(CFG.dataset, CFG.ssl_method)
    subset = Subset(dataset, train_idx)
    loader = DataLoader(subset, batch_size=CFG.batch_size_train, shuffle=True, num_workers=0, drop_last=True if len(subset) > CFG.batch_size_train else False)

    # SSL unlabeled loader:
    # - images use FixMatch weak/strong augmentations on raw images
    # - FordA/UCR uses time-series consistency: weak identity, strong jitter/scaling/time-warp
    # - text/tabular use identity pseudo-label SSL unless a modality-specific branch is added
    unlabeled_loader = None
    unlabeled_iter = None
    unlabeled_view_mode = None
    if CFG.use_ssl and pool_idx is not None and len(pool_idx) > 0 and not CFG.pretrained_features and CFG.dataset in IMAGE_SSL_DATASETS and ssl_method in {"fixmatch", "flexmatch"}:
        # Get normalization constants
        if CFG.dataset == "fashion_mnist":
            norm = ((0.2860,), (0.3530,))
        elif CFG.dataset == "cifar10":
            norm = ((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
        elif CFG.dataset == "cifar100":
            norm = ((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
        elif CFG.dataset == "bloodmnist":
            norm = ((0.8162, 0.6685, 0.6955), (0.2152, 0.2415, 0.1175))

        transform_weak = transforms.Compose([
            transforms.RandomCrop(32 if CFG.dataset in ("cifar10", "cifar100") else 28, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(*norm),
        ])
        transform_strong = get_strong_transforms(CFG.dataset, norm)

        # Use raw_dataset (no transform) as data source 鈥?explicit, no `.dataset` chain
        ssl_data_source = raw_dataset if raw_dataset is not None else dataset
        u_dataset = UnlabeledDataset(ssl_data_source, pool_idx, transform_weak, transform_strong)
        unlabeled_loader = DataLoader(u_dataset, batch_size=CFG.batch_size_train, shuffle=True, num_workers=0, drop_last=True if len(u_dataset) > CFG.batch_size_train else False)
        unlabeled_view_mode = "image_views"
        if ssl_method == "flexmatch" and flex_tracker is None:
            logger.warning("SSL requested with FlexMatch but flex_tracker is None; falling back to fixed threshold.")
    elif CFG.use_ssl and pool_idx is not None and len(pool_idx) > 0 and not CFG.pretrained_features and CFG.dataset in ("forda", "ecg5000", "spoken_arabic", "character_traj") and ssl_method == "ts_consistency":
        u_dataset = TimeSeriesConsistencyDataset(dataset, pool_idx)
        unlabeled_loader = DataLoader(
            u_dataset,
            batch_size=CFG.batch_size_train,
            shuffle=True,
            num_workers=0,
            drop_last=True if len(u_dataset) > CFG.batch_size_train else False,
        )
        unlabeled_view_mode = "ts_views"
        if ssl_method == "flexmatch" and flex_tracker is None:
            logger.warning("SSL requested with FlexMatch but flex_tracker is None; falling back to fixed threshold.")
    elif CFG.use_ssl and pool_idx is not None and len(pool_idx) > 0 and (
        CFG.pretrained_features or CFG.dataset in PSEUDO_LABEL_SSL_DATASETS or ssl_method == "pseudo_label"
    ):
        u_dataset = Subset(dataset, pool_idx)
        unlabeled_loader = DataLoader(
            u_dataset,
            batch_size=CFG.batch_size_train,
            shuffle=True,
            num_workers=0,
            drop_last=True if len(u_dataset) > CFG.batch_size_train else False,
        )
        unlabeled_view_mode = "single_view"
        if ssl_method == "flexmatch" and flex_tracker is None:
            logger.warning("SSL requested with FlexMatch but flex_tracker is None; falling back to fixed threshold.")
        if CFG.pretrained_features:
            logger.info(f"  SSL with pretrained features: using pseudo-label mode (single_view)")
    elif CFG.use_ssl and CFG.dataset not in SSL_SUPPORTED_DATASETS:
        logger.warning(f"SSL requested for unsupported dataset '{CFG.dataset}'; supervised training only.")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs) if CFG.use_scheduler else None

    # --- Loss function selection ---
    if CFG.loss_type == "cb":
        # Class-Balanced Loss: weight_i = (1-β)/(1-β^n_i), β=0.9999
        beta_cb = 0.9999
        # Get n_classes from model
        n_cls = get_num_classes(CFG.dataset)
        # Use labeled subset (train_idx) for class distribution
        if hasattr(dataset, 'targets'):
            labeled_labels_np = np.array([dataset.targets[i] for i in train_idx])
        else:
            labeled_labels_np = np.array([dataset[i][1] for i in train_idx])
        class_counts = np.bincount(labeled_labels_np.astype(int), minlength=n_cls)
        effective_num = 1.0 - np.power(beta_cb, class_counts)
        cb_weights = (1.0 - beta_cb) / np.maximum(effective_num, 1e-8)
        cb_weights = cb_weights / cb_weights.sum() * n_cls  # normalize
        cb_weights_tensor = torch.FloatTensor(cb_weights).to(device)
        criterion = nn.CrossEntropyLoss(weight=cb_weights_tensor)
        logger.info(f"  Using Class-Balanced Loss (β={beta_cb}), weights: {cb_weights.round(3)}")
    elif CFG.loss_type == "focal":
        # Focal Loss with gamma=2.0
        gamma_focal = 2.0
        class FocalLoss(nn.Module):
            def __init__(self, gamma=2.0, reduction='mean'):
                super().__init__()
                self.gamma = gamma
                self.reduction = reduction
            def forward(self, inputs, targets):
                ce_loss = F.cross_entropy(inputs, targets, reduction='none')
                pt = torch.exp(-ce_loss)
                focal_loss = ((1 - pt) ** self.gamma) * ce_loss
                return focal_loss.mean() if self.reduction == 'mean' else focal_loss
        criterion = FocalLoss(gamma=gamma_focal)
        logger.info(f"  Using Focal Loss (γ={gamma_focal})")
    else:
        criterion = nn.CrossEntropyLoss()
    amp_enabled = bool(CFG.use_amp and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    if loss_pred_module is not None:
        loss_pred_module.train()
        lp_optimizer = torch.optim.Adam(loss_pred_module.parameters(), lr=lr * 0.1, weight_decay=1e-4)
        loss_pred_criterion = nn.MSELoss()

    best_val_f1 = -1.0
    patience_counter = 0
    best_state = None

    n_pseudo_labeled = 0
    pseudo_correct = 0
    pseudo_total_checked = 0
    # Collect per-class stats from the last epoch
    last_epoch_pseudo_stats = None

    t0 = time.time()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    for epoch in range(n_epochs):
        model.train()
        if loss_pred_module is not None:
            loss_pred_module.train()

        # --- Pseudo-label dynamic refresh (Direction 2) ---
        # NOTE: pseudo_refresh_freq now operates at AL-round level (see run_single_strategy).
        # Epoch-level refresh has been removed because with n_epochs_base=5,
        # k>=5 never fires. Round-level refresh is more meaningful.

        if unlabeled_loader is not None:
            unlabeled_iter = iter(unlabeled_loader)

        # Warm up SSL once across the whole AL run, not from zero every round.
        ssl_progress_epoch = warmup_offset_epochs + epoch + 1
        warmup_factor = min(1.0, ssl_progress_epoch / max(1, CFG.ssl_lambda_u_warmup_epochs))

        epoch_pseudo_labels, epoch_mask, epoch_true_labels = [], [], []

        for batch_data in loader:
            optimizer.zero_grad(set_to_none=True)
            features = None

            # --- Labeled Data Forward ---
            with torch.amp.autocast("cuda", enabled=amp_enabled):
                if CFG.pretrained_features or CFG.dataset in ["agnews", "adult"]:
                    feats, labels = batch_data
                    feats, labels = feats.to(device), labels.to(device)
                    if loss_pred_module is not None and hasattr(model, "get_features"):
                        features = model.get_features(feats)
                        logits = model.classifier(features)
                    else:
                        logits = model(feats)
                    loss_x = criterion(logits, labels)
                else:
                    images, labels = batch_data
                    images, labels = images.to(device), labels.to(device)
                    if loss_pred_module is not None and hasattr(model, "get_features"):
                        features = model.get_features(images)
                        if isinstance(model, ResNet18):
                            logits = model.backbone.fc(features)
                        elif hasattr(model, "classifier"):
                            logits = model.classifier(features)
                        else:
                            logits = model(images)
                    else:
                        logits = model(images)
                    loss_x = criterion(logits, labels)

                loss = loss_x

                # --- Unlabeled Data Forward (FixMatch) ---
                if unlabeled_loader is not None:
                    try:
                        u_batch = next(unlabeled_iter)
                    except StopIteration:
                        unlabeled_iter = iter(unlabeled_loader)
                        u_batch = next(unlabeled_iter)

                    if unlabeled_view_mode == "image_views":
                        img_w, img_s, u_idx = u_batch
                        weak_inputs = img_w.to(device)
                        strong_inputs = img_s.to(device)
                        true_u_labels = None
                    elif unlabeled_view_mode == "ts_views":
                        series_w, series_s, true_u_labels = u_batch
                        weak_inputs = series_w.to(device)
                        strong_inputs = series_s.to(device)
                        true_u_labels = true_u_labels.to(device)
                        u_idx = None
                    else:
                        weak_inputs, true_u_labels = u_batch
                        weak_inputs = weak_inputs.to(device)
                        true_u_labels = true_u_labels.to(device)
                        if ssl_method == "dropout_consistency":
                            strong_inputs = tfidf_dropout(weak_inputs)
                        else:
                            strong_inputs = weak_inputs
                        u_idx = None

                    with torch.no_grad():
                        if ssl_method == "dropout_consistency":
                            was_training = model.training
                            model.train()
                            logits_passes = [model(weak_inputs) for _ in range(CFG.dropout_consistency_passes)]
                            probs_w = torch.stack(
                                [torch.softmax(logits_i, dim=-1) for logits_i in logits_passes],
                                dim=0,
                            ).mean(dim=0)
                            if not was_training:
                                model.eval()
                        else:
                            logits_w = model(weak_inputs)
                            probs_w = torch.softmax(logits_w, dim=-1)
                        max_probs, pseudo_labels = torch.max(probs_w, dim=-1)

                    if ssl_method == "vat":
                        strong_inputs = vat_perturb(model, weak_inputs, probs_w, CFG.vat_epsilon)

                    if flex_tracker is not None:
                        # Per-batch FlexMatch EMA update (matching original paper)
                        flex_tracker.update_per_batch(probs_w.float().cpu())
                        current_thresholds = flex_tracker.current_thresholds
                        threshold_tensor = torch.tensor(
                            [current_thresholds[l.item()] for l in pseudo_labels],
                            device=device, dtype=torch.float32
                        )
                    else:
                        threshold_tensor = torch.full_like(max_probs, CFG.ssl_threshold)
                    mask = max_probs.ge(threshold_tensor).float()

                    logits_s = model(strong_inputs)

                    # Compute unlabeled loss
                    if CFG.ssl_class_weighted and CFG.use_ssl:
                        # 创新点3: 类别加权一致性损失
                        # w_c = (1/C) / (n_c + 1), 归一化后均值=1
                        # 使用联合分布（labeled + pseudo-labels from previous round）
                        n_classes_ssl = flex_tracker.n_classes if flex_tracker is not None else get_num_classes(CFG.dataset)
                        if train_idx is not None and len(train_idx) > 0:
                            train_labels_arr = np.array([dataset[i][1] for i in train_idx])
                            # Use joint distribution if flag is set
                            use_joint = getattr(CFG, 'ssl_joint_distribution', False)
                            if use_joint and flex_tracker is not None and hasattr(flex_tracker, '_prev_pseudo_labels') and flex_tracker._prev_pseudo_labels is not None:
                                joint_for_weights = np.concatenate([train_labels_arr, flex_tracker._prev_pseudo_labels])
                            else:
                                joint_for_weights = train_labels_arr
                            class_counts = np.bincount(joint_for_weights.astype(int), minlength=n_classes_ssl).astype(np.float32)
                            mean_freq = 1.0 / n_classes_ssl
                            weights = mean_freq / (class_counts + 1.0)
                            weights = weights / weights.mean()  # 归一化使均值=1
                            weights_tensor = torch.tensor(weights, device=device, dtype=torch.float32)
                            sample_weights = weights_tensor[pseudo_labels]  # [B]
                        else:
                            sample_weights = torch.ones(pseudo_labels.shape[0], device=device)
                        loss_u = (F.cross_entropy(logits_s, pseudo_labels, reduction='none') * sample_weights * mask).mean()
                    else:
                        loss_u = (F.cross_entropy(logits_s, pseudo_labels, reduction='none') * mask).mean()

                    # Warmup-modulated loss combination
                    loss = loss_x + CFG.ssl_lambda_u * warmup_factor * loss_u

                    # Track pseudo-label diagnostics on the last epoch only.
                    if epoch == n_epochs - 1:
                        n_pseudo_labeled += int(mask.sum().item())

                        # Get true labels from raw_dataset for accuracy checks.
                        if raw_dataset is not None and u_idx is not None:
                            true_l = []
                            for real_idx in u_idx:
                                item = raw_dataset[int(real_idx)]
                                if isinstance(item, tuple):
                                    true_l.append(int(item[1]))
                                else:
                                    true_l.append(-1)
                            true_l = torch.tensor(true_l, device=device)

                            epoch_pseudo_labels.append(pseudo_labels.detach().cpu())
                            epoch_mask.append(mask.detach().cpu())
                            epoch_true_labels.append(true_l.detach().cpu())

                            correct = int(((pseudo_labels == true_l) * mask).sum().item())
                            pseudo_correct += correct
                            pseudo_total_checked += int(mask.sum().item())
                        elif true_u_labels is not None:
                            epoch_pseudo_labels.append(pseudo_labels.detach().cpu())
                            epoch_mask.append(mask.detach().cpu())
                            epoch_true_labels.append(true_u_labels.detach().cpu())

                            correct = int(((pseudo_labels == true_u_labels) * mask).sum().item())
                            pseudo_correct += correct
                            pseudo_total_checked += int(mask.sum().item())

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            if loss_pred_module is not None and features is not None:
                with torch.no_grad():
                    target_losses = F.cross_entropy(
                        logits.detach().float(), labels, reduction="none"
                    ).unsqueeze(1)
                lp_optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast("cuda", enabled=amp_enabled):
                    pred_losses = loss_pred_module(features.detach())
                    lp_loss = loss_pred_criterion(pred_losses, target_losses)
                scaler.scale(lp_loss).backward()
                scaler.step(lp_optimizer)
                scaler.update()

        if epoch == n_epochs - 1 and len(epoch_pseudo_labels) > 0:
            all_pl = torch.cat(epoch_pseudo_labels)
            all_mask = torch.cat(epoch_mask)
            all_true = torch.cat(epoch_true_labels)
            n_classes = flex_tracker.n_classes if flex_tracker is not None else get_num_classes(CFG.dataset)
            last_epoch_pseudo_stats = eval_pseudo_labels(all_pl, all_mask, all_true, n_classes)

        if early_stopping and val_dataset is not None and val_idx is not None:
            _, val_f1 = evaluate(model, val_dataset, val_idx, device)
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                patience_counter = 0
                best_state = deepcopy(model.state_dict())
            else:
                patience_counter += 1
                if patience_counter >= CFG.early_stopping_patience:
                    break

        if scheduler is not None:
            scheduler.step()

    if early_stopping and best_state is not None:
        model.load_state_dict(best_state)

    train_time = time.time() - t0
    gpu_mem_mb = 0
    if torch.cuda.is_available():
        gpu_mem_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)

    pseudo_acc = pseudo_correct / max(1, pseudo_total_checked)
    # Collect last-epoch pseudo-labels for joint distribution sensing
    last_epoch_pseudo_labels_np = None
    if len(epoch_pseudo_labels) > 0:
        all_pl = torch.cat(epoch_pseudo_labels)
        last_epoch_pseudo_labels_np = all_pl.numpy().astype(int)
    return train_time, gpu_mem_mb, n_pseudo_labeled, pseudo_acc, last_epoch_pseudo_stats or {}, last_epoch_pseudo_labels_np


@torch.no_grad()
def evaluate(model, dataset, test_idx, device):
    model.eval()
    subset = Subset(dataset, test_idx)
    loader = DataLoader(subset, batch_size=CFG.batch_size_infer, shuffle=False, num_workers=0)
    amp_enabled = bool(CFG.use_amp and device.type == "cuda")

    all_preds, all_labels = [], []
    for batch_data in loader:
        with torch.amp.autocast("cuda", enabled=amp_enabled):
            if CFG.pretrained_features or CFG.dataset in ["agnews", "adult"]:
                feats, labels = batch_data
                feats = feats.to(device)
                outputs = model(feats)
            else:
                images, labels = batch_data
                images = images.to(device)
                outputs = model(images)
        preds = outputs.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.numpy())

    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='macro')
    return acc, f1


@torch.no_grad()
def get_probs_and_features(model, dataset, indices, device, keep_train_mode=False):
    if not keep_train_mode:
        model.eval()
    subset = Subset(dataset, indices)
    loader = DataLoader(subset, batch_size=CFG.batch_size_infer, shuffle=False, num_workers=0)
    amp_enabled = bool(CFG.use_amp and device.type == "cuda")

    all_probs, all_features = [], []
    for batch_data in loader:
        with torch.amp.autocast("cuda", enabled=amp_enabled):
            # Direction 3: pretrained features are flat vectors, use tabular path
            if CFG.pretrained_features or CFG.dataset in ["agnews", "adult"]:
                feats, _ = batch_data
                feats = feats.to(device)
                logits, features = model(feats, return_features=True)
            else:
                images, _ = batch_data
                images = images.to(device)
                if isinstance(model, ResNet18):
                    features = model.get_features(images)
                    logits = model.backbone.fc(features)
                elif hasattr(model, 'classifier'):
                    features = model.get_features(images)
                    logits = model.classifier(features)
                else:
                    logits = model(images)
                    features = model.get_features(images)
        probs = F.softmax(logits, dim=1)
        all_probs.append(probs.cpu().numpy())
        all_features.append(features.cpu().numpy())

    return np.vstack(all_probs), np.vstack(all_features)


def subsample_pool(pool_idx, rng, max_size):
    if len(pool_idx) <= max_size:
        return pool_idx, set()
    subpool = rng.choice(pool_idx, max_size, replace=False).tolist()
    return subpool, set(pool_idx) - set(subpool)


def _jsonable(obj):
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    return obj


def get_flex_tracker_state(flex_tracker):
    if flex_tracker is None:
        return None
    return {
        "n_classes": flex_tracker.n_classes,
        "base_threshold": flex_tracker.base_threshold,
        "ema_momentum": flex_tracker.ema_momentum,
        "min_threshold": flex_tracker.min_threshold,
        "classwise_acc": flex_tracker.classwise_acc.cpu().tolist(),
        "current_thresholds": flex_tracker.current_thresholds.tolist(),
        "class_seen": flex_tracker._class_seen.cpu().tolist(),
    }


def restore_flex_tracker_state(flex_tracker, state):
    if flex_tracker is None or not state:
        return
    flex_tracker.classwise_acc = torch.tensor(state["classwise_acc"], dtype=torch.float32)
    flex_tracker.current_thresholds = np.array(state["current_thresholds"], dtype=np.float32)
    flex_tracker._class_seen = torch.tensor(state["class_seen"], dtype=torch.bool)


def create_qbc_member(member_idx, device):
    if CFG.pretrained_features or CFG.dataset in ["agnews", "adult"]:
        return create_model(CFG.dataset, device)
    if CFG.dataset == "forda":
        return TimeSeriesCNN(
            num_classes=get_num_classes(CFG.dataset),
            in_channels=1,
            hidden_dim=32 if member_idx == 2 else 64,
            dropout=0.2 + 0.05 * member_idx,
        ).to(device)
    if CFG.dataset == "ecg5000":
        return TimeSeriesCNN(
            num_classes=get_num_classes(CFG.dataset),
            in_channels=1,
            hidden_dim=32 if member_idx == 2 else 64,
            dropout=0.2 + 0.05 * member_idx,
        ).to(device)
    if CFG.dataset == "spoken_arabic":
        return TimeSeriesCNN(
            num_classes=get_num_classes(CFG.dataset),
            in_channels=13,
            hidden_dim=32 if member_idx == 2 else 64,
            dropout=0.2 + 0.05 * member_idx,
        ).to(device)
    if CFG.dataset == "character_traj":
        return TimeSeriesCNN(
            num_classes=get_num_classes(CFG.dataset),
            in_channels=3,
            hidden_dim=32 if member_idx == 2 else 64,
            dropout=0.2 + 0.05 * member_idx,
        ).to(device)

    arch = "smallcnn" if member_idx == 2 else "simplecnn"
    if arch == "smallcnn":
        in_ch = 1 if CFG.dataset == "fashion_mnist" else 3
        return SmallCNN(num_classes=get_num_classes(CFG.dataset), in_channels=in_ch).to(device)
    return create_model(CFG.dataset, device)


def train_committee(dataset, train_idx, device, n_committee, n_epochs=None):
    # Committee members are intentionally supervised-only; SSL is reserved for
    # the main AL model so QBC disagreement remains independent.
    committee = []
    for i in range(n_committee):
        member = create_qbc_member(i, device)
        torch.manual_seed(42 + i * 1000)
        train_model(member, dataset, train_idx, device, n_epochs=n_epochs or CFG.committee_epochs)
        committee.append(member)
    return committee


@torch.no_grad()
def get_committee_probs(committee, dataset, indices, device):
    all_member_probs = []
    for member in committee:
        probs, _ = get_probs_and_features(member, dataset, indices, device)
        all_member_probs.append(probs)
    return all_member_probs


def save_checkpoint(
    output_dir,
    strategy,
    seed,
    round_idx,
    labeled_idx,
    pool_idx,
    results,
    full_sup_done=False,
    full_sup_result=None,
    model_state=None,
    loss_pred_state=None,
    rng_state=None,
    torch_rng_state=None,
    cuda_rng_state_all=None,
    flex_tracker_state=None,
    committee_state=None,
):
    ckpt_dir = os.path.join(output_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, f"{strategy}_seed{seed}.json")
    ckpt = {
        "strategy": strategy,
        "seed": seed,
        "round_idx": round_idx,
        "labeled_idx": sorted(labeled_idx),
        "pool_idx": sorted(pool_idx),
        "results": results,
        "full_sup_done": full_sup_done,
        "full_sup_result": full_sup_result,
        "rng_state": _jsonable(rng_state),
        "flex_tracker_state": _jsonable(flex_tracker_state),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(ckpt_path, "w") as f:
        json.dump(ckpt, f)
    if model_state is not None:
        model_ckpt_path = os.path.join(ckpt_dir, f"{strategy}_seed{seed}_model.pt")
        torch.save({
            "model": model_state,
            "loss_pred": loss_pred_state,
            "torch_rng_state": torch_rng_state,
            "cuda_rng_state_all": cuda_rng_state_all,
            "committee": committee_state,
        }, model_ckpt_path)
    return ckpt_path


def load_checkpoint(output_dir, strategy, seed):
    ckpt_path = os.path.join(output_dir, "checkpoints", f"{strategy}_seed{seed}.json")
    if not os.path.exists(ckpt_path):
        return None
    with open(ckpt_path, "r") as f:
        ckpt = json.load(f)
    model_ckpt_path = os.path.join(output_dir, "checkpoints", f"{strategy}_seed{seed}_model.pt")
    if os.path.exists(model_ckpt_path):
        ckpt["model_state"] = torch.load(model_ckpt_path, map_location="cpu")
    return ckpt


def make_checkpoint_state(rng, flex_tracker=None, committee_cache=None):
    committee_state = None
    if committee_cache is not None:
        committee_state = [member.state_dict() for member in committee_cache]
    return {
        "rng_state": rng.bit_generator.state,
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state_all": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "flex_tracker_state": get_flex_tracker_state(flex_tracker),
        "committee_state": committee_state,
    }


def compute_statistical_tests(all_results):
    strategies = sorted(set(r["strategy"] for r in all_results))
    strat_final_f1s = {}
    for strat in strategies:
        strat_final_f1s[strat] = [r["f1_scores"][-1] for r in all_results if r["strategy"] == strat]

    stat_results = {}
    pairs = []
    for i, s1 in enumerate(strategies):
        for s2 in strategies[i+1:]:
            f1_1 = strat_final_f1s.get(s1, [])
            f1_2 = strat_final_f1s.get(s2, [])
            if len(f1_1) >= 2 and len(f1_2) >= 2:
                min_len = min(len(f1_1), len(f1_2))
                f1_1_arr = np.array(f1_1[:min_len])
                f1_2_arr = np.array(f1_2[:min_len])
                try:
                    t_stat, p_value = stats.ttest_rel(f1_1_arr, f1_2_arr)
                except Exception:
                    t_stat, p_value = 0.0, 1.0
                diff = f1_1_arr - f1_2_arr
                pooled_std = np.sqrt((f1_1_arr.std()**2 + f1_2_arr.std()**2) / 2)
                cohens_d = float(diff.mean() / pooled_std) if pooled_std > 1e-10 else 0.0
                pairs.append({
                    "strategy_a": s1, "strategy_b": s2,
                    "mean_diff": float(diff.mean()),
                    "t_statistic": float(t_stat),
                    "p_value": float(p_value),
                    "cohens_d": cohens_d,
                    "significant_005": bool(p_value < 0.05),
                    "significant_001": bool(p_value < 0.01),
                    "effect_size": "large" if abs(cohens_d) >= 0.8 else ("medium" if abs(cohens_d) >= 0.5 else "small"),
                })
    stat_results["paired_comparisons"] = pairs
    stat_results["n_seeds"] = min(len(v) for v in strat_final_f1s.values()) if strat_final_f1s else 0
    return stat_results


def generate_tsne_plot(model, dataset, labeled_idx, pool_idx, device, output_path, round_idx):
    try:
        from sklearn.manifold import TSNE
    except ImportError:
        return

    all_idx = labeled_idx[:500] + pool_idx[:500]
    if len(all_idx) < 10:
        return

    probs, features = get_probs_and_features(model, dataset, all_idx, device)
    if features.shape[1] > 50:
        from sklearn.decomposition import PCA
        pca = PCA(n_components=50)
        features = pca.fit_transform(features)

    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(all_idx)-1))
    embedded = tsne.fit_transform(features)

    fig, ax = plt.subplots(figsize=(8, 8))
    n_labeled = min(len(labeled_idx), 500)
    ax.scatter(embedded[:n_labeled, 0], embedded[:n_labeled, 1], c='red', s=10, alpha=0.6, label='Labeled')
    ax.scatter(embedded[n_labeled:, 0], embedded[n_labeled:, 1], c='blue', s=5, alpha=0.3, label='Pool')
    ax.set_title(f"t-SNE Feature Space (Round {round_idx+1})")
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=100)
    plt.close()


def run_single_strategy(strategy_name, train_dataset, test_dataset, infer_dataset, device,
                        train_idx_all, test_idx, seed, raw_dataset=None):
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    n_initial = CFG.n_initial
    n_query = CFG.n_query

    ckpt = None
    if CFG.resume:
        ckpt = load_checkpoint(CFG.output_dir, strategy_name, seed)

    if ckpt is not None:
        labeled_idx = ckpt["labeled_idx"]
        pool_idx = ckpt["pool_idx"]
        results = ckpt["results"]
        start_round = ckpt["round_idx"] + 1
        labeled_set = set(labeled_idx)
        logger.info(f"  [{strategy_name}] Seed={seed} Resumed from R{ckpt['round_idx']+1}")
    else:
        all_indices = list(range(len(train_dataset))) if train_idx_all is None else list(train_idx_all)
        initial_idx = rng.choice(all_indices, n_initial, replace=False).tolist()
        labeled_idx = initial_idx[:]
        labeled_set = set(labeled_idx)
        pool_idx = [i for i in all_indices if i not in labeled_set]
        results = {
            "strategy": strategy_name, "seed": seed,
            "f1_scores": [], "accuracies": [],
            "n_human_labeled": [], "query_times": [], "train_times": [],
            "gpu_memory_mb": [], "n_pseudo_labeled": [], "query_class_dist": [],
            "pseudo_acc_history": [], "pseudo_stats_history": [],
        }
        start_round = 0

    model = create_model(CFG.dataset, device)
    if ckpt is not None and ckpt.get("model_state") is not None:
        model.load_state_dict(ckpt["model_state"]["model"])
        logger.info(f"  [{strategy_name}] Seed={seed} Restored model weights from checkpoint")
    flex_tracker = None
    base_flex_tracker = None  # 渐进式SSL: Base SSL tracker
    innov_flex_tracker = None  # 渐进式SSL: Innov SSL tracker
    progressive_ssl = False  # 渐进式SSL标志
    if CFG.use_ssl and resolve_ssl_method(CFG.dataset, CFG.ssl_method) == "flexmatch":
        progressive_ssl = (CFG.ssl_deficit_threshold and CFG.ssl_deficit_start_round > 0)
        if progressive_ssl:
            # 渐进式SSL: 创建两个tracker，后续按轮次切换
            base_flex_tracker = FlexMatchTracker(get_num_classes(CFG.dataset), threshold=CFG.ssl_threshold)
            innov_flex_tracker = DeficitAwareFlexMatchTracker(
                get_num_classes(CFG.dataset), threshold=CFG.ssl_threshold,
                deficit_alpha=CFG.ssl_deficit_alpha
            )
            flex_tracker = base_flex_tracker  # 初始使用Base SSL
            logger.info(f"  Progressive SSL: Base SSL for rounds 0-{CFG.ssl_deficit_start_round-1}, "
                        f"Innov SSL for round {CFG.ssl_deficit_start_round}+")
        elif CFG.ssl_deficit_threshold:
            flex_tracker = DeficitAwareFlexMatchTracker(
                get_num_classes(CFG.dataset), threshold=CFG.ssl_threshold,
                deficit_alpha=CFG.ssl_deficit_alpha
            )
        else:
            flex_tracker = FlexMatchTracker(get_num_classes(CFG.dataset), threshold=CFG.ssl_threshold)
        if ckpt is not None and ckpt.get("flex_tracker_state") is not None:
            restore_flex_tracker_state(flex_tracker, ckpt["flex_tracker_state"])

    loss_pred_module = None
    if strategy_name == "learning_loss":
        feature_dim = model._feature_dim
        loss_pred_module = LossPredictionModule(feature_dim).to(device)
        if ckpt is not None and ckpt.get("model_state") is not None and ckpt["model_state"].get("loss_pred") is not None:
            loss_pred_module.load_state_dict(ckpt["model_state"]["loss_pred"])

    committee_cache = None
    if strategy_name == "qbc" and CFG.qbc_reuse_committee and ckpt is not None:
        committee_states = (ckpt.get("model_state") or {}).get("committee")
        if committee_states is not None:
            committee_cache = []
            for i, state in enumerate(committee_states):
                member = create_qbc_member(i, device)
                member.load_state_dict(state)
                committee_cache.append(member)

    if ckpt is not None:
        if ckpt.get("rng_state") is not None:
            rng.bit_generator.state = ckpt["rng_state"]
        model_state = ckpt.get("model_state") or {}
        if model_state.get("torch_rng_state") is not None:
            torch.set_rng_state(model_state["torch_rng_state"])
        if torch.cuda.is_available() and model_state.get("cuda_rng_state_all") is not None:
            torch.cuda.set_rng_state_all(model_state["cuda_rng_state_all"])

    tsne_dir = os.path.join(CFG.output_dir, "tsne")
    if CFG.enable_tsne:
        os.makedirs(tsne_dir, exist_ok=True)

    for rd in range(start_round, CFG.n_rounds):
        # --- Progressive SSL: switch from Base to Innov at specified round ---
        if (innov_flex_tracker is not None and base_flex_tracker is not None
                and rd == CFG.ssl_deficit_start_round):
            flex_tracker = innov_flex_tracker
            # Transfer learned class statistics from base to innov tracker
            if hasattr(base_flex_tracker, 'class_acc') and hasattr(innov_flex_tracker, 'class_acc'):
                innov_flex_tracker.class_acc = base_flex_tracker.class_acc.copy()
            if hasattr(innov_flex_tracker, 'set_labeled_distribution'):
                labeled_labels_arr = np.array([train_dataset[i][1] for i in labeled_idx])
                innov_flex_tracker.set_labeled_distribution(labeled_labels_arr)
            logger.info(f"  Progressive SSL: switched to Innov SSL at round {rd}")

        # --- Direction 4: Staged AL-SSL decoupling ---
        # Determine whether SSL is active this round.
        rd_pool_idx = pool_idx
        rd_flex_tracker = flex_tracker
        if CFG.use_ssl and pool_idx is not None and len(pool_idx) > 0:
            # Strategy B: warmup — pure supervised for first N rounds
            if CFG.ssl_warmup_rounds > 0 and rd < CFG.ssl_warmup_rounds:
                rd_pool_idx = None
                rd_flex_tracker = None
                if rd == start_round:
                    logger.info(f"  SSL warmup: rounds 0-{CFG.ssl_warmup_rounds-1} supervised, "
                                f"round {CFG.ssl_warmup_rounds}+ SSL enabled")
            # Strategy C: adaptive — enable SSL only when pseudo-label accuracy is high enough
            elif CFG.ssl_adaptive:
                if len(results["pseudo_acc_history"]) == 0:
                    rd_pool_idx = None
                    rd_flex_tracker = None
                    if rd == start_round:
                        logger.info(f"  Adaptive SSL: paused (round 0, no history yet)")
                elif results["pseudo_acc_history"][-1] > 0:
                    prev_ps_acc = results["pseudo_acc_history"][-1]
                    if prev_ps_acc < CFG.ssl_adaptive_threshold:
                        rd_pool_idx = None
                        rd_flex_tracker = None
                        logger.info(f"  Adaptive SSL: paused (prev pseudo_acc={prev_ps_acc:.3f} < "
                                    f"threshold={CFG.ssl_adaptive_threshold})")
                    else:
                        if rd <= 2 or prev_ps_acc >= CFG.ssl_adaptive_threshold:
                            logger.info(f"  Adaptive SSL: active (prev pseudo_acc={prev_ps_acc:.3f} >= "
                                        f"threshold={CFG.ssl_adaptive_threshold})")
                else:
                    # ps_acc == 0 means SSL was off last round; probe model quality
                    # by checking validation F1 as proxy for pseudo-label readiness
                    val_f1 = results["f1_scores"][-1] if results["f1_scores"] else 0.0
                    f1_threshold = 0.15
                    if val_f1 < f1_threshold:
                        rd_pool_idx = None
                        rd_flex_tracker = None
                        logger.info(f"  Adaptive SSL: paused (val_f1={val_f1:.4f} < {f1_threshold}, model not ready)")
                    else:
                        logger.info(f"  Adaptive SSL: probing enabled (val_f1={val_f1:.4f} >= {f1_threshold})")

        ssl_active_rounds = 0
        if CFG.ssl_warmup_rounds > 0:
            ssl_active_rounds = max(0, rd - CFG.ssl_warmup_rounds)
        elif CFG.ssl_adaptive and len(results["pseudo_acc_history"]) > 0:
            ssl_active_rounds = sum(1 for a in results["pseudo_acc_history"] if a >= CFG.ssl_adaptive_threshold)
        else:
            ssl_active_rounds = rd

        # --- Direction 2: Round-level pseudo-label refresh ---
        # When pseudo_refresh_freq > 0, every K AL rounds we log the pseudo-label
        # quality improvement. For pseudo_label SSL method, this also triggers
        # re-inference on the pool to regenerate pseudo-labels from the improved model.
        pseudo_refresh_this_round = (
            CFG.pseudo_refresh_freq > 0
            and rd > 0
            and rd % CFG.pseudo_refresh_freq == 0
            and CFG.use_ssl
            and rd_pool_idx is not None
            and len(rd_pool_idx) > 0
        )
        if pseudo_refresh_this_round:
            logger.info(f"  Pseudo-label refresh at round {rd} (freq={CFG.pseudo_refresh_freq})")

        # 创新点2: 更新deficit-based阈值的标注分布
        if CFG.ssl_deficit_threshold and rd_flex_tracker is not None and hasattr(rd_flex_tracker, 'set_labeled_distribution'):
            labeled_labels_arr = np.array([train_dataset[i][1] for i in labeled_idx])
            # Use joint distribution if flag is set and pseudo-labels available
            use_joint = getattr(CFG, 'ssl_joint_distribution', False)
            if use_joint and hasattr(rd_flex_tracker, '_prev_pseudo_labels') and rd_flex_tracker._prev_pseudo_labels is not None:
                joint_labels = np.concatenate([labeled_labels_arr, rd_flex_tracker._prev_pseudo_labels])
                rd_flex_tracker.set_labeled_distribution(joint_labels)
            else:
                rd_flex_tracker.set_labeled_distribution(labeled_labels_arr)

        # 渐进式SSL: 按轮次切换class_weighted标志
        orig_class_weighted = CFG.ssl_class_weighted
        if progressive_ssl and rd < CFG.ssl_deficit_start_round:
            CFG.ssl_class_weighted = False  # 前N轮关闭类别加权

        train_t, gpu_mem, n_ps_labeled, ps_acc, ps_stats, last_pseudo_labels = train_model(
            model, train_dataset, labeled_idx, device,
            loss_pred_module=loss_pred_module, pool_idx=rd_pool_idx,
            flex_tracker=rd_flex_tracker, raw_dataset=raw_dataset,
            warmup_offset_epochs=ssl_active_rounds * CFG.n_epochs_base,
        )
        # Save pseudo-labels for next round's joint distribution
        if rd_flex_tracker is not None and last_pseudo_labels is not None:
            rd_flex_tracker._prev_pseudo_labels = last_pseudo_labels
        # 渐进式SSL: 恢复原始class_weighted标志
        if progressive_ssl:
            CFG.ssl_class_weighted = orig_class_weighted

        acc, f1 = evaluate(model, test_dataset, test_idx, device)

        results["f1_scores"].append(float(f1))
        results["accuracies"].append(float(acc))
        results["n_human_labeled"].append(len(labeled_idx))
        results["train_times"].append(train_t)
        results["gpu_memory_mb"].append(gpu_mem)
        results["n_pseudo_labeled"].append(n_ps_labeled)
        results["pseudo_acc_history"].append(ps_acc)
        results["pseudo_stats_history"].append(ps_stats)

        if CFG.enable_tsne and rd % 3 == 0:
            tsne_path = os.path.join(tsne_dir, f"{strategy_name}_seed{seed}_r{rd}.png")
            generate_tsne_plot(model, infer_dataset, labeled_idx, pool_idx, device, tsne_path, rd)

        if rd == CFG.n_rounds - 1:
            results["query_times"].append(0.0)  # no query in final round
            ckpt_state = make_checkpoint_state(rng, flex_tracker, committee_cache)
            save_checkpoint(CFG.output_dir, strategy_name, seed, rd, labeled_idx, pool_idx, results,
                            model_state=model.state_dict(),
                            loss_pred_state=loss_pred_module.state_dict() if loss_pred_module else None,
                            **ckpt_state)
            break

        subpool_idx, excluded = subsample_pool(pool_idx, rng, CFG.max_pool_subsample)
        if len(subpool_idx) == 0:
            results["query_times"].append(0.0)
            ckpt_state = make_checkpoint_state(rng, flex_tracker, committee_cache)
            save_checkpoint(CFG.output_dir, strategy_name, seed, rd, labeled_idx, pool_idx, results,
                            model_state=model.state_dict(),
                            loss_pred_state=loss_pred_module.state_dict() if loss_pred_module else None,
                            **ckpt_state)
            logger.info(f"  [{strategy_name}] Seed={seed} R{rd+1}/{CFG.n_rounds} "
                  f"F1={f1:.4f} Acc={acc:.4f} Labeled={len(labeled_idx)} "
                  f"QueryTime=0.0s GPU={gpu_mem:.0f}MB (pool empty)")
            continue
        # Use infer_dataset (no augmentation) for consistent feature extraction.
        probs, features, labeled_features = None, None, None
        if strategy_name == "coreset":
            query_idx = subpool_idx + labeled_idx
            query_probs, query_features = get_probs_and_features(model, infer_dataset, query_idx, device)
            n_subpool = len(subpool_idx)
            probs = query_probs[:n_subpool]
            features = query_features[:n_subpool]
            labeled_features = query_features[n_subpool:]
        elif strategy_name in ["least_confidence", "margin", "entropy", "badge", "learning_loss", "class_aware_entropy", "gap_aware_entropy", "adaptive_gap_entropy", "two_stage_entropy_balance", "curriculum_penalty_entropy", "calibrated_entropy", "calibrated_margin", "class_aware_entropy_ssl", "gap_aware_entropy_ssl"]:
            probs, features = get_probs_and_features(model, infer_dataset, subpool_idx, device)
        elif strategy_name in ["dynamic_typiclust_margin", "dynamic_typiclust_entropy", "dynamic_coreset_margin", "dynamic_coverage_uncertainty"]:
            probs, features = get_probs_and_features(model, infer_dataset, subpool_idx, device)
        elif strategy_name in ["noise_aware_entropy", "boundary_aware_entropy", "direct_style"]:
            probs, features = get_probs_and_features(model, infer_dataset, subpool_idx, device)

        query_t0 = time.time()
        if strategy_name == "random":
            selected = select_random(subpool_idx, n_query, rng)
        elif strategy_name == "least_confidence":
            selected = select_least_confidence(probs, subpool_idx, n_query)
        elif strategy_name == "margin":
            selected = select_margin(probs, subpool_idx, n_query)
        elif strategy_name == "entropy":
            selected = select_uncertainty(probs, subpool_idx, n_query)
        elif strategy_name == "class_aware_entropy":
            labeled_labels_arr = np.array([train_dataset[i][1] for i in labeled_idx])
            if isinstance(labeled_labels_arr[0], torch.Tensor):
                labeled_labels_arr = np.array([l.item() for l in labeled_labels_arr])
            n_cls = get_num_classes(CFG.dataset)
            lam = getattr(CFG, "class_aware_lambda", 0.5)
            use_adaptive = getattr(CFG, "class_aware_adaptive", True)
            use_soft = getattr(CFG, "class_aware_soft_weighting", True)
            selected = select_class_aware_entropy(
                probs, subpool_idx, n_query, labeled_labels_arr, n_cls, lam=lam,
                adaptive_lambda=use_adaptive, soft_weighting=use_soft)
        elif strategy_name == "gap_aware_entropy":
            labeled_labels_arr = np.array([train_dataset[i][1] for i in labeled_idx])
            if isinstance(labeled_labels_arr[0], torch.Tensor):
                labeled_labels_arr = np.array([l.item() for l in labeled_labels_arr])
            n_cls = get_num_classes(CFG.dataset)
            lam = getattr(CFG, "class_aware_lambda", 0.5)
            selected = select_gap_aware_entropy(
                probs, subpool_idx, n_query, labeled_labels_arr, n_cls, lam=lam)
        elif strategy_name == "adaptive_gap_entropy":
            labeled_labels_arr = np.array([train_dataset[i][1] for i in labeled_idx])
            if isinstance(labeled_labels_arr[0], torch.Tensor):
                labeled_labels_arr = np.array([l.item() for l in labeled_labels_arr])
            n_cls = get_num_classes(CFG.dataset)
            lam_max = getattr(CFG, "class_aware_lambda", 0.5)
            selected = select_adaptive_gap_entropy(
                probs, subpool_idx, n_query, labeled_labels_arr, n_cls, lam_max=lam_max)
        elif strategy_name == "two_stage_entropy_balance":
            labeled_labels_arr = np.array([train_dataset[i][1] for i in labeled_idx])
            if isinstance(labeled_labels_arr[0], torch.Tensor):
                labeled_labels_arr = np.array([l.item() for l in labeled_labels_arr])
            n_cls = get_num_classes(CFG.dataset)
            selected = select_two_stage_entropy_balance(
                probs, subpool_idx, n_query, labeled_labels_arr, n_cls, coarse_factor=3)
        elif strategy_name == "curriculum_penalty_entropy":
            labeled_labels_arr = np.array([train_dataset[i][1] for i in labeled_idx])
            if isinstance(labeled_labels_arr[0], torch.Tensor):
                labeled_labels_arr = np.array([l.item() for l in labeled_labels_arr])
            n_cls = get_num_classes(CFG.dataset)
            lam = getattr(CFG, "class_aware_lambda", 0.5)
            warmup_rounds = getattr(CFG, "curriculum_warmup_rounds", 5)
            selected = select_curriculum_penalty_entropy(
                probs, subpool_idx, n_query, labeled_labels_arr, n_cls,
                lam=lam, current_round=rd, warmup_rounds=warmup_rounds)
        elif strategy_name == "class_aware_entropy_ssl":
            labeled_labels_arr = np.array([train_dataset[i][1] for i in labeled_idx])
            if isinstance(labeled_labels_arr[0], torch.Tensor):
                labeled_labels_arr = np.array([l.item() for l in labeled_labels_arr])
            n_cls = get_num_classes(CFG.dataset)
            lam = getattr(CFG, "class_aware_lambda", 0.5)
            use_adaptive = getattr(CFG, "class_aware_adaptive", False)
            use_soft = getattr(CFG, "class_aware_soft_weighting", False)
            # Progressive joint distribution: use labeled-only for early rounds
            joint_start = getattr(CFG, "joint_start_round", 0)
            if probs is not None and rd >= joint_start:
                pseudo_labels = probs.argmax(axis=1)
                if use_adaptive or use_soft:
                    max_probs = probs.max(axis=1)
                    threshold = getattr(CFG, "ssl_threshold", 0.95)
                    pseudo_labels[max_probs < threshold] = -1
            else:
                pseudo_labels = None
            selected = select_class_aware_entropy_ssl(
                probs, subpool_idx, n_query, labeled_labels_arr, n_cls,
                pseudo_labels=pseudo_labels, lam=lam,
                adaptive_lambda=use_adaptive, soft_weighting=use_soft)
        elif strategy_name == "gap_aware_entropy_ssl":
            labeled_labels_arr = np.array([train_dataset[i][1] for i in labeled_idx])
            if isinstance(labeled_labels_arr[0], torch.Tensor):
                labeled_labels_arr = np.array([l.item() for l in labeled_labels_arr])
            n_cls = get_num_classes(CFG.dataset)
            lam = getattr(CFG, "class_aware_lambda", 0.5)
            # Progressive joint distribution
            joint_start = getattr(CFG, "joint_start_round", 0)
            if probs is not None and rd >= joint_start:
                pseudo_labels = probs.argmax(axis=1)
            else:
                pseudo_labels = None
            selected = select_gap_aware_entropy_ssl(
                probs, subpool_idx, n_query, labeled_labels_arr, n_cls,
                pseudo_labels=pseudo_labels, lam=lam)
        elif strategy_name == "coreset":
            selected = select_coreset(features, subpool_idx, n_query, rng, labeled_features=labeled_features)
        elif strategy_name == "badge":
            grad_embeds = compute_grad_embeddings(probs, features)
            selected = select_badge(grad_embeds, subpool_idx, n_query, rng)
        elif strategy_name == "qbc":
            if CFG.qbc_reuse_committee:
                if committee_cache is None:
                    committee_cache = train_committee(train_dataset, labeled_idx, device, CFG.n_committee)
                else:
                    for member in committee_cache:
                        train_model(
                            member, train_dataset, labeled_idx, device,
                            n_epochs=CFG.qbc_incremental_epochs,
                        )
                committee = committee_cache
            else:
                committee = train_committee(train_dataset, labeled_idx, device, CFG.n_committee)
            committee_probs = get_committee_probs(committee, infer_dataset, subpool_idx, device)
            selected = select_qbc(committee_probs, subpool_idx, n_query)
            if not CFG.qbc_reuse_committee:
                del committee
        elif strategy_name == "bald":
            model.train()
            mc_probs_list = []
            for _ in range(CFG.bald_mc_samples):
                with torch.no_grad():
                    p, _ = get_probs_and_features(model, infer_dataset, subpool_idx, device, keep_train_mode=True)
                mc_probs_list.append(p)
            model.eval()
            selected = select_bald(mc_probs_list, subpool_idx, n_query)
        elif strategy_name == "learning_loss":
            with torch.no_grad():
                feat_tensor = torch.from_numpy(features).float().to(device)
                pred_losses = loss_pred_module(feat_tensor).squeeze(-1).cpu().numpy()
            top_k = np.argsort(pred_losses)[-min(n_query, len(subpool_idx)):]
            selected = [subpool_idx[i] for i in top_k]
        # --- Direction 1: Dynamic strategy switching ---
        elif strategy_name == "dynamic_typiclust_margin":
            selected = select_dynamic_switch(
                probs, features, subpool_idx, n_query, rng,
                current_round=rd, n_rounds=CFG.n_rounds,
                labeled_f1_history=results["f1_scores"],
                strategy="typiclust_to_margin",
                switch_point=CFG.dynamic_switch_point,
                competence_method=CFG.dynamic_competence_method,
            )
        elif strategy_name == "dynamic_typiclust_entropy":
            selected = select_dynamic_switch(
                probs, features, subpool_idx, n_query, rng,
                current_round=rd, n_rounds=CFG.n_rounds,
                labeled_f1_history=results["f1_scores"],
                strategy="typiclust_to_entropy",
                switch_point=CFG.dynamic_switch_point,
                competence_method=CFG.dynamic_competence_method,
            )
        elif strategy_name == "dynamic_coreset_margin":
            # Need labeled_features for coreset
            query_idx = subpool_idx + labeled_idx
            query_probs, query_features = get_probs_and_features(model, infer_dataset, query_idx, device)
            n_subpool = len(subpool_idx)
            labeled_features = query_features[n_subpool:]
            selected = select_dynamic_switch(
                probs, features, subpool_idx, n_query, rng,
                current_round=rd, n_rounds=CFG.n_rounds,
                labeled_f1_history=results["f1_scores"],
                strategy="coreset_to_margin",
                switch_point=CFG.dynamic_switch_point,
                competence_method=CFG.dynamic_competence_method,
            )
        elif strategy_name == "dynamic_coverage_uncertainty":
            selected = select_dynamic_switch(
                probs, features, subpool_idx, n_query, rng,
                current_round=rd, n_rounds=CFG.n_rounds,
                labeled_f1_history=results["f1_scores"],
                strategy="coverage_to_uncertainty",
                switch_point=CFG.dynamic_switch_point,
                competence_method=CFG.dynamic_competence_method,
            )
        # --- Direction 2: Calibrated uncertainty sampling ---
        elif strategy_name == "calibrated_entropy":
            selected, temperature = select_calibrated_entropy(
                model, infer_dataset, subpool_idx, labeled_idx, n_query, device,
                batch_size=CFG.batch_size_infer,
            )
            if rd == 0 or rd == CFG.n_rounds - 1:
                logger.info(f"  [{strategy_name}] Temperature={temperature:.3f}")
        elif strategy_name == "calibrated_margin":
            selected, temperature = select_calibrated_margin(
                model, infer_dataset, subpool_idx, labeled_idx, n_query, device,
                batch_size=CFG.batch_size_infer,
            )
            if rd == 0 or rd == CFG.n_rounds - 1:
                logger.info(f"  [{strategy_name}] Temperature={temperature:.3f}")
        # --- Direction 3: Confirmation Bias robustness ---
        elif strategy_name == "noise_aware_entropy":
            labeled_labels_arr = np.array([train_dataset[i][1] for i in labeled_idx])
            if isinstance(labeled_labels_arr[0], torch.Tensor):
                labeled_labels_arr = np.array([l.item() for l in labeled_labels_arr])
            selected = select_noise_aware_query(
                probs, subpool_idx, n_query, labeled_idx, labeled_labels_arr,
                model, infer_dataset, device,
                noise_threshold=0.3, alpha=0.5,
            )
        # --- Direction 4: Boundary-aware AL ---
        elif strategy_name == "boundary_aware_entropy":
            labeled_labels_arr = np.array([train_dataset[i][1] for i in labeled_idx])
            if isinstance(labeled_labels_arr[0], torch.Tensor):
                labeled_labels_arr = np.array([l.item() for l in labeled_labels_arr])
            n_cls = get_num_classes(CFG.dataset)
            selected = select_boundary_aware_entropy(
                probs, features, subpool_idx, n_query,
                labeled_idx, labeled_labels_arr, n_cls,
                beta=CFG.boundary_aware_beta,
                gamma=CFG.boundary_aware_gamma,
            )
        elif strategy_name == "direct_style":
            labeled_labels_arr = np.array([train_dataset[i][1] for i in labeled_idx])
            if isinstance(labeled_labels_arr[0], torch.Tensor):
                labeled_labels_arr = np.array([l.item() for l in labeled_labels_arr])
            n_cls = get_num_classes(CFG.dataset)
            selected = select_direct_style_query(
                probs, features, subpool_idx, n_query,
                labeled_idx, labeled_labels_arr, n_cls,
                lambda_sep=0.5,
            )
        else:
            raise ValueError(f"Unknown strategy: {strategy_name}")

        query_time = time.time() - query_t0
        results["query_times"].append(query_time)
        
        # --- Visualization Instrumentation ---
        # 1. Track Query Class Distribution
        try:
            if hasattr(train_dataset, 'dataset') and hasattr(train_dataset.dataset, 'targets'):
                true_labels = [int(train_dataset.dataset.targets[idx]) for idx in selected]
            elif hasattr(train_dataset, 'labels'):
                true_labels = [int(train_dataset.labels[idx]) for idx in selected]
            else:
                true_labels = [int(train_dataset[idx][1]) for idx in selected]
                
            dist = {str(k): v for k, v in dict(Counter(true_labels)).items()}
            results["query_class_dist"].append(dist)
        except Exception as e:
            results["query_class_dist"].append({})
            
        # 2. Grad-CAM for most uncertain (only for image datasets and uncertainty strategies)
        if CFG.dataset in ["fashion_mnist", "cifar10", "cifar100", "bloodmnist"] and strategy_name in ["entropy", "least_confidence", "margin"]:
            gradcam_dir = os.path.join(CFG.output_dir, "gradcam", f"{strategy_name}_seed{seed}")
            os.makedirs(gradcam_dir, exist_ok=True)
            # The selected array is sorted by uncertainty (highest first) for these strategies
            generate_and_save_gradcam(model, train_dataset, selected[:3], device, gradcam_dir, prefix=f"r{rd}")
        

        for idx in selected:
            if idx not in labeled_set:
                labeled_idx.append(idx)
                labeled_set.add(idx)
        pool_idx = [i for i in pool_idx if i not in labeled_set]

        ckpt_state = make_checkpoint_state(rng, flex_tracker, committee_cache)
        save_checkpoint(CFG.output_dir, strategy_name, seed, rd, labeled_idx, pool_idx, results,
                        model_state=model.state_dict(),
                        loss_pred_state=loss_pred_module.state_dict() if loss_pred_module else None,
                        **ckpt_state)

        logger.info(f"  [{strategy_name}] Seed={seed} R{rd+1}/{CFG.n_rounds} "
              f"F1={f1:.4f} Acc={acc:.4f} Labeled={len(labeled_idx)} "
              f"QueryTime={query_time:.1f}s GPU={gpu_mem:.0f}MB"
              f"{' SSL+' + str(results['n_pseudo_labeled'][-1]) if CFG.use_ssl else ''}")

    return results


def run_full_supervised(train_dataset, test_dataset, test_idx, device, train_idx_all=None):
    all_idx = list(range(len(train_dataset))) if train_idx_all is None else list(train_idx_all)
    model = create_model(CFG.dataset, device)
    torch.manual_seed(42)
    train_t, gpu_mem, _, _, _, _ = train_model(
        model, train_dataset, all_idx, device,
        n_epochs=CFG.full_supervised_epochs,
        early_stopping=True,
        val_dataset=test_dataset,
        val_idx=test_idx,
    )
    acc, f1 = evaluate(model, test_dataset, test_idx, device)
    return {"accuracy": float(acc), "f1": float(f1), "train_time": train_t, "gpu_memory_mb": gpu_mem, "epochs": CFG.full_supervised_epochs}


def aggregate_results(all_results):
    strategies = sorted(set(r["strategy"] for r in all_results))
    aggregated = {}
    for strat in strategies:
        strat_results = [r for r in all_results if r["strategy"] == strat]
        n_rounds = len(strat_results[0]["f1_scores"])
        f1_arrays = np.array([r["f1_scores"] for r in strat_results])
        acc_arrays = np.array([r["accuracies"] for r in strat_results])
        aggregated[strat] = {
            "f1_mean": f1_arrays.mean(axis=0).tolist(),
            "f1_std": f1_arrays.std(axis=0).tolist(),
            "acc_mean": acc_arrays.mean(axis=0).tolist(),
            "acc_std": acc_arrays.std(axis=0).tolist(),
            "final_f1_mean": float(f1_arrays[:, -1].mean()),
            "final_f1_std": float(f1_arrays[:, -1].std()),
            "best_f1_mean": float(f1_arrays.max(axis=1).mean()),
            "best_f1_std": float(f1_arrays.max(axis=1).std()),
            "n_seeds": len(strat_results),
            "seeds": [r["seed"] for r in strat_results],
        }
        if strat_results[0].get("train_times"):
            train_times = np.array([r["train_times"] for r in strat_results])
            aggregated[strat]["avg_train_time_per_round"] = float(train_times.mean())
            aggregated[strat]["total_train_time"] = float(train_times.sum())
        if strat_results[0].get("gpu_memory_mb"):
            gpu_mems = np.array([r["gpu_memory_mb"] for r in strat_results])
            aggregated[strat]["peak_gpu_memory_mb"] = float(gpu_mems.max())
        if strat_results[0].get("query_times"):
            query_times = np.array([r["query_times"] for r in strat_results])
            aggregated[strat]["avg_query_time"] = float(query_times.mean())
            aggregated[strat]["total_query_time"] = float(query_times.sum())
        if strat_results[0].get("n_human_labeled"):
            n_labeled = np.array([r["n_human_labeled"] for r in strat_results])
            aggregated[strat]["n_human_labeled_mean"] = n_labeled.mean(axis=0).tolist()
        if strat_results[0].get("n_pseudo_labeled"):
            pseudo = np.array([r["n_pseudo_labeled"] for r in strat_results])
            aggregated[strat]["total_pseudo_labeled"] = int(pseudo.sum())
    return aggregated


def plot_results(aggregated, output_dir, all_results=None):
    os.makedirs(output_dir, exist_ok=True)
    strategies = sorted([k for k in aggregated.keys() if k != "full_supervision" and "f1_mean" in aggregated[k]])
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(strategies), 1)))

    # --- 1. Strategy comparison: X-axis = labeled sample count ---
    fig, ax = plt.subplots(figsize=(10, 6))
    for i, strat in enumerate(strategies):
        data = aggregated[strat]
        x_axis = data.get("n_human_labeled_mean", list(range(1, len(data["f1_mean"]) + 1)))
        ax.plot(x_axis, data["f1_mean"], label=strat, color=colors[i], marker='o', markersize=3)
        ax.fill_between(x_axis,
                         np.array(data["f1_mean"]) - np.array(data["f1_std"]),
                         np.array(data["f1_mean"]) + np.array(data["f1_std"]),
                         color=colors[i], alpha=0.15)
    if "full_supervision" in aggregated:
        fs = aggregated["full_supervision"]
        ax.axhline(y=fs["f1"], color='black', linestyle='--', label=f'Full Supervised ({fs["f1"]:.4f})')
    ax.set_xlabel("Number of Labeled Samples")
    ax.set_ylabel("Macro-F1")
    ax.set_title(f"AL Strategy Comparison - {CFG.dataset} ({CFG.budget_level}, {CFG.model_type})")
    ax.legend(loc="lower right", fontsize=7)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "strategy_comparison.png"), dpi=150)
    plt.close()

    # --- 2. Final F1 bar chart ---
    fig, ax = plt.subplots(figsize=(10, 6))
    final_f1s = [aggregated[s]["final_f1_mean"] for s in strategies]
    final_stds = [aggregated[s]["final_f1_std"] for s in strategies]
    x_pos = range(len(strategies))
    ax.bar(x_pos, final_f1s, yerr=final_stds, capsize=4,
           color=[colors[i] for i in range(len(strategies))], alpha=0.8)
    if "full_supervision" in aggregated:
        ax.axhline(y=aggregated["full_supervision"]["f1"], color='black', linestyle='--', label='Full Supervised')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(strategies, rotation=30, ha='right', fontsize=8)
    ax.set_ylabel("Final Macro-F1 (mean\u00b1std)")
    ax.set_title(f"Final Performance - {CFG.dataset} ({CFG.budget_level}, {CFG.model_type})")
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "final_f1_bar.png"), dpi=150)
    plt.close()

    # --- 3. Labeling efficiency curve ---
    if "full_supervision" in aggregated:
        fs_f1 = aggregated["full_supervision"]["f1"]
        fig, ax = plt.subplots(figsize=(10, 6))
        for i, strat in enumerate(strategies):
            data = aggregated[strat]
            x_axis = data.get("n_human_labeled_mean", list(range(1, len(data["f1_mean"]) + 1)))
            efficiency = [f / fs_f1 * 100 for f in data["f1_mean"]]
            ax.plot(x_axis, efficiency, label=strat, color=colors[i], marker='o', markersize=3)
        ax.axhline(y=95, color='grey', linestyle=':', alpha=0.7, label='95% of Full Supervised')
        ax.set_xlabel("Number of Labeled Samples")
        ax.set_ylabel("% of Full Supervised F1")
        ax.set_title(f"Labeling Efficiency - {CFG.dataset} ({CFG.budget_level})")
        ax.legend(loc="lower right", fontsize=7)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "labeling_efficiency.png"), dpi=150)
        plt.close()

    # --- 4. Cost-performance Pareto chart ---
    has_cost = all(aggregated[s].get("total_query_time", 0) > 0 for s in strategies)
    if has_cost and strategies:
        fig, ax = plt.subplots(figsize=(10, 6))
        for i, strat in enumerate(strategies):
            data = aggregated[strat]
            total_time = data.get("total_query_time", 0) + data.get("total_train_time", 0)
            ax.scatter(total_time, data["final_f1_mean"], s=120, color=colors[i], zorder=5, edgecolors='black')
            ax.annotate(strat, (total_time, data["final_f1_mean"]), fontsize=8,
                       textcoords="offset points", xytext=(5, 5))
        ax.set_xlabel("Total Computation Time (s)")
        ax.set_ylabel("Final Macro-F1")
        ax.set_title(f"Cost-Performance Pareto - {CFG.dataset} ({CFG.budget_level})")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "cost_performance_pareto.png"), dpi=150)
        plt.close()

    # --- 5. Strategy ranking heatmap ---
    if len(strategies) >= 3:
        fig, ax = plt.subplots(figsize=(max(8, len(strategies)), 4))
        n_rounds = len(aggregated[strategies[0]]["f1_mean"])
        rank_matrix = np.zeros((len(strategies), n_rounds))
        for rd in range(n_rounds):
            round_f1s = [(s, aggregated[s]["f1_mean"][rd]) for s in strategies]
            round_f1s.sort(key=lambda x: x[1], reverse=True)
            for rank, (s, _) in enumerate(round_f1s):
                rank_matrix[strategies.index(s), rd] = rank + 1
        im = ax.imshow(rank_matrix, cmap='RdYlGn_r', aspect='auto', vmin=1, vmax=len(strategies))
        ax.set_xticks(range(n_rounds))
        ax.set_xticklabels([f"R{r+1}" for r in range(n_rounds)], fontsize=7)
        ax.set_yticks(range(len(strategies)))
        ax.set_yticklabels(strategies, fontsize=8)
        for si in range(len(strategies)):
            for ri in range(n_rounds):
                ax.text(ri, si, f"{int(rank_matrix[si, ri])}", ha="center", va="center", fontsize=7)
        plt.colorbar(im, label="Rank")
        ax.set_xlabel("Round")
        ax.set_title(f"Strategy Ranking per Round - {CFG.dataset} ({CFG.budget_level})")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "strategy_ranking_heatmap.png"), dpi=150)
        plt.close()


def main():
    args = parse_args()
    apply_runtime_args(args)

    # --- Structured logging setup ---
    os.makedirs(CFG.output_dir, exist_ok=True)
    log_handlers = [
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(CFG.output_dir, "experiment.log"), encoding="utf-8"),
    ]
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=log_handlers,
    )

    logger.info(f"=== v8 Controlled-Fast AL+SSL Analysis: {CFG.dataset} ===")
    logger.info(f"Budget: {CFG.budget_level}, Model: {CFG.model_type}, Scheduler: {CFG.use_scheduler}")
    logger.info(f"Strategies: {CFG.strategies}")
    logger.info(f"Seeds: {CFG.seeds}")
    logger.info(f"Rounds: {CFG.n_rounds}, Initial: {CFG.n_initial}, Query: {CFG.n_query}")
    logger.info(f"SSL: {CFG.use_ssl}, Method: {resolve_ssl_method(CFG.dataset, CFG.ssl_method)}, t-SNE: {CFG.enable_tsne}, Resume: {CFG.resume}")
    if CFG.imbalance_ratio > 0:
        logger.info(f"Imbalance mode: max/min class count ratio = {CFG.imbalance_ratio:g}")
    if CFG.cold_start:
        logger.info(f"Cold-start mode: n_initial={CFG.n_initial}, n_query={CFG.n_query}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    if CFG.dataset == "fashion_mnist":
        train_set, test_set, infer_set, raw_set = load_fashion_mnist()
    elif CFG.dataset == "cifar10":
        train_set, test_set, infer_set, raw_set = load_cifar10()
    elif CFG.dataset == "cifar100":
        train_set, test_set, infer_set, raw_set = load_cifar100()
    elif CFG.dataset == "agnews":
        train_set, test_set, infer_set, raw_set = load_agnews()
    elif CFG.dataset == "bloodmnist":
        train_set, test_set, infer_set, raw_set = load_bloodmnist()
    elif CFG.dataset == "adult":
        train_set, test_set, infer_set, raw_set = load_adult()
        CFG.adult_input_dim = train_set.features.shape[1]
    elif CFG.dataset == "forda":
        train_set, test_set, infer_set, raw_set = load_ucr_dataset("forda")
    elif CFG.dataset == "ecg5000":
        train_set, test_set, infer_set, raw_set = load_ucr_dataset("ecg5000")
    elif CFG.dataset == "spoken_arabic":
        train_set, test_set, infer_set, raw_set = load_ucr_dataset("spoken_arabic")
    elif CFG.dataset == "character_traj":
        train_set, test_set, infer_set, raw_set = load_ucr_dataset("character_traj")

    # --- Direction 3: Pretrained feature extraction ---
    if CFG.pretrained_features:
        logger.info(f"Extracting pretrained features: {CFG.pretrained_features}")
        pf_model = CFG.pretrained_features
        train_indices = list(range(len(train_set)))
        test_indices = list(range(len(test_set)))

        train_features, train_labels = extract_pretrained_features(
            train_set, train_indices, pf_model, CFG.dataset, device)
        test_features, test_labels = extract_pretrained_features(
            test_set, test_indices, pf_model, CFG.dataset, device)

        # Infer feature dimension and store for model creation
        CFG.pretrained_input_dim = train_features.shape[1]
        n_classes = int(train_labels.max()) + 1
        logger.info(f"  Feature dim: {CFG.pretrained_input_dim}, Classes: {n_classes}")

        # Replace datasets with feature datasets
        train_set = PrecomputedFeatureDataset(train_features, train_labels)
        test_set = PrecomputedFeatureDataset(test_features, test_labels)
        infer_set = train_set  # use train features for consistent AL query (no augmentation needed for precomputed)
        raw_set = None
        logger.info(f"  Replaced datasets: train={len(train_set)}, test={len(test_set)}, infer={len(infer_set)}")

    train_idx_all = list(range(len(train_set)))
    test_idx = list(range(len(test_set)))

    # --- Class imbalance simulation (Standard Long-tail Exponential Decay) ---
    if CFG.imbalance_ratio > 0:
        # Gather all targets for stratified sampling
        all_train_targets = []
        for i in train_idx_all:
            _, label = train_set[i]
            if isinstance(label, torch.Tensor):
                label = label.item()
            all_train_targets.append(int(label))
            
        filtered_idx = make_longtail_indices(
            all_train_targets, 
            imbalance_ratio=CFG.imbalance_ratio, 
            distribution="exp",
            seed=42
        )
        # filtered_idx returns indices relative to the all_train_targets list.
        # Since all_train_targets matches train_idx_all 1:1 in order, we map back to global indices.
        train_idx_all = [train_idx_all[i] for i in filtered_idx]
        logger.info(f"Imbalance filter (Exponential): {len(all_train_targets)} -> {len(train_idx_all)} training samples")

    all_results = []
    failed_runs = []
    for strategy in CFG.strategies:
        logger.info(f"\n--- Strategy: {strategy} ---")
        for seed in CFG.seeds:
            try:
                result = run_single_strategy(
                    strategy, train_set, test_set, infer_set, device,
                    train_idx_all, test_idx, seed, raw_dataset=raw_set)
                all_results.append(result)
            except Exception as e:
                logger.error(f"  ERROR [{strategy}] seed={seed}: {e}")
                failed_runs.append({"strategy": strategy, "seed": seed, "error": str(e)})
                traceback.print_exc()

    if not all_results:
        status_path = os.path.join(CFG.output_dir, "run_status.json")
        with open(status_path, "w", encoding="utf-8") as f:
            json.dump({"completed": False, "failed_runs": failed_runs}, f, indent=2)
        raise RuntimeError("No strategy runs completed; see experiment.log and run_status.json")

    # Full Supervised Baseline
    skip_full_sup = getattr(CFG, 'skip_full_sup', False) or args.skip_full_sup
    existing_full_sup = None

    # 从当前目录或同级目录查找已有的 Full Supervised 结果
    search_dirs = [CFG.output_dir]
    # 如果是 progressive_ssl_full 的子目录，也查找 no_ssl 目录
    parent_dir = os.path.dirname(CFG.output_dir)
    if "progressive_ssl_full" in parent_dir:
        rho_suffix = os.path.basename(CFG.output_dir).split("_rho")[-1] if "_rho" in os.path.basename(CFG.output_dir) else ""
        if rho_suffix:
            search_dirs.append(os.path.join(parent_dir, f"no_ssl_rho{rho_suffix}"))

    for search_dir in search_dirs:
        agg_path = os.path.join(search_dir, "aggregated_results.json")
        if os.path.exists(agg_path):
            try:
                with open(agg_path) as ef:
                    agg_data = json.load(ef)
                if "full_supervision" in agg_data and agg_data["full_supervision"].get("f1", 0) > 0:
                    existing_full_sup = agg_data["full_supervision"]
                    break
            except: pass

    if skip_full_sup and existing_full_sup:
        full_sup = existing_full_sup
        logger.info(f"\n--- Full Supervised Baseline (reused from existing) ---")
        logger.info(f"Full Supervised: Acc={full_sup['accuracy']:.4f}, F1={full_sup['f1']:.4f}")
    elif skip_full_sup:
        # 不计算 Full Supervised，但不保存零值
        full_sup = None
        logger.info("\n--- Full Supervised Baseline (skipped) ---")
    else:
        logger.info("\n--- Full Supervised Baseline ---")
        full_sup = run_full_supervised(train_set, test_set, test_idx, device, train_idx_all=train_idx_all)
        logger.info(f"Full Supervised: Acc={full_sup['accuracy']:.4f}, F1={full_sup['f1']:.4f} "
                    f"(epochs={full_sup['epochs']}, time={full_sup['train_time']:.1f}s)")

    aggregated = aggregate_results(all_results)
    if full_sup is not None:
        aggregated["full_supervision"] = full_sup

    stat_tests = compute_statistical_tests(all_results)

    with open(os.path.join(CFG.output_dir, "aggregated_results.json"), "w") as f:
        json.dump(aggregated, f, indent=2)
    with open(os.path.join(CFG.output_dir, "raw_results.json"), "w") as f:
        json.dump(all_results, f, indent=2)
    with open(os.path.join(CFG.output_dir, "statistical_tests.json"), "w") as f:
        json.dump(stat_tests, f, indent=2)

    config_save = {
        "dataset": CFG.dataset, "budget_level": CFG.budget_level, "model_type": CFG.model_type,
        "n_initial": CFG.n_initial, "n_query": CFG.n_query, "n_rounds": CFG.n_rounds,
        "seeds": CFG.seeds, "strategies": CFG.strategies,
        "use_ssl": CFG.use_ssl, "ssl_method": resolve_ssl_method(CFG.dataset, CFG.ssl_method),
        "ssl_method_requested": CFG.ssl_method,
        "full_supervised_epochs": CFG.full_supervised_epochs,
        "early_stopping_patience": CFG.early_stopping_patience,
        "use_scheduler": CFG.use_scheduler, "imbalance_ratio": CFG.imbalance_ratio,
        "cold_start": CFG.cold_start, "ablation": CFG.ablation,
        "bald_mc_samples": CFG.bald_mc_samples,
        "fast_4060_preset": CFG.fast_4060_preset,
        "qbc_reuse_committee": CFG.qbc_reuse_committee,
        "qbc_incremental_epochs": CFG.qbc_incremental_epochs,
        "max_pool_subsample": CFG.max_pool_subsample,
        "full_supervised_train_size": len(train_idx_all),
        "ssl_lambda_u_warmup_epochs": CFG.ssl_lambda_u_warmup_epochs,
        "ts_jitter_std": CFG.ts_jitter_std,
        "ts_scaling_std": CFG.ts_scaling_std,
        "ts_time_warp_sigma": CFG.ts_time_warp_sigma,
        "ts_use_time_warp": CFG.ts_use_time_warp,
        "dropout_consistency_passes": CFG.dropout_consistency_passes,
        "vat_epsilon": CFG.vat_epsilon,
        "use_amp": CFG.use_amp,
        "class_aware_lambda": CFG.class_aware_lambda,
        "class_aware_adaptive": CFG.class_aware_adaptive,
        "class_aware_soft_weighting": CFG.class_aware_soft_weighting,
        "curriculum_warmup_rounds": CFG.curriculum_warmup_rounds,
        "pseudo_refresh_freq": CFG.pseudo_refresh_freq,
        "ssl_warmup_rounds": CFG.ssl_warmup_rounds,
        "ssl_adaptive": CFG.ssl_adaptive,
        "ssl_adaptive_threshold": CFG.ssl_adaptive_threshold,
        "pretrained_features": CFG.pretrained_features,
    }
    with open(os.path.join(CFG.output_dir, "config.json"), "w") as f:
        json.dump(config_save, f, indent=2)

    with open(os.path.join(CFG.output_dir, "run_status.json"), "w", encoding="utf-8") as f:
        json.dump({
            "completed": len(failed_runs) == 0,
            "failed_runs": failed_runs,
            "n_completed_runs": len(all_results),
        }, f, indent=2)

    plot_results(aggregated, CFG.output_dir, all_results=all_results)

    if failed_runs:
        raise RuntimeError(f"{len(failed_runs)} strategy run(s) failed; see run_status.json")

    logger.info(f"\n=== Results saved to {CFG.output_dir} ===")
    logger.info("\nSummary:")
    logger.info(f"{'Strategy':<20} {'Final F1':>12} {chr(177)+'std':>8} {'p<0.05 vs Random':>18}")
    logger.info("-" * 60)
    for strat in sorted(aggregated.keys()):
        if strat == "full_supervision":
            continue
        data = aggregated[strat]
        sig_mark = ""
        for pc in stat_tests.get("paired_comparisons", []):
            if (pc["strategy_a"] == strat and pc["strategy_b"] == "random") or \
               (pc["strategy_b"] == strat and pc["strategy_a"] == "random"):
                if pc["significant_005"]:
                    sig_mark = "*" if pc["mean_diff"] > 0 else "x"
                break
        logger.info(f"{strat:<20} {data['final_f1_mean']:>12.4f} {data['final_f1_std']:>8.4f} {sig_mark:>18}")
    logger.info(f"{'Full Supervised':<20} {full_sup['f1']:>12.4f}")

    if stat_tests.get("paired_comparisons"):
        logger.info("\nStatistical Tests (selected):")
        for pc in stat_tests["paired_comparisons"]:
            if pc["significant_005"] and abs(pc["mean_diff"]) > 0.005:
                sig = "***" if pc["p_value"] < 0.001 else ("**" if pc["p_value"] < 0.01 else "*")
                logger.info(f"  {pc['strategy_a']} vs {pc['strategy_b']}: "
                            f"diff={pc['mean_diff']:+.4f}, p={pc['p_value']:.4f}{sig}, "
                            f"d={pc['cohens_d']:.3f} ({pc['effect_size']})")


if __name__ == "__main__":
    main()

