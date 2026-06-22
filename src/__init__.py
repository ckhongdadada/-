"""
主动学习期末项目公共模块。

当前主线实验位于 experiments/v8_controlled_fast_al_ssl.py，本包提供这些实验复用的
模型、主动学习查询、半监督伪标签和评估工具。
"""

from .al_utils import (
    aggregate_results_across_seeds,
    compute_pseudo_label_quality,
    select_typiclust,
    subsample_pool,
)
from .deep_query_utils import (
    compute_grad_embeddings,
    compute_ssl_class_adaptive_threshold,
    compute_ssl_class_weights,
    select_badge,
    select_bald,
    select_class_aware_entropy,
    select_class_aware_entropy_ssl,
    select_coreset,
    select_dacs,
    select_gap_aware_entropy,
    select_gap_aware_entropy_ssl,
    select_least_confidence,
    select_margin,
    select_adaptive_gap_entropy,
    select_qbc,
    select_random,
    select_two_stage_entropy_balance,
    select_curriculum_penalty_entropy,
    select_uncertainty,
)
from .metrics import (
    aggregate_seed_results,
    compute_labeling_efficiency,
    compute_metrics,
    format_mean_std,
    summarize_pseudo_label_quality,
)
from .models import BertClassifier, LossPredictionModule, SimpleCNN, TextMLPClassifier, TimeSeriesCNN
from .ssl_utils import (
    PseudoDataset,
    apply_pseudo_labels,
    select_high_confidence_samples,
)

__all__ = [
    "BertClassifier",
    "LossPredictionModule",
    "SimpleCNN",
    "TextMLPClassifier",
    "TimeSeriesCNN",
    "PseudoDataset",
    "aggregate_results_across_seeds",
    "aggregate_seed_results",
    "apply_pseudo_labels",
    "compute_grad_embeddings",
    "compute_labeling_efficiency",
    "compute_metrics",
    "compute_pseudo_label_quality",
    "compute_ssl_class_adaptive_threshold",
    "compute_ssl_class_weights",
    "format_mean_std",
    "select_adaptive_gap_entropy",
    "select_badge",
    "select_bald",
    "select_class_aware_entropy",
    "select_class_aware_entropy_ssl",
    "select_coreset",
    "select_curriculum_penalty_entropy",
    "select_dacs",
    "select_gap_aware_entropy",
    "select_gap_aware_entropy_ssl",
    "select_least_confidence",
    "select_margin",
    "select_qbc",
    "select_random",
    "select_typiclust",
    "select_high_confidence_samples",
    "select_two_stage_entropy_balance",
    "select_uncertainty",
    "subsample_pool",
    "summarize_pseudo_label_quality",
]
