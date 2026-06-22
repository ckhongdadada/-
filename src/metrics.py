"""
公共评估指标工具函数
==================
提取自 v2 实验脚本的评估逻辑
"""

from typing import List, Dict, Any, Optional
from collections import Counter

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def compute_metrics(
    y_true: List[int],
    y_pred: List[int],
    average: str = "macro",
) -> Dict[str, float]:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred, average=average, zero_division=0),
        "precision": precision_score(y_true, y_pred, average=average, zero_division=0),
        "recall": recall_score(y_true, y_pred, average=average, zero_division=0),
    }


def aggregate_seed_results(
    results_list: List[Dict[str, Any]],
) -> Dict[str, Any]:
    if not results_list:
        return {}

    n_seeds = len(results_list)
    aggregated: Dict[str, Any] = {
        "n_seeds": n_seeds,
    }

    scalar_keys = ["final_f1", "best_f1", "total_train_time", "final_accuracy"]
    for key in scalar_keys:
        values = [r.get(key, 0) for r in results_list]
        aggregated[f"{key}_mean"] = float(np.mean(values))
        aggregated[f"{key}_std"] = float(np.std(values))

    list_keys = ["f1_scores", "accuracies", "n_human_labeled", "n_pseudo_labeled", "n_total_training"]
    for key in list_keys:
        arrays = [np.array(r[key]) for r in results_list if key in r and r[key]]
        if arrays:
            stacked = np.vstack(arrays)
            aggregated[f"{key}_mean"] = stacked.mean(axis=0).tolist()
            aggregated[f"{key}_std"] = stacked.std(axis=0).tolist()

    return aggregated


def format_mean_std(mean: float, std: float, decimals: int = 4) -> str:
    return f"{mean:.{decimals}f} ± {std:.{decimals}f}"


def compute_labeling_efficiency(
    results: Dict[str, Any],
    full_supervised_f1: float,
) -> Dict[str, float]:
    efficiency: Dict[str, float] = {}

    if "f1_scores_mean" in results:
        f1_curve = results["f1_scores_mean"]
        n_labeled = results.get("n_total_training_mean", [])

        for i, (f1, n) in enumerate(zip(f1_curve, n_labeled)):
            if f1 >= full_supervised_f1 * 0.95:
                efficiency["labels_for_95pct"] = n
                efficiency["round_for_95pct"] = i + 1
                break

        if f1_curve:
            efficiency["final_efficiency_ratio"] = f1_curve[-1] / full_supervised_f1 if full_supervised_f1 > 0 else 0

    return efficiency


def summarize_pseudo_label_quality(
    quality_list: List[Dict[str, Any]],
) -> Dict[str, Any]:
    if not quality_list:
        return {}

    summary: Dict[str, Any] = {
        "total_rounds": len(quality_list),
        "total_pseudo_added": sum(q.get("n_added", 0) for q in quality_list),
    }

    accuracies = [q.get("pseudo_accuracy") for q in quality_list if q.get("pseudo_accuracy") is not None]
    if accuracies:
        summary["avg_pseudo_accuracy"] = float(np.mean(accuracies))
        summary["std_pseudo_accuracy"] = float(np.std(accuracies))

    confidences = [q.get("avg_confidence") for q in quality_list if q.get("avg_confidence", 0) > 0]
    if confidences:
        summary["avg_confidence"] = float(np.mean(confidences))

    return summary
