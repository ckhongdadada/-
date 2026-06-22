"""
公共主动学习工具函数
==================
提取自 v2 实验脚本的 AL 逻辑
"""

from typing import List, Dict, Any, Optional, Tuple, Union
from collections import Counter

import numpy as np
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors


def subsample_pool(
    pool_idx: List[int],
    max_size: int,
    rng: np.random.Generator,
) -> List[int]:
    if len(pool_idx) <= max_size:
        return pool_idx
    return rng.choice(pool_idx, max_size, replace=False).tolist()


def select_typiclust(
    features: np.ndarray,
    pool_idx: List[int],
    n_query: int,
    rng: np.random.Generator,
    n_clusters: Optional[int] = None,
    k_neighbors: int = 10,
) -> List[int]:
    if n_clusters is None:
        n_clusters = n_query

    n_select = min(n_query, len(pool_idx))
    if n_select == 0:
        return []

    pool_features = features[pool_idx]

    if pool_features.shape[0] < n_clusters:
        return rng.choice(pool_idx, n_select, replace=False).tolist()

    if pool_features.shape[1] > 50:
        n_components = min(50, pool_features.shape[1], pool_features.shape[0] - 1)
        pca = PCA(n_components=n_components)
        pool_features = pca.fit_transform(pool_features)

    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=n_clusters, random_state=rng.integers(0, 2**31), n_init=1)
    cluster_labels = kmeans.fit_predict(pool_features)

    nbrs = NearestNeighbors(n_neighbors=min(k_neighbors, len(pool_idx)), algorithm='auto').fit(pool_features)
    distances, _ = nbrs.kneighbors(pool_features)
    typicality = 1.0 / (np.mean(distances, axis=1) + 1e-8)

    selected = []
    selected_set = set()
    cluster_to_samples: Dict[int, List[int]] = {}
    for i, label in enumerate(cluster_labels):
        if label not in cluster_to_samples:
            cluster_to_samples[label] = []
        cluster_to_samples[label].append(i)

    sorted_clusters = sorted(cluster_to_samples.items(), key=lambda x: len(x[1]), reverse=True)

    for cluster_id, sample_indices in sorted_clusters:
        if len(selected) >= n_select:
            break

        cluster_typicality = [(idx, typicality[idx]) for idx in sample_indices]
        cluster_typicality.sort(key=lambda x: x[1], reverse=True)

        for idx, _ in cluster_typicality:
            global_idx = pool_idx[idx]
            if global_idx not in selected_set:
                selected.append(global_idx)
                selected_set.add(global_idx)
                break

    while len(selected) < n_select:
        remaining = [idx for idx in pool_idx if idx not in selected_set]
        if not remaining:
            break
        selected.append(rng.choice(remaining))
        selected_set.add(selected[-1])

    return selected[:n_select]


def compute_pseudo_label_quality(
    pseudo_labels: Dict[int, int],
    true_labels: Optional[Union[List[int], Dict[int, int]]],
    confidences: Optional[np.ndarray] = None,
    indices: Optional[List[int]] = None,
) -> Dict[str, Any]:
    quality: Dict[str, Any] = {
        "n_pseudo": len(pseudo_labels),
        "avg_confidence": 0.0,
        "accuracy": None,
        "class_distribution": {},
    }

    if len(pseudo_labels) == 0:
        return quality

    if confidences is not None and indices is not None:
        quality["avg_confidence"] = float(np.mean(confidences))

    if true_labels is not None:
        correct = 0
        for idx, pred_label in pseudo_labels.items():
            if true_labels[idx] == pred_label:
                correct += 1
        quality["accuracy"] = correct / len(pseudo_labels)

    label_counts = Counter(pseudo_labels.values())
    quality["class_distribution"] = dict(label_counts)

    return quality


def aggregate_results_across_seeds(
    all_results: List[Dict[str, Any]],
) -> Dict[str, Any]:
    if not all_results:
        return {}

    aggregated: Dict[str, Any] = {
        "strategy": all_results[0].get("strategy", "unknown"),
        "ssl_mode": all_results[0].get("ssl_mode", "none"),
        "ssl_threshold": all_results[0].get("ssl_threshold", 0.95),
        "n_seeds": len(all_results),
    }

    metric_keys = ["accuracies", "f1_scores", "n_human_labeled", "n_pseudo_labeled", "n_total_training"]
    for key in metric_keys:
        values = [r[key] for r in all_results if key in r]
        if values:
            arr = np.array(values)
            aggregated[f"{key}_mean"] = arr.mean(axis=0).tolist()
            aggregated[f"{key}_std"] = arr.std(axis=0).tolist()

    final_f1s = [r["f1_scores"][-1] if r.get("f1_scores") else 0 for r in all_results]
    best_f1s = [r.get("best_f1", 0) for r in all_results]
    total_times = [r.get("total_train_time", 0) for r in all_results]

    aggregated["final_f1_mean"] = float(np.mean(final_f1s))
    aggregated["final_f1_std"] = float(np.std(final_f1s))
    aggregated["best_f1_mean"] = float(np.mean(best_f1s))
    aggregated["best_f1_std"] = float(np.std(best_f1s))
    aggregated["total_train_time_mean"] = float(np.mean(total_times))
    aggregated["total_train_time_std"] = float(np.std(total_times))

    return aggregated
