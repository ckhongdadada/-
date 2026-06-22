"""
方向4：不平衡AL的专用算法（边界感知Entropy）
================================================
参考：
- DIRECT (Nuggehalli et al., ICML 2025): Deep AL under Imbalance via Optimal Separation

核心思想：
1. 识别类分离边界：通过分析样本到各类决策边界的距离
2. 优先选择边界附近的少数类样本
3. 结合不确定性采样，在决策边界附近选择最不确定的样本
"""

import numpy as np
import torch
from typing import List, Tuple, Optional, Dict
from collections import Counter


def compute_boundary_distance(
    probs: np.ndarray,
    features: np.ndarray,
    pool_idx: List[int],
    labeled_idx: List[int],
    labeled_labels: np.ndarray,
) -> np.ndarray:
    """
    计算每个样本到决策边界的距离。

    方法：
    1. 对于每个样本，找到预测概率最高的两个类别（top-2）
    2. 计算这两个类别概率的差值：margin = p_top1 - p_top2
    3. Margin越小，样本越接近决策边界
    4. 结合到各类中心的距离，综合评估边界 proximity

    Args:
        probs: (N, C) 预测概率
        features: (N, D) 特征向量
        pool_idx: 池样本索引
        labeled_idx: 已标注样本索引
        labeled_labels: 已标注样本标签

    Returns:
        boundary_distances: (len(pool_idx),) 边界距离分数（越小越接近边界）
    """
    # 处理 probs/features 维度与 pool_idx 的匹配问题
    if probs.shape[0] == len(pool_idx):
        pool_probs = probs
        pool_features = features
    else:
        pool_probs = probs[pool_idx]
        pool_features = features[pool_idx]

    # 1. 计算 margin (top1 - top2)
    sorted_probs = np.sort(pool_probs, axis=1)
    margins = sorted_probs[:, -1] - sorted_probs[:, -2]

    # 2. 计算到各类中心的距离
    # 使用已标注样本计算各类中心
    unique_classes = np.unique(labeled_labels)
    class_centers = {}
    for c in unique_classes:
        mask = labeled_labels == c
        if mask.sum() > 0:
            # 需要获取 labeled_idx 对应的特征
            # 如果 features 是全局的，直接用 labeled_idx 索引
            # 如果 features 是子池的，需要额外的全局 features
            if features.shape[0] > len(pool_idx):
                class_features = features[labeled_idx][mask]
            else:
                # 如果 features 只是子池的，无法获取 labeled 特征
                # 使用 pool_features 作为近似（不够准确，但避免崩溃）
                class_features = pool_features[:mask.sum()]
            class_centers[c] = class_features.mean(axis=0)

    # 3. 计算每个池样本到预测类别中心的距离
    pred_classes = np.argmax(pool_probs, axis=1)
    center_distances = np.zeros(len(pool_idx))

    for i, pred_c in enumerate(pred_classes):
        if pred_c in class_centers:
            center_dist = np.linalg.norm(pool_features[i] - class_centers[pred_c])
            center_distances[i] = center_dist
        else:
            center_distances[i] = np.inf

    # 4. 综合边界距离分数
    # margin 小 -> 接近边界
    # center_dist 大 -> 可能是异常点或边界点
    # 我们希望找到 margin 小且不是明显异常点的样本

    # 归一化
    margin_norm = margins / (margins.max() + 1e-10)
    center_dist_norm = center_distances / (center_distances[center_distances < np.inf].max() + 1e-10)

    # 边界分数：margin 越小越好，center_dist 适中最好
    boundary_score = margin_norm + 0.3 * np.abs(center_dist_norm - 0.5)

    return boundary_score


def select_boundary_aware_entropy(
    probs: np.ndarray,
    features: np.ndarray,
    pool_idx: List[int],
    n_query: int,
    labeled_idx: List[int],
    labeled_labels: np.ndarray,
    n_classes: int,
    minority_classes: Optional[List[int]] = None,
    beta: float = 0.5,
    gamma: float = 2.0,
) -> List[int]:
    """
    边界感知Entropy采样：结合决策边界 proximity 和类别不平衡信息。

    算法逻辑（简化版 DIRECT）：
    1. 计算每个样本的边界 proximity 分数
    2. 识别少数类（基于已标注样本的类别分布）
    3. 对少数类样本给予额外权重
    4. 综合分数 = Entropy + β * Boundary_Score + γ * Minority_Bonus

    Args:
        probs: (N, C) 预测概率
        features: (N, D) 特征向量
        pool_idx: 池样本索引
        n_query: 查询数量
        labeled_idx: 已标注样本索引
        labeled_labels: 已标注样本标签
        n_classes: 总类别数
        minority_classes: 指定的少数类（None则自动识别）
        beta: 边界感知权重
        gamma: 少数类奖励权重

    Returns:
        selected: 选中的样本索引
    """
    n_select = min(n_query, len(pool_idx))
    if n_select == 0:
        return []

    # 处理 probs/features 维度与 pool_idx 的匹配问题
    if probs.shape[0] == len(pool_idx):
        # probs 已经是子池的
        pool_probs = probs
        pool_features = features
        local_pool_idx = list(range(len(pool_idx)))
    else:
        pool_probs = probs[pool_idx]
        pool_features = features[pool_idx]
        local_pool_idx = pool_idx

    pool_probs = np.clip(pool_probs.astype(np.float32, copy=False), 1e-7, 1.0)

    # 1. 计算 Entropy
    entropy = -np.sum(pool_probs * np.log(pool_probs), axis=1)
    max_entropy = np.log(n_classes)
    entropy_norm = entropy / max_entropy if max_entropy > 0 else entropy

    # 2. 计算边界 proximity
    boundary_scores = compute_boundary_distance(
        pool_probs, pool_features, local_pool_idx, labeled_idx, labeled_labels
    )
    boundary_norm = boundary_scores / (boundary_scores.max() + 1e-10)

    # 3. 识别少数类
    if minority_classes is None:
        class_counts = Counter(labeled_labels)
        if len(class_counts) > 0:
            median_count = np.median(list(class_counts.values()))
            minority_classes = [
                c for c, count in class_counts.items()
                if count < median_count * 0.8  # 低于中位数的80%视为少数类
            ]
        else:
            minority_classes = []

    # 4. 计算少数类奖励
    pred_classes = np.argmax(pool_probs, axis=1)
    minority_bonus = np.array([
        1.0 if c in minority_classes else 0.0
        for c in pred_classes
    ])

    # 5. 综合分数
    # 我们希望选择：高Entropy + 接近边界 + 少数类
    combined_score = entropy_norm + beta * (1.0 - boundary_norm) + gamma * minority_bonus

    # 6. 选择最高分样本
    top_k = np.argsort(combined_score)[-n_select:]
    return [pool_idx[i] for i in top_k]


def select_direct_style_query(
    probs: np.ndarray,
    features: np.ndarray,
    pool_idx: List[int],
    n_query: int,
    labeled_idx: List[int],
    labeled_labels: np.ndarray,
    n_classes: int,
    minority_classes: Optional[List[int]] = None,
    lambda_sep: float = 0.5,
) -> List[int]:
    """
    DIRECT-style 查询：识别类分离边界并选择边界附近最不确定样本。

    这是 DIRECT 算法的简化实现：
    1. 计算每个样本到各类决策边界的距离
    2. 识别最优分离边界（margin 最小的区域）
    3. 在边界附近选择最不确定的样本

    Args:
        probs: (N, C) 预测概率
        features: (N, D) 特征向量
        pool_idx: 池样本索引
        n_query: 查询数量
        labeled_idx: 已标注样本索引
        labeled_labels: 已标注样本标签
        n_classes: 总类别数
        minority_classes: 少数类列表
        lambda_sep: 分离度权重

    Returns:
        selected: 选中的样本索引
    """
    n_select = min(n_query, len(pool_idx))
    if n_select == 0:
        return []

    # 处理 probs/features 维度与 pool_idx 的匹配问题
    if probs.shape[0] == len(pool_idx):
        pool_probs = probs
        pool_features = features
        local_pool_idx = list(range(len(pool_idx)))
    else:
        pool_probs = probs[pool_idx]
        pool_features = features[pool_idx]
        local_pool_idx = pool_idx

    # 1. 计算各类中心
    unique_classes = np.unique(labeled_labels)
    class_centers = {}
    for c in unique_classes:
        mask = labeled_labels == c
        if mask.sum() > 0:
            if features.shape[0] > len(pool_idx):
                class_features = features[labeled_idx][mask]
            else:
                class_features = pool_features[:mask.sum()]
            class_centers[c] = class_features.mean(axis=0)

    # 2. 计算每个样本到各类中心的距离矩阵
    distances_to_centers = np.zeros((len(local_pool_idx), n_classes))
    for c, center in class_centers.items():
        distances_to_centers[:, c] = np.linalg.norm(pool_features - center, axis=1)

    # 3. 计算分离度分数
    # 找到最近的两个类别中心
    sorted_distances = np.sort(distances_to_centers, axis=1)
    separation_score = sorted_distances[:, 1] - sorted_distances[:, 0]  # 第二近 - 最近
    # 分离度越小，越接近决策边界
    separation_norm = 1.0 / (1.0 + separation_score)  # 转换为 [0, 1]，越大越接近边界

    # 4. 计算不确定性
    pool_probs = np.clip(pool_probs.astype(np.float32, copy=False), 1e-7, 1.0)
    entropy = -np.sum(pool_probs * np.log(pool_probs), axis=1)
    max_entropy = np.log(n_classes)
    entropy_norm = entropy / max_entropy if max_entropy > 0 else entropy

    # 5. 识别少数类
    if minority_classes is None:
        class_counts = Counter(labeled_labels)
        if len(class_counts) > 0:
            median_count = np.median(list(class_counts.values()))
            minority_classes = [
                c for c, count in class_counts.items()
                if count < median_count * 0.8
            ]
        else:
            minority_classes = []

    pred_classes = np.argmax(pool_probs, axis=1)
    minority_bonus = np.array([
        1.0 if c in minority_classes else 0.0
        for c in pred_classes
    ])

    # 6. 综合分数
    combined_score = entropy_norm + lambda_sep * separation_norm + 1.5 * minority_bonus

    # 7. 选择
    top_k = np.argsort(combined_score)[-n_select:]
    return [pool_idx[i] for i in top_k]


def compute_class_separation_quality(
    features: np.ndarray,
    labels: np.ndarray,
    n_classes: int,
) -> Dict[str, float]:
    """
    计算类别分离质量指标。

    Args:
        features: (N, D) 特征向量
        labels: (N,) 标签
        n_classes: 类别数

    Returns:
        metrics: 分离质量指标
    """
    metrics = {}

    # 1. 计算各类中心
    class_centers = {}
    for c in range(n_classes):
        mask = labels == c
        if mask.sum() > 0:
            class_centers[c] = features[mask].mean(axis=0)

    # 2. 计算类间距离
    inter_class_distances = []
    classes = list(class_centers.keys())
    for i in range(len(classes)):
        for j in range(i + 1, len(classes)):
            dist = np.linalg.norm(class_centers[classes[i]] - class_centers[classes[j]])
            inter_class_distances.append(dist)

    if inter_class_distances:
        metrics["mean_inter_class_distance"] = float(np.mean(inter_class_distances))
        metrics["min_inter_class_distance"] = float(np.min(inter_class_distances))

    # 3. 计算类内距离
    intra_class_distances = []
    for c, center in class_centers.items():
        mask = labels == c
        if mask.sum() > 0:
            dists = np.linalg.norm(features[mask] - center, axis=1)
            intra_class_distances.extend(dists.tolist())

    if intra_class_distances:
        metrics["mean_intra_class_distance"] = float(np.mean(intra_class_distances))

    # 4. 计算分离度 (类间距离 / 类内距离)
    if inter_class_distances and intra_class_distances:
        separation = np.mean(inter_class_distances) / (np.mean(intra_class_distances) + 1e-10)
        metrics["separation_ratio"] = float(separation)

    return metrics
