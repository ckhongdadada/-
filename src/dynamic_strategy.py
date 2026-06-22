"""
动态策略切换 (Dynamic Strategy Switching)
==========================================
实现方向1：基于预算/模型能力的动态策略切换
参考：
- DCoM (Mishal & Weinshall, 2024): Competence-Driven Adaptive AL
- TCM (Doucet et al., 2024): TypiClust -> Margin Transition
- UHerding (Bae et al., ICLR 2025): Uncertainty Coverage for all budgets

核心思想：
- 低预算（早期轮次）：使用多样性/覆盖策略（TypiClust, CoreSet）
- 高预算（后期轮次）：使用不确定性策略（Margin, Entropy）
- 切换点由模型能力分数（Competence Score）或固定轮次决定
"""

from typing import List, Optional, Dict, Any
import numpy as np
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors


def compute_competence_score(
    labeled_f1_history: List[float],
    current_round: int,
    n_rounds: int,
    method: str = "slope",
) -> float:
    """
    计算模型能力分数 S_L ∈ [0, 1]，用于判断当前模型处于哪个预算区间。

    Args:
        labeled_f1_history: 已标注数据上的 F1 历史（或验证集 F1）
        current_round: 当前 AL 轮次 (0-indexed)
        n_rounds: 总轮次
        method: "slope" (近期提升斜率) | "absolute" (绝对 F1) | "budget_ratio" (预算比例)

    Returns:
        competence_score ∈ [0, 1]
    """
    if method == "budget_ratio":
        return current_round / max(n_rounds - 1, 1)

    if method == "absolute":
        if not labeled_f1_history:
            return 0.0
        latest_f1 = labeled_f1_history[-1]
        # 假设 F1 范围约 [0, 1]，映射到 [0, 1]
        return min(1.0, max(0.0, latest_f1))

    if method == "slope":
        if len(labeled_f1_history) < 2:
            return 0.0
        # 计算最近 3 轮的平均斜率
        window = min(3, len(labeled_f1_history))
        recent = labeled_f1_history[-window:]
        slopes = [recent[i+1] - recent[i] for i in range(len(recent) - 1)]
        avg_slope = np.mean(slopes) if slopes else 0.0
        # 将斜率映射到 [0, 1]：斜率大表示模型还在快速学习（低能力），斜率小表示趋于饱和（高能力）
        # 使用 sigmoid 反转：高斜率 -> 低能力，低斜率 -> 高能力
        score = 1.0 / (1.0 + np.exp(5.0 * avg_slope))
        return float(score)

    return current_round / max(n_rounds - 1, 1)


def select_typiclust(
    features: np.ndarray,
    pool_idx: List[int],
    n_query: int,
    rng: np.random.Generator,
    n_clusters: Optional[int] = None,
    k_neighbors: int = 10,
) -> List[int]:
    """
    TypiClust: 选择典型（高密度区域中心）且多样的样本。
    适用于低预算/冷启动阶段。

    注意：features 可以是全局特征 (N, D) 或子池特征 (len(pool_idx), D)。
    如果是子池特征，pool_idx 应该是 range(len(pool_idx)) 或局部索引。

    实现：KMeans 聚类 + 局部密度最高的典型样本
    """
    from sklearn.cluster import KMeans

    if n_clusters is None:
        n_clusters = n_query

    n_select = min(n_query, len(pool_idx))
    if n_select == 0:
        return []

    # 处理 features 维度与 pool_idx 的匹配问题
    if features.shape[0] == len(pool_idx):
        # features 已经是子池特征
        pool_features = features
        local_pool_idx = list(range(len(pool_idx)))
    else:
        # features 是全局特征，需要用 pool_idx 索引
        pool_features = features[pool_idx]
        local_pool_idx = pool_idx

    if pool_features.shape[0] < n_clusters:
        return rng.choice(pool_idx, n_select, replace=False).tolist()

    # 降维
    if pool_features.shape[1] > 50:
        n_components = min(50, pool_features.shape[1], pool_features.shape[0] - 1)
        pca = PCA(n_components=n_components)
        pool_features = pca.fit_transform(pool_features)

    # KMeans 聚类
    kmeans = KMeans(n_clusters=n_clusters, random_state=rng.integers(0, 2**31), n_init=1)
    cluster_labels = kmeans.fit_predict(pool_features)

    # 计算局部密度（k-NN 平均距离的倒数）
    nbrs = NearestNeighbors(n_neighbors=min(k_neighbors, len(pool_idx)), algorithm='auto')
    nbrs.fit(pool_features)
    distances, _ = nbrs.kneighbors(pool_features)
    typicality = 1.0 / (np.mean(distances, axis=1) + 1e-8)

    # 从每个聚类中选择典型性最高的样本
    selected = []
    selected_set = set()
    cluster_to_samples: Dict[int, List[int]] = {}
    for i, label in enumerate(cluster_labels):
        if label not in cluster_to_samples:
            cluster_to_samples[label] = []
        cluster_to_samples[label].append(i)

    # 按聚类大小排序，优先从大聚类中选择
    sorted_clusters = sorted(cluster_to_samples.items(), key=lambda x: len(x[1]), reverse=True)

    for cluster_id, sample_indices in sorted_clusters:
        if len(selected) >= n_select:
            break

        # 在当前聚类中选择典型性最高的未选样本
        cluster_typicality = [(idx, typicality[idx]) for idx in sample_indices]
        cluster_typicality.sort(key=lambda x: x[1], reverse=True)

        for idx, _ in cluster_typicality:
            global_idx = local_pool_idx[idx]
            if global_idx not in selected_set:
                selected.append(global_idx)
                selected_set.add(global_idx)
                break

    # 如果还有剩余配额，随机填充
    while len(selected) < n_select:
        remaining = [idx for idx in local_pool_idx if idx not in selected_set]
        if not remaining:
            break
        selected.append(rng.choice(remaining))
        selected_set.add(selected[-1])

    return selected[:n_select]


def select_uncertainty_coverage(
    probs: np.ndarray,
    features: np.ndarray,
    pool_idx: List[int],
    n_query: int,
    rng: np.random.Generator,
    alpha: float = 0.5,
) -> List[int]:
    """
    Uncertainty Coverage: 结合不确定性和覆盖度。
    参考 UHerding (Bae et al., ICLR 2025)。

    score = alpha * normalized_entropy + (1 - alpha) * coverage_score
    coverage_score 基于特征空间中的覆盖度（k-center-greedy 风格）

    Args:
        probs: (N, C) 预测概率
        features: (N, D) 特征向量
        pool_idx: 池样本索引
        n_query: 查询数量
        rng: 随机数生成器
        alpha: 不确定性权重，0 = 纯覆盖，1 = 纯不确定性
    """
    n_select = min(n_query, len(pool_idx))
    if n_select == 0:
        return []

    pool_probs = probs[pool_idx]
    pool_features = features[pool_idx]

    # 不确定性分数：熵
    pool_probs = np.clip(pool_probs.astype(np.float32, copy=False), 1e-7, 1.0)
    entropy = -np.sum(pool_probs * np.log(pool_probs), axis=1)
    n_classes = pool_probs.shape[1]
    max_entropy = np.log(n_classes)
    entropy_norm = entropy / max_entropy if max_entropy > 0 else entropy

    # 覆盖度分数：基于特征距离的 k-center-greedy
    # 简化为基于到已选样本的最小距离
    features_norm = pool_features / (np.linalg.norm(pool_features, axis=1, keepdims=True) + 1e-10)

    # 综合分数：先按不确定性预选，再在预选中保证覆盖度
    # 阶段1：不确定性粗筛（选 3*n_query 候选）
    n_candidates = min(n_select * 3, len(pool_idx))
    candidate_local = np.argsort(entropy_norm)[-n_candidates:]

    # 阶段2：在候选中用 k-center-greedy 保证覆盖度
    selected_local = []
    if len(candidate_local) > 0:
        first_idx = int(rng.integers(len(candidate_local)))
        selected_local.append(candidate_local[first_idx])

        min_distances = np.full(len(candidate_local), np.inf, dtype=np.float32)
        for i, cand_i in enumerate(candidate_local):
            if i != first_idx:
                diff = features_norm[cand_i] - features_norm[candidate_local[first_idx]]
                min_distances[i] = np.linalg.norm(diff)
        min_distances[first_idx] = -np.inf

        for _ in range(min(n_select - 1, len(candidate_local) - 1)):
            next_idx = int(np.argmax(min_distances))
            selected_local.append(candidate_local[next_idx])
            for i, cand_i in enumerate(candidate_local):
                if min_distances[i] > 0:
                    diff = features_norm[cand_i] - features_norm[candidate_local[next_idx]]
                    new_dist = np.linalg.norm(diff)
                    min_distances[i] = min(min_distances[i], new_dist)
            min_distances[next_idx] = -np.inf

    # selected_local contains indices from candidate_local (which are pool-local indices)
    return [pool_idx[i] for i in selected_local[:n_select]]


def select_dynamic_switch(
    probs: np.ndarray,
    features: np.ndarray,
    pool_idx: List[int],
    n_query: int,
    rng: np.random.Generator,
    current_round: int,
    n_rounds: int,
    labeled_f1_history: Optional[List[float]] = None,
    strategy: str = "typiclust_to_margin",
    switch_point: Optional[float] = None,
    competence_method: str = "budget_ratio",
) -> List[int]:
    """
    动态策略切换：根据当前轮次或模型能力选择不同策略。

    Args:
        probs: (N, C) 预测概率
        features: (N, D) 特征向量
        pool_idx: 池样本索引
        n_query: 查询数量
        rng: 随机数生成器
        current_round: 当前 AL 轮次 (0-indexed)
        n_rounds: 总轮次
        labeled_f1_history: 验证 F1 历史（用于 competence-based 切换）
        strategy: 切换策略名称
            - "typiclust_to_margin": TypiClust -> Margin
            - "typiclust_to_entropy": TypiClust -> Entropy
            - "coreset_to_margin": CoreSet -> Margin
            - "coverage_to_uncertainty": Coverage -> Uncertainty (UHerding 风格)
        switch_point: 手动指定切换点（0-1 的比例或绝对轮次）。
                      None 则使用 competence score 自适应。
        competence_method: "budget_ratio" | "slope" | "absolute"

    Returns:
        selected_indices
    """
    n_select = min(n_query, len(pool_idx))
    if n_select == 0:
        return []

    # 确定当前阶段
    if switch_point is not None:
        # 固定切换点
        if switch_point <= 1.0:
            # 比例形式
            is_low_budget = current_round < switch_point * n_rounds
        else:
            # 绝对轮次
            is_low_budget = current_round < int(switch_point)
    else:
        # 基于 competence score
        competence = compute_competence_score(
            labeled_f1_history or [], current_round, n_rounds, method=competence_method
        )
        # competence 低 -> 低预算阶段（多样性），competence 高 -> 高预算阶段（不确定性）
        is_low_budget = competence < 0.5

    # 处理 probs/features 维度与 pool_idx 的匹配问题
    # probs 和 features 可能是子池的（长度 = len(pool_idx)）或全局的
    if probs is not None and probs.shape[0] == len(pool_idx):
        # probs 是子池的，使用局部索引
        local_probs = probs
        local_features = features
        local_pool_idx = list(range(len(pool_idx)))
    else:
        # probs 是全局的
        local_probs = probs[pool_idx] if probs is not None else None
        local_features = features[pool_idx] if features is not None else None
        local_pool_idx = pool_idx

    # 根据策略选择具体方法
    if strategy in ("typiclust_to_margin", "typiclust_to_entropy"):
        if is_low_budget:
            return select_typiclust(features, pool_idx, n_query, rng)
        else:
            if strategy == "typiclust_to_margin":
                sorted_probs = np.sort(local_probs, axis=1)
                margins = sorted_probs[:, -1] - sorted_probs[:, -2]
                top_k = np.argsort(margins)[:n_select]
                return [pool_idx[i] for i in top_k]
            else:  # typiclust_to_entropy
                pool_probs = np.clip(local_probs.astype(np.float32, copy=False), 1e-7, 1.0)
                entropy = -np.sum(pool_probs * np.log(pool_probs), axis=1)
                top_k = np.argsort(entropy)[-n_select:]
                return [pool_idx[i] for i in top_k]

    elif strategy == "coreset_to_margin":
        if is_low_budget:
            # CoreSet 选择
            from .deep_query_utils import select_coreset
            return select_coreset(features, pool_idx, n_query, rng, labeled_features=None)
        else:
            sorted_probs = np.sort(local_probs, axis=1)
            margins = sorted_probs[:, -1] - sorted_probs[:, -2]
            top_k = np.argsort(margins)[:n_select]
            return [pool_idx[i] for i in top_k]

    elif strategy == "coverage_to_uncertainty":
        # UHerding 风格：平滑插值
        if switch_point is not None and switch_point <= 1.0:
            alpha = min(1.0, current_round / max(switch_point * n_rounds, 1))
        else:
            alpha = compute_competence_score(
                labeled_f1_history or [], current_round, n_rounds, method=competence_method
            )
        return select_uncertainty_coverage(local_probs, local_features, local_pool_idx, n_query, rng, alpha=alpha)

    else:
        raise ValueError(f"Unknown dynamic strategy: {strategy}")


def get_default_switch_point(dataset: str, n_rounds: int) -> float:
    """
    根据数据集特性返回推荐的默认切换点。

    Returns:
        切换轮次（绝对轮次）
    """
    # 基于实验观察：CIFAR-10 约在第 3-4 轮切换，表格数据约在第 2-3 轮
    defaults = {
        "cifar10": max(2, n_rounds // 3),
        "fashion_mnist": max(2, n_rounds // 3),
        "agnews": max(2, n_rounds // 4),
        "adult": max(1, n_rounds // 4),
        "bloodmnist": max(2, n_rounds // 3),
        "forda": max(1, n_rounds // 4),
        "ecg5000": max(1, n_rounds // 4),
        "spoken_arabic": max(2, n_rounds // 3),
        "character_traj": max(2, n_rounds // 3),
    }
    return defaults.get(dataset, max(2, n_rounds // 3))
