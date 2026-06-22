"""
方向3：AL-SSL协同中的Confirmation Bias抑制
=============================================
参考：
- Gilhuber & Hvingelby (2023): How To Overcome Confirmation Bias in SSL By AL
- Werner et al.: The Role of AL in Modern Deep Learning

核心思想：
1. 标签噪声鲁棒性测试：在初始标签中注入可控噪声，观察AL-SSL的表现
2. 噪声感知查询：优先选择模型高置信度但标签可能错误的样本进行人工复核
3. 伪标签质量监控：跟踪伪标签的确认偏误程度
"""

import numpy as np
import torch
from typing import List, Tuple, Optional, Dict
from collections import defaultdict


def inject_label_noise(
    labels: np.ndarray,
    noise_ratio: float = 0.1,
    noise_type: str = "uniform",  # "uniform" | "pairflip" | "asymmetric"
    n_classes: int = 10,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, Dict[int, int]]:
    """
    向标签中注入可控噪声。

    Args:
        labels: 原始标签数组
        noise_ratio: 噪声比例 (0.0 - 1.0)
        noise_type: 噪声类型
            - uniform: 均匀随机翻转
            - pairflip: 相邻类别对翻转 (如 0<->1, 2<->3)
            - asymmetric: 特定类别翻转模式 (如 0->1, 1->2, ...)
        n_classes: 类别数
        rng: 随机数生成器

    Returns:
        noisy_labels: 带噪声的标签
        noise_tracker: 记录被修改的样本索引和原始标签
    """
    if rng is None:
        rng = np.random.default_rng(42)

    noisy_labels = labels.copy()
    n_samples = len(labels)
    n_noisy = int(n_samples * noise_ratio)

    # 随机选择要修改的样本
    noisy_indices = rng.choice(n_samples, n_noisy, replace=False)
    noise_tracker = {}

    for idx in noisy_indices:
        original_label = int(labels[idx])
        noise_tracker[idx] = original_label

        if noise_type == "uniform":
            # 均匀随机选择其他类别
            new_label = rng.integers(0, n_classes)
            while new_label == original_label:
                new_label = rng.integers(0, n_classes)
            noisy_labels[idx] = new_label

        elif noise_type == "pairflip":
            # 相邻类别对翻转
            if original_label % 2 == 0:
                new_label = min(original_label + 1, n_classes - 1)
            else:
                new_label = max(original_label - 1, 0)
            noisy_labels[idx] = new_label

        elif noise_type == "asymmetric":
            # 特定翻转模式: c -> (c+1) % n_classes
            new_label = (original_label + 1) % n_classes
            noisy_labels[idx] = new_label

    return noisy_labels, noise_tracker


def compute_confirmation_bias_metrics(
    model_probs: np.ndarray,
    pseudo_labels: np.ndarray,
    true_labels: Optional[np.ndarray] = None,
    labeled_mask: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """
    计算Confirmation Bias相关指标。

    Args:
        model_probs: (N, C) 模型预测概率
        pseudo_labels: (N,) 伪标签
        true_labels: (N,) 真实标签（如果有）
        labeled_mask: (N,) 是否已被人工标注

    Returns:
        metrics: 包含以下指标的字典
            - pseudo_label_accuracy: 伪标签准确率（需要true_labels）
            - confidence_calibration_error: 置信度校准误差
            - high_confidence_error_rate: 高置信度样本的错误率
            - confirmation_bias_score: 综合确认偏误分数
    """
    metrics = {}

    # 1. 伪标签准确率
    if true_labels is not None:
        pseudo_acc = np.mean(pseudo_labels == true_labels)
        metrics["pseudo_label_accuracy"] = float(pseudo_acc)

    # 2. 置信度校准误差 (ECE简化版)
    confidences = np.max(model_probs, axis=1)
    if true_labels is not None:
        accuracies = (pseudo_labels == true_labels).astype(float)
        # 分桶计算ECE
        n_bins = 10
        bin_edges = np.linspace(0, 1, n_bins + 1)
        ece = 0.0
        for i in range(n_bins):
            mask = (confidences >= bin_edges[i]) & (confidences < bin_edges[i + 1])
            if i == n_bins - 1:  # 最后一个桶包含右边界
                mask = (confidences >= bin_edges[i]) & (confidences <= bin_edges[i + 1])
            if mask.sum() > 0:
                avg_confidence = confidences[mask].mean()
                avg_accuracy = accuracies[mask].mean()
                ece += mask.sum() * abs(avg_confidence - avg_accuracy)
        ece /= len(confidences)
        metrics["confidence_calibration_error"] = float(ece)

    # 3. 高置信度样本错误率
    high_conf_mask = confidences > 0.9
    if high_conf_mask.sum() > 0 and true_labels is not None:
        high_conf_errors = (pseudo_labels[high_conf_mask] != true_labels[high_conf_mask]).mean()
        metrics["high_confidence_error_rate"] = float(high_conf_errors)
    else:
        metrics["high_confidence_error_rate"] = 0.0

    # 4. 确认偏误综合分数
    # 高置信度但错误的样本比例（确认偏误的指示器）
    if true_labels is not None:
        wrong_mask = pseudo_labels != true_labels
        if wrong_mask.sum() > 0:
            cb_score = confidences[wrong_mask].mean()
            metrics["confirmation_bias_score"] = float(cb_score)
        else:
            metrics["confirmation_bias_score"] = 0.0

    return metrics


def select_noise_aware_query(
    model_probs: np.ndarray,
    pool_idx: List[int],
    n_query: int,
    labeled_idx: List[int],
    labeled_labels: np.ndarray,
    model: torch.nn.Module,
    dataset: torch.utils.data.Dataset,
    device: torch.device,
    noise_threshold: float = 0.3,
    alpha: float = 0.5,
) -> List[int]:
    """
    噪声感知查询策略：优先选择可能标签错误的样本进行人工复核。

    策略逻辑：
    1. 计算每个样本的"标签可疑度"：模型预测与当前标签的不一致程度
    2. 结合不确定性，选择高可疑度 + 高不确定性的样本
    3. 这有助于主动发现标签噪声，抑制确认偏误

    Args:
        model_probs: (N, C) 预测概率
        pool_idx: 池样本索引
        n_query: 查询数量
        labeled_idx: 已标注样本索引
        labeled_labels: 已标注样本的标签
        model: 模型
        dataset: 数据集
        device: 设备
        noise_threshold: 噪声检测阈值
        alpha: 不确定性与可疑度的平衡系数

    Returns:
        selected: 选中的样本索引
    """
    n_select = min(n_query, len(pool_idx))
    if n_select == 0:
        return []

    # 处理 model_probs 维度与 pool_idx 的匹配问题
    if model_probs.shape[0] == len(pool_idx):
        # model_probs 已经是子池的
        pool_probs = model_probs
    else:
        pool_probs = model_probs[pool_idx]
    pool_probs = np.clip(pool_probs.astype(np.float32, copy=False), 1e-7, 1.0)

    # 1. 计算不确定性（熵）
    entropy = -np.sum(pool_probs * np.log(pool_probs), axis=1)
    n_classes = pool_probs.shape[1]
    max_entropy = np.log(n_classes)
    entropy_norm = entropy / max_entropy if max_entropy > 0 else entropy

    # 2. 计算标签可疑度
    # 对于已标注样本，检查模型预测是否与标签一致
    # 对于未标注样本，使用模型置信度作为可疑度指标
    label_suspiciousness = np.zeros(len(pool_idx))

    for i, idx in enumerate(pool_idx):
        pred_class = np.argmax(pool_probs[i])
        confidence = pool_probs[i][pred_class]

        if idx in labeled_idx:
            # 已标注样本：模型预测与标签不一致程度
            label_idx = labeled_idx.index(idx)
            true_label = labeled_labels[label_idx]
            if pred_class != true_label:
                # 模型预测与标签不一致，可疑度 = 模型对预测类别的置信度
                label_suspiciousness[i] = confidence
            else:
                label_suspiciousness[i] = 0.0
        else:
            # 未标注样本：使用预测置信度的倒数作为可疑度
            # 低置信度 = 高可疑度（模型不确定）
            label_suspiciousness[i] = 1.0 - confidence

    # 3. 综合分数
    combined_score = alpha * entropy_norm + (1 - alpha) * label_suspiciousness

    # 4. 选择最高分的样本
    top_k = np.argsort(combined_score)[-n_select:]
    return [pool_idx[i] for i in top_k]


def compute_pseudo_label_quality_evolution(
    round_history: List[Dict[str, float]],
) -> Dict[str, List[float]]:
    """
    分析伪标签质量的演化趋势。

    Args:
        round_history: 每轮的指标历史

    Returns:
        evolution: 各指标的演化序列
    """
    evolution = defaultdict(list)

    for metrics in round_history:
        for key, value in metrics.items():
            evolution[key].append(value)

    return dict(evolution)


def detect_confirmation_bias_spike(
    recent_metrics: List[Dict[str, float]],
    window_size: int = 3,
    threshold: float = 0.1,
) -> bool:
    """
    检测确认偏误是否出现激增。

    Args:
        recent_metrics: 最近几轮的指标
        window_size: 滑动窗口大小
        threshold: 激增阈值

    Returns:
        is_spike: 是否检测到激增
    """
    if len(recent_metrics) < window_size + 1:
        return False

    # 计算最近窗口的平均确认偏误分数
    recent_cb_scores = [
        m.get("confirmation_bias_score", 0.0)
        for m in recent_metrics[-window_size:]
    ]
    previous_cb_scores = [
        m.get("confirmation_bias_score", 0.0)
        for m in recent_metrics[-(window_size + 1):-1]
    ]

    recent_avg = np.mean(recent_cb_scores)
    previous_avg = np.mean(previous_cb_scores)

    # 如果确认偏误分数显著上升，则检测到激增
    return recent_avg > previous_avg + threshold
