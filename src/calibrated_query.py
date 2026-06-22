"""
校准不确定性采样 (Calibrated Uncertainty Sampling)
====================================================
实现方向2：Temperature Scaling + Calibrated Entropy/Margin
参考：Bui et al. (2025) "Calibrated Uncertainty Sampling for Active Learning"
"""

from typing import List, Optional, Tuple
import numpy as np
import torch
import torch.nn.functional as F


def temperature_scaling_fit(
    logits: torch.Tensor,
    labels: torch.Tensor,
    max_iter: int = 50,
    lr: float = 0.01,
) -> float:
    """
    在验证集上拟合温度参数 T，使得 softmax(logits / T) 的 ECE 最小。

    Args:
        logits: (N, C) 模型输出 logits
        labels: (N,) 真实标签
        max_iter: 优化迭代次数
        lr: 学习率

    Returns:
        最优温度 T
    """
    temperature = torch.ones(1, device=logits.device, requires_grad=True)
    optimizer = torch.optim.LBFGS([temperature], lr=lr, max_iter=max_iter)

    def eval_loss():
        optimizer.zero_grad()
        scaled_logits = logits / temperature
        loss = F.cross_entropy(scaled_logits, labels)
        loss.backward()
        return loss

    optimizer.step(eval_loss)
    return temperature.item()


def apply_temperature_scaling(
    logits: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    """应用温度缩放。"""
    return logits / max(temperature, 1e-6)


def compute_ece(
    probs: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 15,
) -> float:
    """
    计算 Expected Calibration Error (ECE)。

    Args:
        probs: (N, C) 预测概率
        labels: (N,) 真实标签
        n_bins: 置信度分箱数

    Returns:
        ECE 值
    """
    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    accuracies = (predictions == labels).astype(np.float32)

    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
        prop = in_bin.mean()
        if prop > 0:
            avg_confidence = confidences[in_bin].mean()
            avg_accuracy = accuracies[in_bin].mean()
            ece += np.abs(avg_confidence - avg_accuracy) * prop
    return float(ece)


def calibrate_probs_with_labeled_data(
    model: torch.nn.Module,
    dataset: torch.utils.data.Dataset,
    labeled_idx: List[int],
    device: torch.device,
    batch_size: int = 128,
) -> Tuple[np.ndarray, float]:
    """
    使用已标注数据拟合温度参数，并返回校准后的概率和温度值。

    Args:
        model: 当前模型
        dataset: 数据集
        labeled_idx: 已标注样本索引
        device: 计算设备
        batch_size: 推理 batch size

    Returns:
        (calibrated_probs, temperature)
    """
    from torch.utils.data import DataLoader, Subset

    model.eval()
    subset = Subset(dataset, labeled_idx)
    loader = DataLoader(subset, batch_size=batch_size, shuffle=False, num_workers=0)

    all_logits = []
    all_labels = []

    with torch.no_grad():
        for batch_data in loader:
            if hasattr(dataset, 'dataset') and hasattr(dataset.dataset, 'targets'):
                # 处理嵌套 Dataset 的情况
                feats, labels = batch_data
                feats = feats.to(device)
                logits = model(feats)
            else:
                # 通用路径
                if isinstance(batch_data, (list, tuple)) and len(batch_data) >= 2:
                    data, labels = batch_data[0], batch_data[1]
                else:
                    continue
                data = data.to(device)
                logits = model(data)
            all_logits.append(logits.cpu())
            all_labels.append(labels if isinstance(labels, torch.Tensor) else torch.tensor(labels))

    if not all_logits:
        return None, 1.0

    all_logits = torch.cat(all_logits, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    # 拟合温度
    temperature = temperature_scaling_fit(all_logits, all_labels)

    # 返回校准后的概率（在 labeled 数据上）
    calibrated_probs = F.softmax(all_logits / temperature, dim=1).numpy()

    return calibrated_probs, temperature


def get_calibrated_probs_for_pool(
    model: torch.nn.Module,
    dataset: torch.utils.data.Dataset,
    pool_idx: List[int],
    device: torch.device,
    temperature: float,
    batch_size: int = 128,
) -> np.ndarray:
    """
    对未标注池应用温度缩放，返回校准后的概率。

    Args:
        model: 当前模型
        dataset: 数据集
        pool_idx: 池样本索引
        device: 计算设备
        temperature: 已拟合的温度参数
        batch_size: 推理 batch size

    Returns:
        calibrated_probs: (N, C) 校准后的概率
    """
    from torch.utils.data import DataLoader, Subset

    model.eval()
    subset = Subset(dataset, pool_idx)
    loader = DataLoader(subset, batch_size=batch_size, shuffle=False, num_workers=0)

    all_probs = []
    with torch.no_grad():
        for batch_data in loader:
            if isinstance(batch_data, (list, tuple)) and len(batch_data) >= 2:
                data = batch_data[0]
            else:
                data = batch_data
            data = data.to(device)
            logits = model(data)
            probs = F.softmax(logits / max(temperature, 1e-6), dim=1)
            all_probs.append(probs.cpu().numpy())

    if not all_probs:
        return np.array([])
    return np.vstack(all_probs)


def select_calibrated_entropy(
    model: torch.nn.Module,
    dataset: torch.utils.data.Dataset,
    pool_idx: List[int],
    labeled_idx: List[int],
    n_query: int,
    device: torch.device,
    batch_size: int = 128,
) -> Tuple[List[int], float]:
    """
    校准不确定性采样：使用温度缩放后的概率计算熵，选择最高熵样本。

    Returns:
        (selected_indices, temperature)
    """
    # Step 1: 用 labeled 数据拟合温度
    _, temperature = calibrate_probs_with_labeled_data(
        model, dataset, labeled_idx, device, batch_size
    )

    # Step 2: 对 pool 应用校准
    calibrated_probs = get_calibrated_probs_for_pool(
        model, dataset, pool_idx, device, temperature, batch_size
    )

    if len(calibrated_probs) == 0:
        return [], temperature

    # Step 3: 基于校准概率选择最高熵样本
    n_select = min(n_query, len(pool_idx))
    probs = np.clip(calibrated_probs.astype(np.float32, copy=False), 1e-7, 1.0)
    entropy = -np.sum(probs * np.log(probs), axis=1)
    top_k = np.argsort(entropy)[-n_select:]

    return [pool_idx[i] for i in top_k], temperature


def select_calibrated_margin(
    model: torch.nn.Module,
    dataset: torch.utils.data.Dataset,
    pool_idx: List[int],
    labeled_idx: List[int],
    n_query: int,
    device: torch.device,
    batch_size: int = 128,
) -> Tuple[List[int], float]:
    """
    校准 Margin 采样：使用温度缩放后的概率计算 margin，选择最小 margin 样本。

    Returns:
        (selected_indices, temperature)
    """
    # Step 1: 用 labeled 数据拟合温度
    _, temperature = calibrate_probs_with_labeled_data(
        model, dataset, labeled_idx, device, batch_size
    )

    # Step 2: 对 pool 应用校准
    calibrated_probs = get_calibrated_probs_for_pool(
        model, dataset, pool_idx, device, temperature, batch_size
    )

    if len(calibrated_probs) == 0:
        return [], temperature

    # Step 3: 基于校准概率选择最小 margin 样本
    n_select = min(n_query, len(pool_idx))
    sorted_probs = np.sort(calibrated_probs, axis=1)
    margins = sorted_probs[:, -1] - sorted_probs[:, -2]
    top_k = np.argsort(margins)[:n_select]

    return [pool_idx[i] for i in top_k], temperature
