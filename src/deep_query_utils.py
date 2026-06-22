"""
Shared query helpers for deep active learning experiment scripts.
"""

from typing import List, Optional

import numpy as np


def select_random(pool_idx: List[int], n_query: int, rng: np.random.Generator) -> List[int]:
    """Randomly select global indices from the current pool."""
    n_select = min(n_query, len(pool_idx))
    return rng.choice(pool_idx, n_select, replace=False).tolist()


def select_uncertainty(probs: np.ndarray, pool_idx: List[int], n_query: int) -> List[int]:
    """Select samples with the largest predictive entropy."""
    n_select = min(n_query, len(pool_idx))
    probs = np.clip(probs.astype(np.float32, copy=False), 1e-7, 1.0)
    entropy = -np.sum(probs * np.log(probs), axis=1)
    top_k = np.argsort(entropy)[-n_select:]
    return [pool_idx[i] for i in top_k]


def select_class_aware_entropy(
    probs: np.ndarray,
    pool_idx: List[int],
    n_query: int,
    labeled_labels: np.ndarray,
    n_classes: int,
    lam: float = 0.5,
    adaptive_lambda: bool = False,
    soft_weighting: bool = False,
) -> List[int]:
    """Class-aware entropy with optional adaptive lambda and soft weighting.

    score = entropy_norm + effective_lam * penalty_norm

    Default behavior (adaptive_lambda=False, soft_weighting=False):
        Original V2: fixed lambda, hard argmax penalty lookup.
        This matches the implementation used to generate existing experiment results.

    Optional V3 improvements (set both to True):
    1. Adaptive lambda: lam * skewness, where skewness = 1 - min/max of class
       counts. When data is balanced (ρ=1), skewness≈0 → degenerates to pure
       entropy, avoiding noise injection from the penalty term.
    2. Soft probability weighting: replaces hard argmax penalty lookup with
       expected penalty E[penalty] = Σ_c p(x,c) · penalty(c).

    Args:
        probs: (n_pool, n_classes) predicted probabilities.
        pool_idx: global indices of pool samples.
        n_query: number of samples to select.
        labeled_labels: integer label array of currently labeled samples.
        n_classes: total number of classes.
        lam: balance coefficient (max if adaptive, fixed if not).
        adaptive_lambda: if True, scale lam by class distribution skewness.
        soft_weighting: if True, use soft probability weighting instead of hard argmax.
    """
    n_select = min(n_query, len(pool_idx))
    probs = np.clip(probs.astype(np.float32, copy=False), 1e-7, 1.0)

    entropy = -np.sum(probs * np.log(probs), axis=1)
    max_entropy = np.log(n_classes)
    entropy_norm = entropy / max_entropy if max_entropy > 0 else entropy

    class_counts = np.bincount(labeled_labels.astype(int), minlength=n_classes).astype(np.float32)
    penalty = 1.0 / np.log(class_counts + 2.0)
    penalty_max = penalty.max()
    penalty_norm = penalty / penalty_max if penalty_max > 0 else penalty

    # Effective lambda
    if adaptive_lambda:
        freq_nonzero = class_counts[class_counts > 0]
        if len(freq_nonzero) > 0:
            skewness = 1.0 - freq_nonzero.min() / (freq_nonzero.max() + 1e-10)
        else:
            skewness = 0.0
        effective_lam = lam * skewness
    else:
        effective_lam = lam

    # Penalty per sample
    if soft_weighting:
        # Soft: expected penalty under predicted distribution
        sample_penalty = (probs * penalty_norm).sum(axis=1)
    else:
        # Hard: penalty of argmax class
        pred_classes = np.argmax(probs, axis=1)
        sample_penalty = penalty_norm[pred_classes]

    # Normalize
    sp_max = sample_penalty.max()
    sample_penalty_norm = sample_penalty / (sp_max + 1e-10)

    score = entropy_norm + effective_lam * sample_penalty_norm
    top_k = np.argsort(score)[-n_select:]
    return [pool_idx[i] for i in top_k]


def select_gap_aware_entropy(
    probs: np.ndarray,
    pool_idx: List[int],
    n_query: int,
    labeled_labels: np.ndarray,
    n_classes: int,
    lam: float = 0.5,
) -> List[int]:
    """Gap-aware entropy: normalized Entropy(x) + λ · normalized gap_score(x).

    Instead of penalizing by predicted class (unreliable at low budget),
    computes each sample's expected contribution to filling the class
    distribution gap: gap_score(x) = Σ_c p(x,c) · deficit(c),
    where deficit(c) = max(uniform_freq - labeled_freq_c, 0).

    Args:
        probs: (n_pool, n_classes) predicted probabilities for pool samples.
        pool_idx: global indices of pool samples.
        n_query: number of samples to select.
        labeled_labels: integer label array of currently labeled samples.
        n_classes: total number of classes.
        lam: balance coefficient (0 = pure entropy, larger = stronger gap fill).
    """
    n_select = min(n_query, len(pool_idx))
    probs = np.clip(probs.astype(np.float32, copy=False), 1e-7, 1.0)

    entropy = -np.sum(probs * np.log(probs), axis=1)
    max_entropy = np.log(n_classes)
    entropy_norm = entropy / max_entropy if max_entropy > 0 else entropy

    class_counts = np.bincount(labeled_labels.astype(int), minlength=n_classes).astype(np.float32)
    labeled_freq = class_counts / (class_counts.sum() + 1e-10)
    uniform_freq = np.ones(n_classes, dtype=np.float32) / n_classes
    deficit = np.maximum(uniform_freq - labeled_freq, 0.0)

    gap_score = (probs * deficit).sum(axis=1)
    gap_max = gap_score.max()
    gap_norm = gap_score / gap_max if gap_max > 0 else gap_score

    score = entropy_norm + lam * gap_norm
    top_k = np.argsort(score)[-n_select:]
    return [pool_idx[i] for i in top_k]


def select_two_stage_entropy_balance(
    probs: np.ndarray,
    pool_idx: List[int],
    n_query: int,
    labeled_labels: np.ndarray,
    n_classes: int,
    coarse_factor: int = 3,
    soft_weighting: bool = False,
) -> List[int]:
    """Two-stage sampling: Entropy coarse filter + class-balanced greedy selection.

    Stage 1: Select top-K (K = n_query * coarse_factor) candidates by entropy.
    Stage 2: Greedily pick n_query samples from candidates, prioritizing those
             that bring the labeled set's class distribution closest to uniform.

    This fully decouples uncertainty from diversity, avoiding scale competition.

    Args:
        probs: (n_pool, n_classes) predicted probabilities for pool samples.
        pool_idx: global indices of pool samples.
        n_query: number of samples to select.
        labeled_labels: integer label array of currently labeled samples.
        n_classes: total number of classes.
        coarse_factor: multiplier for stage-1 candidate size (default 3).
        soft_weighting: if True, use soft probability update for class counts
                       instead of hard argmax.
    """
    n_select = min(n_query, len(pool_idx))
    probs = np.clip(probs.astype(np.float32, copy=False), 1e-7, 1.0)

    entropy = -np.sum(probs * np.log(probs), axis=1)
    n_candidates = min(n_select * coarse_factor, len(pool_idx))
    candidate_local = np.argsort(entropy)[-n_candidates:]
    candidate_probs = probs[candidate_local]

    class_counts = np.bincount(labeled_labels.astype(int), minlength=n_classes).astype(np.float32)
    total_labeled = class_counts.sum() + 1e-10
    target_freq = np.ones(n_classes, dtype=np.float32) / n_classes

    selected_local: List[int] = []
    current_counts = class_counts.copy()

    for _ in range(n_select):
        best_idx = -1
        best_score = -np.inf

        current_freq = current_counts / (current_counts.sum() + 1e-10)
        deficit = np.maximum(target_freq - current_freq, 0.0)

        for j, cl in enumerate(candidate_local):
            if cl in selected_local:
                continue
            p = candidate_probs[j]
            expected_gain = float((p * deficit).sum())
            if expected_gain > best_score:
                best_score = expected_gain
                best_idx = cl

        if best_idx == -1:
            remaining = [cl for cl in candidate_local if cl not in selected_local]
            if remaining:
                best_idx = remaining[0]
            else:
                break

        selected_local.append(best_idx)
        if soft_weighting:
            # Soft: update counts by expected class probability
            current_counts += probs[best_idx]
        else:
            # Hard: update count of argmax class
            pred_class = int(probs[best_idx].argmax())
            current_counts[pred_class] += 1

    return [pool_idx[i] for i in selected_local]


def select_curriculum_penalty_entropy(
    probs: np.ndarray,
    pool_idx: List[int],
    n_query: int,
    labeled_labels: np.ndarray,
    n_classes: int,
    lam: float = 0.5,
    current_round: int = 0,
    warmup_rounds: int = 5,
    soft_weighting: bool = False,
) -> List[int]:
    """Curriculum Penalty: gradually introduce class penalty over AL rounds.

    score = entropy_norm + lambda_current * penalty_norm
    lambda_current = lam * min(1.0, current_round / warmup_rounds)

    In early rounds (low model quality), lambda is small → pure entropy.
    As model improves, lambda grows → stronger class balance enforcement.
    Consistent with D4's phased decoupling philosophy.

    Args:
        probs: (n_pool, n_classes) predicted probabilities for pool samples.
        pool_idx: global indices of pool samples.
        n_query: number of samples to select.
        labeled_labels: integer label array of currently labeled samples.
        n_classes: total number of classes.
        lam: maximum balance coefficient (reached after warmup).
        current_round: current AL round (0-indexed).
        warmup_rounds: number of rounds before full penalty.
        soft_weighting: if True, use soft probability weighting instead of hard argmax.
    """
    n_select = min(n_query, len(pool_idx))
    probs = np.clip(probs.astype(np.float32, copy=False), 1e-7, 1.0)

    lambda_current = lam * min(1.0, current_round / max(warmup_rounds, 1))

    entropy = -np.sum(probs * np.log(probs), axis=1)
    max_entropy = np.log(n_classes)
    entropy_norm = entropy / max_entropy if max_entropy > 0 else entropy

    class_counts = np.bincount(labeled_labels.astype(int), minlength=n_classes).astype(np.float32)
    penalty = 1.0 / np.log(class_counts + 2.0)
    penalty_max = penalty.max()
    penalty_norm = penalty / penalty_max if penalty_max > 0 else penalty

    if soft_weighting:
        sample_penalty = (probs * penalty_norm).sum(axis=1)
    else:
        pred_classes = probs.argmax(axis=1)
        sample_penalty = penalty_norm[pred_classes]

    sp_max = sample_penalty.max()
    sample_penalty_norm = sample_penalty / (sp_max + 1e-10)

    score = entropy_norm + lambda_current * sample_penalty_norm
    top_k = np.argsort(score)[-n_select:]
    return [pool_idx[i] for i in top_k]


def select_class_aware_entropy_ssl(
    probs: np.ndarray,
    pool_idx: List[int],
    n_query: int,
    labeled_labels: np.ndarray,
    n_classes: int,
    pseudo_labels: Optional[np.ndarray] = None,
    lam: float = 0.5,
    adaptive_lambda: bool = False,
    soft_weighting: bool = False,
) -> List[int]:
    """Class-aware entropy for AL+SSL with optional adaptive lambda and soft weighting.

    score = entropy_norm + effective_lam * penalty_norm

    Key difference from pure AL version:
    - penalty is computed on labeled_labels + pseudo_labels (joint distribution)
    - avoids AL re-selecting classes already covered by SSL pseudo-labels
    - AL focuses on classes that SSL has NOT covered (true tail classes)

    Default behavior (adaptive_lambda=False, soft_weighting=False):
        Original V2: fixed lambda, hard argmax penalty lookup.

    Optional V3 improvements (set both to True):
    1. Adaptive lambda: lam * skewness of joint_counts. When joint distribution
       is balanced (ρ=1), skewness≈0 → degenerates to pure entropy.
    2. Soft probability weighting: replaces hard argmax penalty lookup with
       expected penalty E[penalty] = Σ_c p(x,c) · penalty(c).

    Args:
        probs: (n_pool, n_classes) predicted probabilities for pool samples.
        pool_idx: global indices of pool samples.
        n_query: number of samples to select.
        labeled_labels: integer label array of currently labeled samples.
        n_classes: total number of classes.
        pseudo_labels: integer label array of SSL pseudo-labeled samples (optional).
        lam: balance coefficient (max if adaptive, fixed if not).
        adaptive_lambda: if True, scale lam by joint distribution skewness.
        soft_weighting: if True, use soft probability weighting instead of hard argmax.
    """
    n_select = min(n_query, len(pool_idx))
    probs = np.clip(probs.astype(np.float32, copy=False), 1e-7, 1.0)

    entropy = -np.sum(probs * np.log(probs), axis=1)
    max_entropy = np.log(n_classes)
    entropy_norm = entropy / max_entropy if max_entropy > 0 else entropy

    # Joint distribution: labeled + pseudo-labels (filter out -1 = low confidence)
    if pseudo_labels is not None and len(pseudo_labels) > 0:
        valid_pseudo = pseudo_labels[pseudo_labels >= 0]
        if len(valid_pseudo) > 0:
            joint_labels = np.concatenate([labeled_labels, valid_pseudo])
        else:
            joint_labels = labeled_labels
    else:
        joint_labels = labeled_labels

    joint_counts = np.bincount(joint_labels.astype(int), minlength=n_classes).astype(np.float32)
    penalty = 1.0 / np.log(joint_counts + 2.0)
    penalty_max = penalty.max()
    penalty_norm = penalty / penalty_max if penalty_max > 0 else penalty

    # Effective lambda
    if adaptive_lambda:
        freq_nonzero = joint_counts[joint_counts > 0]
        if len(freq_nonzero) > 0:
            skewness = 1.0 - freq_nonzero.min() / (freq_nonzero.max() + 1e-10)
        else:
            skewness = 0.0
        effective_lam = lam * skewness
    else:
        effective_lam = lam

    # Penalty per sample
    if soft_weighting:
        sample_penalty = (probs * penalty_norm).sum(axis=1)
    else:
        pred_classes = np.argmax(probs, axis=1)
        sample_penalty = penalty_norm[pred_classes]

    # Normalize
    sp_max = sample_penalty.max()
    sample_penalty_norm = sample_penalty / (sp_max + 1e-10)

    score = entropy_norm + effective_lam * sample_penalty_norm
    top_k = np.argsort(score)[-n_select:]
    return [pool_idx[i] for i in top_k]


def select_gap_aware_entropy_ssl(
    probs: np.ndarray,
    pool_idx: List[int],
    n_query: int,
    labeled_labels: np.ndarray,
    n_classes: int,
    pseudo_labels: Optional[np.ndarray] = None,
    lam: float = 0.5,
) -> List[int]:
    """Gap-aware entropy for AL+SSL: considers pseudo-label distribution.

    Key difference from pure AL version:
    - deficit is computed against (labeled + pseudo) distribution
    - if SSL has already filled the gap for a class, AL won't re-select it
    - AL focuses on classes where SSL pseudo-labels are insufficient

    This creates a virtuous cycle:
      SSL generates head-class pseudo-labels → AL deficit shifts to tail classes
      AL selects tail-class samples → model improves on tail → SSL generates better tail pseudo-labels

    Args:
        probs: (n_pool, n_classes) predicted probabilities for pool samples.
        pool_idx: global indices of pool samples.
        n_query: number of samples to select.
        labeled_labels: integer label array of currently labeled samples.
        n_classes: total number of classes.
        pseudo_labels: integer label array of SSL pseudo-labeled samples (optional).
            Samples with label -1 are treated as "not confident" and ignored.
        lam: balance coefficient (0 = pure entropy, larger = stronger gap fill).
    """
    n_select = min(n_query, len(pool_idx))
    probs = np.clip(probs.astype(np.float32, copy=False), 1e-7, 1.0)

    entropy = -np.sum(probs * np.log(probs), axis=1)
    max_entropy = np.log(n_classes)
    entropy_norm = entropy / max_entropy if max_entropy > 0 else entropy

    # Joint distribution: labeled + pseudo-labels (filter out -1 = low confidence)
    if pseudo_labels is not None and len(pseudo_labels) > 0:
        valid_pseudo = pseudo_labels[pseudo_labels >= 0]
        if len(valid_pseudo) > 0:
            joint_labels = np.concatenate([labeled_labels, valid_pseudo])
        else:
            joint_labels = labeled_labels
    else:
        joint_labels = labeled_labels

    joint_counts = np.bincount(joint_labels.astype(int), minlength=n_classes).astype(np.float32)
    joint_freq = joint_counts / (joint_counts.sum() + 1e-10)
    uniform_freq = np.ones(n_classes, dtype=np.float32) / n_classes
    deficit = np.maximum(uniform_freq - joint_freq, 0.0)

    gap_score = (probs * deficit).sum(axis=1)
    gap_max = gap_score.max()
    gap_norm = gap_score / gap_max if gap_max > 0 else gap_score

    score = entropy_norm + lam * gap_norm
    top_k = np.argsort(score)[-n_select:]
    return [pool_idx[i] for i in top_k]


def compute_ssl_class_adaptive_threshold(
    labeled_labels: np.ndarray,
    n_classes: int,
    base_threshold: float = 0.95,
    alpha: float = 0.25,
) -> np.ndarray:
    """Compute class-adaptive confidence threshold for SSL pseudo-labeling.

    Head classes: keep high threshold (0.95) to ensure pseudo-label quality.
    Tail classes: lower threshold (min 0.70) to increase tail pseudo-label count.

    threshold_c = base_threshold - alpha * deficit_norm_c

    Args:
        labeled_labels: integer label array of currently labeled samples.
        n_classes: total number of classes.
        base_threshold: base confidence threshold (default 0.95).
        alpha: max threshold reduction for tail classes (default 0.25).

    Returns:
        thresholds: (n_classes,) per-class confidence thresholds.
    """
    class_counts = np.bincount(labeled_labels.astype(int), minlength=n_classes).astype(np.float32)
    labeled_freq = class_counts / (class_counts.sum() + 1e-10)
    uniform_freq = np.ones(n_classes, dtype=np.float32) / n_classes
    deficit = np.maximum(uniform_freq - labeled_freq, 0.0)
    deficit_norm = deficit / (deficit.max() + 1e-10)
    thresholds = base_threshold - alpha * deficit_norm
    return thresholds


def compute_ssl_class_weights(
    labeled_labels: np.ndarray,
    n_classes: int,
) -> np.ndarray:
    """Compute class weights for SSL consistency loss.

    Tail classes get higher weights (each pseudo-label is more valuable).
    Head classes get lower weights (pseudo-labels are abundant).

    weight_c = 1 / (n_c + 1), normalized so mean weight = 1.0

    Args:
        labeled_labels: integer label array of currently labeled samples.
        n_classes: total number of classes.

    Returns:
        class_weights: (n_classes,) per-class weights for consistency loss.
    """
    class_counts = np.bincount(labeled_labels.astype(int), minlength=n_classes).astype(np.float32)
    class_weights = 1.0 / (class_counts + 1.0)
    class_weights = class_weights / class_weights.mean()  # normalize mean=1
    return class_weights


def select_adaptive_gap_entropy(
    probs: np.ndarray,
    pool_idx: List[int],
    n_query: int,
    labeled_labels: np.ndarray,
    n_classes: int,
    lam_max: float = 1.0,
) -> List[int]:
    """Adaptive gap-aware entropy: λ scales with labeled-set skewness.

    λ = lam_max · skewness, where skewness = 1 - min_freq / max_freq.
    When the labeled set is balanced (skewness→0), λ→0 (pure entropy).
    When highly imbalanced (skewness→1), λ→lam_max (strong gap fill).

    Args:
        probs: (n_pool, n_classes) predicted probabilities.
        pool_idx: global indices of pool samples.
        n_query: number of samples to select.
        labeled_labels: integer label array of currently labeled samples.
        n_classes: total number of classes.
        lam_max: maximum λ when skewness=1.
    """
    n_select = min(n_query, len(pool_idx))
    probs = np.clip(probs.astype(np.float32, copy=False), 1e-7, 1.0)

    entropy = -np.sum(probs * np.log(probs), axis=1)
    max_entropy = np.log(n_classes)
    entropy_norm = entropy / max_entropy if max_entropy > 0 else entropy

    class_counts = np.bincount(labeled_labels.astype(int), minlength=n_classes).astype(np.float32)
    labeled_freq = class_counts / (class_counts.sum() + 1e-10)
    uniform_freq = np.ones(n_classes, dtype=np.float32) / n_classes
    deficit = np.maximum(uniform_freq - labeled_freq, 0.0)

    gap_score = (probs * deficit).sum(axis=1)
    gap_max = gap_score.max()
    gap_norm = gap_score / gap_max if gap_max > 0 else gap_score

    freq_nonzero = labeled_freq[labeled_freq > 0]
    skewness = 1.0 - freq_nonzero.min() / (freq_nonzero.max() + 1e-10) if len(freq_nonzero) > 0 else 0.0
    lam = lam_max * skewness

    score = entropy_norm + lam * gap_norm
    top_k = np.argsort(score)[-n_select:]
    return [pool_idx[i] for i in top_k]


def compute_grad_embeddings(probs: np.ndarray, features: np.ndarray, device=None) -> np.ndarray:
    """Compute BADGE gradient embeddings with predicted labels as pseudo labels."""
    n_samples = probs.shape[0]
    n_classes = probs.shape[1]
    feature_dim = features.shape[1]

    probs = probs.astype(np.float32, copy=False)
    features = features.astype(np.float32, copy=False)
    pred_labels = probs.argmax(axis=1)

    grad_embeds = np.zeros((n_samples, n_classes * feature_dim), dtype=np.float32)
    for c in range(n_classes):
        class_residual = probs[:, c:c + 1].copy()
        class_residual[pred_labels == c] -= 1.0
        grad_embeds[:, c * feature_dim:(c + 1) * feature_dim] = (
            features * class_residual
        )
    return grad_embeds


def _normalize(vectors: np.ndarray) -> np.ndarray:
    return vectors / (np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-10)


def _farthest_first_select(
    vectors: np.ndarray,
    pool_idx: List[int],
    n_query: int,
    rng: np.random.Generator
) -> List[int]:
    """K-center-greedy: maintain min-dist to selected set, pick argmax."""
    n_samples = len(pool_idx)
    n_select = min(n_query, n_samples)
    if n_select == 0:
        return []

    vectors = _normalize(vectors.astype(np.float32, copy=False))

    selected_indices: List[int] = []
    first_idx = int(rng.integers(n_samples))
    selected_indices.append(first_idx)

    min_distances = np.full(n_samples, np.inf, dtype=np.float32)
    chunk_size = 256
    for start in range(0, n_samples, chunk_size):
        end = min(start + chunk_size, n_samples)
        diff = vectors[start:end] - vectors[first_idx]
        min_distances[start:end] = np.linalg.norm(diff, axis=1)
    min_distances[first_idx] = -np.inf

    for _ in range(n_select - 1):
        next_idx = int(np.argmax(min_distances))
        selected_indices.append(next_idx)
        for start in range(0, n_samples, chunk_size):
            end = min(start + chunk_size, n_samples)
            diff = vectors[start:end] - vectors[next_idx]
            new_dists = np.linalg.norm(diff, axis=1)
            min_distances[start:end] = np.minimum(min_distances[start:end], new_dists)
        for si in selected_indices:
            min_distances[si] = -np.inf

    return [pool_idx[i] for i in selected_indices]


def select_badge(
    grad_embeds: np.ndarray,
    pool_idx: List[int],
    n_query: int,
    rng: np.random.Generator
) -> List[int]:
    """Select a diverse batch from gradient embeddings."""
    return _farthest_first_select(grad_embeds, pool_idx, n_query, rng)


def select_coreset(
    features: np.ndarray,
    pool_idx: List[int],
    n_query: int,
    rng: np.random.Generator,
    labeled_features: Optional[np.ndarray] = None,
) -> List[int]:
    """Select samples farthest from the current labeled feature set (k-center-greedy)."""
    n_samples = len(pool_idx)
    n_select = min(n_query, n_samples)
    if n_select == 0:
        return []
    if labeled_features is None or len(labeled_features) == 0:
        return _farthest_first_select(features, pool_idx, n_query, rng)

    pool_features = _normalize(features.astype(np.float32, copy=False))
    labeled_features = _normalize(labeled_features.astype(np.float32, copy=False))

    # Vectorized: compute min distance from each pool sample to labeled set
    min_distances = np.full(n_samples, np.inf, dtype=np.float32)
    labeled_chunk_size = 64
    pool_chunk_size = 256
    for lstart in range(0, len(labeled_features), labeled_chunk_size):
        lend = min(lstart + labeled_chunk_size, len(labeled_features))
        chunk = labeled_features[lstart:lend]
        for pstart in range(0, n_samples, pool_chunk_size):
            pend = min(pstart + pool_chunk_size, n_samples)
            diff = pool_features[pstart:pend, None, :] - chunk[None, :, :]
            dists = np.linalg.norm(diff, axis=2)
            min_distances[pstart:pend] = np.minimum(
                min_distances[pstart:pend], dists.min(axis=1)
            )

    selected_local: List[int] = []
    dist_chunk = 256
    for _ in range(n_select):
        next_idx = int(np.argmax(min_distances))
        selected_local.append(next_idx)
        for start in range(0, n_samples, dist_chunk):
            end = min(start + dist_chunk, n_samples)
            diff = pool_features[start:end] - pool_features[next_idx]
            new_dists = np.linalg.norm(diff, axis=1)
            min_distances[start:end] = np.minimum(min_distances[start:end], new_dists)
        for si in selected_local:
            min_distances[si] = -np.inf

    return [pool_idx[i] for i in selected_local]


def select_least_confidence(probs: np.ndarray, pool_idx: List[int], n_query: int) -> List[int]:
    n_select = min(n_query, len(pool_idx))
    max_probs = probs.max(axis=1)
    top_k = np.argsort(max_probs)[:n_select]
    return [pool_idx[i] for i in top_k]


def select_margin(probs: np.ndarray, pool_idx: List[int], n_query: int) -> List[int]:
    n_select = min(n_query, len(pool_idx))
    if probs.shape[1] < 2:
        # Fallback to least confidence when only one class
        max_probs = probs.max(axis=1)
        top_k = np.argsort(max_probs)[:n_select]
        return [pool_idx[i] for i in top_k]
    sorted_probs = np.sort(probs, axis=1)
    margins = sorted_probs[:, -1] - sorted_probs[:, -2]
    top_k = np.argsort(margins)[:n_select]
    return [pool_idx[i] for i in top_k]


def select_qbc(committee_probs: list, pool_idx: List[int], n_query: int) -> List[int]:
    n_select = min(n_query, len(pool_idx))
    n_committee = len(committee_probs)
    n_classes = committee_probs[0].shape[1]
    n_samples = len(pool_idx)

    vote_entropy = np.zeros(n_samples, dtype=np.float32)
    for c in range(n_classes):
        votes = np.zeros(n_samples, dtype=np.float32)
        for member_probs in committee_probs:
            member_preds = member_probs.argmax(axis=1)
            votes += (member_preds == c).astype(np.float32)
        votes /= n_committee
        mask = votes > 0
        vote_entropy[mask] -= votes[mask] * np.log(votes[mask])

    top_k = np.argsort(vote_entropy)[-n_select:]
    return [pool_idx[i] for i in top_k]


def select_dacs(
    features: np.ndarray,
    pool_idx: List[int],
    n_query: int,
    rng: np.random.Generator,
    labeled_features: Optional[np.ndarray] = None,
    k_neighbors: int = 10,
) -> List[int]:
    """Density-Aware CoreSet Selection (DACS).

    Improves upon CoreSet by prioritizing samples from sparse regions.
    Reference: Kim & Shin, "In Defense of Core-set" (KDD 2022).
    """
    n_samples = len(pool_idx)
    n_select = min(n_query, n_samples)
    if n_select == 0:
        return []

    pool_features = _normalize(features.astype(np.float32, copy=False))

    from sklearn.neighbors import NearestNeighbors
    k = min(k_neighbors, n_samples - 1)
    if k < 1:
        k = 1
    nn = NearestNeighbors(n_neighbors=k + 1, algorithm="auto")
    nn.fit(pool_features)
    distances_knn, _ = nn.kneighbors(pool_features)
    density = distances_knn[:, 1:].mean(axis=1)

    inv_density = 1.0 / (density + 1e-10)
    inv_density = inv_density / (inv_density.max() + 1e-10)

    if labeled_features is not None and len(labeled_features) > 0:
        labeled_features = _normalize(labeled_features.astype(np.float32, copy=False))
        min_distances = np.full(n_samples, np.inf, dtype=np.float32)
        chunk_size = 64
        for start in range(0, len(labeled_features), chunk_size):
            chunk = labeled_features[start:start + chunk_size]  # (C, D)
            dists = np.linalg.norm(
                pool_features[:, None, :] - chunk[None, :, :], axis=2
            )  # (N, C)
            min_distances = np.minimum(min_distances, dists.min(axis=1))

        norm_distances = min_distances / (min_distances.max() + 1e-10)
        scores = norm_distances * inv_density
    else:
        scores = inv_density.copy()

    selected_local: List[int] = []
    for _ in range(n_select):
        temp_scores = scores.copy()
        for si in selected_local:
            temp_scores[si] = -np.inf
        best_idx = int(np.argmax(temp_scores))
        selected_local.append(best_idx)

        new_distances = np.linalg.norm(pool_features - pool_features[best_idx], axis=1)
        median_dist = np.median(new_distances) + 1e-10
        decay = np.exp(-new_distances / median_dist)
        scores = scores * (1 - 0.5 * decay)
        scores[best_idx] = -np.inf

    return [pool_idx[i] for i in selected_local]


def select_bald(
    mc_probs_list: List[np.ndarray],
    pool_idx: List[int],
    n_query: int,
) -> List[int]:
    """Bayesian Active Learning by Disagreement (BALD).

    Uses MC-Dropout probability estimates to compute the mutual information
    between predictions and model parameters.
    BALD score = H[y|x] - E_w[H[y|x,w]]

    Args:
        mc_probs_list: List of probability arrays from multiple forward passes
            with dropout enabled. Each array has shape (n_pool, n_classes).
        pool_idx: Global indices of the pool samples.
        n_query: Number of samples to select.

    Returns:
        List of selected global indices.
    """
    n_select = min(n_query, len(pool_idx))
    mc_probs = np.clip(np.stack(mc_probs_list).astype(np.float32, copy=False), 1e-7, 1.0)

    mean_probs = mc_probs.mean(axis=0)
    predictive_entropy = -np.sum(
        mean_probs * np.log(mean_probs), axis=1
    )

    per_mc_entropy = -np.sum(
        mc_probs * np.log(mc_probs), axis=2
    )
    expected_entropy = per_mc_entropy.mean(axis=0)

    # BALD = mutual information
    bald_scores = predictive_entropy - expected_entropy

    top_k = np.argsort(bald_scores)[-n_select:]
    return [pool_idx[i] for i in top_k]
