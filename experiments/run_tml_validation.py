#!/usr/bin/env python
"""
TML模型验证实验（LR、RF）
验证AL策略在传统机器学习模型上的通用性（仅图像数据集）
支持纯AL和AL+SSL（self-training）两种模式
数据集: CIFAR-10, FashionMNIST
"""

import os
import sys
import json
import numpy as np
import argparse
from pathlib import Path
from collections import Counter

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ssl_v7_utils import make_longtail_indices
from deep_query_utils import (
    select_class_aware_entropy,
    select_gap_aware_entropy,
    select_adaptive_gap_entropy,
)


def load_cifar10_flatten():
    """加载CIFAR-10并flatten为向量"""
    import torchvision
    import torchvision.transforms as transforms
    transform = transforms.ToTensor()
    train_set = torchvision.datasets.CIFAR10(root=str(PROJECT_ROOT / "data"), train=True, download=True, transform=transform)
    test_set = torchvision.datasets.CIFAR10(root=str(PROJECT_ROOT / "data"), train=False, download=True, transform=transform)
    X_train = train_set.data.reshape(len(train_set), -1).astype(np.float32) / 255.0
    y_train = np.array(train_set.targets)
    X_test = test_set.data.reshape(len(test_set), -1).astype(np.float32) / 255.0
    y_test = np.array(test_set.targets)
    return X_train, y_train, X_test, y_test, 10


def load_fashionmnist_flatten():
    """加载FashionMNIST并flatten为向量"""
    import torchvision
    import torchvision.transforms as transforms
    transform = transforms.ToTensor()
    train_set = torchvision.datasets.FashionMNIST(root=str(PROJECT_ROOT / "data"), train=True, download=True, transform=transform)
    test_set = torchvision.datasets.FashionMNIST(root=str(PROJECT_ROOT / "data"), train=False, download=True, transform=transform)
    X_train = train_set.data.numpy().reshape(len(train_set), -1).astype(np.float32) / 255.0
    y_train = np.array(train_set.targets)
    X_test = test_set.data.numpy().reshape(len(test_set), -1).astype(np.float32) / 255.0
    y_test = np.array(test_set.targets)
    return X_train, y_train, X_test, y_test, 10


def load_cifar100_flatten():
    """加载CIFAR-100并flatten为向量"""
    import torchvision
    import torchvision.transforms as transforms
    transform = transforms.ToTensor()
    train_set = torchvision.datasets.CIFAR100(root=str(PROJECT_ROOT / "data"), train=True, download=True, transform=transform)
    test_set = torchvision.datasets.CIFAR100(root=str(PROJECT_ROOT / "data"), train=False, download=True, transform=transform)
    X_train = train_set.data.reshape(len(train_set), -1).astype(np.float32) / 255.0
    y_train = np.array(train_set.targets)
    X_test = test_set.data.reshape(len(test_set), -1).astype(np.float32) / 255.0
    y_test = np.array(test_set.targets)
    return X_train, y_train, X_test, y_test, 100


DATASET_LOADERS = {
    "cifar10": load_cifar10_flatten,
    "fashionmnist": load_fashionmnist_flatten,
    "cifar100": load_cifar100_flatten,
}


class TMLExperiment:
    """TML模型AL实验（图像数据集）"""

    def __init__(self, model_type, dataset_name, rho, seed=42, use_ssl=False, ssl_threshold=0.9):
        self.model_type = model_type
        self.dataset_name = dataset_name
        self.rho = rho
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        self.n_classes = 10
        self.use_ssl = use_ssl
        self.ssl_threshold = ssl_threshold
        self.load_data()

    def load_data(self):
        loader = DATASET_LOADERS[self.dataset_name]
        X_train, y_train, X_test, y_test, n_classes = loader()
        self.n_classes = n_classes

        # 标准化
        self.scaler = StandardScaler()
        self.X_train = self.scaler.fit_transform(X_train)
        self.X_test = self.scaler.transform(X_test)
        self.y_test = y_test

        # 创建长尾分布
        if self.rho > 1:
            indices = make_longtail_indices(y_train, self.rho, self.seed)
            self.X_train = self.X_train[indices]
            self.y_train = y_train[indices]
        else:
            self.y_train = y_train

        print(f"Dataset: {self.dataset_name}, rho={self.rho}")
        print(f"Train: {len(self.X_train)}, Test: {len(self.X_test)}, Classes: {self.n_classes}")
        print(f"Class distribution: {Counter(self.y_train)}")

    def create_model(self):
        if self.model_type == "lr":
            return LogisticRegression(max_iter=1000, solver='lbfgs',
                                      random_state=self.seed, class_weight='balanced')
        elif self.model_type == "rf":
            return RandomForestClassifier(n_estimators=100, max_depth=10,
                                          random_state=self.seed, class_weight='balanced', n_jobs=-1)
        else:
            raise ValueError(f"Unknown model: {self.model_type}")

    def margin_query(self, probs, pool_idx, n_query):
        """Margin: 选择最大概率与次大概率差值最小的样本"""
        sorted_probs = np.sort(probs, axis=1)
        margin = sorted_probs[:, -1] - sorted_probs[:, -2]
        top_k = np.argsort(margin)[:n_query]
        return [pool_idx[i] for i in top_k]

    def badge_query(self, pool_idx, n_query):
        """BADGE: Batch Active learning by Diverse Gradient Embeddings (simplified for TML)"""
        n_select = min(n_query, len(pool_idx))
        # 使用梯度嵌入的简化版本：基于预测不确定性和特征多样性
        probs = self.model_current.predict_proba(self.X_train[pool_idx])
        # 不确定性权重
        entropy = -np.sum(probs * np.log(probs + 1e-10), axis=1)
        # 随机投影实现多样性
        n_features = min(32, self.X_train.shape[1])
        rng_proj = np.random.RandomState(self.seed)
        proj_matrix = rng_proj.randn(self.X_train.shape[1], n_features)
        proj_matrix /= np.linalg.norm(proj_matrix, axis=1, keepdims=True)
        embeddings = self.X_train[pool_idx] @ proj_matrix
        # 加权嵌入
        weighted_emb = embeddings * entropy[:, None]
        # k-means++ 初始化选择多样化样本
        selected_local = []
        # 第一个选不确定性最高的
        first = np.argmax(entropy)
        selected_local.append(first)
        distances = np.full(len(pool_idx), np.inf)
        for _ in range(n_select - 1):
            last_emb = weighted_emb[selected_local[-1]]
            new_dists = np.linalg.norm(weighted_emb - last_emb, axis=1)
            distances = np.minimum(distances, new_dists)
            distances[selected_local] = -1
            next_idx = np.argmax(distances)
            selected_local.append(next_idx)
        return [pool_idx[i] for i in selected_local]

    def entropy_query(self, probs, pool_idx, n_query):
        entropy = -np.sum(probs * np.log(probs + 1e-10), axis=1)
        top_k = np.argsort(entropy)[-n_query:]
        return [pool_idx[i] for i in top_k]

    def adaptive_gap_entropy_query(self, probs, pool_idx, n_query, labeled_labels):
        """Adaptive Gap-Aware Entropy: delegates to deep_query_utils for consistency."""
        return select_adaptive_gap_entropy(
            probs, pool_idx, n_query, labeled_labels, self.n_classes, lam_max=1.0)

    def class_aware_entropy_query(self, probs, pool_idx, n_query, labeled_labels):
        """Class-Aware Entropy: delegates to deep_query_utils for consistency."""
        return select_class_aware_entropy(
            probs, pool_idx, n_query, labeled_labels, self.n_classes,
            lam=0.5, adaptive_lambda=True, soft_weighting=True)

    def gap_aware_entropy_query(self, probs, pool_idx, n_query, labeled_labels):
        """Gap-Aware Entropy: delegates to deep_query_utils for consistency."""
        return select_gap_aware_entropy(
            probs, pool_idx, n_query, labeled_labels, self.n_classes, lam=0.5)

    def coreset_query(self, labeled_idx, pool_idx, n_query):
        """CoreSet: k-center greedy on PCA-reduced features"""
        n_select = min(n_query, len(pool_idx))

        # PCA降维到50维（或数据维度的较小值）
        n_components = min(50, self.X_train.shape[1])
        pca = PCA(n_components=n_components, random_state=self.seed)
        X_pca = pca.fit_transform(self.X_train)

        labeled_set = set(labeled_idx)
        selected = []

        # 计算每个pool样本到最近labeled样本的距离
        pool_features = X_pca[pool_idx]
        labeled_features = X_pca[labeled_idx] if len(labeled_idx) > 0 else np.zeros((1, n_components))

        for _ in range(n_select):
            # 计算pool中每个样本到最近labeled样本的距离
            dists = np.linalg.norm(
                pool_features[:, None, :] - labeled_features[None, :, :], axis=2
            )
            min_dists = dists.min(axis=1)

            # 选择距离最大的样本（最远点）
            best_local = np.argmax(min_dists)
            best_global = pool_idx[best_local]

            selected.append(best_global)
            labeled_idx.append(best_global)
            labeled_set.add(best_global)

            # 更新labeled特征
            labeled_features = np.vstack([labeled_features, pool_features[best_local:best_local+1, :]])

            # 从pool中移除
            pool_idx = [i for i in pool_idx if i != best_global]
            pool_features = X_pca[pool_idx]

        return selected

    def qbc_query(self, pool_idx, n_query, n_committee=5):
        """QBC: Query-by-Committee with vote entropy"""
        n_select = min(n_query, len(pool_idx))

        # 训练多个不同随机种子的模型组成委员会
        committee_predictions = []
        for i in range(n_committee):
            model = self.create_model()
            # 使用bootstrap采样训练不同模型
            bootstrap_idx = self.rng.choice(
                len(self.labeled_idx_current), size=len(self.labeled_idx_current), replace=True
            )
            X_boot = self.X_train[self.labeled_idx_current][bootstrap_idx]
            y_boot = self.y_train[self.labeled_idx_current][bootstrap_idx]
            model.fit(X_boot, y_boot)
            preds = model.predict(self.X_train[pool_idx])
            committee_predictions.append(preds)

        # 计算投票熵
        vote_matrix = np.array(committee_predictions)  # [n_committee, n_pool]
        vote_entropy = np.zeros(len(pool_idx))
        for c in range(self.n_classes):
            vote_fraction = np.mean(vote_matrix == c, axis=0)
            vote_entropy -= vote_fraction * np.log(vote_fraction + 1e-10)

        top_k = np.argsort(vote_entropy)[-n_select:]
        return [pool_idx[i] for i in top_k]

    def generate_pseudo_labels(self, model, unlabeled_idx):
        """Self-training SSL: 为高置信度未标注样本生成伪标签"""
        if len(unlabeled_idx) == 0:
            return [], []

        probs = model.predict_proba(self.X_train[unlabeled_idx])
        max_probs = np.max(probs, axis=1)
        pred_labels = np.argmax(probs, axis=1)

        # 选择置信度超过阈值的样本
        confident_mask = max_probs >= self.ssl_threshold
        pseudo_idx = np.array(unlabeled_idx)[confident_mask]
        pseudo_labels = pred_labels[confident_mask]

        return pseudo_idx.tolist(), pseudo_labels.tolist()

    def run_al(self, strategy, n_initial=100, n_query=100, n_rounds=10):
        labeled_idx = list(self.rng.choice(len(self.X_train), size=n_initial, replace=False))
        pool_idx = [i for i in range(len(self.X_train)) if i not in labeled_idx]

        results = {"f1_scores": [], "acc_scores": [], "labeled_sizes": [], "pseudo_sizes": []}

        for rd in range(n_rounds):
            # 训练模型（如有SSL，加入伪标签数据）
            if self.use_ssl and rd > 0:
                pseudo_idx, pseudo_labels = self.generate_pseudo_labels(model, pool_idx)
                if len(pseudo_idx) > 0:
                    X_train_ssl = np.vstack([self.X_train[labeled_idx], self.X_train[pseudo_idx]])
                    y_train_ssl = np.concatenate([self.y_train[labeled_idx], pseudo_labels])
                    model_ssl = self.create_model()
                    model_ssl.fit(X_train_ssl, y_train_ssl)
                    results["pseudo_sizes"].append(len(pseudo_idx))
                else:
                    model_ssl = self.create_model()
                    model_ssl.fit(self.X_train[labeled_idx], self.y_train[labeled_idx])
                    results["pseudo_sizes"].append(0)
            else:
                model_ssl = self.create_model()
                model_ssl.fit(self.X_train[labeled_idx], self.y_train[labeled_idx])
                results["pseudo_sizes"].append(0)

            # 评估
            y_pred = model_ssl.predict(self.X_test)
            acc = accuracy_score(self.y_test, y_pred)
            f1 = f1_score(self.y_test, y_pred, average='macro')

            results["f1_scores"].append(f1)
            results["acc_scores"].append(acc)
            results["labeled_sizes"].append(len(labeled_idx))

            ssl_tag = "+SSL" if self.use_ssl else ""
            print(f"  [{strategy}{ssl_tag}] R{rd+1}/{n_rounds} F1={f1:.4f} Acc={acc:.4f} Labeled={len(labeled_idx)} Pseudo={results['pseudo_sizes'][-1]}")

            if rd == n_rounds - 1:
                break

            n_select = min(n_query, len(pool_idx))
            if n_select == 0:
                break

            # 使用当前模型（可能是SSL模型）做查询
            model = model_ssl
            probs = model.predict_proba(self.X_train[pool_idx])

            if strategy == "random":
                selected_local = self.rng.choice(len(pool_idx), size=n_select, replace=False)
                selected = [pool_idx[i] for i in selected_local]
            elif strategy == "entropy":
                selected = self.entropy_query(probs, pool_idx, n_select)
            elif strategy == "margin":
                selected = self.margin_query(probs, pool_idx, n_select)
            elif strategy == "badge":
                self.model_current = model
                selected = self.badge_query(list(pool_idx), n_select)
            elif strategy == "adaptive_gap_entropy":
                selected = self.adaptive_gap_entropy_query(probs, pool_idx, n_select, self.y_train[labeled_idx])
            elif strategy == "class_aware_entropy":
                selected = self.class_aware_entropy_query(probs, pool_idx, n_select, self.y_train[labeled_idx])
            elif strategy == "gap_aware_entropy":
                selected = self.gap_aware_entropy_query(probs, pool_idx, n_select, self.y_train[labeled_idx])
            elif strategy == "coreset":
                selected = self.coreset_query(list(labeled_idx), list(pool_idx), n_select)
            elif strategy == "qbc":
                self.labeled_idx_current = labeled_idx
                selected = self.qbc_query(list(pool_idx), n_select)
            else:
                raise ValueError(f"Unknown strategy: {strategy}")

            labeled_idx.extend(selected)
            pool_idx = [i for i in pool_idx if i not in selected]

        return results


def main():
    parser = argparse.ArgumentParser(description="TML模型验证实验（图像数据集）")
    parser.add_argument("--model", type=str, choices=["lr", "rf"], required=True)
    parser.add_argument("--dataset", type=str, choices=["cifar10", "fashionmnist", "cifar100"], default="cifar10")
    parser.add_argument("--rho", type=int, default=10)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456])
    parser.add_argument("--strategies", type=str, nargs="+",
                        default=["random", "entropy", "margin", "badge", "adaptive_gap_entropy", "class_aware_entropy", "gap_aware_entropy", "coreset", "qbc"])
    parser.add_argument("--use-ssl", action="store_true", help="启用self-training SSL")
    parser.add_argument("--ssl-threshold", type=float, default=0.9, help="SSL伪标签置信度阈值")
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = str(PROJECT_ROOT / "output" / "tml_validation")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ssl_tag = "+SSL" if args.use_ssl else ""
    print("\n" + "=" * 80)
    print(f"TML模型验证实验{ssl_tag}: {args.model.upper()} on {args.dataset} (ρ={args.rho})")
    print("=" * 80)

    all_results = {}

    for strategy in args.strategies:
        strategy_results = []
        for seed in args.seeds:
            print(f"\n--- Strategy: {strategy}{ssl_tag}, Seed: {seed} ---")
            exp = TMLExperiment(args.model, args.dataset, args.rho, seed,
                              use_ssl=args.use_ssl, ssl_threshold=args.ssl_threshold)
            results = exp.run_al(strategy)
            strategy_results.append(results)

        final_f1s = [r["f1_scores"][-1] for r in strategy_results]
        mean_f1 = float(np.mean(final_f1s))
        std_f1 = float(np.std(final_f1s))

        all_results[strategy] = {
            "mean_f1": mean_f1,
            "std_f1": std_f1,
            "seeds": args.seeds,
            "all_f1_scores": [r["f1_scores"] for r in strategy_results]
        }
        print(f"\n[{strategy}{ssl_tag}] Final F1: {mean_f1:.4f} ± {std_f1:.4f}")

    # 保存结果
    ssl_suffix = "_ssl" if args.use_ssl else ""
    output_file = output_dir / f"{args.model}_{args.dataset}_rho{args.rho}{ssl_suffix}_results.json"
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved: {output_file}")

    # 打印汇总
    print("\n" + "=" * 80)
    print(f"Summary{ssl_tag}")
    print("=" * 80)
    print(f"{'Strategy':25s} {'Final F1':>10s} {'±std':>8s}")
    print("-" * 50)
    for strategy, data in all_results.items():
        print(f"{strategy:25s} {data['mean_f1']:.4f}    ±{data['std_f1']:.4f}")

    if "random" in all_results:
        random_f1 = all_results["random"]["mean_f1"]
        print(f"\nvs Random:")
        for strategy, data in all_results.items():
            if strategy != "random":
                delta = data["mean_f1"] - random_f1
                pct = delta / random_f1 * 100 if random_f1 > 0 else 0
                print(f"  {strategy}: {delta:+.4f} ({pct:+.2f}%)")


if __name__ == "__main__":
    main()
