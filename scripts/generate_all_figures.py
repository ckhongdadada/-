"""
论文图表生成脚本
================
从 output/ 目录读取实验结果，生成论文所需的全部图表和统计检验。

生成内容:
  1. 学习曲线对比图 (F1 vs Round)
  2. 标签效率曲线 (F1 vs 标注数量)
  3. 消融实验热力图
  4. 标注类别分布变化 (堆叠面积图)
  5. 伪标签质量分析
  6. 跨rho热力图 (策略 x rho)
  7. 统计检验表
  8. 计算成本对比

运行方式:
    python scripts/generate_all_figures.py
    python scripts/generate_all_figures.py --only 1,2,6
"""

import os
import sys
import json
import argparse
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "output"
CIFAR100_DIR = PROJECT_ROOT / "output" / "cifar100"
FIGURE_DIR = PROJECT_ROOT / "figures"
FIGURE_DIR.mkdir(exist_ok=True)

RHO_LIST = [1, 5, 10, 20, 50, 100]
STD_STRATEGIES = ["random", "entropy", "margin", "coreset", "badge", "qbc"]
INNOVATIVE_STRATEGIES = ["class_aware_entropy", "gap_aware_entropy", "adaptive_gap_entropy"]
ALL_STRATEGIES = STD_STRATEGIES + INNOVATIVE_STRATEGIES
SEEDS = [42, 123, 456, 789, 1024]

STRATEGY_NAMES = {
    "random": "Random", "entropy": "Entropy", "margin": "Margin",
    "coreset": "CoreSet", "badge": "BADGE", "qbc": "QBC",
    "class_aware_entropy": "Class-Aware", "gap_aware_entropy": "Gap-Aware",
    "adaptive_gap_entropy": "Adaptive Gap",
}


def load_aggregated(group, rho, base_dir=None):
    if base_dir is None:
        base_dir = OUTPUT_DIR
    f = base_dir / group / f"rho{rho}" / "aggregated_results.json"
    if f.exists():
        with open(f) as fh:
            return json.load(fh)
    return None


def load_checkpoint(group, rho, strategy, seed, base_dir=None):
    if base_dir is None:
        base_dir = OUTPUT_DIR
    f = base_dir / group / f"rho{rho}" / "checkpoints" / f"{strategy}_seed{seed}.json"
    if f.exists():
        with open(f) as fh:
            return json.load(fh)
    return None


def find_checkpoint(strategy, seed, rho):
    """在所有可能的目录中查找checkpoint"""
    for group in ["std_al", "al_ssl", "innovative_al_ssl", "innovative_al_ssl_basic",
                   "al_ssl_innovative", "tail_aware_100"]:
        ck = load_checkpoint(group, rho, strategy, seed)
        if ck:
            return ck
    # CIFAR-100
    for group in ["std_al", "al_ssl", "innovative_al_ssl"]:
        ck = load_checkpoint(group, rho, strategy, seed, base_dir=CIFAR100_DIR)
        if ck:
            return ck
    return None


# ============================================================
# 1. 学习曲线
# ============================================================
def plot_learning_curves():
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("[SKIP] matplotlib not installed")
        return

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for idx, rho in enumerate(RHO_LIST):
        ax = axes[idx]
        data = load_aggregated("std_al", rho)
        if data is None:
            ax.set_title(f"rho={rho} (no data)")
            continue

        for strategy in STD_STRATEGIES:
            if strategy not in data:
                continue
            r = data[strategy]
            n = len(r["f1_mean"])
            rounds = list(range(1, n + 1))
            ax.plot(rounds, r["f1_mean"], label=STRATEGY_NAMES.get(strategy, strategy),
                    marker='o', markersize=3)
            ax.fill_between(rounds,
                            [m - s for m, s in zip(r["f1_mean"], r["f1_std"])],
                            [m + s for m, s in zip(r["f1_mean"], r["f1_std"])],
                            alpha=0.15)

        ax.set_xlabel("Round")
        ax.set_ylabel("Macro-F1")
        ax.set_title(f"rho={rho}")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    plt.suptitle("Learning Curves: Standard AL Strategies (CIFAR-10)", fontsize=14)
    plt.tight_layout()
    out = FIGURE_DIR / "01_learning_curves.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"[OK] {out}")


# ============================================================
# 2. 标签效率曲线
# ============================================================
def plot_label_efficiency():
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("[SKIP] matplotlib not installed")
        return

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for idx, rho in enumerate(RHO_LIST):
        ax = axes[idx]
        std_data = load_aggregated("std_al", rho)
        ssl_data = load_aggregated("al_ssl", rho)
        innov_data = load_aggregated("innovative_al_ssl", rho)

        if std_data:
            for strategy in ["random", "entropy", "qbc"]:
                if strategy in std_data:
                    r = std_data[strategy]
                    ax.plot(r["n_human_labeled_mean"], r["f1_mean"],
                            label=f"{STRATEGY_NAMES[strategy]} (AL)", marker='o', markersize=3)

        if ssl_data:
            for strategy in ["random", "entropy"]:
                if strategy in ssl_data:
                    r = ssl_data[strategy]
                    ax.plot(r["n_human_labeled_mean"], r["f1_mean"],
                            label=f"{STRATEGY_NAMES[strategy]} (AL+SSL)", marker='s', markersize=3, linestyle='--')

        if innov_data:
            for strategy in ["class_aware_entropy", "gap_aware_entropy"]:
                if strategy in innov_data:
                    r = innov_data[strategy]
                    ax.plot(r["n_human_labeled_mean"], r["f1_mean"],
                            label=f"{STRATEGY_NAMES[strategy]} (Innov)", marker='^', markersize=3, linestyle=':')

        ax.set_xlabel("Labeled Samples")
        ax.set_ylabel("Macro-F1")
        ax.set_title(f"rho={rho}")
        ax.legend(fontsize=6)
        ax.grid(True, alpha=0.3)

    plt.suptitle("Label Efficiency: F1 vs Number of Labeled Samples", fontsize=14)
    plt.tight_layout()
    out = FIGURE_DIR / "02_label_efficiency.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"[OK] {out}")


# ============================================================
# 3. 消融实验热力图
# ============================================================
def plot_ablation_heatmap():
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("[SKIP] matplotlib not installed")
        return

    groups = [
        ("std_al", "AL only\n(no SSL)"),
        ("al_ssl", "Base AL\n+ Base SSL"),
        ("innovative_al_ssl_basic", "Innov AL\n+ Base SSL"),
        ("al_ssl_innovative", "Base AL\n+ Innov SSL"),
        ("innovative_al_ssl", "Innov AL\n+ Innov SSL"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    for idx, rho in enumerate([10, 50, 100]):
        ax = axes[idx]
        plot_strategies = ["random", "entropy", "class_aware_entropy", "gap_aware_entropy", "adaptive_gap_entropy"]
        x = np.arange(len(groups))
        width = 0.15

        for i, strategy in enumerate(plot_strategies):
            vals = []
            for group, _ in groups:
                data = load_aggregated(group, rho)
                if data and strategy in data:
                    vals.append(data[strategy]["final_f1_mean"])
                else:
                    vals.append(0)
            offset = (i - len(plot_strategies) / 2) * width
            ax.bar(x + offset, vals, width, label=STRATEGY_NAMES.get(strategy, strategy))

        ax.set_xlabel("Ablation Configuration")
        ax.set_ylabel("Final Macro-F1")
        ax.set_title(f"rho={rho}")
        ax.set_xticks(x)
        ax.set_xticklabels([g[1] for g in groups], fontsize=7)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3, axis='y')

    plt.suptitle("Ablation Study: AL Innovation vs SSL Innovation", fontsize=14)
    plt.tight_layout()
    out = FIGURE_DIR / "03_ablation_heatmap.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"[OK] {out}")


# ============================================================
# 4. 标注类别分布变化
# ============================================================
def plot_class_distribution():
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("[SKIP] matplotlib not installed")
        return

    rho = 10
    strategies_to_plot = ["random", "entropy", "class_aware_entropy", "gap_aware_entropy"]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    for idx, strategy in enumerate(strategies_to_plot):
        ax = axes[idx]
        all_dists = []
        for seed in SEEDS:
            ckpt = find_checkpoint(strategy, seed, rho)
            if ckpt and "results" in ckpt:
                dists = ckpt["results"].get("query_class_dist", [])
                if dists:
                    all_dists.append(dists)

        if not all_dists:
            ax.set_title(f"{STRATEGY_NAMES.get(strategy, strategy)} (no data)")
            continue

        n_rounds = len(all_dists[0])
        n_classes = 10
        avg_dist = np.zeros((n_rounds, n_classes))
        for dists in all_dists:
            for r, d in enumerate(dists):
                for c in range(n_classes):
                    avg_dist[r, c] += d.get(str(c), 0)
        avg_dist /= len(all_dists)

        colors = plt.cm.tab10(np.linspace(0, 1, n_classes))
        cum_dist = np.cumsum(avg_dist, axis=0)
        for c in range(n_classes):
            bottom = cum_dist[:, c - 1] if c > 0 else np.zeros(n_rounds)
            ax.fill_between(range(n_rounds), bottom, cum_dist[:, c],
                            alpha=0.7, color=colors[c], label=f"Class {c}")

        ax.set_xlabel("Round")
        ax.set_ylabel("Cumulative Labeled Samples")
        ax.set_title(f"{STRATEGY_NAMES.get(strategy, strategy)} (rho={rho})")
        ax.legend(fontsize=6, ncol=2)

    plt.suptitle("Class Distribution of Selected Samples Over AL Rounds", fontsize=14)
    plt.tight_layout()
    out = FIGURE_DIR / "04_class_distribution.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"[OK] {out}")


# ============================================================
# 5. 伪标签质量分析
# ============================================================
def plot_pseudo_label_quality():
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("[SKIP] matplotlib not installed")
        return

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for idx, rho in enumerate(RHO_LIST):
        ax = axes[idx]

        # 从 checkpoint 中获取 per-round 伪标签数据
        strategies_with_pseudo = ["random", "entropy", "margin"]
        for strategy in strategies_with_pseudo:
            all_pseudo = []
            for seed in SEEDS[:3]:  # 只用前3个seed避免太慢
                ckpt = load_checkpoint("al_ssl", rho, strategy, seed)
                if ckpt and "results" in ckpt:
                    pseudo = ckpt["results"].get("n_pseudo_labeled", [])
                    if pseudo:
                        all_pseudo.append(pseudo)

            if all_pseudo:
                avg_pseudo = np.mean(all_pseudo, axis=0)
                rounds = list(range(1, len(avg_pseudo) + 1))
                ax.plot(rounds, avg_pseudo, label=STRATEGY_NAMES.get(strategy, strategy),
                        marker='o', markersize=3)

        ax.set_xlabel("Round")
        ax.set_ylabel("Pseudo-Labels Generated")
        ax.set_title(f"rho={rho}")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    plt.suptitle("SSL Pseudo-Label Generation per Round", fontsize=14)
    plt.tight_layout()
    out = FIGURE_DIR / "05_pseudo_label_quality.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"[OK] {out}")


# ============================================================
# 6. 跨rho热力图
# ============================================================
def plot_cross_rho_heatmap():
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("[SKIP] matplotlib not installed")
        return

    # 收集所有组的数据
    all_groups = {
        "std_al": STD_STRATEGIES,
        "innovative_al_ssl": INNOVATIVE_STRATEGIES,
    }

    matrix = {}
    for group_name, strategies in all_groups.items():
        for rho in RHO_LIST:
            data = load_aggregated(group_name, rho)
            if data:
                for strategy in strategies:
                    if strategy not in data:
                        continue
                    if strategy not in matrix:
                        matrix[strategy] = {}
                    matrix[strategy][rho] = data[strategy]["final_f1_mean"]

    if not matrix:
        print("[SKIP] No data for heatmap")
        return

    strategies = list(matrix.keys())
    rhos = RHO_LIST
    values = np.zeros((len(strategies), len(rhos)))
    for i, s in enumerate(strategies):
        for j, r in enumerate(rhos):
            values[i, j] = matrix[s].get(r, 0)

    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(values, cmap='YlOrRd', aspect='auto')

    ax.set_xticks(range(len(rhos)))
    ax.set_xticklabels([f"rho={r}" for r in rhos])
    ax.set_yticks(range(len(strategies)))
    ax.set_yticklabels([STRATEGY_NAMES.get(s, s) for s in strategies])

    for i in range(len(strategies)):
        for j in range(len(rhos)):
            val = values[i, j]
            if val > 0:
                ax.text(j, i, f"{val:.3f}", ha="center", va="center", fontsize=8)

    plt.colorbar(im, label="Final Macro-F1")
    plt.title("Strategy Performance Across Imbalance Ratios (CIFAR-10)")
    plt.tight_layout()
    out = FIGURE_DIR / "06_cross_rho_heatmap.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"[OK] {out}")


# ============================================================
# 7. 统计检验表
# ============================================================
def generate_statistical_tests():
    try:
        from scipy import stats
    except ImportError:
        print("[SKIP] scipy not installed")
        return

    results = []
    for rho in [10, 50, 100]:
        for group_name in ["std_al", "innovative_al_ssl"]:
            data = load_aggregated(group_name, rho)
            if data is None:
                continue

            strategies = [s for s in data if s != "full_supervision"]
            for s1 in strategies:
                for s2 in strategies:
                    if s1 >= s2:
                        continue
                    f1s1, f1s2 = [], []
                    seeds_list = data[s1].get("seeds", SEEDS)
                    for seed in seeds_list:
                        ck1 = load_checkpoint(group_name, rho, s1, seed)
                        ck2 = load_checkpoint(group_name, rho, s2, seed)
                        if ck1 and ck2:
                            f1s1.append(ck1["results"]["f1_scores"][-1])
                            f1s2.append(ck2["results"]["f1_scores"][-1])

                    if len(f1s1) >= 3 and len(f1s2) >= 3:
                        diff = np.mean(f1s1) - np.mean(f1s2)
                        t_stat, p_val = stats.ttest_rel(f1s1, f1s2)
                        pooled_std = np.sqrt((np.std(f1s1, ddof=1) ** 2 + np.std(f1s2, ddof=1) ** 2) / 2)
                        cohens_d = diff / pooled_std if pooled_std > 0 else 0
                        d_abs = abs(cohens_d)
                        effect = "large" if d_abs >= 0.8 else "medium" if d_abs >= 0.5 else "small" if d_abs >= 0.2 else "negligible"
                        sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"

                        results.append({
                            "rho": rho, "group": group_name, "s1": s1, "s2": s2,
                            "diff": diff, "t": t_stat, "p": p_val,
                            "cohens_d": cohens_d, "effect": effect, "sig": sig, "n": len(f1s1),
                        })

    out = FIGURE_DIR / "07_statistical_tests.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*80}")
    print("Statistical Tests (paired t-test, alpha=0.05)")
    print(f"{'='*80}")
    print(f"{'rho':>4} {'Group':<25} {'S1':<20} {'S2':<20} {'diff':>7} {'t':>7} {'p':>8} {'d':>6} {'Effect':<10} {'Sig':<4}")
    print("-" * 80)
    for r in results:
        print(f"{r['rho']:>4} {r['group']:<25} {r['s1']:<20} {r['s2']:<20} "
              f"{r['diff']:>7.4f} {r['t']:>7.2f} {r['p']:>8.4f} {r['cohens_d']:>6.2f} {r['effect']:<10} {r['sig']:<4}")
    print(f"\n[OK] {out}")


# ============================================================
# 8. 计算成本对比
# ============================================================
def plot_computational_cost():
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("[SKIP] matplotlib not installed")
        return

    rho = 10
    groups = ["std_al", "al_ssl", "innovative_al_ssl"]
    group_labels = ["Standard AL", "AL + Base SSL", "Innovative AL + SSL"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for idx, (metric, ylabel) in enumerate([("avg_train_time_per_round", "Seconds/round"),
                                             ("total_pseudo_labeled", "Count")]):
        ax = axes[idx]
        x = np.arange(len(STD_STRATEGIES))
        width = 0.25

        for g_idx, (group, label) in enumerate(zip(groups, group_labels)):
            data = load_aggregated(group, rho)
            if data is None:
                continue
            vals = []
            for strategy in STD_STRATEGIES:
                if strategy in data:
                    vals.append(data[strategy].get(metric, 0))
                else:
                    vals.append(0)
            ax.bar(x + g_idx * width, vals, width, label=label)

        ax.set_xlabel("Strategy")
        ax.set_ylabel(ylabel)
        ax.set_title(metric.replace("_", " ").title())
        ax.set_xticks(x + width)
        ax.set_xticklabels([STRATEGY_NAMES.get(s, s) for s in STD_STRATEGIES], rotation=45)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, axis='y')

    plt.suptitle(f"Computational Cost Comparison (rho={rho})", fontsize=14)
    plt.tight_layout()
    out = FIGURE_DIR / "08_computational_cost.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"[OK] {out}")


# ============================================================
# 主函数
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="Generate all paper figures")
    parser.add_argument("--only", type=str, default=None,
                        help="Comma-separated figure numbers (e.g., 1,2,6)")
    args = parser.parse_args()

    selected = [int(x) for x in args.only.split(",")] if args.only else list(range(1, 9))

    print(f"Output: {FIGURE_DIR}")
    print(f"Figures: {selected}")
    print("=" * 60)

    generators = {
        1: ("Learning Curves", plot_learning_curves),
        2: ("Label Efficiency", plot_label_efficiency),
        3: ("Ablation Heatmap", plot_ablation_heatmap),
        4: ("Class Distribution", plot_class_distribution),
        5: ("Pseudo-Label Quality", plot_pseudo_label_quality),
        6: ("Cross-rho Heatmap", plot_cross_rho_heatmap),
        7: ("Statistical Tests", generate_statistical_tests),
        8: ("Computational Cost", plot_computational_cost),
    }

    for num in selected:
        if num in generators:
            name, func = generators[num]
            print(f"\n[{num}] {name}...")
            try:
                func()
            except Exception as e:
                print(f"[ERROR] {name}: {e}")
                import traceback
                traceback.print_exc()

    print(f"\n{'='*60}")
    print(f"Done! Figures saved to {FIGURE_DIR}")


if __name__ == "__main__":
    main()
