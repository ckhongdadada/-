"""
从已有实验结果生成论文所需的分析文件
======================================
生成:
1. tail_aware_summary.json — 尾类感知策略汇总
2. comprehensive_analysis.json — 统计检验（配对t检验 + Cohen's d）
3. cross_rho_comparison/ — 跨ρ值对比数据

运行方式:
    python experiments/generate_analysis_files.py
"""

import json
import numpy as np
from pathlib import Path
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "output"
CROSS_RHO_DIR = OUTPUT_DIR / "cross_rho_comparison"


def load_aggregated(experiment_dir, rho):
    """加载某个实验组某个ρ的聚合结果"""
    path = OUTPUT_DIR / experiment_dir / f"rho{rho}" / "aggregated_results.json"
    if not path.exists():
        return None
    with open(path, 'r') as f:
        return json.load(f)


def load_raw(experiment_dir, rho):
    """加载原始结果"""
    path = OUTPUT_DIR / experiment_dir / f"rho{rho}" / "raw_results.json"
    if not path.exists():
        return None
    with open(path, 'r') as f:
        return json.load(f)


def load_tail_aware_seeds(rho):
    """加载tail_aware的种子级结果"""
    tail_dir = OUTPUT_DIR / "tail_aware_100"
    if not tail_dir.exists():
        return None
    seeds = [42, 123, 456, 789, 1024]
    all_f1 = []
    for seed in seeds:
        path = tail_dir / f"cifar10_rho{rho}_tail_aware_entropy_seed{seed}.json"
        if path.exists():
            with open(path, 'r') as f:
                data = json.load(f)
                if "f1_scores" in data:
                    all_f1.append(data["f1_scores"][-1])
                elif "all_f1_scores" in data:
                    all_f1.append(data["all_f1_scores"][-1])
    return all_f1 if all_f1 else None


def generate_tail_aware_summary():
    """生成tail_aware_summary.json"""
    print("生成 tail_aware_summary.json ...")
    summary = {}

    for rho in [1, 5, 10, 20, 50, 100]:
        rho_data = {}

        # 加载标准策略（从std_al）
        std = load_aggregated("std_al", rho)
        if std:
            for strategy, data in std.items():
                if strategy == "full_supervision":
                    continue
                rho_data[strategy] = {
                    "mean_f1": data.get("final_f1_mean", 0),
                    "std_f1": data.get("final_f1_std", 0),
                }

        # 加载tail_aware策略
        tail_f1s = load_tail_aware_seeds(rho)
        if tail_f1s:
            rho_data["tail_aware_entropy"] = {
                "mean_f1": float(np.mean(tail_f1s)),
                "std_f1": float(np.std(tail_f1s)),
                "seeds": [42, 123, 456, 789, 1024],
                "all_f1_scores": tail_f1s,
            }

        summary[f"rho_{rho}"] = rho_data

    # 保存
    path = OUTPUT_DIR / "tail_aware_summary.json"
    with open(path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"  已保存: {path}")
    return summary


def generate_comprehensive_analysis():
    """生成comprehensive_analysis.json — 统计检验"""
    print("生成 comprehensive_analysis.json ...")
    analysis = {}

    for rho in [1, 5, 10, 20, 50, 100]:
        rho_analysis = {}

        # 加载各实验组的聚合结果
        std = load_aggregated("std_al", rho)
        al_ssl = load_aggregated("al_ssl", rho)
        innov = load_aggregated("innovative_al_ssl", rho)

        if std is None:
            continue

        # 提取各策略的最终F1（所有种子）
        def extract_final_f1s(agg_data):
            result = {}
            if not agg_data:
                return result
            for strategy, data in agg_data.items():
                if strategy == "full_supervision":
                    continue
                if "all_f1_scores" in data:
                    result[strategy] = [s[-1] for s in data["all_f1_scores"]]
            return result

        std_f1s = extract_final_f1s(std)
        al_ssl_f1s = extract_final_f1s(al_ssl)
        innov_f1s = extract_final_f1s(innov)

        # 配对t检验: 创新策略 vs 最佳基线
        if innov_f1s and std_f1s:
            best_baseline = max(std_f1s.items(), key=lambda x: np.mean(x[1]))
            best_name, best_scores = best_baseline

            for innov_name, innov_scores in innov_f1s.items():
                if len(innov_scores) >= 2 and len(best_scores) >= 2:
                    min_len = min(len(innov_scores), len(best_scores))
                    t_stat, p_value = stats.ttest_rel(innov_scores[:min_len], best_scores[:min_len])
                    cohens_d = (np.mean(innov_scores) - np.mean(best_scores)) / np.std(
                        [a - b for a, b in zip(innov_scores[:min_len], best_scores[:min_len])]
                    ) if np.std([a - b for a, b in zip(innov_scores[:min_len], best_scores[:min_len])]) > 0 else 0

                    rho_analysis[f"{innov_name}_vs_{best_name}"] = {
                        "t_statistic": float(t_stat),
                        "p_value": float(p_value),
                        "cohens_d": float(cohens_d),
                        "significant": p_value < 0.05,
                        "innov_mean": float(np.mean(innov_scores)),
                        "baseline_mean": float(np.mean(best_scores)),
                        "improvement": float(np.mean(innov_scores) - np.mean(best_scores)),
                    }

        # AL+SSL vs 纯AL（最佳策略对比）
        if al_ssl_f1s and std_f1s:
            best_al = max(std_f1s.items(), key=lambda x: np.mean(x[1]))
            best_ssl = max(al_ssl_f1s.items(), key=lambda x: np.mean(x[1]))
            rho_analysis["al_ssl_vs_std_al"] = {
                "best_al_strategy": best_al[0],
                "best_al_f1": float(np.mean(best_al[1])),
                "best_ssl_strategy": best_ssl[0],
                "best_ssl_f1": float(np.mean(best_ssl[1])),
                "ssl_improvement": float(np.mean(best_ssl[1]) - np.mean(best_al[1])),
            }

        analysis[f"rho_{rho}"] = rho_analysis

    path = OUTPUT_DIR / "comprehensive_analysis.json"
    with open(path, 'w') as f:
        json.dump(analysis, f, indent=2)
    print(f"  已保存: {path}")
    return analysis


def generate_cross_rho_comparison():
    """生成cross_rho_comparison/目录下的跨ρ对比数据"""
    print("生成 cross_rho_comparison/ ...")
    CROSS_RHO_DIR.mkdir(parents=True, exist_ok=True)

    # 1. 所有策略跨ρ对比
    strategies = ["random", "entropy", "margin", "coreset", "badge", "qbc",
                  "class_aware_entropy", "gap_aware_entropy", "adaptive_gap_entropy"]
    rhos = [1, 5, 10, 20, 50, 100]

    cross_rho = {}
    for strategy in strategies:
        strategy_data = {}
        for rho in rhos:
            # 先查std_al
            std = load_aggregated("std_al", rho)
            if std and strategy in std:
                strategy_data[f"rho_{rho}"] = {
                    "mean_f1": std[strategy].get("final_f1_mean", 0),
                    "std_f1": std[strategy].get("final_f1_std", 0),
                    "source": "std_al"
                }
            else:
                # 查innovative_al_ssl
                innov = load_aggregated("innovative_al_ssl", rho)
                if innov and strategy in innov:
                    strategy_data[f"rho_{rho}"] = {
                        "mean_f1": innov[strategy].get("final_f1_mean", 0),
                        "std_f1": innov[strategy].get("final_f1_std", 0),
                        "source": "innovative_al_ssl"
                    }
        cross_rho[strategy] = strategy_data

    path = CROSS_RHO_DIR / "all_strategies_cross_rho.json"
    with open(path, 'w') as f:
        json.dump(cross_rho, f, indent=2)
    print(f"  已保存: {path}")

    # 2. 各ρ最佳策略对比
    best_per_rho = {}
    for rho in rhos:
        std = load_aggregated("std_al", rho)
        if not std:
            continue
        best = max(
            [(s, d) for s, d in std.items() if s != "full_supervision"],
            key=lambda x: x[1].get("final_f1_mean", 0)
        )
        best_per_rho[f"rho_{rho}"] = {
            "best_strategy": best[0],
            "mean_f1": best[1].get("final_f1_mean", 0),
            "std_f1": best[1].get("final_f1_std", 0),
        }

    path = CROSS_RHO_DIR / "best_strategy_per_rho.json"
    with open(path, 'w') as f:
        json.dump(best_per_rho, f, indent=2)
    print(f"  已保存: {path}")

    # 3. 创新策略跨ρ对比
    innov_strategies = ["class_aware_entropy", "gap_aware_entropy", "adaptive_gap_entropy"]
    innov_cross = {}
    for strategy in innov_strategies:
        strategy_data = {}
        for rho in rhos:
            innov = load_aggregated("innovative_al_ssl", rho)
            if innov and strategy in innov:
                strategy_data[f"rho_{rho}"] = {
                    "mean_f1": innov[strategy].get("final_f1_mean", 0),
                    "std_f1": innov[strategy].get("final_f1_std", 0),
                }
        innov_cross[strategy] = strategy_data

    path = CROSS_RHO_DIR / "innovative_strategies_cross_rho.json"
    with open(path, 'w') as f:
        json.dump(innov_cross, f, indent=2)
    print(f"  已保存: {path}")


def main():
    print("=" * 60)
    print("生成论文所需分析文件")
    print("=" * 60)

    generate_tail_aware_summary()
    generate_comprehensive_analysis()
    generate_cross_rho_comparison()

    print("\n全部完成！")


if __name__ == "__main__":
    main()
