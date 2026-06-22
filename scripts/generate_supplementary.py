"""生成补充实验图表"""
import json, os, sys, numpy as np
from pathlib import Path

BASE = Path("C:/Users/28414/Desktop/机器学习—图像分类-期末汇报/output")
FIG = Path("C:/Users/28414/Desktop/机器学习—图像分类-期末汇报/figures/supplementary")
FIG.mkdir(exist_ok=True)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ============================================================
# 1. 学习曲线
# ============================================================
print("1. Learning curves...")
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
rhos = [1, 5, 10, 20, 50, 100]
strategies = ['random', 'entropy', 'margin', 'class_aware_entropy', 'gap_aware_entropy', 'adaptive_gap_entropy']
colors = ['#6c757d', '#0d6efd', '#198754', '#dc3545', '#fd7e14', '#6f42c1']

for idx, rho in enumerate(rhos):
    ax = axes[idx // 3][idx % 3]
    for si, s in enumerate(strategies):
        f1s_all = []
        for seed in [42, 123, 456]:
            ckpt = BASE / "std_al" / f"rho{rho}" / "checkpoints" / f"{s}_seed{seed}.json"
            if ckpt.exists():
                d = json.load(open(ckpt))
                f1s_all.append(d['results']['f1_scores'])
        if f1s_all:
            mean_f1 = np.mean(f1s_all, axis=0)
            rounds = list(range(1, len(mean_f1) + 1))
            ax.plot(rounds, mean_f1, label=s.replace('_entropy','').replace('_',' '),
                   color=colors[si], linewidth=1.5)
    ax.set_xlabel('Round')
    ax.set_ylabel('F1')
    ax.set_title(f'rho={rho}')
    ax.grid(True, alpha=0.3)
    if idx == 0:
        ax.legend(fontsize=7, loc='lower right')

plt.suptitle('Learning Curves: F1 vs AL Round', fontsize=14)
plt.tight_layout()
plt.savefig(FIG / 'learning_curves.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Done")

# ============================================================
# 3. 标注类别分布
# ============================================================
print("3. Class distribution...")
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
rho = 50
plot_strategies = ['entropy', 'class_aware_entropy', 'gap_aware_entropy', 'adaptive_gap_entropy']
labels = ['Entropy', 'Class-Aware', 'Gap-Aware', 'Adaptive Gap']

for idx, (s, label) in enumerate(zip(plot_strategies, labels)):
    ax = axes[idx // 2][idx % 2]
    all_dists = []
    for seed in [42, 123, 456]:
        for group in ['std_al', 'innovative_al_ssl']:
            ckpt = BASE / group / f"rho{rho}" / "checkpoints" / f"{s}_seed{seed}.json"
            if ckpt.exists():
                d = json.load(open(ckpt))
                dists = d['results'].get('query_class_dist', [])
                if dists:
                    all_dists.append(dists)
                    break

    if all_dists:
        n_rounds = len(all_dists[0])
        n_classes = 10
        avg_dist = np.zeros((n_rounds, n_classes))
        for dists in all_dists:
            for r, dist in enumerate(dists):
                for c in range(n_classes):
                    avg_dist[r, c] += dist.get(str(c), 0)
        avg_dist /= len(all_dists)
        colors_class = plt.cm.tab10(np.linspace(0, 1, n_classes))
        cum = np.cumsum(avg_dist, axis=1)
        for c in range(n_classes):
            bottom = cum[:, c-1] if c > 0 else np.zeros(n_rounds)
            ax.fill_between(range(n_rounds), bottom, cum[:, c], alpha=0.7, color=colors_class[c], label=f'C{c}')

    ax.set_xlabel('Round')
    ax.set_ylabel('Cumulative Samples')
    ax.set_title(f'{label} (rho={rho})')
    if idx == 0:
        ax.legend(fontsize=5, ncol=2, loc='upper left')

plt.suptitle(f'Class Distribution Over AL Rounds (rho={rho})', fontsize=14)
plt.tight_layout()
plt.savefig(FIG / 'class_distribution_rho50.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Done")

# ============================================================
# 4. 伪标签质量
# ============================================================
print("4. Pseudo-label quality...")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for idx, rho in enumerate([10, 50]):
    ax = axes[idx]
    for s, label in [('entropy', 'Entropy+SSL'), ('class_aware_entropy', 'ClassAware+SSL')]:
        accs_all = []
        counts_all = []
        for seed in [42, 123, 456]:
            ckpt = BASE / "al_ssl" / f"rho{rho}" / "checkpoints" / f"{s}_seed{seed}.json"
            if ckpt.exists():
                d = json.load(open(ckpt))
                accs = d['results'].get('pseudo_acc_history', [])
                counts = d['results'].get('n_pseudo_labeled', [])
                if accs:
                    accs_all.append(accs)
                    counts_all.append(counts)
        if accs_all:
            mean_acc = np.mean(accs_all, axis=0)
            mean_count = np.mean(counts_all, axis=0)
            rounds = list(range(1, len(mean_acc) + 1))
            ax.plot(rounds, mean_acc, label=f'{label} acc', linewidth=2)
            ax2 = ax.twinx()
            ax2.bar(rounds, mean_count, alpha=0.3, label=f'{label} count')
            ax2.set_ylabel('Count')
    ax.set_xlabel('Round')
    ax.set_ylabel('Pseudo-label Accuracy')
    ax.set_title(f'rho={rho}')
    ax.legend(fontsize=7, loc='upper left')
    ax.grid(True, alpha=0.3)

plt.suptitle('Pseudo-Label Quality Over Rounds', fontsize=14)
plt.tight_layout()
plt.savefig(FIG / 'pseudo_label_quality.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Done")

# ============================================================
# 5. 计算成本
# ============================================================
print("5. Computational cost...")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
rho = 10
strategies = ['random', 'entropy', 'margin', 'class_aware_entropy', 'gap_aware_entropy']
labels = ['Random', 'Entropy', 'Margin', 'ClassAware', 'GapAware']

# Training time
ax = axes[0]
std_times, ssl_times = [], []
for s in strategies:
    for group, lst in [('std_al', std_times), ('al_ssl', ssl_times)]:
        f = BASE / group / f"rho{rho}" / "aggregated_results.json"
        if f.exists():
            d = json.load(open(f))
            lst.append(d.get(s, {}).get('avg_train_time_per_round', 0))
x = np.arange(len(strategies))
ax.bar(x - 0.2, std_times, 0.4, label='Pure AL', color='#0d6efd')
ax.bar(x + 0.2, ssl_times, 0.4, label='AL+SSL', color='#dc3545')
ax.set_xlabel('Strategy')
ax.set_ylabel('Time per Round (s)')
ax.set_title(f'Training Time (rho={rho})')
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=45)
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

# Query time
ax = axes[1]
std_qtimes = []
for s in strategies:
    f = BASE / "std_al" / f"rho{rho}" / "aggregated_results.json"
    if f.exists():
        d = json.load(open(f))
        std_qtimes.append(d.get(s, {}).get('avg_query_time', 0))
ax.bar(x, std_qtimes, 0.6, color='#198754')
ax.set_xlabel('Strategy')
ax.set_ylabel('Query Time (s)')
ax.set_title(f'AL Query Time (rho={rho})')
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=45)
ax.grid(True, alpha=0.3, axis='y')

plt.suptitle('Computational Cost', fontsize=14)
plt.tight_layout()
plt.savefig(FIG / 'computational_cost.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Done")

# ============================================================
# 2. Lambda sensitivity (需要跑实验)
# ============================================================
print("\n2. Lambda sensitivity - running experiment...")
V8 = str(PROJECT_ROOT / "experiments" / "v8_controlled_fast_al_ssl.py") if 'PROJECT_ROOT' in dir() else 'experiments/v8_controlled_fast_al_ssl.py'
V8 = str(Path("C:/Users/28414/Desktop/机器学习—图像分类-期末汇报/experiments/v8_controlled_fast_al_ssl.py"))

lam_results = {}
rho = 50
seed = 42

for lam in [0.1, 0.3, 0.5, 0.7, 1.0]:
    out = f'output/lambda_sensitivity/lam{lam}'
    os.makedirs(out, exist_ok=True)
    agg_file = f'{out}/aggregated_results.json'

    if os.path.exists(agg_file):
        d = json.load(open(agg_file))
        lam_results[lam] = d.get('class_aware_entropy', {}).get('final_f1_mean', 0)
        print(f"  lambda={lam}: {lam_results[lam]:.4f} (cached)")
    else:
        import subprocess
        cmd = [sys.executable, V8,
            '--dataset', 'cifar10', '--budget-level', 'ultra_low', '--model-type', 'simplecnn',
            '--strategies', 'class_aware_entropy', '--seeds', str(seed),
            '--imbalance-ratio', str(rho), '--output-dir', out,
            '--class-aware-lambda', str(lam)]
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if os.path.exists(agg_file):
            d = json.load(open(agg_file))
            lam_results[lam] = d.get('class_aware_entropy', {}).get('final_f1_mean', 0)
            print(f"  lambda={lam}: {lam_results[lam]:.4f}")
        else:
            print(f"  lambda={lam}: FAILED")

if lam_results:
    fig, ax = plt.subplots(figsize=(8, 5))
    lams = sorted(lam_results.keys())
    f1s = [lam_results[l] for l in lams]
    ax.plot(lams, f1s, 'o-', color='#0d6efd', linewidth=2, markersize=8)
    ax.set_xlabel('Lambda')
    ax.set_ylabel('F1')
    ax.set_title(f'Class-Aware Lambda Sensitivity (rho={rho}, seed={seed})')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0.2585, color='#6c757d', linestyle='--', label='Entropy baseline')
    ax.legend()
    plt.tight_layout()
    plt.savefig(FIG / 'lambda_sensitivity.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Done: best lambda={lams[np.argmax(f1s)]} with F1={max(f1s):.4f}")

print("\nAll done!")
