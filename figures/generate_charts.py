import matplotlib.pyplot as plt
import numpy as np

# Set Chinese font for Windows
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# Set style
plt.rcParams['figure.facecolor'] = '#0f172a'
plt.rcParams['axes.facecolor'] = '#0f172a'
plt.rcParams['axes.edgecolor'] = '#334155'
plt.rcParams['axes.labelcolor'] = '#94a3b8'
plt.rcParams['xtick.color'] = '#94a3b8'
plt.rcParams['ytick.color'] = '#94a3b8'
plt.rcParams['text.color'] = '#e2e8f0'
plt.rcParams['font.size'] = 11

# ==================== Chart 1: Basic AL Results (s10) ====================
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
fig.patch.set_facecolor('#0f172a')

rho = [1, 10, 50, 100]
x = np.arange(len(rho))
width = 0.3

# ResNet18 data
resnet_random = [0.560, 0.537, 0.456, 0.407]
resnet_best =   [0.570, 0.544, 0.481, 0.412]
resnet_labels = ['CoreSet', 'BADGE', 'BADGE', 'Margin']

# SimpleCNN data
cnn_random = [0.443, 0.356, 0.248, 0.221]
cnn_best =   [0.451, 0.377, 0.268, 0.261]
cnn_labels = ['QBC', 'QBC', 'Margin', 'Margin']

# Left: ResNet18
ax = axes[0]
bars1 = ax.bar(x - width/2, resnet_random, width, label='Random', color='#64748b', edgecolor='#334155', linewidth=0.5)
bars2 = ax.bar(x + width/2, resnet_best, width, label='Best AL', color='#3b82f6', edgecolor='#60a5fa', linewidth=0.5)
ax.set_xlabel('Imbalance Ratio ρ', fontsize=12)
ax.set_ylabel('Macro-F1', fontsize=12)
ax.set_title('ResNet18', fontsize=13, fontweight='bold', color='#e2e8f0')
ax.set_xticks(x)
ax.set_xticklabels([f'ρ={r}' for r in rho])
ax.legend(loc='upper right', facecolor='#1e293b', edgecolor='#334155')
ax.set_ylim(0, 0.65)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(axis='y', alpha=0.2, color='#334155')

# Add value labels and gain annotations
for i, (r, b, label) in enumerate(zip(resnet_random, resnet_best, resnet_labels)):
    ax.text(i - width/2, r + 0.01, f'{r:.3f}', ha='center', va='bottom', fontsize=9, color='#94a3b8')
    ax.text(i + width/2, b + 0.01, f'{b:.3f}\n({label})', ha='center', va='bottom', fontsize=9, color='#60a5fa')
    gain = (b - r) / r * 100
    if gain > 2:
        ax.annotate(f'+{gain:.1f}%', xy=(i, max(r, b) + 0.03), ha='center', fontsize=9, color='#22c55e', fontweight='bold')

# Right: SimpleCNN
ax = axes[1]
bars1 = ax.bar(x - width/2, cnn_random, width, label='Random', color='#64748b', edgecolor='#334155', linewidth=0.5)
bars2 = ax.bar(x + width/2, cnn_best, width, label='Best AL', color='#22c55e', edgecolor='#4ade80', linewidth=0.5)
ax.set_xlabel('Imbalance Ratio ρ', fontsize=12)
ax.set_ylabel('Macro-F1', fontsize=12)
ax.set_title('SimpleCNN', fontsize=13, fontweight='bold', color='#e2e8f0')
ax.set_xticks(x)
ax.set_xticklabels([f'ρ={r}' for r in rho])
ax.legend(loc='upper right', facecolor='#1e293b', edgecolor='#334155')
ax.set_ylim(0, 0.55)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(axis='y', alpha=0.2, color='#334155')

for i, (r, b, label) in enumerate(zip(cnn_random, cnn_best, cnn_labels)):
    ax.text(i - width/2, r + 0.008, f'{r:.3f}', ha='center', va='bottom', fontsize=9, color='#94a3b8')
    ax.text(i + width/2, b + 0.008, f'{b:.3f}\n({label})', ha='center', va='bottom', fontsize=9, color='#4ade80')
    gain = (b - r) / r * 100
    if gain > 2:
        ax.annotate(f'+{gain:.1f}%', xy=(i, max(r, b) + 0.025), ha='center', fontsize=9, color='#22c55e', fontweight='bold')

fig.suptitle('基础AL策略对比：Random vs 最佳策略', fontsize=14, fontweight='bold', color='#e2e8f0', y=1.02)
plt.tight_layout()
plt.savefig('c:/Users/28414/Desktop/机器学习—图像分类-期末汇报/figures/chart_s10_basic_al.png', dpi=150, bbox_inches='tight', facecolor='#0f172a')
plt.close()

# ==================== Chart 2: AL Innovation Results (s12a) ====================
fig, ax = plt.subplots(figsize=(12, 5.5))
fig.patch.set_facecolor('#0f172a')

strategies = ['Random', 'Entropy', 'Class-Aware', 'Gap-Aware']
rho_labels = ['ρ=1', 'ρ=10', 'ρ=50', 'ρ=100']

data = {
    'ρ=1':  [0.443, 0.425, 0.409, 0.422],
    'ρ=10': [0.356, 0.344, 0.359, 0.341],
    'ρ=50': [0.248, 0.259, 0.244, 0.249],
    'ρ=100':[0.221, 0.258, 0.231, 0.255],
}

colors = ['#64748b', '#f59e0b', '#3b82f6', '#22c55e']
x = np.arange(len(rho_labels))
width = 0.18

for i, (strategy, color) in enumerate(zip(strategies, colors)):
    values = [data[r][i] for r in rho_labels]
    bars = ax.bar(x + i * width - 1.5 * width, values, width, label=strategy, color=color, edgecolor='#334155', linewidth=0.5)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005, f'{val:.3f}', ha='center', va='bottom', fontsize=8, color=color)

ax.set_xlabel('Imbalance Ratio', fontsize=12)
ax.set_ylabel('Macro-F1', fontsize=12)
ax.set_title('纯AL策略对比：尾类感知策略 (SimpleCNN, 无SSL)', fontsize=13, fontweight='bold', color='#e2e8f0')
ax.set_xticks(x)
ax.set_xticklabels(rho_labels)
ax.legend(loc='upper right', facecolor='#1e293b', edgecolor='#334155')
ax.set_ylim(0, 0.52)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(axis='y', alpha=0.2, color='#334155')

# Highlight best for each rho
best_indices = [3, 2, 3, 3]  # Gap-Aware, Class-Aware, Gap-Aware, Gap-Aware
for i, best_idx in enumerate(best_indices):
    bar_x = x[i] + best_idx * width - 1.5 * width
    bar_y = data[rho_labels[i]][best_idx]
    ax.plot([bar_x, bar_x], [bar_y + 0.035, bar_y + 0.055], color='#22c55e', linewidth=2)
    ax.plot([bar_x - 0.03, bar_x + 0.03], [bar_y + 0.055, bar_y + 0.055], color='#22c55e', linewidth=2)
    ax.text(bar_x, bar_y + 0.065, 'Best', ha='center', fontsize=9, color='#22c55e', fontweight='bold')

plt.tight_layout()
plt.savefig('c:/Users/28414/Desktop/机器学习—图像分类-期末汇报/figures/chart_s12a_al_innovation.png', dpi=150, bbox_inches='tight', facecolor='#0f172a')
plt.close()

print("Charts generated successfully!")
