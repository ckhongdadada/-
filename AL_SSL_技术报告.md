# AL+SSL联合策略技术报告

## 1. 策略概述

本报告详细描述主动学习（AL）与半监督学习（SSL）联合策略的完整技术实现。该策略在低标注预算（n_initial=100, n_query=100, n_rounds=10）下，针对CIFAR-10长尾分布图像分类任务，通过AL选择最有价值的样本标注，同时利用FlexMatch从大量未标注数据中学习。

**核心设计思想**：AL和SSL并非简单串联，而是通过共享模型状态形成正反馈循环——每轮AL查询后，SSL利用当前模型对未标注数据生成伪标签，扩充训练集，提升模型能力，进而影响下一轮AL的查询决策。

## 2. 核心技术组件

### 2.1 FlexMatch动态阈值机制

**问题**：传统FixMatch使用统一置信度阈值τ=0.95，在长尾分布下尾类样本难以达到该阈值，导致伪标签严重偏向头类。

**解决方案**：FlexMatch通过EMA（指数移动平均）跟踪每个类别的学习状态，动态调整各类别的置信度阈值。

**数学形式化**：

给定类别c在第t个batch的学习效果估计：

$$\hat{a}_c^{(t)} = \frac{1}{|B_t^c|} \sum_{i \in B_t^c} \mathbb{1}[\max_j p_w(x_i, j) \geq \tau_{base}]$$

EMA更新：

$$a_c^{(t)} = \alpha \cdot a_c^{(t-1)} + (1 - \alpha) \cdot \hat{a}_c^{(t)}$$

其中α=0.9为EMA动量。

自适应阈值：

$$\tau_c = \max(\tau_{min}, \tau_{base} \cdot \frac{a_c^{(t)}}{\max_{c'} a_{c'}^{(t)}})$$

其中τ_base=0.95，τ_min=0.70。

**实现代码**：

```python
class FlexMatchTracker:
    def __init__(self, n_classes, threshold=0.95, ema_momentum=0.9, min_threshold=0.70):
        self.classwise_acc = torch.zeros(n_classes)  # EMA状态
        self.current_thresholds = np.full(n_classes, threshold)
    
    def update_per_batch(self, weak_probs):
        max_probs, pseudo_labels = weak_probs.max(dim=1)
        batch_exceeds = (max_probs >= self.base_threshold).float()
        
        for c in range(self.n_classes):
            mask = pseudo_labels == c
            if mask.sum() > 0:
                class_effect = batch_exceeds[mask].mean()
                # EMA更新：α=0.9平滑单batch统计噪声
                self.classwise_acc[c] = (
                    self.ema_momentum * self.classwise_acc[c] 
                    + (1 - self.ema_momentum) * class_effect
                )
        
        # 重新计算阈值
        max_effect = self.classwise_acc.max().item()
        if max_effect > 1e-8:
            for c in range(self.n_classes):
                beta = self.classwise_acc[c].item() / max_effect
                self.current_thresholds[c] = max(
                    self.min_threshold, 
                    self.base_threshold * beta
                )
```

**阈值调整效果**：
- 头类（学习充分）：a_c接近1 → beta接近1 → τ_c ≈ 0.95（保持高质量）
- 尾类（学习不足）：a_c小 → beta小 → τ_c ≈ 0.70（降低门槛，增加伪标签数量）
- 阈值在每个batch实时更新，而非每epoch更新，匹配原始FlexMatch论文

### 2.2 FixMatch一致性正则化

**核心思想**：对同一样本施加不同强度的增强，要求模型对两种增强产生一致的预测。

**双增强管线**：

| 增强类型 | 方法 | 参数 | 目的 |
|----------|------|------|------|
| 弱增强 | RandomCrop(32, padding=4) + HorizontalFlip | - | 生成稳定预测，用于伪标签 |
| 强增强 | RandomCrop(32, padding=4) + HorizontalFlip + RandAugment(num_ops=2, magnitude=10) | - | 增加难度，用于一致性训练 |

**伪标签生成与筛选**：

```python
# 1. 弱增强前向传播（无梯度）
with torch.no_grad():
    logits_w = model(weak_inputs)           # [B, C]
    probs_w = torch.softmax(logits_w, dim=-1)  # [B, C]
    max_probs, pseudo_labels = torch.max(probs_w, dim=-1)  # [B], [B]

# 2. FlexMatch动态阈值筛选
flex_tracker.update_per_batch(probs_w.float().cpu())
current_thresholds = flex_tracker.current_thresholds
threshold_tensor = torch.tensor(
    [current_thresholds[l.item()] for l in pseudo_labels],
    device=device, dtype=torch.float32
)
mask = max_probs.ge(threshold_tensor).float()  # [B], 0或1

# 3. 强增强前向传播 + 一致性损失
logits_s = model(strong_inputs)  # [B, C]
loss_u = (F.cross_entropy(logits_s, pseudo_labels, reduction='none') * mask).mean()
```

**关键设计**：
- 伪标签来自弱增强预测（更稳定），一致性损失来自强增强预测（更鲁棒）
- mask机制：只有置信度超过类别自适应阈值的样本才参与SSL训练
- 未通过阈值的样本不产生梯度，避免噪声伪标签污染模型

### 2.3 SSL损失预热（Warmup）

**问题**：训练初期模型不稳定，伪标签质量差，直接使用高权重的SSL损失会引入大量噪声。

**解决方案**：线性预热SSL损失权重：

```python
# warmup_offset_epochs: 从上一轮AL结束后的epoch继续
ssl_progress_epoch = warmup_offset_epochs + epoch + 1
warmup_factor = min(1.0, ssl_progress_epoch / max(1, CFG.ssl_lambda_u_warmup_epochs))

# 损失组合
loss = loss_x + CFG.ssl_lambda_u * warmup_factor * loss_u
```

其中：
- loss_x：标注数据的交叉熵损失
- loss_u：未标注数据的FixMatch一致性损失
- λ_u = 1.0：SSL损失权重
- warmup_factor：从0线性增长到1，跨越前10个epoch

**预热策略的跨轮延续**：warmup_offset_epochs在AL轮间累积，确保SSL损失在首轮AL后的前几轮逐渐增加，而非每轮从零开始。

### 2.4 AL查询策略

系统支持6种标准AL查询策略和7种创新策略，每轮在SSL训练完成后，使用当前模型对未标注池进行查询：

**标准策略（6种）**：

| 策略 | 数学形式 | 核心思想 |
|------|----------|----------|
| Random | 随机采样 | 基线，无偏好 |
| Entropy | H(x) = -Σ p(c\|x) log p(c\|x) | 选择预测不确定性最高的样本 |
| Margin | p₁ - p₂ | 选择最大与次大概率差最小的样本 |
| CoreSet | min-max距离 | k-center贪婪算法，最大化特征空间覆盖 |
| Badge | ∇_θ ℓ(x) | 梯度嵌入空间中的不确定性+多样性 |
| QBC | Vote Entropy | 5个异构CNN委员会投票熵 |

**创新策略（7种）**：

| 策略 | 核心思想 |
|------|----------|
| Class-Aware Entropy | 熵 + 类别惩罚，低频类样本获得更高权重 |
| Gap-Aware Entropy | 熵 + 分布差距填充，引导选择弥补类别缺口的样本 |
| Adaptive Gap Entropy | 自适应λ的Gap-Aware，λ随偏度自动缩放 |
| Two-Stage Entropy Balance | 两阶段：熵粗筛 + 类别平衡贪心选择 |
| Curriculum Penalty Entropy | 课程学习式惩罚，早期纯熵后期引入类别平衡 |
| Class-Aware Entropy (SSL) | 基于labeled+pseudo联合分布的类别感知 |
| Gap-Aware Entropy (SSL) | 基于labeled+pseudo联合分布的差距填充 |

**Class-Aware Entropy（尾类感知策略，V3改进版）**：

$$\text{score}(x) = \frac{H(p(x))}{\log C} + \lambda_{\text{eff}} \cdot \frac{\text{penalty}(x)}{\max_x \text{penalty}(x)}$$

其中：
- **自适应λ**：$\lambda_{\text{eff}} = \lambda \cdot (1 - \frac{\min_c n_c}{\max_c n_c})$，当数据均衡时λ→0，退化为纯熵
- **软加权惩罚**：$\text{penalty}(x) = \sum_c p(c|x) \cdot \frac{1/\log(n_c + 2)}{\max_c 1/\log(n_c + 2)}$，用概率加权替代硬argmax

**Gap-Aware Entropy（分布差距感知策略）**：

$$\text{score}(x) = \frac{H(p(x))}{\log C} + \lambda \cdot \frac{\sum_c p(c|x) \cdot \text{deficit}(c)}{\max_x \sum_c p(c|x) \cdot \text{deficit}(c)}$$

其中deficit(c) = max(1/C - f_c, 0)，f_c为类别c的已标注频率。

### 2.5 联合分布感知采样（论文创新点1）

**问题**：简单AL+SSL组合中，AL的deficit仅基于labeled分布，不知道SSL已为头类生成大量伪标签，导致AL重复选择头类样本。

**解决方案**：将Class-Aware和Gap-Aware策略扩展为SSL版本，deficit基于labeled + pseudo的联合分布。

**Class-Aware Entropy (SSL版，V3改进)**：

$$\text{Penalty}(c) = \frac{1}{\log(n_c^{\text{labeled}} + n_c^{\text{pseudo}} + 2)}$$

其中n_c^{pseudo}为SSL为类别c生成的伪标签数量。V3改进支持自适应λ和软加权惩罚。

**代码实现**：

```python
def select_class_aware_entropy_ssl(
    probs, pool_idx, n_query, labeled_labels, n_classes,
    pseudo_labels=None, lam=0.5, adaptive_lambda=True, soft_weighting=True
):
    """Class-Aware Entropy SSL版：基于联合分布的类别惩罚"""
    n_select = min(n_query, len(pool_idx))
    probs = np.clip(probs.astype(np.float32), 1e-7, 1.0)

    # 归一化熵
    entropy = -np.sum(probs * np.log(probs), axis=1)
    entropy_norm = entropy / np.log(n_classes)

    # 联合分布：labeled + pseudo
    if pseudo_labels is not None and len(pseudo_labels) > 0:
        joint_labels = np.concatenate([labeled_labels, pseudo_labels])
    else:
        joint_labels = labeled_labels
    joint_counts = np.bincount(joint_labels.astype(int), minlength=n_classes).astype(np.float32)

    # 类别惩罚：n_c越小，惩罚越大
    penalty = 1.0 / np.log(joint_counts + 2.0)
    penalty_max = penalty.max()
    penalty_norm = penalty / penalty_max if penalty_max > 0 else penalty

    # 自适应λ：偏度越大，λ越强
    if adaptive_lambda:
        freq_nonzero = joint_counts[joint_counts > 0]
        skewness = 1.0 - freq_nonzero.min() / (freq_nonzero.max() + 1e-10) if len(freq_nonzero) > 0 else 0.0
        effective_lam = lam * skewness
    else:
        effective_lam = lam

    # 软加权惩罚 vs 硬argmax
    if soft_weighting:
        sample_penalty = (probs * penalty_norm).sum(axis=1)
    else:
        pred_classes = probs.argmax(axis=1)
        sample_penalty = penalty_norm[pred_classes]

    sample_penalty_norm = sample_penalty / (sample_penalty.max() + 1e-10)
    score = entropy_norm + effective_lam * sample_penalty_norm
    top_k = np.argsort(score)[-n_select:]
    return [pool_idx[i] for i in top_k]
```

**Gap-Aware Entropy (SSL版)**：

$$\text{deficit}(c) = \max\left(\frac{1}{C} - f_c^{\text{joint}}, 0\right)$$

其中联合分布频率：

$$f_c^{\text{joint}} = \frac{n_c^{\text{labeled}} + n_c^{\text{pseudo}}}{N^{\text{joint}}}$$

**代码实现**：

```python
def select_gap_aware_entropy_ssl(
    probs, pool_idx, n_query, labeled_labels, n_classes,
    pseudo_labels=None, lam=0.5
):
    """Gap-Aware Entropy SSL版：基于联合分布的deficit填充"""
    n_select = min(n_query, len(pool_idx))
    probs = np.clip(probs.astype(np.float32), 1e-7, 1.0)

    # 归一化熵
    entropy = -np.sum(probs * np.log(probs), axis=1)
    entropy_norm = entropy / np.log(n_classes)

    # 联合分布频率
    if pseudo_labels is not None and len(pseudo_labels) > 0:
        joint_labels = np.concatenate([labeled_labels, pseudo_labels])
    else:
        joint_labels = labeled_labels
    joint_counts = np.bincount(joint_labels.astype(int), minlength=n_classes).astype(np.float32)
    joint_freq = joint_counts / (joint_counts.sum() + 1e-10)

    # deficit = max(1/C - f_c^joint, 0)
    uniform_freq = np.ones(n_classes, dtype=np.float32) / n_classes
    deficit = np.maximum(uniform_freq - joint_freq, 0.0)

    # Gap得分：概率加权deficit（只归一化gap_score，不双重归一化）
    gap_score = (probs * deficit).sum(axis=1)
    gap_max = gap_score.max()
    gap_norm = gap_score / gap_max if gap_max > 0 else gap_score

    score = entropy_norm + lam * gap_norm
    top_k = np.argsort(score)[-n_select:]
    return [pool_idx[i] for i in top_k]
```

**正反馈循环机制**：

```
SSL为头类生成伪标签 → 联合deficit中头类gap被填充
→ AL的gap_score自动偏向尾类 → AL选择尾类样本标注
→ 模型尾类能力提升 → SSL生成更准确的尾类伪标签
→ 联合分布更均衡 → 循环迭代
```

### 2.6 类别自适应置信度阈值（论文创新点2）

**问题**：统一阈值τ=0.95对尾类过于严格，FlexMatch的EMA机制虽然有效，但未直接利用类别分布信息。

**解决方案**：在FlexMatch基础上，引入基于deficit的阈值调整：

$$\tau_c = \tau_{\text{base}} - \alpha \cdot \text{deficit}_{\text{norm}}(c)$$

其中：
- τ_base = 0.95（基础阈值）
- α = 0.25（调整系数）
- deficit_norm(c) = deficit(c) / max_c deficit(c)（归一化deficit）

**代码实现**：

```python
class AdaptiveThresholdTracker:
    """结合FlexMatch EMA和deficit的类别自适应阈值"""
    def __init__(self, n_classes, tau_base=0.95, alpha=0.25, tau_min=0.70):
        self.n_classes = n_classes
        self.tau_base = tau_base
        self.alpha = alpha
        self.tau_min = tau_min

    def compute_thresholds(self, labeled_labels, pseudo_labels=None):
        """根据当前标注分布（和可选的伪标签分布）计算各类别阈值"""
        labeled_counts = np.bincount(labeled_labels.astype(int), minlength=self.n_classes).astype(np.float32)

        if pseudo_labels is not None and len(pseudo_labels) > 0:
            pseudo_counts = np.bincount(pseudo_labels.astype(int), minlength=self.n_classes).astype(np.float32)
            joint_counts = labeled_counts + pseudo_counts
        else:
            joint_counts = labeled_counts

        # deficit = max(1/C - f_c, 0)
        joint_freq = joint_counts / (joint_counts.sum() + 1e-10)
        mean_freq = 1.0 / self.n_classes
        deficit = np.maximum(mean_freq - joint_freq, 0.0)

        # 归一化
        deficit_max = deficit.max()
        if deficit_max > 1e-8:
            deficit_norm = deficit / deficit_max
        else:
            deficit_norm = np.zeros(self.n_classes)

        # 自适应阈值
        thresholds = self.tau_base - self.alpha * deficit_norm
        thresholds = np.maximum(thresholds, self.tau_min)

        return thresholds
```

**阈值效果**：
- 头类：deficit_norm ≈ 0 → τ_c ≈ 0.95（保持高质量）
- 尾类：deficit_norm ≈ 1 → τ_c ≈ 0.70（降低门槛，增加尾类伪标签）

### 2.7 类别加权一致性损失（论文创新点3）

**问题**：标准FixMatch对所有样本的SSL损失权重相同，尾类伪标签数量少但权重不应低于头类。

**解决方案**：对SSL一致性损失按类别加权，尾类伪标签权重更高：

$$\mathcal{L}_u = \frac{1}{|B_u|} \sum_{x \in B_u} w_{\hat{y}(x)} \cdot \mathbb{1}\left[\max p_w(x) \geq \tau_c\right] \cdot H(p_s(x), \hat{y}_w(x))$$

其中类别权重：

$$w_c = \frac{1/C}{n_c^{\text{labeled}} + 1}$$

归一化后均值=1，确保总损失量级不变。

**代码实现**：

```python
def compute_class_weighted_ssl_loss(
    logits_s, pseudo_labels, mask, labeled_labels, n_classes
):
    """类别加权的FixMatch一致性损失"""
    # 计算类别权重：n_c越小，权重越大
    class_counts = np.bincount(labeled_labels.astype(int), minlength=n_classes).astype(np.float32)
    mean_freq = 1.0 / n_classes
    weights = mean_freq / (class_counts + 1.0)  # w_c = (1/C) / (n_c + 1)

    # 归一化使均值=1
    weights = weights / weights.mean()
    weights_tensor = torch.tensor(weights, device=logits_s.device, dtype=torch.float32)

    # 每个样本的权重：根据其伪标签类别
    sample_weights = weights_tensor[pseudo_labels]  # [B]

    # 加权交叉熵损失
    per_sample_loss = F.cross_entropy(logits_s, pseudo_labels, reduction='none')  # [B]
    weighted_loss = (per_sample_loss * sample_weights * mask).mean()

    return weighted_loss
```

**效果**：尾类n_c小 → w_c大 → 尾类伪标签的梯度贡献更大 → 模型更关注尾类学习。

### 2.8 AL+SSL联合训练完整流程

**每轮完整流程**：

```
┌─────────────────────────────────────────────────────────────────┐
│ 第r轮 AL+SSL联合训练                                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│ 阶段1: SSL训练 (n_epochs_base=5轮)                              │
│ ├── 每个batch:                                                  │
│ │   ├── 标注数据前向:                                            │
│ │   │   logits_l = model(x_labeled)                             │
│ │   │   loss_x = CE(logits_l, y_labeled)                        │
│ │   │                                                          │
│ │   ├── 未标注数据前向:                                          │
│ │   │   probs_w = softmax(model(weak_aug(x_unlabeled)))        │
│ │   │   FlexMatch.update_per_batch(probs_w)  # EMA更新阈值      │
│ │   │   τ_c = FlexMatch.get_thresholds()     # 类别自适应阈值    │
│ │   │   mask = max(probs_w) >= τ_c            # 置信度筛选       │
│ │   │   pseudo_labels = argmax(probs_w)       # 伪标签           │
│ │   │   logits_s = model(strong_aug(x_unlabeled))              │
│ │   │   loss_u = w_c · CE(logits_s, pseudo_labels) · mask      │
│ │   │                                                          │
│ │   └── 总损失:                                                  │
│ │       loss = loss_x + λ_u · warmup_factor · loss_u           │
│ │                                                              │
│ ├── FlexMatch EMA: 每batch更新各类别学习状态                      │
│ └── 伪标签诊断: 最后一个epoch统计准确率和覆盖率                     │
│                                                                 │
│ 阶段2: AL查询（使用SSL训练后的模型）                               │
│ ├── 对未标注池预测: probs_pool = model(x_pool)                   │
│ ├── 计算联合分布: f_c^joint = (n_c^labeled + n_c^pseudo) / N    │
│ ├── 应用查询策略:                                                │
│ │   ├── Entropy: score = H(probs_pool)                          │
│ │   ├── Gap-Aware: score = H_norm + λ · Gap_norm               │
│ │   │   其中 Gap(x) = Σ_c p(c|x) · deficit(c)                  │
│ │   │   deficit(c) = max(1/C - f_c^joint, 0)                   │
│ │   └── 其他策略: Margin/CoreSet/Badge/QBC                      │
│ └── 选择top-100样本: selected = argtopk(score, 100)             │
│                                                                 │
│ 阶段3: 更新标注集                                                │
│ ├── labeled_idx += selected                                    │
│ ├── pool_idx -= selected                                       │
│ └── warmup_offset_epochs += n_epochs_base  # 预热跨轮累积       │
│                                                                 │
│ 进入第r+1轮...                                                  │
└─────────────────────────────────────────────────────────────────┘
```

**关键交互点**：
1. **SSL→AL**：SSL训练后的模型状态直接影响AL查询决策，不是用旧模型查询
2. **AL→SSL**：AL选择的样本在下一轮被标注后，影响SSL的loss_x和模型能力
3. **联合分布**：deficit基于labeled+pseudo联合分布，AL自动感知SSL已覆盖的类别
4. **阈值联动**：FlexMatch阈值随模型能力变化动态调整，与deficit阈值叠加
5. **预热累积**：warmup_offset_epochs在AL轮间累积，避免每轮从零预热

## 3. 实验配置

### 3.1 统一配置

| 参数 | 值 | 说明 |
|------|-----|------|
| 数据集 | CIFAR-10 | 10类32×32彩色图像 |
| 模型 | SimpleCNN | 3层卷积(32→64→128)+FC(256→10) |
| n_initial | 100 | 初始标注量（占总数据0.2%） |
| n_query | 100 | 每轮查询量 |
| n_rounds | 10 | AL轮数（最终标注1100） |
| n_epochs_base | 5 | 每轮训练epoch数 |
| 不平衡比率 | ρ∈{1,10,50,100} | 从均衡到极端长尾 |
| 种子 | 42,123,456,789,1024 | 5种子统计 |
| 评估指标 | Macro-F1 | 对尾类敏感 |

### 3.2 SSL配置

| 参数 | 值 | 说明 |
|------|-----|------|
| SSL方法 | FlexMatch | 动态阈值FixMatch |
| 基础阈值τ_base | 0.95 | 头类阈值 |
| 最低阈值τ_min | 0.70 | 尾类阈值下限 |
| EMA动量α | 0.9 | 学习状态平滑系数 |
| λ_u | 1.0 | SSL损失权重 |
| warmup_epochs | 10 | SSL损失预热轮数 |
| 强增强 | RandAugment(num_ops=2, magnitude=10) | 随机增强 |
| 弱增强 | RandomCrop+HorizontalFlip | 标准增强 |

### 3.3 模型架构

#### 3.3.1 SimpleCNN

3层卷积+全连接的轻量级CNN，参数量约1.1M，作为基础实验模型。

| 层 | 输入通道 | 输出通道 | 核大小 | 步长 | 输出尺寸 |
|----|---------|---------|-------|-----|---------|
| Conv1+BN+ReLU+MaxPool | 3 | 32 | 3×3 | 1 | 16×16 |
| Conv2+BN+ReLU+MaxPool | 32 | 64 | 3×3 | 1 | 8×8 |
| Conv3+BN+ReLU+MaxPool | 64 | 128 | 3×3 | 1 | 4×4 |
| FC1+ReLU+Dropout | 2048 | 256 | - | - | 256 |
| FC2 | 256 | 10 | - | - | 10 |

**参数量计算**：
- Conv1: 3×32×3×3 + 32(bias) = 896
- Conv2: 32×64×3×3 + 64(bias) = 18,496
- Conv3: 64×128×3×3 + 128(bias) = 73,856
- FC1: 2048×256 + 256 = 524,544
- FC2: 256×10 + 10 = 2,570
- BN参数: 32×2 + 64×2 + 128×2 = 448
- **总计: ~620K参数**

#### 3.3.2 ResNet-18

残差网络，参数量约11.2M，用于验证策略在深层网络上的表现。基于torchvision的resnet18实现，修改最后的全连接层适配CIFAR-10的10类输出。

**核心结构**：Stem(7×7 Conv + MaxPool) → Layer1-4(各2×BasicBlock) → AvgPool → FC(512→10)

**BasicBlock**：Conv3×3 → BN → ReLU → Conv3×3 → BN → (+shortcut) → ReLU

**与SimpleCNN的对比**：

| 特性 | SimpleCNN | ResNet-18 |
|------|-----------|-----------|
| 参数量 | ~620K | ~11.2M |
| 深度 | 3层卷积 | 18层（含残差） |
| 特征维度 | 2048 (128×4×4) | 512 |
| 残差连接 | 无 | 有（跳跃连接） |
| 适用场景 | 快速验证 | 高精度需求 |
| 训练速度 | 快 | 慢约3-5倍 |

> 完整的ResNet-18架构解析（含逐层维度变化、参数量计算、BasicBlock详解、CIFAR-10适配分析）见 [ResNet18_架构解析.md](ResNet18_架构解析.md)。

## 4. 实验结果

### 4.1 AL+SSL联合策略在CIFAR-10上的表现

| 策略 | ρ=1 | ρ=10 | ρ=50 | ρ=100 |
|------|-----|------|------|-------|
| Random | 0.4393±0.005 | 0.3583±0.019 | 0.2300±0.011 | 0.1886±0.012 |
| Entropy | 0.4152±0.017 | 0.3639±0.024 | 0.2922±0.019 | 0.2480±0.026 |
| Margin | 0.4627±0.007 | 0.3605±0.024 | 0.2623±0.036 | 0.2267±0.015 |
| Badge | 0.4484±0.016 | 0.3772±0.035 | 0.2764±0.020 | 0.2204±0.016 |
| CoreSet | 0.4087±0.013 | 0.3371±0.021 | 0.2519±0.028 | 0.2066±0.017 |
| QBC | 0.4602±0.006 | 0.3570±0.046 | 0.2403±0.013 | 0.2043±0.017 |

> 全监督基线（FlexMatch SSL）：ρ=1时F1=0.8553，ρ=10时F1=0.7646，ρ=50时F1=0.6298，ρ=100时F1=0.5590。

### 4.2 伪标签质量分析

FlexMatch的动态阈值机制在长尾分布下的效果：

- ρ=1（均衡）：各类别阈值均接近0.95，伪标签数量少但质量高
- ρ=100（极端长尾）：头类阈值≈0.95，尾类阈值≈0.70，尾类伪标签数量增加但质量下降

**权衡**：降低阈值增加尾类伪标签数量的同时，可能引入错误伪标签。FlexMatch通过EMA平滑和beta归一化在两者之间取得平衡。

### 4.3 关键发现

1. **平衡数据（ρ=1）下Margin最优**：Margin（0.4627）表现最佳，QBC（0.4602）和Badge（0.4484）紧随其后。Random（0.4393）排第四，Entropy（0.4152）和CoreSet（0.4087）最差。说明在均衡数据上，多样性采样（Badge）和委员会查询（QBC）仍有优势。

2. **长尾数据（ρ≥50）下Entropy最优**：ρ=50时Entropy（0.2922）比Random（0.2300）提升+27.0%；ρ=100时Entropy（0.2480）比Random（0.1886）提升+31.5%。不确定性采样在长尾分布下的价值随不平衡程度增加而增大。

3. **全监督性能随ρ急剧下降**：从ρ=1的0.8553降至ρ=100的0.5590，验证长尾分布对分类性能的严重挑战。

4. **SSL的双重作用**：
   - 正面：为头类生成大量伪标签，扩大有效训练集
   - 负面：在长尾分布下伪标签偏向头类，FlexMatch缓解但未完全消除

## 5. 与现有方法的对比

| 方法 | 思路 | 低预算效果 | 长尾适应性 | 机制 |
|------|------|-----------|-----------|------|
| LDAM+DRW | 修改损失函数 | ❌ 差 | ❌ 低预算下失效 | margin惩罚 |
| 标准AL | 选择有价值的样本 | ✅ 一般 | ❌ 未考虑类别分布 | 单轮查询 |
| 标准SSL | 利用未标注数据 | ✅ 一般 | ❌ 伪标签偏向头类 | 固定阈值 |
| **AL+SSL联合** | **AL选择+SSL扩充** | **✅ 最优** | **✅ FlexMatch自适应** | **动态阈值+预热** |

## 6. 局限性与未来工作

1. **计算成本**：Badge/CoreSet需要计算梯度嵌入或成对距离，计算量O(n²)
2. **伪标签质量**：即使使用FlexMatch，尾类伪标签的准确性仍低于头类
3. **策略选择**：不同ρ值下最优策略不同，缺乏自动策略选择机制
4. **模型容量**：SimpleCNN参数量约1.1M，可能限制复杂策略的效果

## 7. 代码结构

```
experiments/
├── v8_controlled_fast_al_ssl.py   # 主实验框架（核心训练循环）
│   ├── train_one_round()          # 单轮训练（SSL+AL查询）
│   ├── FlexMatchTracker           # 动态阈值跟踪
│   └── UnlabeledDataset           # 双增强数据集
├── run_al_ssl_100.py             # AL+SSL实验入口
├── run_std_al_100.py             # 标准AL实验（无SSL）
├── run_tail_aware_100.py         # 尾类感知实验
└── run_ldam_baseline.py          # LDAM基线实验

src/
├── ssl_v7_utils.py               # FlexMatch/FixMatch核心实现
├── deep_query_utils.py           # AL查询策略（6标准+7创新）
├── models.py                     # SimpleCNN/ResNet18
└── metrics.py                    # 评估指标
```
