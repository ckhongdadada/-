# 项目整理记录

## 操作时间
2026-06-11

## 操作内容

### 1. 数据验证与问题发现

通过 `verify_paper_data.py` 和 `summarize_all.py` 脚本，对3个项目目录中的CIFAR-10实验数据进行了全面验证，发现以下问题：

| 问题 | 详情 |
|------|------|
| 论文§4.1.1数据不存在 | 论文声称6策略×8种子(n_initial=100)，实际无此数据 |
| 实验配置不统一 | 标准AL和尾类感知用500/500，AL+SSL和LDAM用100/100 |
| 尾类感知混入SSL | 旧版尾类感知实验使用了FixMatch，与标准AL不可直接对比 |
| 标准AL策略不全 | 500/500配置只有3策略(random/entropy/margin)，缺少Badge/CoreSet/QBC |

### 2. 统一配置决策

经用户确认，所有实验统一为 **100/100配置**：
- n_initial=100, n_query=100, n_rounds=10
- 种子: 42, 123, 456, 789, 1024（5种子）
- 模型: SimpleCNN
- 评估: Macro-F1

### 3. 文件复制

从 `C:\Users\28414\Desktop\主动学习期末汇报版本\主动学习v8` 复制以下文件：

**src/（8个核心模块）：**
| 文件 | 来源 | 说明 |
|------|------|------|
| `__init__.py` | src/ | 模块入口，导出所有公共接口 |
| `al_utils.py` | src/ | AL工具函数(subsample_pool, select_typiclust, aggregate) |
| `models.py` | src/ | 模型定义(SimpleCNN, TextMLPClassifier等) |
| `metrics.py` | src/ | 评估指标(compute_metrics, format_mean_std等) |
| `ssl_utils.py` | src/ | SSL工具(PseudoDataset, apply_pseudo_labels) |
| `ssl_v7_utils.py` | src/ | 长尾SSL工具(FixMatch, FlexMatch, make_longtail_indices) |
| `deep_query_utils.py` | src/ | 查询策略(Badge, CoreSet, QBC, Margin等) |
| `visualize_utils.py` | src/ | 可视化工具 |

**experiments/（7个脚本）：**
| 文件 | 来源 | 说明 |
|------|------|------|
| `v8_controlled_fast_al_ssl.py` | experiments/ | 主实验框架，支持多种数据集和策略 |
| `v8_phase_transition.py` | experiments/ | 相变实验框架，被尾类感知脚本依赖 |
| `run_ldam_baseline.py` | experiments/ | LDAM-DRW基线实验 |
| `run_tml_validation.py` | experiments/ | 传统机器学习模型验证 |
| `run_tail_aware.py` | innovation_d4_tail_aware/ | 尾类感知策略(旧版500/500+SSL) |
| `run_std_al_100.py` | **新建** | 标准AL策略(100/100配置) |
| `run_tail_aware_100.py` | **新建** | 尾类感知策略(100/100配置，无SSL) |

**output/（已有数据）：**
| 目录 | 来源 | 配置 | 状态 |
|------|------|------|------|
| `al_ssl_rho1/` | output/al_ssl_rho1/ | 100/100, 有SSL | ✅ 可用 |
| `al_ssl_rho10/` | output/al_ssl_rho10/ | 100/100, 有SSL | ✅ 可用 |
| `al_ssl_rho50/` | output/al_ssl_rho50/ | 100/100, 有SSL | ✅ 可用 |
| `al_ssl_joint/` | output/al_ssl_joint/ | 100/100, 有SSL(ρ=100) | ✅ 可用 |
| `ldam_baseline/` | output/ldam_baseline/ | 100/100 | ✅ 可用 |
| `tail_aware/` | innovation_d4_tail_aware/output/ | 500/500+SSL | ⚠️ 配置不匹配，仅供参考 |

**其他：**
| 文件 | 来源 | 说明 |
|------|------|------|
| `期末论文主干.md` | 主动学习期末汇报版本/ | 论文草稿 |
| `README.md` | **新建** | 项目说明、实验进度、运行指南 |

### 4. 新建脚本说明

#### `run_std_al_100.py` — 标准AL策略实验
- 调用 `v8_controlled_fast_al_ssl.py` 主框架
- 使用 `--budget-level ultra_low` 实现100/100配置
- 6策略: random, entropy, margin, coreset, badge, qbc
- 6个ρ值: 1, 5, 10, 20, 50, 100
- 5种子: 42, 123, 456, 789, 1024
- 无SSL（`--no-use-ssl`）
- 输出到 `output/std_al/rho{ρ}/`

#### `run_tail_aware_100.py` — 尾类感知策略实验
- 独立脚本，基于 `run_tail_aware.py` 修改
- 5策略: random, entropy, margin, adaptive_gap_entropy, tail_aware_entropy
- 3个ρ值: 10, 50, 100
- 5种子: 42, 123, 456, 789, 1024
- 无SSL（与标准AL公平对比）
- 输出到 `output/tail_aware_100/`
- 自动跳过已完成的实验（断点续跑）
- 运行完毕自动汇总结果

### 5. 未复制的文件（及原因）

| 文件/目录 | 原因 |
|-----------|------|
| `src/boundary_aware.py` | 非图像分类核心模块 |
| `src/calibrated_query.py` | 非图像分类核心模块 |
| `src/confirmation_bias.py` | 非图像分类核心模块 |
| `src/dynamic_strategy.py` | 非图像分类核心模块 |
| `src/tabular_models.py` | 表格数据模型，非图像分类 |
| `output/v8_cifar10_ir*_low_al_5s/` | 500/500配置，不匹配 |
| `experiments/run_*_experiments.py` | 非CIFAR-10核心实验 |
| `experiments/compare_tabular_models.py` | 非图像分类 |
| `scripts/verify_paper_data.py` | 验证脚本，已完成使命 |
| `scripts/summarize_all.py` | 汇总脚本，已完成使命 |
| `scripts/copy_to_target.py` | 临时复制脚本 |

### 6. 数据集复制

| 数据集 | 来源路径 | 目标路径 | 说明 |
|--------|---------|---------|------|
| CIFAR-10 | data/cifar-10-batches-py/ | data/cifar-10-batches-py/ | 主实验数据集(8文件) |

### 7. 待办事项

1. **运行 `run_std_al_100.py`** — 标准AL基线（6策略×5种子×6ρ值）
2. **运行 `run_tail_aware_100.py`** — 尾类感知策略（5策略×5种子×3ρ值）
3. **统计检验** — 基于标准AL数据做配对t检验和Cohen's d
4. **更新论文** — 用新实验数据替换论文§4.1.1和§4.4中的数据
