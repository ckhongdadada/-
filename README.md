# 低标注预算下长尾分布的主动学习与半监督学习联合策略研究

## 项目概述

本项目研究在低标注预算（n_init=100, n_query=100, n_rounds=10）下，针对CIFAR-10长尾分布图像分类任务的主动学习（AL）与半监督学习（SSL）联合策略。

## 统一实验配置

| 参数 | 值 | 说明 |
|------|-----|------|
| 数据集 | CIFAR-10 | 10类图像分类 |
| 模型 | SimpleCNN | 3层卷积+FC，参数量约1.1M |
| n_initial | 100 | 初始标注量 |
| n_query | 100 | 每轮查询量 |
| n_rounds | 10 | AL轮数（最终标注1100） |
| 不平衡比率 | ρ∈{1,5,10,20,50,100} | 从均衡到极端长尾 |
| 种子 | 42,123,456,789,1024 | 5种子统计 |
| 评估指标 | Macro-F1 | 对尾类敏感 |

## 环境依赖

- Python >= 3.10
- PyTorch >= 2.0

```bash
pip install -r requirements.txt
```

可选依赖（扩展研究模块需要，本论文核心实验不需要）：`transformers`, `lightgbm`, `xgboost`。详见 `requirements.txt` 中的注释说明。

## 文件夹结构

```
机器学习—图像分类-期末汇报/
├── README.md                    # 本文件
├── requirements.txt             # 环境依赖
├── 期末论文主干.md               # 论文草稿
├── AL_SSL_技术报告.md            # 技术报告
├── src/                         # 核心代码
│   ├── __init__.py
│   ├── al_utils.py              # AL工具函数
│   ├── models.py                # SimpleCNN等模型
│   ├── metrics.py               # 评估指标
│   ├── ssl_utils.py             # SSL工具
│   ├── ssl_v7_utils.py          # 长尾SSL工具（FlexMatch等）
│   ├── deep_query_utils.py      # AL查询策略（10+种）
│   ├── visualize_utils.py       # 可视化（GradCAM等）
│   ├── boundary_aware.py        # (扩展) 边界感知AL
│   ├── calibrated_query.py      # (扩展) 校准不确定性采样
│   ├── confirmation_bias.py     # (扩展) 确认偏差抑制
│   ├── dynamic_strategy.py      # (扩展) 动态策略切换
│   └── tabular_models.py        # (扩展) 表格数据模型
├── experiments/                 # 实验脚本
│   ├── v8_controlled_fast_al_ssl.py  # 主实验框架（通用）
│   ├── run_std_al_100.py        # 标准AL策略(100/100)
│   ├── run_tail_aware_100.py    # 尾类感知策略(100/100)
│   ├── run_al_ssl_100.py        # AL+SSL联合实验
│   ├── run_ldam_baseline.py     # LDAM基线实验
│   ├── run_cb_focal_baseline.py # CB/Focal基线实验
│   ├── run_tml_validation.py    # TML模型验证
│   └── run_final_scheduler.py   # 综合实验调度器
├── output/                      # 实验结果（全部已完成）
│   ├── std_al/                  # 标准AL (6策略×6ρ×5种子)
│   ├── al_ssl/                  # AL+SSL (6策略×6ρ×5种子)
│   ├── innovative_al_ssl/       # 创新AL+创新SSL (3策略×6ρ×5种子)
│   ├── ldam_baseline/           # LDAM基线
│   ├── cb_focal_baseline/       # CB/Focal基线
│   ├── progressive_ssl_full/    # 渐进式SSL
│   ├── resnet18_full/           # ResNet-18验证
│   ├── cifar100/                # CIFAR-100交叉验证
│   └── tml_validation/          # TML模型验证
├── figures/                     # 论文图表（8张）
├── scripts/                     # 辅助脚本
└── data/                        # 数据集
```

### 扩展研究模块说明

`src/` 中标注为 **(扩展)** 的模块是通用框架 `v8_controlled_fast_al_ssl.py` 为其他研究方向提供的扩展模块，**未用于本论文的核心实验**：

| 模块 | 研究方向 | 说明 |
|------|---------|------|
| `boundary_aware.py` | 边界感知AL | DIRECT-style边界距离策略 |
| `calibrated_query.py` | 校准不确定性 | 温度缩放校准的Entropy/Margin |
| `confirmation_bias.py` | 确认偏差抑制 | 噪声感知查询策略 |
| `dynamic_strategy.py` | 动态策略切换 | 基于能力的自适应策略选择 |
| `tabular_models.py` | 表格数据兼容 | XGBoost/LightGBM/TabNet包装器 |

## 实验完成情况

所有实验均已完成（n_initial=100, n_query=100, n_rounds=10, 5种子）：

| 实验 | 配置 | 状态 | 输出目录 |
|------|------|------|---------|
| 标准AL策略 (6策略×6ρ) | 100/100×10轮 | ✅ | output/std_al/ |
| AL+SSL联合策略 (6策略×6ρ) | 100/100×10轮 | ✅ | output/al_ssl/ |
| 创新AL+创新SSL (3策略×6ρ) | 100/100×10轮 | ✅ | output/innovative_al_ssl/ |
| 消融: 创新AL+基础SSL | 100/100×10轮 | ✅ | output/innovative_al_ssl_basic/ |
| 消融: 基础AL+创新SSL | 100/100×10轮 | ✅ | output/al_ssl_innovative/ |
| LDAM基线 | 全量数据×3种子 | ✅ | output/ldam_baseline/ |
| CB/Focal基线 | 100/100×3种子 | ✅ | output/cb_focal_baseline/ |
| 渐进式SSL | 6配置×4策略×3种子 | ✅ | output/progressive_ssl_full/ |
| ResNet-18验证 | 全策略×全ρ×3种子 | ✅ | output/resnet18_full/ |
| CIFAR-100交叉验证 | 1种子+额外种子 | ✅ | output/cifar100/ |
| TML模型验证 (LR/RF) | 5策略×6ρ×3种子 | ✅ | output/tml_validation/ |
| 统计检验 | 配对t+Cohen's d | ✅ | figures/07_statistical_tests.json |

## 论文章节与数据对应

| 章节 | 需要的实验 | 数据来源 |
|------|-----------|---------|
| §4.1 标准AL在分布谱系上的表现 | 6策略×5种子×6个ρ值 | ✅ output/std_al/ |
| §4.4 尾类感知策略实验 | 5策略×5种子×3个ρ值 | ✅ output/tail_aware_100/ |
| §5.4 AL+SSL联合策略实验 | 4策略×5种子×4个ρ值 | ✅ output/al_ssl/ |
| §6.1 vs LDAM+DRW基线 | CE/LDAM×Full/AL | ✅ output/ldam_baseline/ |
| §6.4 统计检验 | 基于标准AL数据 | ✅ figures/07_statistical_tests.json |
