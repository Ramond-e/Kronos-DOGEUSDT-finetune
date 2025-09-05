# DOGE微调项目进度记录

## 📋 项目概述

**目标**: 使用DOGE/USDT 1小时K线数据对预训练的Kronos模型进行微调

**数据来源**: 从Binance获取的DOGEUSDT历史数据 (2020-07-10 到 2025-09-03)

**预训练模型**: Kronos-small (NeoQuasar/Kronos-small)

## ✅ 已完成的工作

### 阶段1: 项目准备和清理
- [x] 删除不必要的项目文件夹 (examples, figures, webui, get_DOGEUSDT_data等)
- [x] 保留核心模型文件 (model/, finetune/, requirements.txt等)
- [x] 创建DOGE专用目录结构
- [x] **清理不必要的文件** (train_predictor.py, train_tokenizer.py, qlib相关文件等)

### 阶段2: 数据预处理
- [x] **创建 `doge_preprocess.py`** - 数据预处理脚本
- [x] **创建 `doge_config.py`** - DOGE专用配置文件
- [x] **创建 `doge_dataset.py`** - 数据集类
- [x] **数据标准化优化** - 使用中位数和MAD方法

#### 数据处理详情

**原始数据统计**:
- 数据文件: `doge_data/raw/dogeusdt_1h_all_klines.csv`
- 数据条数: 45,149条记录
- 时间范围: 2020-07-10 09:00:00 到 2025-09-03 13:00:00
- 数据列: ['open_time', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_asset_volume', 'number_of_trades', 'taker_buy_base', 'taker_buy_quote', 'ignore']

**数据预处理步骤**:
1. **时间戳转换**: 将毫秒时间戳转换为datetime
2. **特征选择**: 选择 ['timestamps', 'open', 'high', 'low', 'close', 'volume']
3. **添加amount列**: volume * 平均价格
4. **时间特征生成**: minute, hour, weekday, day, month
5. **滑动窗口创建**: 400历史 + 120预测
6. **数据分割**: 训练70% + 验证15% + 测试15%
7. **改进的Z-score标准化**: 使用中位数和MAD，限制异常值到[-5, 5]

**处理结果**:
- 训练集: 31,240个窗口 (976批次)
- 验证集: 6,695个窗口 (210批次)
- 测试集: 6,695个窗口 (210批次)
- 每个窗口: 400历史 + 120预测
- 特征维度: 6个价格/成交量特征 + 5个时间特征

**生成的文件**:
- `doge_data/processed/doge_processed.csv` (3.9MB)
- `doge_data/splits/train_windows.pkl` (1.2GB)
- `doge_data/splits/val_windows.pkl` (260MB)
- `doge_data/splits/test_windows.pkl` (260MB)

### 阶段3: 配置和数据集
- [x] **配置管理**: 创建DOGE专用配置类
- [x] **数据集类**: 实现DOGEDataset和DOGEDataLoader
- [x] **数据测试**: 验证数据加载和批次生成
- [x] **学习率优化**: 降低学习率以提高稳定性

**配置详情**:
- 预训练模型: NeoQuasar/Kronos-small
- 批次大小: 32 (适合8GB显存)
- 学习率: 2e-6 (大幅降低以提高稳定性)
- 训练轮数: 20 (5轮tokenizer + 15轮predictor)
- 设备: cuda:0
- 梯度裁剪: 3.0 (增加阈值)

## 🔄 当前状态

**最后执行时间**: 2025-09-04
**当前阶段**: 🎉 **项目完全完成** - 模型已成功部署到Hugging Face和GitHub
**项目状态**: ✅ **已完成所有核心任务**

## ✅ 已完成的任务

### 阶段4: 模型训练 ✅
- [x] **创建 `doge_train.py`** - 两阶段训练脚本 ✅
  - [x] 实现tokenizer微调 ✅
  - [x] 实现predictor微调 ✅
  - [x] 实现训练监控和早停 ✅
  - [x] 实现模型保存和加载 ✅
- [x] **完成完整训练** - 7小时训练过程 ✅
  - [x] Tokenizer训练: 5轮 (最佳: Epoch 5, 验证损失: 0.047491) ✅
  - [x] Predictor训练: 7轮 (最佳: Epoch 4, 验证损失: 1.971007) ✅
  - [x] 过拟合检测: 自动早停 ✅

### 阶段5: 评估和可视化 ✅
- [x] **创建 `visualize_training_progress.py`** - 训练进度可视化 ✅
  - [x] 实现训练损失曲线 ✅
  - [x] 实现过拟合区域标注 ✅
  - [x] 实现数值标签显示 ✅
  - [x] 生成训练报告和图表 ✅

### 阶段6: 模型部署 ✅
- [x] **Hugging Face Hub部署** ✅
  - [x] DOGE Tokenizer: [Ramond-e/doge-kronos-tokenizer](https://huggingface.co/Ramond-e/doge-kronos-tokenizer) (15.9MB) ✅
  - [x] DOGE Predictor: [Ramond-e/doge-kronos-predictor](https://huggingface.co/Ramond-e/doge-kronos-predictor) (99.0MB) ✅
- [x] **GitHub代码仓库** ✅
  - [x] 项目地址: [Ramond-e/Kronos-DOGEUSDT-finetune](https://github.com/Ramond-e/Kronos-DOGEUSDT-finetune) ✅
  - [x] 完整的README文档 ✅
  - [x] 分支迁移: master → main ✅
- [x] **项目文档** ✅
  - [x] 详细使用指南 ✅
  - [x] 模型下载链接 ✅
  - [x] 训练结果可视化 ✅

### 阶段7: 项目清理 ✅
- [x] **代码清理** ✅
  - [x] 删除临时文件 (progress.txt, TRAINING_RECOVERY_GUIDE.md等) ✅
  - [x] 删除测试脚本 (test_training_env.py, quick_train_test.py等) ✅
  - [x] 优化.gitignore配置 ✅
- [x] **Git仓库优化** ✅
  - [x] 分支合并 (master → main) ✅
  - [x] 远程仓库清理 ✅

## 🛠️ 技术细节

### 数据格式
```python
# 单个样本格式
{
    'hist_features': torch.Size([400, 6]),      # 历史特征
    'hist_timestamps': torch.Size([400, 5]),   # 历史时间特征
    'pred_features': torch.Size([120, 6]),     # 预测目标
    'pred_timestamps': torch.Size([120, 5]),   # 预测时间特征
    'window_idx': torch.tensor(idx),           # 窗口索引
    'stats': {                                 # 标准化统计信息
        'price_mean': torch.tensor(...),
        'price_std': torch.tensor(...),
        'vol_mean': torch.tensor(...),
        'vol_std': torch.tensor(...)
    }
}
```

### 批次格式
```python
# 批次数据格式
{
    'hist_features': torch.Size([32, 400, 6]),    # [batch_size, seq_len, n_features]
    'hist_timestamps': torch.Size([32, 400, 5]),  # [batch_size, seq_len, n_time_features]
    'pred_features': torch.Size([32, 120, 6]),    # [batch_size, pred_len, n_features]
    'pred_timestamps': torch.Size([32, 120, 5]),  # [batch_size, pred_len, n_time_features]
    'window_idx': torch.Size([32]),               # [batch_size]
    'stats': {...}                                # 标准化统计信息
}
```

### 特征说明
**价格特征**: ['open', 'high', 'low', 'close']
**成交量特征**: ['volume', 'amount']
**时间特征**: ['minute', 'hour', 'weekday', 'day', 'month']

## 🎯 训练策略

### 两阶段训练
1. **阶段1 - Tokenizer微调** (5轮)
   - 微调KronosTokenizer以适应DOGE数据分布
   - 学习率: 5e-6 (降低)
   - 目标: 优化量化参数

2. **阶段2 - Predictor微调** (15轮)
   - 微调Kronos模型进行预测
   - 学习率: 2e-6 (大幅降低)
   - 目标: 优化预测性能

### 训练配置
- 优化器: AdamW
- 学习率调度: 余弦衰减
- 正则化: Dropout 0.1, 权重衰减 0.01
- 梯度裁剪: 3.0 (增加阈值)
- 混合精度: 启用
- 早停: 验证损失5轮不下降

## 📁 项目结构

```
finetune/
├── doge_data/                    # DOGE数据目录
│   ├── raw/                      # 原始数据
│   │   └── dogeusdt_1h_all_klines.csv
│   ├── processed/                # 处理后数据
│   │   └── doge_processed.csv
│   └── splits/                   # 数据分割
│       ├── train_windows.pkl
│       ├── val_windows.pkl
│       └── test_windows.pkl
├── doge_outputs/                 # 输出目录
│   ├── tokenizer/                # 微调后的tokenizer
│   ├── predictor/                # 微调后的predictor
│   └── logs/                     # 训练日志
├── utils/                        # 工具函数
├── doge_preprocess.py            # 数据预处理脚本 ✅
├── doge_config.py                # 配置文件 ✅
├── doge_dataset.py               # 数据集类 ✅
├── doge_train.py                 # 训练脚本 ✅
├── test_training_env.py           # 环境测试脚本 ✅
├── quick_train_test.py           # 快速训练测试 ✅
├── check_data_distribution.py    # 数据分布检查 ✅
└── DOGE_PROGRESS.md              # 进度记录 (本文件)
```

## 🔍 关键决策记录

### 1. 不使用qlib
**原因**: 已有干净的CSV数据，qlib会增加不必要的复杂性
**替代方案**: 直接使用pandas进行数据处理

### 2. 选择Kronos-small
**原因**: 适合8GB显存，训练效率高
**配置**: 512维度，12层，8头注意力

### 3. 数据标准化策略改进
**原始方法**: Z-score标准化 (均值和标准差)
**改进方法**: 使用中位数和MAD (中位数绝对偏差)
**优势**: 对异常值更鲁棒，减少异常值对统计量的影响
**限制**: 异常值限制到[-5, 5]范围
**效果**: 完全消除异常值问题，训练稳定性大幅提升

### 4. 滑动窗口设计
**历史窗口**: 400个时间点 (约16.7天)
**预测窗口**: 120个时间点 (5天)
**重叠**: 最大化数据利用率

### 5. 学习率优化
**原始学习率**: 1e-5
**优化后学习率**: 
- Tokenizer: 5e-6 (降低50%)
- Predictor: 2e-6 (降低80%)
**原因**: 微调需要更小的学习率，避免破坏预训练知识

## 📊 性能预期

### 数据质量
- ✅ 无缺失值
- ✅ 时间连续性良好
- ✅ 价格范围合理 (历史最低0.002455)
- ✅ 数据量充足 (45K+记录)
- ✅ **异常值完全消除** (0.00%)
- ✅ **数据分布稳定** (标准差~2.13)

### 训练预期
- 训练时间: 预计2-4小时 (RTX 4060 Laptop)
- 内存使用: 约8GB显存
- 收敛轮数: 预计10-15轮
- **训练稳定性**: 损失稳定下降，批次间波动小

## 🎯 **项目最终成果**

### **训练结果**
| 模型 | 最佳轮次 | 验证损失 | 训练时间 | 模型大小 |
|------|----------|----------|----------|----------|
| **DOGE Tokenizer** | Epoch 5 | 0.047491 | ~30分钟/轮 | 15.9MB |
| **DOGE Predictor** | Epoch 4 | 1.971007 | ~1小时/轮 | 99.0MB |

### **部署链接**
- 🤗 **Tokenizer**: [Ramond-e/doge-kronos-tokenizer](https://huggingface.co/Ramond-e/doge-kronos-tokenizer)
- 🤗 **Predictor**: [Ramond-e/doge-kronos-predictor](https://huggingface.co/Ramond-e/doge-kronos-predictor)
- 📚 **代码仓库**: [Ramond-e/Kronos-DOGEUSDT-finetune](https://github.com/Ramond-e/Kronos-DOGEUSDT-finetune)

## 🎉 **项目完成状态**

**完成度**: 100% ✅  
**项目状态**: 已完全完成并成功部署 

## 📈 最新改进成果

### 数据标准化改进效果
| 指标 | 改进前 | 改进后 | 改善程度 |
|------|--------|--------|----------|
| **异常值比例** | 0.25% - 0.47% | 0.00% | ✅ 完全消除 |
| **数据范围** | -1.11 ~ 51.66 | -2.38 ~ 5.00 | ✅ 大幅压缩 |
| **损失趋势** | 递增 (不稳定) | 递减 (稳定) | ✅ 完美修复 |
| **训练时间** | 43.61秒/批次 | 42.92秒/批次 | ✅ 略有提升 |

### 训练环境验证
- ✅ 所有核心文件导入成功
- ✅ 数据加载器正常工作
- ✅ 模型加载成功
- ✅ 训练流程稳定
- ✅ 梯度范数合理
- ✅ 内存使用正常

---

**最后更新**: 2025-09-5

