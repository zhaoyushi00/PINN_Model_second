# GRU物理感知模型预测指南

## 🎯 概述

本项目实现了基于**Randomized SVD + GRU + Rollout + Scheduled Sampling + XGBoost残差纠偏 + 物理特征工程**的增强泛化热仿真预测模型。

## 🏗️ 架构特点

### 【增强泛化工程】GRU + Rollout + Scheduled Sampling + XGBoost残差纠偏 + 物理特征工程
- **主干网络**：GRU（门控循环单元）+ 物理特征输入
- **训练策略**：Rollout多步预测 + Scheduled Sampling退火
- **纠偏机制**：XGBoost残差学习 + 物理派生特征
- **泛化增强**：域适应 + 参数变异性处理
- **优势**：强泛化性，长时稳定性好

## 🚀 快速开始

### 1. 训练模型

```bash
# 使用增强泛化工程（推荐）
python main.py --architecture enhanced

# 使用传统工程（对比）
python main.py --architecture traditional

# 查看架构对比
python main.py --show_comparison
```

### 2. 预测使用

#### 方式一：使用简化示例脚本
```bash
# 基本预测（使用默认模型）
python predict_example.py

# 显示使用示例
python predict_example.py --show_examples
```

#### 方式二：使用完整预测脚本

**自动选择工况预测：**
```bash
# 基本预测（自动选择训练数据中的工况）
python predict_gru_physics.py --checkpoint model/gru_physics.pth

# 指定预测参数
python predict_gru_physics.py \
    --checkpoint model/gru_physics.pth \
    --num_predictions 5 \
    --output_dir my_predictions \
    --sequence_length 240 \
    --horizon 12
```

**预设工况预测（推荐）：**
```bash
# 使用预设工况进行预测
python predict_gru_physics.py \
    --checkpoint model/gru_physics.pth \
    --preset low \
    --output_dir preset_predictions

# 可选预设工况：
# --preset low      : RPM=9000, Ambient=15°C (低转速低温)
# --preset medium   : RPM=12000, Ambient=22°C (中转速常温)
# --preset high     : RPM=15000, Ambient=30°C (高转速高温)
# --preset extreme  : RPM=18000, Ambient=25°C (极高转速)
```

**手动指定工况预测：**
```bash
# 指定具体工况进行预测
python predict_gru_physics.py \
    --checkpoint model/gru_physics.pth \
    --rpm 12000 \
    --ambient 25.0 \
    --num_predictions 1 \
    --output_dir manual_predictions

# 查看帮助
python predict_gru_physics.py --help
```

**使用示例脚本：**
```bash
# 测试预设工况功能
python test_presets.py

# 显示预设工况说明和使用示例
python predict_gru_physics.py --preset medium --help
```

## 📊 训练结果示例

```
开始训练GRU物理感知模型 - 输入维度: 5, 序列长度: 240, 预测步长: 12
[Epoch 1/100] train_loss=980063.211493 main_loss=980063.211493 domain_loss=0.000000 TF_ratio=1.000
  VAL: coeff_mse=1044504.241833 T(R2)=26.78% MAE=5.6522 RMSE=6.7884
...
[Epoch 100/100] train_loss=9383.495471 main_loss=9383.495471 domain_loss=0.000000 TF_ratio=0.000
  VAL: coeff_mse=971248.583868 T(R2)=26.74% MAE=5.6548 RMSE=6.7904

GRU物理感知训练完成！最终模型保存到: model/gru_physics.pth
```

## 📁 文件结构

```
├── main.py                          # 主入口，支持两种架构选择
├── train_gru_physics.py             # GRU物理感知训练脚本
├── predict_gru_physics.py           # GRU预测脚本（支持预设工况）
├── test_presets.py                  # 预设工况测试脚本
├── gru_models.py                    # GRU模型定义
├── physics_feature_engineering.py   # 物理特征工程
├── xgboost_residual.py              # XGBoost残差纠偏
├── model/
│   ├── gru_physics.pth              # 训练好的GRU模型
│   ├── gru_training_metrics.csv     # 训练指标日志
│   ├── pod_phi.npy                  # POD基底
│   ├── pod_mean.npy                 # POD均值
│   └── pod_coords.npy               # POD坐标
├── prediction_results/              # 自动选择工况的预测结果
│   ├── prediction_summary.json      # 预测汇总
│   └── prediction_sample_*.npz      # 各样本详细结果
├── preset_predictions/              # 预设工况的预测结果
│   └── prediction_sample_*.npz      # 预设工况详细结果
├── manual_predictions/              # 手动指定工况的预测结果
│   └── prediction_manual_*.npz      # 手动工况详细结果
└── data/snapshots/                  # 快照数据
```

## 🔧 预测结果

### 输出文件说明

**prediction_summary.json** - 预测结果汇总
```json
{
  "total_samples": 3,
  "model_checkpoint": "model/gru_physics.pth",
  "input_dim": 5,
  "sequence_length": 240,
  "horizon": 12,
  "samples": [
    {
      "sample_id": 1,
      "rpm": 9000,
      "ambient": 22.9,
      "coeffs_shape": [12, 5],
      "temperatures_shape": [12, 3600],
      "temp_range": [20.5, 85.2]
    }
  ]
}
```

**prediction_sample_*.npz** - 单个样本详细结果
```python
# 加载自动选择工况的结果
import numpy as np
data = np.load('prediction_results/prediction_sample_1.npz')
pred_coeffs = data['pred_coeffs']          # [horizon, r]
pred_temperatures = data['pred_temperatures']  # [horizon, N]
rpm = data['rpm']                          # 转速
ambient = data['ambient']                  # 环境温度
```

**prediction_preset_*.npz** - 预设工况的结果
```python
# 加载预设工况的结果
import numpy as np
data = np.load('preset_predictions/prediction_sample_1.npz')
pred_coeffs = data['pred_coeffs']          # [horizon, r]
pred_temperatures = data['pred_temperatures']  # [horizon, N]
rpm = data['rpm']                          # 转速
ambient = data['ambient']                  # 环境温度
```

**prediction_manual_*.npz** - 手动指定工况的结果
```python
# 加载手动指定工况的结果
import numpy as np
data = np.load('manual_predictions/prediction_manual_rpm12000_ambient25.0.npz')
pred_coeffs = data['pred_coeffs']          # [horizon, r]
pred_temperatures = data['pred_temperatures']  # [horizon, N]
rpm = data['rpm']                          # 转速
ambient = data['ambient']                  # 环境温度
input_shape = data['input_sequence_shape'] # 输入序列形状
```

## ⚙️ 核心技术特性

### 1. **Rollout训练**
- 多步预测一致性训练，避免长时漂移
- 教师强制逐步退火，提升自回归能力

### 2. **物理特征工程**
- 基于工程公式的派生物理特征计算
- 考虑环境温度对热源和散热的影响
- 参数变异性注入，增强泛化性

### 3. **XGBoost残差纠偏**
- 学习GRU预测的残差模式
- 多输出回归，支持同时纠偏多个POD系数
- 在线增量学习，支持模型持续优化

### 4. **域适应**
- 基于热负荷的自动域聚类
- 域适应损失，最小化跨域特征差异

## 🎛️ 参数配置

### 【增强泛化工程】推荐参数
```python
# 模型参数
HIDDEN_DIM = 128           # GRU隐藏维度
NUM_LAYERS = 2             # GRU层数
DROPOUT = 0.1              # Dropout概率

# 训练参数
SEQ_LEN = 240              # 输入序列长度
HORIZON = 12               # 预测步长
ROLLOUT_HORIZON = 12       # Rollout步长
SCHEDULED_SAMPLING_EPOCHS = 20  # 教师强制退火周期

# 物理特征参数
PHYSICS_VARIATION_LEVEL = 'normal'  # 物理参数变异水平
DOMAIN_ADAPTATION_WEIGHT = 0.1      # 域适应损失权重

# XGBoost残差参数
XGB_ESTIMATORS = 200       # XGBoost树数量
XGB_DEPTH = 6              # 树最大深度
XGB_LR = 0.05              # 学习率
```

## 🔍 故障排除

### 常见问题

1. **模型文件不存在**
   ```bash
   # 先运行训练
   python main.py --architecture enhanced
   ```

2. **内存不足**
   ```bash
   # 减少批次大小或序列长度
   python main.py --architecture enhanced  # 在train_gru_physics.py中修改BATCH_SIZE
   ```

3. **数据文件缺失**
   - 确保`data/snapshots/`目录存在所需的CSV文件
   - 检查POD文件是否完整生成

4. **预测结果异常**
   - 检查输入序列长度是否足够（默认240）
   - 确认工况参数在训练范围内（RPM 6k-22k, Ambient 15-35°C）
   - 对于手动指定的工况，如果超出训练范围，结果可能不准确

5. **预设工况预测问题**
   - 使用`--preset low/medium/high/extreme`选择预设工况
   - 预设工况会自动设置对应的RPM和环境温度
   - 如果预设工况超出训练范围，预测结果可能不准确

6. **手动工况预测问题**
   - 确保`--rpm`和`--ambient`参数都已指定
   - 如果工况在训练数据中不存在，将使用虚拟输入序列
   - 虚拟预测的结果仅供参考，不代表实际物理行为

## 📈 性能监控

训练过程中会生成以下监控文件：

- **`model/gru_training_metrics.csv`** - 详细训练指标
- **`prediction_results/prediction_summary.json`** - 预测结果汇总

可以使用pandas等工具分析这些文件：

```python
import pandas as pd

# 分析训练指标
metrics = pd.read_csv('model/gru_training_metrics.csv')
print(metrics[['epoch', 'train_loss', 'val_T_r2']].tail())

# 分析预测结果
import json
with open('prediction_results/prediction_summary.json', 'r') as f:
    summary = json.load(f)
    print(f"预测样本数: {summary['total_samples']}")
```

## 🤝 贡献指南

欢迎提交Issue和Pull Request来改进这个项目！

---

**🎉 享受您的热仿真预测之旅！**
