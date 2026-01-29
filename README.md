# DiffusionIMU4D

本项目是一个基于 **扩散模型 (Diffusion Models)** 的 IMU 轨迹估计系统，专门设计用于从原始惯性数据 (Gyro + Accel) 中生成或修正 4D 轨迹 (速度 + 时间)。

本项目提供了两种主要的架构变体：
- **Variant A (残差/修正方案)**: 使用 ResNet1D 作为先验网络 (PriorNet) 估计粗略轨迹，随后使用扩散模型预测残差值。
- **Variant B (端到端条件生成方案)**: 将 ResNet1D 仅作为特征编码器，通过特征注入 (FiLM) 指导扩散模型从噪声中生成轨迹。

## 📂 项目结构

```
├── configs/             # 训练配置文件 (Variant A/B, WandB)
├── data/                # 数据加载逻辑与 Dataset 封装
├── models/              # 模型定义 (ResNet1D, DiffUNet1D, DiffusionSystem)
├── utils/               # 工具函数 (几何变换、日志记录、指标计算)
├── train_diff.py        # 扩散模型训练主脚本
├── test_diff.py         # 推理与评估脚本
├── train_ronin.py       # Baseline (RoNIN) 训练脚本
└── requirements.txt     # 项目依赖
```

## 🚀 快速开始

### 1. 环境安装

```bash
pip install -r requirements.txt
```

### 2. 数据准备

请将 RoNIN 数据集放置在 `data/RoNIN` 目录下。结构应如下所示：
```
data/RoNIN/
  ├── extracted/       # 处理后的 .hdf5 文件
  └── lists/          # list_train.txt, list_val.txt 等列表文件
```

### 3. 模型训练

**Variant A (残差方案):**
```bash
python train_diff.py --config configs/diffusion_variant_a.yaml
```

**Variant B (端到端方案):**
```bash
python train_diff.py --config configs/diffusion_variant_b.yaml
```

### 4. 推理与评估

```bash
# 使用训练好的 Checkpoint 进行评估
python test_diff.py --config configs/diffusion_variant_a.yaml --checkpoint experiments/checkpoints/diff_residual_epoch_99.pth
```

## 📊 实验监控 (WandB)

Weights & Biases 的日志记录通过 `configs/wandb.yaml` 进行配置。
你可以在此处修改项目名称或切换离线模式。

```yaml
project_name: "Diffusion4d-Diff"
mode: "online" # 或 "offline"
```

## 📄 开源协议

MIT
