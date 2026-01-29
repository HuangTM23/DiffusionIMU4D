# DiffusionIMU4D

本项目是一个基于 **扩散模型 (Diffusion Models)** 的 IMU 速度序列估计算法，专门设计用于从原始惯性数据 (Gyro + Accel) 中重建高精度的速度序列轨迹。

## 🌟 核心方案

本项目提供了两种不同的扩散模型应用策略，用户可以根据需求选择：

### 方案 A：残差修正 (Residual Refinement)
*   **逻辑**: 使用 ResNet1D 作为先验网络预测一个粗略的速度序列，扩散模型则专注于学习并补偿该先验与真值之间的**残差 (Residual)**。
*   **训练**:
    ```bash
    python train_diff.py --config configs/diffusion_variant_a.yaml
    ```
*   **推理与评估**:
    ```bash
    python test_diff.py --config configs/diffusion_variant_a.yaml --checkpoint experiments/checkpoints/diff_residual_epoch_99.pth
    ```

### 方案 B：端到端条件生成 (End-to-End Conditional)
*   **逻辑**: ResNet1D 仅作为特征编码器，提取 IMU 的深层语义特征。扩散模型以此为**条件 (Condition)**，直接从高斯噪声中去噪生成最终的速度序列。
*   **训练**:
    ```bash
    python train_diff.py --config configs/diffusion_variant_b.yaml
    ```
*   **推理与评估**:
    ```bash
    python test_diff.py --config configs/diffusion_variant_b.yaml --checkpoint experiments/checkpoints/diff_end2end_epoch_99.pth
    ```

---

## 📂 项目结构

```
├── configs/             # 实验配置 (模型超参、WandB 路径等)
├── data/                # 数据加载与处理逻辑 (支持 RoNIN 数据格式)
├── models/              # 模型定义 (ResNet1D, DiffUNet1D, DiffusionSystem)
├── utils/               # 物理积分、几何变换、评价指标等工具类
├── train_diff.py        # 扩散模型训练脚本
├── test_diff.py         # 轨迹重建与精度评估脚本
├── train_ronin.py       # Baseline (ResNet1D) 训练脚本
└── requirements.txt     # 项目依赖环境
```

## 🚀 环境与数据

### 1. 安装依赖
```bash
pip install -r requirements.txt
```

### 2. 数据准备
将 RoNIN 数据集存放至 `data/RoNIN`，确保包含 `extracted` (HDF5文件) 和 `lists` (训练/测试列表)。

### 3. 日志监控 (WandB)
在 `configs/wandb.yaml` 中配置你的项目信息或切换离线模式：
```yaml
project_name: "Diffusion4d-Diff"
entity: ""       # 默认使用本地 wandb login 账号
mode: "online"   # 可选 "offline"
```

## 📄 开源协议
MIT