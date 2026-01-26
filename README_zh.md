# DiffusionIMU4D

[English](./README.md) | [中文](./README_zh.md)

**DiffusionIMU4D** 是一个探索将 **条件扩散模型 (Conditional Diffusion Models)** 应用于惯性导航领域的研究项目。通过将“IMU 到速度”的任务重新建模为条件序列生成问题，我们旨在解决传统回归方法（如 RoNIN, TLIO）中常见的长期漂移和轨迹平滑问题。

## 🚀 项目目标

本项目的主要目标是验证生成式模型是否能够从含噪 IMU 数据中恢复高保真的 3D 速度序列，从而为端到端回归提供一种鲁棒的替代方案。

**主要交付物：**
*   **扩散核心算法**：基于 1D UNet 的 IMU 条件扩散模型。
*   **混合架构**：支持 **残差精修 (Residual Refinement)**（修正粗略估计）和 **端到端生成 (End-to-End Generation)** 两种模式。
*   **评估系统**：完整的轨迹积分与误差分析指标 (ATE, RTE)。

## 🏗️ 系统架构

系统包含三个核心组件：

1.  **条件编码器 (`ResNet1D`)**:
    *   输入原始 IMU 数据（加速度计 + 陀螺仪）。
    *   提取深层时序特征以指导生成过程。
2.  **去噪网络 (`DiffUNet1D`)**:
    *   一个 1D U-Net，负责将随机高斯噪声序列迭代去噪为清晰的速度曲线。
    *   使用 **Cross-Attention / FiLM** 机制注入 IMU 特征。
3.  **物理积分模块**:
    *   将生成的速度序列积分为 3D 轨迹。

### 两种模式
*   **Variant A (残差方案)**: 先由轻量级的 PriorNet (类似 RoNIN) 预测粗略速度，扩散模型生成 **残差** 来补充高频细节。
*   **Variant B (端到端方案)**: 扩散模型直接从纯噪声生成完整的速度序列。

## 🛠️ 安装与使用

### 1. 环境配置
```bash
conda create -n DiffM python=3.9
conda activate DiffM
pip install -r requirements.txt
```

### 2. 数据准备
本项目使用 **RoNIN 数据集**。
*   下载数据并放置在 `data/RoNIN/extracted/`。
*   确保列表文件存在于 `data/RoNIN/lists/`。

### 3. 训练
**训练残差方案 (推荐优先尝试):**
```bash
python train_diff.py --config configs/diffusion_variant_a.yaml
```

**训练端到端方案:**
```bash
python train_diff.py --config configs/diffusion_variant_b.yaml
```

### 4. 评估
```bash
python test_diff.py --config configs/diffusion_variant_a.yaml --checkpoint experiments/checkpoints/your_ckpt.pth
```

## 📄 许可证
MIT License
