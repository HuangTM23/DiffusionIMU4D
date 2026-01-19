# IMU 条件扩散模型 (Diffusion4d) 产品指南

> 基于条件扩散生成模型的深度惯性导航研究平台

---

## 1. 核心目标 (Project Goal)

本项目旨在探索和验证**生成式模型（Generative Models）**在惯性导航领域的应用潜力。通过将“IMU 速度估计”重新建模为“条件序列生成问题”，解决传统回归模型（如 LSTM/ResNet）产生的轨迹平滑效应和长期漂移问题。

**主要交付物：**
1.  **核心算法**：实现多种架构的条件扩散模型（Conditional Diffusion Models），能够从含噪 IMU 数据中恢复高精度的 3D 速度序列。
2.  **模块化框架**：一个高度可配置的 PyTorch 训练框架，支持不同模型架构（回归 vs. 扩散）的快速切换和并行对比。
3.  **评估系统**：复用 RoNIN 标准评估指标，提供详细的轨迹误差分析和可视化工具。

---

## 2. 技术路线与方案 (Technical Approach)

### 2.1 核心思想
不同于传统的端到端回归（End-to-End Regression），本项目采用**Velocity-first（速度优先）**策略，并利用扩散模型强大的分布建模能力，学习真实速度序列的概率分布，从而隐式地抑制传感器噪声和漂移。

### 2.2 模型架构变体 (Model Variants)
本项目将并行实现和对比以下方案：

*   **Baseline (基准)**: 复现 **RoNIN (ResNet)**，作为纯回归方法的性能基准。
*   **Variant A (串行 Refinement)**:
    *   利用 Pre-trained PriorNet (RoNIN) 输出粗略速度。
    *   扩散模型在此基础上预测“残差”或进行“精细化去噪”。
*   **Variant B (端到端条件)**:
    *   IMU Encoder 直接提取特征作为条件（Condition）。
    *   扩散模型从纯高斯噪声开始，逐步去噪生成速度序列。

### 2.3 关键网络结构
*   **Backbone**: 采用 **1D U-Net** 架构。
*   **核心组件**: **混合模式 (Hybrid)**。
    *   Encoder/Decoder 使用 **1D ResNet Blocks** 提取局部特征。
    *   Bottleneck 层引入 **Transformer/Self-Attention** 机制以捕捉长程时间依赖。
*   **条件注入**: 通过 Cross-Attention 或 FiLM (Feature-wise Linear Modulation) 层注入 IMU 特征。

---

## 3. 数据与实验设置 (Data & Experiments)

### 3.1 数据集
*   **首选**: **RoNIN Dataset** (Vicon Ground Truth)。
*   **预处理**: 复用 RoNIN 官方源码的预处理逻辑（坐标系对齐、重力去除、归一化等）和数据加载器，确保对比公平性。

### 3.2 训练策略
*   采用 **混合模式 (Hybrid Mode)** 开发策略：
    1.  **离线高精度 (Phase 1)**: 优先在离线场景下验证理论精度，不计较推理耗时。
    2.  **性能优化 (Phase 2)**: 理论验证成功后，探索快速采样算法（如 DDIM）以提升推理速度。

---

## 4. 成功标准 (Success Criteria)

1.  **代码跑通**: 能够成功加载 RoNIN 数据并完成 Baseline 和 Diffusion 模型的训练流程。
2.  **收敛性验证**: 扩散模型的 Loss 能够正常下降，生成的轨迹在视觉上具有物理合理性。
3.  **性能对比**: 在相同测试集上，扩散模型方案在关键指标（如 ATE, RTE）上能够达到或接近 RoNIN Baseline 的水平（进阶目标：超越 Baseline）。

---

## 5. 项目范围 (Scope)

*   **In Scope**:
    *   IMU 数据加载与处理
    *   PriorNet (ResNet) 实现
    *   Diffusion Model (U-Net + Attention) 实现
    *   训练 Loop、验证 Loop、Tensorboard 记录
    *   轨迹积分与可视化脚本
*   **Out of Scope** (初期不考虑):
    *   手机端/嵌入式部署
    *   Visual-Inertial (VIO) 融合
    *   实时在线运行优化
