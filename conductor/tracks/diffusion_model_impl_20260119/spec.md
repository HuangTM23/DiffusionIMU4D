# Track Specification - 实现 IMU 条件扩散模型 (diffusion_model_impl_20260119)

## 1. 业务背景 (Background)
根据项目产品指南，本项目的核心目标是验证生成式模型在惯性导航中的潜力。在建立了 RoNIN ResNet Baseline 之后，现在的关键任务是实现条件扩散模型（Conditional Diffusion Model），并将其与 Baseline 进行公平对比。

## 2. 目标 (Goals)
- 实现基于 1D U-Net 的扩散模型架构，支持时间序列生成。
- 集成 `diffusers` 库的 Scheduler (DDPM/DDIM) 管理加噪和去噪过程。
- 实现两种核心变体（Variant A: Refinement, Variant B: End-to-End），并通过配置文件切换。
- 升级训练管线，支持扩散模型的损失计算（噪声预测误差）和采样评估。

## 3. 技术设计 (Technical Design)

### 3.1 模型架构 (Model Architecture)
- **Backbone**: `DiffUnet1D`。基于 `diffusers` 的 `UNet1DModel` 或自定义 1D U-Net。
    - **输入**: 噪声速度序列 `(B, 3, L)` + 时间步 `t` + 条件特征 `c`。
    - **输出**: 预测噪声 `(B, 3, L)`。
- **条件编码器 (Condition Encoder)**:
    - 复用已实现的 `ResNet1D`（去除最后的回归层）作为特征提取器，输出 IMU 嵌入 `(B, D, L)`。
- **条件注入**:
    - 使用 FiLM (Feature-wise Linear Modulation) 或 Cross-Attention 将 IMU 嵌入注入到 U-Net 的 ResBlock 中。

### 3.2 训练策略 (Training Strategy)
- **Variant A (Refinement)**:
    - 预训练的 ResNet 输出粗略速度 $v_{prior}$。
    - 扩散模型学习残差分布：$v_{gt} = v_{prior} + v_{residual}$。
- **Variant B (End-to-End)**:
    - 扩散模型直接以 ResNet 提取的特征为条件，从 $x_T \sim \mathcal{N}(0, I)$ 生成 $v_{gt}$。

### 3.3 采样与评估 (Sampling & Eval)
- 在验证阶段，使用 DDPM 或 DDIM Scheduler 进行多步采样。
- 将生成的 $v_{pred}$ 积分得到轨迹，计算 ATE/RTE 并记录到 WandB。

## 4. 验收标准 (Acceptance Criteria)
- [ ] `DiffUnet1D` 模型能够成功初始化并处理输入张量。
- [ ] 训练脚本支持通过 `--config` 切换 Baseline 和 Diffusion 模式。
- [ ] 扩散模型能够收敛，且生成的轨迹在视觉上合理（不发散）。
- [ ] 能够输出采样生成速度与 Ground Truth 的对比图。
