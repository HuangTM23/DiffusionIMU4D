# Track Specification - 搭建基础实验环境与 Baseline 复现 (baseline_setup_20260119)

## 1. 业务背景 (Background)
为了验证条件扩散模型在惯导任务中的优越性，首先需要建立一个稳健的基础实验环境。这包括配置正确的计算环境、实现与 RoNIN 数据集的无缝对接，并复现 RoNIN 论文中的 ResNet Baseline 模型。这将作为后续所有扩散模型改进的基准。

## 2. 目标 (Goals)
- 验证 Conda 环境 `DiffM` 中的核心库（PyTorch, Diffusers, WandB）可用性。
- 实现 RoNIN 数据集的预处理和高效加载，确保输入格式与官方一致。
- 复现 ResNet-18 (RoNIN 变体) 模型架构。
- 完成训练管线搭建，确保 Baseline 能够收敛并产生合理的评价指标。

## 3. 技术设计 (Technical Design)

### 3.1 环境配置
- 确保 `requirements.txt` 涵盖所有必要依赖。
- 集成 WandB 初始化逻辑。

### 3.2 数据加载 (Data Loading)
- 复用 RoNIN 的坐标对齐逻辑。
- 实现 `RoNINDataset` 类，支持训练、验证和测试集的划分。
- 输出数据维度应为 `(Batch, Time, Channels)`，其中 Channels 包含 IMU 的 6 轴数据，目标为 3 轴速度。

### 3.3 Baseline 模型
- 架构：基于 ResNet-18 的 1D 卷积变体。
- 损失函数：MSE Loss（预测速度序列与真实速度序列）。
- 评估：集成 RTE/ATE 计算工具。

## 4. 验收标准 (Acceptance Criteria)
- [ ] 数据加载器能正确读取 RoNIN 数据集并进行归一化。
- [ ] Baseline 模型能够在 GPU 上进行训练，且 Loss 随时间下降。
- [ ] 训练过程中的指标（MSE, RTE）能够实时同步到 WandB。
- [ ] 能够生成并保存第一个 Baseline 的权重文件。
