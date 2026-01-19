# IMU 条件扩散模型 (Diffusion4d) 开发准则

> 本文档定义了项目在代码实现、实验管理和文档记录方面的核心原则和风格指南。

---

## 1. 代码设计准则 (Code Design Principles)

### 1.1 极致模块化 (Extreme Modularity)
*   **解耦设计**：模型架构 (`models/`)、数据预处理与加载 (`data/`)、训练逻辑 (`trainer.py`) 以及评估逻辑 (`eval/`) 必须严格解耦。
*   **配置驱动**：所有的超参数（随机种子、学习率、扩散步数、网络层数等）应通过配置文件（如 `config.yml`）或命令行参数管理，严禁硬编码在逻辑中。
*   **插件化模型组件**：支持通过配置轻松切换不同的 Backbone（如 1D-CNN, Transformer）和去噪策略。

### 1.2 严谨的代码规范
*   **命名规范**：遵循标准学术英语。类名使用 `PascalCase`（如 `InertialDiffusionModel`），变量和函数名使用 `snake_case`。
*   **注释风格**：
    *   **Docstring**：使用中文编写详细的 Docstring，解释函数的输入输出、逻辑和数学含义。
    *   **数学注释**：对于扩散模型的加噪和去噪公式（如 $\mu_\theta, \sigma_\theta$），在代码对应位置使用 Markdown/LaTeX 风格注释说明数学来源。

---

## 2. 实验管理与可复现性 (Experiment Management)

### 2.1 实验记录 (Experiment Tracking)
*   **全面日志**：所有实验必须自动记录 `config`、训练曲线、权重快照和计算设备信息。
*   **可视化集成**：默认集成 **Tensorboard** 或 **WandB**。每轮 epoch 必须记录核心指标（Loss, MSE）以及生成速度序列的采样结果图。

### 2.2 可复现性 (Reproducibility)
*   **固定随机性**：代码库应提供全局随机种子设置函数，确保在同一硬件下结果可复现。
*   **环境一致性**：定期导出 `requirements.txt` 或 `environment.yml`。

---

## 3. 评估与可视化 (Evaluation & Visualization)

### 3.1 指标驱动
*   复用 **RoNIN** 的标准评估指标：RMSE（均方根误差）、ATE（绝对轨迹误差）、RTE（相对轨迹误差）。
*   输出结果应包含所有指标的详细均值和标准差。

### 3.2 强可视化
*   **轨迹绘图**：评估脚本必须能自动绘制生成轨迹与真实轨迹的对比图。
*   **误差分布图**：通过可视化手段展示不同运动模式（步行、跑步、握持方式）下的误差分布情况。

---

## 4. 文档与科研笔记 (Documentation & Research Notes)

### 4.1 文档风格
*   **科研笔记风格**：README 不仅是技术手册，更应记录实验设想、观察到的现象、模型失效案例分析以及消融实验的结论。
*   **公式友好**：文档中涉及的物理建模和扩散模型推导应清晰排版。

### 4.2 目录结构 (Academic Style)
```text
Diffusion4d/
├── configs/            # 实验配置文件
├── data/               # 数据加载与预处理逻辑
├── models/             # 模型架构定义
├── scripts/            # 数据清洗与转换脚本
├── utils/              # 物理积分、可视化等辅助工具
├── experiments/        # 存放训练日志与权重模型
├── eval.py             # 评估主入口
├── train.py            # 训练主入口
└── README.md           # 科研笔记与项目说明
```
