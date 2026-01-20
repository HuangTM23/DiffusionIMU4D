# IMU 条件扩散模型 (Diffusion4d) 技术栈

> 本文档列出了项目开发所使用的编程语言、框架、核心库及其版本要求。

---

## 1. 核心语言与环境 (Core Language & Environment)

*   **Python**: 3.9+
*   **环境管理器**: **Conda**
    *   **当前激活环境**: `DiffM` (用户指定环境名)
*   **操作系统**: Linux (当前开发环境)

---

## 2. 深度学习框架 (Deep Learning Frameworks)

*   **核心框架**: **PyTorch 2.0+** (支持 `torch.compile` 以加速训练)
*   **生成式库**: **HuggingFace `diffusers`**
    *   **用途**: 复用高效的 Schedulers (DDPM, DDIM) 和去噪 Pipeline 架构，加速扩散模型开发。
*   **训练加速**: `accelerate` (可选，用于分布式训练和混合精度)

---

## 3. 实验记录与可视化 (Experiment Tracking & Visualization)

*   **核心实验管理**: **WandB (Weights & Biases)**
    *   **用途**: 记录超参数、Loss 曲线、模型版本、消融实验结果对比以及生成的轨迹采样图。
*   **日志辅助**: `tensorboardX` (用于某些官方组件的兼容性)。
*   **绘图库**:
    *   `matplotlib`: 基础绘图。
    *   `plotly`: 生成可交互的 3D 轨迹对比图。

---

## 4. 数据处理与物理计算 (Data Processing & Physics)

*   **基础计算**: `numpy`, `scipy` (用于信号处理和坐标变换)。
*   **旋转数学**: `numpy-quaternion` (处理四元数旋转)。
*   **数据管理**: `pandas`, `h5py` / `pickle` (加载 RoNIN 格式数据)。
*   **物理仿真/积分**: 自定义惯性积分算法 (Inertial Integration)。

---

## 5. 项目工程工具 (Engineering Tools)

*   **配置管理**: `pyyaml` 或 `hydra` (支持分层配置)。
*   **代码规范**: `ruff` (快速的 Linter 和 Formatter)。
*   **依赖记录**: `requirements.txt` 或 `environment.yml`。

---

## 6. 开发优先级 (Tech Priorities)

1.  **PyTorch & Diffusers 深度集成**: 确保 IMU 条件能正确喂入 `diffusers` 提供的 UNet 结构。
2.  **WandB 自动同步**: 训练脚本启动时自动创建 WandB run 并上传 config。
3.  **RoNIN 数据适配**: 确保 `data_loader` 能够无缝读取 RoNIN 数据集。
