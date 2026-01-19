# Implementation Plan - 搭建基础实验环境与 Baseline 复现 (baseline_setup_20260119)

## Phase 1: 环境准备与项目初始化 (Environment & Initialization) [checkpoint: 0f2cf5b]
- [x] Task: 初始化开发环境与依赖配置文件 fddce48
    - [ ] 编写 `requirements.txt` 包含 `torch`, `diffusers`, `wandb`, `scipy`, `pandas`
    - [ ] 创建 `utils/logger.py` 集成 WandB 初始化逻辑
- [x] Task: 实现基础工具类 (Utils) b4b4c24
    - [ ] 编写测试验证坐标变换工具函数
    - [ ] 实现 `utils/physics.py` 用于 IMU 积分和坐标系对齐
- [x] Task: Conductor - User Manual Verification '环境准备与项目初始化' (Protocol in workflow.md) 0f2cf5b

## Phase 2: RoNIN 数据适配 (Data Adaptation)
- [x] Task: 实现 RoNIN 数据集加载器 aa1ba0a
    - [ ] 编写测试验证数据加载的维度和归一化
    - [ ] 在 `data/dataset.py` 中实现 `RoNINDataset` 类
- [x] Task: 验证 DataLoader 性能 05bbc88
    - [ ] 编写测试确保 DataLoader 在多进程模式下稳定运行
- [ ] Task: Conductor - User Manual Verification 'RoNIN 数据适配' (Protocol in workflow.md)

## Phase 3: Baseline 模型实现 (Baseline Model Implementation)
- [ ] Task: 实现 ResNet-18 (RoNIN 变体) 架构
    - [ ] 编写测试验证模型的 Input/Output 维度
    - [ ] 在 `models/resnet_baseline.py` 中实现模型
- [ ] Task: 编写训练主循环 (Training Loop)
    - [ ] 在 `train.py` 中实现基础训练和验证逻辑，支持 WandB 记录
- [ ] Task: 验证 Baseline 训练闭环
    - [ ] 使用少量数据运行 1 个 epoch，验证从数据读取到 Loss 更新的完整流程
- [ ] Task: Conductor - User Manual Verification 'Baseline 模型实现' (Protocol in workflow.md)
