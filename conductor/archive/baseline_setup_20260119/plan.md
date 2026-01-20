# Implementation Plan - 搭建基础实验环境与 Baseline 复现 (baseline_setup_20260119)

## Phase 1: 环境准备与项目初始化 (Environment & Initialization) [checkpoint: 0f2cf5b]
- [x] Task: 初始化开发环境与依赖配置文件 fddce48
    - [ ] 编写 `requirements.txt` 包含 `torch`, `diffusers`, `wandb`, `scipy`, `pandas`
    - [ ] 创建 `utils/logger.py` 集成 WandB 初始化逻辑
- [x] Task: 实现基础工具类 (Utils) b4b4c24
    - [ ] 编写测试验证坐标变换工具函数
    - [ ] 实现 `utils/physics.py` 用于 IMU 积分和坐标系对齐
- [x] Task: Conductor - User Manual Verification '环境准备与项目初始化' (Protocol in workflow.md) 0f2cf5b

## Phase 2: RoNIN 数据适配 (Data Adaptation) [checkpoint: f91ab07]
- [x] Task: 实现 RoNIN 数据集加载器 47c6c23
- [x] Task: 验证 DataLoader 性能 cfe97e0
- [x] Task: Conductor - User Manual Verification 'RoNIN 数据适配' (Protocol in workflow.md) f91ab07

## Phase 3: Baseline 模型实现 (Baseline Model Implementation)
- [x] Task: 实现 ResNet-18 (RoNIN 变体) 架构 f600bb7
- [x] Task: 编写训练主循环 (Training Loop) e70ccad
- [x] Task: 验证 Baseline 训练闭环 29824a7
- [x] Task: Conductor - User Manual Verification 'Baseline 模型实现' (Protocol in workflow.md) 29824a7
