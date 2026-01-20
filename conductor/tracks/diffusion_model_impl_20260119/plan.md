# Implementation Plan - 实现 IMU 条件扩散模型 (diffusion_model_impl_20260119)

## Phase 1: 扩散模型架构搭建 (Architecture)
- [x] Task: 定义 DiffUnet1D 骨干网络 a97a2ee
    - [ ] 编写测试验证 1D U-Net 的输入输出维度
    - [ ] 在 `models/diffusion_unet.py` 中实现基于 1D 卷积和 ResBlock 的 U-Net
- [x] Task: 实现条件注入模块 (Conditioning) ce4ac5b
    - [ ] 实现 FiLM 或 Cross-Attention 模块
    - [ ] 修改 `ResNet1D` 以支持输出特征图而非最终预测
- [ ] Task: Conductor - User Manual Verification '架构搭建' (Protocol in workflow.md)

## Phase 2: 扩散训练管线 (Pipeline)
- [x] Task: 集成 Diffusers Scheduler ac16a07
    - [ ] 创建 `models/diffusion_system.py` 封装 Scheduler 和 U-Net
    - [ ] 实现 `forward` (计算 Loss) 和 `sample` (生成序列) 方法
- [ ] Task: 升级 Trainer 支持扩散模型
    - [ ] 修改 `train.py` (或新建 `train_diff.py`) 以适配噪声预测 Loss
    - [ ] 添加 WandB 采样可视化回调
- [ ] Task: Conductor - User Manual Verification '训练管线' (Protocol in workflow.md)

## Phase 3: 变体实现与验证 (Variants & Validation)
- [ ] Task: 实现 Variant A (Refinement) 逻辑
    - [ ] 在数据加载或模型前向中加入 PriorNet 推理
- [ ] Task: 编写扩散模型配置文件
    - [ ] 创建 `configs/diffusion_variant_a.yaml` 和 `configs/diffusion_variant_b.yaml`
- [ ] Task: 跑通小规模训练验证
    - [ ] 使用 Debug 数据集验证两种变体的收敛性
- [ ] Task: Conductor - User Manual Verification '变体验证' (Protocol in workflow.md)
