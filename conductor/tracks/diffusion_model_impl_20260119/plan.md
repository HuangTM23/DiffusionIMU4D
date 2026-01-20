# Implementation Plan - 实现 IMU 条件扩散模型 (diffusion_model_impl_20260119)

## Phase 1: 扩散模型架构搭建 (Architecture)
- [x] Task: 定义 DiffUnet1D 骨干网络 a97a2ee
    - [ ] 编写测试验证 1D U-Net 的输入输出维度
    - [ ] 在 `models/diffusion_unet.py` 中实现基于 1D 卷积和 ResBlock 的 U-Net
- [x] Task: 实现条件注入模块 (Conditioning) ce4ac5b
    - [ ] 实现 FiLM 或 Cross-Attention 模块
    - [ ] 修改 `ResNet1D` 以支持输出特征图而非最终预测
- [ ] Task: Conductor - User Manual Verification '架构搭建' (Protocol in workflow.md)

## Phase 2: 扩散训练管线 (Pipeline) [checkpoint: 967be05]
- [x] Task: 集成 Diffusers Scheduler ac16a07
- [x] Task: 升级 Trainer 支持扩散模型 b279b77
- [x] Task: Conductor - User Manual Verification '训练管线' (Protocol in workflow.md) 967be05

## Phase 3: 变体实现与验证 (Variants & Validation)
- [x] Task: 实现 Variant A (Refinement) 逻辑 ac16a07
- [x] Task: 编写扩散模型配置文件 b279b77
- [x] Task: 跑通小规模训练验证 b279b77
- [x] Task: Conductor - User Manual Verification '变体验证' (Protocol in workflow.md) b279b77
