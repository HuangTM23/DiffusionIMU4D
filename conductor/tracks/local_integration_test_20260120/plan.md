# Implementation Plan - 本地集成测试与验证 (local_integration_test_20260120)

## Phase 1: 准备过拟合实验 (Setup Overfit)
- [x] Task: 创建过拟合配置文件 eb23d24
    - [ ] 编写 `configs/local_overfit.yaml`，设置 epoch=50, lr=1e-3, data=list_debug
- [ ] Task: 运行过拟合训练
    - [ ] 执行 `train_diff.py`
    - [ ] 观察并记录 Loss 变化
- [ ] Task: Conductor - User Manual Verification '过拟合训练' (Protocol in workflow.md)

## Phase 2: 推理与可视化验证 (Inference & Viz)
- [ ] Task: 运行推理脚本
    - [ ] 使用训练好的 checkpoint 运行 `test_diff.py`
- [ ] Task: 分析轨迹图
    - [ ] 检查 `experiments/results` 下的图片
    - [ ] 确认轨迹形状匹配度
- [ ] Task: Conductor - User Manual Verification '轨迹验证' (Protocol in workflow.md)
