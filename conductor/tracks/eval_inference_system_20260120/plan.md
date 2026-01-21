# Implementation Plan - 完善评估与推理系统 (eval_inference_system_20260120)

## Phase 1: 轨迹重建与拼接 (Trajectory Reconstruction)
- [x] Task: 实现长序列滑动窗口推理 325cba7
    - [ ] 编写 `utils/inference_utils.py`，实现 `reconstruct_trajectory` 函数，支持重叠窗口平均
    - [ ] 编写单元测试验证拼接逻辑的正确性（输入已知序列，检查拼接结果）
- [ ] Task: 实现位置积分逻辑
    - [ ] 复用并封装 RoNIN 的积分代码，确保与 Baseline 一致
- [ ] Task: Conductor - User Manual Verification '轨迹重建' (Protocol in workflow.md)

## Phase 2: 全量评估脚本 (Evaluation Script)
- [ ] Task: 编写 `test_diff.py`
    - [ ] 支持加载模型、遍历测试列表
    - [ ] 集成 `reconstruct_trajectory`
    - [ ] 计算 ATE/RTE 并保存结果
- [ ] Task: 集成 DDIM 采样器
    - [ ] 在 `test_diff.py` 和 `DiffusionSystem` 中添加 Scheduler 切换逻辑
- [ ] Task: 验证评估脚本
    - [ ] 使用 Debug 模型和少量数据跑通评估流程
- [ ] Task: Conductor - User Manual Verification '评估脚本' (Protocol in workflow.md)

## Phase 3: 可视化与报告 (Visualization)
- [ ] Task: 实现批量绘图功能
    - [ ] 为每个测试序列生成 GT vs Pred 轨迹图
    - [ ] 生成误差分布图
- [ ] Task: Conductor - User Manual Verification '可视化' (Protocol in workflow.md)
