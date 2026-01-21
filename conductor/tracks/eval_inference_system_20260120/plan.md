# Implementation Plan - 完善评估与推理系统 (eval_inference_system_20260120)

## Phase 1: 轨迹重建与拼接 (Trajectory Reconstruction) [checkpoint: 1b7c8b7]
- [x] Task: 实现长序列滑动窗口推理 325cba7
- [x] Task: 实现位置积分逻辑 55fb1e7
- [x] Task: Conductor - User Manual Verification '轨迹重建' (Protocol in workflow.md) 1b7c8b7

## Phase 2: 全量评估脚本 (Evaluation Script) [checkpoint: 213dcaf]
- [x] Task: 编写 test_diff.py 213dcaf
- [x] Task: 集成 DDIM 采样器 213dcaf
- [x] Task: 验证评估脚本 213dcaf
- [x] Task: Conductor - User Manual Verification '评估脚本' (Protocol in workflow.md) 213dcaf

## Phase 3: 可视化与报告 (Visualization) [checkpoint: 41bff5f]
- [x] Task: 实现批量绘图功能 41bff5f
- [x] Task: Conductor - User Manual Verification '可视化' (Protocol in workflow.md) 41bff5f
