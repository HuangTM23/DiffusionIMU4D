# Track Specification - 完善评估与推理系统 (eval_inference_system_20260120)

## 1. 业务背景
为了验证扩散模型的有效性，必须建立一套完整的评估体系。这不仅包括计算单一窗口的损失，更重要的是评估**长时轨迹**的精度（ATE/RTE）。此外，为了提高推理效率，需要集成加速采样算法（DDIM）。

## 2. 目标
- 实现全量测试集评估脚本 `test_diff.py`。
- 实现长序列轨迹拼接与重建（Sliding Window Inference）。
- 集成 DDPM 和 DDIM 采样器，并支持参数配置。
- 自动生成评估报告（CSV）和轨迹对比图。

## 3. 技术设计

### 3.1 推理逻辑 (Inference Logic)
- **长序列处理**: 测试集序列长度通常远超 200 帧。
    - 策略：使用滑动窗口（Stride < Window Size），对重叠部分进行加权平均。
    - 确保生成的轨迹连续平滑。
- **轨迹积分**:
    - $P_t = P_{t-1} + V_t \cdot \Delta t$。
    - 初始位置设为 $(0,0,0)$ 或 GT 初始位置。

### 3.2 采样加速
- 在 `DiffusionSystem` 或 `test_diff.py` 中，允许通过配置切换 `DDPMScheduler` 和 `DDIMScheduler`。
- DDIM 通常只需 50 步即可达到较好效果，大幅提升评估速度。

### 3.3 指标计算
- 复用 RoNIN 的 `compute_ate_rte`。
- 输出：总体 ATE/RTE 均值，以及每个序列的详细指标。

## 4. 验收标准
- [ ] `test_diff.py` 能够加载训练好的 Checkpoint 并运行测试集。
- [ ] 能够生成长时轨迹（>1分钟），且拼接处无明显跳变。
- [ ] 支持通过 `--scheduler ddim` 切换采样器。
- [ ] 输出包含 ATE/RTE 的 `metrics.csv` 和轨迹图 `*.png`。
