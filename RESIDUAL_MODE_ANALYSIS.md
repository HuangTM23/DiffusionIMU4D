# Residual 模式问题诊断分析

## 你的现象

- 训练/验证损失收敛良好
- 但测试 ATE/RTE 很高（seen: ~24m, unseen: ~27m）

## Residual 模式的工作原理

```
Input IMU
    ↓
Encoder (ResNet1D) → Features (B, 512, L/32)
    ↓
┌─────────────────────────────────────────┐
│  Branch 1: prior_head                   │
│  → v_prior (粗糙速度预测)                │
│                                         │
│  Branch 2: UNet (Diffusion)             │
│  → residual (学习残差)                   │
│  输入: 噪声 + v_prior 作为条件           │
└─────────────────────────────────────────┘
    ↓
Final Output = v_prior + residual
```

## 可能的问题根源

### 1. **v_prior 质量问题（最可能）**

训练时的监督方式：
```python
# diffusion_system.py 第115-119行
if self.mode == "residual":
    prior_loss = torch.nn.functional.mse_loss(v_prior, gt_vel)
    loss = loss + prior_loss  # residual loss + prior loss
```

**问题分析**：
- `prior_loss` 和 `residual` 损失是同时优化的
- 但 UNet 的条件是 `cond_feat` (encoder输出)，不是 `v_prior`
- 这导致 encoder 需要同时满足两个目标：
  1. 生成好的特征给 UNet
  2. 生成好的 v_prior
- 这两个目标可能冲突！

**诊断方法**：
运行分析脚本，查看 `residual_analysis.png`：
- 如果 `v_prior` (蓝色线) 与 GT (黑线) 差距很大 → v_prior 没学好
- 如果 `residual` (绿色线) 幅度很大 → diffusion 负担过重

### 2. **条件信息传递问题**

当前代码：
```python
# UNet 的条件是 encoder 特征，不是 v_prior
model_pred = self.unet(noisy_x, timesteps, cond_feat)  # cond_feat 是特征，不是 v_prior
```

**问题**：UNet 并不知道 v_prior 的具体值，只知道 encoder 特征。

在推理时：
```python
v_prior = prior_head(cond_feat)  # 根据特征生成 v_prior
residual = unet.sample(...)      # UNet 只看到了 cond_feat
output = v_prior + residual      # 但 UNet 没明确知道 v_prior 的值
```

**理想设计**：UNet 应该把 v_prior 作为显式条件：
```python
# 更好的设计
v_prior = prior_head(cond_feat)
model_pred = self.unet(noisy_x, timesteps, v_prior)  # 用 v_prior 作为条件
```

### 3. **残差分布问题**

如果 v_prior 已经很好了，残差应该：
- 均值为 0
- 方差很小
- 分布接近高斯

但如果 v_prior 很差：
- 残差幅度很大
- 分布复杂，diffusion 难学
- 训练和测试的分布不一致

### 4. **滑动窗口不连续性问题**

每个窗口独立采样：
```python
# Window 1
v_prior_1 = prior_head(encoder(imu[0:200]))
residual_1 = diffusion.sample(...)
pred_1 = v_prior_1 + residual_1

# Window 2 (stride=100，与 Window 1 重叠100帧)
v_prior_2 = prior_head(encoder(imu[100:300]))
residual_2 = diffusion.sample(...)
pred_2 = v_prior_2 + residual_2
```

**问题**：
- `pred_1[100:200]` 和 `pred_2[0:100]` 是同一时刻的预测
- 但由于采样随机性，两者可能差异很大
- 拼接后的速度序列在窗口边界处不连续

**诊断**：查看 `window_continuity.png` 中的边界差异柱状图

## 使用分析脚本诊断

```bash
python test_diff_analysis.py \
    --config configs/diffusion_variant_a.yaml \
    --model_path experiments/checkpoints/diff_residual_epoch_XX.pth \
    --out_dir experiments/analysis_residual \
    --plot \
    --max_seqs 5  # 先分析5个序列
```

### 关键观察指标

#### 1. v_prior 质量 (`residual_analysis.png`)

| 观察 | 含义 | 解决方案 |
|-----|------|---------|
| v_prior 与 GT 很接近 | v_prior 学习良好 | 检查 residual 是否必要 |
| v_prior 与 GT 差距大 | prior_head 没学好 | 增加 prior_loss 权重，或单独预训练 |
| residual 幅度很大 | diffusion 负担过重 | 改进 v_prior 质量 |
| residual 呈噪声状 | 可能工作正常 | - |
| residual 有明显结构 | v_prior 有系统性偏差 | 分析偏差模式 |

#### 2. 窗口连续性 (`window_continuity.png`)

| 边界差异 | 含义 | 解决方案 |
|---------|------|---------|
| < 0.1 m/s | 连续性良好 | - |
| 0.1-0.5 m/s | 轻微不连续 | 增加采样步数，或使用确定性采样 |
| > 0.5 m/s | 严重不连续 | 需要自回归或一致性约束 |

#### 3. 速度误差 vs ATE (`global_analysis.png`)

- 如果速度误差小但 ATE 大 → 积分累积误差，速度有系统偏差
- 如果速度误差大 → 速度预测本身有问题

## 推荐的解决方案

### 方案 A：显式条件 Residual（推荐）

修改 `diffusion_system.py`，让 UNet 以 v_prior 为显式条件：

```python
class DiffusionSystem(nn.Module):
    def __init__(...):
        # ...
        # 修改 UNet 接受 v_prior 作为条件
        self.unet = DiffUNet1D(
            in_channels=target_dim, 
            out_channels=target_dim, 
            cond_channels=target_dim,  # 改为 target_dim，直接输入 v_prior
            # ...
        )
    
    def forward(self, imu, gt_vel):
        cond_feat = self.encoder(imu)
        v_prior = self.prior_head(cond_feat)
        v_prior_up = F.interpolate(v_prior, size=gt_vel.shape[-1], ...)
        
        target_x0 = gt_vel - v_prior_up
        
        # 使用 v_prior_up 作为 UNet 的条件
        model_pred = self.unet(noisy_x, timesteps, v_prior_up)
        # ...
```

### 方案 B：级联训练

1. **第一阶段**：只训练 prior_head，冻结 encoder
   ```python
   # 冻结 encoder
   for param in system.encoder.parameters():
       param.requires_grad = False
   
   # 只优化 prior_head
   optimizer = optim.Adam(system.prior_head.parameters(), lr=config['lr'])
   ```

2. **第二阶段**：固定 prior_head，训练 diffusion
   ```python
   # 冻结 prior_head
   for param in system.prior_head.parameters():
       param.requires_grad = False
   
   # 训练 diffusion
   ```

### 方案 C：窗口一致性约束

在推理时使用前一窗口信息初始化：

```python
def sample_with_continuity(self, imu_windows, prev_residual=None):
    """
    自回归采样，使用前一窗口的末尾作为当前窗口的初始化
    """
    results = []
    for i, imu in enumerate(imu_windows):
        cond_feat = self.encoder(imu)
        v_prior = self.prior_head(cond_feat)
        
        # 如果不是第一个窗口，用前一窗口的信息初始化
        if i > 0 and prev_residual is not None:
            # 初始化 xt 时考虑连续性
            xt = self._init_with_continuity(v_prior, prev_residual, overlap_size)
        else:
            xt = torch.randn(...)
        
        # 采样
        for t in self.scheduler.timesteps:
            # ...
        
        prev_residual = xt - v_prior  # 保存当前残差供下一窗口使用
        results.append(xt + v_prior)
    
    return results
```

### 方案 D：端到端训练（放弃 Residual）

如果 residual 模式难以调好，可以直接使用 end2end 模式：

```yaml
# configs/diffusion_variant_b.yaml
mode: "end2end"
stride: 10  # 更细的粒度
```

end2end 模式没有 v_prior 和 residual 的分解问题，可能更稳定。

## 快速验证建议

1. **先用分析脚本可视化几个序列**，确认问题类型
2. **单独测试 v_prior**：注释掉 diffusion，只用 v_prior 预测，看 ATE 是多少
3. **尝试不同的 stride**：stride=10 vs stride=100，看拼接误差的影响
4. **对比 end2end 模式**：用相同配置训练 end2end 版本，看是否 residual 模式特有的问题
