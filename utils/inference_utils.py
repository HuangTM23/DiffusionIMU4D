import numpy as np
import torch

def reconstruct_trajectory(model_func, imu_seq, window_size=200, stride=10, device='cpu', batch_size=32):
    """
    长序列滑动窗口推理与拼接。
    
    Args:
        model_func (callable): 模型推理函数，输入 (B, 6, window_size)，输出 (B, 3, window_size)。
        imu_seq (np.ndarray): 完整 IMU 序列 (L, 6)。
        window_size (int): 窗口大小。
        stride (int): 滑动步长。
        device (str): 计算设备。
        batch_size (int): 推理 Batch Size。
        
    Returns:
        np.ndarray: 重建的速度序列 (L, 3)。
    """
    L = imu_seq.shape[0]
    
    # 1. 准备窗口
    # 如果序列比窗口短，Padding (这里简化处理，假设 L >= window_size)
    if L < window_size:
        raise ValueError(f"Sequence length {L} is smaller than window size {window_size}")
        
    starts = np.arange(0, L - window_size + 1, stride)
    
    # 如果最后一段没覆盖到结尾，补一个从结尾往前数的窗口
    if starts[-1] + window_size < L:
        starts = np.concatenate([starts, [L - window_size]])
        
    windows = []
    for s in starts:
        windows.append(imu_seq[s:s+window_size])
    
    windows = np.stack(windows) # (N, window_size, 6)
    
    # 2. 批量推理
    num_windows = windows.shape[0]
    outputs = []
    
    for i in range(0, num_windows, batch_size):
        batch_in = windows[i:i+batch_size]
        batch_in_torch = torch.from_numpy(batch_in).float().permute(0, 2, 1).to(device) # (B, 6, W)
        
        with torch.no_grad():
            # Model output: (B, 3, W)
            batch_out = model_func(batch_in_torch)
            batch_out_np = batch_out.permute(0, 2, 1).cpu().numpy() # (B, W, 3)
            outputs.append(batch_out_np)
            
    outputs = np.concatenate(outputs, axis=0) # (N, W, C)
    C = outputs.shape[2]
    
    # 3. 拼接与平滑 (Weighted Average)
    recon_vel = np.zeros((L, C))
    weights = np.zeros((L, 1))
    
    # 使用三角形权重窗口以平滑边界
    # linear weight: [0, 1, ..., 1, 0] ? 
    # 简单起见，先用全 1 权重 (平均)
    window_weight = np.ones((window_size, 1))
    
    # 或者用 Hanning 窗减少边界效应
    # window_weight = np.hanning(window_size)[:, None] 
    
    for i, s in enumerate(starts):
        recon_vel[s:s+window_size] += outputs[i] * window_weight
        weights[s:s+window_size] += window_weight
        
    # 避免除零 (虽然逻辑上应该都有覆盖)
    weights[weights == 0] = 1.0
    recon_vel /= weights
    
    return recon_vel

def integrate_trajectory(velocity, initial_pos=None, dt=0.01):
    """
    积分速度得到位置。
    Args:
        velocity: (L, 3) or (L, 2)
        initial_pos: (3,) or (2,)
        dt: float, time step
    Returns:
        pos: (L, 3) or (L, 2)
    """
    if initial_pos is None:
        initial_pos = np.zeros(velocity.shape[1])
        
    pos = np.zeros_like(velocity)
    pos[0] = initial_pos
    
    # 累加：p_t = p_{t-1} + v_t * dt
    # 注意：RoNIN 习惯是 v_t 是从 t 到 t+1 的平均速度
    # 累积和: cumsum
    pos[1:] = np.cumsum(velocity[:-1] * dt, axis=0) + initial_pos
    
    return pos
