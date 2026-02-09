import numpy as np
import torch

def reconstruct_trajectory(model_func, imu_seq, window_size=200, stride=10, device='cpu', batch_size=32):
    """
    长序列滑动窗口推理与拼接。支持返回 (Final, Prior) 双轨迹。
    """
    L = imu_seq.shape[0]
    if L < window_size:
        raise ValueError(f"Sequence length {L} is smaller than window size {window_size}")
        
    starts = np.arange(0, L - window_size + 1, stride)
    if starts[-1] + window_size < L:
        starts = np.concatenate([starts, [L - window_size]])
        
    windows = []
    for s in starts:
        windows.append(imu_seq[s:s+window_size])
    windows = np.stack(windows)
    
    num_windows = windows.shape[0]
    outputs = []
    priors = []
    
    for i in range(0, num_windows, batch_size):
        batch_in = windows[i:i+batch_size]
        batch_in_torch = torch.from_numpy(batch_in).float().permute(0, 2, 1).to(device)
        
        with torch.no_grad():
            batch_res = model_func(batch_in_torch)
            
            # 判断模型函数返回的是单一结果还是 (Final, Prior) 元组
            if isinstance(batch_res, tuple):
                batch_out, batch_prior = batch_res
                priors.append(batch_prior.permute(0, 2, 1).cpu().numpy())
            else:
                batch_out = batch_res
                
            outputs.append(batch_out.permute(0, 2, 1).cpu().numpy())
            
    outputs = np.concatenate(outputs, axis=0)
    C = outputs.shape[2]
    recon_vel = np.zeros((L, C))
    weights = np.zeros((L, 1))
    
    recon_prior = None
    if len(priors) > 0:
        priors = np.concatenate(priors, axis=0)
        recon_prior = np.zeros((L, C))

    window_weight = np.ones((window_size, 1))
    
    for i, s in enumerate(starts):
        recon_vel[s:s+window_size] += outputs[i] * window_weight
        if recon_prior is not None:
            recon_prior[s:s+window_size] += priors[i] * window_weight
        weights[s:s+window_size] += window_weight
        
    recon_vel /= (weights + 1e-8)
    if recon_prior is not None:
        recon_prior /= (weights + 1e-8)
    
    return (recon_vel, recon_prior) if recon_prior is not None else recon_vel

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
