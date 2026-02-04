"""
Diffusion Model Advanced Inference Script
实现方案 A：自回归采样 (Autoregressive In-painting)
通过强制约束重叠区域的一致性来解决窗口拼接断裂问题
"""
import torch
import torch.nn as nn
import os
import argparse
import yaml
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd

from data.data_glob_speed import GlobSpeedSequence
from models.model_resnet1d import ResNet1D, BasicBlock1D
from models.diffusion_unet import DiffUNet1D
from models.diffusion_system import DiffusionSystem
from utils.inference_utils import integrate_trajectory
from utils.metric import compute_ate_rte

def reconstruct_autoregressive(system, imu_seq, window_size, stride, device, num_steps, mode="end2end"):
    """
    使用自回归 In-painting 策略重建轨迹
    
    原理：
    在生成第 k 个窗口时，将其前部（与第 k-1 个窗口重叠的部分）强制固定为
    第 k-1 个窗口已生成的后部。
    这通过在 Diffusion 的每一步反向过程中进行 latent replacement 实现。
    """
    L = imu_seq.shape[0]
    
    # 动态获取 target_dim (通常是 2 或 3)
    # 通过查看 unet 的输出层
    target_dim = system.unet.out_conv[-1].out_channels
    
    # 准备窗口起始点
    starts = np.arange(0, L - window_size + 1, stride)
    if starts[-1] + window_size < L:
        starts = np.concatenate([starts, [L - window_size]])
    
    # 初始化全局结果容器
    # 我们直接填充到这个大数组里，不再需要加权平均
    # 因为我们相信 In-painting 保证了完美衔接
    full_pred = np.zeros((L, target_dim))
    
    # 记录每个位置被写入了多少次（理论上除了头尾，中间都应该是 1 次，因为我们只保留新生成部分）
    # 但为了保险，我们还是用重写逻辑
    
    print(f"  [AutoReg] Processing {len(starts)} windows with In-painting...")
    
    # 缓冲区，用于保存上一个窗口的预测结果（用于 In-painting 的参考）
    # 格式: (window_size, target_dim)
    prev_window_pred = None
    
    system.eval()
    system.scheduler.set_timesteps(num_steps)
    
    for i, start_idx in enumerate(tqdm(starts, desc="In-painting", leave=False)):
        end_idx = start_idx + window_size
        
        # 1. 准备当前窗口数据
        imu_window = imu_seq[start_idx:end_idx] # (W, 6)
        imu_tensor = torch.from_numpy(imu_window).float().unsqueeze(0).permute(0, 2, 1).to(device) # (1, 6, W)
        
        # 2. 编码 Condition
        with torch.no_grad():
            cond_feat = system.encoder(imu_tensor)
            
            # 如果是 Residual 模式 (注意：这里假设是用旧模型，即未修改 Explicit Prior 的版本)
            # 如果是新模型，需要修改这里
            v_prior = None
            if mode == 'residual':
                v_prior_feat = system.prior_head(cond_feat)
                v_prior = torch.nn.functional.interpolate(
                    v_prior_feat, size=window_size, mode='linear', align_corners=False
                )
        
        # 3. 确定 In-painting 的 Mask 和参考值
        mask = None
        masked_image = None # 这里的 image 指的是 latent x0 或者 xt
        
        if i > 0:
            # 计算与上一个窗口的重叠量
            # 上一个窗口范围: [prev_start, prev_start + window_size]
            # 当前窗口范围:   [start_idx,  start_idx + window_size]
            prev_start = starts[i-1]
            overlap_len = (prev_start + window_size) - start_idx
            
            if overlap_len > 0:
                # 我们需要当前窗口的前 overlap_len 长度与上一个窗口的后 overlap_len 长度一致
                
                # Mask: 1 表示已知区域 (需要 In-paint)，0 表示未知区域 (需要生成)
                mask = torch.zeros((1, 1, window_size), device=device)
                mask[:, :, :overlap_len] = 1.0
                
                # Reference: 上一个窗口的重叠部分
                # prev_window_pred 是 (W, C) numpy
                ref_part = prev_window_pred[-overlap_len:] # (overlap, C)
                
                # 构造当前窗口的参考 latent (1, C, W)
                # 只有前 overlap_len 是有效的，后面填 0 即可
                ref_full = np.zeros((window_size, target_dim))
                ref_full[:overlap_len] = ref_part
                
                masked_image = torch.from_numpy(ref_full).float().unsqueeze(0).permute(0, 2, 1).to(device) # (1, C, W)
                
                # 如果是 Residual 模式，Diffusion 预测的是残差
                # 所以参考值也应该是残差： residual = pred - v_prior
                if mode == 'residual':
                    # 当前窗口的 v_prior (1, C, W)
                    curr_v_prior_np = v_prior.cpu().numpy() # (1, C, W)
                    curr_v_prior_np = np.transpose(curr_v_prior_np, (0, 2, 1)) # (1, W, C)
                    curr_v_prior_part = curr_v_prior_np[0, :overlap_len, :]
                    
                    # 真正的 residual 参考值
                    ref_residual = ref_part - curr_v_prior_part
                    
                    # 更新 masked_image 为 residual
                    masked_image_np = np.zeros((window_size, target_dim))
                    masked_image_np[:overlap_len] = ref_residual
                    masked_image = torch.from_numpy(masked_image_np).float().unsqueeze(0).permute(0, 2, 1).to(device)

        # 4. 采样循环 (In-painting)
        # 初始化噪声
        xt = torch.randn(1, target_dim, window_size, device=device)
        
        for t in system.scheduler.timesteps:
            # --- In-painting Step ---
            # 如果有 Mask，强制将已知区域替换为加噪后的参考值
            if mask is not None:
                # 计算参考值在当前时刻 t 的加噪状态
                # q_sample(x_start, t, noise)
                noise_ref = torch.randn_like(masked_image)
                # 注意：diffusers 的 add_noise 需要 t 是 tensor
                t_tensor = torch.tensor([t], device=device)
                
                # 获取该时刻的“已知部分”的噪声状态
                xt_known = system.scheduler.add_noise(masked_image, noise_ref, t_tensor)
                
                # 融合：已知区域用 xt_known，未知区域用当前的 xt
                xt = mask * xt_known + (1 - mask) * xt
            # ------------------------
            
            # Predict noise
            t_batch = torch.full((1,), t, device=device, dtype=torch.long)
            
            # 兼容旧模型接口 (unet 输入不含 v_prior)
            # 如果您是在测试新模型，这里需要修改
            model_output = system.unet(xt, t_batch, cond_feat)
            
            # Step
            xt = system.scheduler.step(model_output, t, xt).prev_sample
            
        # 5. 保存结果
        pred_curr = xt # (1, C, W)
        
        # 如果是 Residual 模式，加上 v_prior
        if mode == 'residual':
            pred_curr = pred_curr + v_prior
            
        pred_curr_np = pred_curr.squeeze(0).detach().permute(1, 0).cpu().numpy() # (W, C)
        
        # 更新全局结果
        # 策略：直接覆盖。因为我们强制约束了前部，所以覆盖也不会有跳变
        # 但为了最平滑，非重叠部分直接写入，重叠部分其实理论上是一样的
        
        if i == 0:
            full_pred[start_idx:end_idx] = pred_curr_np
        else:
            # 只写入新生成的部分 (非重叠部分)
            # prev_start + window_size 是上一窗口的结束点，也就是当前窗口 overlap_len 的位置
            new_part_start_in_window = overlap_len
            global_write_start = start_idx + new_part_start_in_window
            
            full_pred[global_write_start:end_idx] = pred_curr_np[new_part_start_in_window:]
            
        # 更新缓冲区供下一次使用
        prev_window_pred = pred_curr_np

    return full_pred


def reconstruct_weighted(system, imu_seq, window_size, stride, device, num_steps, mode="end2end"):
    """
    使用线性加权平均 (Weighted Blending) 策略重建轨迹
    这是一种比硬切 (Hard Cut) 更稳健的非自回归策略，用于诊断拼接问题。
    """
    L = imu_seq.shape[0]
    target_dim = system.unet.out_conv[-1].out_channels
    starts = np.arange(0, L - window_size + 1, stride)
    if starts[-1] + window_size < L:
        starts = np.concatenate([starts, [L - window_size]])
    
    full_pred = np.zeros((L, target_dim))
    weights = np.zeros((L, 1))
    
    # 构造窗口权重 (两头小中间大，平滑过渡)
    # 也可以简单使用线性权重
    window_weight = np.ones(window_size)
    ramp_len = min(stride, window_size // 4)
    if ramp_len > 0:
        ramp = np.linspace(0, 1, ramp_len)
        window_weight[:ramp_len] = ramp
        window_weight[-ramp_len:] = ramp[::-1]
    window_weight = window_weight[:, np.newaxis]

    system.eval()
    system.scheduler.set_timesteps(num_steps)
    
    for start_idx in tqdm(starts, desc="Weighted Blending", leave=False):
        end_idx = start_idx + window_size
        imu_window = imu_seq[start_idx:end_idx]
        imu_tensor = torch.from_numpy(imu_window).float().unsqueeze(0).permute(0, 2, 1).to(device)
        
        with torch.no_grad():
            cond_feat = system.encoder(imu_tensor)
            v_prior = None
            if mode == 'residual':
                v_prior_feat = system.prior_head(cond_feat)
                v_prior = torch.nn.functional.interpolate(v_prior_feat, size=window_size, mode='linear', align_corners=False)
            
            xt = torch.randn(1, target_dim, window_size, device=device)
            for t in system.scheduler.timesteps:
                t_batch = torch.full((1,), t, device=device, dtype=torch.long)
                model_output = system.unet(xt, t_batch, cond_feat)
                xt = system.scheduler.step(model_output, t, xt).prev_sample
            
            pred_curr = xt
            if mode == 'residual':
                pred_curr = pred_curr + v_prior
            
            pred_curr_np = pred_curr.squeeze(0).detach().permute(1, 0).cpu().numpy()
            
            # 累加预测值和权重
            full_pred[start_idx:end_idx] += pred_curr_np * window_weight
            weights[start_idx:end_idx] += window_weight

    # 归一化
    full_pred /= (weights + 1e-8)
    return full_pred


def plot_detailed_diagnosis(gt_vel, pred_vel, gt_pos, pred_pos, seq_name, out_dir):
    """
    生成深度诊断图：轨迹对比 + Vx/Vy 分量对比
    """
    fig = plt.figure(figsize=(15, 10))
    
    # 1. 轨迹图
    ax1 = plt.subplot(2, 2, 1)
    ax1.plot(gt_pos[:, 0], gt_pos[:, 1], 'k-', label='GT')
    ax1.plot(pred_pos[:, 0], pred_pos[:, 1], 'r--', label='Pred')
    ax1.set_title(f"Trajectory: {seq_name}")
    ax1.legend()
    ax1.axis('equal')
    
    # 2. Vx 速度分量
    ax2 = plt.subplot(2, 2, 3)
    ax2.plot(gt_vel[:, 0], 'k-', alpha=0.5, label='GT Vx')
    ax2.plot(pred_vel[:, 0], 'r-', alpha=0.8, label='Pred Vx')
    ax2.set_title("Velocity X-component")
    ax2.set_xlabel("Time Step")
    ax2.legend()
    
    # 3. Vy 速度分量
    ax3 = plt.subplot(2, 2, 4)
    ax3.plot(gt_vel[:, 1], 'k-', alpha=0.5, label='GT Vy')
    ax3.plot(pred_vel[:, 1], 'g-', alpha=0.8, label='Pred Vy')
    ax3.set_title("Velocity Y-component")
    ax3.set_xlabel("Time Step")
    ax3.legend()

    # 4. 速度误差分布
    ax4 = plt.subplot(2, 2, 2)
    error = np.linalg.norm(pred_vel - gt_vel, axis=1)
    ax4.hist(error, bins=50, color='gray', alpha=0.7)
    ax4.axvline(np.mean(error), color='r', linestyle='--', label=f'Mean: {np.mean(error):.4f}')
    ax4.set_title("Velocity Error Distribution")
    ax4.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"diag_{seq_name}.png"), dpi=150)
    plt.close()


def evaluate_autoregressive(args, config):
    device = torch.device('cuda:0' if torch.cuda.is_available() and not args.cpu else 'cpu')
    print(f"Using device: {device}")
    
    # 1. 加载模型
    input_dim = 6
    target_dim = config.get('target_dim', 2)
    mode = config.get('mode', 'end2end')
    
    print(f"Mode: {mode}, Target dim: {target_dim}")
    print(f"Method: {args.method}")
    
    encoder = ResNet1D(num_inputs=input_dim, num_outputs=None, block_type=BasicBlock1D, group_sizes=[2, 2, 2, 2], output_block=None)
    unet = DiffUNet1D(in_channels=target_dim, out_channels=target_dim, cond_channels=512, base_channels=64, channel_mults=(1, 2, 4, 8))
    
    system = DiffusionSystem(encoder, unet, mode=mode).to(device)
    
    if mode == 'residual' and system.prior_head.out_channels != target_dim:
        system.prior_head = nn.Conv1d(512, target_dim, kernel_size=1).to(device)
    
    if args.model_path and os.path.exists(args.model_path):
        print(f"Loading model from {args.model_path}")
        system.load_state_dict(torch.load(args.model_path, map_location=device))
    elif args.model_path:
        print(f"Warning: Model not found at {args.model_path}, using random weights for dry run.")
    
    system.set_scheduler(args.scheduler)
    system.eval()
    
    results_dir = args.out_dir
    os.makedirs(results_dir, exist_ok=True)
    
    test_lists = {
        "seen": config.get('test_seen_list', 'data/RoNIN/lists/list_test_seen.txt'),
        "unseen": config.get('test_unseen_list', 'data/RoNIN/lists/list_test_unseen.txt')
    }
    
    all_metrics = []
    
    for split_name, list_path in test_lists.items():
        if not os.path.exists(list_path): continue
        
        print(f"\nEvaluating {split_name} split...")
        with open(list_path, 'r') as f:
            seq_names = [line.strip() for line in f if line.strip() and line[0] != '#']
        
        # 排除异常序列 (如 009, 011)
        exclude_patterns = ['a009', 'a011']
        filtered_seqs = []
        for sn in seq_names:
            if any(p in sn for p in exclude_patterns):
                print(f"  [Skip] Excluding anomalous sequence: {sn}")
                continue
            filtered_seqs.append(sn)
        seq_names = filtered_seqs
        
        if args.max_seqs > 0:
            seq_names = seq_names[:args.max_seqs]
            
        for seq_name in tqdm(seq_names, desc=split_name):
            seq_path = os.path.join(config['data_dir'], seq_name)
            try:
                seq_data = GlobSpeedSequence(seq_path)
            except:
                continue
                
            imu = seq_data.get_feature()
            gt_vel = seq_data.get_target()
            gt_pos = seq_data.gt_pos[:, :2]
            
            min_len = min(imu.shape[0], gt_vel.shape[0], gt_pos.shape[0])
            imu, gt_vel, gt_pos = imu[:min_len], gt_vel[:min_len], gt_pos[:min_len]

            # 自动清洗 GT 速度异常点 (防止 Tango 视觉跳变干扰评估)
            # 人类步行速度上限约为 3m/s，微分异常可能导致 20m/s+ 的假值
            v_norm = np.linalg.norm(gt_vel, axis=1)
            bad_mask = v_norm > 5.0 # 设为 5m/s 以防万一 (极速奔跑)
            if np.any(bad_mask):
                valid_indices = np.where(~bad_mask)[0]
                if len(valid_indices) > 0:
                    # 简单插值处理
                    for i in np.where(bad_mask)[0]:
                        nearest_valid = valid_indices[np.abs(valid_indices - i).argmin()]
                        gt_vel[i] = gt_vel[nearest_valid]
            
            window_size = config.get('window_size', 200)
            
            # 推理策略选择
            if args.method == 'autoregressive':
                pred_vel = reconstruct_autoregressive(system, imu, window_size, args.stride, device, args.steps, mode)
            else:
                pred_vel = reconstruct_weighted(system, imu, window_size, args.stride, device, args.steps, mode)
            
            min_len_final = min(pred_vel.shape[0], gt_pos.shape[0])
            pred_vel, gt_pos, gt_vel = pred_vel[:min_len_final], gt_pos[:min_len_final], gt_vel[:min_len_final]
            
            pred_pos = integrate_trajectory(pred_vel, initial_pos=gt_pos[0], dt=0.005)
            ate, rte = compute_ate_rte(pred_pos[:, :2], gt_pos, 200 * 60)
            
            all_metrics.append({
                "split": split_name, "seq": seq_name, "ate": ate, "rte": rte, "mean_vel_error": np.mean(np.linalg.norm(pred_vel - gt_vel, axis=1))
            })
            
            if args.plot:
                plot_detailed_diagnosis(gt_vel, pred_vel, gt_pos, pred_pos, seq_name, results_dir)

    df = pd.DataFrame(all_metrics)
    df.to_csv(os.path.join(results_dir, "metrics.csv"), index=False)
    print("\nSummary:")
    print(df.groupby('split')[['ate', 'rte', 'mean_vel_error']].mean())
    return df


def plot_traj(gt, pred, name, ate, out_dir):
    plt.figure(figsize=(6, 6))
    plt.plot(gt[:, 0], gt[:, 1], 'k-', label='GT')
    plt.plot(pred[:, 0], pred[:, 1], 'r-', label=f'AutoReg (ATE={ate:.2f})')
    plt.legend()
    plt.title(name)
    plt.axis('equal')
    plt.savefig(os.path.join(out_dir, f"{name}.png"))
    plt.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--out_dir', type=str, default='experiments/results_diagnostic')
    parser.add_argument('--method', type=str, choices=['autoregressive', 'weighted'], default='autoregressive',
                        help='Stitching method: autoregressive (in-painting) or weighted (linear blending)')
    parser.add_argument('--scheduler', type=str, default='ddim')
    parser.add_argument('--steps', type=int, default=50)
    parser.add_argument('--stride', type=int, default=100)
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--plot', action='store_true')
    parser.add_argument('--max_seqs', type=int, default=-1)
    
    args = parser.parse_args()
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
        
    evaluate_autoregressive(args, config)
