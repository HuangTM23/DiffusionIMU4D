"""
扩散模型测试脚本 - 带完整可视化分析（专门针对 Residual 模式）

功能：
1. 速度可视化（GT vs Pred，以及 v_prior + residual 分解）
2. 速度残差可视化
3. 轨迹可视化
4. 滑动窗口重叠区域连续性分析
5. v_prior 单独评估
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
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
from utils.inference_utils import reconstruct_trajectory, integrate_trajectory
from utils.metric import compute_ate_rte


def evaluate_with_analysis(args, config):
    """执行评估并生成详细分析可视化"""
    device = torch.device('cuda:0' if torch.cuda.is_available() and not args.cpu else 'cpu')
    print(f"Using device: {device}")
    
    # 1. 加载模型
    input_dim = 6
    target_dim = config.get('target_dim', 2)
    mode = config.get('mode', 'end2end')
    
    print(f"Mode: {mode}, Target dim: {target_dim}")
    
    encoder = ResNet1D(
        num_inputs=input_dim, 
        num_outputs=None, 
        block_type=BasicBlock1D, 
        group_sizes=[2, 2, 2, 2],
        output_block=None
    )
    
    unet = DiffUNet1D(
        in_channels=target_dim, 
        out_channels=target_dim, 
        cond_channels=512, 
        base_channels=64,
        channel_mults=(1, 2, 4, 8)
    )
    
    system = DiffusionSystem(encoder, unet, mode=mode).to(device)
    
    # 修复 residual 模式的 prior_head
    if mode == 'residual' and system.prior_head.out_channels != target_dim:
        system.prior_head = nn.Conv1d(512, target_dim, kernel_size=1).to(device)
    
    # 加载权重
    if args.model_path and os.path.exists(args.model_path):
        print(f"Loading model from {args.model_path}")
        system.load_state_dict(torch.load(args.model_path, map_location=device))
    else:
        raise ValueError(f"Model not found: {args.model_path}")
    
    system.set_scheduler(args.scheduler)
    system.eval()
    
    # 2. 准备输出目录
    results_dir = args.out_dir or 'experiments/results_analysis'
    os.makedirs(results_dir, exist_ok=True)
    
    # 3. 加载测试序列
    test_lists = {
        "seen": config.get('test_seen_list', 'data/RoNIN/lists/list_test_seen.txt'),
        "unseen": config.get('test_unseen_list', 'data/RoNIN/lists/list_test_unseen.txt')
    }
    
    all_metrics = []
    
    for split_name, list_path in test_lists.items():
        if not os.path.exists(list_path):
            print(f"List {list_path} not found, skipping {split_name}.")
            continue
        
        print(f"\n{'='*60}")
        print(f"Evaluating {split_name} split...")
        print(f"{'='*60}")
        
        with open(list_path, 'r') as f:
            seq_names = [line.strip() for line in f if line.strip() and line[0] != '#']
        
        # 限制分析序列数量
        if args.max_seqs > 0:
            seq_names = seq_names[:args.max_seqs]
        
        for seq_name in tqdm(seq_names, desc=f"Processing {split_name}"):
            seq_path = os.path.join(config['data_dir'], seq_name)
            
            try:
                seq_data = GlobSpeedSequence(seq_path)
            except Exception as e:
                print(f"Error loading {seq_name}: {e}")
                continue
            
            imu = seq_data.get_feature()  # (L, 6)
            gt_vel = seq_data.get_target()  # (L, 2)
            gt_pos = seq_data.gt_pos[:, :2]  # (L, 2)
            
            # 4. 推理
            window_size = config.get('window_size', 200)
            stride = args.stride
            
            # 获取详细的窗口预测（用于分析连续性）
            pred_vel, window_details = reconstruct_trajectory_with_details(
                system, imu, window_size, stride, device, args.steps, mode
            )
            
            # 5. 积分得到轨迹
            pred_pos = integrate_trajectory(pred_vel, initial_pos=gt_pos[0], dt=0.005)
            
            # 6. 计算指标
            pred_per_min = 200 * 60
            ate, rte = compute_ate_rte(pred_pos[:, :2], gt_pos, pred_per_min)
            
            # 计算速度误差
            vel_error = np.linalg.norm(pred_vel - gt_vel, axis=1)
            mean_vel_error = np.mean(vel_error)
            max_vel_error = np.max(vel_error)
            
            all_metrics.append({
                "split": split_name,
                "seq": seq_name,
                "ate": ate,
                "rte": rte,
                "mean_vel_error": mean_vel_error,
                "max_vel_error": max_vel_error,
                "length": len(gt_vel)
            })
            
            # 7. 生成可视化
            if args.plot:
                seq_viz_dir = os.path.join(results_dir, f"{split_name}_{seq_name}")
                os.makedirs(seq_viz_dir, exist_ok=True)
                
                # 速度可视化
                plot_velocity_comparison(
                    gt_vel, pred_vel, window_details, 
                    seq_name, mode, os.path.join(seq_viz_dir, "velocity.png")
                )
                
                # 轨迹可视化
                plot_trajectory_comparison(
                    gt_pos, pred_pos, seq_name, ate, rte,
                    os.path.join(seq_viz_dir, "trajectory.png")
                )
                
                # 误差时序图
                plot_error_timeseries(
                    gt_vel, pred_vel, gt_pos, pred_pos, seq_name,
                    os.path.join(seq_viz_dir, "errors.png")
                )
                
                # Residual 模式特有：v_prior 和 residual 分析
                if mode == 'residual' and window_details is not None:
                    plot_residual_analysis(
                        gt_vel, window_details, seq_name,
                        os.path.join(seq_viz_dir, "residual_analysis.png")
                    )
                
                # 滑动窗口连续性分析
                if window_details is not None and len(window_details) > 1:
                    plot_window_continuity(
                        window_details, stride, seq_name,
                        os.path.join(seq_viz_dir, "window_continuity.png")
                    )
    
    # 8. 保存汇总结果
    df = pd.DataFrame(all_metrics)
    summary_path = os.path.join(results_dir, "metrics_detailed.csv")
    df.to_csv(summary_path, index=False)
    
    # 9. 生成全局分析图表
    plot_global_analysis(df, results_dir)
    
    print("\n" + "="*60)
    print("Evaluation Summary:")
    print("="*60)
    print(df.groupby('split')[['ate', 'rte', 'mean_vel_error']].mean())
    print(f"\nResults saved to {results_dir}")
    
    return df


def reconstruct_trajectory_with_details(system, imu_seq, window_size, stride, device, 
                                        num_steps, mode):
    """
    重建轨迹并返回每个窗口的详细信息（用于分析）
    
    Returns:
        pred_vel: (L, C) 重建的速度序列
        window_details: list of dict，每个窗口的详细信息
    """
    L = imu_seq.shape[0]
    target_dim = system.unet.out_conv[-1].out_channels
    
    # 准备窗口
    if L < window_size:
        raise ValueError(f"Sequence length {L} < window size {window_size}")
    
    starts = np.arange(0, L - window_size + 1, stride)
    if starts[-1] + window_size < L:
        starts = np.concatenate([starts, [L - window_size]])
    
    windows = [imu_seq[s:s+window_size] for s in starts]
    windows = np.stack(windows)  # (N, W, 6)
    
    # 批量推理
    batch_size = 32
    num_windows = len(windows)
    window_details = []
    
    for i in range(0, num_windows, batch_size):
        batch_in = windows[i:i+batch_size]
        batch_starts = starts[i:i+batch_size]
        batch_in_torch = torch.from_numpy(batch_in).float().permute(0, 2, 1).to(device)
        
        with torch.no_grad():
            # 获取 encoder 特征
            cond_feat = system.encoder(batch_in_torch)
            
            if mode == 'residual':
                # 获取 v_prior
                v_prior_feat = system.prior_head(cond_feat)
                v_prior = torch.nn.functional.interpolate(
                    v_prior_feat, size=window_size, mode='linear', align_corners=False
                )
                v_prior_np = v_prior.permute(0, 2, 1).cpu().numpy()  # (B, W, C)
            
            # 扩散采样
            pred_window = system.sample(batch_in_torch, num_inference_steps=num_steps)
            pred_window_np = pred_window.permute(0, 2, 1).cpu().numpy()  # (B, W, C)
            
            # 如果是 residual 模式，计算残差
            if mode == 'residual':
                residual_np = pred_window_np - v_prior_np
            else:
                v_prior_np = None
                residual_np = None
        
        # 保存每个窗口的详细信息
        for j, (start_idx, pred_vel, prior, residual) in enumerate(
            zip(batch_starts, pred_window_np, 
                v_prior_np if v_prior_np is not None else [None]*len(pred_window_np),
                residual_np if residual_np is not None else [None]*len(pred_window_np))
        ):
            window_details.append({
                'start': start_idx,
                'end': start_idx + window_size,
                'pred_vel': pred_vel,  # (W, C)
                'v_prior': prior,  # (W, C) or None
                'residual': residual,  # (W, C) or None
            })
    
    # 拼接（使用加权平均）
    recon_vel = np.zeros((L, target_dim))
    weights = np.zeros((L, 1))
    window_weight = np.ones((window_size, 1))
    
    for detail in window_details:
        s, e = detail['start'], detail['end']
        recon_vel[s:e] += detail['pred_vel'] * window_weight
        weights[s:e] += window_weight
    
    weights[weights == 0] = 1.0
    recon_vel /= weights
    
    return recon_vel, window_details


def plot_velocity_comparison(gt_vel, pred_vel, window_details, seq_name, mode, save_path):
    """绘制速度对比图"""
    fig, axes = plt.subplots(2, 1, figsize=(15, 8))
    
    # X 方向速度
    axes[0].plot(gt_vel[:, 0], label='GT Vx', color='black', linewidth=1.5)
    axes[0].plot(pred_vel[:, 0], label='Pred Vx', color='red', linewidth=1.0, alpha=0.8)
    axes[0].set_ylabel('Vx (m/s)')
    axes[0].set_title(f'Sequence: {seq_name} - Velocity Comparison ({mode} mode)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Y 方向速度
    axes[1].plot(gt_vel[:, 1], label='GT Vy', color='black', linewidth=1.5)
    axes[1].plot(pred_vel[:, 1], label='Pred Vy', color='red', linewidth=1.0, alpha=0.8)
    axes[1].set_ylabel('Vy (m/s)')
    axes[1].set_xlabel('Frame')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_trajectory_comparison(gt_pos, pred_pos, seq_name, ate, rte, save_path):
    """绘制轨迹对比图"""
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # 2D 轨迹
    axes[0].plot(gt_pos[:, 0], gt_pos[:, 1], label='Ground Truth', 
                 color='black', linewidth=2, alpha=0.7)
    axes[0].plot(pred_pos[:, 0], pred_pos[:, 1], label='Predicted', 
                 color='red', linewidth=1.5, alpha=0.8)
    axes[0].set_xlabel('X (m)')
    axes[0].set_ylabel('Y (m)')
    axes[0].set_title(f'Trajectory Comparison\nATE: {ate:.3f}m, RTE: {rte:.3f}m/min')
    axes[0].axis('equal')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # 位置误差随时间变化
    pos_errors = np.linalg.norm(gt_pos - pred_pos[:, :2], axis=1)
    axes[1].plot(pos_errors, color='blue', linewidth=1)
    axes[1].axhline(y=np.mean(pos_errors), color='r', linestyle='--', 
                    label=f'Mean: {np.mean(pos_errors):.2f}m')
    axes[1].set_xlabel('Frame')
    axes[1].set_ylabel('Position Error (m)')
    axes[1].set_title('Position Error over Time')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_error_timeseries(gt_vel, pred_vel, gt_pos, pred_pos, seq_name, save_path):
    """绘制误差时序分析图"""
    fig, axes = plt.subplots(3, 1, figsize=(15, 10))
    
    # 速度误差
    vel_error = np.linalg.norm(pred_vel - gt_vel, axis=1)
    axes[0].plot(vel_error, color='green', linewidth=1)
    axes[0].axhline(y=np.mean(vel_error), color='r', linestyle='--',
                    label=f'Mean: {np.mean(vel_error):.3f} m/s')
    axes[0].set_ylabel('Velocity Error (m/s)')
    axes[0].set_title(f'Sequence: {seq_name} - Error Analysis')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # 各方向速度误差
    axes[1].plot(np.abs(pred_vel[:, 0] - gt_vel[:, 0]), label='|Vx Error|', alpha=0.7)
    axes[1].plot(np.abs(pred_vel[:, 1] - gt_vel[:, 1]), label='|Vy Error|', alpha=0.7)
    axes[1].set_ylabel('Absolute Velocity Error (m/s)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # 位置误差
    pos_errors = np.linalg.norm(gt_pos - pred_pos[:, :2], axis=1)
    axes[2].plot(pos_errors, color='blue', linewidth=1)
    axes[2].axhline(y=np.mean(pos_errors), color='r', linestyle='--',
                    label=f'Mean: {np.mean(pos_errors):.2f}m')
    axes[2].set_ylabel('Position Error (m)')
    axes[2].set_xlabel('Frame')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_residual_analysis(gt_vel, window_details, seq_name, save_path):
    """
    Residual 模式特有：分析 v_prior 和 residual
    显示：GT, v_prior, residual, 最终预测
    """
    # 重建完整的 v_prior 和 residual 序列
    L = len(gt_vel)
    v_prior_full = np.zeros_like(gt_vel)
    residual_full = np.zeros_like(gt_vel)
    weights = np.zeros((L, 1))
    
    window_weight = np.ones((window_details[0]['pred_vel'].shape[0], 1))
    
    for detail in window_details:
        s, e = detail['start'], detail['end']
        v_prior_full[s:e] += detail['v_prior'] * window_weight
        residual_full[s:e] += detail['residual'] * window_weight
        weights[s:e] += window_weight
    
    weights[weights == 0] = 1.0
    v_prior_full /= weights
    residual_full /= weights
    
    fig, axes = plt.subplots(4, 2, figsize=(16, 12))
    
    # Vx 方向
    axes[0, 0].plot(gt_vel[:, 0], label='GT', color='black')
    axes[0, 0].set_ylabel('GT Vx')
    axes[0, 0].set_title(f'{seq_name} - Residual Mode Decomposition (Vx)')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[1, 0].plot(v_prior_full[:, 0], label='v_prior', color='blue')
    axes[1, 0].set_ylabel('v_prior Vx')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[2, 0].plot(residual_full[:, 0], label='residual', color='green')
    axes[2, 0].set_ylabel('Residual Vx')
    axes[2, 0].legend()
    axes[2, 0].grid(True, alpha=0.3)
    
    pred_vel = v_prior_full + residual_full
    axes[3, 0].plot(gt_vel[:, 0], label='GT', color='black', alpha=0.5)
    axes[3, 0].plot(pred_vel[:, 0], label='Pred (prior+residual)', color='red')
    axes[3, 0].set_ylabel('Final Vx')
    axes[3, 0].set_xlabel('Frame')
    axes[3, 0].legend()
    axes[3, 0].grid(True, alpha=0.3)
    
    # Vy 方向
    axes[0, 1].plot(gt_vel[:, 1], label='GT', color='black')
    axes[0, 1].set_ylabel('GT Vy')
    axes[0, 1].set_title(f'{seq_name} - Residual Mode Decomposition (Vy)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    axes[1, 1].plot(v_prior_full[:, 1], label='v_prior', color='blue')
    axes[1, 1].set_ylabel('v_prior Vy')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    axes[2, 1].plot(residual_full[:, 1], label='residual', color='green')
    axes[2, 1].set_ylabel('Residual Vy')
    axes[2, 1].legend()
    axes[2, 1].grid(True, alpha=0.3)
    
    axes[3, 1].plot(gt_vel[:, 1], label='GT', color='black', alpha=0.5)
    axes[3, 1].plot(pred_vel[:, 1], label='Pred (prior+residual)', color='red')
    axes[3, 1].set_ylabel('Final Vy')
    axes[3, 1].set_xlabel('Frame')
    axes[3, 1].legend()
    axes[3, 1].grid(True, alpha=0.3)
    
    # 计算并显示统计信息
    prior_error = np.linalg.norm(v_prior_full - gt_vel, axis=1)
    final_error = np.linalg.norm(pred_vel - gt_vel, axis=1)
    
    fig.text(0.5, 0.02, 
             f'v_prior MAE: {np.mean(prior_error):.3f} m/s | '
             f'Final MAE: {np.mean(final_error):.3f} m/s | '
             f'Improvement: {(1-np.mean(final_error)/np.mean(prior_error))*100:.1f}%',
             ha='center', fontsize=10, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout(rect=[0, 0.03, 1, 1])
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_window_continuity(window_details, stride, seq_name, save_path):
    """
    分析滑动窗口之间的连续性
    显示相邻窗口在重叠区域的差异
    """
    if len(window_details) < 2:
        return
    
    # 计算相邻窗口在边界处的差异
    boundary_diffs = []
    boundary_positions = []
    
    for i in range(len(window_details) - 1):
        curr_end = window_details[i]['end']
        next_start = window_details[i+1]['start']
        
        # 找到重叠区域
        overlap_start = max(window_details[i]['start'], next_start)
        overlap_end = min(curr_end, window_details[i+1]['end'])
        
        if overlap_start < overlap_end:
            # 计算重叠区域内的差异
            curr_overlap = window_details[i]['pred_vel'][overlap_start - window_details[i]['start']:
                                                         overlap_end - window_details[i]['start']]
            next_overlap = window_details[i+1]['pred_vel'][overlap_start - window_details[i+1]['start']:
                                                           overlap_end - window_details[i+1]['start']]
            
            diff = np.mean(np.linalg.norm(curr_overlap - next_overlap, axis=1))
            boundary_diffs.append(diff)
            boundary_positions.append(overlap_start)
    
    if len(boundary_diffs) == 0:
        return
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    
    # 绘制边界差异
    axes[0].bar(range(len(boundary_diffs)), boundary_diffs, color='coral', alpha=0.7)
    axes[0].axhline(y=np.mean(boundary_diffs), color='red', linestyle='--',
                    label=f'Mean: {np.mean(boundary_diffs):.3f} m/s')
    axes[0].set_ylabel('Boundary Discontinuity (m/s)')
    axes[0].set_title(f'{seq_name} - Window Continuity Analysis (stride={stride})')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # 绘制第一个窗口和第二个窗口的重叠区域对比
    if len(window_details) >= 2:
        w1, w2 = window_details[0], window_details[1]
        overlap_start = w2['start']
        overlap_end = min(w1['end'], w2['end'])
        
        w1_overlap = w1['pred_vel'][overlap_start - w1['start']:]
        w2_overlap = w2['pred_vel'][:overlap_end - w2['start']]
        
        x_axis = np.arange(overlap_start, overlap_end)
        
        axes[1].plot(x_axis, w1_overlap[:, 0], label='Window 1 Vx', color='blue', alpha=0.7)
        axes[1].plot(x_axis, w2_overlap[:, 0], label='Window 2 Vx', color='red', alpha=0.7)
        axes[1].axvline(x=w1['start'] + stride, color='green', linestyle='--', 
                        label=f'Window boundary (stride={stride})')
        axes[1].set_ylabel('Vx (m/s)')
        axes[1].set_xlabel('Frame')
        axes[1].set_title('Example: First Two Windows Overlap')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_global_analysis(df, results_dir):
    """生成全局分析图表"""
    if df.empty:
        return
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # 按 split 分组的 ATE 分布
    for split in df['split'].unique():
        split_data = df[df['split'] == split]
        axes[0, 0].hist(split_data['ate'], bins=15, alpha=0.5, label=split)
    axes[0, 0].set_xlabel('ATE (m)')
    axes[0, 0].set_title('ATE Distribution by Split')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 按 split 分组的 RTE 分布
    for split in df['split'].unique():
        split_data = df[df['split'] == split]
        axes[0, 1].hist(split_data['rte'], bins=15, alpha=0.5, label=split)
    axes[0, 1].set_xlabel('RTE (m/min)')
    axes[0, 1].set_title('RTE Distribution by Split')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 速度误差 vs ATE
    axes[0, 2].scatter(df['mean_vel_error'], df['ate'], c=df['split'].astype('category').cat.codes, 
                       alpha=0.6, cmap='viridis')
    axes[0, 2].set_xlabel('Mean Velocity Error (m/s)')
    axes[0, 2].set_ylabel('ATE (m)')
    axes[0, 2].set_title('Velocity Error vs ATE')
    axes[0, 2].grid(True, alpha=0.3)
    
    # 序列长度 vs ATE
    axes[1, 0].scatter(df['length'], df['ate'], alpha=0.6)
    axes[1, 0].set_xlabel('Sequence Length')
    axes[1, 0].set_ylabel('ATE (m)')
    axes[1, 0].set_title('Sequence Length vs ATE')
    axes[1, 0].grid(True, alpha=0.3)
    
    # ATE vs RTE
    axes[1, 1].scatter(df['ate'], df['rte'], alpha=0.6)
    axes[1, 1].set_xlabel('ATE (m)')
    axes[1, 1].set_ylabel('RTE (m/min)')
    axes[1, 1].set_title('ATE vs RTE')
    axes[1, 1].grid(True, alpha=0.3)
    
    # 统计表格
    axes[1, 2].axis('off')
    summary = df.groupby('split')[['ate', 'rte', 'mean_vel_error']].agg(['mean', 'std'])
    table_data = []
    for split in summary.index:
        row = [split,
               f"{summary.loc[split, ('ate', 'mean')]:.2f}±{summary.loc[split, ('ate', 'std')]:.2f}",
               f"{summary.loc[split, ('rte', 'mean')]:.2f}±{summary.loc[split, ('rte', 'std')]:.2f}",
               f"{summary.loc[split, ('mean_vel_error', 'mean')]:.3f}±{summary.loc[split, ('mean_vel_error', 'std')]:.3f}"]
        table_data.append(row)
    
    table = axes[1, 2].table(cellText=table_data,
                              colLabels=['Split', 'ATE (m)', 'RTE (m/min)', 'Vel Error (m/s)'],
                              cellLoc='center',
                              loc='center',
                              bbox=[0, 0.3, 1, 0.4])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    axes[1, 2].set_title('Summary Statistics')
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'global_analysis.png'), dpi=150)
    plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Diffusion Model Test with Detailed Analysis')
    parser.add_argument('--config', type=str, required=True, help='Config file path')
    parser.add_argument('--model_path', type=str, required=True, help='Model checkpoint path')
    parser.add_argument('--out_dir', type=str, default='experiments/results_analysis', 
                        help='Output directory')
    parser.add_argument('--scheduler', type=str, default='ddim', choices=['ddpm', 'ddim'],
                        help='Sampling scheduler')
    parser.add_argument('--steps', type=int, default=50, help='Number of inference steps')
    parser.add_argument('--stride', type=int, default=100, help='Sliding window stride')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for inference')
    parser.add_argument('--cpu', action='store_true', help='Use CPU')
    parser.add_argument('--plot', action='store_true', default=True, help='Generate visualizations')
    parser.add_argument('--max_seqs', type=int, default=-1, 
                        help='Max sequences to analyze (-1 for all)')
    
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    evaluate_with_analysis(args, config)
