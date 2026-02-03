"""
单独测试 v_prior 的质量（不使用 Diffusion）
用于快速诊断 Residual 模式下 prior_head 的学习效果
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
from utils.inference_utils import reconstruct_trajectory, integrate_trajectory
from utils.metric import compute_ate_rte


def evaluate_prior_only(args, config):
    """仅使用 v_prior 进行预测，不使用 diffusion"""
    device = torch.device('cuda:0' if torch.cuda.is_available() and not args.cpu else 'cpu')
    print(f"Using device: {device}")
    print("="*60)
    print("评估 v_prior 单独性能（无 Diffusion）")
    print("="*60)
    
    # 1. 加载模型
    input_dim = 6
    target_dim = config.get('target_dim', 2)
    mode = config.get('mode', 'end2end')
    
    if mode != 'residual':
        print("警告：当前配置不是 residual 模式，但继续执行...")
    
    encoder = ResNet1D(
        num_inputs=input_dim, 
        num_outputs=None, 
        block_type=BasicBlock1D, 
        group_sizes=[2, 2, 2, 2],
        output_block=None
    )
    
    # [Modified] Residual 模式输入通道翻倍 (保持与其他脚本一致)
    unet_in_channels = target_dim * 2 if mode == 'residual' else target_dim
    
    unet = DiffUNet1D(
        in_channels=unet_in_channels, 
        out_channels=target_dim, 
        cond_channels=512, 
        base_channels=64,
        channel_mults=(1, 2, 4, 8)
    )
    
    system = DiffusionSystem(encoder, unet, mode=mode).to(device)
    
    if mode == 'residual' and system.prior_head.out_channels != target_dim:
        system.prior_head = nn.Conv1d(512, target_dim, kernel_size=1).to(device)
    
    # 加载权重
    if args.model_path and os.path.exists(args.model_path):
        print(f"Loading model from {args.model_path}")
        system.load_state_dict(torch.load(args.model_path, map_location=device))
    else:
        raise ValueError(f"Model not found: {args.model_path}")
    
    system.eval()
    
    # 2. 准备输出
    results_dir = args.out_dir or 'experiments/prior_only_results'
    os.makedirs(results_dir, exist_ok=True)
    
    # 3. 加载测试数据
    test_lists = {
        "seen": config.get('test_seen_list', 'data/RoNIN/lists/list_test_seen.txt'),
        "unseen": config.get('test_unseen_list', 'data/RoNIN/lists/list_test_unseen.txt')
    }
    
    all_metrics = []
    all_residual_stats = []
    
    for split_name, list_path in test_lists.items():
        if not os.path.exists(list_path):
            continue
        
        print(f"\nEvaluating {split_name} split...")
        with open(list_path, 'r') as f:
            seq_names = [line.strip() for line in f if line.strip() and line[0] != '#']
        
        if args.max_seqs > 0:
            seq_names = seq_names[:args.max_seqs]
        
        for seq_name in tqdm(seq_names, desc=split_name):
            seq_path = os.path.join(config['data_dir'], seq_name)
            
            try:
                seq_data = GlobSpeedSequence(seq_path)
            except Exception as e:
                print(f"Error loading {seq_name}: {e}")
                continue
            
            imu = seq_data.get_feature()  # (L, 6)
            gt_vel = seq_data.get_target()  # (L, 2)
            gt_pos = seq_data.gt_pos[:, :2]  # (L, 2)
            
            # 4. 仅使用 v_prior 预测
            window_size = config.get('window_size', 200)
            stride = args.stride
            
            pred_vel_prior, residual_stats = predict_with_prior_only(
                system, imu, window_size, stride, device, gt_vel
            )
            
            # 5. 计算轨迹
            pred_pos_prior = integrate_trajectory(pred_vel_prior, initial_pos=gt_pos[0], dt=0.005)
            
            # [Fix] 再次对齐轨迹和GT位置，防止 predict_with_prior_only 内部截断导致的不匹配
            min_pos_len = min(pred_pos_prior.shape[0], gt_pos.shape[0], gt_vel.shape[0])
            pred_pos_prior = pred_pos_prior[:min_pos_len]
            gt_pos = gt_pos[:min_pos_len]
            gt_vel = gt_vel[:min_pos_len]
            pred_vel_prior = pred_vel_prior[:min_pos_len]
            
            # 6. 计算指标
            pred_per_min = 200 * 60
            ate, rte = compute_ate_rte(pred_pos_prior[:, :2], gt_pos, pred_per_min)
            
            vel_error = np.linalg.norm(pred_vel_prior - gt_vel, axis=1)
            
            all_metrics.append({
                "split": split_name,
                "seq": seq_name,
                "ate": ate,
                "rte": rte,
                "mean_vel_error": np.mean(vel_error),
                "max_vel_error": np.max(vel_error),
            })
            
            all_residual_stats.append({
                "seq": seq_name,
                **residual_stats
            })
            
            # 7. 可视化
            if args.plot:
                plot_comparison(gt_vel, pred_vel_prior, gt_pos, pred_pos_prior, 
                               seq_name, ate, rte, residual_stats, results_dir)
    
    # 8. 汇总结果
    df = pd.DataFrame(all_metrics)
    summary_path = os.path.join(results_dir, "prior_only_metrics.csv")
    df.to_csv(summary_path, index=False)
    
    print("\n" + "="*60)
    print("v_prior 单独性能总结")
    print("="*60)
    print(df.groupby('split')[['ate', 'rte', 'mean_vel_error']].mean())
    
    # 残差统计
    df_stats = pd.DataFrame(all_residual_stats)
    print("\n" + "="*60)
    print("需要的 Residual 幅度统计")
    print("="*60)
    print(df_stats[['residual_mean', 'residual_std', 'residual_max']].describe())
    
    # 保存对比
    save_comparison_with_full_model(df, results_dir)
    
    return df


def predict_with_prior_only(system, imu_seq, window_size, stride, device, gt_vel):
    """
    仅使用 prior_head 预测速度
    同时计算需要的残差统计信息
    """
    L = imu_seq.shape[0]
    target_dim = system.prior_head.out_channels
    
    # 处理长度不匹配问题（features 和 targets 可能差1）
    gt_len = gt_vel.shape[0]
    if L != gt_len:
        print(f"  Warning: Length mismatch - imu: {L}, gt_vel: {gt_len}, using min")
        L = min(L, gt_len)
        imu_seq = imu_seq[:L]
        gt_vel = gt_vel[:L]
    
    # 准备窗口
    starts = np.arange(0, L - window_size + 1, stride)
    if starts[-1] + window_size < L:
        starts = np.concatenate([starts, [L - window_size]])
    
    windows = [imu_seq[s:s+window_size] for s in starts]
    windows = np.stack(windows)
    
    # 批量预测 v_prior
    batch_size = 32
    num_windows = len(windows)
    all_priors = []
    
    for i in range(0, num_windows, batch_size):
        batch_in = windows[i:i+batch_size]
        batch_in_torch = torch.from_numpy(batch_in).float().permute(0, 2, 1).to(device)
        
        with torch.no_grad():
            cond_feat = system.encoder(batch_in_torch)
            v_prior_feat = system.prior_head(cond_feat)
            v_prior = torch.nn.functional.interpolate(
                v_prior_feat, size=window_size, mode='linear', align_corners=False
            )
            v_prior_np = v_prior.permute(0, 2, 1).cpu().numpy()
        
        all_priors.append(v_prior_np)
    
    all_priors = np.concatenate(all_priors, axis=0)
    
    # 拼接
    v_prior_full = np.zeros((L, target_dim))
    weights = np.zeros((L, 1))
    window_weight = np.ones((window_size, 1))
    
    for i, (start, prior) in enumerate(zip(starts, all_priors)):
        v_prior_full[start:start+window_size] += prior * window_weight
        weights[start:start+window_size] += window_weight
    
    weights[weights == 0] = 1.0
    v_prior_full /= weights
    
    # [Fix] 再次确保长度一致，防止拼接过程中的微小差异
    min_len = min(gt_vel.shape[0], v_prior_full.shape[0])
    if gt_vel.shape[0] != v_prior_full.shape[0]:
        print(f"  [Debug] Truncating shapes for residual: gt {gt_vel.shape} vs prior {v_prior_full.shape} -> {min_len}")
    
    gt_vel = gt_vel[:min_len]
    v_prior_full = v_prior_full[:min_len]
    
    # 计算残差统计
    residual = gt_vel - v_prior_full
    residual_stats = {
        'residual_mean': np.mean(np.linalg.norm(residual, axis=1)),
        'residual_std': np.std(np.linalg.norm(residual, axis=1)),
        'residual_max': np.max(np.linalg.norm(residual, axis=1)),
        'residual_vx_mean': np.mean(np.abs(residual[:, 0])),
        'residual_vy_mean': np.mean(np.abs(residual[:, 1])),
    }
    
    return v_prior_full, residual_stats


def plot_comparison(gt_vel, pred_vel_prior, gt_pos, pred_pos, seq_name, ate, rte, 
                   residual_stats, results_dir):
    """绘制 v_prior 预测结果对比"""
    fig, axes = plt.subplots(3, 2, figsize=(16, 12))
    
    # 速度对比 - Vx
    axes[0, 0].plot(gt_vel[:, 0], label='GT', color='black', linewidth=1.5)
    axes[0, 0].plot(pred_vel_prior[:, 0], label='v_prior only', color='blue', linewidth=1)
    axes[0, 0].set_ylabel('Vx (m/s)')
    axes[0, 0].set_title(f'{seq_name} - v_prior Prediction (Vx)')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 速度对比 - Vy
    axes[0, 1].plot(gt_vel[:, 1], label='GT', color='black', linewidth=1.5)
    axes[0, 1].plot(pred_vel_prior[:, 1], label='v_prior only', color='blue', linewidth=1)
    axes[0, 1].set_ylabel('Vy (m/s)')
    axes[0, 1].set_title(f'{seq_name} - v_prior Prediction (Vy)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 速度误差
    vel_error = np.linalg.norm(pred_vel_prior - gt_vel, axis=1)
    axes[1, 0].plot(vel_error, color='red', linewidth=1)
    axes[1, 0].axhline(y=np.mean(vel_error), color='red', linestyle='--',
                       label=f'Mean: {np.mean(vel_error):.3f} m/s')
    axes[1, 0].set_ylabel('Velocity Error (m/s)')
    axes[1, 0].set_title('Velocity Error')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 残差分布（GT - v_prior）
    residual = gt_vel - pred_vel_prior
    axes[1, 1].hist(residual[:, 0], bins=50, alpha=0.5, label='Residual Vx')
    axes[1, 1].hist(residual[:, 1], bins=50, alpha=0.5, label='Residual Vy')
    axes[1, 1].set_xlabel('Residual (m/s)')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title('Residual Distribution (GT - v_prior)')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    # 轨迹
    axes[2, 0].plot(gt_pos[:, 0], gt_pos[:, 1], label='GT', color='black', linewidth=2)
    axes[2, 0].plot(pred_pos[:, 0], pred_pos[:, 1], label='v_prior', color='blue', linewidth=1.5)
    axes[2, 0].set_xlabel('X (m)')
    axes[2, 0].set_ylabel('Y (m)')
    axes[2, 0].set_title(f'Trajectory - ATE: {ate:.2f}m, RTE: {rte:.2f}m/min')
    axes[2, 0].axis('equal')
    axes[2, 0].legend()
    axes[2, 0].grid(True, alpha=0.3)
    
    # 残差时序
    axes[2, 1].plot(residual[:, 0], label='Residual Vx', alpha=0.7, linewidth=0.8)
    axes[2, 1].plot(residual[:, 1], label='Residual Vy', alpha=0.7, linewidth=0.8)
    axes[2, 1].axhline(y=0, color='black', linestyle='--', alpha=0.3)
    axes[2, 1].set_xlabel('Frame')
    axes[2, 1].set_ylabel('Residual (m/s)')
    axes[2, 1].set_title('Residual over Time')
    axes[2, 1].legend()
    axes[2, 1].grid(True, alpha=0.3)
    
    # 添加统计信息文本
    stats_text = (f"Residual Stats:\n"
                  f"  Mean: {residual_stats['residual_mean']:.3f} m/s\n"
                  f"  Std:  {residual_stats['residual_std']:.3f} m/s\n"
                  f"  Max:  {residual_stats['residual_max']:.3f} m/s")
    fig.text(0.5, 0.02, stats_text, ha='center', fontsize=10,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.savefig(os.path.join(results_dir, f"{seq_name}_prior_only.png"), dpi=150)
    plt.close()


def save_comparison_with_full_model(df_prior, results_dir):
    """保存与完整模型的对比说明"""
    with open(os.path.join(results_dir, "README.txt"), 'w') as f:
        f.write("="*60 + "\n")
        f.write("v_prior 单独性能评估\n")
        f.write("="*60 + "\n\n")
        
        f.write("这组结果是仅使用 v_prior（prior_head 输出）的速度预测，\n")
        f.write("没有使用 Diffusion 模型生成的残差。\n\n")
        
        f.write("评估目的：\n")
        f.write("1. 判断 prior_head 的学习质量\n")
        f.write("2. 了解需要 Diffusion 补充的残差幅度\n")
        f.write("3. 对比完整模型结果，评估 Diffusion 的改进效果\n\n")
        
        f.write("诊断指南：\n")
        f.write("- 如果 v_prior 的 ATE 已经很低（< 5m）:\n")
        f.write("  → prior_head 学习良好，问题可能在 Diffusion 或拼接\n")
        f.write("- 如果 v_prior 的 ATE 很高（> 20m）:\n")
        f.write("  → prior_head 学习不足，需要改进训练策略\n")
        f.write("- 如果残差幅度很大（> 1 m/s）:\n")
        f.write("  → Diffusion 负担过重，残差分布可能难以学习\n\n")
        
        f.write("当前结果：\n")
        summary = df_prior.groupby('split')[['ate', 'rte', 'mean_vel_error']].mean()
        f.write(summary.to_string())
        f.write("\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test v_prior only (no diffusion)')
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--out_dir', type=str, default='experiments/prior_only_results')
    parser.add_argument('--stride', type=int, default=100)
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--plot', action='store_true', default=True)
    parser.add_argument('--max_seqs', type=int, default=-1)
    
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    evaluate_prior_only(args, config)
