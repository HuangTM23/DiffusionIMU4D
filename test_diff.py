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

def evaluate(args, config):
    device = torch.device('cuda:0' if torch.cuda.is_available() and not args.cpu else 'cpu')
    
    # 1. Configs
    target_dim = config.get('target_dim', 2)
    mode = config.get('mode', 'end2end')
    print(f"Mode: {mode}, Target Dim: {target_dim}")
    
    # 2. Model
    encoder = ResNet1D(
        num_inputs=6, 
        num_outputs=None, 
        block_type=BasicBlock1D, 
        group_sizes=[2, 2, 2, 2],
        output_block=None
    )
    
    # [Modified] Residual 模式输入通道翻倍
    unet_in_channels = target_dim * 2 if mode == 'residual' else target_dim
    
    unet = DiffUNet1D(
        in_channels=unet_in_channels, 
        out_channels=target_dim, 
        cond_channels=512, 
        base_channels=64,
        channel_mults=(1, 2, 4, 8)
    )
    
    system = DiffusionSystem(encoder, unet, mode=mode).to(device)
    
    # Fix: Update prior_head output dimension for residual mode
    if config.get('mode') == 'residual' and system.prior_head.out_channels != target_dim:
        system.prior_head = nn.Conv1d(512, target_dim, kernel_size=1).to(device)
    
    # Load Weights
    if args.model_path and os.path.exists(args.model_path):
        print(f"Loading model from {args.model_path}")
        system.load_state_dict(torch.load(args.model_path, map_location=device))
    else:
        print("Warning: No model path provided or file not found. Evaluating random model.")
        
    # Switch Scheduler
    system.set_scheduler(args.scheduler)
    
    system.eval()
    
    # 2. Test Lists
    test_lists = {
        "seen": config.get('test_seen_list', 'data/RoNIN/lists/list_test_seen.txt'),
        "unseen": config.get('test_unseen_list', 'data/RoNIN/lists/list_test_unseen.txt')
    }
    
    results_dir = args.out_dir or 'experiments/results'
    os.makedirs(results_dir, exist_ok=True)
    
    all_metrics = []
    
    for split_name, list_path in test_lists.items():
        if not os.path.exists(list_path):
            print(f"List {list_path} not found, skipping {split_name}.")
            continue
            
        print(f"Evaluating {split_name} split...")
        with open(list_path, 'r') as f:
            seq_names = [line.strip() for line in f if line.strip() and line[0] != '#']
            
        for seq_name in tqdm(seq_names):
            # Load full sequence data
            seq_path = os.path.join(config['data_dir'], seq_name)
            # We use GlobSpeedSequence directly to get the whole IMU and GT
            try:
                seq_data = GlobSpeedSequence(seq_path)
            except Exception as e:
                print(f"Error loading {seq_name}: {e}")
                continue
                
            imu = seq_data.get_feature() # (L, 6)
            gt_vel = seq_data.get_target() # (L, 2 or 3)
            gt_pos = seq_data.gt_pos[:, :2] # (L, 2)
            
            # 3. Inference
            # Define a simple wrapper for system.sample
            def model_func(x):
                # x is already (B, 6, W) from reconstruct_trajectory
                return system.sample(x, num_inference_steps=args.steps)
                
            pred_vel = reconstruct_trajectory(
                model_func, 
                imu, 
                window_size=config.get('window_size', 200),
                stride=args.stride,
                device=device,
                batch_size=args.batch_size
            )
            
            min_len_final = min(pred_vel.shape[0], gt_pos.shape[0], gt_vel.shape[0])
            pred_vel, gt_pos, gt_vel = pred_vel[:min_len_final], gt_pos[:min_len_final], gt_vel[:min_len_final]

            # 自动清洗 GT 速度异常点 (防止 Tango 视觉跳变干扰评估)
            v_norm = np.linalg.norm(gt_vel, axis=1)
            bad_mask = v_norm > 5.0
            if np.any(bad_mask):
                valid_indices = np.where(~bad_mask)[0]
                if len(valid_indices) > 0:
                    for i in np.where(bad_mask)[0]:
                        nearest_valid = valid_indices[np.abs(valid_indices - i).argmin()]
                        gt_vel[i] = gt_vel[nearest_valid]

            # 4. Reconstruct Trajectory
            pred_pos = integrate_trajectory(pred_vel, initial_pos=gt_pos[0], dt=0.005)
            
            # 5. Metrics
            ate, rte = compute_ate_rte(pred_pos[:, :2], gt_pos, 200 * 60)
            
            all_metrics.append({
                "split": split_name, "seq": seq_name, "ate": ate, "rte": rte,
                "mean_vel_error": np.mean(np.linalg.norm(pred_vel - gt_vel, axis=1))
            })
            
            # 6. Plotting
            if args.plot:
                plot_detailed_comparison(gt_pos, pred_pos, gt_vel, pred_vel, seq_name, ate, results_dir)
                
    # 7. Save Summary
    df = pd.DataFrame(all_metrics)
    summary_path = os.path.join(results_dir, "metrics.csv")
    df.to_csv(summary_path, index=False)
    
    print("\nEvaluation Summary:")
    print(df.groupby('split')[['ate', 'rte', 'mean_vel_error']].mean())
    print(f"Results saved to {results_dir}")

def plot_detailed_comparison(gt_pos, pred_pos, gt_vel, pred_vel, name, ate, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # Left: Trajectory
    axes[0].plot(gt_pos[:, 0], gt_pos[:, 1], 'k-', label='GT', alpha=0.6)
    axes[0].plot(pred_pos[:, 0], pred_pos[:, 1], 'r--', label='Pred')
    axes[0].set_title(f"Trajectory: {name} (ATE: {ate:.2f}m)")
    axes[0].axis('equal')
    axes[0].legend()
    axes[0].grid(True)
    
    # Right: Velocity Components
    ax_v = axes[1]
    ax_v.plot(gt_vel[:, 0], 'k-', label='GT Vx', alpha=0.4)
    ax_v.plot(pred_vel[:, 0], 'r-', label='Pred Vx', alpha=0.7)
    ax_v.plot(gt_vel[:, 1], 'k--', label='GT Vy', alpha=0.4)
    ax_v.plot(pred_vel[:, 1], 'g--', label='Pred Vy', alpha=0.7)
    ax_v.set_title("Velocity Components (X & Y)")
    ax_v.set_xlabel("Time Step")
    ax_v.set_ylabel("m/s")
    ax_v.legend(loc='upper right')
    ax_v.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{name}_comparison.png"), dpi=150)
    plt.close()

def save_trajectory_plot(gt_pos, pred_pos, seq_name, ate, rte, path):
    plt.figure(figsize=(8, 8))
    plt.plot(gt_pos[:, 0], gt_pos[:, 1], label='Ground Truth', color='black', alpha=0.5)
    plt.plot(pred_pos[:, 0], pred_pos[:, 1], label='Predicted', color='red', alpha=0.8)
    plt.title(f"Seq: {seq_name}\nATE: {ate:.3f}m, RTE: {rte:.3f}m/min")
    plt.axis('equal')
    plt.legend()
    plt.grid(True)
    plt.savefig(path)
    plt.close()

def save_error_plot(gt_pos, pred_pos, seq_name, path):
    # Compute Euclidean distance at each step
    errors = np.linalg.norm(gt_pos - pred_pos[:, :2], axis=1)
    
    plt.figure(figsize=(10, 4))
    plt.plot(errors, color='blue')
    plt.title(f"Position Error over Time - {seq_name}")
    plt.xlabel("Step")
    plt.ylabel("Error (m)")
    plt.grid(True)
    plt.savefig(path)
    plt.close()

def save_distribution_plot(df, path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # ATE Distribution
    df['ate'].hist(ax=axes[0], bins=20, color='skyblue', edgecolor='black')
    axes[0].set_title("ATE Distribution")
    axes[0].set_xlabel("ATE (m)")
    
    # RTE Distribution
    df['rte'].hist(ax=axes[1], bins=20, color='salmon', edgecolor='black')
    axes[1].set_title("RTE Distribution")
    axes[1].set_xlabel("RTE (m/min)")
    
    plt.tight_layout()
    plt.savefig(path)
    plt.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--model_path', type=str)
    parser.add_argument('--out_dir', type=str)
    parser.add_argument('--scheduler', type=str, default='ddim', choices=['ddpm', 'ddim'])
    parser.add_argument('--steps', type=int, default=50)
    parser.add_argument('--stride', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--plot', action='store_true')
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
        
    evaluate(args, config)
