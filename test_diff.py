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
    
    # 1. Models
    input_dim = 6
    target_dim = config.get('target_dim', 2) # Default to 2 for RoNIN
    
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
    
    system = DiffusionSystem(encoder, unet, mode=config.get('mode', 'end2end')).to(device)
    
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
            
            # 4. Reconstruct Trajectory
            # Initial position from GT
            pred_pos = integrate_trajectory(pred_vel, initial_pos=gt_pos[0], dt=0.005) # 200Hz -> dt=0.005
            
            # 5. Metrics
            pred_per_min = 200 * 60
            ate, rte = compute_ate_rte(pred_pos[:, :2], gt_pos, pred_per_min)
            
            all_metrics.append({
                "split": split_name,
                "seq": seq_name,
                "ate": ate,
                "rte": rte
            })
            
            # 6. Plotting
            if args.plot:
                plot_path = os.path.join(results_dir, f"{split_name}_{seq_name}.png")
                save_plot(gt_pos, pred_pos, seq_name, ate, rte, plot_path)
                
    # 7. Save Summary
    df = pd.DataFrame(all_metrics)
    summary_path = os.path.join(results_dir, "metrics.csv")
    df.to_csv(summary_path, index=False)
    
    print("\nEvaluation Summary:")
    print(df.groupby('split')[['ate', 'rte']].mean())
    print(f"Results saved to {results_dir}")

def save_plot(gt_pos, pred_pos, seq_name, ate, rte, path):
    plt.figure(figsize=(8, 8))
    plt.plot(gt_pos[:, 0], gt_pos[:, 1], label='Ground Truth', color='black', alpha=0.5)
    plt.plot(pred_pos[:, 0], pred_pos[:, 1], label='Predicted', color='red', alpha=0.8)
    plt.title(f"Seq: {seq_name}\nATE: {ate:.3f}m, RTE: {rte:.3f}m/min")
    plt.axis('equal')
    plt.legend()
    plt.grid(True)
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
