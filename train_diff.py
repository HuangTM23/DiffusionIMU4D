import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import argparse
import yaml
import numpy as np
import matplotlib.pyplot as plt

import math
from data.data_glob_speed import GlobSpeedSequence, SequenceToSequenceDataset
from utils.math_util import get_random_rotation
from utils.transformations import RandomHoriRotate
from models.model_resnet1d import ResNet1D, BasicBlock1D
from models.diffusion_unet import DiffUNet1D
from models.diffusion_system import DiffusionSystem
from utils.logger import init_logger, log_metrics

def get_dataset(root_dir, data_list, config, mode='train'):
    random_shift, shuffle, transforms, grv_only = 0, False, None, False
    if mode == 'train':
        random_shift = config.get('stride', 10) // 2
        shuffle = True
        transforms = RandomHoriRotate(math.pi * 2)
    elif mode == 'val':
        shuffle = True
    elif mode == 'test':
        shuffle = False
        grv_only = True

    seq_type = GlobSpeedSequence
    
    cache_path = os.path.join(config.get('data_dir', ''), 'cache')
    os.makedirs(cache_path, exist_ok=True)

    # 使用 SequenceToSequenceDataset
    dataset = SequenceToSequenceDataset(
        seq_type, root_dir, data_list, cache_path, 
        step_size=config.get('stride', 10), 
        window_size=config.get('window_size', 200),
        random_shift=random_shift, 
        transform=transforms,
        shuffle=shuffle,
        # grv_only=grv_only, # Seq2Seq doesn't seem to take grv_only directly in __init__ based on code reading, but let's check
    )
    return dataset


def get_dataset_from_list(root_dir, list_path, config, mode='train'):
    with open(list_path) as f:
        data_list = [s.strip().split(',' or ' ')[0] for s in f.readlines() if len(s) > 0 and s[0] != '#']
    return get_dataset(root_dir, data_list, config, mode=mode)

def main(config, args):
    # WandB
    init_logger(project_name="Diffusion4d-Diff", config=config)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on Device: {device}")
    
    # 1. Dataset
    train_dataset = get_dataset_from_list(config['data_dir'], config['train_list'], config, mode='train')
    val_dataset = get_dataset_from_list(config['data_dir'], config['val_list'], config, mode='val')
    
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=4)
    
    # Check dimensions
    # RoNIN usually has target_dim=2 (vx, vy)
    target_dim = train_dataset.target_dim 
    input_dim = train_dataset.feature_dim
    
    # 2. Models
    # Encoder: ResNet1D (output features)
    encoder = ResNet1D(
        num_inputs=input_dim, 
        num_outputs=None, # Feature extraction mode
        block_type=BasicBlock1D, 
        group_sizes=[2, 2, 2, 2],
        output_block=None
    )
    
    # UNet
    unet = DiffUNet1D(
        in_channels=target_dim, 
        out_channels=target_dim, 
        cond_channels=512, 
        base_channels=64,
        channel_mults=(1, 2, 4, 8)
    )
    
    # System
    system = DiffusionSystem(encoder, unet, mode=config.get('mode', 'end2end')).to(device)
    
    # Update prior_head if residual mode and dimensions don't match
    if config.get('mode') == 'residual' and system.prior_head.out_channels != target_dim:
         system.prior_head = nn.Conv1d(512, target_dim, kernel_size=1).to(device)
    
    # Optimizer
    optimizer = optim.Adam(system.parameters(), lr=config['lr'])
    
    # 3. Training Loop
    import time
    print(f"Start Training... Total Epochs: {config['epochs']}")
    for epoch in range(config['epochs']):
        epoch_start_time = time.time()
        system.train()
        train_loss = 0
        
        # print(f"--- Starting Epoch {epoch} ---", flush=True)
        iterator = iter(train_loader)
        
        i = 0
        while True:
            try:
                batch = next(iterator)
                imu, vel, _, _ = batch
            except StopIteration:
                break
            
            if args.dry_run and i > 2: break
            
            # (B, L, C) -> (B, C, L)
            imu = imu.transpose(1, 2).to(device)
            vel = vel.transpose(1, 2).to(device)
            
            optimizer.zero_grad()
            loss = system(imu, vel)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            # if i % config['log_interval'] == 0:
            #     print(f"  [Epoch {epoch}][Batch {i}] Loss: {loss.item():.4f}", flush=True)
            #     log_metrics({"train_loss": loss.item()}, step=epoch * len(train_loader) + i)
            
            i += 1
                
        avg_train_loss = train_loss / (i if i > 0 else 1)
        epoch_duration = time.time() - epoch_start_time
        print(f"Epoch {epoch} Finished. Duration: {epoch_duration:.2f}s | Avg Loss: {avg_train_loss:.6f}", flush=True)
        log_metrics({"epoch": epoch, "train_loss": avg_train_loss})
        
        # Validation & Sampling
        if (epoch + 1) % config['save_interval'] == 0 or args.dry_run:
            validate_and_sample(system, val_loader, device, epoch, config, args)
            
            # Save Checkpoint
            os.makedirs('experiments/checkpoints', exist_ok=True)
            torch.save(system.state_dict(), f'experiments/checkpoints/diff_{config["mode"]}_epoch_{epoch}.pth')

def validate_and_sample(system, val_loader, device, epoch, config, args):
    system.eval()
    val_loss = 0
    
    # 1. Compute Val Loss (Noise Prediction MSE)
    with torch.no_grad():
        for i, (imu, vel, _, _) in enumerate(val_loader):
            if args.dry_run and i > 1: break
            imu = imu.transpose(1, 2).to(device)
            vel = vel.transpose(1, 2).to(device)
            loss = system(imu, vel)
            val_loss += loss.item()
            
    avg_val_loss = val_loss / (len(val_loader) if not args.dry_run else 2)
    log_metrics({"epoch": epoch, "val_loss": avg_val_loss})
    print(f"Validation Loss: {avg_val_loss:.6f}")
    
    # 2. Sampling Visualization
    # Pick first batch to sample
    imu_sample, vel_sample, _, _ = next(iter(val_loader))
    imu_sample = imu_sample[:1].transpose(1, 2).to(device) # Take 1 sample
    vel_gt = vel_sample[:1].transpose(1, 2).to(device)
    
    # Sample
    num_steps = config.get('num_inference_steps', 50)
    if args.dry_run: num_steps = 2
    
    pred_vel = system.sample(imu_sample, num_inference_steps=num_steps)
    
    # Plot
    pred_vel_np = pred_vel.squeeze().cpu().numpy()
    gt_vel_np = vel_gt.squeeze().cpu().numpy()
    
    num_channels = pred_vel_np.shape[0]
    fig, ax = plt.subplots(num_channels, 1, figsize=(10, 3 * num_channels))
    labels = ['Vx', 'Vy', 'Vz']
    
    # Handle case where num_channels=1 (unlikely but safe)
    if num_channels == 1:
        ax = [ax]
        
    for k in range(num_channels):
        ax[k].plot(gt_vel_np[k], label='GT', alpha=0.7)
        ax[k].plot(pred_vel_np[k], label='Pred', alpha=0.7)
        ax[k].set_ylabel(labels[k] if k < 3 else f'Ch{k}')
        ax[k].legend()
    plt.suptitle(f'Epoch {epoch} Sampling ({config["mode"]})')
    
    # Log image to wandb
    import wandb
    if wandb.run is not None:
        wandb.log({"sample_plot": wandb.Image(fig)}, step=epoch)
    
    plt.close(fig)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/diffusion.yaml')
    parser.add_argument('--dry_run', action='store_true', help="Run a few batches for debugging")
    parser.add_argument('--epochs', type=int, help="Override epochs in config")
    parser.add_argument('--lr', type=float, help="Override learning rate in config")
    parser.add_argument('--batch_size', type=int, help="Override batch size in config")
    args = parser.parse_args()
    
    if os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
        
        if args.epochs: config['epochs'] = args.epochs
        if args.lr: config['lr'] = args.lr
        if args.batch_size: config['batch_size'] = args.batch_size
            
        main(config, args)
    else:
        print(f"Config file {args.config} not found.")
