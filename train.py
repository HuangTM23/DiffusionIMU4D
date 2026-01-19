import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import argparse
import yaml

from data.dataset import RoNINDataset
from models.resnet_baseline import ResNet18Baseline
from utils.logger import init_logger, log_metrics

class Trainer:
    def __init__(self, model, optimizer, criterion, device, scheduler=None):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.scheduler = scheduler

    def train_step(self, imu, velocity):
        self.model.train()
        self.optimizer.zero_grad()
        
        # 转换输入维度: (B, L, C) -> (B, C, L)
        imu = imu.transpose(1, 2).to(self.device)
        velocity = velocity.transpose(1, 2).to(self.device)
        
        output = self.model(imu)
        loss = self.criterion(output, velocity)
        
        loss.backward()
        self.optimizer.step()
        
        return loss.item()

    def val_step(self, imu, velocity):
        self.model.eval()
        with torch.no_grad():
            imu = imu.transpose(1, 2).to(self.device)
            velocity = velocity.transpose(1, 2).to(self.device)
            
            output = self.model(imu)
            loss = self.criterion(output, velocity)
            
        return loss.item()

def main(config):
    # 初始化 WandB
    init_logger(project_name="Diffusion4d-Baseline", config=config)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 数据加载
    train_dataset = RoNINDataset(
        config['data_dir'], 
        config['train_list'], 
        window_size=config['window_size'], 
        stride=config['stride']
    )
    val_dataset = RoNINDataset(
        config['data_dir'], 
        config['val_list'], 
        window_size=config['window_size'], 
        stride=config['window_size'] # 验证集不重叠
    )
    
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=4)
    
    # 模型、优化器
    model = ResNet18Baseline(in_channels=6, out_channels=3).to(device)
    optimizer = optim.Adam(model.parameters(), lr=config['lr'])
    criterion = nn.MSELoss()
    
    trainer = Trainer(model, optimizer, criterion, device)
    
    # 训练循环
    for epoch in range(config['epochs']):
        train_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
        for i, (imu, vel) in enumerate(pbar):
            loss = trainer.train_step(imu, vel)
            train_loss += loss
            pbar.set_postfix({"loss": loss})
            
            if i % config['log_interval'] == 0:
                log_metrics({"train_loss": loss}, step=epoch * len(train_loader) + i)
        
        avg_train_loss = train_loss / len(train_loader)
        
        # 验证
        val_loss = 0
        for imu, vel in val_loader:
            val_loss += trainer.val_step(imu, vel)
        avg_val_loss = val_loss / len(val_loader)
        
        print(f"Epoch {epoch}: Train Loss={avg_train_loss:.6f}, Val Loss={avg_val_loss:.6f}")
        log_metrics({"epoch": epoch, "avg_train_loss": avg_train_loss, "avg_val_loss": avg_val_loss})
        
        # 保存权重
        if (epoch + 1) % config['save_interval'] == 0:
            os.makedirs('experiments/checkpoints', exist_ok=True)
            torch.save(model.state_dict(), f'experiments/checkpoints/resnet_epoch_{epoch}.pth')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/resnet_baseline.yaml')
    args = parser.parse_args()
    
    if os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
        main(config)
    else:
        print(f"Config file {args.config} not found.")
