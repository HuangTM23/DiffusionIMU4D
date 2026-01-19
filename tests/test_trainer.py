import unittest
import torch
from torch.utils.data import DataLoader, TensorDataset
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from train import Trainer
from models.resnet_baseline import ResNet18Baseline

class TestTrainer(unittest.TestCase):
    def test_training_step(self):
        """测试单个训练步是否能正常运行并产生梯度"""
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = ResNet18Baseline(in_channels=6, out_channels=3).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = torch.nn.MSELoss()
        
        # 模拟数据 (Batch, Length, Channels)
        imu = torch.randn(4, 200, 6).to(device)
        vel = torch.randn(4, 200, 3).to(device)
        
        trainer = Trainer(model, optimizer, criterion, device)
        loss = trainer.train_step(imu, vel)
        
        self.assertGreater(loss, 0)
        # 检查是否产生了梯度
        for param in model.parameters():
            if param.requires_grad:
                self.assertIsNotNone(param.grad)
                break

if __name__ == '__main__':
    unittest.main()
