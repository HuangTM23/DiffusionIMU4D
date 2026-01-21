import unittest
import torch
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from models.diffusion_system import DiffusionSystem
from models.model_resnet1d import ResNet1D, BasicBlock1D
from models.diffusion_unet import DiffUNet1D

class TestDiffusionSystem(unittest.TestCase):
    def setUp(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 初始化子模块
        self.encoder = ResNet1D(6, None, BasicBlock1D, [2, 2, 2, 2], output_block=None)
        self.unet = DiffUNet1D(
            in_channels=3, 
            out_channels=3, 
            cond_channels=512, # ResNet1D output channels
            base_channels=32,
            channel_mults=(1, 2, 4),
            layers_per_block=1
        )
        
    def test_forward_end2end(self):
        """测试 End-to-End 模式下的 Loss 计算"""
        system = DiffusionSystem(self.encoder, self.unet, mode="end2end").to(self.device)
        
        imu = torch.randn(2, 6, 200).to(self.device)
        gt_vel = torch.randn(2, 3, 200).to(self.device)
        
        loss = system(imu, gt_vel)
        self.assertGreater(loss.item(), 0)

    def test_forward_residual(self):
        """测试 Residual 模式下的 Loss 计算和残差逻辑"""
        # 强制设置 mode 为 residual
        system = DiffusionSystem(self.encoder, self.unet, mode="residual").to(self.device)
        
        imu = torch.randn(2, 6, 200).to(self.device)
        gt_vel = torch.randn(2, 3, 200).to(self.device)
        
        # 1. 测试前向传播是否产生 Loss
        loss = system(imu, gt_vel)
        self.assertGreater(loss.item(), 0)
        
        # 2. 验证内部逻辑：确认 prior_head 存在且输出维度正确
        self.assertTrue(hasattr(system, 'prior_head'))
        cond_feat = system.encoder(imu)
        v_prior_feat = system.prior_head(cond_feat)
        v_prior = torch.nn.functional.interpolate(v_prior_feat, size=200, mode='linear', align_corners=False)
        self.assertEqual(v_prior.shape, (2, 3, 200))

    def test_sample_residual(self):
        """测试 Residual 模式下的采样（应包含 v_prior 加和）"""
        system = DiffusionSystem(self.encoder, self.unet, mode="residual").to(self.device)
        imu = torch.randn(1, 6, 200).to(self.device)
        
        sampled_vel = system.sample(imu, num_inference_steps=5)
        self.assertEqual(sampled_vel.shape, (1, 3, 200))

    def test_sample(self):
        """测试采样生成"""
        system = DiffusionSystem(self.encoder, self.unet, mode="end2end").to(self.device)
        imu = torch.randn(1, 6, 200).to(self.device)
        
        # 采样
        sampled_vel = system.sample(imu, num_inference_steps=10) # 快速采样
        self.assertEqual(sampled_vel.shape, (1, 3, 200))

if __name__ == '__main__':
    unittest.main()
