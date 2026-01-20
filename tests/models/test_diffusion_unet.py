import unittest
import torch
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from models.diffusion_unet import DiffUNet1D

class TestDiffUNet1D(unittest.TestCase):
    def test_model_output_shape(self):
        """测试 1D U-Net 的输入输出维度"""
        batch_size = 4
        seq_len = 200
        in_channels = 3 # 速度 3维
        cond_channels = 512 # 假设 IMU Encoder 输出 512 维特征
        
        model = DiffUNet1D(
            in_channels=in_channels, 
            out_channels=in_channels,
            cond_channels=cond_channels,
            base_channels=64,
            channel_mults=(1, 2, 4, 8),
            layers_per_block=2
        )
        
        # 模拟输入: 噪声速度 (B, C, L), 时间步 (B,), 条件特征 (B, D, L)
        x = torch.randn(batch_size, in_channels, seq_len)
        timesteps = torch.randint(0, 1000, (batch_size,))
        cond = torch.randn(batch_size, cond_channels, seq_len)
        
        output = model(x, timesteps, cond)
        
        # 输出应与输入 x 同形状 (B, C, L)
        self.assertEqual(output.shape, (batch_size, in_channels, seq_len))

if __name__ == '__main__':
    unittest.main()
