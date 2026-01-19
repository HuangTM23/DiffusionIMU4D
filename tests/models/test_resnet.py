import unittest
import torch
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from models.resnet_baseline import ResNet18Baseline

class TestResNetBaseline(unittest.TestCase):
    def test_model_output_shape(self):
        """测试 ResNet Baseline 的输入输出维度"""
        batch_size = 8
        window_size = 200
        in_channels = 6  # acce(3) + gyro(3)
        out_channels = 3 # vx, vy, vz
        
        model = ResNet18Baseline(in_channels=in_channels, out_channels=out_channels)
        
        # PyTorch 1D Conv 预期输入: (Batch, Channels, Length)
        x = torch.randn(batch_size, in_channels, window_size)
        output = model(x)
        
        self.assertEqual(output.shape, (batch_size, out_channels, window_size))

if __name__ == '__main__':
    unittest.main()
