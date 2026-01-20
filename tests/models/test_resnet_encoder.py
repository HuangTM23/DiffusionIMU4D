import unittest
import torch
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from models.model_resnet1d import ResNet1D, BasicBlock1D

class TestResNetEncoder(unittest.TestCase):
    def test_encoder_output(self):
        """测试 ResNet 作为特征提取器的输出"""
        batch_size = 4
        seq_len = 200
        in_channels = 6
        
        # 禁用 output_block，使其输出特征
        model = ResNet1D(
            num_inputs=in_channels, 
            num_outputs=None, # None 表示不加最后的分类/回归层
            block_type=BasicBlock1D, 
            group_sizes=[2, 2, 2, 2],
            output_block=None
        )
        
        x = torch.randn(batch_size, in_channels, seq_len)
        feat = model(x)
        
        # ResNet1D 有 4 个 stage，下采样率为 2^(4-1) = 8? 
        # 让我们检查 model_resnet1d.py 中的 strides。
        # strides = [1] + [2] * (len(group_sizes) - 1) = [1, 2, 2, 2]
        # Input block stride=2, MaxPool stride=2 -> 4x downsample before residuals
        # Residuals downsample: 1, 2, 2, 2 -> 8x more downsample
        # Total downsample: 4 * 8 = 32x
        # 200 / 32 = 6.25 -> 7
        
        expected_len = seq_len // 32 + (1 if seq_len % 32 != 0 else 0)
        expected_channels = 64 * (2**3) # 512
        
        self.assertEqual(feat.shape, (batch_size, expected_channels, expected_len))

if __name__ == '__main__':
    unittest.main()
