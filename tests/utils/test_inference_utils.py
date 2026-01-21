import unittest
import numpy as np
import torch
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from utils.inference_utils import reconstruct_trajectory

class TestInferenceUtils(unittest.TestCase):
    def test_reconstruct_trajectory_shape(self):
        """测试轨迹重建的输出形状"""
        seq_len = 1000
        window_size = 200
        stride = 100
        
        # 模拟模型：输入 (B, C, L)，输出 (B, 3, L)
        # 假设只输出 0
        def mock_model_func(x):
            B, C, L = x.shape
            return torch.zeros(B, 3, L)
            
        imu = np.random.randn(seq_len, 6)
        
        # 重建
        # reconstruct_trajectory 应该返回 (seq_len, 3) 的速度序列
        recon_vel = reconstruct_trajectory(mock_model_func, imu, window_size, stride, device='cpu')
        
        self.assertEqual(recon_vel.shape, (seq_len, 3))

    def test_integrate_trajectory(self):
        """测试积分逻辑"""
        from utils.inference_utils import integrate_trajectory
        
        # 匀速运动 v=1
        L = 100
        dt = 0.1
        vel = np.ones((L, 3))
        
        pos = integrate_trajectory(vel, dt=dt)
        
        # p_t = t * v
        # t=0 -> 0
        # t=1 -> 0.1
        # t=99 -> 9.9
        
        self.assertAlmostEqual(pos[0, 0], 0.0)
        self.assertAlmostEqual(pos[10, 0], 1.0) # 10 steps * 0.1
        self.assertAlmostEqual(pos[-1, 0], (L-1) * dt)

if __name__ == '__main__':
    unittest.main()
