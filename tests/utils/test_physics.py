import unittest
import numpy as np
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from utils.physics import remove_gravity, rotate_imu_data

class TestPhysics(unittest.TestCase):
    def test_remove_gravity(self):
        """测试去除重力。假设重力在 Z 轴，值为 9.81"""
        accel = np.array([[0, 0, 9.81], [1.0, 0, 10.81]])
        expected = np.array([[0, 0, 0], [1.0, 0, 1.0]])
        
        result = remove_gravity(accel, gravity=9.81)
        np.testing.assert_array_almost_equal(result, expected)

    def test_rotate_imu_data(self):
        """测试 IMU 数据旋转。测试绕 Z 轴旋转 90 度"""
        # 旋转矩阵：绕 Z 轴 90 度
        # [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
        rot_matrix = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
        data = np.array([[1.0, 0, 0], [0, 1.0, 0]])
        expected = np.array([[0, 1.0, 0], [-1.0, 0, 0]]) # data @ rot_matrix.T or rot_matrix @ data.T?
        # RoNIN 习惯通常是 v_world = R @ v_imu
        # 我们的实现将明确这一点。
        
        result = rotate_imu_data(data, rot_matrix)
        # 如果是 R @ data.T -> [0, 1, 0] 和 [-1, 0, 0]
        expected = np.array([[0, 1.0, 0], [-1.0, 0, 0]])
        np.testing.assert_array_almost_equal(result, expected)

if __name__ == '__main__':
    unittest.main()
