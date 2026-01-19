import unittest
import numpy as np
import os
import h5py
import json
import torch
from torch.utils.data import DataLoader
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from data.dataset import RoNINDataset

class TestRoNINDataset(unittest.TestCase):
    def setUp(self):
        """创建符合 RoNIN 格式的临时 HDF5 数据"""
        self.test_root = "tests/test_ronin_root"
        self.data_dir = os.path.join(self.test_root, "extracted")
        self.list_dir = os.path.join(self.test_root, "lists")
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.list_dir, exist_ok=True)
        
        # 创建一个序列目录
        self.seq_name = "test_seq_01"
        self.seq_dir = os.path.join(self.data_dir, self.seq_name)
        os.makedirs(self.seq_dir, exist_ok=True)
        
        # 创建 data.hdf5
        self.h5_path = os.path.join(self.seq_dir, "data.hdf5")
        n_samples = 1000
        with h5py.File(self.h5_path, 'w') as f:
            synced = f.create_group('synced')
            synced.create_dataset('acce', data=np.random.randn(n_samples, 3).astype(np.float32))
            synced.create_dataset('gyro', data=np.random.randn(n_samples, 3).astype(np.float32))
            synced.create_dataset('time', data=np.linspace(0, 10, n_samples).astype(np.float64))
            synced.create_dataset('rv', data=np.tile([0, 0, 0, 1], (n_samples, 1)).astype(np.float32))
            
            pose = f.create_group('pose')
            pose.create_dataset('tango_pos', data=np.cumsum(np.random.randn(n_samples, 3) * 0.1, axis=0).astype(np.float32))
            pose.create_dataset('tango_ori', data=np.tile([0, 0, 0, 1], (n_samples, 1)).astype(np.float32))

        # 创建 info.json
        self.info_path = os.path.join(self.seq_dir, "info.json")
        with open(self.info_path, 'w') as f:
            json.dump({"start_frame": 10}, f)
            
        # 创建列表文件
        self.list_file = os.path.join(self.list_dir, "list_train.txt")
        with open(self.list_file, 'w') as f:
            f.write(self.seq_name + "\n")

    def tearDown(self):
        """清理临时数据"""
        import shutil
        if os.path.exists(self.test_root):
            shutil.rmtree(self.test_root)

    def test_dataset_initialization(self):
        """测试 Dataset 初始化和基本长度"""
        window_size = 200
        stride = 100
        # 样本数 1000，start_frame 10 -> 有效样本 990
        # (990 - 200) // 100 + 1 = 7 + 1 = 8
        dataset = RoNINDataset(self.data_dir, self.list_file, window_size=window_size, stride=stride)
        self.assertEqual(len(dataset), 8)

    def test_dataset_item_shape(self):
        """测试获取的数据项维度"""
        window_size = 200
        dataset = RoNINDataset(self.data_dir, self.list_file, window_size=window_size)
        
        imu, velocity = dataset[0]
        
        # IMU 应包含 acce(3) + gyro(3) = 6 轴
        self.assertEqual(imu.shape, (window_size, 6))
        # 目标速度为 3 轴 (vx, vy, vz)
        self.assertEqual(velocity.shape, (window_size, 3))

if __name__ == '__main__':
    unittest.main()