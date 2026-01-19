import unittest
import numpy as np
import os
import pickle
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from data.dataset import RoNINDataset

class TestRoNINDataset(unittest.TestCase):
    def setUp(self):
        """创建测试用的临时数据"""
        self.test_dir = "tests/test_data_dir"
        os.makedirs(self.test_dir, exist_ok=True)
        
        # 创建一个模拟的 pickle 文件
        self.sample_data = {
            "imu": np.random.randn(1000, 6).astype(np.float32),
            "velocity": np.random.randn(1000, 3).astype(np.float32),
            "ts": np.linspace(0, 10, 1000).astype(np.float64)
        }
        self.pickle_path = os.path.join(self.test_dir, "seq_01.pickle")
        with open(self.pickle_path, 'wb') as f:
            pickle.dump(self.sample_data, f)
            
        # 列表文件
        self.list_file = os.path.join(self.test_dir, "list.txt")
        with open(self.list_file, 'w') as f:
            f.write("seq_01.pickle\n")

    def tearDown(self):
        """清理临时数据"""
        import shutil
        shutil.rmtree(self.test_dir)

    def test_dataset_initialization(self):
        """测试 Dataset 初始化和基本长度"""
        window_size = 200
        stride = 10
        dataset = RoNINDataset(self.test_dir, self.list_file, window_size=window_size, stride=stride)
        
        # 计算预期窗口数量: (1000 - 200) // 10 + 1 = 80 + 1 = 81
        self.assertEqual(len(dataset), 81)

    def test_dataset_item_shape(self):
        """测试获取的数据项维度"""
        window_size = 200
        dataset = RoNINDataset(self.test_dir, self.list_file, window_size=window_size)
        
        imu, velocity = dataset[0]
        
        self.assertEqual(imu.shape, (window_size, 6))
        self.assertEqual(velocity.shape, (window_size, 3))

if __name__ == '__main__':
    unittest.main()
