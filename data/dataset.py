import os
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset

class RoNINDataset(Dataset):
    """
    RoNIN 数据集加载器。
    采用滑动窗口方式切分长序列为固定长度的训练样本。
    """
    def __init__(self, data_dir, list_file, window_size=200, stride=10, transform=None):
        """
        Args:
            data_dir (str): 包含 pickle 文件的目录。
            list_file (str): 包含待加载文件名列表的文本文件。
            window_size (int): 窗口大小（时间步数）。
            stride (int): 滑动步长。
            transform (callable, optional): 可选的数据变换。
        """
        self.data_dir = data_dir
        self.window_size = window_size
        self.stride = stride
        self.transform = transform
        
        with open(list_file, 'r') as f:
            self.filenames = [line.strip() for line in f if line.strip()]
            
        self.index_map = []  # 存储 (file_idx, start_idx)
        
        for file_idx, fname in enumerate(self.filenames):
            path = os.path.join(data_dir, fname)
            with open(path, 'rb') as f:
                data = pickle.load(f)
                n_samples = len(data['imu'])
                
                if n_samples < window_size:
                    continue
                
                # 计算该序列中的窗口起始位置
                starts = np.arange(0, n_samples - window_size + 1, stride)
                for s in starts:
                    self.index_map.append((file_idx, s))
        
        self.data_cache = {} # 简单缓存已加载的数据

    def __len__(self):
        return len(self.index_map)

    def _load_file(self, file_idx):
        if file_idx not in self.data_cache:
            path = os.path.join(self.data_dir, self.filenames[file_idx])
            with open(path, 'rb') as f:
                self.data_cache[file_idx] = pickle.load(f)
        return self.data_cache[file_idx]

    def __getitem__(self, idx):
        file_idx, start_idx = self.index_map[idx]
        data = self._load_file(file_idx)
        
        end_idx = start_idx + self.window_size
        
        imu = data['imu'][start_idx:end_idx].astype(np.float32)
        velocity = data['velocity'][start_idx:end_idx].astype(np.float32)
        
        if self.transform:
            imu, velocity = self.transform(imu, velocity)
            
        return torch.from_numpy(imu), torch.from_numpy(velocity)
