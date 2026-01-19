import os
import h5py
import json
import numpy as np
import torch
from torch.utils.data import Dataset

class RoNINDataset(Dataset):
    """
    RoNIN 数据集加载器 (适配 HDF5 格式)。
    """
    def __init__(self, data_dir, list_file, window_size=200, stride=10, transform=None):
        """
        Args:
            data_dir (str): 包含序列文件夹 (如 a000_1) 的目录。
            list_file (str): 包含序列名称列表的文件。
            window_size (int): 窗口大小。
            stride (int): 滑动步长。
            transform (callable, optional): 可选变换。
        """
        self.data_dir = data_dir
        self.window_size = window_size
        self.stride = stride
        self.transform = transform
        
        with open(list_file, 'r') as f:
            self.seq_names = [line.strip() for line in f if line.strip()]
            
        self.index_map = []  # (seq_idx, start_idx)
        
        for seq_idx, seq_name in enumerate(self.seq_names):
            seq_path = os.path.join(data_dir, seq_name)
            h5_path = os.path.join(seq_path, 'data.hdf5')
            info_path = os.path.join(seq_path, 'info.json')
            
            if not os.path.exists(h5_path):
                continue
                
            with h5py.File(h5_path, 'r') as f:
                n_samples = f['synced/acce'].shape[0]
            
            start_frame = 0
            if os.path.exists(info_path):
                with open(info_path, 'r') as info_f:
                    info = json.load(info_f)
                    start_frame = info.get('start_frame', 0)
            
            valid_n = n_samples - start_frame
            if valid_n < window_size:
                continue
                
            starts = np.arange(start_frame, n_samples - window_size + 1, stride)
            for s in starts:
                self.index_map.append((seq_idx, s))
        
        self.data_cache = {}

    def __len__(self):
        return len(self.index_map)

    def _load_seq(self, seq_idx):
        if seq_idx not in self.data_cache:
            seq_name = self.seq_names[seq_idx]
            h5_path = os.path.join(self.data_dir, seq_name, 'data.hdf5')
            with h5py.File(h5_path, 'r') as f:
                # 加载 IMU 数据 (acce + gyro)
                acce = f['synced/acce'][:]
                gyro = f['synced/gyro'][:]
                imu = np.concatenate([acce, gyro], axis=-1).astype(np.float32)
                
                # 计算 Ground Truth 速度 (从 tango_pos 差分获取)
                # 注意：synced/time 是秒，频率通常为 200Hz
                pos = f['pose/tango_pos'][:]
                time = f['synced/time'][:]
                
                # 计算速度 v = dp / dt
                # 为了保持序列长度一致，我们简单地使用向前差分并补齐最后一个点
                vel = np.zeros_like(pos)
                dt = np.diff(time)
                # 避免除以 0
                dt[dt == 0] = 1e-6
                vel[:-1] = np.diff(pos, axis=0) / dt[:, None]
                vel[-1] = vel[-2]
                
                self.data_cache[seq_idx] = {
                    'imu': imu,
                    'vel': vel.astype(np.float32)
                }
        return self.data_cache[seq_idx]

    def __getitem__(self, idx):
        seq_idx, start_idx = self.index_map[idx]
        data = self._load_seq(seq_idx)
        
        end_idx = start_idx + self.window_size
        imu_win = data['imu'][start_idx:end_idx]
        vel_win = data['vel'][start_idx:end_idx]
        
        if self.transform:
            imu_win, vel_win = self.transform(imu_win, vel_win)
            
        return torch.from_numpy(imu_win), torch.from_numpy(vel_win)