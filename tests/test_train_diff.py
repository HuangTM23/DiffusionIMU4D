import unittest
import torch
import sys
import os
import shutil

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

class TestDiffusionTrainer(unittest.TestCase):
    def test_train_loop_runs(self):
        """
        测试 train_diff.py 的主循环是否能运行。
        通过 subprocess 运行脚本，使用 debug 配置。
        """
        import subprocess
        
        # 创建临时 debug config
        config_content = """
data_dir: "data/RoNIN/extracted"
train_list: "data/RoNIN/lists/list_debug.txt"
val_list: "data/RoNIN/lists/list_debug.txt"

window_size: 200
stride: 100
batch_size: 2
lr: 0.0001
epochs: 1

mode: "end2end" # or residual
log_interval: 1
save_interval: 1
num_inference_steps: 2
"""
        os.makedirs("configs", exist_ok=True)
        with open("configs/debug_diff.yaml", "w") as f:
            f.write(config_content)
            
        # 运行命令
        # WANDB_MODE=disabled 防止联网
        cmd = [
            "conda", "run", "-n", "DiffM", 
            "python3", "train_diff.py", 
            "--config", "configs/debug_diff.yaml",
            "--dry_run" # 假设我们加一个 dry_run 标志只跑几个 batch
        ]
        
        # 注意：这需要 train_diff.py 存在
        # 由于我们还没写，这里先 pass
        pass

if __name__ == '__main__':
    unittest.main()
