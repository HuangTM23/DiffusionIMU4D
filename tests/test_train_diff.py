import unittest
import sys
import os
import subprocess

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

class TestDiffusionTrainer(unittest.TestCase):
    def test_train_loop_runs(self):
        """
        测试 train_diff.py 的主循环是否能运行。
        通过 subprocess 运行脚本，使用 debug 配置。
        """
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
        with open("configs/debug_diff_test.yaml", "w") as f:
            f.write(config_content)
            
        # 运行命令
        cmd = [
            "conda", "run", "-n", "DiffM", 
            "python3", "train_diff.py", 
            "--config", "configs/debug_diff_test.yaml",
            "--dry_run"
        ]
        
        # 设置环境变量禁用 WandB
        env = os.environ.copy()
        env["WANDB_MODE"] = "disabled"
        
        # 执行
        result = subprocess.run(cmd, env=env, capture_output=True, text=True)
        
        if result.returncode != 0:
            print("STDOUT:", result.stdout)
            print("STDERR:", result.stderr)
            
        self.assertEqual(result.returncode, 0, f"Training script failed: {result.stderr}")

if __name__ == '__main__':
    unittest.main()