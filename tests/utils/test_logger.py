import unittest
from unittest.mock import patch, MagicMock
import sys
import os

# Ensure we can import utils
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from utils.logger import init_logger, log_metrics

class TestLogger(unittest.TestCase):
    @patch('utils.logger.wandb')
    def test_init_logger(self, mock_wandb):
        """测试 logger 初始化是否正确调用了 wandb.init"""
        config = {"lr": 0.001, "batch_size": 32}
        project_name = "Diffusion4d"
        
        init_logger(project_name=project_name, config=config)
        
        mock_wandb.init.assert_called_once()
        call_args = mock_wandb.init.call_args[1]
        self.assertEqual(call_args['project'], project_name)
        self.assertEqual(call_args['config'], config)

    @patch('utils.logger.wandb')
    def test_log_metrics(self, mock_wandb):
        """测试 log_metrics 是否正确调用了 wandb.log"""
        metrics = {"loss": 0.5, "mse": 0.01}
        step = 10
        
        log_metrics(metrics, step=step)
        
        mock_wandb.log.assert_called_once_with(metrics, step=step)

if __name__ == '__main__':
    unittest.main()
