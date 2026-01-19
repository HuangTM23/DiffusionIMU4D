try:
    import wandb
except ImportError:
    wandb = None
    print("Warning: wandb not installed. Logging will be disabled.")

def init_logger(project_name="Diffusion4d", config=None, entity=None):
    """
    初始化 WandB 日志记录器。
    
    Args:
        project_name (str): WandB 项目名称。
        config (dict): 超参数配置字典。
        entity (str): WandB 实体（用户名或团队名）。
    """
    if wandb is not None:
        wandb.init(project=project_name, config=config, entity=entity)
    else:
        print(f"Mock Init: Project={project_name}, Config={config}")

def log_metrics(metrics, step=None):
    """
    记录实验指标。
    
    Args:
        metrics (dict): 指标字典，如 {"loss": 0.5}。
        step (int): 当前训练步数或 epoch。
    """
    if wandb is not None:
        wandb.log(metrics, step=step)
    else:
        print(f"Mock Log: Step={step}, Metrics={metrics}")
