from torch.optim.lr_scheduler import LambdaLR
from omegaconf import DictConfig

def lr_linear_warmup(epoch):
    warmup_steps = 10
    if epoch < warmup_steps:
        return float(epoch + 1) / float(warmup_steps)
    else:
        return 1.0
    
def get_scheduler(optimizer, config: DictConfig):
    if config.type == 'const':
        if config.warm_up == False:
            warmup_steps = 0
        else:
            warmup_steps = config.warmup_steps
        scheduler = LambdaLR(optimizer, lambda ep: float(ep + 1) / float(warmup_steps) if ep < warmup_steps else 1.0)
    else:
        raise ValueError(f"The lr_scheduler_type {config.type} is not supported!")
    return scheduler