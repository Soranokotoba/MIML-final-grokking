import torch.optim as optim
from omegaconf import DictConfig

def get_optimizer(model, config: DictConfig):
    if config.type.lower() == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=config.lr.init_lr, betas=(config.beta1, config.beta2))
    else:
        raise ValueError(f"The optimizer_type {config.type} is not supported!")
    return optimizer
