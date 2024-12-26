import torch.optim as optim
from omegaconf import DictConfig

def get_optimizer(model, config: DictConfig):
    match config.type.lower():
        case 'adamw':
            optimizer = optim.AdamW(model.parameters(), lr=config.lr.init_lr, betas=(config.beta1, config.beta2))
        case 'adam':
            optimizer = optim.Adam(model.parameters(), lr=config.lr.init_lr, betas=(config.beta1, config.beta2))
        case 'rmsprop':
            optimizer = optim.RMSprop(model.parameters(), lr=config.lr.init_lr, alpha=config.alpha, eps=config.eps, weight_decay=config.weight_decay, momentum=config.momentum, centered=config.centered)
        case 'sgd':
            optimizer = optim.SGD(model.parameters(), lr=config.lr.init_lr, momentum=config.momentum, weight_decay=config.weight_decay, nesterov=config.nesterov)
        case _:
            raise ValueError(f"The optimizer_type {config.type} is not supported!")
    return optimizer
