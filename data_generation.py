from math import ceil
import torch
from torch.utils.data import Dataset, DataLoader

# generate training data
def generate_data(p):
    x = torch.arange(0, p)
    y = torch.arange(0, p)
    x, y = torch.cartesian_prod(x, y).T

    eq = torch.ones_like(x) * p
    op = torch.ones_like(x) * (p + 1)

    labels = (x + y) % p

    inputs = torch.stack([x, op, y, eq], dim=1)

    return inputs, labels

# construct dataset
def get_dataset(p, alpha, batch_size, **args):
    X, y = generate_data(p)
    total_size = X.shape[0]
    
    num_train = int(total_size * alpha)
    num_eval = total_size - num_train

    dataset = torch.utils.data.TensorDataset(X, y)

    train_dataset, eval_dataset = torch.utils.data.random_split(dataset, [num_train, num_eval])

    batch_size = min(batch_size, ceil(total_size / 2))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    eval_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, eval_loader
