from abc import ABC
from enum import Enum
from math import ceil
from typing import Callable, Final, Iterable, Tuple
import torch
from torch.utils.data import Dataset, DataLoader

CONSIDERING_ASSOC: Final[bool] = False
DATA_TAKE_PROP: Final[float] = 0.9

# Exp := Value | (Exp Op Exp)
class Exp(ABC):
    def eval(self) -> int:
        raise NotImplementedError
    def print(self) -> str:
        raise NotImplementedError
    def encode(self) -> torch.Tensor:
        raise NotImplementedError

class Value(Exp):
    def __init__(self, a: int) -> None:
        self.a = a
    def eval(self) -> int:
        return self.a
    def print(self) -> str:
        return str(self.a)
    def encode(self) -> torch.Tensor:
        return torch.tensor([self.a])

class Op(ABC):
    def calc(self, a: int, b: int) -> int:
        raise NotImplementedError
    def encode(self) -> int:
        raise NotImplementedError

class OpAddModP(Op):
    def __init__(self, p: int) -> None:
        self.p = p
    def calc(self, a: int, b: int) -> int:
        return (a + b) % self.p
    def encode(self) -> int:
        return self.p + 1

encode_eq: Final[Callable[[int], int]] = lambda p: p
encode_left_bracket: Final[Callable[[int], int]] = lambda p: p + 2
encode_right_bracket: Final[Callable[[int], int]] = lambda p: p + 3
class ExpOpExp(Exp):
    def __init__(self, left: Exp, op: Op, right: Exp) -> None:
        self.left = left
        self.op = op
        self.right = right
    def eval(self) -> int:
       return self.op.calc(self.left.eval(), self.right.eval())
    def print(self) -> str:
        if CONSIDERING_ASSOC:
            return f"({self.left.print()} + {self.right.print()})"
        else:
            return f"{self.left.print()} + {self.right.print()}"
    def encode(self) -> torch.Tensor:
        if CONSIDERING_ASSOC:
            return torch.cat([
                torch.tensor([encode_left_bracket(self.op.encode())]),
                self.left.encode(),
                torch.tensor([self.op.encode()]),
                self.right.encode(),
                torch.tensor([encode_right_bracket(self.op.encode())])
            ])
        else:
            return torch.cat([
                self.left.encode(),
                torch.tensor([self.op.encode()]),
                self.right.encode()
            ])

def enum_exp_aux(p: int, remaining_k: int) -> Iterable[Exp]:
    if remaining_k == 1:
        for a in range(p):
            yield Value(a)
    else:
        if CONSIDERING_ASSOC:
            for left_k in range(1, remaining_k):
                right_k = remaining_k - left_k
                for left in enum_exp_with_prop_aux(p, left_k):
                    for right in enum_exp_with_prop_aux(p, right_k):
                        for op in [OpAddModP(p)]:
                            yield ExpOpExp(left, op, right)
        else:
            for left in enum_exp_aux(p, 1):
                for right in enum_exp_aux(p, remaining_k - 1):
                    for op in [OpAddModP(p)]:
                        yield ExpOpExp(left, op, right)
        
def enum_exp_K(p: int, K: int) -> Iterable[Exp]:
    return enum_exp_aux(p, K)

def enum_exp_with_prop(p: int, K: int, prop: float) -> Iterable[Exp]:
    for exp in enum_exp_K(p, K):
        if torch.rand(1) < prop:
            yield exp

def enum_exp_tensor_and_label_with_prop(p: int, K: int, prop: float) -> Tuple[torch.Tensor, torch.Tensor]:
    exps = []
    labels = []
    for exp in enum_exp_with_prop(p, K, prop):
        exps.append(torch.cat([exp.encode(), torch.tensor([encode_eq(p)]),]))
        labels.append(exp.eval())
    return torch.stack(exps), torch.tensor(labels)

def enum_test():
    p = 7
    K = 3
    enum_exp_tensor_and_label_with_prop(p, K, 0.3)

# generate training data
def generate_data(p: int, K: int):
    # x = torch.arange(0, p)
    # y = torch.arange(0, p)
    # x, y = torch.cartesian_prod(x, y).T

    # eq = torch.ones_like(x) * p
    # op = torch.ones_like(x) * (p + 1)

    # labels = (x + y) % p

    # inputs = torch.stack([x, op, y, eq], dim=1)
    # return inputs, labels
    res = enum_exp_tensor_and_label_with_prop(p, K, DATA_TAKE_PROP)
    print(res)
    return res

# construct dataset
def get_dataset(p, alpha, batch_size, device: torch.device, K: int = 2, **args):
    X, y = generate_data(p, K)
    total_size = X.shape[0]
    X = X.to(device)
    y = y.to(device)
    num_train = int(total_size * alpha)
    num_eval = total_size - num_train

    dataset = torch.utils.data.TensorDataset(X, y)

    train_dataset, eval_dataset = torch.utils.data.random_split(dataset, [num_train, num_eval])

    batch_size = min(batch_size, ceil(total_size / 2))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    eval_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, eval_loader
