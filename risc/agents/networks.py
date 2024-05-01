from hive.agents.qnets.base import FunctionApproximator
import torch
from typing import List
import inspect
from hive.utils.registry import registry
import numpy as np


class Sequential(torch.nn.Module):
    def __init__(self, in_dim, modules: List[FunctionApproximator]) -> None:
        super().__init__()
        modules[0] = modules[0](np.prod(in_dim))
        for idx in range(1, len(modules)):
            modules[idx] = modules[idx]()
        self.network = torch.nn.Sequential(*modules)

    def forward(self, x):
        return self.network(x)


nn_modules = {
    x: getattr(torch.nn, x)
    for x in dir(torch.nn)
    if inspect.isclass(getattr(torch.nn, x))
    and issubclass(getattr(torch.nn, x), torch.nn.Module)
}
nn_modules["Sequential"] = Sequential

registry.register_all(FunctionApproximator, nn_modules)
