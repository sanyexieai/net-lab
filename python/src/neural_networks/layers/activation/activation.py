import torch
import torch.nn as nn

class ReLU(nn.Module):
    """ReLU激活函数实现"""
    def __init__(self, inplace: bool = False):
        super().__init__()
        self.relu = nn.ReLU(inplace=inplace)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(x)

class Sigmoid(nn.Module):
    """Sigmoid激活函数实现"""
    def __init__(self):
        super().__init__()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.sigmoid(x) 