import torch
import torch.nn as nn

class Linear(nn.Module):
    """线性层实现"""
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x) 