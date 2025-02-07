import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    def __init__(self, dim: int, epsilon: float = 1e-8):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.epsilon = epsilon

    def norm(self, x: torch.Tensor) -> torch.Tensor:
        x = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.epsilon)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x.float()).type_as(x)
        return self.weight * x
