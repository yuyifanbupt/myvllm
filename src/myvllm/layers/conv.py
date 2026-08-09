import math

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn


class Conv3D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple[int, int, int],
        stride: tuple[int, int, int] = (1, 1, 1),
        padding: tuple[int, int, int] = (0, 0, 0),
        groups: int = 1,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.groups = groups

        self.input_size = in_channels * math.prod(kernel_size)
        self.enable_linear = (self.kernel_size == self.stride) and not any(self.padding) and self.groups == 1

        self.weight = nn.Parameter(
            torch.empty(
                out_channels,
                in_channels // groups,
                *kernel_size,
            ),
        )

        if bias:
            self.bias = nn.Parameter(torch.empty(self.out_channels))
        else:
            self.register_parameter("bias", None)

    def _forward_matmul(self, x: torch.Tensor) -> torch.Tensor:
        K1, K2, K3 = self.kernel_size

        x = rearrange(x, "... c (t k1) (h k2) (w k3) -> ... t h w (c k1 k2 k3)", k1=K1, k2=K2, k3=K3)
        x = F.linear(x, self.weight.flatten(1), bias=self.bias)
        x = rearrange(x, "... t h w c -> ... c t h w")

        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.enable_linear:
            return self._forward_matmul(x)

        return F.conv3d(x, self.weight, self.bias, stride=self.stride, padding=self.padding, groups=self.groups)
