import torch
from torch import nn


class RMSNorm(nn.Module):
    def __init__(self, normalized_shape: tuple[int, ...], eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.eps = eps

    @property
    def gamma(self):
        """Backward compatibility: gamma alias for weight"""
        return self.weight

    def rms_forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.pow(2).mean(dim=-1, keepdim=True) + self.eps
        x_norm = x * variance.rsqrt() * self.weight

        return x_norm

    def residual_rms_forward(self, x: torch.Tensor, residual: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = x + residual
        return self.rms_forward(x), x

    @torch.compile
    def forward(
        self, x: torch.Tensor, residual: torch.Tensor | None = None
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if residual is not None:
            return self.residual_rms_forward(x, residual)
        else:
            return self.rms_forward(x)


class LayerNorm(nn.Module):
    def __init__(self, normalized_shape: tuple[int, ...], eps: float = 1e-05) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps

    def _layer_norm(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.pow(2).mean(dim=-1, keepdim=True) + self.eps
        x_norm = (x - x.mean(dim=-1, keepdim=True)) * variance.rsqrt() * self.weight + self.bias

        return x_norm

    def residual_layer_norm_forward(self, x: torch.Tensor, residual: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = x + residual

        return self._layer_norm(x), x

    @torch.compile
    def forward(self, x: torch.Tensor, residual: None | torch.Tensor = None) -> tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            return self._layer_norm(x), x

        return self.residual_layer_norm_forward(x, residual)
