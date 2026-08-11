import torch
from einops import einsum, rearrange
from torch import nn


class RotaryEmbedding(nn.Module):
    inv_freq: torch.Tensor

    def __init__(self, dim: int, theta: float = 10000.0) -> None:
        super().__init__()
        self.dim = dim
        self.theta = theta
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, position_ids: torch.Tensor) -> torch.Tensor:
        return einsum(position_ids, self.inv_freq, "... i, j -> ... i j")


def apply_rotary_pos_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """
    x.shape: (seq_len, num_heads, head_dim)
    cos.shape: (seq_len, head_dim // 2)
    sin.shape: (seq_len, head_dim // 2)
    """
    cos = rearrange(cos, "seq_len half_dim -> seq_len 1 half_dim")
    sin = rearrange(sin, "seq_len half_dim -> seq_len 1 half_dim")
    x1, x2 = x.chunk(2, dim=-1)
    x_emb = torch.cat((x1 * cos - x2 * sin, x2 * cos + x1 * sin), dim=-1)

    return x_emb
