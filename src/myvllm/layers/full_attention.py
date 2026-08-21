import torch
import torch.distributed as dist
from torch import nn

from myvllm.layers.linear import ColumnParallelLinear, GatedQParallelLinear
from myvllm.layers.norm import RMSNorm


class GatedAttention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        qk_head_dim: int,
        num_q_heads: int,
        v_head_dim: int,
        num_kv_heads: int,
        attention_bias: bool,
        rms_norm_eps: float,
    ) -> None:
        super().__init__()
        self.tp_rank = dist.get_rank()
        self.tp_size = dist.get_world_size()

        self.hidden_size = hidden_size
        self.qk_head_dim = qk_head_dim
        self.v_head_dim = v_head_dim
        self.total_num_q_heads = num_q_heads
        self.total_num_kv_heads = num_kv_heads
        self.attention_bias = attention_bias
        self.rms_norm_eps = rms_norm_eps
        self.scale = self.qk_head_dim**-0.5

        assert self.total_num_q_heads % self.tp_size == 0
        assert self.total_num_kv_heads % self.tp_size == 0

        self.q_proj = GatedQParallelLinear(
            self.hidden_size, self.qk_head_dim, self.total_num_q_heads, bias=self.attention_bias
        )
        self.k_proj = ColumnParallelLinear(
            self.hidden_size, self.total_num_kv_heads * self.qk_head_dim, bias=self.attention_bias
        )
        self.v_proj = ColumnParallelLinear(
            self.hidden_size, self.total_num_kv_heads * self.v_head_dim, bias=self.attention_bias
        )
        self.q_norm = RMSNorm((self.qk_head_dim,), self.rms_norm_eps, True)
        self.k_nrom = RMSNorm((self.v_head_dim,), self.rms_norm_eps, True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.empty(0)
