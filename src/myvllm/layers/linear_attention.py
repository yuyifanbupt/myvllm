from collections.abc import Callable

import torch
import torch.distributed as dist
import torch.nn.functional as F
from einops import rearrange
from torch import nn

from myvllm.layers.linear import ColumnParallelLinear, QKVParallelLinear, RowParallelLinear
from myvllm.layers.norm import RMSNormGated
from myvllm.ops.causal_conv1d import causal_conv1d_fn, causal_conv1d_update
from myvllm.ops.gated_delta_rule.gated_delta_rule import chunk_gated_delta_rule, recurrent_gated_delta_rule


class GatedDeltaNet(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_k_heads: int,
        num_v_heads: int,
        k_head_dim: int,
        v_head_dim: int,
        conv_kernel_size: int,
        act_fn: Callable[[torch.Tensor], torch.Tensor] = F.silu,
        rms_norm_eps: float = 1e-06,
    ) -> None:
        super().__init__()
        self.tp_rank = dist.get_rank()
        self.tp_size = dist.get_world_size()

        assert num_k_heads % self.tp_size == 0
        assert num_v_heads % self.tp_size == 0

        self.hidden_size = hidden_size
        self.total_num_k_heads = num_k_heads
        self.total_num_v_heads = num_v_heads
        self.num_k_heads = num_k_heads // self.tp_size
        self.num_v_heads = num_v_heads // self.tp_size
        self.k_head_dim = k_head_dim
        self.v_head_dim = v_head_dim
        self.total_k_size = self.k_head_dim * self.total_num_k_heads
        self.total_v_size = self.v_head_dim * self.total_num_v_heads
        self.k_size = self.k_head_dim * self.num_k_heads
        self.v_size = self.v_head_dim * self.num_v_heads
        self.conv_kernel_size = conv_kernel_size
        self.act_fn = act_fn
        self.rms_norm_eps = rms_norm_eps
        self.conv1d = QKVParallelLinear(
            conv_kernel_size,
            self.k_head_dim,
            self.k_head_dim,
            self.v_head_dim,
            self.total_num_k_heads,
            self.total_num_k_heads,
            self.total_num_v_heads,
        )
        self.dt_bias = nn.Parameter(torch.ones(self.num_v_heads))
        self.A_log = nn.Parameter(torch.log(torch.empty(self.num_v_heads).uniform_(0, 16)))
        self.norm = RMSNormGated((self.v_head_dim,), self.rms_norm_eps)
        self.out_proj = RowParallelLinear(self.total_v_size, self.hidden_size, bias=False)
        self.in_proj_qkv = QKVParallelLinear(
            self.hidden_size,
            self.k_head_dim,
            self.k_head_dim,
            self.v_head_dim,
            self.total_num_k_heads,
            self.total_num_k_heads,
            self.total_num_v_heads,
            bias=False,
        )
        self.in_proj_z = ColumnParallelLinear(self.hidden_size, self.total_v_size, bias=False)
        self.in_proj_b = ColumnParallelLinear(self.hidden_size, self.total_num_v_heads, bias=False)
        self.in_proj_a = ColumnParallelLinear(self.hidden_size, self.total_num_v_heads, bias=False)
        self.recurrent_state: torch.Tensor = torch.empty(0)
        self.conv_state: torch.Tensor = torch.empty(0)

    def forward(
        self,
        x: torch.Tensor,
        cu_seqlens: torch.Tensor,
        conv_state_indices: torch.Tensor,
        recurrent_state_indices: torch.Tensor,
        is_prefill: bool,
    ) -> torch.Tensor:
        qkv = self.in_proj_qkv(x)
        z: torch.Tensor = self.in_proj_z(x)
        a: torch.Tensor = self.in_proj_a(x)
        b: torch.Tensor = self.in_proj_b(x)
        beta = b.sigmoid()
        g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)

        if is_prefill:
            qkv = causal_conv1d_fn(x, self.conv1d.weight, self.conv_state, conv_state_indices, cu_seqlens)
        else:
            qkv = causal_conv1d_update(x, self.conv1d.weight, self.conv_state, conv_state_indices)

        q, k, v = torch.split(qkv, [self.k_size, self.k_size, self.v_size], dim=-1)
        q = rearrange(
            q,
            "total_tokens (num_k_heads k_head_dim) -> total_tokens num_k_heads k_head_dim",
            k_head_dim=self.k_head_dim,
        )
        k = rearrange(
            k,
            "total_tokens (num_k_heads k_head_dim) -> total_tokens num_k_heads k_head_dim",
            k_head_dim=self.k_head_dim,
        )
        v = rearrange(
            v,
            "total_tokens (num_v_heads v_head_dim) -> total_tokens num_v_heads v_head_dim",
            v_head_dim=self.v_head_dim,
        )
        q = q * torch.rsqrt(q.pow(2).sum(dim=-1, keepdim=True) + 1e-6)  # l2 norm
        k = k * torch.rsqrt(k.pow(2).sum(dim=-1, keepdim=True) + 1e-6)  # l2 norm

        if is_prefill:
            o = chunk_gated_delta_rule(q, k, v, g, beta, self.recurrent_state, recurrent_state_indices, cu_seqlens)
        else:
            o = recurrent_gated_delta_rule(q, k, v, g, beta, self.recurrent_state, recurrent_state_indices)

        o = self.norm(o, z)
        output = self.out_proj(z)

        return output
