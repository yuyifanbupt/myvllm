from collections.abc import Callable

import torch
import torch.distributed as dist
import torch.nn.functional as F
from einops import rearrange
from torch import nn

from myvllm.layers.attention import flash_attention_prefill
from myvllm.layers.linear import QKVParallelLinear, RowParallelLinear
from myvllm.layers.rotary_embedding import apply_rotary_pos_emb


class VisionAttention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        head_dim: int,
        num_heads: int,
        head_v_dim: None | int = None,
        num_kv_heads: None | int = None,
    ) -> None:
        super().__init__()
        self.tp_rank = dist.get_rank()
        self.tp_size = dist.get_world_size()
        self.total_num_heads = num_heads
        self.total_num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads
        self.head_dim = head_dim
        self.head_v_dim = head_v_dim if head_v_dim is not None else head_dim

        assert self.total_num_heads % self.tp_size == 0
        assert self.total_num_kv_heads % self.tp_size == 0

        self.num_heads = self.total_num_heads // self.tp_size
        self.num_kv_heads = self.total_num_kv_heads // self.tp_size
        self.q_size = self.head_dim * self.num_heads
        self.k_size = self.head_dim * self.num_kv_heads
        self.v_size = self.head_v_dim * self.num_kv_heads
        self.scale = self.head_dim**-0.5

        self.qkv = QKVParallelLinear(
            hidden_size,
            head_dim,
            num_heads,
            total_num_kv_heads=num_kv_heads,
            v_head_size=head_v_dim,
            bias=True,
        )
        self.proj = RowParallelLinear(
            self.head_v_dim * self.total_num_kv_heads,
            hidden_size,
            bias=True,
        )

    def forward(
        self,
        x: torch.Tensor,
        cu_seqlens: torch.Tensor,
        rotary_pos_emb_cos: torch.Tensor | None,
        rotary_pos_emb_sin: torch.Tensor | None,
    ) -> torch.Tensor:
        """
        x.shape (seq_len, hidden_size)
        """
        # TODO: 标记入参和出参 shape
        qkv: torch.Tensor = self.qkv(x)
        q, k, v = qkv.split([self.q_size, self.k_size, self.v_size], dim=-1)
        q = rearrange(q, "seq_len (num_heads head_dim) -> seq_len num_heads head_dim", head_dim=self.head_dim)
        k = rearrange(k, "seq_len (num_kv_heads head_dim) -> seq_len num_kv_heads head_dim", head_dim=self.head_dim)
        v = rearrange(
            v, "seq_len (num_kv_heads head_v_dim) -> seq_len num_kv_heads head_dim", head_v_dim=self.head_v_dim
        )

        if rotary_pos_emb_cos is not None and rotary_pos_emb_sin is not None:
            q = apply_rotary_pos_emb(q, rotary_pos_emb_cos, rotary_pos_emb_sin)
            k = apply_rotary_pos_emb(k, rotary_pos_emb_cos, rotary_pos_emb_sin)

        o = flash_attention_prefill(
            q,
            k,
            v,
            cu_seqlens,
            self.scale,
            self.num_heads,
            self.num_kv_heads,
            self.head_dim,
            self.head_v_dim,
            False,
        )
        o = rearrange(o, "seq_len num_heads head_v_dim -> seq_len (num_heads head_v_dim)")

        return self.proj(o)


class VisionBlock(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        head_dim: int,
        num_heads: int,
        mlp_hidden_dim: int,
        act_fn: Callable[[torch.Tensor], torch.Tensor] = F.silu,
    ) -> None:
        super().__init__()
