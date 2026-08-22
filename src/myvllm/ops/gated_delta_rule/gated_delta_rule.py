# TODO: 想下这里要怎么支持 paged cache
import torch

from myvllm.ops.gated_delta_rule.chunk_scaled_dot_kkt import chunk_scaled_dotkkt
from myvllm.ops.gated_delta_rule.cumsum import chunk_local_cumsum
from myvllm.ops.gated_delta_rule.fused_recurrent_gdn import fused_recurrent_gdn
from myvllm.ops.gated_delta_rule.h import chunk_compute_h
from myvllm.ops.gated_delta_rule.o import chunk_compute_o
from myvllm.ops.gated_delta_rule.solve_tril import solve_tril
from myvllm.ops.gated_delta_rule.wu import gdn_compute_w_u

RCP_LN2 = 1.4426950216


def prepare_chunk_indices(cu_seqlens: torch.Tensor, chunk_size: int) -> tuple[torch.Tensor, torch.torch.Tensor]:
    seqlens = cu_seqlens[1:] - cu_seqlens[:-1]
    chunk_cnts = (seqlens + chunk_size - 1) // chunk_size
    seq_indices = [
        torch.zeros(int(chunk_cnt.item()), device=cu_seqlens.device, dtype=cu_seqlens.dtype) + i
        for (i, chunk_cnt) in enumerate(chunk_cnts)
    ]
    chunk_offsets = [
        torch.arange(0, chunk_cnt.item(), device=cu_seqlens.device, dtype=cu_seqlens.dtype) for chunk_cnt in chunk_cnts
    ]
    chunk_indices = torch.stack((torch.cat(seq_indices), torch.cat(chunk_offsets)), dim=0)
    cu_chunk_cnts = torch.cat((torch.zeros(1, dtype=cu_seqlens.dtype, device=cu_seqlens.device), chunk_cnts.cumsum(0)))

    return chunk_indices, cu_chunk_cnts


def chunk_gated_delta_rule(
    q: torch.Tensor,  # (total_tokens, num_k_heads, k_head_dim)
    k: torch.Tensor,  # (total_tokens, num_k_heads, k_head_dim)
    v: torch.Tensor,  # (total_tokens, num_v_heads, v_head_dim)
    g: torch.Tensor,  # (total_tokens, num_v_heads, v_head_dim)
    beta: torch.Tensor,  # (total_tokens, num_v_heads)
    recurrent_state: torch.Tensor,  # (max_seq_cnt, num_v_heads, k_head_dim, v_head_dim)
    recurrent_state_indices: torch.Tensor,  # (seq_cnt, )
    cu_seqlens: torch.Tensor,  # (seq_cnt+1, )
) -> torch.Tensor:
    """
    for prefill stage
    """
    CHUNK_SIZE = 32
    chunk_indices, cu_chunk_cnts = prepare_chunk_indices(cu_seqlens, CHUNK_SIZE)
    scale = k.shape[-1] ** -0.5

    g = chunk_local_cumsum(g, RCP_LN2, cu_seqlens, chunk_indices, CHUNK_SIZE)
    A = chunk_scaled_dotkkt(k, g, beta, cu_seqlens, chunk_indices, CHUNK_SIZE)
    A = solve_tril(A, cu_seqlens, chunk_indices)
    w, u = gdn_compute_w_u(k, v, beta, A, g, cu_seqlens, chunk_indices)
    h, v_new = chunk_compute_h(
        k,
        w,
        u,
        g,
        recurrent_state,
        recurrent_state_indices,
        cu_seqlens,
        cu_chunk_cnts,
        int(cu_chunk_cnts[-1].item()),
        CHUNK_SIZE,
    )
    o = chunk_compute_o(q, k, v_new, h, g, cu_seqlens, chunk_indices, scale, CHUNK_SIZE)

    return o


def recurrent_gated_delta_rule(
    q: torch.Tensor,  # (seq_cnt, num_k_heads, k_head_dim)
    k: torch.Tensor,  # (seq_cnt, num_k_heads, k_head_dim)
    v: torch.Tensor,  # (seq_cnt, num_v_heads, v_head_dim)
    g: torch.Tensor,  # (seq_cnt, num_v_heads)
    beta: torch.Tensor,  # (seq_cnt, num_v_heads)
    recurrent_state: torch.Tensor,  # (max_seq_cnt, num_v_heads, k_head_dim, v_head_dim)
    recurrent_state_indices: torch.Tensor,  # (seq_cnt, )
) -> torch.Tensor:
    """
    For decode stage.
    One token for each seq.
    """
    o = fused_recurrent_gdn(
        q,
        k,
        v,
        g,
        beta,
        recurrent_state,
        recurrent_state_indices,
    )

    return o
