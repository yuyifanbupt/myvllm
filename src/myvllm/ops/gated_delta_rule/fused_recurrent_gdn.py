import torch
import triton
import triton.language as tl


@triton.jit
def _fused_recurrent_gdn(
    q_ptr,  # (seq_cnt, num_k_heads, k_head_dim)
    stride_q_seq,
    stride_q_head,
    stride_q_dim,
    k_ptr,  # (seq_cnt, num_k_heads, k_head_dim)
    stride_k_seq,
    stride_k_head,
    stride_k_dim,
    v_ptr,  # (seq_cnt, num_v_heads, v_head_dim)
    stride_v_seq,
    stride_v_head,
    stride_v_dim,
    o_ptr,  # (seq_cnt, num_v_heads, v_head_dim)
    stride_o_seq,
    stride_o_head,
    stride_o_dim,
    g_ptr,  # (seq_cnt, num_v_heads)
    stride_g_seq,
    stride_g_head,
    beta_ptr,  # (seq_cnt, num_v_heads)
    stride_beta_seq,
    stride_beta_head,
    state_ptr,  # (max_seq_cnt, num_v_heads, k_head_dim, v_head_dim)
    stride_state_seq,
    stride_state_head,
    stride_state_k_dim,
    stride_state_v_dim,
    state_indices_ptr,  # (seq_cnt, )
    scale,
    num_k_heads: tl.constexpr,
    k_head_dim: tl.constexpr,
    num_v_heads: tl.constexpr,
    v_head_dim: tl.constexpr,
    BLOCK_V: tl.constexpr,
):
    seq_idx = tl.program_id(0)
    v_head_idx = tl.program_id(1)
    v_block_idx = tl.program_id(2)
    state_idx = tl.load(state_indices_ptr + seq_idx)
    k_head_idx = v_head_idx // (num_v_heads // num_k_heads)

    g = tl.exp(tl.load(g_ptr + seq_idx * stride_g_seq + v_head_idx * stride_g_head).to(tl.float32))
    beta = tl.load(beta_ptr + seq_idx * stride_beta_seq + v_head_idx * stride_beta_head).to(tl.float32)
    state_block_ptr = tl.make_block_ptr(
        state_ptr + state_idx * stride_state_seq + v_head_idx * stride_state_head,
        shape=(k_head_dim, v_head_dim),
        strides=(stride_state_k_dim, stride_state_v_dim),
        offsets=(0, v_block_idx * BLOCK_V),
        block_shape=(k_head_dim, BLOCK_V),
        order=(1, 0),
    )
    state_block = tl.load(state_block_ptr, boundary_check=(1,), padding_option="zero").to(tl.float32)
    q_block_ptr = tl.make_block_ptr(
        q_ptr + seq_idx * stride_q_seq + k_head_idx * stride_q_head,
        shape=(k_head_dim,),
        strides=(stride_q_dim,),
        offsets=(0,),
        block_shape=(k_head_dim,),
        order=(0,),
    )
    q_block = tl.load(q_block_ptr).to(tl.float32) * scale
    k_block_ptr = tl.make_block_ptr(
        k_ptr + seq_idx * stride_k_seq + k_head_idx * stride_k_head,
        shape=(k_head_dim,),
        strides=(stride_k_dim,),
        offsets=(0,),
        block_shape=(k_head_dim,),
        order=(0,),
    )
    k_block = tl.load(k_block_ptr).to(tl.float32)
    v_block_ptr = tl.make_block_ptr(
        v_ptr + seq_idx * stride_v_seq + v_head_idx * stride_v_head,
        shape=(v_head_dim,),
        strides=(stride_v_dim,),
        offsets=(v_block_idx * BLOCK_V,),
        block_shape=(BLOCK_V,),
        order=(0,),
    )
    v_block = tl.load(v_block_ptr, boundary_check=(0,), padding_option="zero").to(tl.float32)
    o_block_ptr = tl.make_block_ptr(
        o_ptr + seq_idx * stride_o_seq + v_head_idx * stride_o_head,
        shape=(v_head_dim,),
        strides=(stride_v_dim,),
        offsets=(v_block_idx * BLOCK_V,),
        block_shape=(BLOCK_V,),
        order=(0,),
    )

    state_block *= g
    v_block = beta * (v_block - tl.sum(k_block[:, None] * state_block, axis=0))
    state_block += k_block[:, None] * v_block[None, :]
    o_block = tl.sum(q_block[:, None] * state_block, axis=0)

    tl.store(state_block_ptr, state_block, boundary_check=(1,))
    tl.store(o_block_ptr, o_block, boundary_check=(0,))


def fused_recurrent_gdn(
    q: torch.Tensor,  # (seq_cnt, num_k_heads, k_head_dim)
    k: torch.Tensor,  # (seq_cnt, num_k_heads, k_head_dim)
    v: torch.Tensor,  # (seq_cnt, num_v_heads, v_head_dim)
    g: torch.Tensor,  # (seq_cnt, num_v_heads)
    beta: torch.Tensor,  # (seq_cnt, num_v_heads)
    recurrent_state: torch.Tensor,  # (max_seq_cnt, num_v_heads, k_head_dim, v_head_dim)
    recurrent_state_indices: torch.Tensor,  # (seq_cnt, )
) -> torch.Tensor:
    seq_cnt, num_k_heads, k_head_dim = q.shape
    _, num_v_heads, v_head_dim = v.shape
    o = torch.zeros_like(v)
    scale = k.shape[-1] ** -0.5
    BLOCK_V = 8

    stride_q_seq, stride_q_head, stride_q_dim = q.stride()
    stride_k_seq, stride_k_head, stride_k_dim = k.stride()
    stride_v_seq, stride_v_head, stride_v_dim = v.stride()
    stride_o_seq, stride_o_head, stride_o_dim = o.stride()
    stride_g_seq, stride_g_head = g.stride()
    stride_beta_seq, stride_beta_head = beta.stride()
    stride_state_seq, stride_state_head, stride_state_k_dim, stride_state_v_dim = recurrent_state.stride()
    grid = [seq_cnt, num_v_heads, tl.cdiv(v_head_dim, BLOCK_V)]

    _fused_recurrent_gdn[grid](
        q,
        stride_q_seq,
        stride_q_head,
        stride_q_dim,
        k,
        stride_k_seq,
        stride_k_head,
        stride_k_dim,
        v,
        stride_v_seq,
        stride_v_head,
        stride_v_dim,
        o,
        stride_o_seq,
        stride_o_head,
        stride_o_dim,
        g,
        stride_g_seq,
        stride_g_head,
        beta,
        stride_beta_seq,
        stride_beta_head,
        recurrent_state,
        stride_state_seq,
        stride_state_head,
        stride_state_k_dim,
        stride_state_v_dim,
        recurrent_state_indices,
        scale,
        num_k_heads,
        k_head_dim,
        num_v_heads,
        v_head_dim,
        BLOCK_V,
    )

    return o
