import torch
import triton
import triton.language as tl


@triton.jit
def _chunk_compute_o_kernel(
    q_ptr,  # (total_tokens, num_k_heads, k_head_dim)
    stride_q_token,
    stride_q_head,
    stride_q_dim,
    k_ptr,  # (total_tokens, num_k_heads, k_head_dim)
    stride_k_token,
    stride_k_head,
    stride_k_dim,
    v_new_ptr,  # (total_tokens, num_v_heads, v_head_dim)
    stride_v_new_token,
    stride_v_new_head,
    stride_v_new_dim,
    h_ptr,  # (chunk_cnt, num_v_heads, k_head_dim, v_head_dim) 这里跟 gated delta network 论文的shape不一样, 原论文里面是 (..., v_head_dim, k_head_dim)
    stride_h_chunk,
    stride_h_head,
    stride_h_k_dim,
    stride_h_v_dim,
    g_ptr,  # (total_tokens, num_v_heads)
    stride_g_token,
    stride_g_head,
    o_ptr,  # (total_tokens, num_v_heads, v_head_dim)
    stride_o_token,
    stride_o_head,
    stride_o_dim,
    cu_seqlens_ptr,  # (seq_cnt+1, )
    chunk_indices_ptr,  # (chunk_cnt, 2)
    scale: float,
    num_k_heads: tl.constexpr,
    k_head_dim: tl.constexpr,
    num_v_heads: tl.constexpr,
    v_head_dim: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_V: tl.constexpr,
    CHUNK_SIZE: tl.constexpr,
):
    chunk_idx = tl.program_id(0)
    v_head_idx = tl.program_id(1)
    v_block_idx = tl.program_id(2)

    seq_idx = tl.load(chunk_indices_ptr + chunk_idx * 2)
    chunk_offset = tl.load(chunk_indices_ptr + chunk_idx * 2 + 1)
    seq_token_start = tl.load(cu_seqlens_ptr + seq_idx)
    seq_token_end = tl.load(cu_seqlens_ptr + seq_idx + 1)
    seqlen = seq_token_end - seq_token_start
    k_head_idx = v_head_idx // (num_v_heads // num_k_heads)

    q_block_ptr = tl.make_block_ptr(
        q_ptr + seq_token_start * stride_q_token + k_head_idx * stride_q_head,
        shape=(seqlen, k_head_dim),
        strides=(stride_q_token, stride_q_dim),
        offsets=(chunk_offset * CHUNK_SIZE, 0),
        block_shape=(CHUNK_SIZE, BLOCK_K),
        order=(1, 0),
    )
    k_block_ptr = tl.make_block_ptr(
        k_ptr + seq_token_start * stride_k_token + k_head_idx * stride_k_head,
        shape=(seqlen, k_head_dim),
        strides=(stride_k_token, stride_k_dim),
        offsets=(chunk_offset * CHUNK_SIZE, 0),
        block_shape=(CHUNK_SIZE, BLOCK_K),
        order=(1, 0),
    )
    h_block_ptr = tl.make_block_ptr(
        h_ptr + chunk_idx * stride_h_chunk + v_head_idx * stride_h_head,
        shape=(k_head_dim, v_head_dim),
        strides=(stride_h_k_dim, stride_h_v_dim),
        offsets=(0, v_block_idx * BLOCK_V),
        block_shape=(BLOCK_K, BLOCK_V),
        order=(1, 0),
    )
    o_block = tl.zeros([CHUNK_SIZE, BLOCK_V], dtype=tl.float32)
    a_block = tl.zeros([CHUNK_SIZE, CHUNK_SIZE], dtype=tl.float32)

    for i in range(tl.cdiv(k_head_dim, BLOCK_K)):
        q_block = tl.load(q_block_ptr, boundary_check=(0, 1), padding_option="zero")
        q_block_ptr = tl.advance(q_block_ptr, (0, 1))
        k_block = tl.load(k_block_ptr, boundary_check=(0, 1), padding_option="zero")
        k_block_ptr = tl.advance(k_block_ptr, (0, 1))
        h_block = tl.load(h_block_ptr, boundary_check=(0, 1), padding_option="zero")
        h_block_ptr = tl.advance(h_block_ptr, (1, 0))

        o_block += tl.dot(q_block, h_block)
        a_block += tl.dot(q_block, tl.trans(k_block))

    g_block_ptr = tl.make_block_ptr(
        g_ptr + seq_token_start * stride_g_token + v_head_idx * stride_g_head,
        shape=(seqlen,),
        strides=(stride_g_token,),
        offsets=(chunk_offset * CHUNK_SIZE,),
        block_shape=(CHUNK_SIZE,),
        order=(0,),
    )
    g_block = tl.load(g_block_ptr, boundary_check=(0,), padding_option="zero")
    o_block = o_block * tl.math.exp2(g_block)[:, None]
    # 原论文这里的公式只使用了 causal_mask, 但实际上这里应该还需要 decay.
    # 这个 issue 也提了相同的问题: https://github.com/NVlabs/GatedDeltaNet/issues/15
    a_block = a_block * tl.math.exp2(g_block[:, None] - g_block[None, :])

    # apply causal mask
    l = tl.arange(0, CHUNK_SIZE)
    chunk_mask = l + chunk_offset * CHUNK_SIZE < seqlen
    causal_mask = (l[:, None] >= l[None, :]) & (chunk_mask[:, None] & chunk_mask)
    a_block = tl.where(a_block, causal_mask, a_block, 0)

    v_new_block_ptr = tl.make_block_ptr(
        v_new_ptr + seq_token_start * stride_v_new_token + v_head_idx * stride_v_new_head,
        shape=(seqlen, v_head_dim),
        strides=(stride_v_new_token, stride_v_new_dim),
        offsets=(chunk_offset * CHUNK_SIZE, v_block_idx * BLOCK_V),
        block_shape=(CHUNK_SIZE, BLOCK_V),
        order=(1, 0),
    )
    v_new_block = tl.load(v_new_block_ptr, boundary_check=(0, 1), padding_option="zero")
    o_block = o_block * scale + tl.dot(a_block, v_new_block) * scale

    o_block_ptr = tl.make_block_ptr(
        o_ptr + seq_token_start * stride_o_token + v_head_idx * stride_o_head,
        shape=(seqlen, v_head_dim),
        strides=(stride_o_token, stride_o_dim),
        offsets=(chunk_offset * CHUNK_SIZE, v_block_idx * BLOCK_V),
        block_shape=(CHUNK_SIZE, BLOCK_V),
        order=(1, 0),
    )
    tl.store(o_block_ptr, o_block, boundary_check=(0, 1))


def chunk_compute_o(
    q: torch.Tensor,  # (total_tokens, num_k_heads, k_head_dim)
    k: torch.Tensor,  # (total_tokens, num_k_heads, k_head_dim)
    v_new: torch.Tensor,  # (total_tokens, num_v_heads, v_head_dim)
    h: torch.Tensor,  # (chunk_cnt, num_v_heads, k_head_dim, v_head_dim)
    g: torch.Tensor,  # (chunk_cnt, num_v_heads)
    cu_seqlens: torch.Tensor,  # (seq_cnt+1, )
    chunk_indices: torch.Tensor,  # (chunk_cnt, 2)
    scale: float,
    chunk_size: int = 32,
) -> torch.Tensor:
    _, num_k_heads, k_head_dim = k.shape
    _, num_v_heads, v_head_dim = v_new.shape
    o = torch.zeros_like(v_new)
    stride_q_token, stride_q_head, stride_q_dim = q.stride()
    stride_k_token, stride_k_head, stride_k_dim = k.stride()
    stride_v_new_token, stride_v_new_head, stride_v_new_dim = v_new.stride()
    stride_h_chunk, stride_h_head, stride_h_k_dim, stride_h_v_dim = h.stride()
    stride_g_token, stride_g_head = g.stride()
    stride_o_token, stride_o_head, stride_o_dim = o.stride()
    chunk_cnt = chunk_indices.size(0)
    BLOCK_K, BLOCK_V = 32, 32
    grid = [chunk_cnt, num_v_heads, tl.cdiv(v_head_dim, BLOCK_V)]

    _chunk_compute_o_kernel[grid](
        q,
        stride_q_token,
        stride_q_head,
        stride_q_dim,
        k,
        stride_k_token,
        stride_k_head,
        stride_k_dim,
        v_new,
        stride_v_new_token,
        stride_v_new_head,
        stride_v_new_dim,
        h,
        stride_h_chunk,
        stride_h_head,
        stride_h_k_dim,
        stride_h_v_dim,
        g,
        stride_g_token,
        stride_g_head,
        o,
        stride_o_token,
        stride_o_head,
        stride_o_dim,
        cu_seqlens,
        chunk_indices,
        scale,
        num_k_heads,
        k_head_dim,
        num_v_heads,
        v_head_dim,
        BLOCK_K,
        BLOCK_V,
        chunk_size,
    )

    return o
