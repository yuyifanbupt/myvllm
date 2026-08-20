import torch
import triton
import triton.language as tl


@triton.jit
def _chunk_compute_h_blockk32_kernel(
    k_ptr,  # (total_tokens, num_k_heads, k_head_dim)
    stride_k_token,
    stride_k_head,
    stride_k_dim,
    u_ptr,  # (total_tokens, num_v_heads, v_head_dim)
    stride_u_token,
    stride_u_head,
    stride_u_dim,
    w_ptr,  # (total_tokens, num_v_heads, k_head_dim),
    stride_w_token,
    stride_w_head,
    stride_w_dim,
    g_ptr,  # (total_tokens, num_v_heads)
    stride_g_token,
    stride_g_head,
    h_ptr,  # (chunk_cnt, num_v_heads, k_head_dim, v_head_dim)
    stride_h_chunk,
    stride_h_head,
    stride_h_k_dim,
    stride_h_v_dim,
    v_new_ptr,  # (total_tokens, num_v_heads, v_head_dim)
    stride_v_new_token,
    stride_v_new_head,
    stride_v_new_dim,
    recurrent_state_ptr,  # (max_seq_cnt, num_v_heads, k_head_dim, v_head_dim)
    stride_state_seq,
    stride_state_head,
    stride_state_k_dim,
    stride_state_v_dim,
    recurrent_state_indices_ptr,  # (seq_cnt, )
    cu_seqlens_ptr,  # (seq_cnt + 1, )
    cu_chunk_cnts_ptr,  # (seq_cnt + 1, )
    num_k_heads: tl.constexpr,
    k_head_dim: tl.constexpr,
    num_v_heads: tl.constexpr,
    v_head_dim: tl.constexpr,
    BLOCK_V: tl.constexpr,
    CHUNK_SIZE: tl.constexpr,
):
    BLOCK_K: tl.constexpr = 32  # # pyright: ignore[reportAssignmentType]
    seq_idx = tl.program_id(0)
    v_head_idx = tl.program_id(1)
    block_v_idx = tl.program_id(2)

    k_head_idx = v_head_idx // (num_v_heads // num_k_heads)
    seq_token_start = tl.load(cu_seqlens_ptr + seq_idx)
    seq_token_end = tl.load(cu_seqlens_ptr + seq_idx + 1)
    seqlen = seq_token_end - seq_token_start
    chunk_start_idx = tl.load(cu_chunk_cnts_ptr + seq_idx)
    chunk_end_idx = tl.load(cu_chunk_cnts_ptr + seq_idx + 1)
    seq_chunk_cnt = chunk_end_idx - chunk_start_idx
    recurrent_state_idx = tl.load(recurrent_state_indices_ptr + seq_idx)

    h_block1 = tl.zeros([1, BLOCK_K, BLOCK_V], dtype=tl.float32)
    if k_head_dim > BLOCK_K:
        h_block2 = tl.zeros([1, BLOCK_K, BLOCK_V], dtype=tl.float32)
    if k_head_dim > 2 * BLOCK_K:
        h_block3 = tl.zeros([1, BLOCK_K, BLOCK_V], dtype=tl.float32)
    if k_head_dim > 3 * BLOCK_K:
        h_block4 = tl.zeros([1, BLOCK_K, BLOCK_V], dtype=tl.float32)

    h_block_ptr = tl.make_block_ptr(
        h_ptr + chunk_start_idx * stride_h_chunk + v_head_idx * stride_h_head,
        shape=(seq_chunk_cnt, k_head_dim, v_head_dim),
        strides=(stride_h_chunk, stride_h_k_dim, stride_h_v_dim),
        offsets=(0, 0, block_v_idx * BLOCK_V),
        block_shape=(1, BLOCK_K, BLOCK_V),
        order=(2, 1, 0),
    )
    w_block_ptr = tl.make_block_ptr(
        w_ptr + seq_token_start * stride_w_token + v_head_idx * stride_w_head,
        shape=(seqlen, k_head_dim),
        strides=(stride_w_token, stride_w_dim),
        offsets=(0, 0),
        block_shape=(CHUNK_SIZE, BLOCK_K),
        order=(1, 0),
    )
    v_block_ptr = tl.make_block_ptr(
        u_ptr + seq_token_start * stride_u_token + v_head_idx * stride_u_head,
        shape=(seqlen, v_head_dim),
        strides=(stride_u_token, stride_u_dim),
        offsets=(0, block_v_idx * BLOCK_V),
        block_shape=(CHUNK_SIZE, BLOCK_V),
        order=(1, 0),
    )
    v_new_block_ptr = tl.make_block_ptr(
        v_new_ptr + seq_token_start * stride_v_new_token + v_head_idx * stride_v_new_head,
        shape=(seqlen, v_head_dim),
        strides=(stride_v_new_token, stride_v_new_dim),
        offsets=(0, block_v_idx * BLOCK_V),
        block_shape=(CHUNK_SIZE, BLOCK_V),
        order=(1, 0),
    )
    g_block_ptr = tl.make_block_ptr(
        g_ptr + seq_token_start * stride_g_token + v_head_idx * stride_g_head,
        shape=(seqlen,),
        strides=(stride_g_token,),
        offsets=(0,),
        block_shape=(CHUNK_SIZE,),
        order=(0,),
    )
    k_block_ptr = tl.make_block_ptr(
        k_ptr + seq_token_start * stride_k_token + k_head_idx * stride_k_head,
        shape=(seqlen, k_head_dim),
        strides=(stride_k_token, stride_k_dim),
        offsets=(0, 0),
        block_shape=(CHUNK_SIZE, BLOCK_K),
        order=(1, 0),
    )

    for i in range(seq_chunk_cnt):
        token_mask = tl.arange(0, CHUNK_SIZE) + i * CHUNK_SIZE < seqlen
        tl.store(h_block_ptr, h_block1, boundary_check=(1, 2))
        if k_head_dim > BLOCK_K:
            tl.store(tl.advance(h_block_ptr, (0, BLOCK_K, 0)), h_block2, boundary_check=(1, 2))
        if k_head_dim > 2 * BLOCK_K:
            tl.store(tl.advance(h_block_ptr, (0, 2 * BLOCK_K, 0)), h_block3, boundary_check=(1, 2))
        if k_head_dim > 3 * BLOCK_K:
            tl.store(tl.advance(h_block_ptr, (0, 3 * BLOCK_K, 0)), h_block4, boundary_check=(1, 2))
        h_block_ptr = tl.advance(h_block_ptr, (1, 0, 0))

        w_block = tl.load(w_block_ptr, boundary_check=(0, 1), padding_option="zero")
        v_new_block = tl.dot(w_block[None, :, :], h_block1.to(w_block.dtype))
        if k_head_dim > BLOCK_K:
            w_block = tl.load(tl.advance(w_block_ptr, (0, BLOCK_K)), boundary_check=(0, 1), padding_option="zero")
            v_new_block += tl.dot(w_block[None, :, :], h_block2.to(w_block.dtype))
        if k_head_dim > 2 * BLOCK_K:
            w_block = tl.load(tl.advance(w_block_ptr, (0, 2 * BLOCK_K)), boundary_check=(0, 1), padding_option="zero")
            v_new_block += tl.dot(w_block[None, :, :], h_block3.to(w_block.dtype))
        if k_head_dim > 3 * BLOCK_K:
            w_block = tl.load(tl.advance(w_block_ptr, (0, 3 * BLOCK_K)), boundary_check=(0, 1), padding_option="zero")
            v_new_block += tl.dot(w_block[None, :, :], h_block4.to(w_block.dtype))
        w_block_ptr = tl.advance(w_block_ptr, (CHUNK_SIZE, 0))

        v_block = tl.load(v_block_ptr, boundary_check=(0, 1), padding_option="zero")
        v_block_ptr = tl.advance(v_block_ptr, (1, 0))
        v_new_block = v_block - v_new_block
        tl.store(v_new_block_ptr, v_new_block, boundary_check=(0, 1))
        v_new_block_ptr = tl.advance(v_new_block_ptr, (CHUNK_SIZE, 0))

        g_block = tl.load(g_block_ptr, boundary_check=(0,), padding_option="zero").to(tl.float32)
        g_block_ptr = tl.advance(g_block_ptr, (CHUNK_SIZE,))
        last_tk_idx = min((i + 1) * CHUNK_SIZE, seqlen) - 1
        g_last = tl.load(g_ptr + last_tk_idx * stride_g_token + v_head_idx * stride_g_head).to(tl.float32)
        v_new_block = v_new_block * tl.where(token_mask, tl.math.exp2(g_last - g_block), 0)[:, None]
        g_last = tl.math.exp2(g_last)
        h_block1 *= g_last
        if k_head_dim > BLOCK_K:
            h_block2 *= g_last
        if k_head_dim > 2 * BLOCK_K:
            h_block3 *= g_last
        if k_head_dim > 3 * BLOCK_K:
            h_block4 *= g_last

        v_new_block = v_new_block.to(k_ptr.dtype.element_ty)

        k_block1 = tl.load(k_block_ptr, boundary_check=(0, 1), padding_option="zero")
        h_block1 += tl.dot(tl.trans(k_block1), v_new_block)
        if k_head_dim > BLOCK_K:
            k_block2 = tl.load(tl.advance(k_block_ptr, (0, BLOCK_K)), boundary_check=(0, 1), padding_option="zero")
            h_block2 += tl.dot(tl.trans(k_block2), v_new_block)
        if k_head_dim > 2 * BLOCK_K:
            k_block3 = tl.load(tl.advance(k_block_ptr, (0, 2 * BLOCK_K)), boundary_check=(0, 1), padding_option="zero")
            h_block3 += tl.dot(tl.trans(k_block3), v_new_block)
        if k_head_dim > 3 * BLOCK_K:
            k_block4 = tl.load(tl.advance(k_block_ptr, (0, 3 * BLOCK_K)), boundary_check=(0, 1), padding_option="zero")
            h_block4 += tl.dot(tl.trans(k_block4), v_new_block)
        k_block_ptr = tl.advance(k_block_ptr, (CHUNK_SIZE, 0))

    # store final state
    state_block_ptr = tl.make_block_ptr(
        recurrent_state_ptr + recurrent_state_idx * stride_state_seq + v_head_idx * stride_state_head,
        shape=(k_head_dim, v_head_dim),
        strides=(stride_state_k_dim, stride_state_v_dim),
        offsets=(0, block_v_idx * BLOCK_V),
        block_shape=(BLOCK_K, BLOCK_V),
        order=(1, 0),
    )
    tl.store(state_block_ptr, h_block1, boundary_check=(0, 1))
    if k_head_dim > BLOCK_K:
        tl.store(tl.advance(state_block_ptr, (BLOCK_K, 0)), h_block2, boundary_check=(0, 1))
    if k_head_dim > 2 * BLOCK_K:
        tl.store(tl.advance(state_block_ptr, (2 * BLOCK_K, 0)), h_block3, boundary_check=(0, 1))
    if k_head_dim > 3 * BLOCK_K:
        tl.store(tl.advance(state_block_ptr, (3 * BLOCK_K, 0)), h_block4, boundary_check=(0, 1))


def chunk_compute_h(
    k: torch.Tensor,  # (total_tokens, num_k_heads, k_head_dim)
    w: torch.Tensor,  # (total_tokens, num_v_heads, k_head_dim)
    u: torch.Tensor,  # (total_tokens, num_v_heads, v_head_dim)
    g: torch.Tensor,  # (total_tokens, num_v_heads)
    recurrent_state: torch.Tensor,  # (max_seq_cnt, num_v_heads, k_head_dim, v_head_dim)
    recurrent_state_indices: torch.Tensor,  # (seq_cnt, )
    cu_seqlens: torch.Tensor,  # (seq_cnt+1, )
    cu_chunk_cnts: torch.Tensor,  # (seq_cnt+1, )
    chunk_cnt: int,
    chunk_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    _, num_k_heads, k_head_dim = k.shape
    _, num_v_heads, v_head_dim = u.shape
    seq_cnt = cu_seqlens.size(0) - 1
    BLOCK_V = 32
    assert k_head_dim <= 128, "_chunk_compute_h_blockk32_kernel requires k_head_dim <= 128"

    h = torch.zeros(chunk_cnt, num_v_heads, k_head_dim, v_head_dim, dtype=k.dtype, device=k.device)
    v_new = torch.zeros_like(u)
    stride_k_token, stride_k_head, stride_k_dim = k.stride()
    stride_u_token, stride_u_head, stride_u_dim = u.stride()
    stride_w_token, stride_w_head, stride_w_dim = w.stride()
    stride_g_token, stride_g_head = g.stride()
    stride_h_chunk, stride_h_head, stride_h_k_dim, stride_h_v_dim = h.stride()
    stride_v_new_token, stride_v_new_head, stride_v_new_dim = v_new.stride()
    stride_state_seq, stride_state_head, stride_state_k_dim, stride_state_v_dim = recurrent_state.stride()
    grid = (seq_cnt, num_v_heads, tl.cdiv(v_head_dim, BLOCK_V))

    _chunk_compute_h_blockk32_kernel[grid](
        k,
        stride_k_token,
        stride_k_head,
        stride_k_dim,
        u,
        stride_u_token,
        stride_u_head,
        stride_u_dim,
        w,
        stride_w_token,
        stride_w_head,
        stride_w_dim,
        g,
        stride_g_token,
        stride_g_head,
        h,
        stride_h_chunk,
        stride_h_head,
        stride_h_k_dim,
        stride_h_v_dim,
        v_new,
        stride_v_new_token,
        stride_v_new_head,
        stride_v_new_dim,
        recurrent_state,
        stride_state_seq,
        stride_state_head,
        stride_state_k_dim,
        stride_state_v_dim,
        recurrent_state_indices,
        cu_seqlens,
        cu_chunk_cnts,
        num_k_heads,
        k_head_dim,
        num_v_heads,
        v_head_dim,
        BLOCK_V,
        chunk_size,
    )

    return h, v_new
