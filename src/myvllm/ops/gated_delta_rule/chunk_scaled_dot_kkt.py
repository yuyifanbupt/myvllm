import torch
import triton
import triton.language as tl


@triton.jit
def _chunk_scaled_dot_kkt_kernel(
    k_ptr,  # (total_tokens, num_k_heads, k_head_dim)
    stride_k_token,
    stride_k_head,
    stride_k_dim,
    g_ptr,  # (total_tokens, num_v_heads)
    stride_g_token,
    stride_g_head,
    beta_ptr,  # (total_tokens, num_v_heads)
    stride_beta_token,
    stride_beta_head,
    A_ptr,  # (total_tokens, num_v_heads, chunk_size)
    stride_A_token,
    stride_A_head,
    stride_A_chunk,
    cu_seqlens_ptr,  # (seq_cnt+1, )
    chunk_indices_ptr,  # (chunk_cnt, 2)
    num_k_heads: tl.constexpr,
    k_head_dim: tl.constexpr,
    num_v_heads: tl.constexpr,
    BLOCK_K: tl.constexpr,
    CHUNK_SIZE: tl.constexpr,
):
    """
    num_v_heads >= num_k_heads && num_v_heads % num_k_heads == 0
    """
    chunk_idx = tl.program_id(0)
    v_head_idx = tl.program_id(1)
    k_head_idx = v_head_idx // (num_v_heads // num_k_heads)
    seq_idx = tl.load(chunk_indices_ptr + chunk_idx * 2)
    chunk_offset = tl.load(chunk_indices_ptr + chunk_idx * 2 + 1)
    token_offset_start = tl.load(cu_seqlens_ptr + seq_idx)
    token_offset_end = tl.load(cu_seqlens_ptr + seq_idx + 1)
    seqlen = token_offset_end - token_offset_start

    k_block_ptr = tl.make_block_ptr(
        k_ptr + token_offset_start * stride_k_token + k_head_idx * stride_k_head,
        shape=(seqlen, k_head_dim),
        strides=(stride_k_token, stride_k_dim),
        offsets=(chunk_offset * CHUNK_SIZE, 0),
        block_shape=(CHUNK_SIZE, BLOCK_K),
        order=(1, 0),
    )
    a_block = tl.zeros([CHUNK_SIZE, CHUNK_SIZE], dtype=tl.float32)

    for i in range(tl.cdiv(k_head_dim, BLOCK_K)):
        k_block = tl.load(k_block_ptr, boundary_check=(0, 1), padding_option="zero")
        k_block_ptr = tl.advance(k_block_ptr, (0, 1))
        a_block += tl.dot(k_block, tl.trans(k_block))

    g_block_ptr = tl.make_block_ptr(
        g_ptr + token_offset_start * stride_g_token,
        shape=(seqlen, num_v_heads),
        strides=(stride_g_token, stride_g_head),
        offsets=(chunk_offset * CHUNK_SIZE, v_head_idx),
        block_shape=(CHUNK_SIZE, 1),
        order=(1, 0),
    )
    g_block = tl.unsqueeze(tl.load(g_block_ptr, boundary_check=(0,), padding_option="zero"), 1)  # (CHUNK_SIZE, )
    g_block_diff = g_block[:, None] - g_block[None, :]
    a_block *= tl.math.exp2(g_block_diff)

    beta_block_ptr = tl.make_block_ptr(
        beta_ptr + token_offset_start * stride_beta_token,
        shape=(seqlen, num_v_heads),
        strides=(stride_beta_token, stride_beta_head),
        offsets=(chunk_offset * CHUNK_SIZE, v_head_idx),
        block_shape=(CHUNK_SIZE, 1),
        order=(1, 0),
    )
    beta_block = tl.load(beta_block_ptr, boundary_check=(0,), padding_option="zero")
    a_block *= beta_block

    # apply strict lower triangle
    l = tl.arange(0, CHUNK_SIZE) + chunk_offset * CHUNK_SIZE
    seqlen_mask = l < seqlen
    a_mask = l[:, None] > l[None, :] & (seqlen_mask[:, None] & seqlen_mask)
    a_block = tl.where(a_mask, a_block, 0)

    # store result
    a_block_ptr = tl.make_block_ptr(
        A_ptr + token_offset_start * stride_A_token + v_head_idx * stride_A_head,
        shape=(seqlen, CHUNK_SIZE),
        strides=(stride_A_token, stride_A_chunk),
        offsets=(chunk_offset * CHUNK_SIZE, 0),
        block_shape=(CHUNK_SIZE, CHUNK_SIZE),
        order=(1, 0),
    )
    tl.store(a_block_ptr, a_block, boundary_check=(0,))


def chunk_scaled_dotkkt(
    k: torch.Tensor,  # (total_tokens, num_k_heads, k_head_dim)
    g: torch.Tensor,  # (total_tokens, num_v_heads)
    beta: torch.Tensor,  # (total_tokens, num_v_heads)
    cu_seqlens: torch.Tensor,  # (seq_cnt+1, )
    chunk_indices: torch.Tensor,  # (chunk_cnt, 2)
    chunk_size: int = 32,
) -> torch.Tensor:
    stride_k_token, stride_k_head, stride_k_dim = k.stride()
    stride_g_token, stride_g_head = g.stride()
    stride_beta_token, stride_beta_head = beta.stride()
    total_tokens, num_v_heads = beta.shape
    A = torch.empty(total_tokens, num_v_heads, chunk_size, device=k.device, dtype=torch.float32)
    stride_A_token, stride_A_head, stride_A_chunk = A.stride()
    _, num_k_heads, k_head_dim = k.shape
    chunk_cnt = chunk_indices.size(0)
    grid = (chunk_cnt, num_v_heads)
    BLOCK_K = 32

    _chunk_scaled_dot_kkt_kernel[grid](
        k,
        stride_k_token,
        stride_k_head,
        stride_k_dim,
        g,
        stride_g_token,
        stride_g_head,
        beta,
        stride_beta_token,
        stride_beta_head,
        A,
        stride_A_token,
        stride_A_head,
        stride_A_chunk,
        cu_seqlens,
        chunk_indices,
        num_k_heads,  # pyright: ignore [reportArgumentType]
        k_head_dim,  # pyright: ignore [reportArgumentType]
        num_v_heads,  # pyright: ignore [reportArgumentType]
        BLOCK_K,  # pyright: ignore [reportArgumentType]
        chunk_size,  # pyright: ignore [reportArgumentType]
    )

    return A
