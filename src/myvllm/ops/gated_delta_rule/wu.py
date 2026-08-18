import torch
import triton
import triton.language as tl


@triton.jit
def _gdn_compute_w_u_kernel(
    k_ptr,  # (total_tokens, num_k_heads, k_head_dim)
    stride_k_token,
    stride_k_head,
    stride_k_dim,
    v_ptr,  # (total_tokens, num_v_heads, v_head_dim)
    stride_v_token,
    stride_v_head,
    stride_v_dim,
    beta_ptr,  # (total_tokens, num_v_heads)
    stride_beta_token,
    stride_beta_head,
    w_ptr,  # (total_tokens, num_v_heads, k_head_dim)
    stride_w_token,
    stride_w_head,
    stride_w_dim,
    u_ptr,  # (total_tokens, num_v_heads, v_head_dim)
    stride_u_token,
    stride_u_head,
    stride_u_dim,
    A_ptr,  # (total_tokens, num_v_heads, chunk_size)
    stride_A_token,
    stride_A_head,
    stride_A_chunk,
    g_ptr,  # (total_tokens, num_v_heads)
    stride_g_token,
    stride_g_head,
    cu_seqlens_ptr,  # (seq_cnt+1, )
    chunk_indices_ptr,  # (chunk_cnt, 2)
    num_k_heads: tl.constexpr,
    k_head_dim: tl.constexpr,
    num_v_heads: tl.constexpr,
    v_head_dim: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_V: tl.constexpr,
    CHUNK_SIZE: tl.constexpr,
):
    chunk_idx = tl.program_id(0)
    seq_idx = tl.load(chunk_indices_ptr + chunk_idx * 2)
    chunk_offset = tl.load(chunk_indices_ptr + chunk_idx * 2 + 1)
    seq_token_start = tl.load(cu_seqlens_ptr + seq_idx)
    seq_token_end = tl.load(cu_seqlens_ptr + seq_idx + 1)
    seqlen = seq_token_end - seq_token_start
    v_head_idx = tl.program_id(1)
    k_head_idx = v_head_idx // (num_v_heads // num_k_heads)

    A_block_ptr = tl.make_block_ptr(
        A_ptr + seq_token_start * stride_A_token + v_head_idx * stride_A_head,
        shape=(seqlen, CHUNK_SIZE),
        strides=(stride_A_token, stride_A_chunk),
        offsets=(chunk_offset * CHUNK_SIZE, 0),
        block_shape=(CHUNK_SIZE, CHUNK_SIZE),
        order=(1, 0),
    )
    A_block = tl.load(A_block_ptr, boundary_check=(0,), padding_option="zero")

    beta_block_ptr = tl.make_block_ptr(
        beta_ptr + seq_token_start * stride_beta_token,
        shape=(seqlen, num_v_heads),
        strides=(stride_beta_token, stride_beta_head),
        offsets=(chunk_offset * CHUNK_SIZE, v_head_idx),
        block_shape=(CHUNK_SIZE, 1),
        order=(1, 0),
    )
    beta_block = tl.load(beta_block_ptr, boundary_check=(0,), padding_option="zero")

    v_block_ptr = tl.make_block_ptr(
        v_ptr + seq_token_start * stride_v_token + v_head_idx * stride_v_head,
        shape=(seqlen, v_head_dim),
        strides=(stride_v_token, stride_v_dim),
        offsets=(chunk_offset * CHUNK_SIZE, 0),
        block_shape=(CHUNK_SIZE, BLOCK_V),
        order=(1, 0),
    )
    u_block_ptr = tl.make_block_ptr(
        u_ptr + seq_token_start * stride_v_token + v_head_idx * stride_u_head,
        shape=(seqlen, v_head_dim),
        strides=(stride_u_token, stride_u_dim),
        offsets=(chunk_offset * CHUNK_SIZE, 0),
        block_shape=(CHUNK_SIZE, BLOCK_V),
        order=(1, 0),
    )

    for i in range(tl.cdiv(v_head_dim, BLOCK_V)):
        v_block = tl.load(v_block_ptr, boundary_check=(0, 1), padding_option="zero")
        v_block_ptr = tl.advance(v_block_ptr, (0, 1))

        beta_v_block = (v_block * beta_block).to(v_block.dtype)
        u_block = tl.dot(A_block, beta_v_block)
        tl.store(u_block_ptr, u_block, boundary_check=(0, 1))
        u_block_ptr = tl.advance(u_block_ptr, (0, 1))

    g_block_ptr = tl.make_block_ptr(
        g_ptr + seq_token_start * stride_g_token,
        shape=(seqlen, num_v_heads),
        strides=(stride_g_token, stride_g_head),
        offsets=(chunk_offset * CHUNK_SIZE, v_head_idx),
        block_shape=(CHUNK_SIZE, 1),
        order=(1, 0),
    )
    g_block = tl.math.exp2(tl.load(g_block_ptr, boundary_check=(0,), padding_option="zero"))

    k_block_ptr = tl.make_block_ptr(
        k_ptr + seq_token_start * stride_k_token + k_head_idx * stride_k_head,
        shape=(seqlen, k_head_dim),
        strides=(stride_k_token, stride_k_dim),
        offsets=(chunk_offset * CHUNK_SIZE, 0),
        block_shape=(CHUNK_SIZE, BLOCK_K),
        order=(1, 0),
    )
    w_block_ptr = tl.make_block_ptr(
        w_ptr + seq_token_start * stride_w_token + v_head_idx * stride_w_head,
        shape=(seqlen, k_head_dim),
        strides=(stride_w_token, stride_w_dim),
        offsets=(chunk_offset * CHUNK_SIZE, 0),
        block_shape=(CHUNK_SIZE, BLOCK_K),
        order=(1, 0),
    )

    for i in range(tl.cdiv(k_head_dim, BLOCK_K)):
        k_block = tl.load(k_block_ptr, boundary_check=(0, 1), padding_option="zero")
        k_block_ptr = tl.advance(k_block_ptr, (0, 1))

        beta_k_block = (k_block * beta_block * g_block).to(k_block.dtype)
        w_block = tl.dot(A_block, beta_k_block)
        tl.store(w_block_ptr, w_block, boundary_check=(0, 1))
        w_block_ptr = tl.advance(w_block_ptr, (0, 1))


def gdn_compute_w_u(
    k: torch.Tensor,  # (total_tokens, num_k_heads, k_head_dim)
    v: torch.Tensor,  # (total_tokens, num_v_heads, v_head_dim)
    beta: torch.Tensor,  # (total_tokens, num_v_heads)
    A: torch.Tensor,  # (total_tokens, num_v_heads, chunk_size)
    g: torch.Tensor,  # (total_tokens, num_v_heads)
    cu_seqlens: torch.Tensor,  # (seq_cnt + 1, )
    chunk_indices: torch.Tensor,  # (chunk_cnt, 2)
) -> tuple[torch.Tensor, torch.Tensor]:
    total_tokens, num_k_heads, k_head_dim = k.shape
    _, num_v_heads, v_head_dim = v.shape
    chunk_size = A.shape[-1]
    chunk_cnt = chunk_indices.size(0)
    BLOCK_K, BLOCK_V = 32, 32  # TODO: 研究下这个值怎么设
    w = torch.zeros(total_tokens, num_v_heads, k_head_dim, dtype=k.dtype, device=k.device)
    u = torch.zeros_like(v)
    grid = (chunk_cnt, num_v_heads)
    stride_k_token, stride_k_head, stride_k_dim = k.stride()
    stride_v_token, stride_v_head, stride_v_dim = v.stride()
    stride_w_token, stride_w_head, stride_w_dim = w.stride()
    stride_u_token, stride_u_head, stride_u_dim = u.stride()
    stride_beta_token, stride_beta_head = beta.stride()
    stride_A_token, stride_A_head, stride_A_chunk = A.stride()
    stride_g_token, stride_g_head = g.stride()

    _gdn_compute_w_u_kernel[grid](
        k,
        stride_k_token,
        stride_k_head,
        stride_k_dim,
        v,
        stride_v_token,
        stride_v_head,
        stride_v_dim,
        beta,
        stride_beta_token,
        stride_beta_head,
        w,
        stride_w_token,
        stride_w_head,
        stride_w_dim,
        u,
        stride_u_token,
        stride_u_head,
        stride_u_dim,
        A,
        stride_A_token,
        stride_A_head,
        stride_A_chunk,
        g,
        stride_g_token,
        stride_g_head,
        cu_seqlens,
        chunk_indices,
        num_k_heads,  # pyright: ignore[reportArgumentType]
        k_head_dim,  # pyright: ignore[reportArgumentType]
        num_v_heads,  # pyright: ignore[reportArgumentType]
        v_head_dim,  # pyright: ignore[reportArgumentType]
        BLOCK_K,  # pyright: ignore[reportArgumentType]
        BLOCK_V,  # pyright: ignore[reportArgumentType]
        chunk_size,  # pyright: ignore[reportArgumentType]
    )

    return w, u
