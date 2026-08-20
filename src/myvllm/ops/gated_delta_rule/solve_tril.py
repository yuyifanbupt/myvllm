import torch
import triton
import triton.language as tl


@triton.jit
def _merge_16x16_to_32x_32_inverse_kernel(
    A_ptr,  # (total_tokens, num_v_heads, chunk_size)
    stride_A_token,
    stride_A_head,
    stride_A_chunk,
    Ai_ptr,  # (total_tokens, num_v_heads, chunk_size)
    stride_Ai_token,
    stride_Ai_head,
    stride_Ai_chunk,
    cu_seqlens_ptr,  # (seq_cnt+1, )
    chunk_indices_ptr,  # (chunk_cnt, 2)
    CHUNK_SIZE: tl.constexpr,
):
    chunk_idx = tl.program_id(0)
    head_idx = tl.program_id(1)

    seq_idx = tl.load(chunk_indices_ptr + chunk_idx * 2)
    chunk_offset = tl.load(chunk_indices_ptr + chunk_idx * 2 + 1)
    seq_token_start = tl.load(cu_seqlens_ptr + seq_idx)
    seq_token_end = tl.load(cu_seqlens_ptr + seq_idx + 1)
    seqlen = seq_token_end - seq_token_start

    A_block_ptr = tl.make_block_ptr(
        A_ptr + seq_token_start * stride_A_token + head_idx * stride_A_head,
        shape=(seqlen, CHUNK_SIZE),
        strides=(stride_A_token, stride_A_chunk),
        offsets=(chunk_offset * CHUNK_SIZE, 0),
        block_shape=(16, 16),
        order=(1, 0),
    )

    A_11 = tl.load(A_block_ptr, boundary_check=(0,), padding_option="zero").to(tl.float32)
    A_block_ptr = tl.advance(A_block_ptr, (1, 0))
    A_21 = tl.load(A_block_ptr, boundary_check=(0,), padding_option="zero").to(tl.float32)
    A_block_ptr = tl.advance(A_block_ptr, (0, 1))
    A_22 = tl.load(A_block_ptr, boundary_check=(0,), padding_option="zero").to(tl.float32)
    Ai_11 = -A_11
    Ai_22 = -A_22
    l = tl.arange(0, 16)

    for i in range(2, min(16, seqlen - chunk_offset * CHUNK_SIZE)):
        a_11_i = -A_11[i]
        a_11_i += tl.sum(a_11_i[:, None] * Ai_11, axis=0)
        Ai_11 = tl.where((l == i)[:, None], a_11_i, Ai_11)
    for i in range(2, min(32, seqlen - chunk_offset * CHUNK_SIZE) - 16):
        a_22_i = -A_22[i]
        a_22_i += tl.sum(a_22_i[:, None] * Ai_22, axis=0)
        Ai_22 = tl.where((l == i)[:, None], a_22_i, Ai_22)

    I = l[:, None] == l[None, :]
    Ai_11 += I
    Ai_22 += I
    Ai_21 = -tl.dot(tl.dot(Ai_22, A_21), Ai_11)

    Ai_block_ptr = tl.make_block_ptr(
        Ai_ptr + seq_token_start * stride_Ai_token + head_idx * stride_Ai_head,
        shape=(seqlen, CHUNK_SIZE),
        strides=(stride_Ai_token, stride_Ai_chunk),
        offsets=(chunk_offset * CHUNK_SIZE, 0),
        block_shape=(16, 16),
        order=(1, 0),
    )
    tl.store(Ai_block_ptr, Ai_11, boundary_check=(0,))
    Ai_block_ptr = tl.advance(Ai_block_ptr, (1, 0))
    tl.store(Ai_block_ptr, Ai_21, boundary_check=(0,))
    Ai_block_ptr = tl.advance(Ai_block_ptr, (0, 1))
    tl.store(Ai_block_ptr, Ai_22, boundary_check=(0,))


def solve_tril(
    A: torch.Tensor,  # (total_tokens, num_v_heads, chunk_size)
    cu_seqlens: torch.Tensor,  # (seq_cnt+1, )
    chunk_indices: torch.Tensor,  # (chunk_cnt, 2)
) -> torch.Tensor:
    """
    Compute the inverse of the matrix I + A
    A should be strictly lower triangular, i.e., A.triu() == 0.

    Args:
        A (torch.Tensor):
            [B, T, H, BT], where BT should only be 16, 32, or 64.
        cu_seqlens (torch.Tensor):
            The cumulative sequence lengths of the input tensor. Default: `None`.
        output_dtype (torch.dtype):
            The dtype of the output tensor. Default: `torch.float`.
            If `None`, the output dtype will be the same as the input dtype.

    Returns:
        (I + A)^-1 with the same shape as A
    """
    _, num_v_heads, chunk_size = A.shape
    assert chunk_size == 32  # only support chunk_size = 32

    stride_A_token, stride_A_head, stride_A_chunk = A.stride()
    Ai = torch.zeros_like(A, dtype=torch.float32)
    stride_Ai_token, stride_Ai_head, stride_Ai_chunk = Ai.stride()
    chunk_cnt = chunk_indices.size(0)
    grid = (chunk_cnt, num_v_heads)

    _merge_16x16_to_32x_32_inverse_kernel[grid](
        A,
        stride_A_token,
        stride_A_head,
        stride_A_chunk,
        Ai,
        stride_Ai_token,
        stride_Ai_head,
        stride_Ai_chunk,
        cu_seqlens,
        chunk_indices,
        chunk_size,  # pyright: ignore[reportArgumentType]
    )

    return Ai
