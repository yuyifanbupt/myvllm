import torch
import triton
import triton.language as tl


# TODO: 明天再看看 g_interleaved 和 g 有什么区别
@triton.jit()
def _chunk_local_cumsum_kernel(
    g_ptr,  # (total_tokens, num_v_heads),
    stride_g_token,
    stride_g_head,
    o_ptr,  # (total_tokens, num_v_heads),
    stride_o_token,
    stride_o_head,
    cu_seq_lens_ptr,  # (seq_cnt+1, )
    chunk_indices_ptr,  # (chunk_cnt, 2)
    scale: float,
    num_v_heads: tl.constexpr,
    CHUNK_SIZE: tl.constexpr,
):
    chunk_idx = tl.program_id(0)
    head_idx = tl.program_id(1)

    seq_idx = tl.load(chunk_indices_ptr + chunk_idx * 2)
    token_offset_start = tl.load(cu_seq_lens_ptr + seq_idx)
    token_offset_end = tl.load(cu_seq_lens_ptr + seq_idx + 1)
    seqlen = token_offset_end - token_offset_start
    chunk_offset = tl.load(chunk_indices_ptr + chunk_idx * 2 + 1)

    g_block_ptr = tl.make_block_ptr(
        g_ptr + token_offset_start * stride_g_token,
        shape=(seqlen, num_v_heads),
        strides=(stride_g_token, stride_g_head),
        offsets=(chunk_offset * CHUNK_SIZE, head_idx),
        block_shape=(CHUNK_SIZE, 1),
        order=(1, 0),
    )
    g_block = tl.load(g_block_ptr, boundary_check=(0,), padding_option="zero").to(tl.float32)

    o_block = tl.cumsum(g_block, axis=0) * scale
    o_block_ptr = tl.make_block_ptr(
        o_ptr + token_offset_start * stride_o_token,
        shape=(seqlen, num_v_heads),
        strides=(stride_o_token, stride_o_head),
        offsets=(chunk_offset * CHUNK_SIZE, head_idx),
        block_shape=(CHUNK_SIZE, 1),
        order=(1, 0),
    )
    tl.store(o_block_ptr, o_block, boundary_check=(0,))


def chunk_local_cumsum(
    g: torch.Tensor,  # (total_tokens, num_v_heads)
    scale: float,
    cu_seqlens: torch.Tensor,  # (seq_cnt+1, )
    chunk_indices: torch.Tensor,  # (chunk_cnt, 2)
    chunk_size: int = 32,
) -> torch.Tensor:
    stride_g_token, stride_g_head = g.stride()
    o = torch.empty_like(g, dtype=torch.float32)
    stride_o_token, stride_o_head = o.stride()
    chunk_cnt = chunk_indices.size(0)
    num_v_heads = g.size(1)
    grid = (chunk_cnt, num_v_heads)

    _chunk_local_cumsum_kernel[grid](
        g,
        stride_g_token,
        stride_g_head,
        o,
        stride_o_token,
        stride_o_head,
        cu_seqlens,
        chunk_indices,
        scale,
        num_v_heads,  # pyright: ignore[reportArgumentType]
        chunk_size,  # pyright: ignore[reportArgumentType]
    )

    return o
