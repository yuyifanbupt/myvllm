import torch
import triton
import triton.language as tl


@triton.jit
def flash_attention_varlen_kernel(
    Q_ptr,
    stride_ql,
    stride_qh,
    stride_qd,
    K_ptr,
    stride_kl,
    stride_kh,
    stride_kd,
    V_ptr,
    stride_vl,
    stride_vh,
    stride_vd,
    O_ptr,
    stride_ol,
    stride_oh,
    stride_od,
    cu_seqlens_ptr,
    scale,
    is_causal: bool,
    num_heads: tl.constexpr,
    num_kv_heads: tl.constexpr,
    head_dim: tl.constexpr,
    head_v_dim: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    q_block_idx = tl.program_id(0)
    head_idx = tl.program_id(1)
    seq_idx = tl.program_id(2)
    kv_head_idx = head_idx // (num_heads // num_kv_heads)
    seq_start = tl.load(cu_seqlens_ptr + seq_idx)
    seq_end = tl.load(cu_seqlens_ptr + seq_idx + 1)
    seq_len = seq_end - seq_start

    if q_block_idx * BLOCK_M >= seq_len:
        return

    Q_block_ptr = tl.make_block_ptr(
        Q_ptr + seq_start * stride_ql + head_idx * stride_qh,
        shape=(seq_len, head_dim),
        strides=(stride_ql, stride_qd),
        offsets=(q_block_idx * BLOCK_M, 0),
        block_shape=(BLOCK_M, head_dim),
        order=(1, 0),
    )
    q = tl.load(Q_block_ptr, boundary_check=(0,), padding_option="zero")

    K_block_ptr = tl.make_block_ptr(
        K_ptr + kv_head_idx * stride_kh,
        shape=(seq_len, head_dim),
        strides=(stride_kl, stride_kd),
        offsets=(0, 0),
        block_shape=(BLOCK_N, head_dim),
        order=(1, 0),
    )
    V_block_ptr = tl.make_block_ptr(
        V_ptr + kv_head_idx * stride_vh,
        shape=(seq_len, head_v_dim),
        strides=(stride_vl, stride_vd),
        offsets=(0, 0),
        block_shape=(BLOCK_N, head_v_dim),
        order=(1, 0),
    )
    O_block_ptr = tl.make_block_ptr(
        O_ptr + seq_start * stride_ol + head_idx * stride_oh,
        shape=(seq_len, head_v_dim),
        strides=(stride_ol, stride_oh),
        offsets=(q_block_idx * BLOCK_M, 0),
        block_shape=(BLOCK_M, head_v_dim),
        order=(1, 0),
    )

    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - 1e10
    acc = tl.zeros([BLOCK_M, head_v_dim], dtype=tl.float32)

    num_blocks = tl.cdiv(seq_len, BLOCK_N)

    for block_n in range(num_blocks):
        mask_n = block_n * BLOCK_N + tl.arange(0, BLOCK_N) < seq_len

        v = tl.load(V_block_ptr, boundary_check=(0,), padding_option="zero")
        k = tl.load(K_block_ptr, boundary_check=(0,), padding_option="zero")
        qk = tl.dot(q, tl.trans(k)) * scale

        if is_causal:
            q_pos_ids = q_block_idx * BLOCK_M + tl.arange(0, BLOCK_M)
            k_pos_ids = block_n * BLOCK_N + tl.arange(0, BLOCK_N)
            mask_causal = q_pos_ids[:, None] >= k_pos_ids[None, :]
            qk = tl.where(mask_causal & mask_n[None, :], qk, -1e10)

        m_ij = tl.max(qk, axis=1)
        m_i_new = tl.maxmum(m_i, m_ij)
        alpha = tl.exp(m_i - m_i_new)
        p = tl.exp(qk - m_i_new[:, None])
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
        l_i = l_i * alpha + tl.sum(p, axis=1)
        m_i = m_i_new

        V_block_ptr = tl.advance(V_block_ptr, (BLOCK_N, 0))
        K_block_ptr = tl.advance(K_block_ptr, (BLOCK_N, 0))

    acc = acc / l_i[:, None]
    tl.store(O_block_ptr, acc.to(O_ptr.dtype.element_ty), boundary_check=(0,))


def flash_attention_prefill(
    q: torch.Tensor,  # shape: (total_seq_len, num_heads, head_dim)
    k: torch.Tensor,  # shape: (total_seq_len, num_kv_heads, head_dim)
    v: torch.Tensor,  # shape: (total_seq_len, num_kv_heads, head_v_dim)
    cu_seqlens: torch.Tensor,  # shape: (num_seq+1)
    scale: float,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    head_v_dim: int,
    is_causal: bool,
) -> torch.Tensor:
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    o = torch.empty(q.shape[0], q.shape[1], head_v_dim)

    if head_dim <= 64:
        BLOCK_M = 64
        BLOCK_N = 64
    elif head_dim <= 128:
        BLOCK_M = 32
        BLOCK_N = 32
    else:
        BLOCK_M = 16
        BLOCK_N = 16

    num_seqs = cu_seqlens.shape[0] - 1
    max_seq_len = (cu_seqlens[1:] - cu_seqlens[:-1]).max().item()
    grid = (tl.cdiv(max_seq_len, BLOCK_M), num_heads, num_seqs)

    flash_attention_varlen_kernel[grid](
        q,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        k,
        k.stride(0),
        k.stride(1),
        k.stride(2),
        v,
        v.stride(0),
        v.stride(1),
        v.stride(2),
        o,
        o.stride(0),
        o.stride(1),
        o.stride(2),
        cu_seqlens,
        scale,
        is_causal,
        num_heads,
        num_kv_heads,
        head_dim,
        head_v_dim,
        BLOCK_M,
        BLOCK_N,
    )

    return o


@triton.jit
def _paged_flash_attention_decode_kernel(
    q_ptr,  # (seq_cnt, num_q_heads, qk_head_dim)
    stride_q_seq,
    stride_q_head,
    stride_q_dim,
    o_ptr,  # (seq_cnt, num_q_heads, v_head_dim)
    stride_o_seq,
    stride_o_head,
    stride_o_dim,
    k_cache_ptr,  # (max_num_cached_blocks, block_size, num_kv_heads, qk_head_dim)
    stride_k_cache_block,
    stride_k_cache_token,
    stride_k_cache_head,
    stride_k_cache_dim,
    v_cache_ptr,  # (max_num_cached_blocks, block_size, num_kv_heads, v_head_dim)
    stride_v_cache_block,
    stride_v_cache_token,
    stride_v_cache_head,
    stride_v_cache_dim,
    block_tables_ptr,  # (seq_cnt, max_num_blocks)
    stride_table_seq,
    stride_table_block,
    context_lens_ptr,  # (seq_cnt, )
    scale: float,
    qk_head_dim: tl.constexpr,
    v_head_dim: tl.constexpr,
    num_q_heads: tl.constexpr,
    num_kv_heads: tl.constexpr,
    block_size: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    seq_idx = tl.program_id(0)
    q_head_idx = tl.program_id(1)

    kv_head_idx = q_head_idx // (num_q_heads // num_kv_heads)
    context_len = tl.load(context_lens_ptr + seq_idx)

    q_block_ptr = tl.make_block_ptr(
        q_ptr + seq_idx * stride_q_seq + q_head_idx * stride_q_head,
        shape=(qk_head_dim,),
        strides=(stride_q_dim,),
        offsets=(0,),
        block_shape=(qk_head_dim,),
        order=(0,),
    )
    q = tl.load(q_block_ptr)  # (qk_head_dim, )

    acc = tl.zeros([v_head_dim], dtype=tl.float32)
    l_i = 0.0
    m_i = -1e10

    for block_n_idx in tl.range(0, tl.cdiv(context_len, BLOCK_N)):
        token_start = block_n_idx * BLOCK_N

        blocks_ptr = (
            block_tables_ptr
            + seq_idx * stride_table_seq
            + ((token_start + tl.arange(0, BLOCK_N)) // block_size) * stride_table_block
        )
        offsets_in_block = (token_start + tl.arange(0, BLOCK_N)) % block_size
        token_mask = token_start + tl.arange(0, BLOCK_N) < context_len
        block_indices = tl.load(blocks_ptr, mask=token_mask, other=0)  # (BLOCK_N, )
        k_ptr = (
            k_cache_ptr
            + block_indices[None, :] * stride_k_cache_block
            + offsets_in_block[None, :] * stride_k_cache_token
            + kv_head_idx * stride_k_cache_head
            + tl.arange(0, qk_head_dim)[:, None] * stride_k_cache_dim
        )
        k = tl.load(k_ptr, mask=token_mask[None, :], other=0.0).to(tl.float32)  # (qk_head_dim, BLOCK_N)
        v_ptr = (
            v_cache_ptr
            + block_indices[:, None] * stride_v_cache_block
            + offsets_in_block[:, None] * stride_v_cache_token
            + kv_head_idx * stride_v_cache_head
            + tl.arange(0, v_head_dim)[None, :] * stride_v_cache_dim
        )
        v = tl.load(v_ptr, mask=token_mask[:, None], other=0.0).to(tl.float32)  # (BLOCK_N, v_head_dim)

        qk = tl.sum(q[:, None] * k, axis=0) * scale  # (BLOCK_N, )
        qk = tl.where(token_mask, qk, -1e10)
        m_ij = tl.max(qk)
        m_i_new = tl.maximum(m_i, m_ij)
        alpha = tl.exp(m_i - m_i_new)
        p = tl.exp(qk - m_i_new)

        acc *= alpha
        l_i *= alpha
        acc += tl.sum(p[:, None] * v, axis=0)
        l_i += tl.sum(p)

        m_i = m_i_new

    o = acc / l_i
    o_block_ptr = tl.make_block_ptr(
        o_ptr + seq_idx * stride_o_seq + q_head_idx * stride_o_head,
        shape=(v_head_dim,),
        strides=(stride_o_dim,),
        offsets=(0,),
        block_shape=(v_head_dim,),
        order=(0,),
    )
    tl.store(o_block_ptr, o)


def paged_flash_attention_decode(
    q: torch.Tensor,  # (seq_cnt, num_q_heads, qk_head_dim)
    k_cache: torch.Tensor,  # (max_num_cached_blocks, block_size, num_kv_heads, qk_head_dim)
    v_cache: torch.Tensor,  # (max_num_cached_blocks, block_size, num_kv_heads, v_head_dim)
    block_tables: torch.Tensor,  # (seq_cnt, max_num_blocks)
    context_lens: torch.Tensor,  # (seq_cnt, )
    scale: float,
    block_size: int,
) -> torch.Tensor:  # (seq_cnt, num_q_heads, v_head_dim)
    seq_cnt, num_q_heads, qk_head_dim = q.shape
    num_kv_heads, v_head_dim = v_cache.shape[-2:]
    o = torch.zeros(seq_cnt, num_q_heads, v_head_dim, dtype=q.dtype, device=q.device)
    stride_q_seq, stride_q_head, stride_q_dim = q.stride()
    stride_o_seq, stride_o_head, stride_o_dim = o.stride()
    stride_k_cache_block, stride_k_cache_token, stride_k_cache_head, stride_k_cache_dim = k_cache.stride()
    stride_v_cache_block, stride_v_cache_token, stride_v_cache_head, stride_v_cache_dim = v_cache.stride()
    stride_table_seq, stride_table_block = block_tables.stride()
    BLOCK_N = 32

    grid = (seq_cnt, num_q_heads)

    _paged_flash_attention_decode_kernel[grid](
        q,
        stride_q_seq,
        stride_q_head,
        stride_q_dim,
        o,
        stride_o_seq,
        stride_o_head,
        stride_o_dim,
        k_cache,
        stride_k_cache_block,
        stride_k_cache_token,
        stride_k_cache_head,
        stride_k_cache_dim,
        v_cache,
        stride_v_cache_block,
        stride_v_cache_token,
        stride_v_cache_head,
        stride_v_cache_dim,
        block_tables,
        stride_table_seq,
        stride_table_block,
        context_lens,
        scale,
        qk_head_dim,  # pyright: ignore[reportArgumentType]
        v_head_dim,  # pyright: ignore[reportArgumentType]
        num_q_heads,  # pyright: ignore[reportArgumentType]
        num_kv_heads,  # pyright: ignore[reportArgumentType]
        block_size,  # pyright: ignore[reportArgumentType]
        BLOCK_N,  # pyright: ignore[reportArgumentType]
    )

    return o


@triton.jit
def _paged_flash_attention_prefill_kernel(
    q_ptr,  # (total_tokens, num_q_heads, qk_head_dim)
    stride_q_token,
    stride_q_head,
    stride_q_dim,
    o_ptr,  # (total_tokens, num_q_heads, v_head_dim)
    stride_o_token,
    stride_o_head,
    stride_o_dim,
    k_cache_ptr,  # (max_num_cached_blocks, block_size, num_kv_heads, qk_head_dim)
    stride_k_cache_block,
    stride_k_cache_token,
    stride_k_cache_head,
    stride_k_cache_dim,
    v_cache_ptr,  # (max_num_cached_blocks, block_size, num_kv_heads, v_head_dim)
    stride_v_cache_block,
    stride_v_cache_token,
    stride_v_cache_head,
    stride_v_cache_dim,
    page_block_tables_ptr,  # (seq_cnt, max_num_blocks)
    stride_table_seq,
    stride_table_block,
    context_lens_ptr,  # (seq_cnt, )
    cu_seqlens_ptr,  # (seq_cnt + 1, )
    block_m_indices_ptr,  # (block_m_cnt, 2)
    scale: float,
    qk_head_dim: tl.constexpr,
    v_head_dim: tl.constexpr,
    num_q_heads: tl.constexpr,
    num_kv_heads: tl.constexpr,
    page_block_size: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    block_m_idx = tl.program_id(0)
    q_head_idx = tl.program_id(1)

    kv_head_idx = q_head_idx // (num_q_heads // num_kv_heads)
    seq_idx = tl.load(block_m_indices_ptr + block_m_idx * 2)
    block_m_offset = tl.load(block_m_indices_ptr * block_m_idx * 2 + 1)
    seq_token_start = tl.load(cu_seqlens_ptr, seq_idx)
    seq_token_end = tl.load(cu_seqlens_ptr, seq_idx + 1)
    seqlen = seq_token_end - seq_token_start
    context_len = tl.load(context_lens_ptr + seq_idx)

    q_block_ptr = tl.make_block_ptr(
        q_ptr + seq_token_start * stride_q_token + q_head_idx * stride_q_head,
        shape=(seqlen, qk_head_dim),
        strides=(stride_q_token, stride_q_dim),
        offsets=(block_m_offset * BLOCK_M, 0),
        block_shape=(BLOCK_M, qk_head_dim),
        order=(1, 0),
    )
    q = tl.load(q_block_ptr, boundary_check=(0,), padding_option="zero")  # (BLOCK_M, qk_head_dim)
    block_m_range = (context_len - seqlen) + block_m_offset * BLOCK_M + tl.arange(0, BLOCK_M)  # (BLOCK_M, )
    acc = tl.zeros([BLOCK_M, v_head_dim], dtype=tl.float32)  # (BLOCK_M, v_head_dim)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)  # (BLOCK_M, )
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - 1e10  # (BLOCK_M)

    block_n_cnt = tl.cdiv(context_len, BLOCK_N)
    for block_n_idx in tl.range(0, block_n_cnt):
        token_start = block_n_idx * BLOCK_N
        block_n_range = token_start + tl.arange(0, BLOCK_N)  # (BLOCK_N, )
        causal_mask = block_m_range[:, None] >= block_n_range[None, :]  # (BLOCK_M, BLOCK_N)
        page_blocks_ptr = (
            page_block_tables_ptr
            + seq_idx * stride_table_seq
            + ((token_start + tl.arange(0, BLOCK_N)) // page_block_size) * stride_table_block
        )
        offsets_in_page_block = (token_start + tl.arange(0, BLOCK_N)) % page_block_size
        token_mask = token_start + tl.arange(0, BLOCK_N) < context_len
        page_block_indices = tl.load(page_blocks_ptr, mask=token_mask, other=0)  # (BLOCK_N, )

        k_ptr = (
            k_cache_ptr
            + page_block_indices[None, :] * stride_k_cache_block
            + offsets_in_page_block[None, :] * stride_k_cache_token
            + kv_head_idx * stride_k_cache_head
            + tl.arange(0, qk_head_dim)[:, None] * stride_k_cache_dim
        )
        k = tl.load(k_ptr, mask=token_mask[None, :], other=0.0)  # (qk_head_dim, BLOCK_N)
        v_ptr = (
            v_cache_ptr
            + page_block_indices[:, None] * stride_v_cache_block
            + offsets_in_page_block[:, None] * stride_v_cache_token
            + kv_head_idx * stride_v_cache_head
            + tl.arange(0, v_head_dim)[None, :] * stride_v_cache_dim
        )
        v = tl.load(v_ptr, mask=token_mask[:, None], other=0.0)  # (BLOCK_N, v_head_dim)

        qk = tl.dot(q, k) * scale  # (BLOCK_M, BLOCK_N)
        qk = tl.where(causal_mask & token_mask[None, :], qk, -1e10)  # (BLOCK_M, BLOCK_N)
        m_ij = tl.max(qk, axis=1)  # (BLOCK_M, )
        m_i_new = tl.maximum(m_i, m_ij)  # (BLOCK_M, )
        alpha = tl.exp(m_i - m_i_new)  # (BLOCK_M, )
        p = tl.exp(qk - m_i_new[:, None])  # (BLOCK_M, BLOCK_N)

        acc *= alpha[:, None]  # (BLOCK_M, v_head_dim)
        l_i *= alpha  # (BLOCK_M, )
        acc += tl.dot(p.to(v.dtype), v)  # (BLOCK_M, v_head_dim)
        l_i += tl.sum(p, axis=1)  # (BLOCK_M, )
        m_i = m_i_new  # (BLOCK_M, )

    o = acc / l_i[:, None]  # (BLOCK_M, v_head_dim)
    o_block_ptr = tl.make_block_ptr(
        o_ptr + seq_token_start * stride_o_token + q_head_idx * stride_o_head,
        shape=(seqlen, v_head_dim),
        strides=(stride_o_token, stride_o_dim),
        offsets=(block_m_offset * BLOCK_M, 0),
        block_shape=(BLOCK_M, v_head_dim),
        order=(1, 0),
    )
    tl.store(o_block_ptr, o, boundary_check=(0,))


def paged_flash_attention_prefill(
    q: torch.Tensor,  # (total_tokens, num_q_heads, qk_head_dim)
    k_cache: torch.Tensor,  # (max_num_cached_blocks, block_size, num_kv_heads, qk_head_dim)
    v_cache: torch.Tensor,  # (max_num_cached_blocks, block_size, num_kv_heads, v_head_dim)
    page_block_tables: torch.Tensor,  # (seq_cnt, max_num_blocks)
    context_lens: torch.Tensor,  # (seq_cnt, )
    cu_seqlens: torch.Tensor,  # (seq_cnt + 1, )
    block_m_indices: torch.Tensor,  # (block_m_cnt, 2) TODO: 想下这里怎么避免重复计算这个参数
    scale: float,
    page_block_size: int,
) -> torch.Tensor:
    total_tokens, num_q_heads, qk_head_dim = q.shape
    num_kv_heads, v_head_dim = v_cache.shape[-2:]
    o = torch.zeros(total_tokens, num_q_heads, v_head_dim, dtype=q.dtype, device=q.device)
    block_m_cnt = block_m_indices.size(0)
    BLOCK_M, BLOCK_N = 16, 16
    stride_q_token, stride_q_head, stride_q_dim = q.stride()
    stride_o_token, stride_o_head, stride_o_dim = o.stride()
    stride_k_cache_block, stride_k_cache_token, stride_k_cache_head, stride_k_cache_dim = k_cache.stride()
    stride_v_cache_block, stride_v_cache_token, stride_v_cache_head, stride_v_cache_dim = v_cache.stride()
    stride_table_seq, stride_table_block = page_block_tables.stride()
    grid = (block_m_cnt, num_q_heads)

    _paged_flash_attention_prefill_kernel[grid](
        q,
        stride_q_token,
        stride_q_head,
        stride_q_dim,
        o,
        stride_o_token,
        stride_o_head,
        stride_o_dim,
        k_cache,
        stride_k_cache_block,
        stride_k_cache_token,
        stride_k_cache_head,
        stride_k_cache_dim,
        v_cache,
        stride_v_cache_block,
        stride_v_cache_token,
        stride_v_cache_head,
        stride_v_cache_dim,
        page_block_tables,
        stride_table_seq,
        stride_table_block,
        context_lens,
        cu_seqlens,
        block_m_indices,
        scale,
        qk_head_dim,  # pyright: ignore[reportArgumentType]
        v_head_dim,  # pyright: ignore[reportArgumentType]
        num_q_heads,  # pyright: ignore[reportArgumentType]
        num_kv_heads,  # pyright: ignore[reportArgumentType]
        page_block_size,  # pyright: ignore[reportArgumentType]
        BLOCK_M,  # pyright: ignore[reportArgumentType]
        BLOCK_N,  # pyright: ignore[reportArgumentType]
    )

    return o
