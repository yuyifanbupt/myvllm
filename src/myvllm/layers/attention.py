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
