import torch
import triton
import triton.language as tl


@triton.jit()
def _causal_conv1d_fwd_kernel(
    x_ptr,  # (total_tokens, dim)
    stride_x_token,
    stride_x_dim,
    block_m_offsets_ptr,
    seq_indices_ptr,
    w_ptr,  # (dim, kernel_size)
    stride_w_dim,
    stride_w_width,
    conv_state_ptr,  # (max_seq_cnt, kernel_size-1, dim)
    stride_conv_state_seq,
    stride_conv_state_token,
    stride_conv_state_dim,
    conv_state_indices_ptr,  # (seq_cnt,)
    cu_seqlens_ptr,  # (seq_cnt+1,)
    o_ptr,  # (total_tokens, dim)
    stride_o_token,
    stride_o_dim,
    dim,
    KERNEL_SIZE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    block_m_idx = tl.program_id(0)
    block_n_idx = tl.program_id(1)

    block_offset = tl.load(block_m_offsets_ptr, block_m_idx)
    seq_idx = tl.load(seq_indices_ptr + block_m_idx)
    token_offset_start = tl.load(cu_seqlens_ptr + seq_idx)
    token_offset_end = tl.load(cu_seqlens_ptr + seq_idx + 1)
    seqlen = token_offset_end - token_offset_start

    if block_offset == 0:
        conv_state_idx = tl.load(conv_state_indices_ptr + seq_idx)
        conv_state_block_ptr = tl.make_block_ptr(
            conv_state_ptr + conv_state_idx * stride_conv_state_seq,
            shape=(KERNEL_SIZE - 1, dim),
            strides=(stride_conv_state_token, stride_conv_state_dim),
            offsets=(0, block_n_idx * BLOCK_N),
            block_shape=(1, BLOCK_N),
            order=(1, 0),
        )

        if KERNEL_SIZE >= 2:
            col0 = tl.load(conv_state_block_ptr, boundary_check=(1,), padding_option="zero")
        if KERNEL_SIZE >= 3:
            tl.advance(conv_state_block_ptr, (1, 0))
            col1 = tl.load(conv_state_block_ptr, boundary_check=(1,), padding_option="zero")
        if KERNEL_SIZE >= 4:
            tl.advance(conv_state_block_ptr, (1, 0))
            col2 = tl.load(conv_state_block_ptr, boundary_check=(1,), padding_option="zero")
        if KERNEL_SIZE >= 5:
            tl.advance(conv_state_block_ptr, (1, 0))
            col3 = tl.load(conv_state_block_ptr, boundary_check=(1,), padding_option="zero")
        if KERNEL_SIZE >= 6:
            tl.advance(conv_state_block_ptr, (1, 0))
            col4 = tl.load(conv_state_block_ptr, boundary_check=(1,), padding_option="zero")
    else:
        conv_state_block_ptr = tl.make_block_ptr(
            x_ptr + (token_offset_start) * stride_x_token,
            shape=(seqlen, dim),
            strides=(stride_x_token, stride_x_dim),
            offsets=(block_offset * BLOCK_M, block_n_idx * BLOCK_N),
            block_shape=(1, BLOCK_N),
            order=(1, 0),
        )

        if KERNEL_SIZE >= 2:
            tl.advance(conv_state_block_ptr, (-1, 0))
            col0 = tl.load(conv_state_block_ptr, boundary_check=(1,), padding_option="zero")
        if KERNEL_SIZE >= 3:
            tl.advance(conv_state_block_ptr, (-1, 0))
            col1 = col0  # pyright: ignore [reportPossiblyUnboundVariable]
            col0 = tl.load(conv_state_block_ptr, boundary_check=(1,), padding_option="zero")
        if KERNEL_SIZE >= 4:
            tl.advance(conv_state_block_ptr, (-1, 0))
            col1, col2 = col0, col1  # pyright: ignore [reportPossiblyUnboundVariable]
            col0 = tl.load(conv_state_block_ptr, boundary_check=(1,), padding_option="zero")
        if KERNEL_SIZE >= 5:
            tl.advance(conv_state_block_ptr, (-1, 0))
            col1, col2, col3 = col0, col1, col2  # pyright: ignore [reportPossiblyUnboundVariable]
            col0 = tl.load(conv_state_block_ptr, boundary_check=(1,), padding_option="zero")
        if KERNEL_SIZE >= 6:
            tl.advance(conv_state_block_ptr, (-1, 0))
            col1, col2, col3, col4 = col0, col1, col2, col3  # pyright: ignore [reportPossiblyUnboundVariable]
            col0 = tl.load(conv_state_block_ptr, boundary_check=(1,), padding_option="zero")

    w_block_ptr = tl.make_block_ptr(
        w_ptr,
        shape=(dim, KERNEL_SIZE),
        strides=(stride_w_dim, stride_w_width),
        offsets=(block_n_idx * BLOCK_N, 0),
        block_shape=(BLOCK_N, 1),
        order=(1, 0),
    )
    w_col0 = tl.trans(tl.load(w_block_ptr, boundary_check=(0,), padding_option="zero"))

    if KERNEL_SIZE >= 2:
        w_block_ptr = tl.advance(w_block_ptr, (0, 1))
        w_col1 = tl.trans(tl.load(w_block_ptr, boundary_check=(0,), padding_option="zero"))
    if KERNEL_SIZE >= 3:
        w_block_ptr = tl.advance(w_block_ptr, (0, 1))
        w_col2 = tl.trans(tl.load(w_block_ptr, boundary_check=(0,), padding_option="zero"))
    if KERNEL_SIZE >= 4:
        w_block_ptr = tl.advance(w_block_ptr, (0, 1))
        w_col3 = tl.trans(tl.load(w_block_ptr, boundary_check=(0,), padding_option="zero"))
    if KERNEL_SIZE >= 5:
        w_block_ptr = tl.advance(w_block_ptr, (0, 1))
        w_col4 = tl.trans(tl.load(w_block_ptr, boundary_check=(0,), padding_option="zero"))
    if KERNEL_SIZE >= 6:
        w_block_ptr = tl.advance(w_block_ptr, (0, 1))
        w_col5 = tl.trans(tl.load(w_block_ptr, boundary_check=(0,), padding_option="zero"))

    x_block_ptr = tl.make_block_ptr(
        x_ptr + token_offset_start * stride_x_token,
        shape=(seqlen, dim),
        strides=(stride_x_token, stride_x_dim),
        offsets=(block_offset * BLOCK_M, block_n_idx * BLOCK_N),
        block_shape=(1, BLOCK_N),
        order=(1, 0),
    )
    o_block_ptr = tl.make_block_ptr(
        o_ptr + token_offset_start * stride_o_token,
        shape=(seqlen, dim),
        strides=(stride_o_token, stride_o_dim),
        offsets=(block_offset * BLOCK_M, block_n_idx * BLOCK_N),
        block_shape=(1, BLOCK_N),
        order=(1, 0),
    )
    acc_preload = tl.zeros((1, BLOCK_N), dtype=tl.float32)

    for token_idx in tl.range(block_offset * BLOCK_M, min(seqlen, (block_offset + 1) * BLOCK_M)):
        acc = acc_preload
        cur_tok = tl.load(x_block_ptr, boundary_check=(1,), padding_option="zero")
        x_block_ptr = tl.advance(x_block_ptr, (1, 0))
        mat_w = w_col0
        mat_x = col0

        for j in tl.static_range(KERNEL_SIZE):
            if KERNEL_SIZE == 2:
                if j == 1:
                    mat_w, mat_x = w_col1, cur_tok
            elif KERNEL_SIZE == 3:
                if j == 1:
                    mat_w, mat_x = w_col1, col1
                elif j == 2:
                    mat_w, mat_x = w_col2, cur_tok
            elif KERNEL_SIZE == 4:
                if j == 1:
                    mat_w, mat_x = w_col1, col1
                elif j == 2:
                    mat_w, mat_x = w_col2, col2
                elif j == 3:
                    mat_w, mat_x = w_col3, cur_tok
            elif KERNEL_SIZE == 5:
                if j == 1:
                    mat_w, mat_x = w_col1, col1
                elif j == 2:
                    mat_w, mat_x = w_col2, col2
                elif j == 3:
                    mat_w, mat_x = w_col3, col3
                elif j == 4:
                    mat_w, mat_x = w_col4, cur_tok
            elif KERNEL_SIZE == 6:
                if j == 1:
                    mat_w, mat_x = w_col1, col1
                elif j == 2:
                    mat_w, mat_x = w_col2, col2
                elif j == 3:
                    mat_w, mat_x = w_col3, col3
                elif j == 4:
                    mat_w, mat_x = w_col4, col4
                elif j == 5:
                    mat_w, mat_x = w_col5, cur_tok

            acc += mat_x * mat_w

        if KERNEL_SIZE == 2:
            col0 = mat_x
        if KERNEL_SIZE == 3:
            col0, col1 = col1, mat_x
        if KERNEL_SIZE == 4:
            col0, col1, col2 = col1, col2, mat_x
        if KERNEL_SIZE == 5:
            col0, col1, col2, col3 = col1, col2, col3, mat_x
        if KERNEL_SIZE == 6:
            col0, col1, col2, col3, col4 = col1, col2, col3, col4, mat_x

        acc = acc / (1 + tl.exp(-acc))
        tl.store(o_block_ptr, acc, boundary_check=(1,))
        o_block_ptr = tl.advance(o_block_ptr, (1, 0))

    if (block_offset + 1) * BLOCK_M >= seqlen:
        conv_state = col0
        if KERNEL_SIZE >= 3:
            conv_state = tl.cat(conv_state, col1, dim=0)
        if KERNEL_SIZE >= 4:
            conv_state = tl.cat(conv_state, col2, dim=0)
        if KERNEL_SIZE >= 5:
            conv_state = tl.cat(conv_state, col3, dim=0)
        if KERNEL_SIZE >= 6:
            conv_state = tl.cat(conv_state, col4, dim=0)

        conv_state_full_block_ptr = tl.make_block_ptr(
            conv_state_ptr + conv_state_idx * stride_conv_state_seq,
            shape=(KERNEL_SIZE - 1, dim),
            strides=(stride_conv_state_token, stride_conv_state_dim),
            offsets=(0, block_n_idx * BLOCK_N),
            block_shape=(KERNEL_SIZE - 1, BLOCK_N),
            order=(1, 0),
        )
        tl.store(conv_state_full_block_ptr, conv_state, boundary_check=(1,))


def causal_conv1d_fn(
    x: torch.Tensor,  # (total_tokens, q_dim + k_dim + v_dim)
    weight: torch.Tensor,  # (q_dim + k_dim + v_dim, kernel_size)
    conv_state: torch.Tensor,  # (max_seq_cnt, kernel_size-1, q_dim + k_dim + v_dim)
    conv_state_indices: torch.Tensor,  # (seq_cnt, )
    cu_seqlens: torch.Tensor,  # (seq_cnt+1, )
) -> torch.Tensor:
    KERNEL_SIZE = weight.size(1)
    BLOCK_M, BLOCK_N = 8, 256
    seqlens = (cu_seqlens[1:] - cu_seqlens[:-1]).tolist()
    block_m_cnts = [tl.cdiv(seqlen, BLOCK_M) for seqlen in seqlens]

    seq_indices_nested = [[i for _ in range(block_m_cnts[i])] for i in range(len(block_m_cnts))]
    seq_indices_flatten = [idx for sublist in seq_indices_nested for idx in sublist]
    seq_indices_tensor = torch.tensor(seq_indices_flatten, device=x.device, dtype=torch.int32)

    seq_block_offsets_nested = [list(range(block_m_cnts[i])) for i in range(len(block_m_cnts))]
    seq_block_offsets_flatten = [x for sublist in seq_block_offsets_nested for x in sublist]
    seq_block_offsets_tensor = torch.tensor(seq_block_offsets_flatten, device=x.device, dtype=torch.int32)

    o = torch.zeros_like(x)
    stride_o_token, stride_o_dim = o.stride()
    stride_x_token, stride_x_dim = x.stride()
    stride_w_dim, stride_w_width = weight.stride()
    stride_conv_state_seq, stride_conv_state_token, stride_conv_state_dim = conv_state.stride()
    dim = x.size(1)
    grid = (sum(block_m_cnts), tl.cdiv(dim, BLOCK_N))

    _causal_conv1d_fwd_kernel[grid](
        x,
        stride_x_token,
        stride_x_dim,
        seq_block_offsets_tensor,
        seq_indices_tensor,
        weight,
        stride_w_dim,
        stride_w_width,
        conv_state,
        stride_conv_state_seq,
        stride_conv_state_token,
        stride_conv_state_dim,
        conv_state_indices,
        cu_seqlens,
        o,
        stride_o_token,
        stride_o_dim,
        dim,
        KERNEL_SIZE,
        BLOCK_M,
        BLOCK_N,
    )

    return o


# for decode
@triton.jit()
def _causal_conv1d_update_kernel(
    x_ptr,  # (seq_cnt, dim)
    stride_x_seq,
    stride_x_dim,
    w_ptr,  # (dim, kernel_size)
    stride_w_dim,
    stride_w_width,
    conv_state_ptr,  # (max_seq_cnt, kernel_size-1, dim)
    stride_conv_state_seq,
    stride_conv_state_token,
    stride_conv_state_dim,
    conv_state_indices_ptr,  # (seq_cnt, )
    o_ptr,  # (seq_cnt, dim)
    stride_o_seq,
    stride_o_dim,
    dim,  # q_dim + k_dim + v_dim
    KERNEL_SIZE: tl.constexpr,
    BLOCK_N: tl.constexpr,
) -> None:
    seq_idx = tl.program_id(0)
    block_n_idx = tl.program_id(1)
    conv_state_idx = tl.load(conv_state_indices_ptr + seq_idx)
    conv_state_block_ptr = tl.make_block_ptr(
        conv_state_ptr + conv_state_idx * stride_conv_state_seq,
        shape=(KERNEL_SIZE - 1, dim),
        strides=(stride_conv_state_token, stride_conv_state_dim),
        offsets=(0, block_n_idx * BLOCK_N),
        block_shape=(1, BLOCK_N),
        order=(1, 0),
    )

    if KERNEL_SIZE >= 2:
        col0 = tl.load(conv_state_block_ptr, boundary_check=(1,), padding_option="zero")
    if KERNEL_SIZE >= 3:
        tl.advance(conv_state_block_ptr, (1, 0))
        col1 = tl.load(conv_state_block_ptr, boundary_check=(1,), padding_option="zero")
    if KERNEL_SIZE >= 4:
        tl.advance(conv_state_block_ptr, (1, 0))
        col2 = tl.load(conv_state_block_ptr, boundary_check=(1,), padding_option="zero")
    if KERNEL_SIZE >= 5:
        tl.advance(conv_state_block_ptr, (1, 0))
        col3 = tl.load(conv_state_block_ptr, boundary_check=(1,), padding_option="zero")
    if KERNEL_SIZE >= 6:
        tl.advance(conv_state_block_ptr, (1, 0))
        col4 = tl.load(conv_state_block_ptr, boundary_check=(1,), padding_option="zero")

    w_block_ptr = tl.make_block_ptr(
        w_ptr,
        shape=(dim, KERNEL_SIZE),
        strides=(stride_w_dim, stride_w_width),
        offsets=(block_n_idx * BLOCK_N, 0),
        block_shape=(BLOCK_N, 1),
        order=(1, 0),
    )

    w_col0 = tl.trans(tl.load(w_block_ptr, boundary_check=(0,), padding_option="zero"))
    if KERNEL_SIZE >= 2:
        w_block_ptr = tl.advance(w_block_ptr, (0, 1))
        w_col1 = tl.trans(tl.load(w_block_ptr, boundary_check=(0,), padding_option="zero"))
    if KERNEL_SIZE >= 3:
        w_block_ptr = tl.advance(w_block_ptr, (0, 1))
        w_col2 = tl.trans(tl.load(w_block_ptr, boundary_check=(0,), padding_option="zero"))
    if KERNEL_SIZE >= 4:
        w_block_ptr = tl.advance(w_block_ptr, (0, 1))
        w_col3 = tl.trans(tl.load(w_block_ptr, boundary_check=(0,), padding_option="zero"))
    if KERNEL_SIZE >= 5:
        w_block_ptr = tl.advance(w_block_ptr, (0, 1))
        w_col4 = tl.trans(tl.load(w_block_ptr, boundary_check=(0,), padding_option="zero"))
    if KERNEL_SIZE >= 6:
        w_block_ptr = tl.advance(w_block_ptr, (0, 1))
        w_col5 = tl.trans(tl.load(w_block_ptr, boundary_check=(0,), padding_option="zero"))

    x_block_ptr = tl.make_block_ptr(
        x_ptr + seq_idx * stride_x_seq,
        shape=(1, dim),
        strides=(stride_x_seq, stride_x_dim),
        offsets=(0, block_n_idx * BLOCK_N),
        block_shape=(1, BLOCK_N),
        order=(1, 0),
    )
    cur_x = tl.load(x_block_ptr, boundary_check=(1,), padding_option="zero")
    acc = tl.zeros((1, BLOCK_N), dtype=tl.float32)
    mat_w = w_col0
    mat_x = col0

    for j in tl.static_range(KERNEL_SIZE):
        if KERNEL_SIZE == 2:
            if j == 1:
                mat_w, mat_x = w_col1, cur_x
        elif KERNEL_SIZE == 3:
            if j == 1:
                mat_w, mat_x = w_col1, col1
            elif j == 2:
                mat_w, mat_x = w_col2, cur_x
        elif KERNEL_SIZE == 4:
            if j == 1:
                mat_w, mat_x = w_col1, col1
            elif j == 2:
                mat_w, mat_x = w_col2, col2
            elif j == 3:
                mat_w, mat_x = w_col3, cur_x
        elif KERNEL_SIZE == 5:
            if j == 1:
                mat_w, mat_x = w_col1, col1
            elif j == 2:
                mat_w, mat_x = w_col2, col2
            elif j == 3:
                mat_w, mat_x = w_col3, col3
            elif j == 4:
                mat_w, mat_x = w_col4, cur_x
        elif KERNEL_SIZE == 6:
            if j == 1:
                mat_w, mat_x = w_col1, col1
            elif j == 2:
                mat_w, mat_x = w_col2, col2
            elif j == 3:
                mat_w, mat_x = w_col3, col3
            elif j == 4:
                mat_w, mat_x = w_col4, col4
            elif j == 5:
                mat_w, mat_x = w_col5, cur_x

        acc += mat_x * mat_w

    acc = acc / (1 + tl.exp(-acc))
    o_block_ptr = tl.make_block_ptr(
        o_ptr + seq_idx * stride_o_seq,
        shape=(1, dim),
        strides=(stride_o_seq, stride_o_dim),
        offsets=(0, block_n_idx * BLOCK_N),
        block_shape=(1, BLOCK_N),
        order=(1, 0),
    )
    tl.store(o_block_ptr, acc, boundary_check=(1,))

    conv_state = col0
    if KERNEL_SIZE >= 3:
        conv_state = tl.cat(conv_state, col1, dim=0)
    if KERNEL_SIZE >= 4:
        conv_state = tl.cat(conv_state, col2, dim=0)
    if KERNEL_SIZE >= 5:
        conv_state = tl.cat(conv_state, col3, dim=0)
    if KERNEL_SIZE >= 6:
        conv_state = tl.cat(conv_state, col4, dim=0)

    conv_state_full_block_ptr = tl.make_block_ptr(
        conv_state_ptr + conv_state_idx * stride_conv_state_seq,
        shape=(KERNEL_SIZE - 1, dim),
        strides=(stride_conv_state_token, stride_conv_state_dim),
        offsets=(0, block_n_idx * BLOCK_N),
        block_shape=(KERNEL_SIZE - 1, BLOCK_N),
        order=(1, 0),
    )
    tl.store(conv_state_full_block_ptr, conv_state, boundary_check=(1,))


def causal_conv1d_update(
    x: torch.Tensor,  # (seq_cnt, q_dim + k_dim + v_dim)
    weight: torch.Tensor,  # (q_dim+k_dim+v_dim, kernel_size)
    conv_state: torch.Tensor,  # (max_seq_cnt, kernel_size-1, q_dim+k_dim+v_dim)
    conv_state_indices: torch.Tensor,  # (seq_cnt, )
) -> torch.Tensor:
    o = torch.zeros_like(x)
    stride_o_seq, stride_o_dim = o.stride()
    stride_x_seq, stride_x_dim = x.stride()
    stride_w_dim, stride_w_width = weight.stride()
    stride_conv_state_seq, stride_conv_state_token, stride_conv_state_dim = conv_state.stride()
    BLOCK_N = 256
    KERNEL_SIZE = weight.size(1)
    seq_cnt, dim = x.shape
    grid = (seq_cnt, tl.cdiv(dim, BLOCK_N))

    _causal_conv1d_update_kernel[grid](
        x,
        stride_x_seq,
        stride_x_dim,
        weight,
        stride_w_dim,
        stride_w_width,
        conv_state,
        stride_conv_state_seq,
        stride_conv_state_token,
        stride_conv_state_dim,
        conv_state_indices,
        o,
        stride_o_seq,
        stride_o_dim,
        dim,
        KERNEL_SIZE,
        BLOCK_N,
    )

    return o
