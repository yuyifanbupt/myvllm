from torch.nn.modules import padding
import triton
import triton.language as tl


# TODO: add bias?
@triton.jit()
def _causal_conv1d_update_kernel(
    x_ptr,
    stride_x_token,
    stride_x_dim,
    w_ptr,  # (dim, kernel_size)
    stride_w_width,
    stride_w_dim,
    conv_state_ptr,
    stride_conv_state_seq,
    stride_conv_state_token,
    stride_conv_state_dim,
    conv_state_indices_ptr,
    stride_conv_state_indices,
    cu_seqlens_ptr,
    o_ptr,
    stride_o_token,
    stride_o_dim,
    dim,
    KERNEL_SIZE: tl.constexpr,
    BLOCK_N: tl.constexpr,
) -> None:
    seq_idx = tl.program_id(0)
    block_n_idx = tl.program_id(1)

    conv_state_idx = tl.load(conv_state_indices_ptr + seq_idx * stride_conv_state_indices)
    token_offset_start = tl.load(cu_seqlens_ptr + seq_idx)
    token_offset_end = tl.load(cu_seqlens_ptr + seq_idx + 1)
    seqlen = token_offset_end - token_offset_start

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

    conv_state = tl.load(conv_state_block_ptr, boundary_check=(1,), padding_option="zero")
    acc_preload = tl.zeros((1, BLOCK_N), dtype=tl.float32)

    w_block_ptr = tl.make_block_ptr(
        w_ptr,
        shape=(KERNEL_SIZE, dim),
        strides=(stride_w_width, stride_w_dim),
        offsets=(0, block_n_idx * BLOCK_N),
        block_shape=(1, BLOCK_N),
        order=(1, 0),
    )

    w_col0 = tl.load(w_block_ptr, boundary_check=(1,), padding_option="zero")
    if KERNEL_SIZE >= 2:
        w_block_ptr = tl.advance(w_block_ptr, (1, 0))
        w_col1 = tl.load(w_block_ptr, boundary_check=(1,), padding_option="zero")
    if KERNEL_SIZE >= 3:
        w_block_ptr = tl.advance(w_block_ptr, (1, 0))
        w_col2 = tl.load(w_block_ptr, boundary_check=(1,), padding_option="zero")
    if KERNEL_SIZE >= 4:
        w_block_ptr = tl.advance(w_block_ptr, (1, 0))
        w_col3 = tl.load(w_block_ptr, boundary_check=(1,), padding_option="zero")
    if KERNEL_SIZE >= 5:
        w_block_ptr = tl.advance(w_block_ptr, (1, 0))
        w_col4 = tl.load(w_block_ptr, boundary_check=(1,), padding_option="zero")
    if KERNEL_SIZE >= 6:
        w_block_ptr = tl.advance(w_block_ptr, (1, 0))
        w_col5 = tl.load(w_block_ptr, boundary_check=(1,), padding_option="zero")

    x_block_ptr = tl.make_block_ptr(
        x_ptr + token_offset_start * stride_x_token,
        shape=(0, dim),
        strides=(stride_x_token, stride_x_dim),
        offsets=(0, block_n_idx * BLOCK_N),
        block_shape=(1, BLOCK_N),
        order=(1, 0),
    )
    o_block_ptr = tl.make_block_ptr(
        o_ptr + token_offset_start * stride_o_token,
        shape=(0, dim),
        strides=(stride_o_token, stride_o_dim),
        offsets=(0, block_n_idx * BLOCK_N),
        block_shape=(1, BLOCK_N),
        order=(1, 0),
    )

    for token_idx in tl.range(seqlen):
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
