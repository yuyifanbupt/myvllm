import torch
import triton
import triton.language as tl


@triton.jit
def _bilinear_pos_embed_kernel(
    embed_ptr,
    output_ptr,
    H,
    W,
    h_scale,
    w_scale,
    NUM_GRID: tl.constexpr,
    M_SIZE: tl.constexpr,
    HIDDEN_DIM: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Fused bilinear pos-embed interpolation with spatial-merge reorder."""
    pid = tl.program_id(0)
    total_spatial = H * W
    spatial_idx = pid % total_spatial

    num_blocks_w = W // M_SIZE
    block_idx = spatial_idx // (M_SIZE * M_SIZE)
    local_idx = spatial_idx % (M_SIZE * M_SIZE)
    br = block_idx // num_blocks_w
    bc = block_idx % num_blocks_w
    lr = local_idx // M_SIZE
    lc = local_idx % M_SIZE
    row = br * M_SIZE + lr
    col = bc * M_SIZE + lc

    h_frac = row.to(tl.float32) * h_scale
    w_frac = col.to(tl.float32) * w_scale

    hf = tl.math.floor(h_frac).to(tl.int32)
    wf = tl.math.floor(w_frac).to(tl.int32)
    hc = tl.minimum(hf + 1, NUM_GRID - 1)
    wc = tl.minimum(wf + 1, NUM_GRID - 1)

    dh = h_frac - hf.to(tl.float32)
    dw = w_frac - wf.to(tl.float32)
    w11 = dh * dw
    w10 = dh - w11
    w01 = dw - w11
    w00 = 1.0 - dh - w01

    off00 = (hf * NUM_GRID + wf) * HIDDEN_DIM
    off01 = (hf * NUM_GRID + wc) * HIDDEN_DIM
    off10 = (hc * NUM_GRID + wf) * HIDDEN_DIM
    off11 = (hc * NUM_GRID + wc) * HIDDEN_DIM
    out_off = pid * HIDDEN_DIM

    # Cast weights to output dtype so the multiply-accumulate stays
    # in the same precision as the native PyTorch implementation.
    out_dtype = output_ptr.dtype.element_ty
    w00_c = w00.to(out_dtype)
    w01_c = w01.to(out_dtype)
    w10_c = w10.to(out_dtype)
    w11_c = w11.to(out_dtype)

    for d in tl.range(0, HIDDEN_DIM, BLOCK_D):  # pyright: ignore
        cols = d + tl.arange(0, BLOCK_D)
        mask = cols < HIDDEN_DIM

        e00 = tl.load(embed_ptr + off00 + cols, mask=mask)
        e01 = tl.load(embed_ptr + off01 + cols, mask=mask)
        e10 = tl.load(embed_ptr + off10 + cols, mask=mask)
        e11 = tl.load(embed_ptr + off11 + cols, mask=mask)

        val = w00_c * e00 + w01_c * e01 + w10_c * e10 + w11_c * e11

        tl.store(output_ptr + out_off + cols, val, mask=mask)


def triton_pos_embed_interpolate(
    embed_weight: torch.Tensor,
    t: int,
    h: int,
    w: int,
    num_grid_per_side: int,
    m_size: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Launch the fused Triton kernel for one (t,h,w) grid.

    Returns a tensor of shape ``(t * h * w, hidden_dim)`` with the
    bilinearly-interpolated position embeddings in spatial-merge order.
    """
    assert h % m_size == 0 and w % m_size == 0, f"h={h} and w={w} must be divisible by m_size={m_size}"
    hidden_dim = embed_weight.shape[1]
    total_out = t * h * w
    output = torch.empty(
        total_out,
        hidden_dim,
        device=embed_weight.device,
        dtype=dtype,
    )

    h_scale = float(num_grid_per_side - 1) / float(h - 1) if h > 1 else 0.0
    w_scale = float(num_grid_per_side - 1) / float(w - 1) if w > 1 else 0.0

    BLOCK_D = triton.next_power_of_2(hidden_dim)

    _bilinear_pos_embed_kernel[(total_out,)](
        embed_weight,
        output,
        h,
        w,
        h_scale,
        w_scale,
        num_grid_per_side,  # pyright: ignore[reportArgumentType]
        m_size,  # pyright: ignore[reportArgumentType]
        hidden_dim,  # pyright: ignore[reportArgumentType]
        BLOCK_D,
    )
    return output
