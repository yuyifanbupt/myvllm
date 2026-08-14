from collections.abc import Callable
from functools import lru_cache

import torch
import torch.distributed as dist
import torch.nn.functional as F
import triton
import triton.language as tl
from einops import rearrange
from torch import nn

from myvllm.configs.qwen3_5 import Qwen3_5MoeVisionConfig
from myvllm.layers.conv import Conv3D
from myvllm.layers.full_attention import flash_attention_prefill
from myvllm.layers.linear import ColumnParallelLinear, QKVParallelLinear, RowParallelLinear
from myvllm.layers.norm import LayerNorm
from myvllm.layers.rotary_embedding import RotaryEmbedding, apply_rotary_pos_emb


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


class VisionAttention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        head_dim: int,
        num_heads: int,
        head_v_dim: None | int = None,
        num_kv_heads: None | int = None,
    ) -> None:
        super().__init__()
        self.tp_rank = dist.get_rank()
        self.tp_size = dist.get_world_size()
        self.total_num_heads = num_heads
        self.total_num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads
        self.head_dim = head_dim
        self.head_v_dim = head_v_dim if head_v_dim is not None else head_dim

        assert self.total_num_heads % self.tp_size == 0
        assert self.total_num_kv_heads % self.tp_size == 0

        self.num_heads = self.total_num_heads // self.tp_size
        self.num_kv_heads = self.total_num_kv_heads // self.tp_size
        self.q_size = self.head_dim * self.num_heads
        self.k_size = self.head_dim * self.num_kv_heads
        self.v_size = self.head_v_dim * self.num_kv_heads
        self.scale = self.head_dim**-0.5

        self.qkv = QKVParallelLinear(
            hidden_size,
            head_dim,
            num_heads,
            total_num_kv_heads=num_kv_heads,
            v_head_size=head_v_dim,
            bias=True,
        )
        self.proj = RowParallelLinear(
            self.head_v_dim * self.total_num_kv_heads,
            hidden_size,
            bias=True,
        )

    def forward(
        self,
        x: torch.Tensor,
        cu_seqlens: torch.Tensor,
        rotary_pos_emb_cos: torch.Tensor | None,
        rotary_pos_emb_sin: torch.Tensor | None,
    ) -> torch.Tensor:
        """
        x.shape (seq_len, hidden_size)
        """
        # TODO: 标记入参和出参 shape
        qkv: torch.Tensor = self.qkv(x)
        q, k, v = qkv.split([self.q_size, self.k_size, self.v_size], dim=-1)
        q = rearrange(q, "seq_len (num_heads head_dim) -> seq_len num_heads head_dim", head_dim=self.head_dim)
        k = rearrange(k, "seq_len (num_kv_heads head_dim) -> seq_len num_kv_heads head_dim", head_dim=self.head_dim)
        v = rearrange(
            v, "seq_len (num_kv_heads head_v_dim) -> seq_len num_kv_heads head_dim", head_v_dim=self.head_v_dim
        )

        if rotary_pos_emb_cos is not None and rotary_pos_emb_sin is not None:
            q = apply_rotary_pos_emb(q, rotary_pos_emb_cos, rotary_pos_emb_sin)
            k = apply_rotary_pos_emb(k, rotary_pos_emb_cos, rotary_pos_emb_sin)

        o = flash_attention_prefill(
            q,
            k,
            v,
            cu_seqlens,
            self.scale,
            self.num_heads,
            self.num_kv_heads,
            self.head_dim,
            self.head_v_dim,
            False,
        )
        o = rearrange(o, "seq_len num_heads head_v_dim -> seq_len (num_heads head_v_dim)")

        return self.proj(o)


class VisionMLP(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        bias: bool = False,
        act_fn: Callable[[torch.Tensor], torch.Tensor] = F.silu,
    ) -> None:
        super().__init__()
        self.linear_fc1 = ColumnParallelLinear(in_features, hidden_features, bias=bias)
        self.linear_fc2 = RowParallelLinear(hidden_features, in_features, bias=bias)
        self.act_fn = act_fn

    @torch.compile
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear_fc2(self.act_fn(self.linear_fc1(x)))


class VisionBlock(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        head_dim: int,
        num_heads: int,
        mlp_hidden_dim: int,
        act_fn: Callable[[torch.Tensor], torch.Tensor] = F.silu,
    ) -> None:
        super().__init__()
        self.norm1 = LayerNorm((hidden_size,), eps=1e-6)
        self.norm2 = LayerNorm((hidden_size,), eps=1e-6)
        self.attn = VisionAttention(hidden_size, head_dim, num_heads)
        self.mlp = VisionMLP(hidden_size, mlp_hidden_dim, bias=True, act_fn=act_fn)

    def forward(
        self,
        x: torch.Tensor,
        cu_seqlens: torch.Tensor,
        rotary_pos_emb_cos: torch.Tensor | None,
        rotary_pos_emb_sin: torch.Tensor | None,
        residual: None | torch.Tensor = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x, r = self.norm1(x, residual)
        x = self.attn(x, cu_seqlens, rotary_pos_emb_cos, rotary_pos_emb_sin)
        x, r = self.norm2(x, r)
        x = self.mlp(x)

        return x, r


class VisionPatchMerger(nn.Module):
    def __init__(
        self,
        d_model: int,
        context_dim: int,
        spatial_merge_size: int,
        use_postshuffle_norm: bool = False,
    ) -> None:
        super().__init__()
        self.hidden_size = context_dim * (spatial_merge_size**2)
        self.use_postshuffle_norm = use_postshuffle_norm

        if self.use_postshuffle_norm:
            context_dim = self.hidden_size
        self.norm = LayerNorm((context_dim,), eps=1e-6)
        self.linear_fc1 = ColumnParallelLinear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_fc2 = RowParallelLinear(self.hidden_size, d_model, bias=True)
        self.act_fn = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_postshuffle_norm:
            x, _ = self.norm(x.view(-1, self.hidden_size))
        else:
            x, _ = self.norm(x)
            x = x.view(-1, self.hidden_size)

        return self.linear_fc2(self.act_fn(self.linear_fc1(x)))


class VisionPatchEmbed(nn.Module):
    def __init__(self, patch_size: int, temporal_patch_size: int, in_channels: int, hidden_size: int) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.temporal_patch_size = temporal_patch_size
        self.hidden_size = hidden_size

        kernel_size = (temporal_patch_size, patch_size, patch_size)
        self.proj = Conv3D(in_channels, hidden_size, kernel_size, kernel_size, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = rearrange(
            x,
            "L (c tp hp wp) -> L c tp hp wp",
            c=self.in_channels,
            tp=self.temporal_patch_size,
            hp=self.patch_size,
            wp=self.patch_size,
        )
        x = self.proj(x).view(-1, self.hidden_size)

        return x


class VisionTransformer(nn.Module):
    def __init__(self, vision_config: Qwen3_5MoeVisionConfig) -> None:
        super().__init__()
        self.in_channels = vision_config.in_channels
        self.hidden_size = vision_config.hidden_size
        self.num_heads = vision_config.num_heads
        self.num_position_embeddings = vision_config.num_position_embeddings
        self.patch_size = vision_config.patch_size
        self.spatial_merge_size = vision_config.spatial_merge_size
        self.spatial_merge_unit = self.spatial_merge_size**2
        self.temporal_patch_size = vision_config.temporal_patch_size
        self.num_grid_per_side = int(self.num_position_embeddings**0.5)

        self.patch_embed = VisionPatchEmbed(
            self.patch_size,
            self.temporal_patch_size,
            self.in_channels,
            self.hidden_size,
        )

        # TODO: 这里考虑下要不要用tensor parallel
        self.pos_embed = nn.Embedding(self.num_position_embeddings, self.hidden_size)

        head_dim = self.hidden_size // self.num_heads
        self.rotary_pos_emb = RotaryEmbedding(head_dim // 2)
        self.merger = VisionPatchMerger(vision_config.out_hidden_size, self.hidden_size, self.spatial_merge_size)
        self.act_fn = nn.GELU(approximate="tanh")
        self.blocks = nn.ModuleList(
            [
                VisionBlock(
                    self.hidden_size,
                    head_dim,
                    self.num_heads,
                    vision_config.intermediate_size,
                    act_fn=self.act_fn,
                )
                for layer_idx in range(vision_config.depth)
            ]
        )

    @property
    def device(self):
        return self.patch_embed.proj.weight.device

    @property
    def dtype(self):
        return self.patch_embed.proj.weight.dtype

    @staticmethod
    @lru_cache(maxsize=1024)
    def rot_pos_ids(h: int, w: int, spatial_merge_size: int) -> torch.Tensor:
        hpos_ids = torch.arange(h, device="cpu").view(h, 1).expand(h, w)
        wpos_ids = torch.arange(w, device="cpu").view(1, w).expand(h, w)
        hpos_ids = rearrange(hpos_ids, "(hd hm) (wd wm) -> (hd wd hm wm)", hm=spatial_merge_size, wm=spatial_merge_size)
        wpos_ids = rearrange(wpos_ids, "(hd hm) (wd wm) -> (hd wd hm wm)", hm=spatial_merge_size, wm=spatial_merge_size)

        return torch.stack([hpos_ids, wpos_ids], dim=-1)

    def rot_pos_emb(self, grid_thw: torch.Tensor):
        pos_ids = [
            self.rot_pos_ids(h.item(), w.item(), self.spatial_merge_size)
            if t == 1
            else self.rot_pos_ids(h.item(), w.item(), self.spatial_merge_size).repeat(int(t.item()), 1)
            for t, h, w in grid_thw
        ]
        pos_ids = torch.cat(pos_ids, dim=0).to(device=self.device, non_blocking=True)
        rotaries = self.rotary_pos_emb(pos_ids)
        cos, sin = rotaries.cos(), rotaries.sin()

        return cos, sin

    def get_cu_seqlens(self, grid_thw: torch.Tensor) -> torch.Tensor:
        patches_per_frame = grid_thw[:, 1] * grid_thw[:, 2]
        cu_seqlens = torch.repeat_interleave(patches_per_frame, grid_thw[:, 0], dim=0).cumsum(0, dtype=torch.int32)
        cu_seqlens = torch.cat([torch.zeros(1, dtype=torch.int32, device=cu_seqlens.device), cu_seqlens])

        return cu_seqlens

    def pos_embed_interpolate(self, grid_thw: torch.Tensor) -> torch.Tensor:
        outputs = []
        for t, h, w in grid_thw:
            outputs.append(
                triton_pos_embed_interpolate(
                    self.pos_embed.weight,
                    int(t.item()),
                    int(h.item()),
                    int(w.item()),
                    self.num_grid_per_side,
                    self.spatial_merge_size,
                    self.dtype,
                )
            )

        return torch.cat(outputs, dim=0)

    def forward(self, x: torch.Tensor, grid_thw: torch.Tensor) -> torch.Tensor:
        cu_seqlens = self.get_cu_seqlens(grid_thw)
        rotary_pos_emb_cos, rotary_pos_emb_sin = self.rot_pos_emb(grid_thw)
        pos_embeds = self.pos_embed_interpolate(grid_thw)

        hidden_state = x.to(device=self.device, dtype=self.dtype)
        hidden_state = self.patch_embed(hidden_state)
        hidden_state = hidden_state + pos_embeds
        residual = None

        for layer_num, blk in enumerate(self.blocks):
            hidden_state, residual = blk(hidden_state, cu_seqlens, rotary_pos_emb_cos, rotary_pos_emb_sin, residual)

        hidden_state = self.merger(hidden_state)

        return hidden_state
