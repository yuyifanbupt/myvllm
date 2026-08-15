import torch


class LinearAttentionCache:
    def __init__(self, conv_cache: torch.Tensor, recurrent_cache: torch.Tensor) -> None:
        self.conv_cache = conv_cache
        self.recurrent_cache = recurrent_cache

    @staticmethod
    def bytes_per_seq(
        layer_cnt: int,
        conv_kernel_size: int,
        k_head_dim: int,
        v_head_dim: int,
        num_k_heads: int,
        num_v_heads: int,
        dtype: torch.dtype,
    ) -> int:
        conv_cache_size = (
            layer_cnt
            * (conv_kernel_size - 1)
            * (2 * k_head_dim * num_k_heads + v_head_dim * num_v_heads)
            * dtype.itemsize
        )
        # recurr
        return 0

    @staticmethod
    def create_cache_tensor(
        layer_cnt: int,
        max_seq_cnt: int,
        conv_kernel_size: int,
        k_head_dim: int,
        v_head_dim: int,
        num_k_heads: int,
        num_v_heads: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        conv_cache = torch.zeros(
            layer_cnt,
            max_seq_cnt,
            conv_kernel_size - 1,
            2 * k_head_dim * num_k_heads + v_head_dim * num_v_heads,
            dtype=dtype,
            device=device,
        )
        recurrent_cache = torch.zeros(
            layer_cnt,
            max_seq_cnt,
            num_k_heads,
            k_head_dim,
            v_head_dim,
            dtype=torch.float32,
            device=device,
        )

        return conv_cache, recurrent_cache
