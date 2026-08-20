from collections import deque

import torch


class LinearAttentionCache:
    def __init__(
        self,
        layer_cnt: int,
        max_seq_cnt: int,
        conv_kernel_size: int,
        k_head_dim: int,
        v_head_dim: int,
        num_k_heads: int,
        num_v_heads: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        self.layer_cnt = layer_cnt
        self.max_seq_cnt = max_seq_cnt
        self.conv_kernel_size = conv_kernel_size
        self.k_head_dim = k_head_dim
        self.v_head_dim = v_head_dim
        self.num_k_heads = num_k_heads
        self.num_v_heads = num_v_heads
        self.dtype = dtype
        self.device = device
        self.conv_cache = torch.zeros(
            layer_cnt,
            max_seq_cnt,
            conv_kernel_size - 1,
            2 * k_head_dim * num_k_heads + v_head_dim * num_v_heads,
            dtype=dtype,
            device=device,
        )
        self.recurrent_cache = torch.zeros(
            layer_cnt,
            max_seq_cnt,
            num_k_heads,
            k_head_dim,
            v_head_dim,
            dtype=dtype,
            device=device,
        )
        self.free_slot_ids: deque[int] = deque(range(max_seq_cnt))
        self.used_slot_ids: set[int] = set()

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
        recurrent_cache_size = layer_cnt * num_k_heads * k_head_dim * v_head_dim * dtype.itemsize

        return conv_cache_size + recurrent_cache_size

    def can_allocate(self) -> bool:
        return len(self.free_slot_ids) > 0

    def allocate(self) -> int:
        assert len(self.free_slot_ids) > 0

        slot_id = self.free_slot_ids.popleft()
        self.used_slot_ids.add(slot_id)

        return slot_id

    def deallocate(self, slot_id: int):
        assert slot_id in self.used_slot_ids

        for layer_idx in range(self.layer_cnt):
            self.conv_cache[layer_idx, slot_id].zero_()
            self.recurrent_cache[layer_idx, slot_id].zero_()

        self.used_slot_ids.remove(slot_id)
        self.free_slot_ids.appendleft(slot_id)
