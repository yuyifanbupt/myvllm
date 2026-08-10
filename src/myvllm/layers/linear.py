import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import load, nn


class Linear(nn.Module):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
        prefix: str = "",
    ):
        super().__init__()

        # Keep input parameters
        self.input_size = input_size
        self.output_size = output_size
        self.weight = nn.Parameter(torch.empty(output_size, input_size))
        self.weight.__setattr__("weight_loader", self.weight_loader)

        if bias:
            self.bias = nn.Parameter(torch.zeros(output_size))
            self.bias.__setattr__("weight_loader", self.weight_loader)
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight, self.bias)

    def weight_loader(self, param: nn.Parameter, loaded_weights: torch.Tensor):
        param.data.copy_(loaded_weights)


class ColumnParallelLinear(Linear):
    def __init__(self, input_size: int, output_size: int, bias: bool = False, prefix: str = ""):
        self.tp_rank = dist.get_rank()
        self.tp_size = dist.get_world_size()

        assert output_size % self.tp_size == 0

        super().__init__(input_size, output_size // self.tp_size, bias, prefix)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight, self.bias)

    def weight_loader(self, param: nn.Parameter, loaded_weights: torch.Tensor):
        param_data = param.data
        loaded_weights = loaded_weights.narrow(0, self.tp_rank * self.output_size, self.output_size)
        param_data.copy_(loaded_weights)


class QKVParallelLinear(ColumnParallelLinear):
    def __init__(
        self,
        hidden_size: int,
        head_size: int,
        total_num_heads: int,
        total_num_kv_heads: int | None = None,
        bias: bool = False,
        params_dtype: torch.dtype | None = None,
        v_head_size: int | None = None,
    ):
        self.hidden_size = hidden_size
        self.head_size = head_size
        self.v_head_size = v_head_size if v_head_size is not None else head_size
        self.total_num_heads = total_num_heads
        self.total_num_kv_heads = total_num_kv_heads if total_num_kv_heads is not None else total_num_heads

        tp_size = dist.get_world_size()
        assert self.total_num_heads % tp_size == 0
        assert self.total_num_kv_heads % tp_size == 0

        self.num_heads = self.total_num_heads // tp_size
        self.num_kv_heads = self.total_num_kv_heads // tp_size  # TODO: 这里需要想想 tp_size > total_num_kv_heads 怎么办
        output_size = (
            self.head_size * self.total_num_heads
            + self.head_size * self.total_num_kv_heads
            + self.v_head_size * self.total_num_kv_heads
        )

        super().__init__(input_size=hidden_size, output_size=output_size, bias=bias)

    def weight_loader(self, param: nn.Parameter, loaded_weights: torch.Tensor):
        param_data = param.data
        q_shard_size = self.head_size * self.num_heads
        k_shard_size = self.head_size * self.num_kv_heads
        v_shard_size = self.v_head_size * self.num_kv_heads
        q_shard_offset = 0
        k_shard_offset = q_shard_size
        v_shard_offset = q_shard_size + k_shard_size

        q_total_size = self.head_size * self.total_num_heads
        k_total_size = self.head_size * self.total_num_kv_heads
        q_loaded_weights_offset = 0 + self.tp_rank * q_shard_size
        k_loaded_weights_offset = q_total_size + self.tp_rank * k_shard_size
        v_loaded_weights_offset = q_total_size + k_total_size + self.tp_rank * v_shard_size

        q_loaded_weights = loaded_weights.narrow(0, q_loaded_weights_offset, q_shard_size)
        k_loaded_weights = loaded_weights.narrow(0, k_loaded_weights_offset, k_shard_size)
        v_loaded_weights = loaded_weights.narrow(0, v_loaded_weights_offset, v_shard_size)

        q_param_data = param_data.narrow(0, q_shard_offset, q_shard_size)
        k_param_data = param_data.narrow(0, k_shard_offset, k_shard_size)
        v_param_data = param_data.narrow(0, v_shard_offset, v_shard_size)

        q_param_data.copy_(q_loaded_weights)
        k_param_data.copy_(k_loaded_weights)
        v_param_data.copy_(v_loaded_weights)


class RowParallelLinear(Linear):
    def __init__(self, input_size: int, output_size: int, bias: bool = False, prefix: str = ""):
        self.tp_rank = dist.get_rank()
        self.tp_size = dist.get_world_size()

        assert input_size % self.tp_size == 0

        super().__init__(input_size // self.tp_size, output_size, bias, prefix)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        result = F.linear(x, self.weight, bias=self.bias)
        if self.tp_size > 1:
            dist.all_reduce(result, op=dist.ReduceOp.SUM)

        return result

    def weight_loader(self, param: nn.Parameter, loaded_weights: torch.Tensor):
        param_data = param.data
        loaded_weights = loaded_weights.narrow(1, self.tp_rank * self.input_size, self.input_size)
        param_data.copy_(loaded_weights)
