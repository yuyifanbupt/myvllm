import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn


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


class GatedQParallelLinear(ColumnParallelLinear):
    def __init__(self, hidden_size: int, head_dim: int, total_num_heads: int, bias: bool = False):
        self.hidden_size = hidden_size
        self.head_dim = head_dim
        self.total_num_heads = total_num_heads

        tp_size = dist.get_world_size()
        assert self.total_num_heads % tp_size == 0

        self.num_heads = self.total_num_heads // tp_size
        output_size = self.total_num_heads * self.head_dim * 2  # 2 for gate

        super().__init__(input_size=self.hidden_size, output_size=output_size, bias=bias)

    def weight_loader(self, param: nn.Parameter, loaded_weights: torch.Tensor):
        param_data = param.data
        shard_size = self.num_heads * self.head_dim
        loaded_weight_q_offset = self.tp_rank * shard_size
        loaded_weight_gate_offset = self.total_num_heads * self.head_dim + self.tp_rank * shard_size
        loaded_weight_q = loaded_weights.narrow(0, loaded_weight_q_offset, shard_size)
        loaded_weight_gate = loaded_weights.narrow(0, loaded_weight_gate_offset, shard_size)

        param_data_q_offset = 0
        param_data_gate_offset = shard_size
        param_data_q = param_data.narrow(0, param_data_q_offset, shard_size)
        param_data_gate = param_data.narrow(0, param_data_gate_offset, shard_size)

        param_data_q.copy_(loaded_weight_q)
        param_data_gate.copy_(loaded_weight_gate)


class QKVParallelLinear(ColumnParallelLinear):
    def __init__(
        self,
        hidden_size: int,
        q_head_dim: int,
        k_head_dim: int,
        v_head_dim: int,
        total_num_q_heads: int,
        total_num_k_heads: int,
        total_num_v_heads: int,
        bias: bool = False,
        params_dtype: torch.dtype | None = None,
    ):
        self.hidden_size = hidden_size
        self.q_head_dim = q_head_dim
        self.k_head_dim = k_head_dim
        self.v_head_dim = v_head_dim
        self.total_num_q_heads = total_num_q_heads
        self.total_num_k_heads = total_num_k_heads
        self.total_num_v_heads = total_num_v_heads

        tp_size = dist.get_world_size()
        assert self.total_num_q_heads % tp_size == 0
        assert self.total_num_k_heads % tp_size == 0
        assert self.total_num_v_heads % tp_size == 0

        self.num_q_heads = self.total_num_q_heads // tp_size
        self.num_k_heads = self.total_num_k_heads // tp_size
        self.num_v_heads = self.total_num_v_heads // tp_size
        output_size = (
            self.q_head_dim * self.total_num_q_heads
            + self.k_head_dim * self.total_num_k_heads
            + self.v_head_dim * self.total_num_v_heads
        )

        super().__init__(input_size=hidden_size, output_size=output_size, bias=bias)

    def weight_loader(self, param: nn.Parameter, loaded_weights: torch.Tensor):
        param_data = param.data
        q_shard_size = self.q_head_dim * self.num_q_heads
        k_shard_size = self.k_head_dim * self.num_k_heads
        v_shard_size = self.v_head_dim * self.num_v_heads
        q_shard_offset = 0
        k_shard_offset = q_shard_size
        v_shard_offset = q_shard_size + k_shard_size

        q_total_size = self.q_head_dim * self.total_num_q_heads
        k_total_size = self.k_head_dim * self.total_num_k_heads
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
