import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn


class VocabParallelEmbedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__()
        self.tp_size = dist.get_world_size()
        self.tp_rank = dist.get_rank()

        # keep the original num_embeddings
        self.num_embeddings = num_embeddings
        # pad to make it divisible by tp_size
        self.padded_num_embeddings = (num_embeddings + self.tp_size - 1) // self.tp_size * self.tp_size
        # this is the num_embeddings per partition in this current GPU
        self.num_embeddings_per_partition = self.padded_num_embeddings // self.tp_size
        self.embedding_dim = embedding_dim

        self.weight = nn.Parameter(torch.empty(self.num_embeddings_per_partition, embedding_dim))
        self.weight.__setattr__("weight_loader", self.weight_loader)

    def weight_loader(self, param: nn.Parameter, loaded_weights: torch.Tensor):
        param_data = param.data

        offset = self.tp_rank * self.num_embeddings_per_partition
        shard_size = self.num_embeddings_per_partition

        # calculate how much of the original vocab falls in this partition
        actual_start = min(offset, self.num_embeddings)
        actual_end = min(offset + shard_size, self.num_embeddings)
        actual_size = max(0, actual_end - actual_start)

        if actual_size > 0:
            # load the actual weights
            sharded_weights = loaded_weights.narrow(0, actual_start, actual_size)
            param_data[:actual_size].copy_(sharded_weights)

        # pad the rest with zeros if needed
        if actual_size < shard_size:
            param_data[actual_size:].zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # mask for tokens in this partition's range and within original vocab size
        mask = (
            (x >= self.tp_rank * self.num_embeddings_per_partition)
            & (x < (self.tp_rank + 1) * self.num_embeddings_per_partition)
            & (x < self.num_embeddings)
        )
        x = mask * (x - self.tp_rank * self.num_embeddings_per_partition)
        output = F.embedding(x, self.weight)

        if dist.get_world_size() > 1:
            # need to mask again, otherwise the embedding for the out-of-range ids will be the embedding of id 0
            output = mask.unsqueeze(1) * output
            dist.all_reduce(output, op=dist.ReduceOp.SUM)
        return output
