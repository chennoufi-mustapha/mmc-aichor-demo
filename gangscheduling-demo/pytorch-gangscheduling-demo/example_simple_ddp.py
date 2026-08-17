import os

import torch
import torch.distributed as dist
import torch.nn as nn


def main():
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    dist.init_process_group(backend="gloo", init_method="env://", rank=rank, world_size=world_size)

    model = nn.Linear(4, 4)
    ddp_model = nn.parallel.DistributedDataParallel(model)

    x = torch.randn(2, 4)
    loss = ddp_model(x).sum()
    loss.backward()

    print(f"rank {rank}: backward pass complete, gradients synced across {world_size} ranks")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
