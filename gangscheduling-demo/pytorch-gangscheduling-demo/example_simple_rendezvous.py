import os

import torch.distributed as dist


def main():
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    dist.init_process_group(backend="gloo", init_method="env://", rank=rank, world_size=world_size)
    print(f"rank {rank}/{world_size} joined the process group")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
