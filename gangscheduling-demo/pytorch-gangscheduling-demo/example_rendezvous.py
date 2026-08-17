import os

import torch
import torch.distributed as dist


def main():
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    backend = "nccl" if torch.cuda.is_available() else "gloo"

    dist.init_process_group(backend=backend, init_method="env://", rank=rank, world_size=world_size)
    device = torch.device("cuda", rank % torch.cuda.device_count()) if torch.cuda.is_available() else torch.device("cpu")

    dist.barrier()
    print(f"rank {rank}: all {world_size} ranks reached the barrier")

    broadcast_value = torch.tensor([float(rank)], device=device)
    dist.broadcast(broadcast_value, src=0)
    print(f"rank {rank}: received broadcast value {broadcast_value.item()} from rank 0")

    sum_value = torch.tensor([float(rank)], device=device)
    dist.all_reduce(sum_value, op=dist.ReduceOp.SUM)
    expected_sum = world_size * (world_size - 1) // 2
    print(f"rank {rank}: sum of all ranks = {sum_value.item()} (expected {expected_sum})")

    gathered = [torch.zeros(1, device=device) for _ in range(world_size)]
    dist.all_gather(gathered, torch.tensor([float(rank)], device=device))
    print(f"rank {rank}: all_gather across peers = {[t.item() for t in gathered]}")

    if rank == 0:
        scatter_list = [torch.tensor([float(r * 10)], device=device) for r in range(world_size)]
    else:
        scatter_list = None
    scattered_value = torch.zeros(1, device=device)
    dist.scatter(scattered_value, scatter_list=scatter_list, src=0)
    print(f"rank {rank}: received scattered value {scattered_value.item()} from rank 0")

    dist.barrier()
    print(f"rank {rank}: all collectives done, every rank reached the exit barrier")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
