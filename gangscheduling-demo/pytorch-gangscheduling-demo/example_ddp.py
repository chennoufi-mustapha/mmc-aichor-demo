import os

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, DistributedSampler


class SyntheticDataset(Dataset):
    def __init__(self, num_samples, num_features):
        self.x = torch.randn(num_samples, num_features)
        self.y = (self.x.sum(dim=1, keepdim=True) > 0).float()

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


def main():
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    backend = "nccl" if torch.cuda.is_available() else "gloo"

    dist.init_process_group(backend=backend, init_method="env://", rank=rank, world_size=world_size)
    device = torch.device("cuda", rank % torch.cuda.device_count()) if torch.cuda.is_available() else torch.device("cpu")

    num_features = 16
    dataset = SyntheticDataset(num_samples=1024, num_features=num_features)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    loader = DataLoader(dataset, batch_size=32, sampler=sampler)

    model = nn.Sequential(nn.Linear(num_features, 32), nn.ReLU(), nn.Linear(32, 1)).to(device)
    ddp_model = nn.parallel.DistributedDataParallel(model)
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.05)
    loss_fn = nn.BCEWithLogitsLoss()

    num_epochs = 3
    for epoch in range(num_epochs):
        sampler.set_epoch(epoch)
        epoch_loss = torch.zeros(1, device=device)

        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = loss_fn(ddp_model(x), y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.detach()

        dist.all_reduce(epoch_loss, op=dist.ReduceOp.SUM)
        averaged_loss = epoch_loss.item() / (len(loader) * world_size)
        print(f"rank {rank}: epoch {epoch} averaged_loss={averaged_loss:.4f}")

    correct = torch.zeros(1, device=device)
    total = torch.zeros(1, device=device)
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            preds = (torch.sigmoid(ddp_model(x)) > 0.5).float()
            correct += (preds == y).sum()
            total += y.numel()

    dist.all_reduce(correct, op=dist.ReduceOp.SUM)
    dist.all_reduce(total, op=dist.ReduceOp.SUM)
    print(f"rank {rank}: global accuracy={(correct / total).item():.4f}")

    if rank == 0:
        torch.save(model.state_dict(), "/tmp/model.pt")
        print("rank 0: saved final model checkpoint")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
