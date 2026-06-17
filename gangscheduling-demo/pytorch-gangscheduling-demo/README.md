# PyTorch gang scheduling examples

Each script here is tightly coupled: every rank calls `dist.init_process_group` and blocks until all `WORLD_SIZE` ranks have joined. If the cluster only has room for some of the ranks, the ones that are up sit inside that call forever, holding their resources while waiting for ranks that may never be scheduled.

## Environment variables

Injected into every worker container to set up the distribution between the different containers:

| Variable | Description | Example |
|----------|-------------|---------|
| `MASTER_PORT` | Port the master listens on for the distributed rendezvous. | `23456` |
| `MASTER_ADDR` | Address of the master container that the others connect to. | `pytorch-dist-cifar-master-0` |
| `WORLD_SIZE` | Total number of containers in the run. | `3` |
| `RANK` | Rank of the current container, from `0` to `WORLD_SIZE - 1`. | `1` |

`MASTER_ADDR` and `MASTER_PORT` are read by `torch.distributed` itself through the `env://` init method; these examples only read `RANK` and `WORLD_SIZE` directly.

## `example_simple_rendezvous.py`

The bare minimum: joins the process group and exits. Shows the rendezvous itself with nothing else attached.

## `example_rendezvous.py`

Same rendezvous, followed by a tour of collectives that only work once every rank is present: `barrier`, `broadcast`, `all_reduce`, `all_gather`, and `scatter`.

## `example_simple_ddp.py`

Wraps a tiny `nn.Linear` in `DistributedDataParallel` and runs a single backward pass, showing that DDP's gradient sync requires every rank to participate in the same backward call.

## `example_ddp.py`

A fuller distributed training loop: a synthetic dataset split with `DistributedSampler`, several epochs of training, per-epoch loss averaged across ranks with `all_reduce`, a global accuracy check, and a rank-0 checkpoint save at the end.
