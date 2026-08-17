# XGBoost gang scheduling examples

Each script here is tightly coupled: rank 0 starts a tracker that every rank (including rank 0 itself) must connect to before any collective call can proceed. If the cluster only has room for some of the ranks, the ones that are up sit retrying the connection, holding their resources while waiting for ranks that may take a long time to be scheduled. So we see the need here and in similar scenarios to enable the gangscheduling feature to make sure pods don't run needlessly on the cluster waiting for the others to be scheduled.

## Environment variables

Injected into every worker container to set up the distribution between the different containers:

| Variable | Description | Example |
|----------|-------------|---------|
| `MASTER_PORT` | Port the master listens on for the distributed rendezvous. | `9999` |
| `MASTER_ADDR` | Address of the master container that the others connect to. | `xgboost-dist-demo-master-0` |
| `WORLD_SIZE` | Total number of containers in the run. | `3` |
| `RANK` | Rank of the current container, from `0` to `WORLD_SIZE - 1`. | `1` |

`MASTER_ADDR` and `MASTER_PORT` are where rank 0's `RabitTracker` listens; every rank, including rank 0, connects to that address through `xgboost.collective.CommunicatorContext`.

## `example_simple_rendezvous.py`

The bare minimum: rank 0 starts the tracker, every rank joins the collective through `CommunicatorContext`, and exits. Shows the rendezvous itself with nothing else attached.

## `example_rendezvous.py`

Same rendezvous, followed by a couple of collectives that only work once every rank is present: a `broadcast` from rank 0, and an `allreduce` summing a value contributed by each rank.

## `example_simple_train.py`

The bare minimum distributed training run: each rank builds its own small `DMatrix` and calls `xgb.train`, which synchronizes gradients across every rank in the collective under the hood.

## `example_train.py`

A fuller distributed training loop: each rank builds its own train/test split, trains for more boosting rounds, evaluates local accuracy, `allreduce`s it into a global accuracy, and rank 0 saves the final model checkpoint.
