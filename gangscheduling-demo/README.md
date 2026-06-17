# Gang scheduling

In here we will go through what gang scheduling is, what it does, and when we should (and shouldn't) use it, using demos across the different frameworks and operators supported by AIchor: Kuberay, JAX, PyTorch, and XGBoost.

## What is gang scheduling?

Distributed training jobs are made up of several pods that only make progress once all of them are running together (e.g. a PyTorch job's workers all need to rendezvous, a Ray cluster needs its head and workers up, a JAX SPMD job needs every host present before the collective ops can run).

The default behavior before this gangscheduling feature was that the scheduler placed pods one at a time, with no notion that a group of pods belongs together. The result of this was often **Wasted compute**, pods that did get scheduled sit idle waiting for the rest of the group, burning GPU/CPU allocation without doing useful work. This was most common with ray experiments as they would not be submitted until all their workers had started

**Gang scheduling** fixes this by scheduling a group ("gang") of pods as a single unit: either enough resources exist to place the whole group and they all start together, or none of them are scheduled and the job waits as a whole until it can.

## It's about coupling and not not pod count

Having multiple replicas in a job does **not** automatically mean it needs gang scheduling. What matters is whether those replicas are **tightly coupled** (they must all be present at once to make any progress) or **independent** (each replica can do useful work on its own, regardless of whether its siblings are scheduled yet). Every framework here can be used either way:

- **PyTorch**
  - Tightly coupled: a DDP/FSDP job where every worker must rendezvous before training can start stepping — needs gang scheduling.
  - Independent: several separate PyTorch training runs submitted as replicas of the same job (e.g. training multiple model variants, or a sweep), each running its own single-process training loop with no dependency on the others — doesn't need gang scheduling.

- **JAX**
  - Tightly coupled: a multi-host SPMD job where every host must be present before collective ops can execute — needs gang scheduling.
  - Independent: several separate single-host JAX programs run as replicas (e.g. a parameter sweep, each on its own devices) — doesn't need gang scheduling, since there's no cross-replica rendezvous.

- **Ray (kuberay)**
  - Tightly coupled: cluster bootstrap — the head node and whatever minimum worker count a job actually needs to start must come up together — needs gang scheduling for that initial set.
  - Independent: once the cluster is up and autoscaling, additional workers requested as load ramps up don't need to arrive together — Ray puts each new worker to use as soon as it lands. Gang scheduling those incremental requests would work against the point of autoscaling.

- **XGBoost**
  - Tightly coupled: a distributed training job where rank 0 starts the tracker and every rank must connect to it before the collective can run boosting rounds together — needs gang scheduling.
  - Independent: several separate single-node XGBoost training runs submitted as replicas of the same job (e.g. a hyperparameter sweep), each training its own model with no rendezvous between replicas — doesn't need gang scheduling.

## When to use gangscheduling

- You're running **tightly-coupled multi-pod jobs** — replicas that must rendezvous to make progress, such as PyTorch DDP/FSDP, JAX multi-host SPMD, a Ray cluster's initial head + minimum-worker bootstrap, or a distributed XGBoost job whose workers must all connect to the tracker.

## When it's not relevant

- **Independent-replica workloads**, such as a sweep of standalone PyTorch/JAX/XGBoost runs, or an autoscaling Ray cluster's incremental worker additions — each replica is useful on its own, so all-or-nothing scheduling only adds latency for no benefit.
- Single-pod jobs (e.g. a single-worker training script) have nothing to "gang" — there's only one pod to schedule.

## Demos

Each subfolder is a minimal, self-contained project showing where gangscheduling would be relevant with different operators. These examples are the textbook case for gang scheduling — they demonstrate the failure mode (blocking rendezvous, wasted idle resources under partial scheduling) that gang scheduling exists to prevent:

| Demo | Operator | What it shows |
| --- | --- | --- |
| [`pytorch-gangscheduling-demo`](./pytorch-gangscheduling-demo) | `pytorch` | Multi-worker PyTorch distributed job. |
| [`jax-gangscheduling-demo`](./jax-gangscheduling-demo) | `jax` | Multi-host JAX SPMD job. |
| [`kuberay-gangscheduling-demo`](./kuberay-gangscheduling-demo) | `kuberay` | Ray cluster where the head + minimum worker set is gang-scheduled to bootstrap, while additional autoscaled workers join independently as demand ramps up. |
| [`xgboost-gangscheduling-demo`](./xgboost-gangscheduling-demo) | `xgboost` | Multi-worker XGBoost distributed job whose workers must all connect to rank 0's tracker before training can run. |


> More detail on each operator's specific setup lives in that demo's own README.
