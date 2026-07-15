# JAX gang scheduling examples

Each script here is tightly coupled: every process calls `jax.distributed.initialize` and blocks until all `num_processes` hosts have joined. Here we see the clear need for gangscheduling because if the cluster only has room for some of the hosts, the ones that are up sit inside that call forever, holding their resources while waiting for hosts that may take a long time to be scheduled. So we see the need here and in similar scenarios to enable the gangscheduling feature to make sure pods don't run needlessly on the cluster waiting for the others to be scheduled.

## Environment variables

AIchor designates the first replica as the JAX distributed coordinator and injects the following into every worker pod:

| Variable | Description | Example |
|----------|-------------|---------|
| `JAXOPERATOR_COORDINATOR_ADDRESS` | Address of the coordinator (the first pod), host and port included. This is the value these examples pass to `jax.distributed.initialize`. | `worker-0-0.experiment-6f4a:1234` |
| `JAXOPERATOR_COORDINATOR_HOST` | The coordinator pod's hostname, without a port. Not used by these examples since `JAXOPERATOR_COORDINATOR_ADDRESS` already includes the port. | `worker-0-0.experiment-6f4a` |
| `JAXOPERATOR_NUM_PROCESSES` | Total number of JAX pods running in parallel (equals the worker `count`). | `2` |
| `JAXOPERATOR_PROCESS_ID` | Rank of the current pod among all pods in the job, from `0` to `JAXOPERATOR_NUM_PROCESSES - 1`. The coordinator is always `0`. | `0` on the coordinator, `1` on the second worker |

`JAXOPERATOR_COORDINATOR_ADDRESS` is set to the literal value `$(JAXOPERATOR_COORDINATOR_HOST):1234`, shell-expanded when the command starts — these examples can read it directly without appending a port themselves.

## `example_simple_rendezvous.py`

The bare minimum: joins the distributed runtime and exits. Shows the rendezvous itself with nothing else attached.

## `example_rendezvous.py`

Same rendezvous, followed by a tour of collectives that only produce a correct result once every host is present: `psum`, `pmean`, `pmax`, and `all_gather`, all run through `pmap`.

## `example_simple_multihost_pmap.py`

The smallest possible multi-host `pmap`: doubles a local array on each host's devices, showing that the same `pmap` call is executed on every host to form one computation across the whole device mesh.

## `example_multihost_pmap.py`

A small SPMD training loop: params are replicated across local devices, then trained for several steps with `optax.sgd`, averaging gradients and loss across hosts with `pmean` on every step before checking how close the learned weight got to the target.
