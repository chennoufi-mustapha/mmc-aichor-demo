# Ray (kuberay) gang scheduling examples

Ray's own explicit gang scheduling primitive is the `PlacementGroup`: `pg.ready()` blocks until every bundle in the group has been placed together, so either the cluster has room for the whole group or none of it gets scheduled. The actor-based examples show the same requirement without a placement group, by having actors that must all be alive to complete a synchronous round.

Note this only applies to the *fixed* set of actors reserved up front. Workers added later by the cluster autoscaler don't need to go through the same all-or-nothing group, since each one is useful as soon as it lands (see the top-level README).

## Environment variables

Unlike the other operators, KubeRay does not rely on environment variables to set up the distribution itself: the head and workers are already connected into one Ray cluster, via the `ray start` commands the platform runs on every container, before `spec.command` starts. The variables below are only there for your code to connect to (or identify) that already-running cluster.

The head container gets:

| Variable | Description | Example |
|----------|-------------|---------|
| `RAY_CLUSTER_NAME` | Name of the RayCluster this pod belongs to. | `experiment-6f4a850d-2281-raycluster-q5wqj` |

The worker containers get:

| Variable | Description | Example |
|----------|-------------|---------|
| `RAY_CLUSTER_NAME` | Name of the RayCluster this pod belongs to. | `experiment-6f4a850d-2281-raycluster-q5wqj` |
| `RAY_NODE_TYPE_NAME` | Name of the worker group this pod belongs to. | `cpu-workers` |
| `KUBERAY_GEN_RAY_START_CMD` | The `ray start` command used to start Ray on this worker. Not used by these examples. | `ray start ...` |
| `RAY_ADDRESS` | Head address. Passed to `ray.init(address=...)` so it connects from within the cluster — the only one of these variables these examples read. | `10.43.84.237:6379` |
| `RAY_PORT` | Port the head's GCS listens on. Not used by these examples. | `6379` |

## `example_simple_placement_group.py`

The bare minimum: reserves `NUM_WORKERS` bundles as one atomic unit and confirms the group is fully placed.

## `example_placement_group.py`

A fuller layout that mirrors a head+workers setup: one bundle for a `Coordinator` actor and `NUM_WORKERS` bundles for `Worker` actors, all reserved together in a single placement group. Runs several rounds of workers contributing values that the coordinator aggregates, then prints the full run history.

## `example_simple_actor_barrier.py`

The bare minimum: spawns `NUM_WORKERS` actors and waits for all of them to respond to a ping, showing the "every actor must be alive" requirement without a placement group.

## `example_actor_barrier.py`

A synchronous parameter-server loop: each worker holds its own synthetic linear-regression data, computes a local gradient, and the driver averages gradients across all workers every round to update a shared weight vector until it converges toward the target.
