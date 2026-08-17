import os

import ray
from ray.util.placement_group import placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

NUM_WORKERS = 4
NUM_ROUNDS = 3


@ray.remote
class Coordinator:
    def __init__(self, num_workers):
        self.num_workers = num_workers
        self.weight = 0.0
        self.history = []

    def aggregate(self, round_idx, contributions):
        assert len(contributions) == self.num_workers
        averaged = sum(contributions) / self.num_workers
        self.weight += averaged
        self.history.append((round_idx, averaged, self.weight))
        return self.weight

    def get_history(self):
        return self.history


@ray.remote
class Worker:
    def __init__(self, rank):
        self.rank = rank

    def contribute(self, round_idx):
        return self.rank + round_idx


def main():
    ray.init(address=os.environ.get("RAY_ADDRESS", "auto"))

    bundles = [{"CPU": 1}] + [{"CPU": 1} for _ in range(NUM_WORKERS)]
    pg = placement_group(bundles, strategy="STRICT_SPREAD")
    print(f"waiting for a gang of {len(bundles)} bundles (1 coordinator + {NUM_WORKERS} workers) to be placed together")
    ray.get(pg.ready())
    print("gang complete, placement group is fully reserved")

    coordinator = Coordinator.options(
        scheduling_strategy=PlacementGroupSchedulingStrategy(placement_group=pg, placement_group_bundle_index=0)
    ).remote(num_workers=NUM_WORKERS)

    workers = [
        Worker.options(
            scheduling_strategy=PlacementGroupSchedulingStrategy(placement_group=pg, placement_group_bundle_index=i + 1)
        ).remote(rank=i)
        for i in range(NUM_WORKERS)
    ]

    for round_idx in range(NUM_ROUNDS):
        contributions = ray.get([w.contribute.remote(round_idx) for w in workers])
        weight = ray.get(coordinator.aggregate.remote(round_idx, contributions))
        print(f"round {round_idx}: contributions={contributions} weight={weight}")

    history = ray.get(coordinator.get_history.remote())
    print(f"full run history: {history}")

    ray.util.remove_placement_group(pg)


if __name__ == "__main__":
    main()
