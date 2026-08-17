import os

import ray
from ray.util.placement_group import placement_group

NUM_WORKERS = 4


def main():
    ray.init(address=os.environ.get("RAY_ADDRESS", "auto"))

    pg = placement_group([{"CPU": 1} for _ in range(NUM_WORKERS)], strategy="STRICT_SPREAD")
    ray.get(pg.ready())

    print(f"placement group with {NUM_WORKERS} bundles is fully reserved")
    ray.util.remove_placement_group(pg)


if __name__ == "__main__":
    main()
