import os

import ray

NUM_WORKERS = 4


@ray.remote
class Worker:
    def ping(self):
        return True


def main():
    ray.init(address=os.environ.get("RAY_ADDRESS", "auto"))

    workers = [Worker.remote() for _ in range(NUM_WORKERS)]
    ray.get([w.ping.remote() for w in workers])

    print(f"all {NUM_WORKERS} workers are alive")


if __name__ == "__main__":
    main()
