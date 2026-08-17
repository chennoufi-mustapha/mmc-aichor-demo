import os

import numpy as np
import ray

NUM_WORKERS = 4
NUM_ROUNDS = 15
NUM_FEATURES = 4
LEARNING_RATE = 0.1


@ray.remote
class Worker:
    def __init__(self, rank, num_features, seed):
        self.rank = rank
        rng = np.random.default_rng(seed)
        self.x = rng.normal(size=(32, num_features))
        self.true_w = np.ones(num_features)
        self.y = self.x @ self.true_w

    def compute_gradient(self, weight):
        error = self.x @ weight - self.y
        return self.x.T @ error / len(self.x)

    def compute_loss(self, weight):
        error = self.x @ weight - self.y
        return float(np.mean(error ** 2))


def main():
    ray.init(address=os.environ.get("RAY_ADDRESS", "auto"))

    workers = [Worker.remote(rank=i, num_features=NUM_FEATURES, seed=i) for i in range(NUM_WORKERS)]
    print(f"all {NUM_WORKERS} workers are alive")

    weight = np.zeros(NUM_FEATURES)

    for round_idx in range(NUM_ROUNDS):
        gradients = ray.get([w.compute_gradient.remote(weight) for w in workers])
        averaged_gradient = sum(gradients) / len(workers)
        weight = weight - LEARNING_RATE * averaged_gradient

        losses = ray.get([w.compute_loss.remote(weight) for w in workers])
        print(f"round {round_idx}: averaged_loss={np.mean(losses):.4f} weight={np.round(weight, 3)}")

    print(f"final weight={weight}")


if __name__ == "__main__":
    main()
