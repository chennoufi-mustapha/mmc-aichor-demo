import os

import numpy as np
import xgboost as xgb
from xgboost import collective
from xgboost.tracker import RabitTracker


def main():
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    master_addr = os.environ["MASTER_ADDR"]
    master_port = int(os.environ["MASTER_PORT"])

    tracker = None
    if rank == 0:
        tracker = RabitTracker(n_workers=world_size, host_ip=master_addr, port=master_port)
        tracker.start()

    with collective.CommunicatorContext(
        dmlc_tracker_uri=master_addr,
        dmlc_tracker_port=master_port,
        dmlc_task_id=str(rank),
    ):
        rng = np.random.default_rng(rank)
        x = rng.normal(size=(64, 4))
        y = (x.sum(axis=1) > 0).astype(int)
        dtrain = xgb.DMatrix(x, label=y)

        booster = xgb.train({"objective": "binary:logistic", "max_depth": 2}, dtrain, num_boost_round=5)
        print(f"rank {rank}: trained {booster.num_boosted_rounds()} rounds")

    if tracker is not None:
        tracker.wait_for()


if __name__ == "__main__":
    main()
