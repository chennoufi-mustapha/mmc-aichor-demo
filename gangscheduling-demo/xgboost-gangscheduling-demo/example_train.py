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
        x_train = rng.normal(size=(256, 8))
        y_train = (x_train.sum(axis=1) > 0).astype(int)
        x_test = rng.normal(size=(64, 8))
        y_test = (x_test.sum(axis=1) > 0).astype(int)

        dtrain = xgb.DMatrix(x_train, label=y_train)
        dtest = xgb.DMatrix(x_test, label=y_test)

        booster = xgb.train(
            {"objective": "binary:logistic", "max_depth": 3, "eta": 0.3},
            dtrain,
            num_boost_round=20,
        )

        preds = (booster.predict(dtest) > 0.5).astype(int)
        local_accuracy = float((preds == y_test).mean())

        accuracies = np.array([local_accuracy])
        global_accuracy = collective.allreduce(accuracies, collective.Op.SUM) / world_size
        print(f"rank {rank}: local_accuracy={local_accuracy:.4f} global_accuracy={global_accuracy[0]:.4f}")

        if rank == 0:
            booster.save_model("/tmp/model.json")
            print("rank 0: saved final model checkpoint")

    if tracker is not None:
        tracker.wait_for()


if __name__ == "__main__":
    main()
