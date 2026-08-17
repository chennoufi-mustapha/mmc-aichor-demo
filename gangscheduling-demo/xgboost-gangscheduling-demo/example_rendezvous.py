import os

import numpy as np
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
        print(f"rank {rank}: joined the collective, world_size={collective.get_world_size()}")

        message = collective.broadcast(f"hello from rank {rank}" if rank == 0 else None, 0)
        print(f"rank {rank}: received broadcast '{message}'")

        contribution = np.array([float(rank)])
        total = collective.allreduce(contribution, collective.Op.SUM)
        expected = world_size * (world_size - 1) // 2
        print(f"rank {rank}: allreduce sum={total[0]} expected={expected}")

    if tracker is not None:
        tracker.wait_for()


if __name__ == "__main__":
    main()
