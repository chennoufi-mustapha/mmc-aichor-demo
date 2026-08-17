import os

import jax


def main():
    jax.distributed.initialize(
        coordinator_address=os.environ["JAXOPERATOR_COORDINATOR_ADDRESS"],
        num_processes=int(os.environ["JAXOPERATOR_NUM_PROCESSES"]),
        process_id=int(os.environ["JAXOPERATOR_PROCESS_ID"]),
    )
    print(f"process {jax.process_index()}/{jax.process_count()} joined, devices={jax.device_count()}")


if __name__ == "__main__":
    main()
