import os

import jax
import jax.numpy as jnp


def main():
    jax.distributed.initialize(
        coordinator_address=os.environ["JAXOPERATOR_COORDINATOR_ADDRESS"],
        num_processes=int(os.environ["JAXOPERATOR_NUM_PROCESSES"]),
        process_id=int(os.environ["JAXOPERATOR_PROCESS_ID"]),
    )

    x = jnp.arange(jax.local_device_count())
    y = jax.pmap(lambda v: v * 2)(x)
    print(f"process {jax.process_index()}: local result={y}")


if __name__ == "__main__":
    main()
