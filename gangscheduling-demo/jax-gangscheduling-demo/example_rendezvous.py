import os

import jax
import jax.numpy as jnp


def collectives(x):
    total = jax.lax.psum(x, axis_name="hosts")
    mean = jax.lax.pmean(x, axis_name="hosts")
    maximum = jax.lax.pmax(x, axis_name="hosts")
    gathered = jax.lax.all_gather(x, axis_name="hosts")
    return total, mean, maximum, gathered


def main():
    jax.distributed.initialize(
        coordinator_address=os.environ["JAXOPERATOR_COORDINATOR_ADDRESS"],
        num_processes=int(os.environ["JAXOPERATOR_NUM_PROCESSES"]),
        process_id=int(os.environ["JAXOPERATOR_PROCESS_ID"]),
    )
    print(f"process {jax.process_index()}/{jax.process_count()} joined, devices={jax.device_count()}")

    local_value = jnp.ones(jax.local_device_count()) * jax.process_index()
    total, mean, maximum, gathered = jax.pmap(collectives, axis_name="hosts")(local_value)

    print(f"process {jax.process_index()}: psum={total} pmean={mean} pmax={maximum}")
    print(f"process {jax.process_index()}: all_gather shape={gathered.shape} values={gathered[0]}")


if __name__ == "__main__":
    main()
