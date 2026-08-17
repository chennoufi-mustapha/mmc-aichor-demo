import os

import jax
import jax.numpy as jnp
import optax


def init_params(key, num_features):
    return {"w": jax.random.normal(key, (num_features,)), "b": jnp.zeros(())}


def loss_fn(params, x, y):
    preds = x @ params["w"] + params["b"]
    return jnp.mean((preds - y) ** 2)


def make_train_step(tx):
    def train_step(params, opt_state, x, y):
        loss, grads = jax.value_and_grad(loss_fn)(params, x, y)
        grads = jax.lax.pmean(grads, axis_name="hosts")
        loss = jax.lax.pmean(loss, axis_name="hosts")
        updates, opt_state = tx.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss

    return jax.pmap(train_step, axis_name="hosts")


def main():
    jax.distributed.initialize(
        coordinator_address=os.environ["JAXOPERATOR_COORDINATOR_ADDRESS"],
        num_processes=int(os.environ["JAXOPERATOR_NUM_PROCESSES"]),
        process_id=int(os.environ["JAXOPERATOR_PROCESS_ID"]),
    )

    num_features = 8
    local_device_count = jax.local_device_count()

    key = jax.random.PRNGKey(0)
    params = init_params(key, num_features)
    params = jax.tree_util.tree_map(lambda v: jnp.broadcast_to(v, (local_device_count,) + v.shape), params)

    tx = optax.sgd(0.05)
    opt_state = tx.init(params)

    data_key = jax.random.PRNGKey(jax.process_index())
    x = jax.random.normal(data_key, (local_device_count, 64, num_features))
    true_w = jnp.ones(num_features)
    y = x @ true_w

    train_step = make_train_step(tx)

    num_steps = 10
    for step in range(num_steps):
        params, opt_state, loss = train_step(params, opt_state, x, y)
        if step % 2 == 0:
            print(f"process {jax.process_index()}: step {step} global_loss={loss[0]:.4f}")

    final_w = jax.tree_util.tree_map(lambda v: v[0], params)["w"]
    print(f"process {jax.process_index()}: final weight error={jnp.abs(final_w - true_w).mean():.4f}")


if __name__ == "__main__":
    main()
