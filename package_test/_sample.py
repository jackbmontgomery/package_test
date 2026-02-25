import blackjax
import jax
import jax.numpy as jnp
import jax.random as jr
import jax.scipy.stats as stats
import numpy as np

observed = np.random.normal(10, 20, size=1_000)


def logdensity_fn(x):
    logpdf = stats.norm.logpdf(observed, x["loc"], x["scale"])
    return jnp.sum(logpdf)


def run_sampling():
    key = jr.key(42)
    sigma = jnp.array([0.1, 0.01])
    random_walk = blackjax.additive_step_random_walk(
        logdensity_fn, blackjax.mcmc.random_walk.normal(sigma)
    )

    # Initialize the state
    initial_position = {"loc": 1.0, "scale": 2.0}
    state = random_walk.init(initial_position, key)

    # Iterate
    rng_key = jax.random.key(0)
    step = jax.jit(random_walk.step)
    for i in range(100):
        nuts_key = jax.random.fold_in(rng_key, i)
        state, _ = step(nuts_key, state)

    return state.position
