from typing import List, Tuple

import jax
import jax.numpy as jnp
from jax import random


def create_gaussians(n_source: int, n_target: int, dimension: int = 2, means: Tuple[List[float], List[float]] = ([0,0], [1,1]), var_source: float = 1.0, var_target: float = 1.0, seed: int = 0):
    rng = jax.random.PRNGKey(seed)
    rngs = random.split(rng, 2)
    source = random.normal(rngs[0], shape=(n_source, dimension)) * var_source + jnp.array(means[0])
    target = random.normal(rngs[1], shape=(n_target, dimension)) * var_target + jnp.array(means[1])
    return source, target


def create_gaussian_split(n_source: int, n_target: int, dimension: int = 2, means: Tuple[List[float], List[float], List[float]] = ([0,0], [1,0.2], [1,-0.2]), var_source: float = 1.0, var_target: float = 1.0, p: Tuple[float, float] = [0.5, 0.5],seed: int = 0):
    rng = jax.random.PRNGKey(seed)
    rngs = random.split(rng, 4)
    source = random.normal(rngs[0], shape=(n_source, dimension)) * var_source + jnp.array(means[0])
    
    n_target_1 = jnp.sum(random.choice(rngs[1], a=2, p=jnp.array(p), shape=(n_target,)))
    n_target_2 = n_target - n_target_1
    target_1 = random.normal(rngs[2], shape=(n_target_1, dimension)) * var_target + jnp.array(means[1])
    target_2 = random.normal(rngs[3], shape=(n_target_2, dimension)) * var_target + jnp.array(means[2])
    return source, jnp.concatenate([target_1, target_2], axis=0)