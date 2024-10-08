from typing import List, Tuple, Iterable, Union, Optional
from abc import ABC
import jax
import jax.numpy as jnp
from jax import random


def create_gaussians(
    n_source: int,
    n_target: int,
    dimension: int = 2,
    means: Tuple[List[float], List[float]] = ([0, 0], [1, 1]),
    var_source: float = 1.0,
    var_target: float = 1.0,
    seed: int = 0,
):
    rng = jax.random.PRNGKey(seed)
    rngs = random.split(rng, 2)
    source = random.normal(rngs[0], shape=(n_source, dimension)) * var_source + jnp.array(means[0])
    target = random.normal(rngs[1], shape=(n_target, dimension)) * var_target + jnp.array(means[1])
    return source, target


def create_gaussian_split(
    n_source: int,
    n_target: int,
    dimension: int = 2,
    means: Tuple[List[float], List[float], List[float]] = ([0, 0], [1, 0.2], [1, -0.2]),
    var_source: float = 1.0,
    var_target: float = 1.0,
    p: Tuple[float, float] = [0.5, 0.5],
    seed: int = 0,
):
    rng = jax.random.PRNGKey(seed)
    rngs = random.split(rng, 4)
    source = random.normal(rngs[0], shape=(n_source, dimension)) * var_source + jnp.array(means[0])

    n_target_1 = jnp.sum(random.choice(rngs[1], a=2, p=jnp.array(p), shape=(n_target,)))
    n_target_2 = n_target - n_target_1
    target_1 = random.normal(rngs[2], shape=(n_target_1, dimension)) * var_target + jnp.array(means[1])
    target_2 = random.normal(rngs[3], shape=(n_target_2, dimension)) * var_target + jnp.array(means[2])
    return source, jnp.concatenate([target_1, target_2], axis=0)

class BaseSampler(ABC):
    pass


class DataLoader(BaseSampler):
    def __init__(self, data: Optional[jnp.ndarray] = None, batch_size: int = 64) -> None:
        super().__init__()
        self.data = data
        self.batch_size = batch_size

    def __call__(self, key: jax.random.KeyArray) -> jnp.ndarray:
        inds = jax.random.choice(key, len(self.data), shape=[self.batch_size])
        return self.data[inds, :]


class MixtureNormalSampler(BaseSampler):
    def __init__(
        self, rng: jax.random.KeyArray, centers: Iterable[int], dim: int, std: float = 1.0, batch_size: int = 64, weights: Optional[List[float]] = None,
    ) -> None:
        super().__init__()
        if weights is not None:
            assert len(centers) == len(weights)
            self.weights = jnp.asarray(weights)
        else:
            self.weights = None
        self.batch_size = batch_size
        self.centers = jnp.array(centers)
        self.dim = dim
        self.std = std
        self.rng = rng

    def __iter__(self):
        return self

    def __next__(self) -> jnp.ndarray:
        self.rng, rng = jax.random.split(self.rng, 2)
        if len(self.centers) > 1:
            comps_idx = jax.random.choice(
                rng, len(self.centers), p=self.weights, shape=(self.batch_size,)
            ).astype(int)
        else:
            comps_idx = jnp.zeros(
                self.batch_size,
            ).astype(int)
        if self.dim > 1:
            std_normal = jax.random.normal(rng, (self.batch_size, self.dim))
        else:
            std_normal = jax.random.normal(rng, (self.batch_size,))
        self.centers[comps_idx]
        return jnp.reshape(std_normal * self.std + self.centers[comps_idx], (self.batch_size, self.dim))

