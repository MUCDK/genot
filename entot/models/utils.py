from typing import Any
from ott.solvers.nn.models import MLP
import jax.numpy as jnp
import jax
from typing import Dict

class DataLoader:
    def __init__(self, data: jnp.ndarray, batch_size) -> None:
        self.data = data
        self.batch_size= batch_size

    def __call__(self, key: jax.random.KeyArray) -> jnp.ndarray:
        inds = jax.random.choice(key, len(self.data), shape=[self.batch_size])
        return self.data[inds,:]


