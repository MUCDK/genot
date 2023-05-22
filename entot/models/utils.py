from typing import Any, Union, Tuple, Iterable
from ott.solvers.nn.models import ModelBase, NeuralTrainState
from typing import Sequence, Callable
import flax.linen as nn
import jax.numpy as jnp
import jax
import optax
from typing import Dict

class DataLoader:
    def __init__(self, data: jnp.ndarray, batch_size) -> None:
        self.data = data
        self.batch_size= batch_size

    def __call__(self, key: jax.random.KeyArray) -> jnp.ndarray:
        inds = jax.random.choice(key, len(self.data), shape=[self.batch_size])
        return self.data[inds,:]

class MixtureNormalSampler:
    def __init__(self, centers: Iterable[int], dim: int, batch_size: int, var: float = 1.0) -> None:
        self.centers = jnp.array(centers)
        self.dim = dim
        self.var = var
        self.batch_size = batch_size

    def __call__(self, key: jax.random.KeyArray) -> jnp.ndarray:
        if len(self.centers)>1:
            comps_idx = jax.random.categorical(key, jnp.repeat(jnp.log(1.0/len(self.centers)), len(self.centers)), shape=(self.batch_size,)).astype(int)
        else:
            comps_idx = jnp.zeros(self.batch_size,).astype(int)
        if self.dim > 1:
            std_normal = jax.random.normal(key, (self.batch_size,self.dim)) 
        else: 
            std_normal = jax.random.normal(key, (self.batch_size,))
        return std_normal * self.var + self.centers[comps_idx]
        



        


def _concatenate(a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
    return jnp.concatenate((jnp.atleast_2d(a), jnp.atleast_2d(b)),axis=1)




class MLP(ModelBase):

    dim_hidden: Sequence[int]
    is_potential: bool = True
    act_fn: Callable[[jnp.ndarray], jnp.ndarray] = nn.leaky_relu
    noise_dim: int = 0

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:  # noqa: D102
        squeeze = x.ndim == 1
        if squeeze:
            x = jnp.expand_dims(x, 0)
        assert x.ndim == 2, x.ndim
        n_input = x.shape[-1]

        z = x
        for n_hidden in self.dim_hidden:
            Wx = nn.Dense(n_hidden, use_bias=True)
            z = self.act_fn(Wx(z))

        if self.is_potential:
            Wx = nn.Dense(1, use_bias=True)
            z = Wx(z).squeeze(-1)

            quad_term = 0.5 * jax.vmap(jnp.dot)(x, x)
            z += quad_term
        else:
            Wx = nn.Dense(n_input-self.noise_dim, use_bias=True)
            z = Wx(z)

        return z.squeeze(0) if squeeze else z
    
    def create_train_state(
        self,
        rng: jax.random.PRNGKeyArray,
        optimizer: optax.OptState,
        input: Union[int, Tuple[int, ...]],
        **kwargs: Any,
    ) -> NeuralTrainState:
        params = self.init(rng, jnp.ones(input))["params"]

        return NeuralTrainState.create(
            apply_fn=self.apply,
            params=params,
            tx=optimizer,
            potential_value_fn=self.potential_value_fn,
            potential_gradient_fn=self.potential_gradient_fn,
            **kwargs,
        )