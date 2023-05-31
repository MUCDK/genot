from typing import Any, Callable, Dict, Iterable, Sequence, Tuple, Union, Optional

import flax.linen as nn
import jax
import jax.numpy as jnp
import optax
from ott.solvers.nn.models import ModelBase, NeuralTrainState
import jax
import jax.numpy as jnp
from ott.geometry import pointcloud
from ott.solvers.linear import sinkhorn
from ott.problems.linear import linear_problem
import ott.geometry.costs as costs
from typing import Any, Mapping, Callable
from types import MappingProxyType
from abc import ABC, abstractmethod

class BaseSampler(ABC):
    pass

class DataLoader(BaseSampler):
    def __init__(self, data: Optional[jnp.ndarray]=None, batch_size: int = 64) -> None:
        super().__init__()
        self.data = data
        self.batch_size= batch_size

    def __call__(self, key: jax.random.KeyArray) -> jnp.ndarray:
        inds = jax.random.choice(key, len(self.data), shape=[self.batch_size])
        return self.data[inds,:]

class MixtureNormalSampler(BaseSampler):
    def __init__(self, centers: Iterable[int], dim: int, var: float = 1.0, batch_size: int = 64) -> None:
        super().__init__()
        self.batch_size=batch_size
        self.centers = jnp.array(centers)
        self.dim = dim
        self.var = var

    def __call__(self, key: jax.random.KeyArray) -> jnp.ndarray:
        if len(self.centers)>1:
            comps_idx = jax.random.categorical(key, jnp.repeat(jnp.log(1.0/len(self.centers)), len(self.centers)), shape=(self.batch_size,)).astype(int)
        else:
            comps_idx = jnp.zeros(self.batch_size,).astype(int)
        if self.dim > 1:
            std_normal = jax.random.normal(key, (self.batch_size,self.dim)) 
        else: 
            std_normal = jax.random.normal(key, (self.batch_size,))
        self.centers[comps_idx]
        return jnp.reshape(std_normal * self.var + self.centers[comps_idx], (self.batch_size,self.dim))
        



        


def _concatenate(a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
    return jnp.concatenate((jnp.atleast_2d(a), jnp.atleast_2d(b)),axis=1)




class MLP(ModelBase):

    dim_hidden: Sequence[int]
    is_potential: bool = True
    act_fn: Callable[[jnp.ndarray], jnp.ndarray] = nn.leaky_relu
    noise_dim: int = 0
    output_dim: Optional[int] = None

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
        elif self.output_dim is None:
            Wx = nn.Dense(n_input-self.noise_dim, use_bias=True)
            z = Wx(z)
        else:
            Wx = nn.Dense(self.output_dim, use_bias=True)
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
    


class KantorovichGap:
    def __init__(
        self,
        noise_dim: int,
        geometry_kwargs: Mapping[str, Any] = MappingProxyType({}),
        sinkhorn_kwargs: Mapping[str, Any] = MappingProxyType({}),
    ) -> None:
        self.geometry_kwargs = geometry_kwargs
        self.sinkhorn_kwargs = sinkhorn_kwargs
        self.noise_dim = noise_dim

    def __call__(
        self, source: jnp.ndarray, T: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]
    ) -> Union[float, Any]:
        T_xz = T(source)
        
        geom = pointcloud.PointCloud(
            x=jnp.reshape(source[..., :-self.noise_dim], (source.shape[0], -1)), y=jnp.reshape(T_xz, (T_xz.shape[0],-1)),
            **self.geometry_kwargs
        )
        
        id_displacement = jnp.mean(
            jax.vmap(self.cost_fn)(source[..., :-self.noise_dim], T_xz)
        )
        sinkhorn_output = sinkhorn.Sinkhorn(
            **self.sinkhorn_kwargs
        )(
            linear_problem.LinearProblem(geom)
        )
        
        opt_displacement = sinkhorn_output.reg_ot_cost - 2 * geom.epsilon * jnp.log(len(source)) # use Shannon entropy instead of relative entropy as entropic regularizer to ensure Monge gap positivity

        return id_displacement - opt_displacement, sinkhorn_output

    @property
    def cost_fn(self) -> costs.CostFn:
        """"
        Set cost function on which Monge gap is instanciated.
        Default is squared euclidean.
        """
        if "cost_fn" in self.geometry_kwargs:
            return self.geometry_kwargs["cost_fn"]
        else:
            return costs.SqEuclidean()
