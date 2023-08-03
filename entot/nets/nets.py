from typing import Any, Callable, Tuple, Union
import flax.linen as nn
import jax
import jax.numpy as jnp
import optax 
from flax.training import train_state
from ott.solvers.nn.models import ModelBase, NeuralTrainState


class Block(nn.Module):
    dim: int = 128
    num_layers: int = 3
    activation_fn: Any = nn.silu
    out_dim: int = 32

    @nn.compact
    def __call__(self, x):
        for i in range(self.num_layers):
            x = nn.Dense(self.dim, name="fc{0}".format(i))(x)
            x = self.activation_fn(x)
        x = nn.Dense(self.out_dim, name="fc_final")(x)
        return x


class MLP_vector_field(ModelBase):
    output_dim: int
    t_embed_dim: int
    condition_embed_dim: int
    joint_hidden_dim: int
    is_potential: bool = False
    act_fn: Callable[[jnp.ndarray], jnp.ndarray] = nn.silu
    latent_dim: int = 0
    n_frequencies: int = 1

    def time_encoder(self, t: jnp.array) -> jnp.array:
        freq = 2 * jnp.arange(self.n_frequencies) * jnp.pi
        t = freq * t  # [..., None]
        return jnp.concatenate((jnp.cos(t), jnp.sin(t)), axis=-1)

    @nn.compact
    def __call__(self, t: float, condition: jnp.ndarray, latent: jnp.ndarray) -> jnp.ndarray:  # noqa: D102
        t = jnp.full(shape=(len(condition), 1), fill_value=t)
        t = self.time_encoder(t)
        t = Block(dim=self.t_embed_dim, out_dim=self.t_embed_dim, activation_fn=self.act_fn)(t)

        condition = Block(dim=self.condition_embed_dim, out_dim=self.output_dim, activation_fn=self.act_fn)(condition)
        latent = Block(dim=self.condition_embed_dim, out_dim=self.output_dim, activation_fn=self.act_fn)(latent)

        z = jnp.concatenate((condition, t, latent), axis=-1)
        z = Block(num_layers=3, dim=self.joint_hidden_dim, out_dim=self.joint_hidden_dim, activation_fn=self.act_fn)(z)

        Wx = nn.Dense(self.output_dim, use_bias=True, name="final_layer")
        z = Wx(z)

        return z

    def create_train_state(
        self, rng: jax.random.PRNGKeyArray, optimizer: optax.OptState, source_dim: int, latent_dim: int, **kwargs: Any
    ) -> NeuralTrainState:
        params = self.init(rng, jnp.ones((1, 1)), jnp.ones((1, source_dim)), jnp.ones((1, latent_dim)))["params"]
        return train_state.TrainState.create(apply_fn=self.apply, params=params, tx=optimizer)


class Bridge_MLP_mean(ModelBase):
    output_dim: int
    t_embed_dim: int
    condition_embed_dim: int
    is_potential: bool = False
    act_fn: Callable[[jnp.ndarray], jnp.ndarray] = nn.silu

    @nn.compact
    def __call__(self, condition: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:  # noqa: D102
        condition = Block(dim=self.condition_embed_dim, out_dim=self.condition_embed_dim, activation_fn=self.act_fn)(
            condition
        )
        mu = Block(dim=self.condition_embed_dim, out_dim=self.output_dim)(condition)
        return mu, jnp.ones_like(mu)

    def create_train_state(
        self,
        rng: jax.random.PRNGKeyArray,
        optimizer: optax.OptState,
        input_dim: int,
    ) -> NeuralTrainState:
        params = self.init(rng, jnp.ones((1, input_dim)))["params"]
        return train_state.TrainState.create(apply_fn=self.apply, params=params, tx=optimizer)


class Bridge_MLP_full(ModelBase):
    output_dim: int
    t_embed_dim: int
    condition_embed_dim: int
    is_potential: bool = False
    act_fn: Callable[[jnp.ndarray], jnp.ndarray] = nn.silu

    @nn.compact
    def __call__(self, condition: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:  # noqa: D102
        condition = Block(dim=self.condition_embed_dim, out_dim=self.condition_embed_dim, activation_fn=self.act_fn)(
            condition
        )
        mu = Block(dim=self.condition_embed_dim, out_dim=self.output_dim)(condition)
        log_var = Block(dim=self.condition_embed_dim, out_dim=self.output_dim)(condition)
        return mu, jnp.sqrt(jnp.exp(log_var))

    def create_train_state(
        self,
        rng: jax.random.PRNGKeyArray,
        optimizer: optax.OptState,
        output_dim: int,
    ) -> NeuralTrainState:
        params = self.init(rng, jnp.ones((1, output_dim)))["params"]
        return train_state.TrainState.create(apply_fn=self.apply, params=params, tx=optimizer)


class Bridge_MLP_constant(ModelBase):
    output_dim: int
    t_embed_dim: int
    condition_embed_dim: int
    is_potential: bool = False
    act_fn: Callable[[jnp.ndarray], jnp.ndarray] = nn.silu

    @nn.compact
    def __call__(self, condition: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:  # noqa: D102
        return jnp.zeros((*condition.shape[:-1], self.output_dim)), jnp.ones((*condition.shape[:-1], self.output_dim))

    def create_train_state(
        self,
        rng: jax.random.PRNGKeyArray,
        optimizer: optax.OptState,
        input_dim: int,
    ) -> NeuralTrainState:
        params = self.init(rng, jnp.ones((1, input_dim)))["params"]
        return train_state.TrainState.create(apply_fn=self.apply, params=params, tx=optimizer)


class MLP_marginal(ModelBase):
    output_dim: int
    hidden_dim: int
    is_potential: bool = False
    act_fn: Callable[[jnp.ndarray], jnp.ndarray] = nn.selu

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:  # noqa: D102
        x = Block(dim=self.hidden_dim, out_dim=self.hidden_dim, activation_fn=self.act_fn)(x)
        Wx = nn.Dense(self.output_dim, use_bias=True, name="final_layer")
        z = Wx(x)
        return jnp.exp(z)

    def create_train_state(
        self,
        rng: jax.random.PRNGKeyArray,
        optimizer: optax.OptState,
        input_dim: int,
    ) -> NeuralTrainState:
        params = self.init(rng, jnp.ones((1, input_dim)))["params"]
        return train_state.TrainState.create(apply_fn=self.apply, params=params, tx=optimizer)
