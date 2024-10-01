from typing import Callable, Optional
import flax.linen as nn
import optax
from flax.training import train_state
import jax
import jax.numpy as jnp
from ott.solvers.nn.models import ModelBase, NeuralTrainState


class Block(nn.Module):
    """
    Block of a neural network.

    Parameters
    ----------
    dim
        Input dimension.
    out_dim
        Output dimension.
    num_layers
        Number of layers.
    act_fn
        Activation function.
    """

    dim: int = 128
    out_dim: int = 32
    num_layers: int = 3
    act_fn: Callable[[jnp.ndarray], jnp.ndarray] = nn.silu

    @nn.compact
    def __call__(self, x):
        for i in range(self.num_layers):
            x = nn.Dense(self.dim, name=f"fc{i}")(x)
            x = self.act_fn(x)
        return nn.Dense(self.out_dim, name="fc_final")(x)


class MLP_vector_field(ModelBase):
    """
    Neural vector field.

    Parameters
    ----------
    output_dim
        Output dimension.
    latent_embed_dim
        Latent embedding dimension.
    condition_embed_dim
        Condition embedding dimension.
    t_embed_dim
        Time embedding dimension.
    joint_hidden_dim
        Joint hidden dimension.
    num_layers
        Number of layers per block.
    act_fn
        Activation function.
    n_frequencies
        Number of frequencies for time embedding.
    """

    output_dim: int
    latent_embed_dim: int
    condition_embed_dim: Optional[int] = None
    t_embed_dim: Optional[int] = None
    joint_hidden_dim: Optional[int] = None
    num_layers: int = 3
    act_fn: Callable[[jnp.ndarray], jnp.ndarray] = nn.silu
    n_frequencies: int = 1

    def time_encoder(self, t: jnp.array) -> jnp.array:
        freq = 2 * jnp.arange(self.n_frequencies) * jnp.pi
        t = freq * t
        return jnp.concatenate((jnp.cos(t), jnp.sin(t)), axis=-1)

    def __post_init__(self):
        if self.condition_embed_dim is None:
            self.condition_embed_dim = self.latent_embed_dim
        if self.t_embed_dim is None:
            self.t_embed_dim = self.latent_embed_dim

        concat_embed_dim = self.latent_embed_dim + self.condition_embed_dim + self.t_embed_dim
        if self.joint_hidden_dim is not None:
            assert self.joint_hidden_dim >= concat_embed_dim, (
                "joint_hidden_dim must be greater than or equal to the sum of " "all embedded dimensions. "
            )
            self.joint_hidden_dim = self.latent_embed_dim
        else:
            self.joint_hidden_dim = concat_embed_dim
        super().__post_init__()

    @property
    def is_potential(self) -> bool:
        return self.output_dim == 1

    @nn.compact
    def __call__(self, t: float, condition: jnp.ndarray, latent: jnp.ndarray) -> jnp.ndarray:
        condition, latent = jnp.atleast_2d(condition, latent)

        t = jnp.full(shape=(len(condition), 1), fill_value=t)
        t = self.time_encoder(t)
        t = Block(
            dim=self.t_embed_dim,
            out_dim=self.t_embed_dim,
            num_layers=self.num_layers,
            act_fn=self.act_fn,
        )(t)

        condition = Block(
            dim=self.condition_embed_dim,
            out_dim=self.condition_embed_dim,
            num_layers=self.num_layers,
            act_fn=self.act_fn,
        )(condition)

        latent = Block(
            dim=self.latent_embed_dim, out_dim=self.latent_embed_dim, num_layers=self.num_layers, act_fn=self.act_fn
        )(latent)

        concat_embed = jnp.concatenate((t, condition, latent), axis=-1)
        out = Block(
            dim=self.joint_hidden_dim,
            out_dim=self.joint_hidden_dim,
            num_layers=self.num_layers,
            act_fn=self.act_fn,
        )(concat_embed)

        return nn.Dense(self.output_dim, use_bias=True, name="final_layer")(out)

    def create_train_state(
        self,
        rng: jax.random.PRNGKeyArray,
        optimizer: optax.OptState,
        input_dim: int,
    ) -> NeuralTrainState:
        params = self.init(rng, jnp.ones((1, 1)), jnp.ones((1, input_dim)), jnp.ones((1, self.output_dim)))["params"]
        return train_state.TrainState.create(apply_fn=self.apply, params=params, tx=optimizer)


class MLP_marginal(ModelBase):
    """
    Neural network parameterizing a reweighting function.

    Parameters
    ----------
    hidden_dim
        Hidden dimension.
    num_layers
        Number of layers.
    act_fn
        Activation function.
    """

    hidden_dim: int
    num_layers: int = 3
    act_fn: Callable[[jnp.ndarray], jnp.ndarray] = nn.silu

    @property
    def is_potential(self) -> bool:
        return True

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        z = x
        z = Block(dim=self.hidden_dim, out_dim=1, num_layers=self.num_layers, act_fn=self.act_fn)(z)
        return nn.softplus(z)
