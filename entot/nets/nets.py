from typing import Any, Tuple, Union, Callable

import flax.linen as nn
import jax
import jax.numpy as jnp
import optax
from ott.solvers.nn.models import ModelBase, NeuralTrainState


class GNOT_MLP(ModelBase):
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
        variance = Block(dim=self.condition_embed_dim, out_dim=1, activation_fn=self.act_fn)(jnp.concatenate((condition, t), axis=-1))

        condition = condition + latent * variance

        z = jnp.concatenate((condition, t), axis=-1)
        z = Block(num_layers=3, dim=self.joint_hidden_dim, out_dim=self.joint_hidden_dim, activation_fn=self.act_fn)(z)

        Wx = nn.Dense(self.output_dim, use_bias=True, name="final_layer")
        z = Wx(z)

        return z



class Block(nn.Module):
    dim: int = 32
    groups: int = 4
    activation_fn: Any = nn.relu

    @nn.compact
    def __call__(self, inputs):
        conv = nn.Conv(self.dim, (3, 3))(inputs)
        norm = nn.GroupNorm(num_groups=self.groups)(conv)
        activation = self.activation_fn(norm)
        return activation


class ResnetBlock(nn.Module):
    dim: int = 32
    groups: int = 8
    activation_fn: Any = nn.relu

    @nn.compact
    def __call__(self, inputs):
        x = Block(self.dim, self.groups, self.activation_fn)(inputs)
        x = Block(self.dim, self.groups, self.activation_fn)(x)
        res_conv = nn.Conv(self.dim, (1, 1), padding="SAME")(inputs)
        return res_conv + x


class UResNet(ModelBase):  # adapted from https://www.kaggle.com/code/darshan1504/exploring-diffusion-models-with-jax
    base_factor: int = 32
    dim_scale_factor: tuple = (1, 2, 4, 8)
    num_groups: int = 4
    diff_input_output: int = 0
    use_sigmoid: bool = False
    activation_fn: Any = nn.relu

    @nn.compact
    def __call__(self, inputs):
        channels = inputs.shape[-1]
        x = nn.Conv(self.base_factor, (3, 3))(inputs)

        dims = [self.base_factor * i for i in self.dim_scale_factor]
        pre_downsampling = []

        # Downsampling phase
        for index, dim in enumerate(dims):
            x = ResnetBlock(dim, self.num_groups, self.activation_fn)(x)
            x = ResnetBlock(dim, self.num_groups, self.activation_fn)(x)
            norm = nn.GroupNorm(self.num_groups)(x)
            x = norm + x
            # Saving this output for residual connection with the upsampling layer
            pre_downsampling.append(x)
            if index != len(dims) - 1:
                x = nn.Conv(dim, (4, 4), (2, 2))(x)

        # Middle block
        x = ResnetBlock(dims[-1], self.num_groups, self.activation_fn)(x)
        norm = nn.GroupNorm(self.num_groups)(x)
        x = norm + x
        x = ResnetBlock(dims[-1], self.num_groups, self.activation_fn)(x)

        # Upsampling phase
        for index, dim in enumerate(reversed(dims)):
            x = jnp.concatenate([pre_downsampling.pop(), x], -1)
            x = ResnetBlock(dim, self.num_groups, self.activation_fn)(x)
            x = ResnetBlock(dim, self.num_groups, self.activation_fn)(x)
            norm = nn.GroupNorm(self.num_groups)(x)
            x = norm + x
            if index != len(dims) - 1:
                x = nn.ConvTranspose(dim, (4, 4), (2, 2))(x)

        # Final ResNet block and output convolutional layer
        x = ResnetBlock(dim, self.num_groups, self.activation_fn)(x)
        x = nn.Conv(channels - self.diff_input_output, (1, 1), padding="SAME")(x)
        return nn.sigmoid(x) if self.use_sigmoid else x

    def is_potential(self) -> bool:
        return False

    def create_train_state(
        self,
        rng: jax.random.PRNGKeyArray,
        optimizer: optax.OptState,
        input: Union[int, Tuple[int, ...]],
        **kwargs: Any,
    ) -> NeuralTrainState:
        """Create initial training state."""
        params = self.init(rng, jnp.ones(input))["params"]

        return NeuralTrainState.create(
            apply_fn=self.apply,
            params=params,
            tx=optimizer,
            potential_value_fn=self.potential_value_fn,
            potential_gradient_fn=self.potential_gradient_fn,
            **kwargs,
        )


class UNet(ModelBase):  # adapted from https://www.kaggle.com/code/darshan1504/exploring-diffusion-models-with-jax
    base_factor: int = 32
    dim_scale_factor: tuple = (1, 2, 4, 8)
    num_groups: int = 4
    diff_input_output: int = 0
    use_sigmoid: bool = False
    activation_fn: Any = nn.relu

    @nn.compact
    def __call__(self, inputs):
        channels = inputs.shape[-1]
        x = nn.Conv(self.base_factor, (3, 3))(inputs)
        dims = [self.base_factor * i for i in self.dim_scale_factor]
        pre_downsampling = []

        # Downsampling phase
        for index, dim in enumerate(dims):
            x = Block(dim, self.num_groups, self.activation_fn)(x)
            x = Block(dim, self.num_groups, self.activation_fn)(x)
            norm = nn.GroupNorm(self.num_groups)(x)
            x = norm + x
            # Saving this output for residual connection with the upsampling layer
            pre_downsampling.append(x)
            if index != len(dims) - 1:
                x = nn.Conv(dim, (4, 4), (2, 2))(x)

        # Middle block
        x = Block(dims[-1], self.num_groups, self.activation_fn)(x)
        norm = nn.GroupNorm(self.num_groups)(x)
        x = norm + x
        x = Block(dims[-1], self.num_groups, self.activation_fn)(x)
        # Upsampling phase
        for index, dim in enumerate(reversed(dims)):
            x = jnp.concatenate([pre_downsampling.pop(), x], -1)
            x = Block(dim, self.num_groups, self.activation_fn)(x)
            x = Block(dim, self.num_groups, self.activation_fn)(x)
            norm = nn.GroupNorm(self.num_groups)(x)
            x = norm + x
            if index != len(dims) - 1:
                x = nn.ConvTranspose(dim, (4, 4), (2, 2))(x)

        # Final block and output convolutional layer
        x = Block(dim, self.num_groups, self.activation_fn)(x)
        x = nn.Conv(channels - self.diff_input_output, (1, 1), padding="SAME")(x)
        return nn.sigmoid(x) if self.use_sigmoid else x

    def is_potential(self) -> bool:
        return False

    def create_train_state(
        self,
        rng: jax.random.PRNGKeyArray,
        optimizer: optax.OptState,
        input: Union[int, Tuple[int, ...]],
        **kwargs: Any,
    ) -> NeuralTrainState:
        """Create initial training state."""
        params = self.init(rng, jnp.ones(input))["params"]

        return NeuralTrainState.create(
            apply_fn=self.apply,
            params=params,
            tx=optimizer,
            potential_value_fn=self.potential_value_fn,
            potential_gradient_fn=self.potential_gradient_fn,
            **kwargs,
        )
