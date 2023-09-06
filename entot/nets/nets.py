# jax, flax, optax
from jax._src.prng import PRNGKeyArray
import jax
import jax.numpy as jnp
from flax.core import frozen_dict
import flax.linen as nn
from flax.training import train_state
import optax 

# typing
from typing import Any, Callable, Literal, Optional, Tuple

# ott
from ott.solvers.nn.models import ModelBase, NeuralTrainState

Shape = Tuple[int]
Dtype = Any

# default initializers 
default_kernel_init = nn.initializers.variance_scaling(
    scale=1., mode='fan_in', distribution='truncated_normal'
)
default_bias_init = nn.initializers.zeros

class Block(nn.Module):
    dim: int = 128
    out_dim: int = 32
    num_layers: int = 3
    act_fn: Callable[[jnp.ndarray], jnp.ndarray] = nn.silu
    kernel_init: Callable[
        [PRNGKeyArray, Shape, Dtype],
        jnp.ndarray
    ] = default_kernel_init
    bias_init: Callable[
        [PRNGKeyArray, Shape, Dtype],
        jnp.ndarray
    ] = default_bias_init

    @nn.compact
    def __call__(self, x):
        for i in range(self.num_layers):
            x = nn.Dense(
                self.dim, 
                name="fc{0}".format(i),
                kernel_init=self.kernel_init,
                bias_init=self.bias_init
            )(x)
            x = self.act_fn(x)
        x = nn.Dense(
            self.out_dim, 
            name="fc_final",
            kernel_init=self.kernel_init,
            bias_init=self.bias_init
        )(x)
        return x


class MLP_vector_field(ModelBase):
    output_dim: int
    latent_embed_dim: int
    condition_embed_dim: Optional[int] = None
    t_embed_dim: Optional[int] = None
    joint_hidden_dim: Optional[int] = None
    num_layers: int = 3    
    act_fn: Callable[[jnp.ndarray], jnp.ndarray] = nn.silu
    n_frequencies: int = 64
    kernel_init: Callable[
        [PRNGKeyArray, Shape, Dtype],
        jnp.ndarray
    ] = default_kernel_init
    bias_init: Callable[
        [PRNGKeyArray, Shape, Dtype],
        jnp.ndarray
    ] = default_bias_init

    def time_encoder(self, t: jnp.array) -> jnp.array:
        freq = 2 * jnp.arange(self.n_frequencies) * jnp.pi
        t = freq * t  
        return jnp.concatenate((jnp.cos(t), jnp.sin(t)), axis=-1)
        
    def __post_init__(self):
        
        # set embedded dim from latent embedded dim
        if self.condition_embed_dim is None:
            self.condition_embed_dim = self.latent_embed_dim 
        if self.t_embed_dim is None:
            self.t_embed_dim = self.latent_embed_dim
        
        # set joint hidden dim from all embedded dim
        max_embed_dim = max(
            self.latent_embed_dim,
            self.condition_embed_dim,
            self.t_embed_dim
        )
        if self.joint_hidden_dim is not None:
            assert (
                self.joint_hidden_dim >= max_embed_dim
            ), (
                "joint_hidden_dim must be greater than or equal to the max of "
                "all embedded dimensions. "
            )
            self.joint_hidden_dim = self.latent_embed_dim
        else:
            concat_embed_dim = (
                self.latent_embed_dim 
                + self.condition_embed_dim 
                + self.t_embed_dim
            )
            self.joint_hidden_dim = concat_embed_dim
        super().__post_init__()
            
    @property
    def is_potential(self) -> bool:  
        return self.output_dim == 1
        
    @nn.compact
    def __call__(
            self, 
            t: float, condition: jnp.ndarray, latent: jnp.ndarray
        ) -> jnp.ndarray:  
        condition, latent = jnp.atleast_2d(condition, latent)
            
        # time embedding
        t = jnp.full(shape=(len(condition), 1), fill_value=t)
        t = self.time_encoder(t)
        t = Block(
            dim=self.t_embed_dim, 
            out_dim=self.t_embed_dim, 
            num_layers=self.num_layers,
            act_fn=self.act_fn,
        )(t)

        # condition embedding
        condition = Block(
            dim=self.condition_embed_dim, 
            out_dim=self.condition_embed_dim, 
            num_layers=self.num_layers,
            act_fn=self.act_fn,
            kernel_init=self.kernel_init,
            bias_init=self.bias_init
        )(condition)
        
        # latent embedding
        latent = Block(
            dim=self.latent_embed_dim, 
            out_dim=self.latent_embed_dim, 
            num_layers=self.num_layers,
            act_fn=self.act_fn,
            kernel_init=self.kernel_init,
            bias_init=self.bias_init
        )(latent)

        # concatenation of all embeddings
        concat_embed = jnp.concatenate((t, condition, latent), axis=-1)
        out = Block(
            dim=self.joint_hidden_dim, 
            out_dim=self.joint_hidden_dim, 
            num_layers=self.num_layers,
            act_fn=self.act_fn,
            kernel_init=self.kernel_init,
            bias_init=self.bias_init
        )(concat_embed)

        # final layer
        out = nn.Dense(
            self.output_dim, 
            use_bias=True, 
            name="final_layer",
            kernel_init=self.kernel_init,
            bias_init=self.bias_init
        )(out)

        return out

    def create_train_state(
        self, 
        rng: jax.random.PRNGKeyArray, 
        optimizer: optax.OptState, 
        input_dim: int, 
    ) -> NeuralTrainState:
        params = self.init(
            rng, 
            jnp.ones((1, 1)), 
            jnp.ones((1, input_dim)), 
            jnp.ones((1, self.output_dim))
        )["params"]
        return train_state.TrainState.create(
            apply_fn=self.apply, params=params, tx=optimizer
        )
    
    
class MLP_bridge(ModelBase):
    output_dim: int
    hidden_dim: int
    bridge_type: Literal["full", "mean", "constant"] = "constant"
    num_layers: int = 3
    act_fn: Callable[[jnp.ndarray], jnp.ndarray] = nn.silu
    kernel_init: Callable[
        [PRNGKeyArray, Shape, Dtype],
        jnp.ndarray
    ] = default_kernel_init
    bias_init: Callable[
        [PRNGKeyArray, Shape, Dtype],
        jnp.ndarray
    ] = default_bias_init
    
    @property
    def is_potential(self) -> bool:  
        return False
    
    @property
    def learn_mean(self) -> bool:
        assert self.bridge_type in ["full", "mean", "constant"], (
            f"bridge_type: {self.bridge_type} not supported. "
            "Please use 'full','mean' or 'constant'."
        )
        return self.bridge_type in ["full", "mean"]
    
    @property
    def learn_std(self) -> bool:
        return self.bridge_type == "full"
            
    @nn.compact
    def __call__(
        self, condition: jnp.ndarray
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:  
        condition = jnp.atleast_2d(condition)
        
        # constant bridge, mean = 0, std = 1
        if not (self.learn_mean or self.learn_std):
            zeros = jnp.zeros(
                (*condition.shape[:-1], self.output_dim)
            )
            return zeros, jnp.ones_like(zeros)
        
        # bridge learning the mean, std = 1
        elif self.learn_mean and not self.learn_std:
            mean = Block(
                dim=self.hidden_dim, 
                out_dim=self.output_dim,
                num_layers=self.num_layers, 
                act_fn=self.act_fn,
                kernel_init=self.kernel_init,
                bias_init=self.bias_init
            )(condition)
            return mean, jnp.ones_like(mean)
        
        # full bridge, learning both mean and variance
        else:
            out = Block(
                dim=self.hidden_dim, 
                out_dim=2*self.output_dim, # :dim for mean and :dim for log_std
                num_layers=self.num_layers, 
                act_fn=self.act_fn,
                kernel_init=self.kernel_init,
                bias_init=self.bias_init
            )(condition)
            mean = out[:, self.output_dim:]
            log_std = out[:, :self.output_dim]
            return mean, jnp.exp(log_std)

    def create_train_state(
        self,
        rng: jax.random.PRNGKeyArray,
        optimizer: optax.OptState,
        input_dim: int,
    ) -> NeuralTrainState:
        if not (self.learn_mean or self.learn_std):
            params = frozen_dict.FrozenDict({})
        else:
            params = self.init(rng, jnp.ones((1, input_dim)))["params"]
        return train_state.TrainState.create(
            apply_fn=self.apply, params=params, tx=optimizer
        )
        
class MLP_marginal(ModelBase):
    hidden_dim: int
    num_layers: int = 3
    act_fn: Callable[[jnp.ndarray], jnp.ndarray] = nn.silu
    kernel_init: Callable[
        [PRNGKeyArray, Shape, Dtype],
        jnp.ndarray
    ] = default_kernel_init
    bias_init: Callable[
        [PRNGKeyArray, Shape, Dtype],
        jnp.ndarray
    ] = default_bias_init

    @property
    def is_potential(self) -> bool:  
        return True
    
    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:  
        z = x
        z = Block(
            dim=self.hidden_dim, 
            out_dim=1, # is a potential
            num_layers=self.num_layers,
            act_fn=self.act_fn,
            kernel_init=self.kernel_init,
            bias_init=self.bias_init
        )(z)
        return jnp.exp(z) # is positive


# class old_MLP_vector_field(ModelBase):
#     output_dim: int
#     t_embed_dim: int
#     condition_embed_dim: int
#     joint_hidden_dim: int
#     is_potential: bool = False
#     act_fn: Callable[[jnp.ndarray], jnp.ndarray] = nn.silu
#     latent_dim: int = 0
#     n_frequencies: int = 1

#     def time_encoder(self, t: jnp.array) -> jnp.array:
#         freq = 2 * jnp.arange(self.n_frequencies) * jnp.pi
#         t = freq * t  # [..., None]
#         return jnp.concatenate((jnp.cos(t), jnp.sin(t)), axis=-1)

#     @nn.compact
#     def __call__(self, t: float, condition: jnp.ndarray, latent: jnp.ndarray) -> jnp.ndarray:  # noqa: D102
#         t = jnp.full(shape=(len(condition), 1), fill_value=t)
#         t = self.time_encoder(t)
#         t = Block(dim=self.t_embed_dim, out_dim=self.t_embed_dim, act_fn=self.act_fn)(t)

#         condition = Block(dim=self.condition_embed_dim, out_dim=self.output_dim, act_fn=self.act_fn)(condition)
#         latent = Block(dim=self.condition_embed_dim, out_dim=self.output_dim, act_fn=self.act_fn)(latent)

#         z = jnp.concatenate((condition, t, latent), axis=-1)
#         z = Block(num_layers=3, dim=self.joint_hidden_dim, out_dim=self.joint_hidden_dim, act_fn=self.act_fn)(z)

#         Wx = nn.Dense(self.output_dim, use_bias=True, name="final_layer")
#         z = Wx(z)

#         return z

#     def create_train_state(
#         self, rng: jax.random.PRNGKeyArray, optimizer: optax.OptState, source_dim: int, 
#     ) -> NeuralTrainState:
#         params = self.init(
#             rng, 
#             jnp.ones((1, 1)), 
#             jnp.ones((1, source_dim)), 
#             jnp.ones((1, self.output_dim))
#         )["params"]
#         return train_state.TrainState.create(apply_fn=self.apply, params=params, tx=optimizer)
    
# class MLP_vector_field_(ModelBase):
#     output_dim: int
#     latent_condition_embed_dim: int 
#     t_embed_dim: Optional[int] = None
#     joint_hidden_dim: Optional[int] = None
#     num_layers: int = 3    
#     act_fn: Callable[[jnp.ndarray], jnp.ndarray] = nn.silu
#     n_frequencies: int = 64
#     kernel_init: Callable[
#         [PRNGKeyArray, Shape, Dtype],
#         jnp.ndarray
#     ] = nn.initializers.variance_scaling(1., mode='fan_avg', distribution='normal')
#     bias_init: Callable[
#         [PRNGKeyArray, Shape, Dtype],
#         jnp.ndarray
#     ] = nn.initializers.zeros

#     def time_encoder(self, t: jnp.array) -> jnp.array:
#         freq = 2 * jnp.arange(self.n_frequencies) * jnp.pi
#         t = freq * t  
#         return jnp.concatenate((jnp.cos(t), jnp.sin(t)), axis=-1)
        
#     # def __post_init__(self):
        
#     #     # set embedded dim from latent embedded dim
#     #     if self.condition_embed_dim is None:
#     #         self.condition_embed_dim = self.latent_embed_dim 
#     #     if self.t_embed_dim is None:
#     #         self.t_embed_dim = self.latent_embed_dim
        
#     #     # set joint hidden dim from all embedded dim
#     #     max_embed_dim = max(
#     #         self.latent_embed_dim,
#     #         self.condition_embed_dim,
#     #         self.t_embed_dim
#     #     )
#     #     if self.joint_hidden_dim is not None:
#     #         assert (
#     #             self.joint_hidden_dim >= max_embed_dim
#     #         ), (
#     #             "joint_hidden_dim must be greater than or equal to the max of "
#     #             "all embedded dimensions. "
#     #         )
#     #         self.joint_hidden_dim = self.latent_embed_dim
#     #     else:
#     #         concat_embed_dim = (
#     #             self.latent_embed_dim 
#     #             + self.condition_embed_dim 
#     #             + self.t_embed_dim
#     #         )
#     #         self.joint_hidden_dim = concat_embed_dim
#     #     super().__post_init__()
            
#     @property
#     def is_potential(self) -> bool:  
#         return self.output_dim == 1
        
#     @nn.compact
#     def __call__(
#             self, 
#             t: float, condition: jnp.ndarray, latent: jnp.ndarray
#         ) -> jnp.ndarray:  
#         condition, latent = jnp.atleast_2d(condition, latent)
            
#         # time embedding
#         t = jnp.full(shape=(len(condition), 1), fill_value=t)
#         t = self.time_encoder(t)
#         emb_t = Block(
#             dim=self.t_embed_dim, 
#             out_dim=self.t_embed_dim, 
#             num_layers=self.num_layers,
#             act_fn=self.act_fn,
#         )(t)

#         # condition embedding
#         zx = jnp.concatenate((condition, latent), axis=-1)
#         emb_zx = Block(
#             dim=self.latent_condition_embed_dim, 
#             out_dim=self.latent_condition_embed_dim, 
#             num_layers=self.num_layers,
#             act_fn=self.act_fn,
#             kernel_init=self.kernel_init,
#             bias_init=self.bias_init
#         )(zx)

#         # concatenation of all embeddings
#         concat_embed = jnp.concatenate((emb_t, emb_zx), axis=-1)
#         out = Block(
#             dim=self.joint_hidden_dim, 
#             out_dim=self.joint_hidden_dim, 
#             num_layers=self.num_layers,
#             act_fn=self.act_fn,
#             kernel_init=self.kernel_init,
#             bias_init=self.bias_init
#         )(concat_embed)

#         # final layer
#         out = nn.Dense(
#             self.output_dim, 
#             use_bias=True, 
#             name="final_layer",
#             kernel_init=self.kernel_init,
#             bias_init=self.bias_init
#         )(out)

#         return out

#     def create_train_state(
#         self, 
#         rng: jax.random.PRNGKeyArray, 
#         optimizer: optax.OptState, 
#         input_dim: int, 
#     ) -> NeuralTrainState:
#         params = self.init(
#             rng, 
#             jnp.ones((1, 1)), 
#             jnp.ones((1, input_dim)), 
#             jnp.ones((1, self.output_dim))
#         )["params"]
#         return train_state.TrainState.create(
#             apply_fn=self.apply, params=params, tx=optimizer
#         )


# class Bridge_MLP_mean(ModelBase):
#     output_dim: int
#     t_embed_dim: int
#     condition_embed_dim: int
#     is_potential: bool = False
#     act_fn: Callable[[jnp.ndarray], jnp.ndarray] = nn.silu

#     @nn.compact
#     def __call__(self, condition: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:  # noqa: D102
#         condition = Block(
#             dim=self.condition_embed_dim, 
#             out_dim=self.condition_embed_dim, 
#             act_fn=self.act_fn
#         )(condition)
#         mu = Block(dim=self.condition_embed_dim, out_dim=self.output_dim)(condition)
#         return mu, jnp.ones_like(mu)

#     def create_train_state(
#         self,
#         rng: jax.random.PRNGKeyArray,
#         optimizer: optax.OptState,
#         input_dim: int,
#     ) -> NeuralTrainState:
#         params = self.init(rng, jnp.ones((1, input_dim)))["params"]
#         return train_state.TrainState.create(apply_fn=self.apply, params=params, tx=optimizer)


# class Bridge_MLP_full(ModelBase):
#     output_dim: int
#     t_embed_dim: int
#     condition_embed_dim: int
#     is_potential: bool = False
#     act_fn: Callable[[jnp.ndarray], jnp.ndarray] = nn.silu

#     @nn.compact
#     def __call__(self, condition: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:  # noqa: D102
#         condition = jnp.atleast_2d(condition)
#         condition = Block(
#             dim=self.condition_embed_dim, 
#             out_dim=self.condition_embed_dim, 
#             act_fn=self.act_fn
#         )(condition)
#         out = Block(
#             dim=self.condition_embed_dim, 
#             out_dim=2*self.output_dim  # :dim for mean and :dim for log_std
#         )(condition)
#         mean = out[:, self.output_dim:]
#         log_std = out[:, :self.output_dim]
#         return mean, jnp.exp(log_std)

#     def create_train_state(
#         self,
#         rng: jax.random.PRNGKeyArray,
#         optimizer: optax.OptState,
#         output_dim: int,
#     ) -> NeuralTrainState:
#         params = self.init(rng, jnp.ones((1, output_dim)))["params"]
#         return train_state.TrainState.create(apply_fn=self.apply, params=params, tx=optimizer)


# class Bridge_MLP_constant(ModelBase):
#     output_dim: int
#     t_embed_dim: int
#     condition_embed_dim: int
#     is_potential: bool = False
#     act_fn: Callable[[jnp.ndarray], jnp.ndarray] = nn.silu

#     @nn.compact
#     def __call__(self, condition: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:  # noqa: D102
#         return jnp.zeros((*condition.shape[:-1], self.output_dim)), jnp.ones((*condition.shape[:-1], self.output_dim))

#     def create_train_state(
#         self,
#         rng: jax.random.PRNGKeyArray,
#         optimizer: optax.OptState,
#         input_dim: int,
#     ) -> NeuralTrainState:
#         return train_state.TrainState.create(apply_fn=self.apply, params={}, tx=optimizer)
