import time
from functools import partial
from typing import *
from typing import Any

import diffrax
import equinox as eqx  # https://github.com/patrick-kidger/equinox
import flax.linen as nn
import jax
import jax.nn as jnn
import jax.numpy as jnp
import jax.random as jrandom
import matplotlib.pyplot as plt
import numpy as np
import optax  # https://github.com/deepmind/optax
import types
import ott
import pandas as pd
import tensorflow as tf
import tensorflow_datasets as tfds
from flax.training import train_state
from flax.training.train_state import TrainState
from ott.solvers.nn.models import ModelBase, NeuralTrainState
from sklearn.datasets import make_moons, make_swiss_roll
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import jax
import jax.numpy as jnp
import numpy as np

import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
from IPython import display
from matplotlib import animation, cm

from ott.geometry import pointcloud
from ott.problems.quadratic import quadratic_problem
from ott.solvers.quadratic import gromov_wasserstein

from entot.models.utils import MLP, MLP_FM


class MLP_FM(ModelBase):
    output_dim: int
    dim_hidden: Sequence[int]
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
        Wt = nn.Dense(self.dim_hidden[0], use_bias=True)
        t = self.act_fn(Wt(t))

        Wc = nn.Dense(self.dim_hidden[0], use_bias=True)
        condition = self.act_fn(Wc(condition))

        # try to avoid this
        #if condition.ndim == 3:
        #    condition = jnp.squeeze(condition)

        # try to avoid this
        #if latent.ndim == 3:
        #    latent = jnp.squeeze(latent)


        # try to avoid this
        #squeeze = condition.ndim == 1
        #if squeeze:
        #    condition = jnp.expand_dims(condition, 0)


        # try to avoid this
        #squeeze = latent.ndim == 1
        #if squeeze:
        #    latent = jnp.expand_dims(latent, 0)

        assert condition.ndim == 2, condition.ndim
        assert latent.ndim == 2, latent.dim

        z = jnp.concatenate((condition, latent, t), axis=-1)

        for n_hidden in self.dim_hidden:
            Wx = nn.Dense(n_hidden, use_bias=True)
            z = self.act_fn(Wx(z))

        Wx = nn.Dense(self.output_dim, use_bias=True)
        z = Wx(z)

        return z

    def create_train_state(
        self, rng: jax.random.PRNGKeyArray, optimizer: optax.OptState, source_dim: int, latent_dim: int, **kwargs: Any
    ) -> NeuralTrainState:
        params = self.init(rng, jnp.ones((1,1)), jnp.ones((1,source_dim)), jnp.ones((1,latent_dim)))["params"]
        return train_state.TrainState.create(apply_fn=self.apply, params=params, tx=optimizer)


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
    
class MLP_FM2(ModelBase):
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

        condition = Block(dim=self.condition_embed_dim, out_dim=self.condition_embed_dim, activation_fn=self.act_fn)(condition)

        z = jnp.concatenate((condition, latent, t), axis=-1)
        z = Block(dim=self.joint_hidden_dim, out_dim=self.joint_hidden_dim, activation_fn=self.act_fn)(z)

        Wx = nn.Dense(self.output_dim, use_bias=True, name="final_layer")
        z = Wx(z)

        return z

    def create_train_state(
        self, rng: jax.random.PRNGKeyArray, optimizer: optax.OptState, source_dim: int, latent_dim: int, **kwargs: Any
    ) -> NeuralTrainState:
        params = self.init(rng, jnp.ones((1,1)), jnp.ones((1,source_dim)), jnp.ones((1,latent_dim)))["params"]
        return train_state.TrainState.create(apply_fn=self.apply, params=params, tx=optimizer)


class OTFlowMatching:
    def __init__(
        self,
        neural_net: Type[ModelBase],
        input_dim: int,
        output_dim: int,
        iterations: int,
        opt: Optional[Any] = None,
        k_noise_per_x: int = 1,
        noise_std: float = 1.0,
        ot_solver: Optional[Any] = None,
        latent_to_data_solver: Optional[Any] = None,
        epsilon: float = 1e-2,
        eps_match_latent_fn: float = 1e-2,
        cost_fn: str = ott.geometry.costs.SqEuclidean(),
        seed: int = 0,
        **kwargs: Any,
    ) -> None:
        self.optimizer = optax.adamw(learning_rate=1e-4, weight_decay=1e-10) if opt is None else opt
        self.noise_fn = partial(
            jax.random.multivariate_normal, mean=jnp.zeros((output_dim,)), cov=jnp.diag(jnp.ones((output_dim,)))
        )
        self.input_dim = input_dim
        self.output_dim = output_dim  # note that output_dim == latent_dim/noise_dim
        self.noise_std = noise_std
        self.rng = jax.random.PRNGKey(seed)
        self.k_noise_per_x: int = k_noise_per_x
        self.iterations = iterations
        self.neural_net = neural_net
        self.state_mlp: Optional[TrainState] = None
        self.metrics = {"loss": []}
        self.ot_solver = ott.solvers.linear.sinkhorn.Sinkhorn() if ot_solver is None else ot_solver
        self.latent_to_data_solver = latent_to_data_solver
        self.epsilon = epsilon
        self.eps_match_latent_fn = eps_match_latent_fn
        self.cost_fn = cost_fn

        self.setup(**kwargs)

    def setup(self, **kwargs: Any) -> None:
        self.state_mlp = self.neural_net.create_train_state(self.rng, self.optimizer, self.input_dim, self.output_dim)

        self.step_fn = self._get_step_fn()
        if self.cost_fn == "graph":
            self.match_fn = self._get_match_fn_graph(self.ot_solver, epsilon=self.epsilon,  k_neighbors=kwargs.pop("k_neighbors", 30))
        else:
            if isinstance(self.ot_solver, ott.solvers.linear.sinkhorn.Sinkhorn):
                self.match_fn = self._get_match_fn(self.ot_solver, epsilon=self.epsilon, cost_fn=self.cost_fn)
            else:
                self.match_fn = self._get_gromov_match_fn(self.ot_solver, epsilon=self.epsilon, cost_fn=self.cost_fn)
        self.match_latent_fn = self._get_match_latent_fn(self.latent_to_data_solver, epsilon=self.eps_match_latent_fn) if self.latent_to_data_solver is not None else lambda x, *_, **__: x

    def __call__(self, x: jnp.array, y: jnp.array, batch_size_source: int, batch_size_target: int) -> Any:
        x_loader = iter(
            tf.data.Dataset.from_tensor_slices(x).repeat().shuffle(buffer_size=10_000).batch(batch_size_source)
        )
        y_loader = iter(
            tf.data.Dataset.from_tensor_slices(y).repeat().shuffle(buffer_size=10_000).batch(batch_size_target)
        )

        batch: Dict[str, jnp.array] = {}
        for _ in tqdm(range(self.iterations)):
            x_batch, y_batch = tfds.as_numpy(next(x_loader)), tfds.as_numpy(next(y_loader))
            self.rng, rng_time, rng_noise = jax.random.split(self.rng, 3)
            source_batch = jnp.tile(x_batch, (self.k_noise_per_x, *((1,) * (x_batch.ndim - 1))))
            #latent_shape = (
            #    (*source_batch.shape[:-1], self.output_dim)
            #    if len(source_batch.shape[:-1]) > 1
            #    else (source_batch.shape[:-1][0], self.output_dim)
            #)
            latent_shape = (len(x_batch),)
            latent_batch = self.noise_fn(rng_noise, shape=latent_shape) * self.noise_std
         
            source_batch, y_batch = self.match_fn(source_batch, y_batch)
            latent_batch = self.match_latent_fn(latent_batch, y_batch)
            batch["latent"] = latent_batch
            batch["source"] = source_batch
            batch["target"] = y_batch
            batch["time"] = jax.random.uniform(rng_time, (len(source_batch), 1))
         
            loss, self.state_mlp = self.step_fn(self.state_mlp, batch)
            self.metrics["loss"].append(loss)

    def _get_match_fn(self, ot_solver: Any, epsilon: float, cost_fn: str) -> Callable:
        @jax.jit
        def match_pairs(x: jnp.ndarray, y: jnp.ndarray, solver: Any, epsilon: float, cost_fn: str) -> Tuple[jnp.array, jnp.array]:
            geom = ott.geometry.pointcloud.PointCloud(x, y, epsilon=epsilon, scale_cost="mean", cost_fn=cost_fn)
            out = solver(ott.problems.linear.linear_problem.LinearProblem(geom))
            pi_star_inds = jax.random.categorical(
                jax.random.PRNGKey(0), logits=jnp.log(out.matrix.flatten()), shape=(len(x),)
            )
            inds_source = pi_star_inds // len(y)
            inds_target = pi_star_inds % len(y)
            return x[inds_source], y[inds_target]

        return partial(match_pairs, solver=ot_solver, epsilon=epsilon, cost_fn=cost_fn)

    def _get_gromov_match_fn(self, ot_solver: Any, epsilon: float, cost_fn: str) -> Callable:
        @jax.jit
        def match_pairs(x: jnp.ndarray, y: jnp.ndarray, solver: Any, epsilon: float, cost_fn: str) -> Tuple[jnp.array, jnp.array]:
            geom_xx = pointcloud.PointCloud(x=x, y=x, cost_fn=cost_fn, scale_cost="mean")
            geom_yy = pointcloud.PointCloud(x=y, y=y, cost_fn=cost_fn, scale_cost="mean")
            prob = quadratic_problem.QuadraticProblem(geom_xx, geom_yy)
            out = solver(prob)
            
            pi_star_inds = jax.random.categorical(
                jax.random.PRNGKey(0), logits=jnp.log(out.matrix.flatten()), shape=(len(x),)
            )
            inds_source = pi_star_inds // len(y)
            inds_target = pi_star_inds % len(y)
            return x[inds_source], y[inds_target]

        return partial(match_pairs, solver=ot_solver, epsilon=epsilon, cost_fn=cost_fn)
        
    def _get_match_fn_graph(self, ot_solver: Any, epsilon: float, k_neighbors: int) -> Callable:

        def get_nearest_neighbors(
            X: jnp.ndarray, Y: jnp.ndarray, len_x: int, k: int = 30  # type: ignore[name-defined]
        ) -> Tuple[jnp.ndarray, jnp.ndarray]:  # type: ignore[name-defined]
            concat = jnp.concatenate((X,Y), axis=0)
            pairwise_euclidean_distances = ott.geometry.pointcloud.PointCloud(concat, concat).cost_matrix
            distances, indices = jax.lax.approx_min_k(pairwise_euclidean_distances, k=k, recall_target=0.95, aggregate_to_topk=True)
            return distances, indices

        def create_cost_matrix(X: jnp.array, Y: jnp.array, k_neighbors: int) -> jnp.array:
            distances, indices = get_nearest_neighbors(X, Y, len(X), k_neighbors)
            a = jnp.zeros((len(X)+ len(Y), len(X)+ len(Y)))
            adj_matrix = a.at[jnp.repeat(jnp.arange(len(X)+len(Y)), repeats=k_neighbors).flatten(), indices.flatten()].set(distances.flatten())
            return ott.geometry.graph.Graph.from_graph(adj_matrix, normalize=True).cost_matrix
        
        @partial(jax.jit, static_argnames=["solver", "epsilon", "k_neighbors"])
        def match_pairs(x: jnp.ndarray, y: jnp.ndarray, solver: Any, epsilon: float, k_neighbors: int) -> Tuple[jnp.array, jnp.array]:
            cm = create_cost_matrix(x, y, k_neighbors)
            geom = ott.geometry.geometry.Geometry(cost_matrix=cm, epsilon=1e-1) # add scale_cost
            out = solver(ott.problems.linear.linear_problem.LinearProblem(geom))
            pi_star_inds = jax.random.categorical(
                jax.random.PRNGKey(0), logits=jnp.log(out.matrix.flatten()), shape=(len(x),)
            )
            inds_source = pi_star_inds // len(y)
            inds_target = pi_star_inds % len(y)
            return x[inds_source], y[inds_target]

        return partial(match_pairs, solver=ot_solver, epsilon=epsilon, k_neighbors=k_neighbors)

    def _get_match_latent_fn(self, ot_solver: Any, epsilon: float) -> Callable:
        @jax.jit
        def match_left_to_right(x: jnp.ndarray, y: jnp.ndarray, solver: Any, epsilon: float) -> Tuple[jnp.array, jnp.array]:
            geom = ott.geometry.pointcloud.PointCloud(x, y, epsilon=epsilon, scale_cost="mean")
            out = solver(ott.problems.linear.linear_problem.LinearProblem(geom))
            inds_source = jax.vmap(
                lambda tmap: jax.random.categorical(jax.random.PRNGKey(0), logits=jnp.log(tmap), shape=(1,))
            )(out.matrix)

            return x[jnp.squeeze(inds_source)]

        return partial(match_left_to_right, solver=ot_solver, epsilon=epsilon)

    def _get_step_fn(self) -> Callable:
        def psi_t(x_0, x_1, t):
            return (1 - t) * x_0 + t * x_1

        def loss_fn(params_mlp: jnp.array, apply_fn_mlp: Callable, batch: Dict[str, jnp.array]):
            psi_t_eval = psi_t(batch["latent"], batch["target"], batch["time"])
            mlp_pred = apply_fn_mlp({"params": params_mlp}, batch["time"], psi_t_eval, batch["source"])
            d_psi = batch["target"] - batch["latent"]
            if len(mlp_pred.shape) == 1:
                mlp_pred = mlp_pred[:, None]
            return jnp.mean(optax.l2_loss(mlp_pred, d_psi))

        @jax.jit
        def step_fn(state_neural_net: TrainState, batch: Dict[str, jnp.array]):
            grad_fn = jax.value_and_grad(loss_fn, argnums=0, has_aux=False)
            val, grads_mlp = grad_fn(state_neural_net.params, state_neural_net.apply_fn, batch)

            return val, state_neural_net.apply_gradients(grads=grads_mlp)

        return step_fn

    def transport(self, x: jnp.array, samples_per_x: int = 1, seed: int = 0, diffeqsolve_kwargs: Dict[str, Any] = types.MappingProxyType({})) -> Union[jnp.array, diffrax.Solution]:
        diffeqsolve_kwargs = dict(diffeqsolve_kwargs)
        rng = jax.random.PRNGKey(seed)
        source_batch = x  # jnp.tile(x, (samples_per_x, *((1,) * (x.ndim-1))))
        latent_shape = (len(x),)
        latent_batch = self.noise_fn(rng, shape=latent_shape) * self.noise_std

        apply_fn_partial = partial(
            self.state_mlp.apply_fn, condition=source_batch
        )  # note that here we swap source and noise
        solution = diffrax.diffeqsolve(
            diffrax.ODETerm(lambda t, y, args: apply_fn_partial({"params": self.state_mlp.params}, t=t, latent=y)),
            diffeqsolve_kwargs.pop("solver", diffrax.Tsit5()),
            t0=0,
            t1=1,
            dt0=diffeqsolve_kwargs.pop("dt0", None),
            y0=latent_batch,  
            stepsize_controller=diffeqsolve_kwargs.pop("stepsize_controller", diffrax.PIDController(rtol=1e-3, atol=1e-6)),
            # saveat=diffrax.SaveAt(ts=ts),
        )
        t_xz = solution.ys
        return jnp.reshape(t_xz, (samples_per_x, *(x.shape))), solution


class OriginalFlowMatching:
    def __init__(self, neural_net: Type[ModelBase], input_dim: int, iterations: int, opt: Optional[Any]= None, sig_min: float = 0.001) -> None:
        self.optimizer = optax.adamw(learning_rate=1e-4, weight_decay=1e-10) if opt is None else opt
        self.input_dim = input_dim
        self.rng = jax.random.PRNGKey(0)
        self.iterations = iterations
        self.sig_min = sig_min
        
        self.neural_net = neural_net

        self.state_mlp: Optional[TrainState] = None

        self.setup()
    
    def setup(self) -> None:
        self.state_mlp = self.neural_net.create_train_state(self.rng, self.optimizer, self.input_dim, self.input_dim)
        
        self.step_fn = self._get_step_fn()
        self.match_fn = self._get_match_fn()

    def __call__(self, x: jnp.array, y: jnp.array, batch_size_source: int, batch_size_target: int) -> Any:
        x_loader = iter(tf.data.Dataset.from_tensor_slices(x).repeat().shuffle(buffer_size=10_000).batch(batch_size_source))
        y_loader = iter(tf.data.Dataset.from_tensor_slices(y).repeat().shuffle(buffer_size=10_000).batch(batch_size_target))

        batch: Dict[str, jnp.array] = {}
        for step in tqdm(range(self.iterations)):

            x_batch, y_batch = tfds.as_numpy(next(x_loader)), tfds.as_numpy(next(y_loader))
            self.rng, rng_time = jax.random.split(self.rng, 2)
            x_batch, y_batch = self.match_fn(x_batch, y_batch)
            batch["source"] = x_batch
            batch["target"] = y_batch
            batch["time"] = jax.random.uniform(rng_time, (len(x_batch), 1))
            self.state_mlp = self.step_fn(self.state_mlp, batch, self.sig_min)

    def _get_match_fn(self) -> Callable:
        
        @jax.jit
        def match_pairs(x: jnp.ndarray, y: jnp.ndarray) -> Tuple[jnp.array, jnp.array]:
            geom = ott.geometry.pointcloud.PointCloud(x, y, epsilon=1e-1, scale_cost="mean")
            out = ott.solvers.linear.sinkhorn.Sinkhorn(threshold=1e-2)(
                ott.problems.linear.linear_problem.LinearProblem(geom)
            )
            pi_star_inds = jax.random.categorical(
                jax.random.PRNGKey(0), logits=jnp.log(out.matrix.flatten()), shape=(len(x),)
            )
            inds_source = pi_star_inds // len(y)
            inds_target = pi_star_inds % len(y)
            return x[inds_source], y[inds_target]
        return match_pairs
    
    
    def _get_step_fn(self) -> Callable:
        def psi_t(x_0, x_1, t,sig_min):
            return (1 - (1 - sig_min) * t) * x_0 + t * x_1
        
        def loss_fn(params_mlp: jnp.array, apply_fn_mlp: Callable, batch: Dict[str, jnp.array], sig_min: float):
            psi_t_eval = psi_t(batch["source"], batch["target"], batch["time"], sig_min)
            mlp_pred = apply_fn_mlp({"params": params_mlp}, batch["time"], psi_t_eval)
            d_psi = batch["target"] - batch["source"]
            if len(mlp_pred.shape) == 1:
                mlp_pred = mlp_pred[:, None]
            return jnp.mean(optax.l2_loss(mlp_pred, d_psi))
        
        @jax.jit
        def step_fn(state_neural_net: TrainState, batch: Dict[str, jnp.array], sig_min: float):
            grad_fn = jax.value_and_grad(loss_fn, argnums=0, has_aux=False)
            val, grads_mlp = grad_fn(state_neural_net.params, state_neural_net.apply_fn, batch, sig_min)

            return state_neural_net.apply_gradients(grads=grads_mlp)
        
        return step_fn
    
        

    def transport(self, x: jnp.array) -> Union[jnp.array, Any]:
        solution = diffrax.diffeqsolve(
                    diffrax.ODETerm(lambda t,y,args: self.state_mlp.apply_fn({"params":self.state_mlp.params}, t=t, x=y)),
                    diffrax.Tsit5(),
                    t0=0,
                    t1=1,
                    dt0=None,
                    y0=x, # here we swap source and noise
                    stepsize_controller=diffrax.PIDController(rtol=1e-3, atol=1e-6),
                    #saveat=diffrax.SaveAt(ts=ts),
                    )
        return solution.ys, solution



class MLP_no_noise(ModelBase):
    output_dim: int
    dim_hidden: Sequence[int]
    is_potential: bool = False
    act_fn: Callable[[jnp.ndarray], jnp.ndarray] = nn.silu
    n_frequencies: int = 1

    def time_encoder(self, t: jnp.array) -> jnp.array:
        freq = 2 * jnp.arange(self.n_frequencies) * jnp.pi
        t = freq * t  # [..., None]
        return jnp.concatenate((jnp.cos(t), jnp.sin(t)), axis=-1)

    @nn.compact
    def __call__(self, t: float, x: jnp.ndarray) -> jnp.ndarray:  # noqa: D102
        t = jnp.full(shape=(len(x), 1), fill_value=t)
        t = self.time_encoder(t)
        Wt = nn.Dense(self.dim_hidden[0], use_bias=True)
        t = self.act_fn(Wt(t))

        Wc = nn.Dense(self.dim_hidden[0], use_bias=True)
        x = self.act_fn(Wc(x))

        z = jnp.concatenate((x, t), axis=-1)

        for n_hidden in self.dim_hidden:
            Wx = nn.Dense(n_hidden, use_bias=True)
            z = self.act_fn(Wx(z))

        Wx = nn.Dense(self.output_dim, use_bias=True)
        z = Wx(z)

        return z

    def create_train_state(
        self, rng: jax.random.PRNGKeyArray, optimizer: optax.OptState, source_dim: int, latent_dim: int, **kwargs: Any
    ) -> NeuralTrainState:
        params = self.init(rng, jnp.ones((1,1)), jnp.ones((1,source_dim)))["params"]
        return train_state.TrainState.create(apply_fn=self.apply, params=params, tx=optimizer)
