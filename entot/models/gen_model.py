import time
from functools import partial
from typing import *
from typing import Any

import diffrax
import equinox as eqx  # https://github.com/patrick-kidger/equinox
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax  # https://github.com/deepmind/optax
import types
import ott
from ott.solvers.linear import sinkhorn
import pandas as pd
import tensorflow as tf
import tensorflow_datasets as tfds
from flax.training import train_state
from flax.training.train_state import TrainState
from ott.solvers.nn.models import ModelBase, NeuralTrainState
from ott.solvers import was_solver
from sklearn.datasets import make_moons, make_swiss_roll
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from ott.geometry import costs
import jax
import jax.numpy as jnp
import numpy as np
import collections
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3

from ott.geometry import pointcloud, graph, geometry
from ott.problems.quadratic import quadratic_problem
from ott.solvers.quadratic import gromov_wasserstein
from ott.problems.linear import linear_problem

from entot.models.utils import MLP, MLP_FM


Match_fn_T = Callable[[jax.random.PRNGKeyArray, jnp.array, jnp.array], Tuple[jnp.array, jnp.array, jnp.array, jnp.array]]
Match_latent_fn_T = Callable[[jax.random.PRNGKeyArray, jnp.array, jnp.array], Tuple[jnp.array, jnp.array]]

def sample_indices_from_tmap(key: jax.random.PRNGKeyArray, tmat: jnp.ndarray, n_samples: Optional[int]) -> Tuple[jnp.array, jnp.array]:
    n_samples = n_samples if isinstance(n_samples, int) else tmat.shape[1]
    pi_star_inds = jax.random.categorical(
                key, logits=jnp.log(tmat.flatten()), shape=(n_samples,)
            )
    return pi_star_inds // tmat.shape[1], pi_star_inds % tmat.shape[1]

def sample_conditional_indices_from_tmap(key: jax.random.PRNGKeyArray, tmat: jnp.ndarray, k_samples_per_x: Optional[int]) -> Tuple[jnp.array, jnp.array]:
    indices_per_row = jax.vmap(lambda tmat: jax.random.choice(key=key, a=jnp.arange(len(tmat)), p=tmat, shape=(k_samples_per_x,)), in_axes=0, out_axes=0)(tmat)
    return jnp.repeat(jnp.arange(tmat.shape[0]), k_samples_per_x), indices_per_row % tmat.shape[1]
    
def sinkhorn_fn(key, x, y):
    ot_solver = ott.solvers.linear.sinkhorn.Sinkhorn()
    geom = pointcloud.PointCloud(x, y, epsilon=1e-2, scale_cost="mean")
    out = ot_solver(linear_problem.LinearProblem(geom))
    inds_source, inds_target = sample_conditional_indices_from_tmap(key, out.matrix, 1)
    return x[inds_source], y[inds_target]

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

class MLP_FM3(ModelBase):
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
        t = freq * t
        return jnp.concatenate((jnp.cos(t), jnp.sin(t)), axis=-1)

    @nn.compact
    def __call__(self, t: float, condition: jnp.ndarray, latent: jnp.ndarray) -> jnp.ndarray:  # noqa: D102
        t = jnp.full(shape=(len(condition), 1), fill_value=t)
        t = self.time_encoder(t)
        t = Block(dim=self.t_embed_dim, out_dim=self.t_embed_dim, activation_fn=self.act_fn)(t)

        condition = Block(dim=self.condition_embed_dim, out_dim=self.condition_embed_dim, activation_fn=self.act_fn)(condition)

        z = jnp.concatenate((condition, latent, t), axis=-1)
        z = Block(num_layers=1,dim=self.joint_hidden_dim, out_dim=self.joint_hidden_dim, activation_fn=self.act_fn)(z)

        z = jnp.concatenate((condition, z), axis=-1)
        z = Block(num_layers=1,dim=self.joint_hidden_dim, out_dim=self.joint_hidden_dim, activation_fn=self.act_fn)(z)

        z = jnp.concatenate((condition, z), axis=-1)
        z = Block(num_layers=1,dim=self.joint_hidden_dim, out_dim=self.joint_hidden_dim, activation_fn=self.act_fn)(z)

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
        bridge_net: Type[ModelBase],
        input_dim: int,
        output_dim: int,
        iterations: int,
        ot_solver: Type[was_solver.WassersteinSolver],
        optimizer: Optional[Any] = None,
        k_noise_per_x: int = 1, # TODO(@MUCDK) remove?
        n_samples: Optional[int]=None,
        noise_std: float = 0.1,
        solver_latent_to_data: Optional[Type[was_solver.WassersteinSolver]] = None,
        epsilon: float = 1e-2,
        epsilon_latent_to_data: float = 1e-2,
        cost_fn: Union[costs.CostFn, Literal["graph"]] = costs.SqEuclidean(),
        scale_cost: Any = "mean",
        graph_kwargs: Dict[str, Any] = types.MappingProxyType({}),
        fused_penalty: float = 0.0,
        split_dim: int = 0,
        mlp_eta: Callable[[jnp.ndarray], float] = None,
        mlp_xi: Callable[[jnp.ndarray], float] = None,
        tau_a: float = 1.0,
        tau_b: float = 1.0,
        unbalanced_kwargs: Dict[str, Any] = {},
        callback: Optional[Callable[[jnp.ndarray, jnp.ndarray, jnp.ndarray], Any]] = None,
        callback_kwargs: Dict[str, Any] = {},
        callback_iters: int = 10,
        seed: int = 0,
        **kwargs: Any,
    ) -> None:
        # setup parameters
        self.rng = jax.random.PRNGKey(seed)
        self.iterations = iterations
        self.metrics = {"loss": [] ,"loss_eta": [], "loss_xi": []}
        
        # neural parameters
        self.neural_net = neural_net
        self.bridge_net = bridge_net
        self.state_neural_net: Optional[TrainState] = None
        self.state_bridge_net: Optional[TrainState] = None
        self.optimizer = optax.adamw(learning_rate=1e-4, weight_decay=1e-10) if optimizer is None else optimizer
        self.noise_fn = jax.tree_util.Partial(
            jax.random.multivariate_normal, mean=jnp.zeros((output_dim,)), cov=jnp.diag(jnp.ones((output_dim,)))
        )
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.noise_std = noise_std
        self.k_noise_per_x = k_noise_per_x
        self.n_samples = n_samples
        
        # OT data-data matching parameters
        self.ot_solver = ot_solver
        self.epsilon = epsilon
        self.cost_fn = cost_fn
        self.scale_cost = scale_cost
        self.graph_kwargs = graph_kwargs # "k_neighbors", kwargs for graph.Graph.from_graph()
        if fused_penalty != 0 and split_dim == 0:
            raise ValueError("Missing 'split_dim' for FGW.")
        self.fused_penalty = fused_penalty
        self.split_dim = split_dim

        # OT latent-data matching parameters
        self.solver_latent_to_data = sinkhorn.Sinkhorn() if solver_latent_to_data is None else solver_latent_to_data
        self.epsilon_latent_to_data = epsilon_latent_to_data
        
        # unbalancedness parameters
        self.mlp_eta = mlp_eta
        self.mlp_xi = mlp_xi
        self.state_eta: Optional[TrainState] = None
        self.state_xi: Optional[TrainState] = None
        self.tau_a: float = tau_a
        self.tau_b: float = tau_b
        self.unbalanced_kwargs = unbalanced_kwargs # "opt_eta", "opt_xi"

        # callback parameteres
        self.callback = callback
        self.callback_kwargs = callback_kwargs
        self.callback_iters = callback_iters

        self.setup(**kwargs)

    def setup(self, **kwargs: Any) -> None:
        self.state_neural_net = self.neural_net.create_train_state(self.rng, self.optimizer, self.output_dim, self.output_dim)
        self.state_bridge_net = self.bridge_net.create_train_state(self.rng, self.optimizer, self.output_dim)

        self.step_fn = self._get_step_fn()
        if self.cost_fn == "graph":
            self.match_fn = self._get_match_fn_graph(self.ot_solver, epsilon=self.epsilon, tau_a=self.tau_a, tau_b=self.tau_b, **self.graph_kwargs)
        else:
            if isinstance(self.ot_solver, sinkhorn.Sinkhorn):
                self.match_fn = self._get_sinkhorn_match_fn(self.ot_solver, epsilon=self.epsilon, cost_fn=self.cost_fn, tau_a=self.tau_a, tau_b=self.tau_b, scale_cost=self.scale_cost, k_samples_per_x=self.k_noise_per_x)
            else:
                self.match_fn = self._get_gromov_match_fn(self.ot_solver, epsilon=self.epsilon, cost_fn=self.cost_fn, tau_a=self.tau_a, tau_b=self.tau_b, scale_cost=self.scale_cost, split_dim=self.split_dim, fused_penalty=self.fused_penalty, k_samples_per_x=self.k_noise_per_x)

        self.match_latent_fn = self._get_match_latent_fn(self.solver_latent_to_data, epsilon=self.epsilon_latent_to_data) if self.solver_latent_to_data is not None else lambda x, *_, **__: x
        if self.mlp_eta is not None:
            opt_eta = kwargs["opt_eta"] if "opt_eta" in kwargs else optax.adamw(learning_rate=1e-4, weight_decay=1e-10)
            self.state_eta = self.mlp_eta.create_train_state(self.rng, opt_eta, self.input_dim)
        if self.mlp_xi is not None:
            opt_xi = kwargs["opt_xi"] if "opt_xi" in kwargs else optax.adamw(learning_rate=1e-4, weight_decay=1e-10)
            self.state_xi = self.mlp_xi.create_train_state(self.rng, opt_xi, self.output_dim)

    def __call__(self, x: Union[jnp.array, collections.abc.Iterable], y: Union[jnp.array, collections.abc.Iterable], batch_size_source: int, batch_size_target: int) -> Any:
        if not hasattr(x, "shape"):
            x_loader = x
        else:
            x_loader = iter(
                tf.data.Dataset.from_tensor_slices(x).repeat().shuffle(buffer_size=10_000).batch(batch_size_source)
            )
        if not hasattr(y, "shape"):
            y_loader = y
        else:
            y_loader = iter(
                tf.data.Dataset.from_tensor_slices(y).repeat().shuffle(buffer_size=10_000).batch(batch_size_target)
            )

        batch: Dict[str, jnp.array] = {}
        for step in tqdm(range(self.iterations)):
            source_batch, target_batch = tfds.as_numpy(next(x_loader)), tfds.as_numpy(next(y_loader))
            self.rng, rng_time, rng_noise, rng_step_fn = jax.random.split(self.rng, 4)
            batch["source"] = source_batch
            batch["target"] = target_batch
            batch["time"] = jax.random.uniform(rng_time, (len(source_batch)*self.k_noise_per_x, 1))
            batch["noise"] = self.noise_fn(rng_noise, shape=(len(source_batch), self.k_noise_per_x)) * self.noise_std
            metrics, self.state_neural_net, self.state_bridge_net, self.state_eta, self.state_xi, eta_predictions, xi_predictions = self.step_fn(rng_step_fn, self.state_neural_net, self.state_bridge_net, batch, self.state_eta, self.state_xi)
            for key, value in metrics.items():
                self.metrics[key].append(value)

            if self.callback is not None and step % self.callback_iters == 0:
                self.callback(
                    source=batch["source"],
                    target=batch["target"],
                    eta_predictions=eta_predictions,
                    xi_predictions=xi_predictions,
                    **self.callback_kwargs,
                    **self.unbalanced_kwargs,
                )

    def _get_sinkhorn_match_fn(self, ot_solver: Any, epsilon: float, cost_fn: str, tau_a: float, tau_b: float, scale_cost: Any, k_samples_per_x: int) -> Callable:
        @partial(jax.jit, static_argnames=["ot_solver", "epsilon", "cost_fn", "scale_cost", "tau_a", "tau_b", "k_samples_per_x"])
        def match_pairs(key: jax.random.PRNGKeyArray, x: jnp.ndarray, y: jnp.ndarray, ot_solver: Any, tau_a: float, tau_b: float, epsilon: float, cost_fn: str, scale_cost: Any, k_samples_per_x:int) -> Tuple[jnp.array, jnp.array]:
            geom = pointcloud.PointCloud(x, y, epsilon=epsilon, scale_cost=scale_cost, cost_fn=cost_fn)
            out = ot_solver(linear_problem.LinearProblem(geom, tau_a=tau_a, tau_b=tau_b))
            inds_source, inds_target = sample_conditional_indices_from_tmap(key=key, tmat=out.matrix, k_samples_per_x=k_samples_per_x)
            return x[inds_source], y[inds_target], out.matrix.sum(axis=0), out.matrix.sum(axis=1)

        return jax.tree_util.Partial(match_pairs, ot_solver=ot_solver, epsilon=epsilon, cost_fn=cost_fn, tau_a=tau_a, tau_b=tau_b, scale_cost=scale_cost, k_samples_per_x=k_samples_per_x)

    def _get_gromov_match_fn(self, ot_solver: Any, cost_fn: str, tau_a: float, tau_b: float, epsilon: float, scale_cost: Any, split_dim: int, fused_penalty: float, k_samples_per_x:int) -> Callable:
        @partial(jax.jit, static_argnames=["ot_solver", "epsilon", "cost_fn", "scale_cost", "fused_penalty", "split_dim", "tau_a", "tau_b", "k_samples_per_x"])
        def match_pairs(key: jax.random.PRNGKeyArray, x: Tuple[jnp.ndarray, jnp.ndarray], y: Tuple[jnp.ndarray, jnp.ndarray], ot_solver: Any, epsilon: float, tau_a: float, tau_b: float, cost_fn: str, scale_cost, fused_penalty: float, k_samples_per_x: int, split_dim: int = 0) -> Tuple[jnp.array, jnp.array]:
            geom_xx = pointcloud.PointCloud(x=x[..., split_dim:], y=x[..., split_dim:], cost_fn=cost_fn, scale_cost=scale_cost)
            geom_yy = pointcloud.PointCloud(x=y[..., split_dim:], y=y[..., split_dim:], cost_fn=cost_fn, scale_cost=scale_cost)
            if split_dim > 0:
                geom_xy = pointcloud.PointCloud(x=x[..., :split_dim], y=y[..., :split_dim], cost_fn=cost_fn, scale_cost=scale_cost)
            else:
                geom_xy = None
            prob = quadratic_problem.QuadraticProblem(geom_xx, geom_yy, geom_xy, fused_penalty=fused_penalty, tau_a=tau_a, tau_b=tau_b)
            out = ot_solver(prob)
            inds_source, inds_target = sample_conditional_indices_from_tmap(key=key, tmat=out.matrix, k_samples_per_x=k_samples_per_x)
            return x[inds_source], y[inds_target], out.matrix.sum(axis=0), out.matrix.sum(axis=1)

        return jax.tree_util.Partial(match_pairs, ot_solver=ot_solver, epsilon=epsilon, cost_fn=cost_fn, tau_a=tau_a, tau_b=tau_b, scale_cost=scale_cost, split_dim=split_dim, fused_penalty=fused_penalty, k_samples_per_x=k_samples_per_x)
           
    def _get_match_fn_graph(self, ot_solver: Any, epsilon: float, k_neighbors: int, tau_a: float, tau_b: float, scale_cost: Any, **kwargs) -> Callable:

        def get_nearest_neighbors(
            X: jnp.ndarray, Y: jnp.ndarray, len_x: int, k: int = 30  # type: ignore[name-defined]
        ) -> Tuple[jnp.ndarray, jnp.ndarray]:  # type: ignore[name-defined]
            concat = jnp.concatenate((X,Y), axis=0)
            pairwise_euclidean_distances = pointcloud.PointCloud(concat, concat).cost_matrix
            distances, indices = jax.lax.approx_min_k(pairwise_euclidean_distances, k=k, recall_target=0.95, aggregate_to_topk=True)
            return distances, indices

        def create_cost_matrix(X: jnp.array, Y: jnp.array, k_neighbors: int, **kwargs: Any) -> jnp.array:
            distances, indices = get_nearest_neighbors(X, Y, len(X), k_neighbors)
            a = jnp.zeros((len(X)+ len(Y), len(X)+ len(Y)))
            adj_matrix = a.at[jnp.repeat(jnp.arange(len(X)+len(Y)), repeats=k_neighbors).flatten(), indices.flatten()].set(distances.flatten())
            return graph.Graph.from_graph(adj_matrix, normalize=kwargs.pop("normalize", True), **kwargs).cost_matrix
        
        def match_pairs(key: jax.random.PRNGKeyArray, x: jnp.ndarray, y: jnp.ndarray, ot_solver: Any, epsilon: float, k_neighbors: int, n_samples: Optional[int], **kwargs) -> Tuple[jnp.array, jnp.array, jnp.ndarray, jnp.ndarray]:
            cm = create_cost_matrix(x, y, k_neighbors, **kwargs)
            geom = geometry.Geometry(cost_matrix=cm, epsilon=epsilon, scale_cost=scale_cost)
            out = ot_solver(linear_problem.LinearProblem(geom, tau_a=tau_a, tau_b=tau_b))
            inds_source, inds_target = sample_indices_from_tmap(key=key, tmat=out.matrix, n_samples=n_samples)
            return x[inds_source], y[inds_target], out.matrix.sum(axis=0), out.matrix.sum(axis=1)

        return jax.tree_util.Partial(match_pairs, ot_solver=ot_solver, epsilon=epsilon, k_neighbors=k_neighbors, tau_a=tau_a, tau_b=tau_b, **kwargs)

    def _get_match_latent_fn(self, ot_solver: Type[sinkhorn.Sinkhorn], epsilon: float) -> Callable:
        @partial(jax.jit, static_argnames=["ot_solver", "epsilon"])
        def match_latent_to_data(key: jax.random.PRNGKeyArray, x: jnp.ndarray, y: jnp.ndarray, ot_solver: Any, epsilon: float) -> Tuple[jnp.array, jnp.array]:
            geom = pointcloud.PointCloud(x, y, epsilon=epsilon, scale_cost="mean")
            out = ot_solver(linear_problem.LinearProblem(geom))
            inds_source = jax.vmap(
                lambda tmap: jax.random.categorical(key, logits=tmap, shape=(1,))
            )(out.matrix)

            return x[jnp.squeeze(inds_source)]

        return jax.tree_util.Partial(match_latent_to_data, ot_solver=ot_solver, epsilon=epsilon)

    def _get_step_fn(self) -> Callable:
        
        
        def loss_fn(params_mlp: jnp.array, params_bridge_net: jnp.array, apply_fn_mlp: Callable, apply_fn_bridge: Callable, batch: Dict[str, jnp.array], rng_latent_match):
            def psi_t(x_0, x_1, t):
                return (1 -  t) * x_0 + t * x_1
        
    
            mu_0, _ = apply_fn_bridge({"params": params_bridge_net}, condition=batch["source"])
            mu_noisy = mu_0 + batch["noise"] * 1.0 # todo: match latent to target with OT
            #noise_batch = self.match_latent_fn(rng_latent_match, batch["noise"], batch["target"])
            #batch["noise"] = noise_batch
            #psi_t_eval = psi_t(mu_noisy, batch["target"], batch["time"], 0.0001, sigma_0)
            #mlp_pred = apply_fn_mlp({"params": params_mlp}, t=batch["time"], x=psi_t_eval, condition=batch["source"])
            #d_psi = batch["target"] - mu_noisy
            psi_t_eval = psi_t(mu_noisy, batch["target"], batch["time"])
            mlp_pred = apply_fn_mlp({"params": params_mlp}, t=batch["time"], latent=psi_t_eval, condition=batch["source"])
            d_psi = batch["target"] - mu_noisy
        
            if len(mlp_pred.shape) == 1:
                mlp_pred = mlp_pred[:, None]
            return jnp.mean(optax.l2_loss(mlp_pred, d_psi))

        def loss_a_fn(
            params_eta: Optional[jnp.ndarray], apply_fn_eta: Optional[Callable], x: jnp.ndarray, a: jnp.ndarray
        ) -> float:
            eta_predictions = apply_fn_eta({"params": params_eta}, x)
            return optax.l2_loss(eta_predictions[:, 0], a).sum(), eta_predictions

        def loss_b_fn(
            params_xi: Optional[jnp.ndarray], apply_fn_xi: Optional[Callable], x: jnp.ndarray, b: jnp.ndarray
        ) -> float:
            xi_predictions = apply_fn_xi({"params": params_xi}, x)
            return optax.l2_loss(xi_predictions[:, 0], b).sum(), xi_predictions

        @jax.jit
        def step_fn(key: jax.random.PRNGKeyArray, state_neural_net: TrainState, state_bridge_net: TrainState, batch: Dict[str, jnp.array], state_eta: Optional[TrainState] = None, state_xi: Optional[TrainState] = None):
            rng_match, rng_noise, rng_latent_match = jax.random.split(key, 3)
            source_batch, target_batch, a, b = self.match_fn(rng_match, batch["source"], batch["target"])
            rng_noise = jax.random.split(rng_noise, (len(target_batch)))
            noise_matched, conditional_target = jax.vmap(sinkhorn_fn, 0, 0)(rng_noise, batch["noise"], target_batch)

            batch["source"] = jnp.reshape(source_batch, (len(source_batch), -1))
            batch["target"] = jnp.reshape(conditional_target, (len(source_batch), -1))
            batch["noise"] = jnp.reshape(noise_matched, (len(source_batch), -1))

            grad_fn = jax.value_and_grad(loss_fn, argnums=[0,1], has_aux=False)
            loss, (grads_mlp, grads_bridge_net) = grad_fn(state_neural_net.params, state_bridge_net.params, state_neural_net.apply_fn, state_bridge_net.apply_fn, batch, rng_latent_match)
            metrics = {}
            metrics["loss"] = loss

            if state_eta is not None:
                grad_a_fn = jax.value_and_grad(loss_a_fn, argnums=0, has_aux=True)
                (loss_a, eta_predictions), grads_eta = grad_a_fn(
                    state_eta.params,
                    state_eta.apply_fn,
                    batch["source"],
                    a * len(batch["source"]),
                )
                new_state_eta = state_eta.apply_gradients(grads=grads_eta)
                metrics["loss_eta"] = loss_a

            else:
                new_state_eta = eta_predictions = None
            if state_xi is not None:
                grad_b_fn = jax.value_and_grad(loss_b_fn, argnums=0, has_aux=True)
                (loss_b, xi_predictions), grads_xi = grad_b_fn(
                    state_xi.params,
                    state_xi.apply_fn,
                    batch["target"],
                    b * len(batch["target"]),
                )
                new_state_xi = state_xi.apply_gradients(grads=grads_xi)
                metrics["loss_xi"] = loss_b
            else:
                new_state_xi = xi_predictions = None

            return metrics, state_neural_net.apply_gradients(grads=grads_mlp), state_bridge_net.apply_gradients(grads=grads_bridge_net), new_state_eta, new_state_xi, eta_predictions, xi_predictions

        return step_fn

    def transport(self, source: jnp.array, seed: int = 0, diffeqsolve_kwargs: Dict[str, Any] = types.MappingProxyType({})) -> Union[jnp.array, diffrax.Solution]:
        diffeqsolve_kwargs = dict(diffeqsolve_kwargs)
        rng = jax.random.PRNGKey(seed)
        latent_shape = (len(source),)
        latent_batch = self.noise_fn(rng, shape=latent_shape) * self.noise_std

        mu_0, sigma_0 = self.state_bridge_net.apply_fn({"params": self.state_bridge_net.params}, condition=source)

        mu_noisy = mu_0 + latent_batch * sigma_0 
        apply_fn_partial = partial(
            self.state_neural_net.apply_fn, condition=source
        )
        solution = diffrax.diffeqsolve(
            diffrax.ODETerm(lambda t, y, *args: apply_fn_partial({"params": self.state_neural_net.params}, t=t, latent=y)),
            diffeqsolve_kwargs.pop("solver", diffrax.Tsit5()),
            t0=0,
            t1=1,
            dt0=diffeqsolve_kwargs.pop("dt0", None),
            y0=mu_noisy,#latent_batch,  
            stepsize_controller=diffeqsolve_kwargs.pop("stepsize_controller", diffrax.PIDController(rtol=1e-3, atol=1e-6)),
            **diffeqsolve_kwargs
        )
        return solution.ys, solution


class OriginalFlowMatching:
    def __init__(self, neural_net: Type[ModelBase], input_dim: int, iterations: int, opt: Optional[Any]= None, sig_min: float = 0.001) -> None:
        self.optimizer = optax.adamw(learning_rate=1e-4, weight_decay=1e-10) if opt is None else opt
        self.input_dim = input_dim
        self.rng = jax.random.PRNGKey(0)
        self.iterations = iterations
        self.sig_min = sig_min
        self.metrics = {"loss": []}
        self.neural_net = neural_net

        self.state_neural_net: Optional[TrainState] = None

        self.setup()
    
    def setup(self) -> None:
        self.state_neural_net = self.neural_net.create_train_state(self.rng, self.optimizer, self.input_dim, self.input_dim)
        
        self.step_fn = self._get_step_fn()
        self.match_fn = self._get_match_fn()

    def __call__(self, x: jnp.array, y: jnp.array, batch_size_source: int, batch_size_target: int) -> Any:
        x_loader = iter(tf.data.Dataset.from_tensor_slices(x).repeat().shuffle(buffer_size=10_000).batch(batch_size_source))
        y_loader = iter(tf.data.Dataset.from_tensor_slices(y).repeat().shuffle(buffer_size=10_000).batch(batch_size_target))

        batch: Dict[str, jnp.array] = {}
        for step in tqdm(range(self.iterations)):

            x_batch, y_batch = tfds.as_numpy(next(x_loader)), tfds.as_numpy(next(y_loader))
            self.rng, rng_time, rng_noise = jax.random.split(self.rng, 3)
            rng_noise = jax.random.split(rng_noise, len(x_batch))
            x_batch, y_batch = self.match_fn(x_batch, y_batch)
            noise_matched, conditional_target = jax.vmap(sinkhorn_fn, 0, 0)(rng_noise, y_batch.shape), y_batch, jnp.reshape(batch["noise"])
            batch["source"] = x_batch.flatten()
            batch["target"] = conditional_target.flatten()
            batch["noise"] = noise_matched.flatten()
            batch["time"] = jax.random.uniform(rng_time, (len(x_batch)*self.k_noise_per_x, 1))
            loss, self.state_neural_net = self.step_fn(self.state_neural_net, batch, self.sig_min)
            self.metrics["loss"].append(loss)

    def _get_match_fn(self) -> Callable:
        
        @jax.jit
        def match_pairs(x: jnp.ndarray, y: jnp.ndarray) -> Tuple[jnp.array, jnp.array]:
            geom = pointcloud.PointCloud(x, y, epsilon=1e-1, scale_cost="mean")
            out = sinkhorn.Sinkhorn(threshold=1e-2)(
                linear_problem.LinearProblem(geom)
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

            return val, state_neural_net.apply_gradients(grads=grads_mlp)
        
        return step_fn
    
        

    def transport(self, x: jnp.array) -> Union[jnp.array, Any]:
        solution = diffrax.diffeqsolve(
                    diffrax.ODETerm(lambda t,y,args: self.state_neural_net.apply_fn({"params":self.state_neural_net.params}, t=t, x=y)),
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
    t_embed_dim: int
    condition_embed_dim: int
    is_potential: bool = False
    act_fn: Callable[[jnp.ndarray], jnp.ndarray] = nn.silu
    n_frequencies: int = 1

    def time_encoder(self, t: jnp.array) -> jnp.array:
        freq = 2 * jnp.arange(self.n_frequencies) * jnp.pi
        t = freq * t  # [..., None]
        return jnp.concatenate((jnp.cos(t), jnp.sin(t)), axis=-1)

    @nn.compact
    def __call__(self, t: float, x: jnp.ndarray, condition: jnp.ndarray) -> jnp.ndarray:  # noqa: D102

        embedded_condition = Block(dim=self.condition_embed_dim, out_dim=self.output_dim, activation_fn=self.act_fn)(condition)
        t = jnp.full(shape=(len(x), 1), fill_value=t)
        #t = jnp.full(shape=(len(embedded_condition), 1), fill_value=t)
        t = self.time_encoder(t)
        t = Block(dim=self.t_embed_dim, out_dim=self.t_embed_dim, activation_fn=self.act_fn)(t)
        
        z = jnp.concatenate((x, embedded_condition, t), axis=-1)

        z = Block(dim=self.condition_embed_dim,out_dim=self.condition_embed_dim, activation_fn=self.act_fn)(z)
        Wx = nn.Dense(self.output_dim, use_bias=True, name="final_layer")
        z = Wx(z)

        return z

    def create_train_state(
        self, rng: jax.random.PRNGKeyArray, optimizer: optax.OptState, source_dim: int, latent_dim: int, **kwargs: Any
    ) -> NeuralTrainState:
        params = self.init(rng, jnp.ones((1,1)), jnp.ones((1,source_dim)), jnp.ones((1,source_dim)))["params"]
        return train_state.TrainState.create(apply_fn=self.apply, params=params, tx=optimizer)



class MLP_FM_VAE(ModelBase):
    output_dim: int
    t_embed_dim: int
    condition_embed_dim: int
    joint_hidden_dim: int
    is_potential: bool = False
    act_fn: Callable[[jnp.ndarray], jnp.ndarray] = nn.silu
    act_fn_var: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu
    latent_dim: int = 0
    n_frequencies: int = 1

    def time_encoder(self, t: jnp.array) -> jnp.array:
        freq = 2 * jnp.arange(self.n_frequencies) * jnp.pi
        t = freq * t  # [..., None]
        return jnp.concatenate((jnp.cos(t), jnp.sin(t)), axis=-1)
    

    @nn.compact
    def __call__(self, t: float, condition: jnp.ndarray, flow: jnp.ndarray, latent: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:  # noqa: D102
        
        embedded_condition = Block(dim=self.condition_embed_dim, out_dim=self.output_dim, activation_fn=self.act_fn)(condition)

        t = jnp.full(shape=(len(embedded_condition), 1), fill_value=t)
        t = self.time_encoder(t)
        t = Block(dim=self.t_embed_dim, out_dim=self.t_embed_dim, activation_fn=self.act_fn)(t)
        
        variance = Block(dim=self.condition_embed_dim, out_dim=1, activation_fn=self.act_fn_var)(jnp.concatenate((embedded_condition, t), axis=-1))

        embedded_condition_noisy = embedded_condition + latent * variance

        z = jnp.concatenate((flow, t), axis=-1)
        z = Block(num_layers=3, dim=self.joint_hidden_dim, out_dim=self.joint_hidden_dim, activation_fn=self.act_fn)(z)

        Wx = nn.Dense(self.output_dim, use_bias=True, name="final_layer")
        z = Wx(z)

        return z, embedded_condition, variance
        


    def create_train_state(
        self, rng: jax.random.PRNGKeyArray, optimizer: optax.OptState, source_dim: int, latent_dim: int, **kwargs: Any
    ) -> NeuralTrainState:
        params = self.init(rng, jnp.ones((1,1)), jnp.ones((1,source_dim)), jnp.ones((1,latent_dim)), jnp.ones((1,latent_dim)))["params"]
        return train_state.TrainState.create(apply_fn=self.apply, params=params, tx=optimizer)

class MLP_FM_VAE_all_dims(ModelBase):
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
        #latent = Block(dim=self.condition_embed_dim, out_dim=self.output_dim, activation_fn=self.act_fn)(latent)
        variance = Block(dim=self.condition_embed_dim, out_dim=1, activation_fn=self.act_fn)(jnp.concatenate((condition, t), axis=-1))

        condition = condition + latent * variance

        z = jnp.concatenate((condition, t), axis=-1)
        z = Block(num_layers=3, dim=self.joint_hidden_dim, out_dim=self.joint_hidden_dim, activation_fn=self.act_fn)(z)

        Wx = nn.Dense(self.output_dim, use_bias=True, name="final_layer")
        z = Wx(z)

        return z


    def create_train_state(
        self, rng: jax.random.PRNGKeyArray, optimizer: optax.OptState, source_dim: int, latent_dim: int, **kwargs: Any
    ) -> NeuralTrainState:
        params = self.init(rng, jnp.ones((1,1)), jnp.ones((1,source_dim)), jnp.ones((1,latent_dim)))["params"]
        return train_state.TrainState.create(apply_fn=self.apply, params=params, tx=optimizer)


class MLP_FM_VAE2(ModelBase):
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
        #variance = Block(dim=self.condition_embed_dim, out_dim=1, activation_fn=self.act_fn)(condition)

        #condition = condition + latent * variance

        z = jnp.concatenate((condition, t, latent), axis=-1)
        z = Block(num_layers=3, dim=self.joint_hidden_dim, out_dim=self.joint_hidden_dim, activation_fn=self.act_fn)(z)

        Wx = nn.Dense(self.output_dim, use_bias=True, name="final_layer")
        z = Wx(z)

        return z


    def create_train_state(
        self, rng: jax.random.PRNGKeyArray, optimizer: optax.OptState, source_dim: int, latent_dim: int, **kwargs: Any
    ) -> NeuralTrainState:
        params = self.init(rng, jnp.ones((1,1)), jnp.ones((1,source_dim)), jnp.ones((1,latent_dim)))["params"]
        return train_state.TrainState.create(apply_fn=self.apply, params=params, tx=optimizer)


class Bridge_MLP(ModelBase):
    output_dim: int
    t_embed_dim: int
    condition_embed_dim: int
    is_potential: bool = False
    act_fn: Callable[[jnp.ndarray], jnp.ndarray] = nn.silu


    @nn.compact
    def __call__(self, condition: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:  # noqa: D102
        
        condition = Block(dim=self.condition_embed_dim, out_dim=self.condition_embed_dim, activation_fn=self.act_fn)(condition)
        mu = Block(dim=self.condition_embed_dim, out_dim=self.output_dim)(condition)
        log_var = Block(dim=self.condition_embed_dim, out_dim=1)(condition)
        std = jnp.sqrt(jnp.exp(log_var))
        return mu, std

    def create_train_state(
        self, rng: jax.random.PRNGKeyArray, optimizer: optax.OptState, output_dim: int,
    ) -> NeuralTrainState:
        params = self.init(rng, jnp.ones((1,output_dim)))["params"]
        return train_state.TrainState.create(apply_fn=self.apply, params=params, tx=optimizer)


class Bridge_MLP_with_t(ModelBase):
    output_dim: int
    t_embed_dim: int
    condition_embed_dim: int
    is_potential: bool = False
    act_fn: Callable[[jnp.ndarray], jnp.ndarray] = nn.silu


    @nn.compact
    def __call__(self, t: jnp.ndarray, condition: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:  # noqa: D102
        t = jnp.full(shape=(len(condition), 1), fill_value=t)
        condition = Block(dim=self.condition_embed_dim, out_dim=self.condition_embed_dim, activation_fn=self.act_fn)(condition)
        t = Block(dim=self.condition_embed_dim, out_dim=self.condition_embed_dim, activation_fn=self.act_fn)(t)
        mu = Block(dim=self.condition_embed_dim, out_dim=self.output_dim)(jnp.concatenate((condition, t), axis=-1))
        log_var = Block(dim=self.condition_embed_dim, out_dim=1)(jnp.concatenate((condition, t), axis=-1))
        std = jnp.sqrt(jnp.exp(log_var))
        return mu, std

    def create_train_state(
        self, rng: jax.random.PRNGKeyArray, optimizer: optax.OptState, output_dim: int,
    ) -> NeuralTrainState:
        params = self.init(rng, jnp.ones((1,1)), jnp.ones((1,output_dim)))["params"]
        return train_state.TrainState.create(apply_fn=self.apply, params=params, tx=optimizer)