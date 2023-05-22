# jax
from jax.config import config
#config.update("jax_debug_nans", True)
from abc import ABC, abstractmethod
from types import MappingProxyType
from typing import Any, Callable, Dict, Literal, Mapping, Optional

import flax.linen as nn
import jax
import jax.numpy as jnp
import jax.random as random
import jax.scipy as jsp
import jax_dataloader as jdl
import matplotlib.pyplot as plt
import optax

# ott
import ott
import ott.geometry.costs as costs
from flax.training.train_state import TrainState
from ott.geometry import pointcloud
from ott.geometry.geometry import Geometry
from ott.math.utils import logsumexp as lse
from ott.problems.linear import linear_problem
from ott.problems.linear.potentials import EntropicPotentials
from ott.solvers.linear import acceleration, sinkhorn
from ott.solvers.linear.sinkhorn import SinkhornOutput
from tqdm import tqdm

from entot.models.utils import MLP, DataLoader, _concatenate


class DualPotentialModel(ABC):
    def __init__(self, epsilon: float, rng: Optional[jax.random.PRNGKeyArray] = None) -> None:
        self.epsilon = epsilon
        self.rng = jax.random.PRNGKey(0) if rng is None else rng

    @abstractmethod
    def __call__(self, *args: Any, **kwds: Any) -> Any:
        pass

    @abstractmethod
    def transport(self, *args: Any, **kwargs: Any) -> Any:
        pass


class DiscreteOTModel(DualPotentialModel):
    def __init__(self, epsilon: float, sinkhorn_solver_kwargs: Dict[str, Any] = MappingProxyType({}), **kwargs):
        super().__init__(epsilon=epsilon, **kwargs)
        self.solver = sinkhorn.Sinkhorn(**sinkhorn_solver_kwargs)
        self.output: Optional[SinkhornOutput] = None
        self.dual_potentials: Optional[EntropicPotentials] = None
        self.source: Optional[jnp.ndarray] = None
        self.target: Optional[jnp.ndarray] = None

    def __call__(self, x: jnp.ndarray, y: jnp.ndarray) -> Any:
        self.source = x
        self.target = y
        self.output = self.solver(linear_problem.LinearProblem(pointcloud.PointCloud(x, y, epsilon=self.epsilon)))
        self.dual_potentials = self.output.to_dual_potentials()

    def transport(self, x: Optional[jnp.ndarray] = None) -> jnp.ndarray:
        x = x if x is not None else self.source
        return self.dual_potentials.transport(x)

    def get_hessians_f(self, x: jnp.ndarray) -> jnp.ndarray:
        # x = jnp.atleast_2d(x)
        # hess_eval = jax.hessian(lambda source, target, sinkhorn_output, epsilon: 0.5 * jnp.inner(source, source) - self.evaluate_f(source, target, sinkhorn_output, epsilon), argnums=0)
        # vmap_hess_eval = jax.vmap(lambda x: hess_eval(x, self.target, self.output, self.epsilon))
        hess_eval = jax.hessian(lambda x: self.dual_potentials.f(x))
        vmap_hess_eval = jax.vmap(lambda x: hess_eval(x))
        return vmap_hess_eval(x)

    def get_hessians_g(self, y: jnp.ndarray) -> jnp.ndarray:
        # y = jnp.atleast_2d(y)
        # hess_eval = jax.hessian(lambda source, target, sinkhorn_output, epsilon: 0.5 * jnp.inner(target, target) - self.evaluate_g(source, target, sinkhorn_output, epsilon), argnums=1)
        # vmap_hess_eval = jax.vmap(lambda y: hess_eval(self.source, y, self.output, self.epsilon))
        hess_eval = jax.hessian(lambda y: self.dual_potentials.g(y))
        vmap_hess_eval = jax.vmap(lambda y: hess_eval(y))
        return vmap_hess_eval(y)

    # @staticmethod
    def evaluate_f(self, x: jnp.ndarray, y: jnp.ndarray, out: SinkhornOutput, epsilon: float) -> float:
        g = out.g
        geom = pointcloud.PointCloud(x, y)
        cost = geom.cost_matrix
        z = (g - cost) / epsilon
        lse = -epsilon * jsp.special.logsumexp(z, axis=-1)
        return jnp.squeeze(lse)

    @staticmethod
    def evaluate_g(x: jnp.ndarray, y: jnp.ndarray, out: SinkhornOutput, epsilon: float) -> float:
        f = out.f
        geom = pointcloud.PointCloud(x, y)
        cost = jnp.transpose(geom.cost_matrix)
        z = (f - cost) / epsilon
        lse = -epsilon * jsp.special.logsumexp(z, axis=-1)
        return jnp.squeeze(lse)


class NeuralDualPotentialModel(DualPotentialModel, ABC):
    def __init__(
        self,
        epsilon: float,
        batch_size_source: Optional[int] = None,
        batch_size_target: Optional[int] = None,
        iterations: Optional[int] = None,
        input_dim: Optional[int] = None,
        rng: Optional[jax.random.PRNGKeyArray] = None,
    ) -> None:
        super().__init__(epsilon=epsilon, rng=rng)
        self.batch_size_source = batch_size_source
        self.batch_size_target = batch_size_target
        self.iterations = iterations
        self.input_dim = input_dim
        self.x_loader: Optional[DataLoader] = None
        self.y_loader: Optional[DataLoader] = None

    def __call__(self, x: jnp.ndarray, y: jnp.ndarray) -> Any:
        batch: Dict[str, jnp.ndarray] = {}
        self.x_loader = DataLoader(x, batch_size=self.batch_size_source)
        self.y_loader = DataLoader(y, batch_size=self.batch_size_target)
        for _ in tqdm(range(self.iterations)):
            self.rng, rng_x, rng_y = jax.random.split(self.rng, 3)
            batch["source"] = self.x_loader(rng_x)
            batch["target"] = self.y_loader(rng_y)
            metrics = self.train(batch)
            for key, value in metrics.items():
                self.metrics[key].append(value)

    @abstractmethod
    def setup(self):
        pass


class SeguyModel(NeuralDualPotentialModel):
    def __init__(
        self,
        epsilon: float,
        batch_size_source: int,
        batch_size_target: int,
        iterations: int,
        input_dim: int,
        cost_fn: Optional[str] = None,
        **kwargs
    ) -> None:
        super().__init__(
            epsilon=epsilon,
            batch_size_source=batch_size_source,
            batch_size_target=batch_size_target,
            iterations=iterations,
            input_dim=input_dim,
        )
        self.cost_fn = cost_fn
        self.state_f: Optional[TrainState] = None
        self.state_g: Optional[TrainState] = None
        self.state_mlp: Optional[TrainState] = None
        self.train_step: Any = None
        self.train_step_mlp: Any = None
        self.metrics = {"obj": [], "term": [], "f_mean": [], "g_mean": []}
        self.mlp_metrics = {"loss": [], "d_xy": [], "H": []}

        g = MLP(dim_hidden=[256, 256], is_potential=True, act_fn=nn.relu)

        f = MLP(dim_hidden=[256, 256], is_potential=True, act_fn=nn.relu)

        mlp = MLP(
            dim_hidden=[128, 256, 128, 128],
            is_potential=False,
            act_fn=nn.relu,
        )

        opt_f = optax.adamw(learning_rate=1e-3)
        opt_g = optax.adamw(learning_rate=1e-3)
        opt_mlp = optax.adamw(learning_rate=1e-3)

        self.setup(f=f, g=g, mlp=mlp, opt_f=opt_f, opt_g=opt_g, opt_mlp=opt_mlp, input_dim=self.input_dim, rng=self.rng)

    def setup(self, f, g, mlp, opt_f, opt_g, opt_mlp, input_dim, rng):
        rng, rng_f, rng_g, rng_mlp = jax.random.split(rng, 4)

        self.state_f = f.create_train_state(rng_f, opt_f, input_dim)
        self.state_g = g.create_train_state(rng_g, opt_g, input_dim)
        self.state_mlp = mlp.create_train_state(rng_mlp, opt_mlp, input_dim)
        self.train_step = self.get_train_step()

    def train(self, batch: Dict[str, jnp.ndarray]) -> Dict[str, Any]:
        self.state_f, self.state_g, metrics = self.train_step(
            self.state_f, self.state_g, batch, self.epsilon, self.cost_fn
        )
        return metrics

    def get_train_step(self) -> Callable:
        def loss(
            params_f: jnp.ndarray,
            params_g: jnp.ndarray,
            state_f: TrainState,
            state_g: TrainState,
            batch: Dict[str, jnp.ndarray],
            epsilon: float,
            cost_fn: Optional[str] = None,
        ):
            f_x = state_f.apply_fn({"params": params_f}, batch["source"])
            g_y = state_g.apply_fn({"params": params_g}, batch["target"])

            pc = pointcloud.PointCloud(batch["source"], batch["target"], cost_fn=cost_fn)
            c_xy = pc.cost_matrix
            t_1 = jnp.mean(f_x)
            t_2 = jnp.mean(g_y)
            t_3 = -epsilon * jnp.mean(jnp.exp((jnp.reshape(f_x, (-1, 1)) + jnp.reshape(g_y, (1, -1)) - c_xy) / epsilon))
            metrics = t_3, t_1, t_2
            return -(t_1 + t_2 + t_3), metrics

        def step_fn(
            state_f: TrainState,
            state_g: TrainState,
            batch: Dict[str, jnp.ndarray],
            epsilon: float,
            cost_fn: Optional[str] = None,
        ):
            grad_fn = jax.value_and_grad(loss, argnums=[0, 1], has_aux=True)
            (obj_value, metrics), grads = grad_fn(
                state_f.params, state_g.params, state_f, state_g, batch, epsilon, cost_fn
            )
            term, g_mean, f_mean = metrics
            grads_f, grads_g = grads
            metrics = {"obj": obj_value, "term": term, "f_mean": f_mean, "g_mean": g_mean}

            return state_f.apply_gradients(grads=grads_f), state_g.apply_gradients(grads=grads_g), metrics

        return step_fn

    def learn_barycentric_projection(self, n_iters=1000):
        self.train_step_mlp = self.get_train_step_mlp()
        batch: Dict[str, jnp.ndarray] = {}
        for _ in tqdm(range(n_iters)):
            self.rng, rng_x, rng_y = jax.random.split(self.rng, 3)
            batch["source"] = self.x_loader(rng_x)
            batch["target"] = self.y_loader(rng_y)
            self.state_mlp, metrics = self.train_step_mlp(
                self.state_mlp, self.state_f, self.state_g, batch, self.epsilon, self.cost_fn
            )
            for key, value in metrics.items():
                self.mlp_metrics[key].append(value)

        return metrics

    def get_train_step_mlp(self) -> Callable:
        def loss(
            params_mlp: jnp.ndarray,
            state_mlp: TrainState,
            state_f: TrainState,
            state_g: TrainState,
            batch: Dict[str, jnp.ndarray],
            epsilon: float,
            cost_fn: Optional[str] = None,
        ):
            mlp_x = state_mlp.apply_fn({"params": params_mlp}, batch["source"])
            # loss = jnp.sum(mlp_x.flatten())
            f_x = state_f.apply_fn({"params": state_f.params}, batch["source"])
            g_y = state_g.apply_fn({"params": state_g.params}, batch["target"])
            pc = pointcloud.PointCloud(batch["source"], batch["target"], cost_fn=cost_fn)
            c_xy = pc.cost_matrix
            H = jnp.exp((jnp.reshape(f_x, (-1, 1)) + jnp.reshape(g_y, (1, -1)) - c_xy) / epsilon)
            pc2 = pointcloud.PointCloud(mlp_x, batch["target"])
            d_xy = pc2.cost_matrix
            loss = jnp.sum(jnp.multiply(d_xy, H))
            loss = jnp.sum(mlp_x.flatten())
            metrics = d_xy.flatten(), H.flatten()
            return loss, metrics

        def step_fn(
            state_mlp: TrainState,
            state_f: TrainState,
            state_g: TrainState,
            batch: Dict[str, Any],
            epsilon: float,
            cost_fn: Optional[str] = None,
        ):
            grad_fn = jax.value_and_grad(loss, has_aux=True)
            (obj_value, metrics), grads_mlp = grad_fn(
                state_mlp.params, state_mlp, state_f, state_g, batch, epsilon, cost_fn
            )
            metrics = {"loss": obj_value, "d_xy": metrics[0], "H": metrics[1]}
            return state_mlp.apply_gradients(grads=grads_mlp), metrics

        return step_fn

    def transport_mlp(self, x: Optional[jnp.ndarray] = None) -> jnp.ndarray:
        if self.state_mlp is None:
            raise ValueError("Not trained yet.")
        x = x if x is not None else self.x_loader.data
        return self.state_mlp.apply_fn({"params": self.state_mlp.params}, x)

    def transport(self, x: Optional[jnp.ndarray] = None, forward: bool = True) -> Any:
        if forward:
            return x - jax.vmap(
                jax.grad(lambda x: self.state_f.apply_fn({"params": self.state_f.params}, x), argnums=0)
            )(x)
        return x - jax.vmap(jax.grad(lambda x: self.state_g.apply_fn({"params": self.state_g.params}, x), argnums=0))(x)

    def get_hessians_f(self, x: jnp.ndarray) -> jnp.ndarray:
        x = jnp.atleast_2d(x)
        hess_eval = jax.hessian(
            lambda x: 0.5 * jnp.inner(x, x) - self.state_f.apply_fn({"params": self.state_f.params}, x), argnums=0
        )
        vmap_hess_eval = jax.vmap(lambda x: hess_eval(x))
        return vmap_hess_eval(x)

    def get_hessians_g(self, x: jnp.ndarray) -> jnp.ndarray:
        x = jnp.atleast_2d(x)
        hess_eval = jax.hessian(
            lambda x: 0.5 * jnp.inner(x, x) - self.state_g.apply_fn({"params": self.state_f.params}, x), argnums=0
        )
        vmap_hess_eval = jax.vmap(lambda x: hess_eval(x))
        return vmap_hess_eval(x)


class SinkhornModel(NeuralDualPotentialModel):
    def __init__(
        self,
        epsilon: float,
        include_distribution_match: bool,
        batch_size_source: int,
        batch_size_target: int,
        iterations: int,
        input_dim: int,
        cost_fn: Optional[str] = None,
        **kwargs
    ) -> None:
        super().__init__(
            epsilon=epsilon,
            batch_size_source=batch_size_source,
            batch_size_target=batch_size_target,
            iterations=iterations,
            input_dim=input_dim,
        )
        self.cost_fn = cost_fn
        self.include_distribution_match = include_distribution_match
        self.state_f: Optional[TrainState] = None
        self.state_g: Optional[TrainState] = None
        self.train_step: Any = None
        self.metrics = {"obj": [], "f": [], "g": [], "div_transported_source": [], "div_transported_target": []}

        g = MLP(dim_hidden=[256, 256], is_potential=True, act_fn=nn.gelu)

        f = MLP(dim_hidden=[256, 256], is_potential=True, act_fn=nn.gelu)

        opt_f = optax.adamw(learning_rate=1e-3)
        opt_g = optax.adamw(learning_rate=1e-3)

        self.setup(f=f, g=g, opt_f=opt_f, opt_g=opt_g, input_dim=self.input_dim, rng=self.rng)

    def setup(self, f, g, opt_f, opt_g, input_dim, rng):
        rng, rng_f, rng_g = jax.random.split(rng, 3)

        self.state_f = f.create_train_state(rng_f, opt_f, input_dim)
        self.state_g = g.create_train_state(rng_g, opt_g, input_dim)
        self.train_step = self.get_train_step()

    def train(self, batch: Dict[str, jnp.ndarray]) -> Dict[str, Any]:
        self.state_f, self.state_g, metrics = self.train_step(
            self.state_f, self.state_g, batch, self.epsilon, self.include_distribution_match
        )
        return metrics

    def get_train_step(self) -> Callable:
        def loss(
            params_f: jnp.ndarray,
            params_g: jnp.ndarray,
            state_f: TrainState,
            state_g: TrainState,
            batch: Dict[str, jnp.ndarray],
            epsilon: float,
            include_distribution_match: bool,
        ):
            geom = pointcloud.PointCloud(batch["source"], batch["target"], epsilon=epsilon)
            out = sinkhorn.Sinkhorn()(linear_problem.LinearProblem(geom))

            f_x = state_f.apply_fn({"params": params_f}, batch["source"])
            g_y = state_g.apply_fn({"params": params_g}, batch["target"])

            t_1 = jnp.mean(optax.l2_loss(f_x, out.f))
            t_2 = jnp.mean(optax.l2_loss(g_y, out.g))

            if include_distribution_match:
                transported_source = batch["source"] - jax.vmap(
                    jax.grad(lambda x: self.state_f.apply_fn({"params": self.state_f.params}, x), argnums=0)
                )(batch["source"])
                transported_target = batch["target"] - jax.vmap(
                    jax.grad(lambda x: self.state_f.apply_fn({"params": self.state_f.params}, x), argnums=0)
                )(batch["target"])
                t_3 = ott.tools.sinkhorn_divergence.sinkhorn_divergence(
                    pointcloud.PointCloud, transported_source, batch["target"]
                ).divergence
                t_4 = ott.tools.sinkhorn_divergence.sinkhorn_divergence(
                    pointcloud.PointCloud, transported_target, batch["source"]
                ).divergence
                return t_1 + t_2 + t_3 + t_4, (t_1, t_2, t_3, t_4)
            return t_1 + t_2, t_1, t_2

        def step_fn(
            state_f: TrainState,
            state_g: TrainState,
            batch: Dict[str, jnp.ndarray],
            epsilon: float,
            include_distribution_match: bool,
        ):
            grad_fn = jax.value_and_grad(loss, argnums=[0, 1], has_aux=True)
            (obj_value, metrics), grads = grad_fn(
                state_f.params, state_g.params, state_f, state_g, batch, epsilon, include_distribution_match
            )
            grads_f, grads_g = grads
            metrics = {
                "obj": obj_value,
                "f": metrics[0],
                "g": metrics[1],
                "div_transported_source": metrics[2] if include_distribution_match else 0.0,
                "div_transported_target": metrics[3] if include_distribution_match else 0,
            }

            return state_f.apply_gradients(grads=grads_f), state_g.apply_gradients(grads=grads_g), metrics

        return step_fn

    def transport(self, x: Optional[jnp.ndarray] = None) -> Any:
        x = x if x is not None else self.x_loader.data
        return x - jax.vmap(jax.grad(lambda x: self.state_f.apply_fn({"params": self.state_f.params}, x), argnums=0))(x)


class NoiseOutsourcingModel:
    def __init__(
        self,
        epsilon: float,
        batch_size_source: int,
        batch_size_target: int,
        iterations: int,
        input_dim: int,
        noise_dim: int = 2,
        inner_iterations: int = 10,
        std: float = 0.1,
        k_noise_per_x: int = 1,
        callback: Optional[Callable[[jnp.ndarray, jnp.ndarray, jnp.ndarray], Any]] = None,
        callback_iters: int = 10,
        cost_fn: Optional[str] = None,
        rng: Optional[jax.random.PRNGKeyArray] = None,
        **kwargs: Any,
    ) -> None:
        self.input_dim = input_dim
        self.epsilon = epsilon
        self.batch_size_source = batch_size_source
        self.batch_size_target = batch_size_target
        self.iterations = iterations
        self.cost_fn = cost_fn
        self.state_t: Optional[TrainState] = None
        self.state_phi: Optional[TrainState] = None
        self.train_step: Any = None
        self.noise_dim = noise_dim
        self.std = std
        self.metrics = {"t_obj": [], "phi_obj": []}
        self.inner_iterations = inner_iterations
        self.rng = jax.random.PRNGKey(0) if rng is None else rng
        self.noise_fn = jax.random.normal
        self.callback = callback
        self.callback_iters = callback_iters
        self.k_noise_per_x = k_noise_per_x

        t = MLP(dim_hidden=[256, 256, 256, 256], is_potential=False, noise_dim=noise_dim, act_fn=nn.relu)

        phi = MLP(dim_hidden=[256, 256, 256, 256], is_potential=True, act_fn=nn.relu)

        opt_t = optax.adamw(learning_rate=1e-3)
        opt_phi = optax.adamw(learning_rate=1e-3)

        self.setup(
            t=t, phi=phi, opt_t=opt_t, opt_phi=opt_phi, input_dim=self.input_dim, noise_dim=self.noise_dim, rng=self.rng
        )

    def setup(self, t, phi, opt_t, opt_phi, input_dim, noise_dim, rng):
        self.state_t = t.create_train_state(rng, opt_t, input_dim + noise_dim)
        self.state_phi = phi.create_train_state(rng, opt_phi, input_dim * 2)
        self.train_step_t = self.get_train_step(to_optimize="t")
        self.train_step_phi = self.get_train_step(to_optimize="phi")

    def __call__(self, x: jnp.ndarray, y: jnp.ndarray) -> Any:
        batch: Dict[str, jnp.ndarray] = {}
        self.x_loader = DataLoader(x, batch_size=self.batch_size_source)
        self.y_loader = DataLoader(y, batch_size=self.batch_size_target)
        for step in tqdm(range(self.iterations)):
            self.rng, rng_x, rng_y, rng_noise, rng_sample = jax.random.split(self.rng, 5)
            batch["source"] = self.x_loader(rng_x)
            batch["target"] = self.y_loader(rng_y)
            batch["latent"] = (
                self.noise_fn(rng_noise, shape=[len(batch["source"]) * self.k_noise_per_x, self.noise_dim]) * self.std
            )
            metrics, t_xz = self.train(batch, self.epsilon, rng_sample)
            for key, value in metrics.items():
                self.metrics[key].append(value)
            if self.callback is not None and step % self.callback_iters == 0:
                self.callback(batch["source"], batch["target"], t_xz)

    def train(self, batch: Dict[str, jnp.ndarray], epsilon: float, key: jax.random.PRNGKeyArray) -> Dict[str, Any]:
        geom = pointcloud.PointCloud(batch["source"], batch["target"], epsilon=epsilon)
        out = sinkhorn.Sinkhorn()(linear_problem.LinearProblem(geom))
        pi_star_inds = jax.random.categorical(key, logits=jnp.log(out.matrix.flatten()))
        inds_source = pi_star_inds // len(batch["target"])
        inds_target = pi_star_inds % len(batch["target"])
        batch["pi_star_samples"] = _concatenate(batch["source"][inds_source], batch["target"][inds_target])

        for _ in range(self.inner_iterations):
            self.state_phi, metrics_phi = self.train_step_phi(self.state_t, self.state_phi, batch, self.k_noise_per_x)

        self.state_t, metrics_t, t_xz = self.train_step_t(self.state_t, self.state_phi, batch, self.k_noise_per_x)
        return dict(metrics_t, **metrics_phi), t_xz

    def get_train_step(self, to_optimize: Literal["t", "phi"]) -> Callable:
        def loss_t(
            params_t: jnp.ndarray,
            params_phi: jnp.ndarray,
            state_t: TrainState,
            state_phi: TrainState,
            batch: Dict[str, jnp.ndarray],
            k_noise_per_x: int,
        ):
            source_repeated = jnp.repeat(batch["source"], k_noise_per_x, axis=0)
            xz = _concatenate(source_repeated, batch["latent"])
            t_xz = state_t.apply_fn({"params": params_t}, xz)
            # print("t loss ", jnp.mean(state_phi.apply_fn({"params": params_phi}, _concatenate(batch["source"], t_xz))))
            #if jnp.isnan(jnp.mean(state_phi.apply_fn({"params": params_phi}, _concatenate(source_repeated, t_xz)))):
            #    print("encountered in loss_t")
            return jnp.mean(state_phi.apply_fn({"params": params_phi}, _concatenate(source_repeated, t_xz))), t_xz

        def loss_phi(
            params_t: jnp.ndarray,
            params_phi: jnp.ndarray,
            state_t: TrainState,
            state_phi: TrainState,
            batch: Dict[str, jnp.ndarray],
            k_noise_per_x: int,
        ):
            source_repeated = jnp.repeat(batch["source"], k_noise_per_x, axis=0)
            xz = _concatenate(source_repeated, batch["latent"])
            t_xz = state_t.apply_fn({"params": params_t}, xz)
            if jnp.sum(jnp.isnan(t_xz)):
                print("t_xz")
            if jnp.sum(jnp.isnan(_concatenate(source_repeated, t_xz))):
                print("in concatenate")
            print(jnp.sum(jnp.isnan(_concatenate(source_repeated, t_xz))))
            phi_pi_hat = state_phi.apply_fn({"params": params_phi}, _concatenate(source_repeated, t_xz))
            phi_pi_star = state_phi.apply_fn({"params": params_phi}, batch["pi_star_samples"])
            if jnp.isnan(jnp.mean(phi_pi_hat)):
                print('encountered in phi pi hat')
                
            if jnp.isnan(jnp.mean(jnp.exp(phi_pi_star))):
                print("encoutererd in exp")
            kl = jnp.mean(phi_pi_hat) - jax.scipy.special.logsumexp(phi_pi_star)
            return -kl

        def step_fn(state_t: TrainState, state_phi: TrainState, batch: Dict[str, jnp.ndarray], k_noise_per_x: int):
            if to_optimize == "t":
                grad_fn = jax.value_and_grad(loss_t, has_aux=True, argnums=0)
                (obj_value, t_xz), grads = grad_fn(
                    state_t.params, state_phi.params, state_t, state_phi, batch, k_noise_per_x
                )
                metrics = {"t_obj": obj_value}
                return state_t.apply_gradients(grads=grads), metrics, t_xz
            elif to_optimize == "phi":
                grad_fn = jax.value_and_grad(loss_phi, has_aux=False, argnums=1)
                obj_value, grads = grad_fn(state_t.params, state_phi.params, state_t, state_phi, batch, k_noise_per_x)
                metrics = {"phi_obj": obj_value}
                return state_phi.apply_gradients(grads=grads), metrics
            return NotImplementedError()

        return step_fn

    def transport(self, x: jnp.ndarray, rng: Optional[jax.random.PRNGKeyArray] = None) -> jnp.ndarray:
        rng = jax.random.PRNGKey(0) if rng is None else rng
        latent = self.noise_fn(rng, shape=[len(x), self.noise_dim]) * self.std
        return self.state_t.apply_fn({"params": self.state_t.params}, _concatenate(x, latent))

    def sample(self, x: jnp.ndarray, k: int = 10, rng: Optional[jax.random.PRNGKeyArray] = None) -> jnp.ndarray:
        rng = jax.random.PRNGKey(0) if rng is None else rng
        latent = self.noise_fn(rng, shape=[k, self.noise_dim]) * self.std
        return self.state_t.apply_fn({"params": self.state_t.params}, _concatenate(jnp.tile(x, (k, 1)), latent))
