# jax
from abc import ABC, abstractmethod
from types import MappingProxyType
from typing import Any, Callable, Dict, List, Literal, Mapping, Optional, Sequence, Tuple, Union

import flax.linen as nn
import jax
import jax.numpy as jnp
import optax

# ott
import ott
import ott.geometry.costs as costs
import tensorflow as tf
import tensorflow_datasets as tfds
from flax.training.train_state import TrainState
from ott.geometry import pointcloud
from ott.problems.linear import linear_problem
from ott.solvers.nn.models import ModelBase
from tqdm import tqdm

from entot.models.utils import MLP, BaseSampler, DataLoader, KantorovichGap, MixtureNormalSampler, _concatenate


class KantorovichGapModel:
    def __init__(
        self,
        epsilon_kant_gap: float,
        input_dim: Sequence[int],
        neural_net: Optional[ModelBase] = None,
        k_noise_per_x: int = 1,
        epsilon_fitting_loss: float = 1e-2,
        std: float = 0.1,
        noise_dim: int = 4,
        opt: Optional[optax.OptState] = None,
        callback: Optional[Callable[[jnp.ndarray, jnp.ndarray, jnp.ndarray], Any]] = None,
        callback_kwargs: Dict[str, Any] = {},
        callback_iters: int = 10,
        fitting_loss: Optional[Callable[[jnp.ndarray, jnp.ndarray], float]] = None,
        lambda_kant_gap: Union[float, List[float]] = 1.0,
        iterations: int = 10_000,
        mlp_eta: Callable[[jnp.ndarray], float] = None,
        mlp_xi: Callable[[jnp.ndarray], float] = None,
        unbalanced_kwargs: Dict[str, Any] = {},
        scale_cost_kant_gap: Any = 1.0,
        scale_cost_fitting_term: Any = 1.0,
        rng: Optional[jax.random.PRNGKeyArray] = None,
    ) -> None:
        self.input_dim = input_dim
        self.neural_net = neural_net
        self.noise_dim = noise_dim
        self.rng = jax.random.PRNGKey(0) if rng is None else rng
        self.fitting_loss = (
            lambda a, t_x, b, y: ott.tools.sinkhorn_divergence.sinkhorn_divergence(
                pointcloud.PointCloud,
                jnp.reshape(t_x, (t_x.shape[0], -1)),
                jnp.reshape(y, (y.shape[0], -1)),
                epsilon=epsilon_fitting_loss,
                scale_cost=scale_cost_fitting_term,
                a=a,
                b=b,
            ).divergence
            if fitting_loss is None
            else fitting_loss
        )
        self.lambda_kant_gap = lambda_kant_gap if isinstance(lambda_kant_gap, list) else [lambda_kant_gap] * iterations
        self.iterations = iterations
        self.std = std
        self.callback = callback
        self.callback_kwargs = callback_kwargs
        self.callback_iters = callback_iters
        self.noise_fn = jax.random.normal
        self.k_noise_per_x = k_noise_per_x
        self.kant_gap = KantorovichGap(
            noise_dim=noise_dim,
            geometry_kwargs={
                "epsilon": epsilon_kant_gap,
                "scale_cost": scale_cost_kant_gap,
                **{"cost_fn": costs.SqEuclidean()},
            },
        )
        self.mlp_eta = mlp_eta
        self.mlp_xi = mlp_xi
        self.state_eta: Optional[TrainState] = None
        self.state_xi: Optional[TrainState] = None
        self.unbalanced_kwargs = unbalanced_kwargs

        self.metrics = {
            "total_loss": [],
            "fitting_loss": [],
            "scaled_kant_gap": [],
            "kant_gap": [],
            "loss_eta": [],
            "loss_xi": [],
        }

        if neural_net is None:
            self.neural_net = MLP(
                dim_hidden=[128, 128, 128, 128], is_potential=False, noise_dim=noise_dim, act_fn=nn.relu
            )
        else:
            self.neural_net = neural_net

        self.optimizer = optax.adamw(learning_rate=1e-4, weight_decay=1e-10) if opt is None else opt

        self.setup(self.input_dim, self.noise_dim, self.neural_net, self.optimizer)

    def setup(
        self,
        input_dim: int,
        noise_dim: int,
        neural_net: ModelBase,
        optimizer: optax.OptState,
    ) -> None:
        self.state_neural_net = neural_net.create_train_state(
            self.rng, optimizer, (*input_dim[:-1], input_dim[-1] + noise_dim)
        )

        if self.mlp_eta is not None:
            self.state_eta = self.mlp_eta.create_train_state(self.rng, optimizer, input_dim)
        if self.mlp_xi is not None:
            self.state_xi = self.mlp_xi.create_train_state(self.rng, optimizer, input_dim)
        self.step_fn = self._get_step_fn()
        self.pretrain_step_fn = self._get_pretrain_step_fn()

    @property
    def neural_kant_gap(self) -> Callable:
        return lambda params, apply_fn, batch: (
            self.kant_gap(
                a=batch.get("a", None),
                source=batch["source"],
                b=batch.get("a", None),
                T=lambda x: apply_fn({"params": params}, x),
            )
        )

    @property
    def neural_fitting_loss(self) -> Callable:
        return lambda params, apply_fn, batch: (
            self.fitting_loss(
                a=batch.get("a", None),
                t_x=apply_fn({"params": params}, batch["source"]),
                b=batch.get("b", None),
                y=batch["target"],
            )
        )

    def unbalanced_step_1(
        self, batch: Dict[str, jnp.ndarray], t_xz: jnp.ndarray, tau_a: float, tau_b: float, epsilon: float
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        geom = ott.geometry.pointcloud.PointCloud(t_xz, batch["target"], epsilon=epsilon)

        out = ott.solvers.linear.sinkhorn.Sinkhorn()(linear_problem.LinearProblem(geom, tau_a=tau_a, tau_b=tau_b))
        # TODO: adapt for multiple noise per x

        return out

    def unbalanced_step_0(
        self, batch: Dict[str, jnp.ndarray], t_xz: jnp.ndarray, tau_a: float, tau_b: float, epsilon: float
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        geom = ott.geometry.pointcloud.PointCloud(
            batch["source"][..., : -self.noise_dim], batch["target"], epsilon=epsilon
        )

        out = ott.solvers.linear.sinkhorn.Sinkhorn()(linear_problem.LinearProblem(geom, tau_a=tau_a, tau_b=tau_b))
        # TODO: adapt for multiple noise per x

        return out

    def pretrain(
        self,
        x: Union[jnp.ndarray, MixtureNormalSampler],
        y: Union[jnp.ndarray, MixtureNormalSampler],
        batch_size_source: int,
        batch_size_target: int,
        pretrain_iters: int = 0,
    ) -> Any:
        x_loader_sorted = iter(tf.data.Dataset.from_tensor_slices(x).repeat().batch(batch_size_source))
        y_loader_sorted = iter(tf.data.Dataset.from_tensor_slices(y).repeat().batch(batch_size_target))
        batch: Dict[str, jnp.ndarray] = {}
        for i in range(pretrain_iters):
            x_batch, y_batch = tfds.as_numpy(next(x_loader_sorted)), tfds.as_numpy(next(y_loader_sorted))
            self.rng, rng_noise = jax.random.split(self.rng, 2)
            source_batch = jnp.tile(x_batch, (self.k_noise_per_x, *((1,) * (x_batch.ndim - 1))))
            noise_shape = (
                (*source_batch.shape[:-1], self.noise_dim)
                if len(source_batch.shape[:-1]) > 1
                else (source_batch.shape[:-1][0], self.noise_dim)
            )
            latent_batch = self.noise_fn(rng_noise, shape=noise_shape) * self.std
            x_with_noise = jnp.concatenate((source_batch, latent_batch), axis=-1)
            batch["source"] = x_with_noise
            batch["target"] = y_batch
            self.state_neural_net = self.pretrain_step_fn(self.state_neural_net, batch)

    def __call__(
        self,
        x: Union[jnp.ndarray, MixtureNormalSampler],
        y: Union[jnp.ndarray, MixtureNormalSampler],
        batch_size_source: int,
        batch_size_target: int,
    ) -> Any:
        x_loader = iter(
            tf.data.Dataset.from_tensor_slices(x).repeat().shuffle(buffer_size=10_000).batch(batch_size_source)
        )
        y_loader = iter(
            tf.data.Dataset.from_tensor_slices(y).repeat().shuffle(buffer_size=10_000).batch(batch_size_target)
        )

        batch: Dict[str, jnp.ndarray] = {}
        for step in tqdm(range(self.iterations)):
            x_batch, y_batch = tfds.as_numpy(next(x_loader)), tfds.as_numpy(next(y_loader))
            self.rng, rng_noise = jax.random.split(self.rng, 2)
            source_batch = jnp.tile(x_batch, (self.k_noise_per_x, *((1,) * (x_batch.ndim - 1))))
            noise_shape = (
                (*source_batch.shape[:-1], self.noise_dim)
                if len(source_batch.shape[:-1]) > 1
                else (source_batch.shape[:-1][0], self.noise_dim)
            )
            latent_batch = self.noise_fn(rng_noise, shape=noise_shape) * self.std
            x_with_noise = jnp.concatenate((source_batch, latent_batch), axis=-1)
            batch["source"] = x_with_noise
            batch["target"] = y_batch
            (
                self.state_neural_net,
                self.state_eta,
                self.state_xi,
                metrics,
                t_xz,
                sinkhorn_output,
                sinkhorn_output_unbalanced,
                eta_predictions,
                xi_predictions,
            ) = self.step_fn(
                self.state_neural_net,
                self.state_eta,
                self.state_xi,
                batch,
                self.lambda_kant_gap[step],
                self.unbalanced_kwargs,
            )
            for key, value in metrics.items():
                self.metrics[key].append(value)
            if self.callback is not None and step % self.callback_iters == 0:
                self.callback(
                    source=batch["source"][..., : -self.noise_dim],
                    target=batch["target"],
                    t_xz=t_xz,
                    sinkhorn_output=sinkhorn_output,
                    sinkhorn_output_unbalanced=sinkhorn_output_unbalanced,
                    eta_predictions=eta_predictions,
                    xi_predictions=xi_predictions,
                    **self.callback_kwargs,
                    **self.unbalanced_kwargs,
                )

    def _get_step_fn(self) -> Callable:
        def loss_fn(
            params_mlp: jnp.ndarray,
            apply_fn_mlp: Callable,
            batch: Dict[str, jnp.ndarray],
            lambda_kant_gap: float,
            unbalanced_kwargs: Dict[str, Any],
        ) -> Tuple[float, Dict[str, float]]:
            t_xz = apply_fn_mlp({"params": params_mlp}, batch["source"])

            if unbalanced_kwargs:
                if unbalanced_kwargs["model"] == 0:
                    sinkhorn_output_unbalanced = self.unbalanced_step_0(
                        batch,
                        t_xz,
                        unbalanced_kwargs["tau_a"],
                        unbalanced_kwargs["tau_b"],
                        unbalanced_kwargs["epsilon"],
                    )
                elif unbalanced_kwargs["model"] == 1:
                    sinkhorn_output_unbalanced = self.unbalanced_step_1(
                        batch,
                        t_xz,
                        unbalanced_kwargs["tau_a"],
                        unbalanced_kwargs["tau_b"],
                        unbalanced_kwargs["epsilon"],
                    )
                else:
                    raise ValueError("Need unbalanced_kwargs 'model' key")
                a = sinkhorn_output_unbalanced.matrix.sum(axis=1) * len(batch["source"])
                b = sinkhorn_output_unbalanced.matrix.sum(axis=0) * len(batch["target"])
            else:
                sinkhorn_output_unbalanced = a = b = None

            batch["a"] = a
            batch["b"] = b

            val_fitting_loss = self.neural_fitting_loss(
                params_mlp,
                apply_fn_mlp,
                batch,
            )
            val_kant_gap, sinkhorn_output = self.neural_kant_gap(params_mlp, apply_fn_mlp, batch)
            val_tot_loss = val_fitting_loss + lambda_kant_gap * val_kant_gap
            # jax.debug.print("gap loss {x}", x=lambda_kant_gap * val_kant_gap)
            # store training logs
            metrics = {
                "total_loss": val_tot_loss,
                "fitting_loss": val_fitting_loss,
                "scaled_kant_gap": lambda_kant_gap * val_kant_gap,
                "kant_gap": val_kant_gap,
            }

            return val_tot_loss, (metrics, t_xz, sinkhorn_output, sinkhorn_output_unbalanced)

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
        def step_fn(
            state_neural_net: TrainState,
            state_eta: Optional[TrainState],
            state_xi: Optional[TrainState],
            train_batch: Dict[str, jnp.ndarray],
            lambda_kant_gap: float,
            unbalanced_kwargs: Dict[str, Any],
        ) -> Tuple[TrainState, Dict[str, float], jnp.ndarray]:
            grad_fn = jax.value_and_grad(loss_fn, argnums=0, has_aux=True)
            (_, (metrics, t_xz, sinkhorn_output, sinkhorn_output_unbalanced)), grads_mlp = grad_fn(
                state_neural_net.params, state_neural_net.apply_fn, train_batch, lambda_kant_gap, unbalanced_kwargs
            )

            if state_eta is not None:
                grad_a_fn = jax.value_and_grad(loss_a_fn, argnums=0, has_aux=True)
                (loss_a, eta_predictions), grads_eta = grad_a_fn(
                    state_eta.params,
                    state_eta.apply_fn,
                    train_batch["source"][..., : -self.noise_dim],
                    sinkhorn_output_unbalanced.matrix.sum(axis=1) * len(train_batch["source"]),
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
                    train_batch["target"],
                    sinkhorn_output_unbalanced.matrix.sum(axis=0) * len(train_batch["target"]),
                )
                new_state_xi = state_xi.apply_gradients(grads=grads_xi)
                metrics["loss_xi"] = loss_b
            else:
                new_state_xi = xi_predictions = None

            return (
                state_neural_net.apply_gradients(grads=grads_mlp),
                new_state_eta,
                new_state_xi,
                metrics,
                t_xz,
                sinkhorn_output,
                sinkhorn_output_unbalanced,
                eta_predictions,
                xi_predictions,
            )

        return step_fn

    def _get_pretrain_step_fn(self) -> Callable:
        def pretrain_loss_fn(
            params_mlp: jnp.ndarray,
            apply_fn_mlp: Callable,
            batch: Dict[str, jnp.ndarray],
        ) -> Tuple[float, Dict[str, float]]:
            t_xz = apply_fn_mlp({"params": params_mlp}, batch["source"])
            return optax.l2_loss(batch["target"], t_xz).sum()

        @jax.jit
        def pretrain_step_fn(
            state_neural_net: TrainState,
            train_batch: Dict[str, jnp.ndarray],
        ) -> Tuple[TrainState, Dict[str, float], jnp.ndarray]:
            grad_fn = jax.value_and_grad(pretrain_loss_fn, argnums=0, has_aux=False)
            _, grads_mlp = grad_fn(
                state_neural_net.params,
                state_neural_net.apply_fn,
                train_batch,
            )
            return state_neural_net.apply_gradients(grads=grads_mlp)

        return pretrain_step_fn

    def transport(self, x: jnp.ndarray, samples_per_x: int = 1, seed: int = None) -> jnp.ndarray:
        assert x.ndim == len(self.input_dim) + 1
        rng = jax.random.PRNGKey(seed)
        source_batch = jnp.tile(x, (samples_per_x, *((1,) * (x.ndim - 1))))
        noise_shape = (
            (*source_batch.shape[:-1], self.noise_dim)
            if len(source_batch.shape[:-1]) > 1
            else (source_batch.shape[:-1][0], self.noise_dim)
        )
        latent_batch = self.noise_fn(rng, shape=noise_shape) * self.std
        x_with_noise = jnp.concatenate((source_batch, latent_batch), axis=-1)
        t_xz = self.state_neural_net.apply_fn({"params": self.state_neural_net.params}, x_with_noise)
        return jnp.reshape(t_xz, (samples_per_x, *(x.shape)))
