import collections
import types
from functools import partial
from typing import Any, Callable, Dict, Literal, Optional, Tuple, Type, Union

import diffrax
import optax
import tensorflow as tf
import tensorflow_datasets as tfds
from flax.training.train_state import TrainState
from tqdm import tqdm

import jax
import jax.numpy as jnp
from ott.geometry import costs, geometry, graph, pointcloud
from ott.problems.linear import linear_problem
from ott.problems.quadratic import quadratic_problem
from ott.solvers import was_solver
from ott.solvers.linear import sinkhorn
from ott.solvers.quadratic import gromov_wasserstein
import flax.linen as nn
from ott.initializers.quadratic.initializers import QuadraticInitializer

Match_fn_T = Callable[
    [jax.Array, jnp.array, jnp.array], Tuple[jnp.array, jnp.array, jnp.array, jnp.array]
]
Match_latent_fn_T = Callable[[jax.Array, jnp.array, jnp.array], Tuple[jnp.array, jnp.array]]


def sample_conditional_indices_from_tmap(
    key: jax.Array,
    tmat: jnp.ndarray,
    k_samples_per_x: Union[int, jnp.ndarray],
    left_marginals: Optional[jnp.ndarray],
    *,
    is_balanced: bool,
) -> Tuple[jnp.array, jnp.array]:
    if not is_balanced:
        key, key2 = jax.random.split(key, 2)
        indices = jax.random.choice(
            key=key2, a=jnp.arange(len(left_marginals)), p=left_marginals, shape=(len(left_marginals),)
        )
    else:
        indices = jnp.arange(tmat.shape[0])
    tmat_adapted = tmat[indices]
    indices_per_row = jax.vmap(
        lambda tmat_adapted: jax.random.choice(
            key=key, a=jnp.arange(tmat.shape[1]), p=tmat_adapted, shape=(k_samples_per_x,)
        ),
        in_axes=0,
        out_axes=0,
    )(tmat_adapted)

    return jnp.repeat(indices, k_samples_per_x), indices_per_row % tmat.shape[1]

def get_quad_initializer(push_forward: jnp.ndarray, target: jnp.ndarray):
    n = len(push_forward)
    pc = pointcloud.PointCloud(push_forward, target)
    geom = geometry.Geometry(pc.cost_matrix, epsilon=1e-1)
    lp = linear_problem.LinearProblem(geom, a=jnp.ones(n) / n, b=jnp.ones(n) / n)
    return lp

class GENOT:
    def __init__(
        self,
        neural_net: Union[Type[nn.Module], Tuple[Type[nn.Module], Type[nn.Module]]],
        input_dim: int,
        output_dim: int,
        iterations: int,
        ot_solver: Type[was_solver.WassersteinSolver],
        optimizer: Optional[Any] = None,
        initializer: Optional[Literal["linearisation_push_forward"]] = None,
        warmup_iterations: int = 0,
        k_noise_per_x: int = 1,
        t_offset: float = 1e-5,
        epsilon: float = 1e-2,
        cost_fn: Union[costs.CostFn, Literal["graph"]] = costs.SqEuclidean(),
        solver_latent_to_data: Optional[Type[was_solver.WassersteinSolver]] = None,
        latent_to_data_epsilon: float = 1e-2,
        latent_to_data_scale_cost: Any = 1.0,
        scale_cost: Any = 1.0,
        graph_kwargs: Dict[str, Any] = types.MappingProxyType({}),
        fused_penalty: float = 0.0,
        split_dim: int = 0,
        mlp_eta: Callable[[jnp.ndarray], float] = None,
        mlp_xi: Callable[[jnp.ndarray], float] = None,
        tau_a: float = 1.0,
        tau_b: float = 1.0,
        callback: Optional[Callable[[jnp.ndarray, jnp.ndarray, jnp.ndarray], Any]] = None,
        callback_kwargs: Dict[str, Any] = {},
        callback_iters: int = 10,
        seed: int = 0,
        **kwargs: Any,
    ) -> None:
        """
        The GENOT training class.

        Parameters
        ----------
        neural_net
            Neural vector field
        input_dim
            Dimension of the source distribution
        output_dim
            Dimension of the target distribution
        iterations
            Number of iterations to train
        ot_solver
            Solver to match samples from the source to the target distribution
        optimizer
            Optimizer for the neural vector field
        initializer
            Initializer for the OT coupling, only valid if `ot_solver` is `GromovWasserstein`.
            This helps to make the solution of the quadratic OT problem more unique.
        warmup_iterations
            Number of times to use the same batch to bias the solution of a quadratic
            OT problem towards a certain solution. Recommended to be set to ~100.
        k_noise_per_x
            Number of samples to draw from the conditional distribution
        t_offset
            Offset for sampling from the time t
        epsilon
            Entropy regularization parameter for the discrete solver
        cost_fn
            Cost function to use for the discrete OT solver
        solver_latent_to_data
            Linear OT solver to match samples from the noise to the conditional distribution
        latent_to_data_epsilon
            Entropy regularization term for `solver_latent_to_data`
        latent_to_data_scale_cost
            How to scale the cost matrix for the `solver_latent_to_data` solver
        scale_cost
            How to scale the cost matrix in each discrete OT problem
        graph_kwargs
            Keyword arguments for the graph cost computation in case `cost="graph"`
        fused_penalty
            Penalisation term for the linear term in a Fused GW setting
        split_dim
            Dimension to split the data into fused term and purely quadratic term in the FGW setting
        mlp_eta
            Neural network to learn the left rescaling function
        mlp_xi
            Neural network to learn the right rescaling function
        tau_a
            Left unbalancedness parameter
        tau_b
            Right unbalancedness parameter
        callback
            Callback function
        callback_kwargs
            Keyword arguments to the callback function
        callback_iters
            Number of iterations after which to evaluate callback function
        seed
            Random seed
        kwargs
            Keyword arguments passed to `setup`, e.g. custom choice of optimizers for learning rescaling functions
        """
        if isinstance(ot_solver, gromov_wasserstein.GromovWasserstein) and epsilon is not None:
            raise ValueError(
                "If `ot_solver` is `GromovWasserstein`, `epsilon` must be `None`. This check is performed "
                "to ensure that in the (fused) Gromov case the `epsilon` parameter is passed via the `ot_solver`."
            )
        
        if not isinstance(ot_solver, gromov_wasserstein.GromovWasserstein) and warmup_iterations != 0:
            raise ValueError(
                "`warmup_iterations` is only available for solving quadratic problems. It is used for initializing "
                "the orientation of Gromov-Wasserstein problems."
            )
        
        if initializer is not None and initializer!="linearisation_push_forward":
            raise ValueError(
                "Only 'linearisation_push_forward' is a valid initializer. Note it should only be used for the quadratic OT problem."
            )

        # setup parameters
        self.rng = jax.random.PRNGKey(seed)
        self.seed = seed
        self.iterations = iterations
        self.metrics = {"loss": [], "loss_eta": [], "loss_xi": []}

        # neural parameters
        self.neural_net = neural_net
        self.state_neural_net: Optional[TrainState] = None
        self.optimizer = optax.adamw(learning_rate=1e-4, weight_decay=1e-10) if optimizer is None else optimizer
        self.use_init =True if initializer is not None else False
        self.warmup_iterations = warmup_iterations
        self.noise_fn = self.noise_fn = jax.tree_util.Partial(
            jax.random.multivariate_normal, mean=jnp.zeros((output_dim,)), cov=jnp.diag(jnp.ones((output_dim,)))
        )
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.k_noise_per_x = k_noise_per_x
        self.t_offset = t_offset

        # OT data-data matching parameters
        self.ot_solver = ot_solver
        self.epsilon = epsilon
        self.cost_fn = cost_fn
        self.scale_cost = scale_cost
        self.graph_kwargs = graph_kwargs  # "k_neighbors", kwargs for graph.Graph.from_graph()
        if fused_penalty != 0 and split_dim == 0:
            raise ValueError("Missing 'split_dim' for FGW.")
        self.fused_penalty = fused_penalty
        self.split_dim = split_dim

        # OT latent-data matching parameters
        self.solver_latent_to_data = solver_latent_to_data
        self.latent_to_data_epsilon = latent_to_data_epsilon
        self.latent_to_data_scale_cost = latent_to_data_scale_cost

        # unbalancedness parameters
        self.mlp_eta = mlp_eta
        self.mlp_xi = mlp_xi
        self.state_eta: Optional[TrainState] = None
        self.state_xi: Optional[TrainState] = None
        self.tau_a: float = tau_a
        self.tau_b: float = tau_b

        # callback parameteres
        self.callback = callback
        self.callback_kwargs = callback_kwargs
        self.callback_iters = callback_iters

        self.setup(**kwargs)

    def setup(self, **kwargs: Any) -> None:
        """
        Set up the model.

        Parameters
        ----------
        kwargs
            Keyword arguments for the setup function
        """
        self.state_neural_net = self.neural_net.create_train_state(self.rng, self.optimizer, self.input_dim)
        self.step_fn = self._get_step_fn(with_init=False)
        self.step_fn_with_init = self._get_step_fn(with_init=True)
        if self.solver_latent_to_data is not None:
            self.match_latent_to_data_fn = self._get_match_latent_fn(
                self.solver_latent_to_data, self.latent_to_data_epsilon, self.latent_to_data_scale_cost
            )
        else:
            self.match_latent_to_data_fn = lambda key, x, y, **_: (x, y)

        if isinstance(self.ot_solver, sinkhorn.Sinkhorn):
            problem_type = "linear"
        else:
            problem_type = "fused" if self.fused_penalty > 0 else "quadratic"

        if self.cost_fn == "graph":
            self.match_fn = self._get_match_fn_graph(
                problem_type=problem_type,
                ot_solver=self.ot_solver,
                epsilon=self.epsilon,
                tau_a=self.tau_a,
                tau_b=self.tau_b,
                fused_penalty=self.fused_penalty,
                split_dim=self.split_dim,
                k_samples_per_x=self.k_noise_per_x,
                scale_cost=self.scale_cost,
                **self.graph_kwargs,
            )
        else:
            if problem_type == "linear":
                self.match_fn = self._get_sinkhorn_match_fn(
                    self.ot_solver,
                    epsilon=self.epsilon,
                    cost_fn=self.cost_fn,
                    tau_a=self.tau_a,
                    tau_b=self.tau_b,
                    scale_cost=self.scale_cost,
                    k_samples_per_x=self.k_noise_per_x,
                )
            else:
                self.match_fn = self._get_gromov_match_fn(
                    self.ot_solver,
                    cost_fn=self.cost_fn,
                    tau_a=self.tau_a,
                    tau_b=self.tau_b,
                    scale_cost=self.scale_cost,
                    split_dim=self.split_dim,
                    fused_penalty=self.fused_penalty,
                    k_samples_per_x=self.k_noise_per_x,
                )

        if self.mlp_eta is not None:
            opt_eta = kwargs["opt_eta"] if "opt_eta" in kwargs else optax.adamw(learning_rate=1e-4, weight_decay=1e-10)
            self.state_eta = self.mlp_eta.create_train_state(self.rng, opt_eta, self.input_dim)
        if self.mlp_xi is not None:
            opt_xi = kwargs["opt_xi"] if "opt_xi" in kwargs else optax.adamw(learning_rate=1e-4, weight_decay=1e-10)
            self.state_xi = self.mlp_xi.create_train_state(self.rng, opt_xi, self.output_dim)

    def __call__(
        self,
        x: Union[jnp.array, collections.abc.Iterable],
        y: Union[jnp.array, collections.abc.Iterable],
        batch_size_source: Optional[int] = None,
        batch_size_target: Optional[int] = None,
    ) -> None:
        """
        Train GENOT.

        Parameters
        ----------
        x
            Source data as an iterator or data array
        y
            Target data as an iterator or data array
        batch_size_source
            Batch size for the source distribution.
        batch_size_target
            Batch size for the target distribution.
        """
        # prepare data
        if not hasattr(x, "shape"):
            x_loader = x
            x_load_fn = lambda x: x
        else:
            assert batch_size_source is not None, "'batch_size_source' must be specified when 'x' is an array."
            x_loader = iter(
                tf.data.Dataset.from_tensor_slices(x)
                .repeat()
                .shuffle(buffer_size=10_000, seed=self.seed)
                .batch(batch_size_source)
            )
            x_load_fn = tfds.as_numpy
        if not hasattr(y, "shape"):
            y_loader = y
            y_load_fn = lambda x: x
        else:
            assert batch_size_target is not None, "'batch_size_target' must be specified when 'y' is an array."
            y_loader = iter(
                tf.data.Dataset.from_tensor_slices(y)
                .repeat()
                .shuffle(buffer_size=10_000, seed=self.seed)
                .batch(batch_size_target)
            )
            y_load_fn = tfds.as_numpy

        batch: Dict[str, jnp.array] = {}
        for step in tqdm(range(self.iterations)):
            if step < self.warmup_iterations:
                source_batch, target_batch = x[:batch_size_source], y[:batch_size_target] #x_load_fn(next(x_loader)), y_load_fn(next(y_loader))
                step_fn = self.step_fn
            else:
                source_batch, target_batch =x_load_fn(next(x_loader)), y_load_fn(next(y_loader))
                step_fn = self.step_fn_with_init if self.use_init else self.step_fn

            self.rng, rng_time, rng_noise, rng_step_fn = jax.random.split(self.rng, 4)
            batch["source"] = source_batch
            batch["target"] = target_batch
            n_samples = len(source_batch) * self.k_noise_per_x
            t = (jax.random.uniform(rng_time, (1,)) + jnp.arange(n_samples) / n_samples) % (1 - self.t_offset)
            batch["time"] = t[:, None]
            batch["noise"] = self.noise_fn(rng_noise, shape=(len(source_batch), self.k_noise_per_x))
            (
                metrics,
                self.state_neural_net,
                self.state_eta,
                self.state_xi,
                eta_predictions,
                xi_predictions,
            ) = step_fn(
                rng_step_fn,
                self.state_neural_net,
                batch,
                self.state_eta,
                self.state_xi,
            )
            for key, value in metrics.items():
                self.metrics[key].append(value)

            if self.callback is not None and step % self.callback_iters == 0:
                self.callback(
                    source=batch["source"],
                    target=batch["target"],
                    eta_predictions=eta_predictions,
                    xi_predictions=xi_predictions,
                    **self.callback_kwargs,
                )

    def _get_sinkhorn_match_fn(
        self,
        ot_solver: Any,
        epsilon: float,
        cost_fn: str,
        tau_a: float,
        tau_b: float,
        scale_cost: Any,
        k_samples_per_x: int,
    ) -> Callable:
        @partial(
            jax.jit,
            static_argnames=["ot_solver", "epsilon", "cost_fn", "scale_cost", "tau_a", "tau_b", "k_samples_per_x"],
        )
        def match_pairs(
            key: jax.Array,
            x: jnp.ndarray,
            y: jnp.ndarray,
            ot_solver: Any,
            tau_a: float,
            tau_b: float,
            epsilon: float,
            cost_fn: str,
            scale_cost: Any,
            k_samples_per_x: int,
            **_: Any,
        ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
            geom = pointcloud.PointCloud(x, y, epsilon=epsilon, scale_cost=scale_cost, cost_fn=cost_fn)
            out = ot_solver(linear_problem.LinearProblem(geom, tau_a=tau_a, tau_b=tau_b))
            a, b = out.matrix.sum(axis=1), out.matrix.sum(axis=0)
            inds_source, inds_target = sample_conditional_indices_from_tmap(
                key=key,
                tmat=out.matrix,
                k_samples_per_x=k_samples_per_x,
                left_marginals=a,
                is_balanced=(tau_a == 1.0) and (tau_b == 1.0),
            )
            return x[inds_source], y[inds_target], a, b

        return jax.tree_util.Partial(
            match_pairs,
            ot_solver=ot_solver,
            epsilon=epsilon,
            cost_fn=cost_fn,
            tau_a=tau_a,
            tau_b=tau_b,
            scale_cost=scale_cost,
            k_samples_per_x=k_samples_per_x,
        )

    def _get_gromov_match_fn(
        self,
        ot_solver: Any,
        cost_fn: str,
        tau_a: float,
        tau_b: float,
        scale_cost: Any,
        split_dim: int,
        fused_penalty: float,
        k_samples_per_x: int,
    ) -> Callable:
        @partial(
            jax.jit,
            static_argnames=[
                "ot_solver",
                "cost_fn",
                "scale_cost",
                "fused_penalty",
                "split_dim",
                "tau_a",
                "tau_b",
                "k_samples_per_x",
            ],
        )
        def match_pairs(
            key: jax.Array,
            x: Tuple[jnp.ndarray, jnp.ndarray],
            y: Tuple[jnp.ndarray, jnp.ndarray],
            initializer: Any,
            ot_solver: Any,
            tau_a: float,
            tau_b: float,
            cost_fn: str,
            scale_cost,
            fused_penalty: float,
            k_samples_per_x: int,
            split_dim: int = 0,
        ) -> Tuple[jnp.array, jnp.array]:
            geom_xx = pointcloud.PointCloud(
                x=x[..., split_dim:], y=x[..., split_dim:], cost_fn=cost_fn, scale_cost=scale_cost
            )
            geom_yy = pointcloud.PointCloud(
                x=y[..., split_dim:], y=y[..., split_dim:], cost_fn=cost_fn, scale_cost=scale_cost
            )
            if split_dim > 0:
                geom_xy = pointcloud.PointCloud(
                    x=x[..., :split_dim], y=y[..., :split_dim], cost_fn=cost_fn, scale_cost=scale_cost
                )
            else:
                geom_xy = None
            prob = quadratic_problem.QuadraticProblem(
                geom_xx, geom_yy, geom_xy, fused_penalty=fused_penalty, tau_a=tau_a, tau_b=tau_b
            )
            out = ot_solver(prob, init=initializer)
            a, b = out.matrix.sum(axis=1), out.matrix.sum(axis=0)
            inds_source, inds_target = sample_conditional_indices_from_tmap(
                key=key,
                tmat=out.matrix,
                k_samples_per_x=k_samples_per_x,
                left_marginals=a,
                is_balanced=(tau_a == 1.0) and (tau_b == 1.0),
            )
            return x[inds_source], y[inds_target], a, b

        return jax.tree_util.Partial(
            match_pairs,
            ot_solver=ot_solver,
            cost_fn=cost_fn,
            tau_a=tau_a,
            tau_b=tau_b,
            scale_cost=scale_cost,
            split_dim=split_dim,
            fused_penalty=fused_penalty,
            k_samples_per_x=k_samples_per_x,
        )

    def _get_match_fn_graph(
        self,
        problem_type: Literal["linear", "quadratic", "fused"],
        ot_solver: Any,
        epsilon: float,
        k_neighbors: int,
        tau_a: float,
        tau_b: float,
        scale_cost: Any,
        fused_penalty: float,
        split_dim: int,
        k_samples_per_x: int,
        **kwargs,
    ) -> Callable:
        def get_nearest_neighbors(
            X: jnp.ndarray, Y: jnp.ndarray, k: int = 30  # type: ignore[name-defined]
        ) -> Tuple[jnp.ndarray, jnp.ndarray]:  # type: ignore[name-defined]
            concat = jnp.concatenate((X, Y), axis=0)
            pairwise_euclidean_distances = pointcloud.PointCloud(concat, concat).cost_matrix
            distances, indices = jax.lax.approx_min_k(
                pairwise_euclidean_distances, k=k, recall_target=0.95, aggregate_to_topk=True
            )
            return distances, indices

        def create_cost_matrix(X: jnp.array, Y: jnp.array, k_neighbors: int, **kwargs: Any) -> jnp.array:
            distances, indices = get_nearest_neighbors(X, Y, k_neighbors)
            a = jnp.zeros((len(X) + len(Y), len(X) + len(Y)))
            adj_matrix = a.at[
                jnp.repeat(jnp.arange(len(X) + len(Y)), repeats=k_neighbors).flatten(), indices.flatten()
            ].set(distances.flatten())
            return graph.Graph.from_graph(adj_matrix, normalize=kwargs.pop("normalize", True), **kwargs).cost_matrix[
                : len(X), len(X) :
            ]

        @partial(
            jax.jit,
            static_argnames=[
                "ot_solver",
                "problem_type",
                "epsilon",
                "k_neighbors",
                "tau_a",
                "tau_b",
                "k_samples_per_x",
                "fused_penalty",
                "split_dim",
            ],
        )
        def match_pairs(
            key: jax.Array,
            x: jnp.ndarray,
            y: jnp.ndarray,
            ot_solver: Any,
            problem_type: Literal["linear", "quadratic", "fused"],
            epsilon: float,
            tau_a: float,
            tau_b: float,
            fused_penalty: float,
            split_dim: int,
            k_neighbors: int,
            k_samples_per_x: int,
            initializer: Any,
            **kwargs,
        ) -> Tuple[jnp.array, jnp.array, jnp.ndarray, jnp.ndarray]:
            if problem_type == "linear":
                cm = create_cost_matrix(x, y, k_neighbors, **kwargs)
                geom = geometry.Geometry(cost_matrix=cm, epsilon=epsilon, scale_cost=scale_cost)
                out = ot_solver(linear_problem.LinearProblem(geom, tau_a=tau_a, tau_b=tau_b))
            else:
                cm_xx = create_cost_matrix(x[..., split_dim:], x[..., split_dim:], k_neighbors, **kwargs)
                cm_yy = create_cost_matrix(y[..., split_dim:], y[..., split_dim:], k_neighbors, **kwargs)
                geom_xx = geometry.Geometry(cost_matrix=cm_xx, epsilon=epsilon, scale_cost=scale_cost)
                geom_yy = geometry.Geometry(cost_matrix=cm_yy, epsilon=epsilon, scale_cost=scale_cost)
                if problem_type == "quadratic":
                    geom_xy = None
                else:
                    cm_xy = create_cost_matrix(x[..., :split_dim], y[..., :split_dim], k_neighbors, **kwargs)
                    geom_xy = geometry.Geometry(cost_matrix=cm_xy, epsilon=epsilon, scale_cost=scale_cost)
                prob = quadratic_problem.QuadraticProblem(
                    geom_xx, geom_yy, geom_xy, fused_penalty=fused_penalty, tau_a=tau_a, tau_b=tau_b
                )
                out = ot_solver(prob)
            a, b = out.matrix.sum(axis=0), out.matrix.sum(axis=1)
            inds_source, inds_target = sample_conditional_indices_from_tmap(
                key=key,
                tmat=out.matrix,
                k_samples_per_x=k_samples_per_x,
                left_marginals=a,
                is_balanced=(tau_a == 1.0) and (tau_b == 1.0),
            )
            return x[inds_source], y[inds_target], a, b

        return jax.tree_util.Partial(
            match_pairs,
            problem_type=problem_type,
            ot_solver=ot_solver,
            epsilon=epsilon,
            k_neighbors=k_neighbors,
            tau_a=tau_a,
            tau_b=tau_b,
            k_samples_per_x=k_samples_per_x,
            fused_penalty=fused_penalty,
            split_dim=split_dim,
            **kwargs,
        )

    def _get_match_latent_fn(self, ot_solver: Type[sinkhorn.Sinkhorn], epsilon: float, scale_cost: Any) -> Callable:
        @partial(jax.jit, static_argnames=["ot_solver", "epsilon", "scale_cost"])
        def match_latent_to_data(
            key: jax.Array,
            ot_solver: Type[was_solver.WassersteinSolver],
            x: jnp.ndarray,
            y: jnp.ndarray,
            epsilon: float,
            scale_cost: Any,
        ) -> Tuple[jnp.ndarray, jnp.ndarray]:
            geom = pointcloud.PointCloud(x, y, epsilon=epsilon, scale_cost=scale_cost)
            out = ot_solver(linear_problem.LinearProblem(geom))
            inds_source, inds_target = sample_conditional_indices_from_tmap(key, out.matrix, 1, None, is_balanced=True)
            return x[inds_source], y[inds_target]

        return jax.tree_util.Partial(match_latent_to_data, ot_solver=ot_solver, epsilon=epsilon, scale_cost=scale_cost)

    def _get_step_fn(self, with_init: bool = False) -> Callable:
        def loss_fn(
            params_mlp: jnp.array,
            apply_fn_mlp: Callable,
            batch: Dict[str, jnp.array],
        ):
            def phi_t(x_0: jnp.ndarray, x_1: jnp.ndarray, t: jnp.ndarray) -> jnp.ndarray:
                return (1 - t) * x_0 + t * x_1

            def u_t(x_0: jnp.ndarray, x_1: jnp.ndarray) -> jnp.ndarray:
                return x_1 - x_0

            phi_t_eval = phi_t(batch["noise"], batch["target"], batch["time"])
            mlp_pred = apply_fn_mlp(
                {"params": params_mlp}, t=batch["time"], latent=phi_t_eval, condition=batch["source"]
            )
            d_psi = u_t(batch["noise"], batch["target"])

            return jnp.mean(optax.l2_loss(mlp_pred, d_psi))

        def loss_a_fn(
            params_eta: Optional[jnp.ndarray],
            apply_fn_eta: Optional[Callable],
            x: jnp.ndarray,
            a: jnp.ndarray,
            expectation_reweighting: float,
        ) -> float:
            eta_predictions = apply_fn_eta({"params": params_eta}, x)
            return (
                optax.l2_loss(eta_predictions[:, 0], a).mean()
                + optax.l2_loss(jnp.mean(eta_predictions) - expectation_reweighting),
                eta_predictions,
            )

        def loss_b_fn(
            params_xi: Optional[jnp.ndarray],
            apply_fn_xi: Optional[Callable],
            x: jnp.ndarray,
            b: jnp.ndarray,
            expectation_reweighting: float,
        ) -> float:
            xi_predictions = apply_fn_xi({"params": params_xi}, x)
            return (
                optax.l2_loss(xi_predictions, b).mean()
                + optax.l2_loss(jnp.mean(xi_predictions) - expectation_reweighting),
                xi_predictions,
            )

        @jax.jit
        def step_fn(
            key: jax.Array,
            state_neural_net: TrainState,
            batch: Dict[str, jnp.array],
            state_eta: Optional[TrainState] = None,
            state_xi: Optional[TrainState] = None,
        ):
            rng_match, rng_noise = jax.random.split(key, 2)
            original_source_batch = batch["source"]
            original_target_batch = batch["target"]
            source_batch, target_batch, a, b = self.match_fn(rng_match, batch["source"], batch["target"], initializer=None)
            rng_noise = jax.random.split(rng_noise, (len(target_batch)))

            noise_matched, conditional_target = jax.vmap(self.match_latent_to_data_fn, 0, 0)(
                key=rng_noise, x=batch["noise"], y=target_batch
            )

            batch["source"] = jnp.reshape(source_batch, (len(source_batch), -1))
            batch["target"] = jnp.reshape(conditional_target, (len(source_batch), -1))
            batch["noise"] = jnp.reshape(noise_matched, (len(source_batch), -1))

            grad_fn = jax.value_and_grad(loss_fn, argnums=[0], has_aux=False)
            loss, grads_mlp = grad_fn(
                state_neural_net.params,
                state_neural_net.apply_fn,
                batch,
            )
            
            metrics = {}
            metrics["loss"] = loss

            integration_eta = jnp.sum(a)
            integration_xi = jnp.sum(b)

            if state_eta is not None:
                grad_a_fn = jax.value_and_grad(loss_a_fn, argnums=0, has_aux=True)
                (loss_a, eta_predictions), grads_eta = grad_a_fn(
                    state_eta.params,
                    state_eta.apply_fn,
                    original_source_batch[:,],
                    a * len(original_source_batch),
                    integration_xi,
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
                    original_target_batch[:,],
                    (b * len(original_target_batch))[:, None],
                    integration_eta,
                )
                new_state_xi = state_xi.apply_gradients(grads=grads_xi)
                metrics["loss_xi"] = loss_b
            else:
                new_state_xi = xi_predictions = None

            return (
                metrics,
                state_neural_net.apply_gradients(grads=grads_mlp[0]),
                new_state_eta,
                new_state_xi,
                eta_predictions,
                xi_predictions,
            )
        
        @jax.jit
        def step_fn_with_init(
            key: jax.Array,
            state_neural_net: TrainState,
            batch: Dict[str, jnp.array],
            state_eta: Optional[TrainState] = None,
            state_xi: Optional[TrainState] = None,
        ):
            rng_match, rng_noise = jax.random.split(key, 2)
            original_source_batch = batch["source"]
            original_target_batch = batch["target"]
            push_forward = self.transport(original_source_batch)[0][0,...]
            initializer = get_quad_initializer(push_forward, original_target_batch)

            source_batch, target_batch, a, b = self.match_fn(rng_match, batch["source"], batch["target"], initializer=initializer)
            rng_noise = jax.random.split(rng_noise, (len(target_batch)))

            noise_matched, conditional_target = jax.vmap(self.match_latent_to_data_fn, 0, 0)(
                key=rng_noise, x=batch["noise"], y=target_batch
            )

            batch["source"] = jnp.reshape(source_batch, (len(source_batch), -1))
            batch["target"] = jnp.reshape(conditional_target, (len(source_batch), -1))
            batch["noise"] = jnp.reshape(noise_matched, (len(source_batch), -1))

            grad_fn = jax.value_and_grad(loss_fn, argnums=[0], has_aux=False)
            loss, grads_mlp = grad_fn(
                state_neural_net.params,
                state_neural_net.apply_fn,
                batch,
            )
            
            metrics = {}
            metrics["loss"] = loss

            integration_eta = jnp.sum(a)
            integration_xi = jnp.sum(b)

            if state_eta is not None:
                grad_a_fn = jax.value_and_grad(loss_a_fn, argnums=0, has_aux=True)
                (loss_a, eta_predictions), grads_eta = grad_a_fn(
                    state_eta.params,
                    state_eta.apply_fn,
                    original_source_batch[:,],
                    a * len(original_source_batch),
                    integration_xi,
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
                    original_target_batch[:,],
                    (b * len(original_target_batch))[:, None],
                    integration_eta,
                )
                new_state_xi = state_xi.apply_gradients(grads=grads_xi)
                metrics["loss_xi"] = loss_b
            else:
                new_state_xi = xi_predictions = None

            return (
                metrics,
                state_neural_net.apply_gradients(grads=grads_mlp[0]),
                new_state_eta,
                new_state_xi,
                eta_predictions,
                xi_predictions,
            )

        return step_fn_with_init if with_init else step_fn

    def transport(
        self, source: jnp.array, seed: int = 0, diffeqsolve_kwargs: Dict[str, Any] = types.MappingProxyType({})
    ) -> Union[jnp.array, diffrax.Solution, Optional[jnp.ndarray]]:
        """
        Transport the distribution.

        Parameters
        ----------
        source
            Source distribution to transport
        seed
            Random seed for sampling from the latent distribution
        diffeqsolve_kwargs
            Keyword arguments for the ODE solver.

        Returns
        -------
            The transported samples, the solution of the neural ODE, and the rescaling factor.
        """
        diffeqsolve_kwargs = dict(diffeqsolve_kwargs)
        rng = jax.random.PRNGKey(seed)
        latent_shape = (len(source),)
        latent_batch = self.noise_fn(rng, shape=latent_shape)
        apply_fn_partial = partial(self.state_neural_net.apply_fn, condition=source)
        solution = diffrax.diffeqsolve(
            diffrax.ODETerm(
                lambda t, y, *args: apply_fn_partial({"params": self.state_neural_net.params}, t=t, latent=y)
            ),
            diffeqsolve_kwargs.pop("solver", diffrax.Tsit5()),
            t0=0,
            t1=1,
            dt0=diffeqsolve_kwargs.pop("dt0", None),
            y0=latent_batch,
            stepsize_controller=diffeqsolve_kwargs.pop(
                "stepsize_controller", diffrax.PIDController(rtol=1e-3, atol=1e-6)
            ),
            **diffeqsolve_kwargs,
        )
        if self.state_eta is not None:
            weight_factors = self.state_eta.apply_fn({"params": self.state_eta.params}, x=source)
        else:
            weight_factors = jnp.ones(source.shape)
        return solution.ys, solution, weight_factors
