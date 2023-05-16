# jax
import jax
import jax.numpy as jnp
import jax.random as random

# ott
from ott.geometry import pointcloud
from ott.geometry.geometry import Geometry
from ott.solvers.linear import sinkhorn
from ott.problems.linear import linear_problem
import ott.geometry.costs as costs
from ott.solvers.linear import acceleration
from ott.solvers.nn.models import MLP
from typing import Any, Mapping, Callable, Optional
from types import MappingProxyType
import flax.linen as nn
import jax.numpy as jnp
import optax
from tqdm import tqdm
import matplotlib.pyplot as plt
import jax_dataloader as jdl
from flax.training.train_state import TrainState
from typing import Literal, Dict
from ott.math.utils import logsumexp as lse
import jax.scipy as jsp
from ott.solvers.linear.sinkhorn import SinkhornOutput
from abc import abstractmethod, ABC
from ott.problems.linear.potentials import EntropicPotentials


from typing import Any

from entot.models.utils import DataLoader


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
        self.source : Optional[jnp.ndarray] = None
        self.target : Optional[jnp.ndarray] = None

    def __call__(self, x: jnp.ndarray, y:jnp.ndarray) -> Any:
        self.source=x
        self.target=y
        self.output = self.solver(linear_problem.LinearProblem(pointcloud.PointCloud(x, y, epsilon=self.epsilon)))
        self.dual_potentials = self.output.to_dual_potentials()

    def transport(self, x: Optional[jnp.ndarray]=None) -> jnp.ndarray:
        x=x if x is not None else self.source
        return self.dual_potentials.transport(x)
    
    def f_hessian(self, x: jnp.ndarray, statistic: Optional[Literal["total_variance", "largest_eigenvalue"]]=None):
        if statistic is None:
            jax.hessian(self.output.f)
    
    def get_hessians_f(self, x: jnp.ndarray) -> jnp.ndarray:
        x = jnp.atleast_2d(x)
        hess_eval = jax.hessian(lambda source, target, sinkhorn_output, epsilon: 0.5 * jnp.inner(source, source) - self.evaluate_f(source, target, sinkhorn_output, epsilon), argnums=0)
        vmap_hess_eval = jax.vmap(lambda x: hess_eval(x, self.target, self.output, self.epsilon))
        return vmap_hess_eval(x)
    
    def get_hessians_g(self, y: jnp.ndarray) -> jnp.ndarray:
        y = jnp.atleast_2d(y)
        hess_eval = jax.hessian(lambda source, target, sinkhorn_output, epsilon: 0.5 * jnp.inner(target, target) - self.evaluate_g(source, target, sinkhorn_output, epsilon), argnums=1)
        vmap_hess_eval = jax.vmap(lambda y: hess_eval(self.source, y, self.output, self.epsilon))
        return vmap_hess_eval(y)
    
    #@staticmethod
    def evaluate_f(self, x: jnp.ndarray, y: jnp.ndarray, out:SinkhornOutput, epsilon:float):
        g = out.g - jnp.sum(out.g) + jnp.sum(out.f)
        geom = pointcloud.PointCloud(jnp.atleast_2d(x), y)
        cost = geom.cost_matrix
        z = (g - cost) / epsilon
        lse = -epsilon * jsp.special.logsumexp(z, axis=-1)
        return jnp.squeeze(lse)
    
    @staticmethod
    def evaluate_g(x: jnp.ndarray, y: jnp.ndarray, out:SinkhornOutput, epsilon:float):
        f = out.f - jnp.sum(out.f) + jnp.sum(out.g)
        geom = pointcloud.PointCloud(x,y)
        cost = geom.cost_matrix
        z = (f - cost) / epsilon
        lse = -epsilon * jsp.special.logsumexp(z, axis=-1)
        return jnp.squeeze(lse)
            
    

class NeuralDualPotentialModel(DualPotentialModel, ABC):

    def __init__(self, epsilon: float, batch_size_source: Optional[int]=None, batch_size_target: Optional[int]=None, iterations: Optional[int]=None, input_dim: Optional[int]=None, rng: Optional[jax.random.PRNGKeyArray] = None) -> None:
        
        super().__init__(epsilon=epsilon, rng=rng)
        self.batch_size_source = batch_size_source
        self.batch_size_target = batch_size_target
        self.iterations = iterations
        self.input_dim = input_dim
        self.x_loader : Optional[DataLoader] = None
        self.y_loader : Optional[DataLoader] = None
        

    def __call__(self, x: jnp.ndarray, y:jnp.ndarray) -> Any:
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

    def __init__(self, epsilon: float, batch_size_source: int, batch_size_target: int, iterations: int, input_dim: int, cost_fn: Optional[str]=None, **kwargs) -> None:
        super().__init__(epsilon=epsilon, batch_size_source=batch_size_source, batch_size_target=batch_size_target, iterations=iterations, input_dim=input_dim)
        self.cost_fn = cost_fn
        self.state_f: Optional[TrainState] = None
        self.state_g: Optional[TrainState] = None
        self.state_mlp: Optional[TrainState] = None
        self.train_step: Any = None
        self.train_step_mlp: Any = None
        self.metrics = {"obj": [], "term": [], "f_mean": [], "g_mean": []}
        self.mlp_metrics = {"loss": [], "d_xy": [], "H": []}

        g = MLP(
          dim_hidden=[256, 256],
          is_potential=True,
          act_fn=nn.gelu)

        f = MLP(
          dim_hidden=[256, 256],
          is_potential=True,
          act_fn=nn.gelu)
        
        mlp = MLP(
            dim_hidden=[128, 256, 128, 128],
            is_potential=False,
            act_fn = nn.relu,
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
        
        self.state_f, self.state_g, metrics = self.train_step(self.state_f, self.state_g, batch, self.epsilon, self.cost_fn)
        return metrics


    def get_train_step(self) -> Callable:
        
        def loss(params_f: jnp.ndarray, params_g: jnp.ndarray, state_f: TrainState, state_g: TrainState, batch: Dict[str, jnp.ndarray], epsilon: float, cost_fn: Optional[str] = None):
            f_x = state_f.apply_fn({"params": params_f}, batch["source"])
            g_y = state_g.apply_fn({"params": params_g}, batch["target"])
            
            pc = pointcloud.PointCloud(batch["source"], batch["target"], cost_fn=cost_fn)
            c_xy = pc.cost_matrix
            t_1 = jnp.mean(f_x)
            t_2 = jnp.mean(g_y)
            t_3 = -epsilon * jnp.mean(jnp.exp((jnp.reshape(f_x, (-1, 1))+jnp.reshape(g_y, (1, -1))-c_xy)/epsilon))
            metrics = t_3, t_1, t_2
            return -(t_1+t_2-t_3), metrics
            
            
        def step_fn(state_f: TrainState, state_g: TrainState, batch: Dict[str, jnp.ndarray], epsilon: float, cost_fn: Optional[str]=None):
            grad_fn = jax.value_and_grad(loss, argnums=[0,1], has_aux=True)
            (obj_value, metrics), grads = grad_fn(state_f.params, state_g.params, state_f, state_g, batch, epsilon, cost_fn)
            term, g_mean, f_mean = metrics
            grads_f, grads_g = grads
            metrics = {"obj": obj_value, "term": term, "f_mean": f_mean, "g_mean": g_mean}
            
            return state_f.apply_gradients(grads=grads_f), state_g.apply_gradients(grads=grads_g), metrics
    
        return step_fn
    
    def learn_barycentric_projection(self, n_iters=1000):
    
        self.train_step_mlp = self.get_train_step_mlp()
        batch: Dict[str, jnp.ndarray]= {}
        for _ in tqdm(range(n_iters)):
            self.rng, rng_x, rng_y = jax.random.split(self.rng, 3)
            batch["source"] = self.x_loader(rng_x)
            batch["target"] = self.y_loader(rng_y)
            self.state_mlp, metrics = self.train_step_mlp(self.state_mlp, self.state_f, self.state_g, batch, self.epsilon, self.cost_fn)
            for key, value in metrics.items():
                self.mlp_metrics[key].append(value)

        return metrics
        


    def get_train_step_mlp(self) -> Callable:
        
        def loss(params_mlp: jnp.ndarray, state_mlp: TrainState, state_f: TrainState, state_g: TrainState, batch: Dict[str, jnp.ndarray], epsilon: float, cost_fn: Optional[str]=None):
            mlp_x = state_mlp.apply_fn({"params": params_mlp}, batch["source"])
            #loss = jnp.sum(mlp_x.flatten())
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
                    
        def step_fn(state_mlp: TrainState, state_f: TrainState, state_g: TrainState, batch: Dict[str, Any], epsilon: float, cost_fn: Optional[str]=None):
            grad_fn = jax.value_and_grad(loss, has_aux=True)
            (obj_value, metrics), grads_mlp = grad_fn(state_mlp.params, state_mlp, state_f, state_g, batch, epsilon, cost_fn)
            metrics = {"loss": obj_value, "d_xy": metrics[0], "H": metrics[1]}
            return state_mlp.apply_gradients(grads=grads_mlp), metrics
        
        return step_fn

    def transport(self, x: Optional[jnp.ndarray]=None) -> jnp.ndarray:
        if self.state_mlp is None:
            raise ValueError("Not trained yet.")
        x=x if x is not None else self.x_loader.data
        return self.state_mlp.apply_fn({"params": self.state_mlp.params}, x)
    
    def get_hessians_f(self, x: jnp.ndarray) -> jnp.ndarray:
        x = jnp.atleast_2d(x)
        hess_eval = jax.hessian(lambda x: 0.5 * jnp.inner(x, x) - self.state_f.apply_fn({"params": self.state_f.params}, x), argnums=0)
        vmap_hess_eval = jax.vmap(lambda x: hess_eval(x))
        return vmap_hess_eval(x)

    
        