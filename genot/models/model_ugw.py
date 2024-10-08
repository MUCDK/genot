# ott-version 0.4.3.dev22+gf275dc4




from ott.initializers.quadratic.initializers import BaseQuadraticInitializer
import numpy as np
from ott.geometry import geometry
from ott.geometry.geometry import Geometry
from ott.problems.quadratic import quadratic_problem
from ott.solvers.quadratic import gromov_wasserstein
from ott.geometry import costs, pointcloud
from ott.problems.linear import linear_problem
from ott.solvers.linear import sinkhorn, acceleration
from typing import Any






class MyQuadraticInitializer(BaseQuadraticInitializer):
    def __init__(
          self, init_coupling: Optional[jnp.ndarray] = None, **kwargs: Any
      ):
        super().__init__(**kwargs)
        self.init_coupling = init_coupling

    def _create_geometry(
      self,
      quad_prob: "quadratic_problem.QuadraticProblem",
      *,
      epsilon: float,
      relative_epsilon: Optional[bool] = None,
      **kwargs: Any,
        ) -> geometry.Geometry:
        """Compute initial geometry for linearization.
        
        Args:
          quad_prob: Quadratic OT problem.
          epsilon: Epsilon regularization.
          relative_epsilon: Flag, use `relative_epsilon` or not in geometry.
          kwargs: Keyword arguments for :class:`~ott.geometry.geometry.Geometry`.
        
        Returns:
          The initial geometry used to initialize the linearized problem.
        """
        from ott.problems.quadratic import quadratic_problem
        
        del kwargs
        
        marginal_cost = quad_prob.marginal_dependent_cost(quad_prob.a, quad_prob.b)
        geom_xx, geom_yy = quad_prob.geom_xx, quad_prob.geom_yy
        
        h1, h2 = quad_prob.quad_loss
        if self.init_coupling is None:
          tmp1 = quadratic_problem.apply_cost(geom_xx, quad_prob.a, axis=1, fn=h1)
          tmp2 = quadratic_problem.apply_cost(geom_yy, quad_prob.b, axis=1, fn=h2)
          tmp = jnp.outer(tmp1, tmp2)
        else:
          tmp1 = h1.func(geom_xx.cost_matrix)
          tmp2 = h2.func(geom_yy.cost_matrix)
          tmp = tmp1 @ self.init_coupling @ tmp2.T
        
        if quad_prob.is_balanced:
          cost_matrix = marginal_cost.cost_matrix - tmp
        else:
          # initialize epsilon for Unbalanced GW according to Sejourne et. al (2021)
          init_transport = jnp.outer(quad_prob.a, quad_prob.b)
          marginal_1, marginal_2 = init_transport.sum(1), init_transport.sum(0)
        
          epsilon *= marginal_1.sum()
          unbalanced_correction = quad_prob.cost_unbalanced_correction(
              init_transport, marginal_1, marginal_2, epsilon=epsilon
          )
          cost_matrix = marginal_cost.cost_matrix - tmp + unbalanced_correction
        
        cost_matrix += quad_prob.fused_penalty * quad_prob._fused_cost_matrix
        return geometry.Geometry(
            cost_matrix=cost_matrix,
            epsilon=epsilon,
            relative_epsilon=relative_epsilon
        )
        
        def tree_flatten(self) -> Tuple[Sequence[Any], Dict[str, Any]]:  # noqa: D102
            return [self.init_coupling], self._kwargs