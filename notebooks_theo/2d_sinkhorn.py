# %%

%load_ext autoreload
%autoreload 2

import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"] = "7"

import sys
sys.path.append('..')

# %%
from functools import partial
import seaborn as sns
import jax
import ott
import diffrax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from genot.data.data import MixtureNormalSampler
from genot.models.model import OTFlowMatching
from genot.nets.nets import MLP_vector_field, Bridge_MLP_mean
from genot.plotting.plots import plot_1D_balanced
from genot.data.distributions import (
    GaussianMixture,
    SklearnDistribution, 
    SphereDistribution,
    RectangleUniform,
    BallUniform,
    Dataset
)
from typing import Optional
from genot.plotting.plotters import (
    plot_fitted_map_gromov,
    plot_source_and_target_spherical
)
from ott.geometry import costs, pointcloud
from ott.problems.linear import linear_problem
from ott.solvers.linear import sinkhorn, acceleration
import numpy as np
# %%

# plotter = partial(
#     plot_fitted_map_gromov, 
#     view_init=(20, 40),
#     # num_points_plot=None,
#     title_size=20,
#     legend_size=25,
#     tick_size=20,
#     size_points=150,
#     alpha=.8,
#     # lims={
#     #     'source': [(None, None), (None, None), (None, None)],
#     #     'target': [(-4, 4), (-1.5, 1.5)],
#     # },
#     set_equal_aspect_source=True,
#     set_equal_aspect_target=True,
#     plot_sphere_source=True,
#     num_points=512
# )


def plot_fitted_plan(
    batch, 
    num_points=None,
    seed=0,
    show=True,
    return_fig=False,
    *args,
    **kwargs,
):
    """
    Plot the fitted map on a batch of samples from the source measure.
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    if num_points is None:
        subsample = jnp.arange(len(batch['source']))
    else:
        rng = jax.random.PRNGKey(seed)
        subsample = jax.random.choice(rng, a=len(batch['source']), shape=(num_points,))

    # subsample and plot samples from the batch
    ax.scatter(
        batch["source"][subsample, 0], 
        batch["source"][subsample, 1],
        c="b", 
        edgecolors="k", 
        label="source samples", 
        s=150, alpha=0.8
    )
    ax.scatter(
        batch['target'][subsample, 0],
        batch['target'][subsample, 1],
        c="r",
        edgecolors="k",
        marker="X",
        label="target samples",
        s=150,
        alpha=0.6,
    )
    
    # plot image of the samples by the monge map if available
    monge_map_avaliable = "monge_target" in batch.keys()
    if monge_map_avaliable:
        ax.scatter(
            batch['monge_target'][subsample, 0],
            batch['monge_target'][subsample, 1],
            c="green",
            edgecolors="k",
            label="target",
            s=60,
            alpha=0.8,
        )

    # map and plot samples with the fitted map
    z = batch['mapped_source'] -  batch['source']
    ax.scatter(
        batch['mapped_source'][subsample, 0],
        batch['mapped_source'][subsample, 1],
        c="orange",
        edgecolors="k",
        marker="X",
        label="push-forward samples",
        s=150,
        alpha=0.8,
    )
    ax.quiver(
        batch['source'][subsample, 0],
        batch['source'][subsample, 1],
        z[subsample, 0],
        z[subsample, 1],
        angles="xy",
        scale_units="xy",
        scale=1.0,
        width=0.002,
        headwidth=10,
        headlength=10,
        color="dodgerblue",
        alpha=0.5,
        edgecolor="k",
    )

    # define the title according to available inputs
    if monge_map_avaliable:
        ax.set_title(
            r"Fitted map $T_{\hat{\theta}}$ vs. Monge map $T^*$", fontsize=30, y=1.01
        )
    else:
        ax.set_title(r"Fitted map $\hat{T}_\theta$", fontsize=30, y=1.01)

    ax.tick_params(axis="both", which="major", labelsize=20)
    ax.legend(fontsize=25)

    fig.tight_layout()
    if show:
        plt.show()
    plt.close()
    if return_fig:
        return fig
    


# %%

source = SklearnDistribution(
    name="moon_upper",
)
target = SklearnDistribution(
    name="moon_lower"
)

batch = {
    "source": next(iter(source)),
    "target": next(iter(target)),
}

# %%

geom = pointcloud.PointCloud(
    x=batch['source'],
    y=batch['target'],
    epsilon=1e-2,
    relative_epsilon=True
)
ot_prob = linear_problem.LinearProblem(geom)
solver = sinkhorn.Sinkhorn(
    momentum=acceleration.Momentum(value=1., start=25)
)
ot = solver(ot_prob)

# %%

n = len(batch['source'])
cond_probs = n * ot.matrix
k_samples_per_x = 20
sample_cond_from_index = lambda i: jax.random.choice(
    key=jax.random.PRNGKey(i),
    a=batch["target"],
    p=ot.matrix[i],
    shape=(k_samples_per_x,)
)
cond_samples = jax.vmap(sample_cond_from_index)(jnp.arange(n))
batch["conditional_target"] = cond_samples

# %%

num_points_measures = 100
num_points_cond = 20
indices_plot_cond = jnp.array([0, 500, 1_000, 1_500])
seed = 0
size_points = 200
tick_size = 20
fig, ax = plt.subplots(
    ncols=2,
    nrows=1,
    figsize=(10*2, 8)
)

def set_subsample_indices(
    num_points_to_samples: Optional[int], 
    num_total_points: int,
    rng: jax.random.PRNGKeyArray,
):
    # pick all points in the batch if no subsampling
    if (
        num_points_to_samples is None
        or
        num_points_to_samples == num_total_points
    ):
        return jnp.arange(num_total_points)
    
    # subsampling
    else:
        rng = jax.random.PRNGKey(seed)
        return jax.random.choice(
            rng, 
            a=num_total_points, 
            shape=(num_points_to_samples,)
        )

num_total_points = len(batch['source'])
rng = jax.random.PRNGKey(seed)

# plot source and target measures
rng, _ = jax.random.split(rng, 2)
subsample_measures = set_subsample_indices(
    num_points_to_samples=num_points_measures,
    num_total_points=num_total_points,
    rng=rng
)
for i in range(2):
    ax[i].scatter(
        batch["source"][subsample_measures, 0], 
        batch["source"][subsample_measures, 1],
        c="b", 
        edgecolors="k", 
        label=r"source $\mu$", 
        s=size_points, 
        alpha=0.8
    )
    ax[i].scatter(
        batch['target'][subsample_measures, 0],
        batch['target'][subsample_measures, 1],
        c="r",
        edgecolors="k",
        marker="X",
        label=r"target $\nu$",
        s=size_points,
        alpha=0.6,
    )
# plot fitted target measure
rng, _ = jax.random.split(rng, 2)
select_cond = jax.random.choice(
    key=rng, 
    a=num_points_cond, 
    shape=(num_total_points,)
)
target_samples = batch["conditional_target"][jnp.arange(n), select_cond]
ax[0].scatter(
    target_samples[subsample_measures, 0],
    target_samples[subsample_measures, 1],
    c="orange",
    edgecolors="k",
    marker="X",
    label=r"fitted target $\hat{\pi}_2$",
    s=size_points,
    alpha=0.6,
)

# plot conditional samples
if "conditional_target" in  batch:
    num_total_cond_samples = len(batch['conditional_target'][0])
    subsample_cond = set_subsample_indices(
        num_points_to_samples=num_points_cond,
        num_total_points=num_total_cond_samples,
        rng=rng
    )
    c = 0
    for indx in indices_plot_cond:
        x =  batch['source'][indx]
        label = (
            "" if c > 0 
            else r"conditional $\hat{\pi}(\cdot | \mathbf{x})$"
        )
        c += 1
        ax[1].scatter(
            batch["conditional_target"][indx, subsample_cond, 0], 
            batch["conditional_target"][indx, subsample_cond, 1],
            c="orange",
            edgecolors="k", 
            label=label, 
            marker="X",
            s=size_points, 
            alpha=0.8
        )
        z = batch["conditional_target"][indx] -  x[None, :]
        ax[1].quiver(
            jnp.tile(x, (num_points_cond, 1))[:, 0],
            jnp.tile(x, (num_points_cond, 1))[:, 1],
            z[subsample_cond, 0],
            z[subsample_cond, 1],
            angles="xy",
            scale_units="xy",
            scale=1.0,
            width=0.002,
            headwidth=10,
            headlength=10,
            color="dodgerblue",
            alpha=0.5,
            edgecolor="k",
        )

for i in range(2):
    ax[i].tick_params(
        axis="both", 
        which="major", 
        labelsize=tick_size
    )
    ax[i].grid(alpha=.4)
    ax[i].legend(
        fontsize=15,
        loc='upper right'
    )
plt.show()


# %%

fig, ax = plt.subplots(figsize=(10,8))
ax.scatter(
    batch["source"][:, 0], 
    batch["source"][:, 1],
    c="b", 
    edgecolors="k", 
)
for i in range(100):
    print(i)
    ax.scatter(
        batch["source"][i, 0], 
        batch["source"][i, 1],
        c="green", 
        edgecolors="k", 
    )
    plt.show()
    time.sleep(5)
# %%
