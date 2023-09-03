# %%

%load_ext autoreload
%autoreload 2

import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

import sys
sys.path.append('..')

# %%

import jax
import seaborn as sns
import jax
import ott
import diffrax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from entot.data.data import MixtureNormalSampler
from entot.models.model import OTFlowMatching
from entot.nets.nets import MLP_vector_field, MLP_bridge
from entot.plotting.plots import plot_1D_balanced
from entot.data.distributions import (
    Gaussian, 
    GaussianMixture, 
    SklearnDistribution
)
from typing import Any, Mapping, Optional
from ott.geometry import pointcloud, costs
from ott.tools import sinkhorn_divergence
from ott.solvers.linear import acceleration
from types import MappingProxyType

# %%

def plot_batch(
    batch,
    num_points_measures = 100,
    # num_points_cond = 20,
    seed = 0,
    size_points = 200,
    tick_size = 20,
    set_equal_axis = False,
    compute_sinkhorn_div = False,
    epsilon = 1e-1
    # indices_plot_cond = jnp.array([0, 500, 1_000, 1_500]),
):
    
    _, ax = plt.subplots(
        ncols=1,
        nrows=1,
        figsize=(10, 8)
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
    ax.scatter(
        batch["source"][subsample_measures, 0], 
        batch["source"][subsample_measures, 1],
        c="b", 
        edgecolors="k", 
        label=r"source $\mu$", 
        s=size_points, 
        alpha=0.8
    )
    ax.scatter(
        batch['target'][subsample_measures, 0],
        batch['target'][subsample_measures, 1],
        c="r",
        edgecolors="k",
        marker="X",
        label=r"target $\nu$",
        s=size_points,
        alpha=0.6,
    )
    mapped_samples_available = ("mapped_source" in batch)
    if mapped_samples_available:
        ax.scatter(
            batch['mapped_source'][subsample_measures, 0],
            batch['mapped_source'][subsample_measures, 1],
            c="orange",
            edgecolors="k",
            marker="X",
            label=r"learned target $\hat{\pi}_2$",
            s=size_points,
            alpha=0.8,
        )
    
    if compute_sinkhorn_div:
        assert mapped_samples_available, (
            "Sinkhorn divergence required mapped samples "
            "to be computed."
        )
        sinkhorn_div_value = sinkhorn_div(
            samples=batch['target'],
            mapped_samples=batch['mapped_source'],
            epsilon=epsilon
        )
        ax.set_title(
            r"Sinkhorn divergence: $S_{\varepsilon, \ell_2^2}(\hat{\pi}_2, \nu) = $" 
            + f"{sinkhorn_div_value:.6f}",
            fontsize=20          
        )
        
    ax.tick_params(
        axis="both", 
        which="major", 
        labelsize=tick_size
    )
    ax.grid(alpha=.4)
    ax.legend(
        fontsize=15,
        loc='upper right'
    )
    if set_equal_axis:
        ax.axis("equal")
    plt.show()
    plt.close()
    
def sinkhorn_div(
    samples: jnp.ndarray,
    mapped_samples: jnp.ndarray,
    epsilon: Optional[float] = None,
    sinkhorn_kwargs: Mapping[str, Any] = MappingProxyType({}),
    **kwargs: Any,
) -> Optional[float]:
    r"""Sinkhorn divergence fitting loss."""
    return sinkhorn_divergence.sinkhorn_divergence(
        pointcloud.PointCloud,
        x=samples,
        y=mapped_samples,
        epsilon=epsilon,
        sinkhorn_kwargs=sinkhorn_kwargs,
        **kwargs,
    ).divergence


# %%

source = SklearnDistribution(
    name="moon_upper",
    noise=.05
)
target = SklearnDistribution(
    name="moon_lower",
    noise=.05
)
source, target = iter(source), iter(target)

# %%
source_samples = jnp.concatenate(
    [next(source) for _ in range(5)], 
    axis=0
)
target_samples = jnp.concatenate(
    [next(target) for _ in range(5)], 
    axis=0
)
eval_batch = {
    "source": source_samples,
    "target": target_samples
}
plot_batch(
    batch=eval_batch,
)

# %%

from entot.nets.nets import (
    MLP_vector_field,
    old_MLP_vector_field
)

# vector field 
output_dim, input_dim = 2, 2
latent_embed_dim = 256
n_frequencies = 128
num_layers = 4
vector_field_net = MLP_vector_field(
    output_dim=output_dim, 
    latent_embed_dim=latent_embed_dim, 
    n_frequencies=n_frequencies,
    num_layers=num_layers
) 
# vector_field_net = old_MLP_vector_field(
#     2, 
#     128, 128, 128
# )

# %%

# from entot.nets.nets import (
#     MLP_bridge,
# )

# hidden_dim = 128
# bridge_type = "mean"
# bridge_net = MLP_bridge(
#     output_dim=output_dim,
#     hidden_dim=hidden_dim,
#     bridge_type=bridge_type,
#     num_layers=num_layers
# )

# epsilon = 1e-2
# ot_solver = ott.solvers.linear.sinkhorn.Sinkhorn(
#     momentum=ott.solvers.linear.acceleration.Momentum(
#         value=1., start=25
#     )
# )


# iterations = 10_000
# beta = 0.
# otfm = OTFlowMatching(
#     neural_net=vector_field_net, 
#     bridge_net=bridge_net, 
#     beta=beta, 
#     ot_solver=ot_solver, 
#     epsilon=epsilon, 
#     input_dim=input_dim, 
#     output_dim=output_dim, 
#     iterations=iterations, 
# )
# otfm(
#     x=source, 
#     y=target, 
# )

# # plot loss 
# fig, ax = plt.subplots(figsize=(8, 6))
# ax.plot(
#     np.arange(len(otfm.metrics["loss"])), 
#     otfm.metrics["loss"],   
# ) 
# ax.set_yscale("log")
# ax.set_title(
#     "loss throughout iterations in log-scale",
#     fontsize=20
# )
# plt.show()

# # transport and plot the samples
# mapped_source_samples, _ , _= otfm.transport(
#     eval_batch["source"], 
#     seed=0, 
#     diffeqsolve_kwargs={"max_steps": 10_000}
# )
# eval_batch["mapped_source"] = jnp.squeeze(mapped_source_samples)
# plot_batch(
#     batch=eval_batch,
#     compute_sinkhorn_div=True,
#     epsilon=1e-1
# )

# %%

list_bridge_type= [
    "full",
    "mean",
    "constant"
]
for i, bridge_type in enumerate(list_bridge_type):
    
    print(
        f"bridge_type = {bridge_type}: in progress..."
    )
    hidden_dim = latent_embed_dim
    bridge_net = MLP_bridge(
        output_dim=output_dim,
        hidden_dim=hidden_dim,
        bridge_type=bridge_type,
        num_layers=num_layers
    )
    # beta = (
    #     0. if bridge_type in ["mean", "constant"] 
    #     else .1
    # )
    beta = 0.
    epsilon = 1e-2
    iterations = 10_000
    ot_solver = ott.solvers.linear.sinkhorn.Sinkhorn(
        momentum=ott.solvers.linear.acceleration.Momentum(
            value=1., start=25
        )
    )
    otfm = OTFlowMatching(
        vector_field_net,
        beta=beta, 
        bridge_net=bridge_net, 
        ot_solver=ot_solver, 
        epsilon=epsilon, 
        input_dim=input_dim, 
        output_dim=output_dim, 
        iterations=iterations, 
    )
    otfm(
        x=source, 
        y=target, 
    )
    
    # plot loss 
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(
        np.arange(len(otfm.metrics["loss"])), 
        otfm.metrics["loss"],   
    ) 
    ax.set_yscale("log")
    ax.set_title(
        "loss throughout iterations in log-scale",
        fontsize=20
    )
    plt.show()
    plt.close()
    
    # transport and plot the samples
    mapped_source_samples, _ , _= otfm.transport(
        eval_batch["source"], 
        seed=0, 
        diffeqsolve_kwargs={"max_steps": 10_000}
    )
    eval_batch["mapped_source"] = jnp.squeeze(mapped_source_samples)
    plot_batch(
        batch=eval_batch,
        num_points_measures=200,
        compute_sinkhorn_div=True,
        epsilon=1e-1,
    )
    print(
        f"bridge_type = {bridge_type}: done!\n"
    )


# %%

# source_repeat = jnp.ones((1,)) * 0.0
# source_repeat = jnp.repeat(source_repeat, 20)[:, None]
# ts = np.linspace(0, 1, 20)
# saveat=diffrax.SaveAt(ts=ts)
# res, sol, _ = otfm.transport(
#     source_repeat, 
#     seed=0, diffeqsolve_kwargs={"saveat": saveat}
# )
# tracks = sol.ys[...,0]
# t_vec = ts[1:] - ts[:-1]
# t_vector = np.tile(t_vec[:, None], (1, tracks.shape[-1]))
# y_vector = tracks[1:, ...] - tracks[:-1,...]
# t_augmented = np.tile(ts[:, None], (1, tracks.shape[-1]))[:-1, ...]
# plt.quiver(t_augmented, tracks[:-1], t_vector, y_vector, angles="xy", scale=3)#,  headwidth=5, headlength=2, headaxislength=5)
# plt.xlabel("Time")
# plt.ylabel("Target space")
# plt.title(f"Sampled paths, $\epsilon={epsilon}$");

# %%

# mu_0, _ = otfm.state_bridge_net.apply_fn(
#     {"params": otfm.state_bridge_net.params}, 
#     condition=source_samples
# )
# sns.kdeplot(mu_0)
# # sns.kdeplot(var_0)

# %%
