# %%

%load_ext autoreload
%autoreload 2

import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"] = "7"

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
    Gaussian, GaussianMixture, 
    SklearnDistribution
)
from typing import Optional

# %%

def plot_batch(
    batch,
    num_points_measures = 100,
    # num_points_cond = 20,
    seed = 0,
    size_points = 200,
    tick_size = 20,
    set_equal_axis = False,
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
    if "mapped_source" in batch:
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

# %%

source = SklearnDistribution(
    name="moon_upper",
    noise=.05
)
target = SklearnDistribution(
    name="moon_lower",
    noise=.05
)

# source = Gaussian(
#     mean = jnp.array([0, -1]),
#     cov = jnp.array(
#         [
#             [.7, 0], 
#             [0, .1]
#         ]
#     )
# )
# target = GaussianMixture(
#     means = jnp.array(
#         [
#             [-3, 1], 
#             [0, 1],
#             [3, 1]
#         ]
#     ),
#     covs = jnp.tile(
#         .05*jnp.eye(2), (3, 1, 1)
#     )
# )
source, target = iter(source), iter(target)

# %%

batch = {
    "source": next(source),
    "target": next(target)
}
plot_batch(
    batch=batch,
)


# %%

from entot.nets.nets import (
    MLP_vector_field,
)

# vector field 
output_dim, input_dim = 2, 2
latent_embed_dim = 128
n_frequencies = 10
num_layers = 3
vector_field_net = MLP_vector_field(
    output_dim=output_dim, 
    latent_embed_dim=latent_embed_dim, 
    n_frequencies=n_frequencies,
    num_layers=num_layers
) 

# rng = jax.random.PRNGKey(0)
# input_dim = 2
# params = vector_field_net.init(
#     jax.random.PRNGKey(0), 
#     jnp.ones((1, 1)), 
#     jnp.ones((1, input_dim)), 
#     jnp.ones((1, output_dim))
# )["params"]

# x = batch['source']
# t = jax.random.uniform(
#     rng, 
#     shape=(len(x),)
# )[:, None]
# z = jax.random.normal(
#     rng, 
#     shape=x.shape
# )
# out = vector_field_net.apply(
#     {'params': params}, 
#     t=t, 
#     condition=x,
#     latent=z
# )
# print(
#     out
# )

# %%

from entot.nets.nets import (
    MLP_bridge,
)

hidden_dim = 128
bridge_type = "full"
bridge_net = MLP_bridge(
    output_dim=output_dim,
    hidden_dim=hidden_dim,
    bridge_type=bridge_type,
    num_layers=num_layers
)

epsilon = 1e-2
ot_solver = ott.solvers.linear.sinkhorn.Sinkhorn(
    momentum=ott.solvers.linear.acceleration.Momentum(
        value=1., start=25
    )
)

# %%

iterations = 10_000
beta = 0.
otfm = OTFlowMatching(
    neural_net=vector_field_net, 
    bridge_net=bridge_net, 
    beta=beta, 
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

# %%

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

# %%

# transport the samples
# THIS DOES NOT WORK
source_samples = jnp.concatenate(
    [next(source) for _ in range(4)], 
    axis=0
)
source_samples = next(source)
mapped_source_samples, _ , _= otfm.transport(
    source_samples, 
    seed=0, 
    diffeqsolve_kwargs={"max_steps": 10_000}
)
batch["mapped_source"] = jnp.squeeze(mapped_source_samples)
plot_batch(batch=batch)

# TRYING TO INVESTIGATE THE ERROR
# %%

mapped_source_samples, _ , _= otfm.transport(
    source_samples, 
    seed=0, 
    diffeqsolve_kwargs={"max_steps": 10_000}
)

# %%

seed = 0
diffeqsolve_kwargs = dict({})
rng = jax.random.PRNGKey(seed)
latent_shape = (len(batch['source']),)
latent_batch = otfm.noise_fn(rng, shape=latent_shape)
mu_0, std_0 = otfm.state_bridge_net.apply_fn(
    {"params": otfm.state_bridge_net.params}, 
    condition=batch['source']
)
print(
    mu_0.shape, 
    std_0.shape,
    latent_batch.shape
)
mu_noisy = mu_0 + latent_batch * std_0
print(
    mu_noisy.shape
)
apply_fn_partial = jax.tree_util.Partial(
    otfm.state_neural_net.apply_fn, condition=source
)
solution = diffrax.diffeqsolve(
    diffrax.ODETerm(
        lambda t, y, *args: apply_fn_partial(
            {"params": otfm.state_neural_net.params}, 
            t=t, 
            latent=y
        )
    ),
    diffeqsolve_kwargs.pop("solver", diffrax.Tsit5()),
    t0=0,
    t1=1,
    dt0=diffeqsolve_kwargs.pop("dt0", None),
    y0=mu_noisy,
    stepsize_controller=diffeqsolve_kwargs.pop(
        "stepsize_controller", diffrax.PIDController(rtol=1e-3, atol=1e-6)
    ),
    **diffeqsolve_kwargs,
)

# %%

x = batch['source']
t = jax.random.uniform(
    rng, 
    shape=(len(x),)
)[:, None]
z = jax.random.normal(
    rng, 
    shape=x.shape
)
out = otfm.state_neural_net.apply_fn(
    {'params': otfm.state_neural_net.params}, 
    t=t[0], 
    condition=x[0],
    latent=z[0]
)
print(
    out
)

# %%

# list_bridge_type= [
#     "full",
#     "mean",
#     "const"
# ]
# for i, bridge_type in enumerate(list_bridge_type):
    
#     print(bridge_type)
    
#     hidden_dim = 128
#     bridge_net = MLP_bridge(
#         output_dim=output_dim,
#         hidden_dim=hidden_dim,
#         bridge_type=bridge_type,
#         num_layers=num_layers
#     )
#     otfm = OTFlowMatching(
#         neural_net, 
#         beta=0., #.01, 
#         bridge_net=bridge_net, 
#         ot_solver=ot_solver, 
#         epsilon=epsilon, 
#         input_dim=2, 
#         output_dim=2, 
#         iterations=10_000, 
#         seed=0
#     )
#     otfm(
#         x=source, 
#         y=target, 
#         batch_size_source=1_024, 
#         batch_size_target=1_024
#     )
    
#     # plot loss 
#     fig, ax = plt.subplots(figsize=(8, 6))
#     ax.plot(
#         np.arange(len(otfm.metrics["loss"])), 
#         otfm.metrics["loss"],   
#     ) 
#     ax.set_yscale("log")
#     ax.set_title(
#         "loss throughout iterations in log-scale",
#         fontsize=20
#     )
#     plt.show()

#     source_samples = jnp.concatenate(
#         [next(source) for _ in range(4)], 
#         axis=0
#     )
#     source_samples = next(source)
#     mapped_source_samples, _ , _= otfm.transport(
#         source_samples, 
#         seed=0, 
#         diffeqsolve_kwargs={"max_steps": 10_000}
#     )
#     batch["mapped_source"] = jnp.squeeze(mapped_source_samples)
#     plot_batch(batch=batch)


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
