# %%

%load_ext autoreload
%autoreload 2

import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"] = "7"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

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
from entot.nets.nets import MLP_vector_field, Bridge_MLP_mean, Bridge_MLP_full
from entot.plotting.plots import plot_1D_balanced
from entot.data.distributions import Gaussian, GaussianMixture

# %%

# source = Gaussian(
#     mean=jnp.array([0]),
#     cov=jnp.array([.5])
# )
# target = GaussianMixture(
#     means=jnp.array(
#         [[-2.], [2.]]
#     ),
#     covs=jnp.array(
#         [[.01], [.01]]
#     )
# )
# source, target = iter(source), iter(target)

source = MixtureNormalSampler(
    jax.random.PRNGKey(0), 
    [0], 1,  .5, 
    batch_size=1024
)
target = MixtureNormalSampler(
    jax.random.PRNGKey(1), 
    [-3., 3.], 1,  0.1, 
    batch_size=1024
)

# %%

# plot source and target
source_samples = jnp.concatenate(
    [next(source) for _ in range(10)],
    axis=0
)
target_samples = jnp.concatenate(
    [next(target) for _ in range(10)], 
    axis=0
)
fig, ax = plt.subplots(figsize=(8, 6))
sns.kdeplot(
    jnp.squeeze(source_samples),
    ax=ax,
    c='b',
    label='source',
    linewidth=2
)
sns.kdeplot(
    jnp.squeeze(target_samples),
    ax=ax,
    c='r',
    label='target',
    linewidth=2
)
ax.legend()
plt.show()


# %%

from entot.nets.nets import Bridge_MLP_full, Bridge_MLP_mean, Bridge_MLP_constant
neural_net = MLP_vector_field(
    1, 128, 128, 128, 
    n_frequencies=10
) # this was 10
bridge_net = Bridge_MLP_full(
    1, 128, 128, 128
)
# bridge_net = Bridge_MLP_mean(
#     1, 128, 128, 128
# )
# bridge_net = Bridge_MLP_constant(
#     1, 128, 128, 128
# )

# output_dim = 1
# params = bridge_net.init(
#     jax.random.PRNGKey(0), 
#     jnp.ones(output_dim)
# )['params']

# x = next(source)
# mu, std = bridge_net.apply(
#     {'params': params}, 
#     x
# )
# print(
#     mu.shape, std.shape
# )


# %%
epsilon = 1e-1
ot_solver = ott.solvers.linear.sinkhorn.Sinkhorn(
    momentum=ott.solvers.linear.acceleration.Momentum(
        value=1., start=25
    )
)
otfm = OTFlowMatching(
    neural_net, 
    beta=.01, 
    bridge_net=bridge_net, 
    ot_solver=ot_solver, 
    epsilon=epsilon, 
    input_dim=1, 
    output_dim=1, 
    iterations=2000, 
    seed=0
)
otfm(
    x=source, 
    y=target, 
    batch_size_source=1_024, 
    batch_size_target=1_024
)


# %%

plt.plot(
    np.arange(len(otfm.metrics["loss"])), 
    otfm.metrics["loss"]
)

# %%

source_samples = jnp.concatenate(
    [next(source) for _ in range(4)], 
    axis=0
)
source_samples = next(source)
res, _ , _= otfm.transport(
    source_samples, 
    seed=0, 
    diffeqsolve_kwargs={"max_steps": 10_000}
)
sns.kdeplot(res[0,...])

# %%

source_repeat = jnp.ones((1,)) * 0.0
source_repeat = jnp.repeat(source_repeat, 20)[:, None]
ts = np.linspace(0,1,20)
saveat=diffrax.SaveAt(ts=ts)
res, sol, _ = otfm.transport(
    source_repeat, seed=0, diffeqsolve_kwargs={"saveat": saveat}
)
tracks = sol.ys[...,0]
t_vec = ts[1:] - ts[:-1]
t_vector = np.tile(t_vec[:, None], (1, tracks.shape[-1]))
y_vector = tracks[1:, ...] - tracks[:-1,...]
t_augmented = np.tile(ts[:, None], (1, tracks.shape[-1]))[:-1, ...]
plt.quiver(t_augmented, tracks[:-1], t_vector, y_vector, angles="xy", scale=3)#,  headwidth=5, headlength=2, headaxislength=5)
plt.xlabel("Time")
plt.ylabel("Target space")
plt.title(f"Sampled paths, $\epsilon={epsilon}$")

# %%

mu_0, _ = otfm.state_bridge_net.apply_fn(
    {"params": otfm.state_bridge_net.params}, 
    condition=source_samples
)
sns.kdeplot(mu_0)
# sns.kdeplot(var_0)

# %%
