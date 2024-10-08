# %%

%load_ext autoreload
%autoreload 2

import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import sys
sys.path.append('..')

# %%

from pathlib import Path
import jax
import seaborn as sns
import jax
import ott
import diffrax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from genot.data.data import MixtureNormalSampler
from genot.models.model import OTFlowMatching, OTFlowMatching_
from genot.nets.nets import MLP_vector_field, MLP_bridge
from genot.plotting.plots import plot_1D_balanced
from genot.data.distributions import (
    Gaussian, 
    GaussianMixture, 
    SklearnDistribution
)
from typing import Any, Mapping, Optional
from ott.geometry import pointcloud, costs
from ott.tools import sinkhorn_divergence
from ott.solvers.linear import acceleration
from types import MappingProxyType
from genot.data import utils_cellot
import optax

# %%
    
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
sinkhorn_div_fn = jax.jit(
    jax.tree_util.Partial(
        sinkhorn_div, 
        epsilon=1e-1,
    )
)

# %%

from genot.data import utils_cellot
full_dataset = utils_cellot.load_dataset(
    drug_name = "cisplatin",
    drug_setting = "4i",
    where = "data_space"
)
iterators = utils_cellot.load_iterator(
    drug_name = "cisplatin",
    drug_setting = "4i",
    where = "data_space"
)
print(
   f'train source shape: {full_dataset.train.source.adata.to_df().values.shape}'
)
print(
    f'train target shape: {full_dataset.train.target.adata.to_df().values.shape}'
)
print("")
print(
   f'test source shape: {full_dataset.test.source.adata.to_df().values.shape}'
)
print(
    f'test target shape: {full_dataset.test.target.adata.to_df().values.shape}'
)
eval_batch = {
    "source": full_dataset.test.source.adata.to_df().values,
    "target": full_dataset.test.target.adata.to_df().values,
}

# %%

from genot.nets.nets import MLP_vector_field


# vector field network
x_train = jnp.array(full_dataset.train.source.adata.to_df().values)
y_train = jnp.array(full_dataset.train.target.adata.to_df().values)
input_dim = x_train.shape[1]
output_dim = y_train.shape[1]
latent_embed_dim = 256
n_frequencies = 128
num_layers = 5
vector_field_net = MLP_vector_field(
    output_dim=output_dim, 
    latent_embed_dim=latent_embed_dim, 
    n_frequencies=n_frequencies,
    num_layers=num_layers
) 

# bridge newtork
bridge_type = "constant"
hidden_dim = latent_embed_dim
bridge_net = MLP_bridge(
    output_dim=output_dim,
    hidden_dim=hidden_dim,
    bridge_type=bridge_type,
    num_layers=num_layers
)
beta = 0.
epsilon = 1e-1
iterations = 30_000
ot_solver = ott.solvers.linear.sinkhorn.Sinkhorn(
    momentum=ott.solvers.linear.acceleration.Momentum(
        value=1., start=25
    )
)
optimizer = optax.adamw(
    learning_rate=1e-3,
    # weight_decay=1e-4
)
otfm = OTFlowMatching_(
    vector_field_net,
    beta=beta, 
    bridge_net=bridge_net, 
    ot_solver=ot_solver, 
    epsilon=epsilon, 
    input_dim=input_dim, 
    output_dim=output_dim, 
    iterations=iterations, 
    log_wandb=True,
    eval_batch=eval_batch,
    logging=True,
    log_freq=1_000,
    sink_div_fn=sinkhorn_div_fn,
    optimizer=optimizer
)

# %%

# otfm(
#     x=full_dataset.train.source.adata.to_df().values, 
#     batch_size_source=1_024,
#     y=full_dataset.train.target.adata.to_df().values, 
#     batch_size_target=1_024,
# )
otfm(
    x=iterators.train.source, 
    y=iterators.train.target, 
)


# %%
