# %%

%load_ext autoreload
%autoreload 2

import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import sys
sys.path.append('../..')

# %%
import collections
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
from entot.data.data import MixtureNormalSampler
from entot.models import model, new_model
from entot.nets.nets import MLP_vector_field, MLP_bridge
from entot.plotting.plots import plot_1D_balanced
from entot.data.distributions import (
    Gaussian, 
    GaussianMixture, 
    SklearnDistribution
)
from typing import Any, Callable, Dict, Mapping, Optional
from ott.geometry import pointcloud, costs
from ott.tools import sinkhorn_divergence
from ott.solvers.linear import acceleration
from types import MappingProxyType
from entot.data import utils_cellot
import optax
from ott.solvers.linear import sinkhorn
from ott.problems.linear import linear_problem
import wandb
from entot.utils import metrics

# %%

from entot.data import utils_cellot
full_data_space_dataset = utils_cellot.load_dataset(
    drug_name = "abexinostat",
    drug_setting = "scRNAseq",
    where = "data_space",
    cellot_path=Path("../../cellot/"),
)
full_latent_dataset = utils_cellot.load_dataset(
    drug_name = "abexinostat",
    drug_setting = "scRNAseq",
    where = "latent_space",
    cellot_path=Path("../../cellot/"),
)
encode, decode = utils_cellot.load_encode_decode(
    drug_name = "abexinostat",
    cellot_path=Path("../../cellot/"),
)
print(
   f'train source shape: {full_data_space_dataset.train.source.adata.to_df().values.shape}'
)
print(
    f'train target shape: {full_data_space_dataset.train.target.adata.to_df().values.shape}'
)
print("")
print(
   f'test source shape: {full_data_space_dataset.test.source.adata.to_df().values.shape}'
)
print(
    f'test target shape: {full_data_space_dataset.test.target.adata.to_df().values.shape}'
)
print("")
print(
   f'train source shape: {full_latent_dataset.train.source.adata.to_df().values.shape}'
)
print(
    f'train target shape: {full_latent_dataset.train.target.adata.to_df().values.shape}'
)
print("")
print(
   f'test source shape: {full_latent_dataset.test.source.adata.to_df().values.shape}'
)
print(
    f'test target shape: {full_latent_dataset.test.target.adata.to_df().values.shape}'
)

# %%

eval_batch = {
    "source": full_data_space_dataset.test.source.adata.to_df().values,
    "target": full_data_space_dataset.test.target.adata.to_df().values,
}

# %%

test_entropic_map = False
if test_entropic_map:
    x_train = jnp.array(full_latent_dataset.train.source.adata.to_df().values)
    y_train = jnp.array(full_latent_dataset.train.target.adata.to_df().values)

    x_test = jnp.array(full_data_space_dataset.test.source.adata.to_df().values)
    y_test = jnp.array(full_data_space_dataset.test.target.adata.to_df().values)

    solver = jax.jit(sinkhorn.Sinkhorn())
    def entropic_map(x, y, cost_fn: costs.TICost) -> jnp.ndarray:
        geom = pointcloud.PointCloud(x, y, cost_fn=cost_fn)
        output = solver(linear_problem.LinearProblem(geom))
        dual_potentials = output.to_dual_potentials()
        return dual_potentials.transport

    map = entropic_map(x_train, y_train, costs.SqEuclidean())
    enc_x_test = encode(x_test)
    enc_mapped_samples = map(enc_x_test)
    dec_mapped_samples = decode(enc_mapped_samples)
    print(
        metrics.sinkhorn_divergence_fn(
            samples=y_test,
            mapped_samples=dec_mapped_samples
        )
    )

# %%

# vector field network
x_train = jnp.array(full_latent_dataset.train.source.adata.to_df().values)
y_train = jnp.array(full_latent_dataset.train.target.adata.to_df().values)
input_dim = x_train.shape[1]
output_dim = y_train.shape[1]
latent_embed_dim = 256
joint_hidden_dim = 256
n_frequencies = 128
num_layers = 3
vector_field_net = MLP_vector_field(
    output_dim=output_dim, 
    latent_embed_dim=latent_embed_dim, 
    joint_hidden_dim=joint_hidden_dim,
    n_frequencies=n_frequencies,
    num_layers=num_layers
) 


# bridge newtork
bridge_type = "constant"
hidden_dim = 256
bridge_net = MLP_bridge(
    output_dim=output_dim,
    hidden_dim=256,
    bridge_type=bridge_type,
    num_layers=num_layers
)
beta = 0.
epsilon = 1e-1
iterations = 30_000
ot_solver = ott.solvers.linear.sinkhorn.Sinkhorn(
    momentum=ott.solvers.linear.acceleration.Momentum(
        value=1., start=25
    ),
    threshold=1e-2
)
optimizer = optax.adamw(
    learning_rate=1e-3,
    weight_decay=1e-5
)

# %%

def cellot_callback_fn(
    solver: model.OTFlowMatching,
    test_batch: Dict[str, jnp.ndarray],
    num_sample_conditional: int = 50.,
    encode: Optional[Callable[[jnp.ndarray], jnp.ndarray]] = None,
    decode: Optional[Callable[[jnp.ndarray], jnp.ndarray]] = None,
    epsilon: float = 1e-1,
    seed: int = 0,
    **kwargs: Any,
):
    rng = jax.random.PRNGKey(seed)
    rngs = jax.random.split(rng, num=num_sample_conditional)
    
    # check that a decoder is provided if an encoder is provided
    # and vice versa
    assert (encode is None) == (decode is None), (
        "If an encoder is provided,"
        " a decoder must also be provided and vice-versa."
    )
    encode = encode or (lambda x: x)
    decode = decode or (lambda x: x)
        
    # transport samples
    samples_to_map = encode(test_batch["source"])
    mapped_samples, _ , _= jax.vmap(
        lambda rng: solver.transport(
            samples_to_map, 
            rng=rng, 
            diffeqsolve_kwargs={"max_steps": 10_000}
        )
    )(rngs)
    mapped_samples = decode(
        jnp.squeeze(
            jnp.mean(mapped_samples, axis=0)
        )
    )
    
    # compute sinkhorn divergences
    test_sd_value = metrics.sinkhorn_divergence_fn(
        samples=test_batch["target"],
        mapped_samples=mapped_samples,
        epsilon=epsilon
    )
    dict_metrics = {
        "test_sinkhorn_divergence": test_sd_value,
    }
        
    return dict_metrics

callback_kwargs = {
    "epsilon": 1e-1,
    "num_sample_conditional": 10,
    "seed": 0,
    "encode": encode,
    "decode": decode
}


# %%

otfm = new_model.OTFlowMatching(
    vector_field_net,
    beta=beta, 
    bridge_net=bridge_net, 
    ot_solver=ot_solver, 
    epsilon=epsilon, 
    input_dim=input_dim, 
    output_dim=output_dim, 
    iterations=iterations, 
    log_wandb=True,
    callback_freq=500,
    callback_fn=cellot_callback_fn,
    callback_kwargs=callback_kwargs,
    test_batch=eval_batch,
)

# %%

otfm(
    x=x_train, 
    y=y_train, 
    batch_size_source=1_024,
    batch_size_target=1_024
)
# %%
