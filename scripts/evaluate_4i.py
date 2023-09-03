
# %load_ext autoreload
# %autoreload 2

# import os

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   
# os.environ["CUDA_VISIBLE_DEVICES"] = "7"

import sys

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
from entot.models.model import OTFlowMatching, OTFlowMatching_
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
from entot.data import utils_cellot
from entot.nets.nets import MLP_vector_field
import hydra
from omegaconf import DictConfig, OmegaConf


@hydra.main(config_path="confs/single-cell", config_name="4i.yaml")
def evaluate(cfg: DictConfig):
        
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
    sinkhorn_div_fn = jax.tree_util.Partial(
        sinkhorn_div, 
        epsilon=1e-1,
    )

    # %%

    from entot.data import utils_cellot
    full_dataset = utils_cellot.load_dataset(
        drug_name = cfg.data.drug_name,
        drug_setting = "4i",
        where = "data_space"
    )
    iterators = utils_cellot.load_iterator(
        drug_name = cfg.data.drug_name,
        drug_setting = "4i",
        where = "data_space"
    )
    eval_batch = {
        "source": full_dataset.test.source.adata.to_df().values,
        "target": full_dataset.test.target.adata.to_df().values,
    }

    # vector field network
    input_dim = eval_batch["source"].shape[1]
    output_dim = eval_batch["target"].shape[1]
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
    bridge_type = cfg.model.bridge_type
    hidden_dim = latent_embed_dim
    bridge_net = MLP_bridge(
        output_dim=output_dim,
        hidden_dim=hidden_dim,
        bridge_type=bridge_type,
        num_layers=num_layers
    )
    beta = cfg.model.beta
    epsilon = cfg.model.epsilon
    iterations = 10_000
    ot_solver = ott.solvers.linear.sinkhorn.Sinkhorn(
        momentum=ott.solvers.linear.acceleration.Momentum(
            value=1., start=25
        )
    )
    
    # define wandb run name
    name_run_wandb = (
        f"{cfg.data.drug_name}; "
        f"{cfg.model.bridge_type}; "
        f"beta={cfg.model.beta} "
        f"epsilon={cfg.model.epsilon}"
    )
    
    # define solver
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
        name_run_wandb=name_run_wandb
    )
    otfm(
        x=iterators.train.source, 
        y=iterators.train.target, 
    )

if __name__ == '__main__':
    evaluate()