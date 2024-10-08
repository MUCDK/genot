# jax, optax
import jax
import jax.numpy as jnp
import optax

# typing, types
from typing import Any, Mapping, Optional
from types import MappingProxyType

# hydra, omegaconf
import hydra
from omegaconf import DictConfig

# ott
from ott.geometry import pointcloud
from ott.tools import sinkhorn_divergence
from ott.solvers.linear import acceleration, sinkhorn

# set directories before own code imports 
import sys, os
from pathlib import Path
script_dir = Path(__file__).parent
project_dir = next(
    (
        parent for parent in script_dir.parents 
        if parent.name == "genot"
    ), None
)
if project_dir and (str(project_dir) not in sys.path):
    sys.path.append(str(project_dir))
config_dir = os.path.join(project_dir, "confs/single-cell")
cellot_path = project_dir / "cellot"

# own code
from genot.data import utils_cellot
from genot.models.model import OTFlowMatching_
from genot.nets.nets import MLP_vector_field, MLP_bridge

@hydra.main(config_path=config_dir, config_name="4i.yaml")
def evaluate(cfg: DictConfig) -> None:
    
    ##### UGLY BUT TEMPORARY
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
    ###############

    # set iterators and testing dataset
    global cellot_path
    iterators = utils_cellot.load_iterator(
        drug_name=cfg.data.drug_name,
        drug_setting=cfg.data.drug_setting,
        where=cfg.data.where, 
        cellot_path=cellot_path,
    )
    full_dataset = utils_cellot.load_dataset(
        drug_name=cfg.data.drug_name,
        drug_setting=cfg.data.drug_setting,
        where=cfg.data.where,
        cellot_path=cellot_path,
    )
    test_batch = {
        "source": full_dataset.test.source.adata.to_df().values,
        "target": full_dataset.test.target.adata.to_df().values,
    }

    # set input and output dimensions
    input_dim = test_batch["source"].shape[1]
    output_dim = test_batch["target"].shape[1]
    
    # set vector field network
    vector_field_act_fn = getattr(jax.nn, cfg.model.vector_field.act_fn) 
    vector_field_net = MLP_vector_field(
        output_dim=output_dim, 
        latent_embed_dim=cfg.model.vector_field.latent_embed_dim, 
        n_frequencies=cfg.model.vector_field.n_frequencies,
        num_layers=cfg.model.vector_field.num_layers,
        act_fn=vector_field_act_fn,
    ) 

    # set bridge newtork and hyperparameters
    bridge_act_fn = getattr(jax.nn, cfg.model.bridge.act_fn) 
    bridge_net = MLP_bridge(
        output_dim=output_dim,
        hidden_dim=cfg.model.bridge.hidden_dim,
        bridge_type=cfg.model.bridge.bridge_type,
        num_layers=cfg.model.bridge.num_layers,
        act_fn=bridge_act_fn,
    )
    beta = cfg.model.bridge.beta
    
    # set entropic regularization strength and ot solver
    epsilon = cfg.model.epsilon
    use_momentum = (cfg.model.ot_solver.momentum.value is not None)
    momentum = (
        None if use_momentum 
        else acceleration.Momentum(
            value=cfg.model.ot_solver.momentum.value, 
            start=cfg.model.ot_solver.momentum.start
        )
    )
    ot_solver = sinkhorn.Sinkhorn(
        momentum=momentum
    )
    
    # define wandb run name
    name_run_wandb = (
        f"{cfg.data.drug_name}; "
        f"{cfg.model.bridge.bridge_type}; "
        f"beta={cfg.model.bridge.beta} "
        f"epsilon={cfg.model.epsilon}"
    )
    
    # define optimizer
    iterations = cfg.optim.num_iterations
    optimizer = optax.adamw(
        learning_rate=cfg.optim.learning_rate,
        weight_decay=cfg.optim.weight_decay
    )
    
    # define solver
    otfm = OTFlowMatching_(
        
        # input and output dimension
        input_dim=input_dim, 
        output_dim=output_dim, 
        
        # vector field 
        neural_net=vector_field_net,
        
        # bridge
        bridge_net=bridge_net,
        beta=beta, 
        
        # ot solver and entropic regularization strentgh
        epsilon=epsilon, 
        ot_solver=ot_solver, 
        
        # optimization
        iterations=iterations, 
        optimizer=optimizer,
        
        # logging
        logging=cfg.logging.logging,
        log_freq=cfg.logging.log_freq,
        log_wandb=cfg.logging.log_wandb,
        test_batch=test_batch,
        sink_div_fn=sinkhorn_div_fn,
        name_run_wandb=name_run_wandb
    )
    otfm(
        x=iterators.train.source, 
        y=iterators.train.target, 
    )
    
if __name__ == '__main__':
    evaluate()