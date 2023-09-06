# jax, optax
import jax
import optax

# hydra, omegaconf
import hydra
from omegaconf import DictConfig, OmegaConf

# ott
from ott.solvers.linear import acceleration, sinkhorn

# wandb
import wandb

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
if (project_dir) and (str(project_dir) not in sys.path):
    sys.path.append(str(project_dir))
config_dir = os.path.join(project_dir, "confs/single-cell")
cellot_path = project_dir / "cellot"

# own code
from entot.data import utils_cellot
from entot.models import model, new_model
from entot.nets.nets import MLP_vector_field, MLP_bridge
from entot.utils import callbacks

@hydra.main(config_path=config_dir, config_name="4i.yaml")
def evaluate(cfg: DictConfig) -> None:

    # set iterators and testing dataset
    global cellot_path
    full_ds = utils_cellot.load_dataset(
        drug_name=cfg.data.drug_name,
        drug_setting=cfg.data.drug_setting,
        where=cfg.data.where,
        cellot_path=cellot_path,
    )
    test_batch = {
        "source": full_ds.test.source.adata.to_df().values,
        "target": full_ds.test.target.adata.to_df().values,
    }

    # set input and output dimensions
    input_dim = test_batch["source"].shape[1]
    output_dim = test_batch["target"].shape[1]
    
    # set vector field network
    vector_field_act_fn = getattr(jax.nn, cfg.model.vector_field.act_fn) 
    vector_field_net = MLP_vector_field(
        output_dim=output_dim, 
        latent_embed_dim=cfg.model.vector_field.latent_embed_dim, 
        condition_embed_dim=cfg.model.vector_field.condition_embed_dim,
        t_embed_dim=cfg.model.vector_field.t_embed_dim,
        joint_hidden_dim=cfg.model.vector_field.joint_hidden_dim,
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
        momentum=momentum,
        threshold=cfg.model.ot_solver.threshold,
    )
    
    # set wandb run name and config
    name_run_wandb = (
        f"{cfg.data.drug_name}; "
        f"epsilon={cfg.model.epsilon}"
    )
    config_wandb = OmegaConf.to_container(
        cfg, resolve=True, throw_on_missing=True
    )
    
    # set optimizer
    iterations = cfg.optim.num_iterations
    optimizer = optax.adamw(
        learning_rate=cfg.optim.learning_rate,
        weight_decay=cfg.optim.weight_decay
    )
    
    # set callback kwargs
    callback_fn = (
        None if not cfg.callback.use_callback 
        else callbacks.cellot_callback_fn
    )
    
    with wandb.init(config=config_wandb):
    
        # define solver
        otfm = model.OTFlowMatching(
            
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
            callback_fn=callback_fn,
            callback_kwargs=cfg.callback.kwargs, 
            callback_freq=cfg.callback.callback_freq, 
            log_wandb=cfg.callback.log_wandb,
            test_batch=test_batch,
            name_run_wandb=name_run_wandb
        )
        
        # train model
        otfm(
            x=full_ds.train.source.adata.to_df().values, 
            y=full_ds.train.target.adata.to_df().values, 
            batch_size_source=cfg.optim.batch_size_source,
            batch_size_target=cfg.optim.batch_size_target,
        )
    
if __name__ == '__main__':
    evaluate()