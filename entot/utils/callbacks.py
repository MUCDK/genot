# jax
import jax
import jax.numpy as jnp

# numpy
import numpy as np

# pandas
import pandas as pd

# joblin
from joblib import Parallel, delayed

# types, typing
from types import MappingProxyType
from typing import Any, Callable, Dict, Iterable, Mapping, Optional, Tuple

# ott
from ott.tools import sinkhorn_divergence
from ott.geometry import pointcloud

# own code
from entot.models import model
from entot.utils import metrics

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

        
