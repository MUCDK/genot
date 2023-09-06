# %%

%load_ext autoreload
%autoreload 2

import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import sys
sys.path.append('..')
# %%

import seaborn as sns
import jax
import ott
import diffrax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from entot.models.utils import MixtureNormalSampler
from entot.models.model import OTFlowMatching, MLP_FM_VAE2, Bridge_MLP
from entot.plotting.plots import plot_1D_balanced_new

# %%

x0 = np.linspace(-1, 0, 300)
y0 = np.cos(x0) * 5 / + 0.1 * np.random.randn(len(x0))
data0 = np.concatenate((x0[:,None], y0[:,None]), axis=1)

x1 = np.linspace(0, 1, 300)
y1 = np.cos(x1) * 5 + 0.1 * np.random.randn(len(x1))
data1 = np.concatenate((x1[:,None], y1[:,None]), axis=1)

x0 = np.linspace(-1, 0, 300)
x1 = np.linspace(0, 1, 300)

y0 = 0.3 * (x0**2)
y1 = 0.3 * (x1**2)

source = np.concatenate((x0[:,None], y0[:,None]), axis=1)
target = np.concatenate((x1[:,None], y1[:,None]), axis=1)

plt.scatter(source[:,0], source[:,1])
plt.scatter(target[:,0], target[:,1])

# %%
