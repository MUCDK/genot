import jax.numpy as jnp
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import jax
from IPython.display import clear_output, display
from entot.models.utils import _concatenate


def plot_1D(source, target, T_xz, sinkhorn_output):
    clear_output(wait=True)
    #T_xz_mean = jnp.expand_dims(jnp.mean(target, axis=-1), axis=1)
    fig, axes = plt.subplots(1, 5, figsize=(12, 3), dpi=150)
    
   
    
    axes[0].set_axisbelow(True); axes[0].grid(axis='x')
    axes[2].set_axisbelow(True); axes[2].grid(axis='y')
    axes[4].set_axisbelow(True); axes[4].grid(axis='y')
    axes[0].set_xlim(-2.5, 2.5); axes[0].set_ylim(0, 1.0)
    axes[2].set_ylim(-2.5, 2.5); axes[2].set_xlim(0, 0.8)
    axes[1].set_xlim(-2.5, 2.5); axes[1].set_ylim(-2.5, 2.5)
    axes[3].set_xlim(-2.5, 2.5); axes[3].set_ylim(-2.5, 2.5)
    axes[4].set_ylim(-2.5, 2.5); axes[4].set_xlim(0, 0.8)
    
    # Plotting source distribution
    sns.kdeplot(
        source[:, 0], color='darkseagreen', fill=True,
        edgecolor='black', alpha=0.95,
        ax=axes[0], label=r'$x\sim\mathbb{P}$'
    )  
    axes[0].legend(fontsize=12, loc='upper left', framealpha=1)
    axes[0].set_title(r"Input $\mathbb{P}$ (1D)", fontsize=14)


    # Plotting P_2#pi_hat(X,Z)
    sns.kdeplot(
        y=T_xz[:,0], color='sandybrown', fill=True,
        edgecolor='black', alpha=0.95,
        ax=axes[2], label=r'$T(x,z)\sim T_{\sharp}(\mathbb{P}\times\mathbb{S})$'
    )
    axes[2].legend(fontsize=12, loc='upper left', framealpha=1)   
    axes[2].set_title(r"Mapped $T_{\sharp}(\mathbb{P}\times\mathbb{S})$ (1D)", fontsize=14)

    
    # Plotting learnt plan pi_hat between source and P_2#pi_hat(X,Z)
    joint_dist= jnp.concatenate((source, T_xz), axis=1)
    jd = pd.DataFrame(data =joint_dist, columns=["source", "mapped"])
    sns.kdeplot(
        data = jd , x="source", y="mapped", # xlim=(-2.5, 2.5), ylim=(-2.5, 2.5), # make T_xz 2d
        color='black', alpha=1., ax=axes[1],
        label=r'$(x,\hat{T}(x,z))\sim \hat{\pi}$'
    )
    axes[1].set_title(r"Learned $\hat{\pi}$ (2D), ours", fontsize=14)

    # Plotting ground truth plan between source and P_2#pi_hat(X,Z)
    pi_star_inds = jax.random.categorical(jax.random.PRNGKey(0), logits=jnp.log(sinkhorn_output.matrix.flatten()), shape=(len(source),))
    inds_source = pi_star_inds // len(target)
    inds_target = pi_star_inds % len(target)
    data = _concatenate(source[inds_source], T_xz[inds_target])
    gt = pd.DataFrame(data = data, columns=["source", "target"])
    sns.kdeplot(
        data = gt , x="source", y="target", # xlim=(-2.5, 2.5), ylim=(-2.5, 2.5), # make T_xz 2d
        color='black', alpha=1., ax=axes[3],
        label=r'$(x,\hat{T}(x,z))\sim \pi^*$'
    )
    axes[3].set_title(r"$\pi^*$ (2D)", fontsize=14)


    # Plotting target distribution
    sns.kdeplot(
        y=target[:, 0], color='wheat', fill=True,
        edgecolor='black', alpha=0.95,
        ax=axes[4], label=r'$y\sim\mathbb{Q}$'
    )
    axes[4].legend(fontsize=12, loc='upper left', framealpha=1)   
    axes[4].set_title(r"Target $\mathbb{Q}$ (1D)", fontsize=14)


    fig.tight_layout(pad=0.01)
    fig.show()
    display(fig)
    
    return fig


def plot_images(source, target, T_xz, _, n_samples: int = 2, k_noise_per_sample: int = 2):
    clear_output(wait=True)
    fig, axes = plt.subplots(n_samples, k_noise_per_sample+2, figsize=(12, 3), dpi=150)
    
    for i in range(n_samples):
        for j in range(k_noise_per_sample+2):
            if j==0:
                axes[i][j].imshow(source[i,...])
                axes[i][j].set_title("Source image")
            elif j==1:
                axes[i][j].imshow(target[i,...])
                axes[i][j].set_title("Target image")
            else:
                axes[i][j].imshow(T_xz[i,...])
                axes[i][j].set_title(f"Generated image {j-2}")

    fig.tight_layout(pad=0.01)
    fig.show()
    display(fig)
    
    return fig