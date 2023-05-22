import jax.numpy as jnp
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from IPython.display import clear_output


def plot_1D(source, target, T_xz):
    clear_output(wait=True)
    print("Plotting")
    T_xz_mean = jnp.expand_dims(jnp.mean(target, axis=-1), axis=1)
    fig, axes = plt.subplots(1, 4, figsize=(12, 3), dpi=150)

    axes[2].set_xlim(-2.5, 2.5)
    axes[2].set_ylim(-2.5, 2.5)

    axes[0].set_axisbelow(True)
    axes[0].grid(axis="x")
    axes[1].set_axisbelow(True)
    axes[1].grid(axis="y")
    axes[3].set_axisbelow(True)
    axes[3].grid(axis="y")
    axes[0].set_xlim(-2.5, 2.5)
    axes[0].set_ylim(0, 0.7)
    axes[1].set_ylim(-2.5, 2.5)
    axes[1].set_xlim(0, 0.7)
    axes[3].set_ylim(-2.5, 2.5)
    axes[3].set_xlim(0, 0.7)

    # Plotting X
    sns.kdeplot(
        source[:, 0],
        color="darkseagreen",
        fill=True,
        edgecolor="black",
        alpha=0.95,
        ax=axes[0],
        label=r"$x\sim\mathbb{P}$",
    )
    axes[0].legend(fontsize=12, loc="upper left", framealpha=1)
    axes[0].set_title(r"Input $\mathbb{P}$ (1D)", fontsize=14)

    # Plotting Y
    sns.kdeplot(
        y=target[:, 0], color="wheat", fill=True, edgecolor="black", alpha=0.95, ax=axes[1], label=r"$y\sim\mathbb{Q}$"
    )
    axes[1].legend(fontsize=12, loc="upper left", framealpha=1)
    axes[1].set_title(r"Target $\mathbb{Q}$ (1D)", fontsize=14)

    T_xz_reshaped = jnp.reshape(T_xz, (T_xz.shape[0] * T_xz.shape[2], T_xz.shape[1]))
    source_reshaped = jnp.repeat(source, T_xz.shape[2], axis=0)
    joint_dist = jnp.concatenate((source_reshaped, T_xz_reshaped), axis=1)

    joint_dist.sort(axis=0)
    jd = pd.DataFrame(data=joint_dist, columns=["source", "mapped"])
    sns.kdeplot(
        data=jd,
        x="source",
        y="mapped",  # xlim=(-2.5, 2.5), ylim=(-2.5, 2.5), # make T_xz 2d
        color="black",
        alpha=1.0,
        ax=axes[2],
        label=r"$(x,\hat{T}(x,z))\sim \hat{\pi}$",
    )
    axes[2].set_title(r"Learned $\hat{\pi}$ (2D), ours", fontsize=14)

    axes[2].plot(source[:, 0], T_xz_mean[:, 0], color="sandybrown", linewidth=3, label=r"$x\mapsto \overline{T}(x)$")
    axes[2].legend(fontsize=12, loc="upper left", framealpha=1)

    # Plotting T(X,Z)
    sns.kdeplot(
        y=T_xz_reshaped[:, 0],
        color="sandybrown",
        fill=True,
        edgecolor="black",
        alpha=0.95,
        ax=axes[3],
        label=r"$T(x,z)\sim T_{\sharp}(\mathbb{P}\times\mathbb{S})$",
    )
    axes[3].legend(fontsize=12, loc="upper left", framealpha=1)
    axes[3].set_title(r"Mapped $T_{\sharp}(\mathbb{P}\times\mathbb{S})$ (1D)", fontsize=14)

    fig.tight_layout(pad=0.01)
    fig.show()

    return None
