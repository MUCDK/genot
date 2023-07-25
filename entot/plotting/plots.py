import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import ott
import pandas as pd
import seaborn as sns
from IPython.display import clear_output, display

from entot.models.utils import _concatenate


def plot_1D(source, target, t_xz, sinkhorn_output, **kwargs):
    clear_output(wait=True)
    #
    fig, axes = plt.subplots(1, 5, figsize=(12, 3), dpi=150)

    axes[0].set_axisbelow(True)
    axes[0].grid(axis="x")
    axes[3].set_axisbelow(True)
    axes[3].grid(axis="y")
    axes[4].set_axisbelow(True)
    axes[4].grid(axis="y")
    axes[0].set_xlim(kwargs.pop("0_xlim", (-2.5, 2.5)))
    axes[0].set_ylim(kwargs.pop("0_ylim", (0, 1.0)))
    axes[1].set_xlim(kwargs.pop("1_xlim", (-2.5, 2.5)))
    axes[1].set_ylim(kwargs.pop("1_ylim", (-2.5, 2.5)))
    axes[2].set_xlim(kwargs.pop("2_xlim", (-2.5, 2.5)))
    axes[2].set_ylim(kwargs.pop("2_ylim", (-2.5, 2.5)))
    axes[3].set_xlim(kwargs.pop("3_xlim", (0, 1.0)))
    axes[3].set_ylim(kwargs.pop("3_ylim", (-2.5, 2.5)))
    axes[4].set_xlim(kwargs.pop("4_xlim", (0, 1.0)))
    axes[4].set_ylim(kwargs.pop("4_ylim", (-2.5, 2.5)))

    # Plotting source distribution
    sns.kdeplot(
        source[:, 0],
        color="darkseagreen",
        fill=True,
        edgecolor="black",
        alpha=0.95,
        ax=axes[0],
        label=r"$x\sim\mathbb{P}$",
        bw_adjust=kwargs.get("bw_adjust", 1.0),
    )
    axes[0].legend(fontsize=12, loc="upper left", framealpha=1)
    axes[0].set_title(r"Input $\mathbb{P}$", fontsize=14)

    # Plotting learnt plan pi_hat between source and P_2#pi_hat(X,Z)
    joint_dist = jnp.concatenate((source, t_xz), axis=1)
    jd = pd.DataFrame(data=joint_dist, columns=["source", "mapped"])
    sns.kdeplot(
        data=jd,
        x="source",
        y="mapped",  # xlim=(-2.5, 2.5), ylim=(-2.5, 2.5), # make T_xz 2d
        color="black",
        alpha=1.0,
        ax=axes[1],
        label=r"$(x,\hat{T}(x,z))\sim \hat{\pi}$",
        bw_adjust=kwargs.get("bw_adjust", 1.0),
        fill=True,
        levels=100,
        cmap="mako",
    )
    axes[1].set_title(r"Learned $\hat{\pi}$", fontsize=14)

    # Plotting ground truth plan between source and P_2#pi_hat(X,Z)
    pi_star_inds = jax.random.categorical(
        jax.random.PRNGKey(0), logits=jnp.log(sinkhorn_output.matrix.flatten()), shape=(len(source),)
    )
    inds_source = pi_star_inds // len(target)
    inds_target = pi_star_inds % len(target)
    data = _concatenate(source[inds_source], t_xz[inds_target])
    gt = pd.DataFrame(data=data, columns=["source", "target"])
    sns.kdeplot(
        data=gt,
        x="source",
        y="target",  # xlim=(-2.5, 2.5), ylim=(-2.5, 2.5), # make T_xz 2d
        color="black",
        alpha=1.0,
        ax=axes[2],
        label=r"$(x,\hat{T}(x,z))\sim \pi^*$",
        bw_adjust=kwargs.get("bw_adjust", 1.0),
        fill=True,
        levels=100,
        cmap="mako",
    )
    axes[2].set_title(r"$\pi^*$", fontsize=14)

    # Plotting P_2#pi_hat(X,Z)
    sns.kdeplot(
        y=t_xz[:, 0],
        color="sandybrown",
        fill=True,
        edgecolor="black",
        alpha=0.95,
        bw_adjust=kwargs.get("bw_adjust", 1.0),
        ax=axes[3],
        label=r"$T(x,z)\sim T_{\sharp}(\mathbb{P}\times\mathbb{S})$",
    )
    axes[3].legend(fontsize=12, loc="upper left", framealpha=1)
    axes[3].set_title(r"Mapped $T_{\sharp}(\mathbb{P}\times\mathbb{S})$", fontsize=14)

    # Plotting target distribution
    sns.kdeplot(
        y=target[:, 0],
        color="wheat",
        fill=True,
        edgecolor="black",
        alpha=0.95,
        ax=axes[4],
        label=r"$y\sim\mathbb{Q}$",
        bw_adjust=kwargs.get("bw_adjust", 1.0),
    )
    axes[4].legend(fontsize=12, loc="upper left", framealpha=1)
    axes[4].set_title(r"Target $\mathbb{Q}$", fontsize=14)

    fig.tight_layout(pad=0.01)
    fig.show()
    display(fig)

    return fig


def plot_images(source, target, t_xz, sinkhorn_output, n_samples: int = 2, k_noise_per_sample: int = 2, **kwargs):
    clear_output(wait=True)
    fig, axes = plt.subplots(n_samples, k_noise_per_sample + 2, figsize=(12, 3), dpi=150)
    pi_star_inds = jax.random.categorical(
        jax.random.PRNGKey(1), logits=jnp.log(sinkhorn_output.matrix.flatten()), shape=(n_samples,)
    )
    inds_source = pi_star_inds // len(target)
    inds_target = pi_star_inds % len(target)
    source = source[inds_source, ...]
    target = target[inds_target, ...]
    t_xz = t_xz[inds_target, ...]

    for i in range(n_samples):
        for j in range(k_noise_per_sample + 2):
            if j == 0:
                axes[i][j].imshow(source[i, ...])
                axes[i][j].set_title("Source image")
            elif j == 1:
                axes[i][j].imshow(target[i, ...])
                axes[i][j].set_title("Target image")
            else:
                axes[i][j].imshow(t_xz[i, ...])
                axes[i][j].set_title(f"Generated image {j-2}")

    fig.tight_layout(pad=0.01)
    fig.show()
    display(fig)

    return fig


def plot_1D_unbalanced(
    source, target, t_xz, sinkhorn_output, sinkhorn_output_unbalanced, eta_predictions, xi_predictions, **kwargs
):
    clear_output(wait=True)
    fig, axes = plt.subplots(2, 5, figsize=(24, 16), dpi=150)
    eta_predictions = eta_predictions + 1 if np.sum(eta_predictions) == 0 else eta_predictions
    xi_predictions = xi_predictions + 1 if np.sum(eta_predictions) == 0 else xi_predictions

    a, b = sinkhorn_output_unbalanced.matrix.sum(axis=1), sinkhorn_output_unbalanced.matrix.sum(axis=0)

    # axes[0][0].set_axisbelow(True); axes[0][0].grid(axis='x')
    # axes[0][2].set_axisbelow(True); axes[0][2].grid(axis='y')
    # axes[0][4].set_axisbelow(True); axes[0][4].grid(axis='y')
    axes[0][0].set_xlim(kwargs.pop("00_xlim", (-2.5, 2.5)))
    axes[0][0].set_ylim(kwargs.pop("00_ylim", (0, 2.0)))
    axes[0][1].set_xlim(kwargs.pop("01_xlim", (-2.5, 2.5)))
    axes[0][1].set_ylim(kwargs.pop("01_ylim", (0, 2.0)))
    axes[0][2].set_xlim(kwargs.pop("02_xlim", (-2.5, 2.5)))
    axes[0][2].set_ylim(kwargs.pop("02_ylim", (0, 2.0)))
    axes[0][3].set_xlim(kwargs.pop("03_xlim", (0, 2.0)))
    axes[0][3].set_ylim(kwargs.pop("03_ylim", (-2.5, 2.5)))
    axes[0][4].set_xlim(kwargs.pop("04_xlim", (0, 2.0)))
    axes[0][4].set_ylim(kwargs.pop("04_ylim", (-2.5, 2.5)))
    axes[1][0].set_xlim(kwargs.pop("10_xlim", (-2.5, 2.5)))
    axes[1][0].set_ylim(kwargs.pop("10_ylim", (-2.5, 2.5)))
    axes[1][1].set_xlim(kwargs.pop("11_xlim", (-2.5, 2.5)))
    axes[1][1].set_ylim(kwargs.pop("11_ylim", (-2.5, 2.5)))
    axes[1][2].set_xlim(kwargs.pop("12_xlim", (0, 2.0)))
    axes[1][2].set_ylim(kwargs.pop("12_ylim", (-2.5, 2.5)))
    axes[1][3].set_xlim(kwargs.pop("13_xlim", (0, 2.0)))
    axes[1][3].set_ylim(kwargs.pop("13_ylim", (-2.5, 2.5)))
    axes[1][4].set_xlim(kwargs.pop("14_xlim", (0, 2.0)))
    axes[1][4].set_ylim(kwargs.pop("14_ylim", (-2.5, 2.5)))

    # Plotting source distribution
    sns.kdeplot(
        source[:, 0],
        color="darkseagreen",
        fill=True,
        edgecolor="black",
        alpha=0.95,
        ax=axes[0][0],
        bw_adjust=kwargs.get("bw_adjust", 1.0),
    )
    axes[0][0].set_title(r"Input $\mathbb{P}$", fontsize=14)

    # Plotting rescaled source distribution
    sns.kdeplot(
        x=source[:, 0],
        color="darkseagreen",
        fill=True,
        weights=a,
        edgecolor="black",
        alpha=0.95,
        ax=axes[0][1],
        bw_adjust=kwargs.get("bw_adjust", 1.0),
    )
    axes[0][1].set_title(r"$\eta(x) \cdot \mathbb{P}$", fontsize=14)

    # Plotting learnt rescaled source distribution
    sns.kdeplot(
        x=source[:, 0],
        color="darkseagreen",
        fill=True,
        weights=eta_predictions[:, 0],
        edgecolor="black",
        alpha=0.95,
        ax=axes[0][2],
        bw_adjust=kwargs.get("bw_adjust", 1.0),
    )
    axes[0][2].set_title(r"$\hat{\eta}_{\theta}(x) \cdot \mathbb{P}$", fontsize=14)

    # Plotting P_2#pi_hat(X,Z)
    sns.kdeplot(
        y=t_xz[:, 0],
        color="sandybrown",
        fill=True,
        edgecolor="black",
        alpha=0.95,
        bw_adjust=kwargs.get("bw_adjust", 1.0),
        ax=axes[0][3],
    )
    axes[0][3].set_title(r"$T_{\theta}\#(\mathbb{P}\times\mathbb{S})$", fontsize=14)

    # Plotting \eta(x) * P_2#pi_hat(X,Z)
    sns.kdeplot(
        y=t_xz[:, 0],
        color="sandybrown",
        fill=True,
        weights=a,
        edgecolor="black",
        alpha=0.95,
        bw_adjust=kwargs.get("bw_adjust", 1.0),
        ax=axes[0][4],
    )
    axes[0][4].set_title(r"$\hat{\eta}_{\theta}(x) \cdot T_{\theta}\#(\mathbb{P}\times\mathbb{S})$", fontsize=14)

    # Plotting learnt plan pi_hat between learnt rescaled source and P_2#pi_hat(X,Z)
    joint_dist = jnp.concatenate((source, t_xz), axis=1)
    jd = pd.DataFrame(data=joint_dist, columns=["source", "mapped"])
    sns.kdeplot(
        data=jd,
        x="source",
        y="mapped",
        weights=a,
        color="black",
        alpha=1.0,
        ax=axes[1][0],
        bw_adjust=kwargs.get("bw_adjust", 1.0),
    )
    axes[1][0].set_title(r"$\hat{\pi}_{\theta}$", fontsize=14)

    # Plotting ground truth plan between learnt rescaled source and P_2#pi_hat(X,Z)
    geom = ott.geometry.pointcloud.PointCloud(source, t_xz, epsilon=kwargs["epsilon"])
    out = ott.solvers.linear.sinkhorn.Sinkhorn()(
        ott.problems.linear.linear_problem.LinearProblem(geom, a=a, b=a, tau_a=kwargs["tau_a"], tau_b=kwargs["tau_b"])
    )
    pi_star_inds = jax.random.categorical(
        jax.random.PRNGKey(0), logits=jnp.log(out.matrix.flatten()), shape=(len(source),)
    )
    inds_source = pi_star_inds // len(target)
    inds_target = pi_star_inds % len(target)
    data = _concatenate(source[inds_source], t_xz[inds_target])
    gt = pd.DataFrame(data=data, columns=["source", "mapped_source"])
    sns.kdeplot(
        data=gt,
        x="source",
        y="mapped_source",
        weights=a,
        color="black",
        alpha=1.0,
        ax=axes[1][1],
        bw_adjust=kwargs.get("bw_adjust", 1.0),
    )
    axes[1][1].set_title(r"$\pi^*$", fontsize=14)

    # Plotting rescaled target distribution
    df = pd.DataFrame(data=np.concatenate((target, b[:, None]), axis=1), columns=["target", "weights"])
    sns.kdeplot(
        data=df,
        y="target",
        color="wheat",
        fill=True,
        weights="weights",
        edgecolor="black",
        alpha=0.95,
        ax=axes[1][2],
        bw_adjust=kwargs.get("bw_adjust", 1.0),
    )
    axes[1][2].set_title(r"$\xi(y) \cdot \mathbb{Q}$", fontsize=14)

    # Plotting rescaled learnt target distribution
    sns.kdeplot(
        y=target[:, 0],
        color="wheat",
        fill=True,
        weights=xi_predictions[:, 0],
        edgecolor="black",
        alpha=0.95,
        ax=axes[1][3],
        bw_adjust=kwargs.get("bw_adjust", 1.0),
    )
    axes[1][3].set_title(r"$\hat{\xi}_{\theta}(y) \cdot \mathbb{Q}$", fontsize=14)

    # Plotting target distribution
    sns.kdeplot(
        y=target[:, 0],
        color="wheat",
        fill=True,
        edgecolor="black",
        alpha=0.95,
        ax=axes[1][4],
        bw_adjust=kwargs.get("bw_adjust", 1.0),
    )
    axes[1][4].set_title(r"$\mathbb{Q}$", fontsize=14)

    fig.tight_layout(pad=0.01)
    fig.show()
    display(fig)

    return fig


def scatter_plot_2d(source, target, t_xz, sinkhorn_output, **_):
    clear_output(wait=True)
    fig, axes = plt.subplots(1, 4, figsize=(10, 3), dpi=150)

    axes[0].set_xlim((-1.5, 1.5))
    axes[0].set_ylim((-1, 10.0))

    axes[1].set_xlim((-1.5, 1.5))
    axes[1].set_ylim((-1, 10.0))

    axes[2].set_xlim((-1.5, 1.5))
    axes[2].set_ylim((-1, 10.0))

    axes[3].set_xlim((-1.5, 1.5))
    axes[3].set_ylim((-1, 10.0))

    axes[0].scatter(target[:, 0], target[:, 1])
    axes[0].scatter(t_xz[:, 0], t_xz[:, 1])

    axes[1].scatter(source[:, 0], source[:, 1])
    axes[1].scatter(t_xz[:, 0], t_xz[:, 1])

    pi_star_inds = jax.random.categorical(
        jax.random.PRNGKey(0), logits=jnp.log(sinkhorn_output.matrix.flatten()), shape=(1000,)
    )
    inds_source = pi_star_inds // 300
    inds_target = pi_star_inds % 300
    source_gt = source[inds_source]
    target_gt = t_xz[inds_target]
    indices = jax.random.randint(jax.random.PRNGKey(0), (60,), 0, len(source_gt))
    axes[2].plot([source_gt[indices, 0], target_gt[indices, 0]], [source_gt[indices, 1], target_gt[indices, 1]], "ro-")

    indices = jax.random.randint(jax.random.PRNGKey(0), (60,), 0, len(source_gt))
    axes[3].plot([source[indices, 0], t_xz[indices, 0]], [source[indices, 1], t_xz[indices, 1]], "ro-")

    axes[3].scatter(source[:, 0], source[:, 1])
    axes[3].scatter(t_xz[:, 0], t_xz[:, 1])

    fig.tight_layout(pad=0.01)
    fig.show()
    display(fig)

    return fig


def plot_1D_unbalanced_new(
    source, target, target_predicted, eta_predictions, xi_predictions, epsilon, tau_a, tau_b, **kwargs
):
    with jax.default_device(jax.devices("cpu")[0]):
        fig, axes = plt.subplots(2, 5, figsize=(24, 16), dpi=150)

        geom = ott.geometry.pointcloud.PointCloud(source, target_predicted, epsilon=epsilon, scale_cost="mean")
        out = ott.solvers.linear.sinkhorn.Sinkhorn()(
            ott.problems.linear.linear_problem.LinearProblem(geom, tau_a=tau_a, tau_b=tau_b)
        )
        
        a, b = out.matrix.sum(axis=1), out.matrix.sum(axis=0)

        axes[0][0].set_xlim(kwargs.pop("00_xlim", (-2.5, 2.5)))
        axes[0][0].set_ylim(kwargs.pop("00_ylim", (0, 2.0)))
        axes[0][1].set_xlim(kwargs.pop("01_xlim", (-2.5, 2.5)))
        axes[0][1].set_ylim(kwargs.pop("01_ylim", (0, 2.0)))
        axes[0][2].set_xlim(kwargs.pop("02_xlim", (-2.5, 2.5)))
        axes[0][2].set_ylim(kwargs.pop("02_ylim", (0, 2.0)))
        axes[0][3].set_xlim(kwargs.pop("03_xlim", (0, 2.0)))
        axes[0][3].set_ylim(kwargs.pop("03_ylim", (-2.5, 2.5)))
        axes[0][4].set_xlim(kwargs.pop("04_xlim", (0, 2.0)))
        axes[0][4].set_ylim(kwargs.pop("04_ylim", (-2.5, 2.5)))
        axes[1][0].set_xlim(kwargs.pop("10_xlim", (-2, 3)))
        axes[1][0].set_ylim(kwargs.pop("10_ylim", (-2, 3)))
        axes[1][1].set_xlim(kwargs.pop("11_xlim", (-2.5, 2.5)))
        axes[1][1].set_ylim(kwargs.pop("11_ylim", (-2.5, 2.5)))
        axes[1][2].set_xlim(kwargs.pop("12_xlim", (0, 2.0)))
        axes[1][2].set_ylim(kwargs.pop("12_ylim", (-2.5, 2.5)))
        axes[1][3].set_xlim(kwargs.pop("13_xlim", (0, 2.0)))
        axes[1][3].set_ylim(kwargs.pop("13_ylim", (-2.5, 2.5)))
        axes[1][4].set_xlim(kwargs.pop("14_xlim", (0, 2.0)))
        axes[1][4].set_ylim(kwargs.pop("14_ylim", (-2.5, 2.5)))

        # Plotting source distribution
        sns.kdeplot(
            source[:, 0],
            color="darkseagreen",
            fill=True,
            edgecolor="black",
            alpha=0.95,
            ax=axes[0][0],
            bw_adjust=kwargs.get("bw_adjust", 1.0),
        )
        axes[0][0].set_title(r"$\mathbb{P}$", fontsize=14)

        # Plotting rescaled source distribution
        sns.kdeplot(
            x=source[:, 0],
            color="darkseagreen",
            fill=True,
            weights=a,
            edgecolor="black",
            alpha=0.95,
            ax=axes[0][1],
            bw_adjust=kwargs.get("bw_adjust", 1.0),
        )
        axes[0][1].set_title(r"$\eta(x) \cdot \mathbb{P}$", fontsize=14)

        # Plotting learnt rescaled source distribution
        sns.kdeplot(
            x=source[:, 0],
            color="darkseagreen",
            fill=True,
            weights=eta_predictions[:, 0],
            edgecolor="black",
            alpha=0.95,
            ax=axes[0][2],
            bw_adjust=kwargs.get("bw_adjust", 1.0),
        )
        axes[0][2].set_title(r"$\hat{\eta}_{\theta}(x) \cdot \mathbb{P}$", fontsize=14)

        # Plotting P_2#pi_hat(X,Z)
        sns.kdeplot(
            y=target_predicted[:, 0],
            color="sandybrown",
            fill=True,
            edgecolor="black",
            alpha=0.95,
            bw_adjust=kwargs.get("bw_adjust", 1.0),
            ax=axes[0][3],
        )
        axes[0][3].set_title(r"$T_{\theta}\#\mathbb{P}$", fontsize=14)

        # Plotting \eta(x) * P_2#pi_hat(X,Z)
        sns.kdeplot(
            y=target_predicted[:, 0],
            color="sandybrown",
            fill=True,
            weights=eta_predictions[:,0],
            edgecolor="black",
            alpha=0.95,
            bw_adjust=kwargs.get("bw_adjust", 1.0),
            ax=axes[0][4],
        )
        axes[0][4].set_title(r"$\hat{\eta}_{\theta}(x) \cdot T_{\theta}\#\mathbb{P}$", fontsize=14)

        # Plotting learnt plan pi_hat between learnt rescaled source and P_2#pi_hat(X,Z)
        joint_dist = jnp.concatenate((source, target_predicted), axis=1)
        jd = pd.DataFrame(data=joint_dist, columns=["source", "mapped"])
        sns.kdeplot(
            data=jd,
            x="source",
            y="mapped",
            weights=a,
            color="black",
            alpha=1.0,
            ax=axes[1][0],
            bw_adjust=kwargs.get("bw_adjust", 1.0),
        )
        axes[1][0].set_title(r"$\hat{\pi}_{\theta}$", fontsize=14)

        # Plotting ground truth plan between learnt rescaled source and P_2#pi_hat(X,Z)
        pi_star_inds = jax.random.categorical(
            jax.random.PRNGKey(0), logits=jnp.log(out.matrix.flatten()), shape=(len(source),)
        )
        inds_source = pi_star_inds // len(target)
        inds_target = pi_star_inds % len(target)
        data = _concatenate(source[inds_source], target_predicted[inds_target])
        gt = pd.DataFrame(data=data, columns=["source", "mapped_source"])
        sns.kdeplot(
            data=gt,
            x="source",
            y="mapped_source",
            weights=a,
            color="black",
            alpha=1.0,
            ax=axes[1][1],
            bw_adjust=kwargs.get("bw_adjust", 1.0),
        )
        axes[1][1].set_title(r"$\pi^*$", fontsize=14)

        # Plotting rescaled target distribution
        df = pd.DataFrame(data=np.concatenate((target, b[:, None]), axis=1), columns=["target", "weights"])
        sns.kdeplot(
            data=df,
            y="target",
            color="wheat",
            fill=True,
            weights="weights",
            edgecolor="black",
            alpha=0.95,
            ax=axes[1][2],
            bw_adjust=kwargs.get("bw_adjust", 1.0),
        )
        axes[1][2].set_title(r"$\xi(y) \cdot \mathbb{Q}$", fontsize=14)

        # Plotting rescaled learnt target distribution
        sns.kdeplot(
            y=target[:, 0],
            color="wheat",
            fill=True,
            weights=xi_predictions[:, 0],
            edgecolor="black",
            alpha=0.95,
            ax=axes[1][3],
            bw_adjust=kwargs.get("bw_adjust", 1.0),
        )
        axes[1][3].set_title(r"$\hat{\xi}_{\theta}(y) \cdot \mathbb{Q}$", fontsize=14)

        # Plotting target distribution
        sns.kdeplot(
            y=target[:, 0],
            color="wheat",
            fill=True,
            edgecolor="black",
            alpha=0.95,
            ax=axes[1][4],
            bw_adjust=kwargs.get("bw_adjust", 1.0),
        )
        axes[1][4].set_title(r"$\mathbb{Q}$", fontsize=14)

        fig.tight_layout(pad=0.01)

        return fig



def plot_1D_unbalanced(
    source, target, t_xz, sinkhorn_output, sinkhorn_output_unbalanced, eta_predictions, xi_predictions, **kwargs
):
    clear_output(wait=True)
    fig, axes = plt.subplots(2, 5, figsize=(24, 16), dpi=150)
    eta_predictions = eta_predictions + 1 if np.sum(eta_predictions) == 0 else eta_predictions
    xi_predictions = xi_predictions + 1 if np.sum(eta_predictions) == 0 else xi_predictions

    a, b = sinkhorn_output_unbalanced.matrix.sum(axis=1), sinkhorn_output_unbalanced.matrix.sum(axis=0)

    # axes[0][0].set_axisbelow(True); axes[0][0].grid(axis='x')
    # axes[0][2].set_axisbelow(True); axes[0][2].grid(axis='y')
    # axes[0][4].set_axisbelow(True); axes[0][4].grid(axis='y')
    axes[0][0].set_xlim(kwargs.pop("00_xlim", (-2.5, 2.5)))
    axes[0][0].set_ylim(kwargs.pop("00_ylim", (0, 2.0)))
    axes[0][1].set_xlim(kwargs.pop("01_xlim", (-2.5, 2.5)))
    axes[0][1].set_ylim(kwargs.pop("01_ylim", (0, 2.0)))
    axes[0][2].set_xlim(kwargs.pop("02_xlim", (-2.5, 2.5)))
    axes[0][2].set_ylim(kwargs.pop("02_ylim", (0, 2.0)))
    axes[0][3].set_xlim(kwargs.pop("03_xlim", (0, 2.0)))
    axes[0][3].set_ylim(kwargs.pop("03_ylim", (-2.5, 2.5)))
    axes[0][4].set_xlim(kwargs.pop("04_xlim", (0, 2.0)))
    axes[0][4].set_ylim(kwargs.pop("04_ylim", (-2.5, 2.5)))
    axes[1][0].set_xlim(kwargs.pop("10_xlim", (-2.5, 2.5)))
    axes[1][0].set_ylim(kwargs.pop("10_ylim", (-2.5, 2.5)))
    axes[1][1].set_xlim(kwargs.pop("11_xlim", (-2.5, 2.5)))
    axes[1][1].set_ylim(kwargs.pop("11_ylim", (-2.5, 2.5)))
    axes[1][2].set_xlim(kwargs.pop("12_xlim", (0, 2.0)))
    axes[1][2].set_ylim(kwargs.pop("12_ylim", (-2.5, 2.5)))
    axes[1][3].set_xlim(kwargs.pop("13_xlim", (0, 2.0)))
    axes[1][3].set_ylim(kwargs.pop("13_ylim", (-2.5, 2.5)))
    axes[1][4].set_xlim(kwargs.pop("14_xlim", (0, 2.0)))
    axes[1][4].set_ylim(kwargs.pop("14_ylim", (-2.5, 2.5)))

    # Plotting source distribution
    sns.kdeplot(
        source[:, 0],
        color="darkseagreen",
        fill=True,
        edgecolor="black",
        alpha=0.95,
        ax=axes[0][0],
        bw_adjust=kwargs.get("bw_adjust", 1.0),
    )
    axes[0][0].set_title(r"Input $\mathbb{P}$", fontsize=14)

    # Plotting rescaled source distribution
    sns.kdeplot(
        x=source[:, 0],
        color="darkseagreen",
        fill=True,
        weights=a,
        edgecolor="black",
        alpha=0.95,
        ax=axes[0][1],
        bw_adjust=kwargs.get("bw_adjust", 1.0),
    )
    axes[0][1].set_title(r"$\eta(x) \cdot \mathbb{P}$", fontsize=14)

    # Plotting learnt rescaled source distribution
    sns.kdeplot(
        x=source[:, 0],
        color="darkseagreen",
        fill=True,
        weights=eta_predictions[:, 0],
        edgecolor="black",
        alpha=0.95,
        ax=axes[0][2],
        bw_adjust=kwargs.get("bw_adjust", 1.0),
    )
    axes[0][2].set_title(r"$\hat{\eta}_{\theta}(x) \cdot \mathbb{P}$", fontsize=14)

    # Plotting P_2#pi_hat(X,Z)
    sns.kdeplot(
        y=t_xz[:, 0],
        color="sandybrown",
        fill=True,
        edgecolor="black",
        alpha=0.95,
        bw_adjust=kwargs.get("bw_adjust", 1.0),
        ax=axes[0][3],
    )
    axes[0][3].set_title(r"$T_{\theta}\#(\mathbb{P}\times\mathbb{S})$", fontsize=14)

    # Plotting \eta(x) * P_2#pi_hat(X,Z)
    sns.kdeplot(
        y=t_xz[:, 0],
        color="sandybrown",
        fill=True,
        weights=a,
        edgecolor="black",
        alpha=0.95,
        bw_adjust=kwargs.get("bw_adjust", 1.0),
        ax=axes[0][4],
    )
    axes[0][4].set_title(r"$\hat{\eta}_{\theta}(x) \cdot T_{\theta}\#(\mathbb{P}\times\mathbb{S})$", fontsize=14)

    # Plotting learnt plan pi_hat between learnt rescaled source and P_2#pi_hat(X,Z)
    joint_dist = jnp.concatenate((source, t_xz), axis=1)
    jd = pd.DataFrame(data=joint_dist, columns=["source", "mapped"])
    sns.kdeplot(
        data=jd,
        x="source",
        y="mapped",
        weights=a,
        color="black",
        alpha=1.0,
        ax=axes[1][0],
        bw_adjust=kwargs.get("bw_adjust", 1.0),
    )
    axes[1][0].set_title(r"$\hat{\pi}_{\theta}$", fontsize=14)

    # Plotting ground truth plan between learnt rescaled source and P_2#pi_hat(X,Z)
    geom = ott.geometry.pointcloud.PointCloud(source, t_xz, epsilon=kwargs["epsilon"])
    out = ott.solvers.linear.sinkhorn.Sinkhorn()(
        ott.problems.linear.linear_problem.LinearProblem(geom, a=a, b=a, tau_a=kwargs["tau_a"], tau_b=kwargs["tau_b"])
    )
    pi_star_inds = jax.random.categorical(
        jax.random.PRNGKey(0), logits=jnp.log(out.matrix.flatten()), shape=(len(source),)
    )
    inds_source = pi_star_inds // len(target)
    inds_target = pi_star_inds % len(target)
    data = _concatenate(source[inds_source], t_xz[inds_target])
    gt = pd.DataFrame(data=data, columns=["source", "mapped_source"])
    sns.kdeplot(
        data=gt,
        x="source",
        y="mapped_source",
        weights=a,
        color="black",
        alpha=1.0,
        ax=axes[1][1],
        bw_adjust=kwargs.get("bw_adjust", 1.0),
    )
    axes[1][1].set_title(r"$\pi^*$", fontsize=14)

    # Plotting rescaled target distribution
    df = pd.DataFrame(data=np.concatenate((target, b[:, None]), axis=1), columns=["target", "weights"])
    sns.kdeplot(
        data=df,
        y="target",
        color="wheat",
        fill=True,
        weights="weights",
        edgecolor="black",
        alpha=0.95,
        ax=axes[1][2],
        bw_adjust=kwargs.get("bw_adjust", 1.0),
    )
    axes[1][2].set_title(r"$\xi(y) \cdot \mathbb{Q}$", fontsize=14)

    # Plotting rescaled learnt target distribution
    sns.kdeplot(
        y=target[:, 0],
        color="wheat",
        fill=True,
        weights=xi_predictions[:, 0],
        edgecolor="black",
        alpha=0.95,
        ax=axes[1][3],
        bw_adjust=kwargs.get("bw_adjust", 1.0),
    )
    axes[1][3].set_title(r"$\hat{\xi}_{\theta}(y) \cdot \mathbb{Q}$", fontsize=14)

    # Plotting target distribution
    sns.kdeplot(
        y=target[:, 0],
        color="wheat",
        fill=True,
        edgecolor="black",
        alpha=0.95,
        ax=axes[1][4],
        bw_adjust=kwargs.get("bw_adjust", 1.0),
    )
    axes[1][4].set_title(r"$\mathbb{Q}$", fontsize=14)

    fig.tight_layout(pad=0.01)
    fig.show()
    display(fig)

    return fig


def scatter_plot_2d(source, target, t_xz, sinkhorn_output, **_):
    clear_output(wait=True)
    fig, axes = plt.subplots(1, 4, figsize=(10, 3), dpi=150)

    axes[0].set_xlim((-1.5, 1.5))
    axes[0].set_ylim((-1, 10.0))

    axes[1].set_xlim((-1.5, 1.5))
    axes[1].set_ylim((-1, 10.0))

    axes[2].set_xlim((-1.5, 1.5))
    axes[2].set_ylim((-1, 10.0))

    axes[3].set_xlim((-1.5, 1.5))
    axes[3].set_ylim((-1, 10.0))

    axes[0].scatter(target[:, 0], target[:, 1])
    axes[0].scatter(t_xz[:, 0], t_xz[:, 1])

    axes[1].scatter(source[:, 0], source[:, 1])
    axes[1].scatter(t_xz[:, 0], t_xz[:, 1])

    pi_star_inds = jax.random.categorical(
        jax.random.PRNGKey(0), logits=jnp.log(sinkhorn_output.matrix.flatten()), shape=(1000,)
    )
    inds_source = pi_star_inds // 300
    inds_target = pi_star_inds % 300
    source_gt = source[inds_source]
    target_gt = t_xz[inds_target]
    indices = jax.random.randint(jax.random.PRNGKey(0), (60,), 0, len(source_gt))
    axes[2].plot([source_gt[indices, 0], target_gt[indices, 0]], [source_gt[indices, 1], target_gt[indices, 1]], "ro-")

    indices = jax.random.randint(jax.random.PRNGKey(0), (60,), 0, len(source_gt))
    axes[3].plot([source[indices, 0], t_xz[indices, 0]], [source[indices, 1], t_xz[indices, 1]], "ro-")

    axes[3].scatter(source[:, 0], source[:, 1])
    axes[3].scatter(t_xz[:, 0], t_xz[:, 1])

    fig.tight_layout(pad=0.01)
    fig.show()
    display(fig)

    return fig


def plot_1D_balanced_new(
    source, target, target_predicted, epsilon, **kwargs
):
    with jax.default_device(jax.devices("cpu")[0]):
        fig, axes = plt.subplots(1, 5, figsize=(12, 4), dpi=200)

        axes[0].set_xlim(kwargs.pop("00_xlim", (-2.5, 2.5)))
        axes[0].set_ylim(kwargs.pop("00_ylim", (0, 2.0)))
        axes[1].set_xlim(kwargs.pop("01_xlim", (-2.5, 2.5)))
        axes[1].set_ylim(kwargs.pop("01_ylim", (0, 2.0)))
        axes[2].set_xlim(kwargs.pop("02_xlim", (-2.5, 2.5)))
        axes[2].set_ylim(kwargs.pop("02_ylim", (0, 2.0)))
        axes[3].set_xlim(kwargs.pop("10_xlim", (-2, 3)))
        axes[3].set_ylim(kwargs.pop("10_ylim", (-2, 3)))
        axes[4].set_xlim(kwargs.pop("11_xlim", (-2.5, 2.5)))
        axes[4].set_ylim(kwargs.pop("11_ylim", (-2.5, 2.5)))
        
        # Plotting source distribution
        sns.kdeplot(
            source[:, 0],
            color="darkseagreen",
            fill=True,
            edgecolor="black",
            alpha=0.95,
            ax=axes[0],
            bw_adjust=kwargs.get("bw_adjust", 1.0),
        )
        axes[0].set_title(r"Input $\mathbb{P}$", fontsize=14)
        axes[0].set_xlabel("Source")

        # Plotting P_2#pi_hat(X,Z)
        sns.kdeplot(
            y=target_predicted[:, 0],
            color="sandybrown",
            fill=True,
            edgecolor="black",
            alpha=0.95,
            bw_adjust=kwargs.get("bw_adjust", 1.0),
            ax=axes[1],
        )
        axes[1].set_title(r"$T_{\theta}\#\mathbb{P}$", fontsize=14)
        axes[1].set_ylabel("Mapped Source")

        # Plotting learnt plan pi_hat between source and P_2#pi_hat(X,Z)
        joint_dist = jnp.concatenate((source, target_predicted), axis=1)
        jd = pd.DataFrame(data=joint_dist, columns=["source", "mapped"])
        sns.kdeplot(
            data=jd,
            x="source",
            y="mapped",
            color="black",
            alpha=1.0,
            ax=axes[2],
            bw_adjust=kwargs.get("bw_adjust", 1.0),
        )
        axes[2].set_title(r"$\hat{\pi}_{\theta}$", fontsize=14)
        axes[2].set_xlabel("Source")
        axes[2].set_ylabel("Mapped Source")

        # Plotting ground truth plan between learnt rescaled source and P_2#pi_hat(X,Z)
        geom = ott.geometry.pointcloud.PointCloud(source, target_predicted, epsilon=epsilon)
        out = ott.solvers.linear.sinkhorn.Sinkhorn()(
            ott.problems.linear.linear_problem.LinearProblem(geom)
        )
        pi_star_inds = jax.random.categorical(
            jax.random.PRNGKey(0), logits=jnp.log(out.matrix.flatten()), shape=(len(source),)
        )
        inds_source = pi_star_inds // len(target)
        inds_target = pi_star_inds % len(target)
        data = _concatenate(source[inds_source], target_predicted[inds_target])
        gt = pd.DataFrame(data=data, columns=["source", "mapped_source"])
        sns.kdeplot(
            data=gt,
            x="source",
            y="mapped_source",
            color="black",
            alpha=1.0,
            ax=axes[3],
            bw_adjust=kwargs.get("bw_adjust", 1.0),
        )
        axes[3].set_title(r"$\pi^*$", fontsize=14)
        axes[3].set_xlabel("Source")
        axes[3].set_ylabel("Mapped Source")

        # Plotting target distribution
        sns.kdeplot(
            y=target[:, 0],
            color="wheat",
            fill=True,
            edgecolor="black",
            alpha=0.95,
            ax=axes[4],
            bw_adjust=kwargs.get("bw_adjust", 1.0),
        )
        axes[4].set_title(r"$\mathbb{Q}$", fontsize=14)
        axes[4].set_ylabel("Target")
        fig.tight_layout(pad=0.01)

        return fig