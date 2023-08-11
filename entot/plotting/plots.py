import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import ott
import pandas as pd
import seaborn as sns
from IPython.display import clear_output, display



def plot_1D_unbalanced(
    source, target, target_predicted, eta_predictions, xi_predictions, epsilon, tau_a, tau_b, seed, **kwargs
):
    with jax.default_device(jax.devices("cpu")[0]):
        fig, axes = plt.subplots(2, 5, figsize=(24, 16), dpi=150)

        geom = ott.geometry.pointcloud.PointCloud(source, target_predicted, epsilon=epsilon, scale_cost=1.0)
        out_pred = ott.solvers.linear.sinkhorn.Sinkhorn()(ott.problems.linear.linear_problem.LinearProblem(geom))

        geom = ott.geometry.pointcloud.PointCloud(source, target, epsilon=epsilon, scale_cost=1.0)
        out_true = ott.solvers.linear.sinkhorn.Sinkhorn()(
            ott.problems.linear.linear_problem.LinearProblem(geom, tau_a=tau_a, tau_b=tau_b)
        )

        a_true, b_true = out_true.matrix.sum(axis=1), out_true.matrix.sum(axis=0)

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
        axes[0][0].set_title(r"$\mu$", fontsize=14)
        axes[0][0].set_xlabel("Source")

        # Plotting rescaled source distribution
        sns.kdeplot(
            x=source[:, 0],
            color="darkseagreen",
            fill=True,
            weights=a_true,
            edgecolor="black",
            alpha=0.95,
            ax=axes[0][1],
            bw_adjust=kwargs.get("bw_adjust", 1.0),
        )
        axes[0][1].set_title(r"$\eta(x) \cdot \mu$", fontsize=14)

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
        axes[0][2].set_title(r"$\hat{\eta}_{\theta}(x) \cdot \mu$", fontsize=14)

        # Plotting P_2#pi_hat(X,Y)
        sns.kdeplot(
            y=target_predicted[:, 0],
            color="sandybrown",
            fill=True,
            edgecolor="black",
            alpha=0.95,
            bw_adjust=kwargs.get("bw_adjust", 1.0),
            ax=axes[0][3],
        )
        axes[0][3].set_title(r"$P_2\#\pi_{\theta}$", fontsize=14)
        axes[0][3].set_ylabel("Mapped Source")

        # Plotting \hat{\eta}(x) * P_2#pi_hat(X,Z)
        sns.kdeplot(
            y=target_predicted[:, 0],
            color="sandybrown",
            fill=True,
            weights=eta_predictions[:, 0],
            edgecolor="black",
            alpha=0.95,
            bw_adjust=kwargs.get("bw_adjust", 1.0),
            ax=axes[0][4],
        )
        axes[0][4].set_title(r"$\hat{\eta}_{\theta}(x) \cdot P_2\#\pi_{\theta}$", fontsize=14)

        # Plotting learnt plan pi_hat between learnt rescaled source and P_2#pi_hat(X,Z)
        joint_dist = jnp.concatenate((source, target_predicted), axis=1)
        jd = pd.DataFrame(data=joint_dist, columns=["source", "mapped"])
        sns.kdeplot(
            data=jd,
            x="source",
            y="mapped",
            weights=eta_predictions[:, 0],
            color="black",
            alpha=1.0,
            ax=axes[1][0],
            bw_adjust=kwargs.get("bw_adjust", 1.0),
        )
        axes[1][0].set_title(
            r"$\hat{\pi}_{\theta}(\hat{\eta}_{\theta}\cdot \mu, \hat{\xi}_{\theta}\cdot \nu)$", fontsize=14
        )

        # Plotting ground truth plan between learnt rescaled source and P_2#pi_hat(X,Y)
        pi_star_inds = jax.random.categorical(
            jax.random.PRNGKey(seed), logits=jnp.log(out_pred.matrix.flatten()), shape=(len(source),)
        )
        inds_source = pi_star_inds // len(target)
        inds_target = pi_star_inds % len(target)
        data = jnp.concatenate(
            (
                jnp.atleast_2d(source[inds_source]),
                jnp.atleast_2d(target_predicted[inds_target]),
                jnp.atleast_2d(eta_predictions[inds_source]),
            ),
            axis=1,
        )
        gt = pd.DataFrame(data=data, columns=["source", "mapped_source", "predicted_weights"])
        sns.kdeplot(
            data=gt,
            x="source",
            y="mapped_source",
            weights="predicted_weights",
            color="black",
            alpha=1.0,
            ax=axes[1][1],
            bw_adjust=kwargs.get("bw_adjust", 1.0),
        )
        axes[1][1].set_title(r"$\pi^*(\hat{\eta}_{\theta}\cdot \mu, \hat{\xi}_{\theta}\cdot \nu)$", fontsize=14)

        # Plotting rescaled target distribution
        df = pd.DataFrame(data=np.concatenate((target, b_true[:, None]), axis=1), columns=["target", "weights"])
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
        axes[1][2].set_title(r"$\xi(y) \cdot \nu$", fontsize=14)

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
        axes[1][3].set_title(r"$\hat{\xi}_{\theta}(y) \cdot \nu$", fontsize=14)

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
        axes[1][4].set_title(r"$\nu$", fontsize=14)

        fig.tight_layout(pad=0.01)

        return fig


def plot_1D_balanced(source, target, target_predicted, epsilon, **kwargs):
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
        axes[0].set_title(r"$\mu$", fontsize=14)
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
        axes[1].set_title(r"$P_2\#\hat{\pi}_{\theta}$", fontsize=14)
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
        geom = ott.geometry.pointcloud.PointCloud(source, target_predicted, epsilon=epsilon, scale_cost=1.0)
        out = ott.solvers.linear.sinkhorn.Sinkhorn()(ott.problems.linear.linear_problem.LinearProblem(geom))
        pi_star_inds = jax.random.categorical(
            jax.random.PRNGKey(0), logits=jnp.log(out.matrix.flatten()), shape=(len(source),)
        )
        inds_source = pi_star_inds // len(target)
        inds_target = pi_star_inds % len(target)
        data = jnp.concatenate(
            (jnp.atleast_2d(source[inds_source]), jnp.atleast_2d(target_predicted[inds_target])), axis=1
        )
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
        axes[4].set_title(r"$\nu$", fontsize=14)
        axes[4].set_ylabel("Target")
        fig.tight_layout(pad=0.01)

        return fig
