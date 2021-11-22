from argparse import ArgumentParser
from itertools import repeat, zip_longest
import sys
from typing import Iterable, Optional, Tuple
from botorch.test_functions.base import BaseTestProblem

from matplotlib import pyplot as plt
from matplotlib import patheffects as pe
import numpy as np
from scipy import interpolate, stats
import seaborn as sns
import torch
from torch.functional import Tensor

# sys.path.append("boip")
from boip import objectives

sns.set_theme(style="white", context="talk")


def immediate_regret(Y, optimum: Tensor):
    """calculate the immediate regret for the observed values given the objective

    Parameters
    ----------
    Y : np.ndarray
        an `r x t` array, where r is the number of repeats and t is the number of observations made
        for a given trial. Each entry is the observation made at iteration t
    optimum : Optional[Tensor]
        the optimum of the objective function

    Returns
    -------
    np.ndarray
        an `r x t` array where each row corresponds to a given trial and the each entry is the
        immediate regret (the true maximum minus the observed maximum) at iteration t
    """
    Y_star = np.empty(Y.shape)
    for i in range(Y.shape[1]):
        Y_star[:,i] = Y[:,:i+1].max(1)

    return optimum - Y_star


def interpolate_regret(T, R, budget):
    """interpolate the regret individual regret curves to align them for similar costs

    Parameters
    ----------
    T : np.ndarray
        an `r x t` array, where each entry is the total time spent by iteration t of trial r
    R : np.ndarray
        an `r x t` array, where each entry is the immediate regret at iteration t of trial r
    budget : float
        the total allowable cost

    Returns
    -------
    cost : np.ndarray
        a vector of length N, where each entry c_i is the total cost needed to make
        i+1 objective evaluations and N is equal to `budget / objective_cost`
    R_interp : np.ndarray
        an `r x N` array, where each entry corresponds to the immediate regret
        at a given cost c of trial r
    """
    c = np.linspace(0, budget, 100)
    R_interp = np.empty((len(R), len(c)))

    for i in range(len(T)):
        f = interpolate.interp1d(
            T[i], R[i], kind="previous", fill_value="extrapolate"
        )
        R_interp[i] = f(c)

    return c, R_interp


def plot_IR(
    ax,
    Ys: Iterable[np.ndarray],
    Ts: Iterable[np.ndarray],
    N: int,
    optimum: Tensor,
    labels: Optional[Iterable[str]] = None,
    interpolate: bool = True
):
    """plot the immediate regret curves of the dataset onto axis ax

    Parameters
    ----------
    ax
        the axis on which to plot the curves
    Ys : Iterable[np.ndarray]
        an iterable of `r x t` arrays, where each entry is the observation made at
        iteration t of trial r for a specific dataset
    Ts : Iterable[np.ndarray]
        an iterable of `r x t` arrays, where each entry is the total time spent by iteration t of 
        trial r for a given dataset
    obj : BaseTestProblem
        the objective these datasets are attempting to minimize
    N : int
        the number of initial random observations
    labels : Optional[Iterable[str]]
        the label of each trace
    optimum: Tensor
        the optimal objective value

    Returns
    -------
    ax
        the axis on which the curves were plotted
    """
    for Y, T, label in zip_longest(Ys, Ts, labels):
        R = immediate_regret(Y, optimum)[:, N:]
        c = np.arange(R.shape[1])

        if interpolate:
            c, R = interpolate_regret(T[:, N:], R, 60)

        r = R.mean(0)
        r_se = stats.sem(R, 0)
        ax.plot(
            c,
            r,
            label=label,
            lw=5,
            path_effects=[pe.Stroke(linewidth=7.5, foreground="k"), pe.Normal()],
        )
        ax.fill_between(c, r-r_se, r+r_se, alpha=0.3, dashes=":", lw=2.0, ec="black")

    ax.set_xlabel(r"$\mathrm{CPU}\cdot\mathrm{s}$")

    return ax


def add_random_IR(
    ax,
    obj,
    N: int,
    optimum: Tensor,
    shape: Tuple,
    choices: Optional[Tensor] = None
):
    if choices is None:
        X = torch.distributions.uniform.Uniform(*obj.bounds).sample(shape)
    else:
        idxs = torch.randperm(len(choices))[:np.prod(shape)].reshape(shape)
        X = choices[idxs]

    Y = obj(X)

    R = immediate_regret(Y.numpy(), optimum)[:, N:]
    c = np.arange(R.shape[1])

    r = R.mean(0)
    r_se = stats.sem(R, 0)
    ax.plot(
        c,
        r,
        color="grey",
        label="random",
        lw=5,
        path_effects=[pe.Stroke(linewidth=7.5, foreground="k"), pe.Normal()],
    )
    ax.fill_between(c, r-r_se, r+r_se, alpha=0.3, dashes=":", lw=2.0, ec="black")

    return ax


def plot_surface(
    ax,
    obj: BaseTestProblem,
    Xs: np.ndarray,
    optimal_choice: Tensor,
    title: Optional[str] = None,
    N: int = 0,
):
    """plot the level surface of the given objective onto the specified axis

    Parameters
    ----------
    ax : [type]
        the axis onto which the level surface should be plotted
    obj : BaseTestProblem
        the objective to which the level surface corresponds
    Xs : Iterable[np.ndarray]
        an iterable of `r x t x d` arrays. Where each entry in each array corresponds to the
        d-dimensional point acquired at iteration t of trial r
    title : Optional[str], default=None
        the title to use for the plot. If None, use the name of the objective function
    N : int, default=0
        the number of initial random observations
    optimal_choice: Tensor

    Returns
    -------
    ax
        the axis on which the surface was plotted
    """
    X1_bounds, X2_bounds = zip(*obj.bounds.numpy())
    x1 = np.linspace(*X1_bounds, 100)
    x2 = np.linspace(*X2_bounds, 100)

    X1, X2 = np.meshgrid(x1, x2)
    Z = -obj(torch.from_numpy(np.stack((X1, X2), axis=2))).numpy()

    cs1 = ax.contourf(X1, X2, Z, 8, cmap="coolwarm", zorder=0, alpha=0.7)
    cs2 = ax.contour(X1, X2, Z, cs1.levels, colors='k', alpha=0.5, linestyles='solid')
    # for c in cs2.collections:
    #     c.set_linestyle('solid')
    # ax.clabel(cs2, inline=True, fontsize=10)

    ax.plot(*optimal_choice.numpy(), "y*", ms=12, mec="black", zorder=2)

    ax.set_title(title if title is not None else str(obj)[:-2])
    ax.tick_params(
        labelcolor="none",
        which="both",
        top=False,
        bottom=False,
        left=False,
        right=False,
    )

    # if Xs is not None and strategy is not None:
    #     idx = np.random.randint(len(Xs[strategy.value]))
    #     points = Xs[strategy.value][idx][: N + max_idxs[strategy.value]]
    #     ax.scatter(
    #         points[:, 0],
    #         points[:, 1],
    #         s=10.0,
    #         marker="x",
    #         c=np.arange(len(points)),
    #         cmap="viridis",
    #         vmin=N,
    #         zorder=1,
    #     )

    return ax


def create_figure(
    objs,
    npzdirs,
    titles: Optional[Iterable[str]] = None,
    Ns: Iterable[int] = None,
    choicess: Optional[Iterable[Tensor]] = None,
    interpolate: bool= False
):
    fig, axs = plt.subplots(2, len(objs), figsize=(6 * len(objs), 11))

    titles = titles if titles is not None else repeat(None)
    axs = axs if len(objs) > 1 else [[ax] for ax in axs]

    for i, (obj, npzdir, title, N, choices) in enumerate(
        zip(objs, npzdirs, titles, Ns, choicess)
    ):
        try:
            X_npzfile = np.load(f"{npzdir}/X.npz")
            Xs = [X_npzfile[k] for k in X_npzfile]
        except FileNotFoundError:
            print("Inputs array not found. No points will be overlaid!")
            Xs = None

        Y_npzfile = np.load(f"{npzdir}/Y.npz")
        Ys = [Y_npzfile[k] for k in Y_npzfile]

        T_npzfile = np.load(f"{npzdir}/T.npz")
        Ts = [T_npzfile[k] for k in T_npzfile]

        if choices is not None:
            optimal_idx = obj(choices).argmax()
            optimal_choice = choices[optimal_idx]
            optimum = obj(optimal_choice)
        else:
            optimal_choice = obj.optimizers[0]
            optimum = obj.optimal_value

        plot_surface(axs[0][i], obj, Xs, optimal_choice, npzdir, N)
        ax = plot_IR(axs[1][i], Ys, Ts, N, optimum, Y_npzfile.files, interpolate)
        if not interpolate:
            add_random_IR(ax, obj, N, optimum, Ys[0].shape, choices)

    handles, labels = axs[1][0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        ncol=int(len(labels) / (2 if len(objs) == 1 else 1)),
        bbox_to_anchor=(0.34, 0.0),
        loc="lower center",
    )
    ax = fig.add_subplot(111, frameon=False)
    ax.tick_params(
        labelcolor="none",
        which="both",
        top=False,
        bottom=False,
        left=False,
        right=False,
    )
    # fig.text(
    #     0.945,
    #     0.03,
    #     f"N: {N} | objective cost: {objective_cost:0.1f} | retraining cost: {retraining_cost:0.2f}",
    #     horizontalalignment="right",
    #     verticalalignment="center",
    #     bbox=dict(edgecolor="gray", fc="white", boxstyle="Square, pad=0.3"),
    # )
    axs[1][0].set_ylabel("Immediate Regret")
    fig.tight_layout()
    fig.subplots_adjust(wspace=0.175, hspace=0.175)

    axs[1][0].get_shared_y_axes().join(*axs[1])
    axs[1][0].get_shared_x_axes().join(*axs[1])
    axs[1][-1].autoscale()
    return fig


def main():
    parser = ArgumentParser()
    parser.add_argument("-o", "--objectives", nargs="+")
    parser.add_argument("-i", "--npzdirs", "--input-dirs", nargs="+")
    parser.add_argument("--titles", nargs="+")
    parser.add_argument("-N", type=int, nargs="+", help="the number of initialization points")
    parser.add_argument("-s", "--strategy")
    parser.add_argument(
        "-c",
        "--num-choices",
        type=int,
        default=1000,
        nargs="+",
        help="the number of points with which to discretize the objective function",
    )
    parser.add_argument(
        "-ds",
        "--discretization-seed",
        type=int,
        nargs="+",
        help="the random seed to use for discrete landscapes",
    )
    parser.add_argument("--no-interpolate", action="store_true")
    parser.add_argument("--output", help="the name under which to save the figure")

    args = parser.parse_args()

    objs = [objectives.build_objective(o) for o in args.objectives]
    if args.num_choices is not None:
        num_choices = args.num_choices if len(args.num_choices) > 1 else repeat(args.num_choices[0])
        seeds = (
            args.discretization_seed
            if len(args.discretization_seed) > 1
            else repeat(args.discretization_seed[0])
        )
        choicess = [
            objectives.discretize(obj, N, seed)
            for obj, N, seed in zip(objs, num_choices, seeds)
        ]
    else:
        choicess = None

    # strategy = Strategy[args.strategy.upper()] if args.strategy is not None else None
    fig = create_figure(
        objs,
        args.npzdirs,
        args.titles,
        args.N,
        choicess,
        not args.no_interpolate
    )
    fig.savefig(args.output, dpi=200, bbox_inches="tight")
    print(f"Figure saved to {args.output}")


if __name__ == "__main__":
    main()
