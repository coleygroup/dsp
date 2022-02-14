import argparse
from pathlib import Path
import string
from typing import Optional, Tuple

from botorch.test_functions.base import BaseTestProblem
from matplotlib import pyplot as plt, patheffects as pe
from matplotlib.collections import PathCollection
from matplotlib.ticker import MultipleLocator
import numpy as np
from scipy import stats
import seaborn as sns
import torch
from torch import Tensor

import boip

sns.set_theme("talk", "white")
sns.set_palette("dark")

def plot_surface_discrete(
    ax,
    choices: np.ndarray,
    y: np.ndarray,
    zmin: float,
    zmax: float,
    hits: np.ndarray,
    title: Optional[str] = None,
) -> PathCollection:
    """plot the discretized objective

    Parameters
    ----------
    ax
        the axis onto which the level surface should be plotted
    choices : np.ndarray
        an iterable of `n x d` array corresponding to the design space
    y : np.ndarray
        a vector parallel to choices that contains the y-value for corresponding to each point 
        point in the design space 
    hits: np.ndarray
    title : Optional[str], default=None
        the title to use for the plot

    Returns
    -------
    sc : PathCollection
        the path collection created by the scatter plot
    """
    sc = ax.scatter(
        choices[:, 0], choices[:, 1], c=y, vmin=zmin, vmax=zmax,
        s=10, cmap="coolwarm", alpha=0.7, zorder=0
    )
    ax.plot(hits[:, 0], hits[:, 1], "y*", ms=12, mec="black", zorder=2)

    ax.tick_params(
        labelcolor="none",
        which="both",
        top=False,
        bottom=False,
        left=False,
        right=False,
    )

    buffer = 0.02
    ax.text(
        buffer, 1-buffer, title,
        color="w", ha="left", va="top", transform=ax.transAxes,
    ).set_path_effects([pe.withStroke(linewidth=5, foreground="k")])
    
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)

    return sc
    
def prune(choices: np.ndarray, H: np.ndarray, i: int) -> np.ndarray:
    mask = np.ones(len(choices), bool)
    mask[(0 <= H) & (H <= i)] = False

    return mask

def plot_acquired_points(ax, X, Y, zmin, zmax):
    return ax.scatter(
        X[:,0], X[:,1], s=20, c=Y, vmin=zmin, vmax=zmax,
        cmap="coolwarm", edgecolor='k', linewidth=2, zorder=1, alpha=0.7
    )

def immediate_regret(Y: np.ndarray, optima: np.ndarray) -> np.ndarray:
    """calculate the immediate top-k regret for the observed values given the objective top-k
    optima

    Parameters
    ----------
    Y : np.ndarray
        an `r x t` array, where r is the number of repeats and t is the number of observations made
        for a given trial. Each entry is the observation made at iteration t
    optima : np.ndarray
        a length `k` vector containing the optima of the objective function

    Returns
    -------
    np.ndarray
        an `r x t` array containing the immediate regret at the given iteration, where r is number of repititions and t is number of iterations. "Immediate regret" is formulated as the
        fraction of the total number of optima found. NOTE: it is possible for there to be
        degenerate hits
    """
    k = len(optima)
    Y = np.nan_to_num(Y, nan=-np.inf)
    t = optima.min()
    Y_star = np.empty(Y.shape)
    
    for i in range(k, Y.shape[1]):
        Y_star[:,i] = (np.partition(Y[:,:i+1], -k, 1)[:, -k:] >= t).sum(1)
        
    return Y_star / len(optima)


def add_regret_trace(ax, R: np.ndarray, x, label: str):
    r = R.mean(0)
    r_se = stats.sem(R, 0)
    
    handles = ax.plot(
        x,
        r,
        label=label,
        lw=5,
        path_effects=[pe.Stroke(linewidth=7.5, foreground="k"), pe.Normal()],
    )
    c = handles[0].get_color()

    ax.fill_between(x, r-r_se, r+r_se, alpha=0.5, lw=2.0, color=c, dashes=":", ec="black")

    return handles

def plot_IR(
    ax,
    Y_full: np.ndarray,
    Y_prune: np.ndarray,
    N: int,
    optima: np.ndarray,
    obj,
    choices,
):
    """plot the immediate regret curves of the dataset onto axis ax

    Parameters
    ----------
    ax
        the axis on which to plot the curves
    Ys : Iterable[np.ndarray]
        an iterable of `r x t` arrays, where each entry is the observation made at
        iteration t of trial r for a specific dataset
    obj : BaseTestProblem
        the objective these datasets are attempting to minimize
    N : int
        the number of initial random observations
    labels : Optional[Iterable[str]]
        the label of each trace
    optima : np.ndarray
        the optimal objective values

    Returns
    -------
    List
        the artists corresponding to the plotted curves
    """   
    handles = [] 
    x = np.arange(Y_prune.shape[1])[N:]

    R = immediate_regret(Y_full, optima)[:, N : N + len(x)]
    handles.append(add_regret_trace(ax, R, x, "no pruning"))

    R = immediate_regret(Y_prune, optima)[:, N:]
    handles.append(add_regret_trace(ax, R, x, "pruning"))

    R = gen_R_random(obj, *Y_prune.shape, optima, choices)[:, N:]
    r = R.mean(0)
    r_se = stats.sem(R, 0)
    
    handles.append(ax.plot(
        x,
        r,
        c="grey",
        label="random",
        lw=5,
        path_effects=[pe.Stroke(linewidth=7.5, foreground="k"), pe.Normal()],
    ))
    ax.fill_between(x, r-r_se, r+r_se, color="grey", alpha=0.7, dashes=":", lw=2.0, ec="black")

    return [h[0] for h in handles]

def gen_R_random(
    obj,
    r: int,
    n: int,
    optima: Tensor,
    choices: Tensor,
):
    """add a trace for random acquisition

    Parameters
    ----------
    obj
        the objective to generate the trace for
    r : int
        the number of repeats
    n : int
        the number of samples to take

    Returns
    -------
    np.ndarray
        the immediate regret matrix for random acquisition
    """
    idxs = torch.empty((r, n)).long()
    for i in range(r):
        idxs[i] = torch.randperm(len(choices))[:n]

    X = choices[idxs]
    Y = obj(X).numpy()

    return immediate_regret(Y, optima)

def plot_surface(
    ax,
    obj: BaseTestProblem,
    optimal_choices: np.ndarray,
    title: Optional[str] = None,
):
    """plot the level surface of the given objective onto the specified axis

    Parameters
    ----------
    ax
        the axis onto which the level surface should be plotted
    obj : BaseTestProblem
        the objective to which the level surface corresponds
    Xs : Iterable[np.ndarray]
        an iterable of `r x t x d` arrays. Where each entry in each array corresponds to the
        d-dimensional point acquired at iteration t of trial r
    optimal_choices: np.ndarray
    title : Optional[str], default=None
        the title to use for the plot. If None, use the name of the objective function

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
    ax.contour(X1, X2, Z, cs1.levels, colors='k', alpha=0.5, linestyles='solid')

    ax.plot(optimal_choices[:, 0], optimal_choices[:, 1], "y*", ms=12, mec="black", zorder=2)

    ax.set_title(title if title is not None else str(obj)[:-2], size="large")
    ax.tick_params(
        labelcolor="none",
        which="both",
        top=False,
        bottom=False,
        left=False,
        right=False,
    )

    return ax

def plot_size_H(ax, H, q: int, T: int):
    """plot the input space size using the pruning history

    Parameters
    ----------
    ax
        the axis on which to plot
    H : np.ndarray
        an `r x n x 2` array containing the acquisition and pruning history of each point, where r 
        is the number of repeats and n is the total input space size. The 0th value of each entry is the iteration at
        which the given point was pruned from the input space, with a -1 indicating that the
        point was never pruned
    q : int
        the batch/initialization size
    T : int
        the number of optimzation iterations

    Returns
    -------
    handle
        the matplotlib artist corresponding to the plotted curve
    """
    N = H.shape[1]
    N_prune_t = np.empty((len(H), T+1), int)
    for t in range(N_prune_t.shape[1]):
        N_prune_t[:, t] = (H == t).sum(2).sum(1)

    S = N - np.cumsum(N_prune_t, 1)
    x = q * (1+np.arange(S.shape[1]))
    
    S = S/S[0, 0]
    s = S.mean(0)
    s_se = stats.sem(S, 0)

    handle = ax.plot(
        x,
        s,
        'g--',
        lw=2.5,
        path_effects=[pe.Stroke(linewidth=5, foreground="k"), pe.Normal()],
        label="size",
    )[0]
    ax.fill_between(x, s-s_se, s+s_se, color='g', alpha=0.3, lw=2.0)
    
    return handle

def f_hits_pruned(H, hit_idxs, T):
    hits_pruned_t = np.zeros((len(H), T+1))
    for t in range(hits_pruned_t.shape[1]):
        hits_pruned_t[:, t] = (H[:, hit_idxs] == t).sum(1)

    return np.cumsum(hits_pruned_t, 1) / len(hit_idxs)
    
def plot_IR_all(
    ax,
    x: np.ndarray,
    Y: np.ndarray,
    N: int,
    optima: np.ndarray,
    color,
    label: str
):
    """plot the immediate regret curves of the dataset onto axis ax

    Parameters
    ----------
    ax
        the axis on which to plot the curves
    x : np.ndarray
        a vector of the x-value for each regret trace
    Y : np.ndarray
        an iterable of `r x t` array, where each entry is the observation made at
        iteration t of trial r for a specific dataset
    N : int
        the number of initial random observations
    optima : np.ndarray
        the optimal objective values
    label : str
        the label of the traces

    Returns
    -------
    List
        the artists corresponding to the plotted curves
    """   
    handles = [] 

    R = immediate_regret(Y, optima)[:, N : N + len(x)]
    r = np.median(R, 0)
    handles = ax.plot(
        x,
        r,
        color=color,
        lw=5,
        path_effects=[pe.Stroke(linewidth=7.5, foreground="k"), pe.Normal()],
        label=label,
        zorder=10
    )
    c = handles[0].get_color()

    for r in R:
        ax.plot(x, r, color=c, lw=3, alpha=0.2)

    return handles[0]

def michalewicz(npzdir: Path, gamma_dir: Path, outfile: Path):
    fig = plt.figure(figsize=(5*4, 3*4), constrained_layout=False)

    gs = fig.add_gridspec(2, 1, height_ratios=[1, 1.65], hspace=0.225)

    gs0 = gs[0].subgridspec(1, 5, wspace=0.04,)
    gs1 = gs[1].subgridspec(1, 2, wspace=0.3,)

    axsTop = []
    axsBot = []

    #----------------------------------------- SETUP -----------------------------------------#

    N = 10000
    ds = 42

    k = 10
    n = 10
    q = 10

    obj = boip.objectives.build_objective("michalewicz")
    choices = boip.objectives.discretize(obj, N, ds)

    y_all = -obj(choices).numpy()
    optima_idxs = np.argpartition(y_all, k)[:k]
    optima_xs = choices[optima_idxs].numpy()
    optima_ys = y_all[optima_idxs]

    #---------------------------------------- PANEL A ----------------------------------------#

    xmin, ymin = choices.min(0)[0].numpy()
    xmax, ymax = choices.max(0)[0].numpy()

    zmin, zmax = y_all.min(), y_all.max()

    REP = 0
    X = np.load(f"{npzdir}/X.npz")["PRUNE"][REP]
    Y = -np.load(f"{npzdir}/Y.npz")["PRUNE"][REP]
    H = np.load(f"{npzdir}/H.npz")["PRUNE"][REP]

    its = [0, 1, 5, 10, 20]
    for i, it in zip(range(5), its):
        ax = fig.add_subplot(gs0[i])
        
        mask = prune(choices, H[:,1], it)

        if it == 0:
            sc = plot_surface_discrete(ax, choices[mask], y_all[mask], zmin, zmax, optima_xs, title=it)
        else:
            plot_surface_discrete(ax, choices[mask], y_all[mask], zmin, zmax, optima_xs, title=it)
            plot_acquired_points(ax, X[:q*4*(it-1) + n], Y[:q*4*(it-1) + n], zmin, zmax)
            
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)

        axsTop.append(ax)

    cbar = fig.colorbar(
        sc, ax=axsTop, location="bottom", aspect=125, fraction=0.05, pad=0.05,
        ticks=[y_all[optima_idxs].max()],
    )
    cbar.set_label(label=r"objective ($\leftarrow$)", labelpad=-15)
    cbar.ax.set_xticklabels(
        [r"$\bigstar$"], color="y"
    )[0].set_path_effects([pe.withStroke(linewidth=5, foreground="k")])
    cbar.ax.tick_params("x", direction="inout", pad=6, width=2.5, grid_alpha=0.5)

    axsTop[0].set_title("A", weight="bold", loc="left", fontsize="large")

    #---------------------------------------- PANEL B ----------------------------------------#

    ax = fig.add_subplot(gs1[0])

    Y_npz = np.load(f"{npzdir}/Y.npz")
    Y_full = Y_npz["FULL"]
    Y_prune = Y_npz["PRUNE"]
    H = np.load(f"{npzdir}/H.npz")["PRUNE"]

    handles = plot_IR(ax, Y_full, Y_prune, n, -optima_ys, obj, choices)

    ax_twin = ax.twinx()
    handles.append(plot_size_H(ax_twin, H, q, Y_prune.shape[1] // q))

    ax.xaxis.set_major_locator(MultipleLocator(100))
    ax.xaxis.set_minor_locator(MultipleLocator(50))
    ax.tick_params(axis="x", which="both", direction="out", bottom=True)

    ax.tick_params(axis="both", labelsize=16)
    ax_twin.tick_params(axis="y", colors='green', labelsize=16)
    ax.set_xlabel("Objective evaluations", fontsize=16)
    ax.set_ylabel("Fraction of Top-10 Identified", fontsize=16)
    ax_twin.set_ylabel("Relative Input Space Size", fontsize=16)

    ax_twin.set_ylim(ax.get_ylim())
    ax.grid(True, axis="y", ls="--")

    ax.legend(
        handles=handles, bbox_transform=ax.transAxes, bbox_to_anchor=[0.33, 1], loc="upper center"
    )

    axsBot.append(ax)

    axsBot[0].set_title("B", weight="bold", loc="left", fontsize="large")

    #---------------------------------------- PANEL C ----------------------------------------#

    ax = fig.add_subplot(gs1[1])

    T = 40
    t = np.arange(T+1)

    npzdirs = sorted(gamma_dir.iterdir(), key=lambda d: float(d.name))
    palette = sns.color_palette("magma", len(npzdirs))

    for npzdir, c in zip(npzdirs, palette):
        H = np.load(npzdir / "H.npz")["PRUNE"]

        gamma = float(npzdir.name)
        label=rf"${gamma}\,\hat\sigma^2$"
        
        F_nr = f_hits_pruned(H[:,:,1], optima_idxs, T)
        F_nr_mean = F_nr.mean(0)
        F_nr_sem = stats.sem(F_nr, 0)

        ax.plot(
            t, F_nr_mean, color=c, lw=3,
            path_effects=[pe.Stroke(linewidth=5, foreground="k"), pe.Normal()], label=label
        )
        ax.fill_between(
            t, F_nr_mean - F_nr_sem, F_nr_mean + F_nr_sem,
            color=c, dashes=":", lw=2, ec="black", alpha=0.3
        )

    ax.legend()

    ax.xaxis.set_minor_locator(MultipleLocator(1))
    ax.xaxis.set_major_locator(MultipleLocator(5))
    ax.yaxis.set_major_locator(MultipleLocator(0.1))
    ax.tick_params(axis="x", which="both", direction="out", bottom=True)
    ax.grid(axis="y", ls="--")

    ax.set_ylabel("Fraction of Top-10 Pruned", fontsize=16)
    ax.set_xlabel("Iteration", fontsize=16)
    ax.tick_params(axis="both", labelsize=16)

    axsBot.append(ax)

    axsBot[1].set_title("C", weight="bold", loc="left", fontsize="large")

    #------------------------------------------------------------------------------------#

    fig.savefig(outfile, dpi=400, bbox_inches="tight")

def surface_regret(npzdirs, outfile):
    objs = [boip.objectives.build_objective(p.stem) for p in npzdirs]

    n = 10
    k = 10
    q = 10

    sns.set_palette(sns.color_palette("dark"))
    fig, axs = plt.subplots(2, len(objs), figsize=(6 * len(objs), 11))

    axs_2 = []
    for i, (obj, npzdir, label) in enumerate(zip(objs, npzdirs, string.ascii_uppercase)):
        choices = boip.objectives.discretize(obj, 10000, 42)

        # X_npz = np.load(f"{npzdir}/X.npz")
        Y_npz = np.load(f"{npzdir}/Y.npz")
        Y_full = Y_npz["FULL"]
        Y_prune = Y_npz["PRUNE"]
        H = np.load(f"{npzdir}/H.npz")["PRUNE"]

        y_all = obj(choices).numpy()
        optimal_idxs = np.argpartition(y_all, -k)[-k:]
        optimal_choices = choices[optimal_idxs]
        optima = y_all[optimal_idxs]

        plot_surface(axs[0][i], obj, optimal_choices)

        ax1 = axs[1][i]
        handles = plot_IR(ax1, Y_full, Y_prune, n, optima, obj, choices)

        ax2 = ax1.twinx()
        handles.append(plot_size_H(ax2, H, q, Y_prune.shape[1] // q))
        axs_2.append(ax2)

        ax1.set_ylim(-0.05, 1.05)
        ax2.set_ylim(ax1.get_ylim())
        ax1.grid(True, axis="y", ls="--")

        ax1.xaxis.set_major_locator(MultipleLocator(500))
        ax1.xaxis.set_minor_locator(MultipleLocator(100))
        ax1.tick_params(axis="x", which="both", direction="out", bottom=True)

        axs[0][i].set_title(label, loc="left", weight="bold", fontsize="large")

    for ax1 in axs[1][1:]:
        ax1.sharey(axs[1][0])
        ax1.tick_params('y', labelleft=False)

    for ax2 in axs_2[:-1]:
        ax2.sharey(axs_2[-1])
        ax2.tick_params('y', labelright=False, colors='green')

    axs_2[-1].tick_params('y', colors='green')

    fig.legend(
        handles=handles,
        ncol=4,
        bbox_to_anchor=(0.24, 0),
        loc="lower center",
    )

    fig.supxlabel("Objective Evaluations", fontsize=18)
    axs[1][0].set_ylabel(r"Fraction of Top-$10$ Identified")
    axs_2[-1].set_ylabel("Relative Input Space Size")

    fig.tight_layout()
    fig.subplots_adjust(wspace=0.1, hspace=0.07)

    fig.savefig(outfile, dpi=400, bbox_inches="tight")

def regret(npzdir, objective, outfile):
    fig, ax = plt.subplots(1, 1, figsize=(7, 6))

    n = 10
    k = 10
    q = 10

    obj = boip.objectives.build_objective(objective)
    choices = boip.objectives.discretize(obj, 10000, 42)

    Y_npz = np.load(f"{npzdir}/Y.npz")
    Y_full = Y_npz["FULL"]
    Y_prune = Y_npz["PRUNE"]
    H = np.load(f"{npzdir}/H.npz")["PRUNE"]

    y_all = obj(choices).numpy()
    optimal_idxs = np.argpartition(y_all, -k)[-k:]
    optima = y_all[optimal_idxs]

    handles = plot_IR(ax, Y_full, Y_prune, n, optima, obj, choices)

    ax_twin = ax.twinx()
    handles.append(plot_size_H(ax_twin, H, q, Y_prune.shape[1] // q))

    ax_twin.set_ylim(ax.get_ylim())
    ax.grid(True, axis="y", ls="--")

    ax.xaxis.set_major_locator(MultipleLocator(500))
    ax.xaxis.set_minor_locator(MultipleLocator(100))
    ax.tick_params(axis="x", which="both", direction="out", bottom=True)

    ax_twin.tick_params('y', colors='green')

    ax.legend(handles=handles, loc="best")

    ax.set_xlabel("Objective Evaluations")
    ax.set_ylabel(r"Fraction of Top-$10$ Identified")
    ax_twin.set_ylabel("Relative Input Space Size")

    fig.tight_layout()

    fig.savefig(outfile, dpi=200, bbox_inches="tight")

def gamma_perf(gamma_dir, objective, outfile):
    fig, axs = plt.subplots(1, 3, figsize=(6 * 3, 6), sharey=True)

    N = 10000
    ds = 42
    k = 10
    n = 10
    q = 10

    npzdirs = sorted(gamma_dir.iterdir(), key=lambda d: float(d.name))

    obj = boip.build_objective(objective)
    choices = boip.objectives.discretize(obj, N, ds) 

    y_all = obj(choices).numpy()
    optimal_idxs = np.argpartition(y_all, -k)[-k:]
    optima = y_all[optimal_idxs]

    axs_2 = []
    for ax, npzdir in zip(axs, npzdirs):
        gamma = float(npzdir.name)
        title=rf"${gamma}\,\hat\sigma^2$"
        ax.set_title(title)

        Y_npz = np.load(f"{npzdir}/Y.npz")
        H = np.load(f"{npzdir}/H.npz")["PRUNE"]

        y_all = obj(choices).numpy()
        optimal_idxs = np.argpartition(y_all, -k)[-k:]
        optima = y_all[optimal_idxs]

        handles = plot_IR(ax, Y_npz["FULL"], Y_npz["PRUNE"], n, optima, obj, choices)

        ax2 = ax.twinx()
        handles.append(plot_size_H(ax2, H, q, H.max()+1))
        axs_2.append(ax2)

        ax.xaxis.set_major_locator(MultipleLocator(100))
        ax.tick_params(axis="x", which="both", direction="out", bottom=True)
        ax.grid(True, axis="y", ls="--")
        ax2.set_ylim(ax.get_ylim())

    for ax2 in axs_2[:-1]:
        ax2.sharey(axs_2[-1])
        ax2.tick_params('y', labelright=False, colors='green')
    axs_2[-1].tick_params('y', colors='green')

    fig.legend(
        handles=handles,
        ncol=2,
        bbox_to_anchor=(0.075, -0.05),
        loc="lower left",
    )

    axs[0].set_ylabel(r"Fraction of Top-$10$ Identified")
    axs_2[-1].set_ylabel("Relative Input Space Size")
    fig.supxlabel("Objective Evaluations", fontsize=18)

    fig.tight_layout()

    fig.savefig(outfile, dpi=200, bbox_inches="tight")

def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(
        title="figure", description="the figure you would like to make", dest="figure"
    )

    mich_parser = subparsers.add_parser("michalewicz", help="michalewicz multi-panel figure")
    mich_parser.add_argument("npzdir", type=Path)
    mich_parser.add_argument("gamma_dir", type=Path)
    mich_parser.add_argument("-o", "--outfile", type=Path)

    combo_parser = subparsers.add_parser("combo", help="surface+regret multi-panel figure")
    combo_parser.add_argument("npzdirs", nargs="+", type=Path)
    combo_parser.add_argument("-o", "--outfile", type=Path)

    regret_parser = subparsers.add_parser("regret", help="regret trace for a single objective")
    regret_parser.add_argument("npzdir", type=Path, help="the directory containing the X, Y, and H .npz files")
    regret_parser.add_argument("objective", help="the objective function used for the runs")
    regret_parser.add_argument("-o", "--outfile", type=Path)

    gamma_parser = subparsers.add_parser("gamma-perf", help="gamma sweep multi-panel figure")
    gamma_parser.add_argument("gamma_dir", type=Path)
    gamma_parser.add_argument("objective")
    gamma_parser.add_argument("-o", "--outfile", type=Path)

    args = parser.parse_args()

    if args.figure == "michalewicz":
        michalewicz(args.npzdir, args.gamma_dir, args.outfile)
    elif args.figure == "combo":
        surface_regret(args.npzdirs, args.outfile)
    elif args.figure == "regret":
        regret(args.npzdir, args.objective, args.outfile)
    elif args.figure == "gamma-perf":
        gamma_perf(args.gamma_dir, args.objective, args.outfile)

if __name__ == "__main__":
    main()