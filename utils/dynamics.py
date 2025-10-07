# --------------------------------------------------------------------------
# spectrum_helpers.py  — eigen-spectrum visualisation
# --------------------------------------------------------------------------
from __future__ import annotations
from pathlib import Path
from typing import Union, Sequence, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns     

from sklearn.linear_model import Ridge
from matplotlib import patheffects

from utils.plotting import _resolve_save_path      

_FIG_DIR = Path(__file__).resolve().parent.parent / "figures"
_FIG_DIR.mkdir(parents=True, exist_ok=True)




def _extract_matrix(obj: Union[np.ndarray, "object"]) -> np.ndarray:
    if isinstance(obj, np.ndarray):
        return obj
    for attr in ("W_res", "W_base", "W0", "W", "Ps", "weight", "weights", "W_nn"):
        if hasattr(obj, attr):
            return np.asarray(getattr(obj, attr))
    raise TypeError(
        "Input must be a square numpy array or an object with W_base/W attribute."
    )


def plot_eigen_spectrum(
    matrix_or_model: Union[np.ndarray, "object"],
    dpi: int = 300,
    unit_circle: bool = True,
    spectral_radius: Optional[float] = None,
    title: Optional[str] = None,
    save_as: str | Path | None = None,
    show: bool = True,
) -> Path:
    """Pretty scatter‑plot of eigenvalues in the complex plane."""
    W = _extract_matrix(matrix_or_model)
    if W.shape[0] != W.shape[1]:
        raise ValueError("Weight matrix must be square.")

    eig = np.linalg.eigvals(W)
    mag = np.abs(eig)
    θ = np.linspace(0, 2 * np.pi, 720)

    #  -------- visual style --------
    sns.set_theme(style="white", context="talk")
    fig, ax = plt.subplots(figsize=(6, 6), dpi=dpi, facecolor="white")
    ax.set_facecolor("#f7f7f7")

    #  -------- scatter cloud --------
    sns.scatterplot(
        x=eig.real,
        y=eig.imag,
        hue=mag,
        palette="magma",
        edgecolor="black",
        linewidth=0.15,
        s=28,
        alpha=0.9,
        ax=ax,
    )
    ax.get_legend().remove()

    #  -------- spectral circles --------
    circle_kw = dict(ls="--", lw=1.25, zorder=1,
                     path_effects=[patheffects.withStroke(linewidth=3,
                                                          foreground="white")])
    if unit_circle:
        ax.plot(np.cos(θ), np.sin(θ), color="black", **circle_kw, label="$|\\lambda|=1$")
    if spectral_radius is not None and spectral_radius > 0:
        ax.plot(spectral_radius * np.cos(θ),
                spectral_radius * np.sin(θ),
                color="#d62728",
                **circle_kw,
                label=f"$|\\lambda|={spectral_radius}$")

    #  -------- axes & colour‑bar --------
    ax.axhline(0, lw=0.8, c="grey", alpha=0.5)
    ax.axvline(0, lw=0.8, c="grey", alpha=0.5)
    ax.set_aspect("equal", "box")
    ax.set_xlabel("Re$(\\lambda)$", labelpad=6)
    ax.set_ylabel("Im$(\\lambda)$", labelpad=6)

    cbar = fig.colorbar(
        plt.cm.ScalarMappable(cmap="magma", norm=plt.Normalize(mag.min(), mag.max())),
        ax=ax, pad=0.02, shrink=0.85,
    )
    cbar.set_label("$|\\lambda|$", rotation=270, labelpad=15)

    #  -------- title --------
    if title is None:
        ρ = np.max(mag)
        title = f"Eigen‑spectrum   $N={W.shape[0]}$,  $\\rho={ρ:.3f}$"
    #ax.set_title(title, pad=12)

    sns.despine(fig)
    plt.tight_layout()

    #  -------- save / show --------
    if save_as is None:
        fname = title.lower().replace(" ", "_").replace("$", "").replace("\\", "")
        save_as = f"{fname}.png"
    save_path = Path(save_as).expanduser().resolve()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
    if show:
        plt.show()
    else:
        plt.close(fig)
    return save_path



# ---------------------------------------------------------------------------
# utils/memory.py  —  memory-capacity experiment helpers
# ---------------------------------------------------------------------------
"""Memory-capacity (MC) experiment utilities.

Functions
---------
compute_memory_capacity(esn, ...)
    Returns delays, C_l spectrum, and total MC.

plot_memory_capacity(...)
    Publication-ready Seaborn plot of the MC spectrum.
"""

 


# --------------------------------------------------------------------- #
# MC computation                                                        #
# --------------------------------------------------------------------- #
def compute_memory_capacity(
    esn,
    max_delay: int = 200,
    T: int = 4_000,
    discard: int = 200,
    ridge_alpha: float = 1e-6,
    seed: int = 42,
    use_poly: bool = False,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Estimate the linear memory-capacity (Jaeger, 2001) of an ESN.

    Returns
    -------
    delays       : shape (max_delay,)
    capacities   : shape (max_delay,)
    total_MC     : float  –  Σ C_ℓ
    """
    rng = np.random.default_rng(seed)
    u = rng.uniform(-1.0, 1.0, size=(T, 1)).astype(np.float32)

    # fresh scalar input weights
    esn.W_in = (rng.uniform(-1, 1, size=(esn.N, 1)) * esn.input_scale).astype(
        np.float32
    )
    esn.use_poly = use_poly
    if hasattr(esn, "reset_state"):
        esn.reset_state()
    elif hasattr(esn, "reset"):
        esn.reset()

    # Roll out the reservoir
    states = []
    for t in range(T):
        if hasattr(esn, "_step"):
            esn._step(u[t])
        else:
            esn._update(u[t])
        if t >= discard:
            states.append(esn.x.copy() if hasattr(esn, "x") else esn.state.copy())
    X = np.asarray(states, dtype=np.float32)
    if use_poly:
        X = np.concatenate([X, X**2, np.ones((X.shape[0], 1))], axis=1)

    # Capacity spectrum
    capacities = []
    for l in range(1, max_delay + 1):
        if l >= X.shape[0]:
            capacities.append(0.0)
            continue
        y = u[discard - l : T - l].reshape(-1, 1)
        reg = Ridge(alpha=ridge_alpha, fit_intercept=False)
        reg.fit(X[: y.shape[0]], y)
        y_hat = reg.predict(X[: y.shape[0]])
        c = np.corrcoef(y_hat[:, 0], y[:, 0])[0, 1] ** 2
        capacities.append(float(c))

    delays = np.arange(1, max_delay + 1)
    return delays, np.asarray(capacities), float(np.sum(capacities))


# --------------------------------------------------------------------- #
# MC visualisation                                                      #
# --------------------------------------------------------------------- #
def plot_memory_capacity(
    delays: np.ndarray,
    capacities: np.ndarray,
    title: str = "Memory-Capacity Spectrum",
    dpi: int = 300,
    save_as: str | Path | None = None,
    show: bool = True,
):
    """
    Draw and save a Seaborn-styled plot of the memory-capacity spectrum.
    """
    # Seaborn look-and-feel (local)
    sns.set_theme(context="talk", style="whitegrid", font_scale=0.9)

    save_path = _resolve_save_path(save_as, "memory_capacity.png")
    fig, ax = plt.subplots(figsize=(7, 4), dpi=dpi)


    # Line + markers
    sns.lineplot(
        x=delays,
        y=capacities,
        marker="o",
        markersize=4,
        linewidth=1.5,
        ax=ax,
        color=sns.color_palette("rocket_r", 1)[0],
    )
    # Soft fill under curve for emphasis
    ax.fill_between(
        delays,
        capacities,
        alpha=0.15,
        color=sns.color_palette("rocket_r", 1)[0],
    )

    ax.set_xlabel(r"Delay $\ell$")
    ax.set_ylabel(r"$C_\ell$  (squared correlation)")
    ax.set_title(title, pad=10)
    ax.set_xlim(1, delays.max())
    ax.set_ylim(0, max(0.05, capacities.max() * 1.05))

    plt.tight_layout()
    fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)

    return save_path



# ---------------------------------------------------------------------
# Effective Dimensionality (Participation Ratio)
# ---------------------------------------------------------------------
def estimate_participation_ratio(
    esn,
    T: int = 4000,
    discard: int = 200,
    seed: int = 42,
    use_poly: bool = False,
):
    """
    Estimate the state-space participation ratio of an ESN. A higher 
    participation ratio indicates that the reservoir spreads its 
    variance across many orthogonal directions, suggesting richer
    internal dynamics.

    Parameters
    ----------
    esn       : reservoir instance with methods
                reset_state()/reset() and _step(u) or _update(u).
                Must expose:
                    - esn.N               (reservoir size)
                    - esn.W_in            (initialised input matrix)
                    - esn.input_scale     (scalar)
                    - esn.x or esn.state  (current state vector)
    T         : total drive length.
    discard   : wash-out steps to ignore.
    seed      : RNG seed.
    use_poly  : if True, augment state with x² and bias.

    Returns
    -------
    pr        : float   – participation ratio
    eig_vals  : np.ndarray of variances (descending)

    Notes
    -----
    Participation ratio is  (Σ λ_i)² / Σ λ_i²  where λ_i
    are eigenvalues (variances) of the state covariance matrix.
    """
    rng = np.random.default_rng(seed)
    d_in = esn.W_in.shape[1] if hasattr(esn, "W_in") else 1
    u_seq = rng.uniform(-1, 1, size=(T, d_in)).astype(np.float32)

    # fresh scalar input weights if needed
    if hasattr(esn, "W_in"):
        esn.W_in = (
            rng.uniform(-1, 1, size=(esn.N, d_in)) * esn.input_scale
        ).astype(np.float32)

    # initialise
    if hasattr(esn, "reset_state"):
        esn.reset_state()
    elif hasattr(esn, "reset"):
        esn.reset()

    states = []
    for u in u_seq:
        if hasattr(esn, "_step"):
            esn._step(u)
        else:
            esn._update(u)

        x_vec = esn.x if hasattr(esn, "x") else esn.state
        states.append(x_vec.copy())

    X = np.asarray(states[discard:], dtype=np.float32)   # [T_d, N]
    if use_poly:
        X = np.concatenate([X, X * X, np.ones((X.shape[0], 1))], axis=1)

    # covariance matrix of columns (features)
    C = np.cov(X.T, bias=True)                          # shape (F, F)
    eig_vals = np.sort(np.linalg.eigvalsh(C))[::-1]     # descending

    pr = (eig_vals.sum() ** 2) / (np.square(eig_vals).sum() + 1e-12)
    return float(pr), eig_vals



# --------------------------------------------------------------------- #
# Public API                                                            #
# --------------------------------------------------------------------- #


__all__ = [
    "plot_eigen_spectrum",
    "compute_memory_capacity",
    "plot_memory_capacity",
    "estimate_participation_ratio",  
]


