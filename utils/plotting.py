"""Utility plotting helpers for Lorenz forecasting experiments.

Functions
---------
plot_timeseries(...)
    2‑D line plot of x‑dimension predictions vs ground‑truth.

plot_lorenz_3d(...)
    3‑D trajectory comparison (truth vs one model).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Sequence, Optional

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


# ---------------------------------------------------------------------------
# Helper: ensure the figures directory exists
# ---------------------------------------------------------------------------
_DEFAULT_FIG_DIR = Path(__file__).resolve().parent.parent / "figures"
_DEFAULT_FIG_DIR.mkdir(parents=True, exist_ok=True)


def _resolve_save_path(filename: str | Path | None,
                       default_name: str) -> Path:
    """Return a Path object under *figures/* with given or default name."""
    if filename is None:
        filename = default_name
    p = Path(filename)
    if not p.parent or p.parent == Path('.'):
        # place inside figures/ by default
        p = _DEFAULT_FIG_DIR / p
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


# ---------------------------------------------------------------------------
# 1) 2‑D time‑series plot
# ---------------------------------------------------------------------------

def plot_timeseries(time: Sequence[float],
                    test_target: pd.DataFrame | Sequence[Sequence[float]] | None,
                    preds_dict: Dict[str, Sequence[Sequence[float]]],
                    plot_len: Optional[int] = 3000,
                    dpi: int = 300,
                    save_as: str | Path | None = None,
                    show: bool = True) -> Path:
    """Line plot of x‑dimension ground‑truth and model predictions.

    Parameters
    ----------
    time        : sequence of time points (len ≥ plot_len)
    test_target : array‑like [T,3] – only x‑dim is used for the reference curve
    preds_dict  : mapping model‑name → array‑like [T,3]
    plot_len    : truncate to at most this many steps (None → full length)
    dpi         : figure resolution when saving
    save_as     : filename or Path (defaults to figures/lorenz_timeseries.png)
    show        : whether to call plt.show()

    Returns
    -------
    Path to the saved PNG file.
    """
    import numpy as np  # local import avoids hard dependency here

    if plot_len is None:
        plot_len = len(time)

    df_plot = pd.DataFrame({
        'Time': time[:plot_len],
        'True x(t)': np.asarray(test_target)[:plot_len, 0] if test_target is not None else None,
    })

    # Add each model's x‑dimension
    for name, arr in preds_dict.items():
        df_plot[f"{name} x(t)"] = np.asarray(arr)[:plot_len, 0]

    # Plot ------------------------------------------------------------------
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=df_plot, x='Time', y='True x(t)', label='True x(t)', linestyle='--')

    # Plot only the models present in preds_dict (skip ground‑truth key)
    for name in preds_dict:
        sns.lineplot(data=df_plot, x='Time', y=f"{name} x(t)", label=f"{name} x(t)")

    plt.title('Lorenz Autoregressive Prediction (x-dim)')
    plt.xlabel('Time')
    plt.ylabel('x')
    plt.legend()
    plt.tight_layout()

    save_path = _resolve_save_path(save_as, 'lorenz_timeseries.png')
    plt.savefig(save_path, dpi=dpi)
    if show:
        plt.show()
    else:
        plt.close()
    return save_path


# ---------------------------------------------------------------------------
# 2) 3‑D Lorenz trajectory plot
# ---------------------------------------------------------------------------

def plot_lorenz_3d(test_target,
                   preds,
                   plot_len: Optional[int] = 3000,
                   dpi: int = 300,
                   save_as: str | Path | None = None,
                   show: bool = True) -> Path:
    """3‑D plot of true Lorenz trajectory vs a model prediction.

    Only one prediction series is plotted (pass the model of interest).
    """
    import numpy as np

    if plot_len is None:
        plot_len = len(test_target)

    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 – side‑effect import

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(projection='3d')

    targ = np.asarray(test_target)[:plot_len]
    pred = np.asarray(preds)[:plot_len]

    ax.plot(targ[:, 0], targ[:, 1], targ[:, 2],
            label='True (Test)', linestyle='--', color='k')
    ax.plot(pred[:, 0], pred[:, 1], pred[:, 2],
            label='Prediction', color='blue')

    ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_zlabel('z')
    ax.legend()
    plt.tight_layout()

    save_path = _resolve_save_path(save_as, 'lorenz_3d.png')
    plt.savefig(save_path, dpi=dpi)
    if show:
        plt.show()
    else:
        plt.close()
    return save_path

# ---------------------------------------------------------------------------
__all__ = [
    'plot_timeseries',
    'plot_lorenz_3d',
]
