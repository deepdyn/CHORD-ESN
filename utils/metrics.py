import numpy as np

def mse_dimwise(pred, truth):
        length = min(len(pred), len(truth))
        return np.mean((pred[:length] - truth[:length])**2, axis=0)
    
    
##############################################################################
# NRMSE
##############################################################################


def evaluate_nrmse(all_preds: np.ndarray,
                   test_target: np.ndarray,
                   horizons: list[int]) -> dict[int, float]:
    """
    Compute scalar NRMSE for multiple prediction horizons.

    all_preds   : ndarray [T, d] – autoregressive predictions
    test_target : ndarray [T, d] – ground-truth targets
    horizons    : iterable of ints (steps) – e.g. [1, 5, 10, 25]

    Returns
    -------
    dict mapping horizon → NRMSE  (single scalar per horizon)
    """
    horizon_nrmse = {}
    for H in horizons:
        # slice up to the horizon
        pred_slice    = all_preds[:H].ravel()     # flatten to 1-D
        target_slice  = test_target[:H].ravel()

        mse   = np.mean((pred_slice - target_slice)**2)
        var   = np.var(target_slice)

        # Protect against zero variance (constant target)
        nrmse = np.sqrt(mse / var) if var > 0 else np.nan
        horizon_nrmse[H] = nrmse

    return horizon_nrmse


##############################################################################
# VPT
##############################################################################

def compute_valid_prediction_time(y_true, y_pred, t_vals, threshold=0.4, lambda_max=0.9):
    """
    Compute the Valid Prediction Time (VPT) and compare it to Lyapunov time T_lambda = 1 / lambda_max.
    
    Parameters
    ----------
    y_true : ndarray of shape (N, dim)
        True trajectory over time.
    y_pred : ndarray of shape (N, dim)
        Model's predicted trajectory over time (closed-loop).
    t_vals : ndarray of shape (N,)
        Time values corresponding to the trajectory steps.
    threshold : float, optional
        The error threshold, default is 0.4 as in your snippet.
    lambda_max : float, optional
        Largest Lyapunov exponent. Default=0.9 for Lorenz.
        
    Returns
    -------
    T_VPT : float
        Valid prediction time. The earliest time at which normalized error surpasses threshold
        (or the last time if never surpassed).
    T_lambda : float
        Lyapunov time = 1 / lambda_max
    ratio : float
        How many Lyapunov times the model prediction remains valid, i.e. T_VPT / T_lambda.
    """
    # 1) Average of y_true
    y_mean = np.mean(y_true, axis=0)  # shape (dim,)
    
    # 2) Time-averaged norm^2 of (y_true - y_mean)
    y_centered = y_true - y_mean
    denom = np.mean(np.sum(y_centered**2, axis=1))  # scalar
    
    # 3) Compute the normalized error delta_gamma(t) = ||y_true - y_pred||^2 / denom
    diff = y_true - y_pred
    err_sq = np.sum(diff**2, axis=1)  # shape (N,)
    delta_gamma = err_sq / denom      # shape (N,)
    
    # 4) Find the first time index where delta_gamma(t) exceeds threshold
    idx_exceed = np.where(delta_gamma > threshold)[0]
    if len(idx_exceed) == 0:
        # never exceeds threshold => set T_VPT to the final time
        T_VPT = t_vals[-1]
    else:
        T_VPT = t_vals[idx_exceed[0]]
    
    # 5) Compute T_lambda and ratio
    T_lambda = 1.0 / lambda_max
    ratio = T_VPT / T_lambda
    
    return T_VPT, T_lambda, ratio

##############################################################################
# Attractor Deviation (ADev)
##############################################################################

def compute_attractor_deviation(predictions, targets, cube_size=(0.1, 0.1, 0.1)):
    """
    Compute the Attractor Deviation (ADev) metric.

    Parameters:
        predictions (numpy.ndarray): Predicted trajectories of shape (n, 3).
        targets (numpy.ndarray): True trajectories of shape (n, 3).
        cube_size (tuple): Dimensions of the cube (dx, dy, dz).

    Returns:
        float: The ADev metric.
    """
    # Define the cube grid based on the range of the data and cube size
    min_coords = np.min(np.vstack((predictions, targets)), axis=0)
    max_coords = np.max(np.vstack((predictions, targets)), axis=0)

    # Create a grid of cubes
    grid_shape = ((max_coords - min_coords) / cube_size).astype(int) + 1

    # Initialize the cube occupancy arrays
    pred_cubes = np.zeros(grid_shape, dtype=int)
    target_cubes = np.zeros(grid_shape, dtype=int)

    # Map trajectories to cubes
    pred_indices = ((predictions - min_coords) / cube_size).astype(int)
    target_indices = ((targets - min_coords) / cube_size).astype(int)

    # Mark cubes visited by predictions and targets
    for idx in pred_indices:
        pred_cubes[tuple(idx)] = 1
    for idx in target_indices:
        target_cubes[tuple(idx)] = 1

    # Compute the ADev metric
    adev = np.sum(np.abs(pred_cubes - target_cubes))
    return adev

##############################################################################
# PSD
##############################################################################

def compute_psd(y, dt=0.02):
    z = y[:, 2]  # Extract Z-component

    # Compute PSD using Welch’s method
    freqs, psd = welch(z, fs=1/dt, window='hamming', nperseg=len(z))  # Using Hamming window

    return freqs, psd

##############################################################################
# Lyapunov Exponent Deviation
##############################################################################

lambda_max = {
    "lorenz63": 0.9056,     # classic sigma=10, beta=8/3, rho=28
    "rossler": 0.0714,      # a=0.2, b=0.2, c=5.7
    # ...
}

def compute_lyapunov_exponent(system_name, trajectory, dt,
                               method="rosenstein", min_t=0.0,
                               show=False):
    """
    Estimate the maximal Lyapunov exponent of a trajectory
    with NeuroKit2 (Rosenstein et al. algorithm).

    Parameters
    ----------
    system_name : str
        Lower-case key used to look up the ground-truth exponent in
        TRUE_LAMBDA_MAX (pass "" if you do not want that comparison).
    trajectory  : ndarray, shape (T,) or (T, D)
    dt          : float
        Integration time step (seconds).
    method      : {"rosenstein","wolf","kantz"}, optional
    min_t       : float, optional
        Transient to discard from the start of `trajectory` (seconds).
    show        : bool, optional
        Forwarded to NeuroKit (whether to plot the fit).

    Returns
    -------
    le_hat  : float
        Estimated largest Lyapunov exponent (units: 1/seconds).
    diff    : float or None
        |le_hat – true| if the reference value is known, else None.
    """

    # --- 1. ensure a 2-D array -------------------------------------------
    traj = np.asarray(trajectory, dtype=float)
    if traj.ndim == 1:
        traj = traj[:, None]

    # --- 2. optional transient removal -----------------------------------
    if min_t > 0:
        start_idx = int(np.ceil(min_t / dt))
        traj = traj[start_idx:]

    # --- 3. call NK -------------------------------------------------------
    fs = 1.0 / dt
    les = []
    for k in range(traj.shape[1]):
        # NeuroKit returns a Series -> take the first element
        le_series = nk.complexity_lyapunov(
            signal=traj[:, k],
            sampling_rate=fs,
            method=method,
            show=show,
            #delay=None,    # let NK choose optimal delay
            #dimension=None # let NK choose embedding dimension
        )
        les.append(float(le_series.iloc[0]))
    le_hat = max(les)

    # --- 4. optional ground-truth comparison -----------------------------
    true_val = TRUE_LAMBDA_MAX.get(system_name.lower(), None)
    diff = abs(le_hat - true_val) if true_val is not None else None
    return le_hat, diff
