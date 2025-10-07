import numpy as np

##############################################################################
# 1) Utility: Scale spectral radius
##############################################################################

def scale_spectral_radius(W, target_radius=0.95):
    """
    Scales matrix W so that its largest eigenvalue magnitude = target_radius.
    """
    eigenvals = np.linalg.eigvals(W)
    radius = np.max(np.abs(eigenvals))
    if radius == 0:
        return W
    return (W / radius) * target_radius

##############################################################################
# 2) Generic Helper: building augmented readout vector with squared terms
##############################################################################

def augment_state_with_squares(x):
    """
    Given state vector x in R^N, return [ x, x^2, 1 ] in R^(2N+1).
    We'll use this for both training and prediction.
    """
    x_sq = x**2
    return np.concatenate([x, x_sq, [1.0]])  # shape: 2N+1


##############################################################################
# hyperbolic distance
##############################################################################
def hyperbolic_distance_poincare_d(u: np.ndarray, v: np.ndarray) -> float:
    """
    Hyperbolic distance d(u,v) in the d‑dimensional Poincaré ball (κ = −1).
    """
    eps = 1e-14
    num = 2.0 * np.dot(u - v, u - v)
    den = max(eps, (1.0 - np.dot(u, u)) * (1.0 - np.dot(v, v)))
    return np.arccosh(max(1.0 + num / den, 1.0))