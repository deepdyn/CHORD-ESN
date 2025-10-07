import numpy as np
from sklearn.linear_model import Ridge


class TGCCoupledESN:
    """
    Theta–Gamma Cross-Frequency Coupled Echo-State Network (TGC-ESN)
    ----------------------------------------------------------------
    Reservoir with two coupled oscillatory sub-states plus a mixed state.

    State vectors
    -------------
        θ_t  ∈ ℝᴺ  : theta-band components   (≈ 6 Hz; fast)
        γ_t  ∈ ℝᴺ  : gamma-band components   (≈ 80 Hz; fast)
        x_t  ∈ ℝᴺ  : mixed hidden units      (intermediate)

    Update order per time-step t → t+1
    ----------------------------------
        1) θ ← (1-λ) θ + λ · sin( W_θ θ + V_θ u_t )
        2) γ ← (1-η) γ + η · tanh( W_γ γ + V_γ u_t + c · sin(θ) )
        3) x ← (1-α) x + α · tanh( W_mix x + G_θ θ + G_γ γ )

    For spectrum visualization, a combined block recurrent matrix is exposed
    as `self.W_res` (and alias `self.W`) over the concatenated state [θ; γ; x].
    """

    # ------------------------------------------------------------------
    #                             constructor
    # ------------------------------------------------------------------
    def __init__(
        self,
        reservoir_size: int = 800,          # N
        input_dim: int = 3,
        rho_theta: float = 0.8,             # spectral radius for W_θ
        rho_gamma: float = 0.8,             # spectral radius for W_γ
        rho_mix: float = 0.8,               # spectral radius for W_mix
        lambda_theta: float = 0.3,          # θ leak  (λ)
        eta_gamma: float = 0.3,             # γ leak  (η)
        alpha_mix: float = 0.5,             # x leak  (α)
        c_cfc: float = 1.0,                 # θ–γ coupling coefficient (c)
        input_scale: float = 0.5,
        ridge_alpha: float = 1e-6,
        use_quadratic_readout: bool = True,
        seed: int = 42,
    ):
        self.N = reservoir_size
        self.d_in = input_dim

        # leaks
        self.lmb = lambda_theta
        self.eta = eta_gamma
        self.alpha = alpha_mix
        self.c = c_cfc

        self.ridge_alpha = ridge_alpha
        self.use_quad = use_quadratic_readout

        rng = np.random.default_rng(seed)

        # ---------------- weight initialisation -----------------------
        # Helper to scale matrices by spectral radius
        def _scaled_matrix(shape, rho):
            W = rng.standard_normal(shape).astype(np.float32)
            s_max = np.linalg.svd(W, compute_uv=False)[0]  # largest singular value
            return (rho / (s_max + 1e-12)) * W

        self.W_theta = _scaled_matrix((self.N, self.N), rho_theta)
        self.W_gamma = _scaled_matrix((self.N, self.N), rho_gamma)
        self.W_mix   = _scaled_matrix((self.N, self.N), rho_mix)

        self.V_theta = (
            rng.uniform(-1, 1, size=(self.N, self.d_in)).astype(np.float32) * input_scale
        )
        self.V_gamma = (
            rng.uniform(-1, 1, size=(self.N, self.d_in)).astype(np.float32) * input_scale
        )

        # Diagonal coupling gains G_θ, G_γ (can be dense if desired)
        self.G_theta = np.diag(rng.uniform(0, 1, size=self.N).astype(np.float32))
        self.G_gamma = np.diag(rng.uniform(0, 1, size=self.N).astype(np.float32))

        # ---------------- expose combined recurrent matrix -------------
        # Block matrix over concatenated state [theta; gamma; x]
        N = self.N
        Z = np.zeros((N, N), dtype=np.float32)
        self.W_res = np.block([
            [self.W_theta, Z,            Z           ],
            [Z,            self.W_gamma, Z           ],
            [self.G_theta, self.G_gamma, self.W_mix  ],
        ]).astype(np.float32)
        # alias for helpers that look for "W"
        self.W = self.W_res

        # ---------------- dynamic state -------------------------------
        self.theta = np.zeros(self.N, dtype=np.float32)
        self.gamma = np.zeros(self.N, dtype=np.float32)
        self.x     = np.zeros(self.N, dtype=np.float32)

        self.W_out: np.ndarray | None = None

    # ------------------------------------------------------------------
    #                            helpers
    # ------------------------------------------------------------------
    def reset_state(self):
        self.theta.fill(0.0)
        self.gamma.fill(0.0)
        self.x.fill(0.0)

    def _step(self, u_t: np.ndarray):
        """Single theta–gamma–mixed update."""

        # θ-band update (sine oscillator)
        z_theta = self.W_theta @ self.theta + self.V_theta @ u_t
        theta_new = np.sin(z_theta).astype(np.float32)
        self.theta = (1.0 - self.lmb) * self.theta + self.lmb * theta_new

        # γ-band update (tanh + θ-coupling)
        z_gamma = (
            self.W_gamma @ self.gamma
            + self.V_gamma @ u_t
            + self.c * np.sin(self.theta)
        )
        gamma_new = np.tanh(z_gamma).astype(np.float32)
        self.gamma = (1.0 - self.eta) * self.gamma + self.eta * gamma_new

        # mixed state
        drive = (
            self.W_mix @ self.x
            + self.G_theta @ self.theta
            + self.G_gamma @ self.gamma
        )
        x_new = np.tanh(drive).astype(np.float32)
        self.x = (1.0 - self.alpha) * self.x + self.alpha * x_new

    # ------------------------------------------------------------------
    #                        read-out training
    # ------------------------------------------------------------------
    def fit_readout(self, inputs: np.ndarray, targets: np.ndarray, discard: int = 100):
        """
        Train linear / quadratic ridge read-out.

        inputs  : (T, d_in)
        targets : (T, d_out)
        """
        T, d_in = inputs.shape
        if d_in != self.d_in:
            raise ValueError("input_dim mismatch")
        if T <= discard + 1:
            raise ValueError("sequence too short")

        self.reset_state()
        feats = []
        for t in range(T):
            self._step(inputs[t])
            if t >= discard:
                if self.use_quad:
                    feats.append(
                        np.concatenate(
                            [
                                self.x,
                                self.x * self.x,
                                self.theta,
                                self.gamma,
                                [1.0],
                            ]
                        )
                    )
                else:
                    feats.append(
                        np.concatenate(
                            [self.x, self.theta, self.gamma, [1.0]]
                        )
                    )

        X_feat = np.asarray(feats, dtype=np.float32)
        Y = targets[discard:]

        reg = Ridge(alpha=self.ridge_alpha, fit_intercept=False)
        reg.fit(X_feat, Y)
        self.W_out = reg.coef_.astype(np.float32)

    # ------------------------------------------------------------------
    #                  autoregressive free-run forecast
    # ------------------------------------------------------------------
    def predict_autoregressive(
        self,
        init_input: np.ndarray,
        n_steps: int,
    ) -> np.ndarray:
        """
        Autoregressive rollout for n_steps.

        Parameters
        ----------
        init_input : (d_in,) first input fed into reservoir
        n_steps    : number of timesteps to predict

        Returns
        -------
        preds : (n_steps, d_out)
        """
        if self.W_out is None:
            raise RuntimeError("Call fit_readout() first")

        d_out = self.W_out.shape[0]
        preds = np.empty((n_steps, d_out), dtype=np.float32)

        u_t = init_input.astype(np.float32).copy()

        for t in range(n_steps):
            self._step(u_t)

            if self.use_quad:
                feat_vec = np.concatenate(
                    [self.x, self.x * self.x, self.theta, self.gamma, [1.0]]
                )
            else:
                feat_vec = np.concatenate([self.x, self.theta, self.gamma, [1.0]])

            y_t = (self.W_out @ feat_vec).astype(np.float32)
            preds[t] = y_t
            u_t = y_t[: self.d_in]  # feedback loop

        return preds

