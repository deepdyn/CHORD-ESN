import numpy as np
from sklearn.linear_model import Ridge


# ---------------------------------------------------------------------
# Thalamo–Cortical Burst-Gated Echo-State Network (TCB-ESN)
# ---------------------------------------------------------------------
class TCBESN:
    """
    Thalamo–Cortical Burst-Gated Echo-State Network (TCB-ESN)

    Two interacting pools:
      • Cortex  (fast, recurrent):        x_t ∈ R^{N_c}
      • Thalamus (relay with bursts):     p_t (fast), q_t (slow packet),
                                          h_t (readiness), b_t (burst gate),
                                          r_t (reticular inhibition), y_t (output to cortex)

    Update (per step)
    -----------------
      s_t  = W_tc x_{t-1} + W_tin u_t − γ_r r_{t-1}                    (1)
      p_t  = (1−α_f) p_{t-1} + α_f tanh(s_t)                           (2)
      q_t  = (1−α_s) q_{t-1} + α_s p_t                                 (3)
      h_t  = (1−α_h) h_{t-1} + α_h σ(θ_h − p_{t-1})                    (4)  # readiness from prior hyperpol.
      b_t  = σ(k_b (p_t − θ_b)) ⊙ h_t                                  (5)  # burst = depol × readiness
      y_t  = (1 − b_t) ⊙ p_t + b_t ⊙ q_t                               (6)  # tonic vs packet
      r_t  = (1−α_r) r_{t-1} + α_r σ(W_rt y_{t-1} + W_rc x_{t-1} − θ_r)(7)  # TRN-like slow inhibition
      ξ_t  = tanh(W_cc x_{t-1} + W_ct y_t + W_in u_t)                  (8)
      x_t  = (1−λ) x_{t-1} + λ ξ_t

    Only the linear read-out W_out is trained (ridge regression).

    Parameters
    ----------
    cortex_size : int = 800
        Number of cortical neurons (N_c).
    thalamus_size : int = 200
        Number of thalamic relay units (N_t).
    input_dim : int = 3
        Dimensionality of external input u_t.
    spectral_radius_cc : float = 0.9
        Spectral radius for cortical recurrent matrix W_cc.
    gain_ct : float = 0.3
        Scale for cortex←thalamus matrix W_ct.
    gain_tc : float = 0.3
        Scale for thalamus←cortex matrix W_tc.
    input_scale_cortex : float = 0.5
        Scale for cortical input weights W_in (uniform ±scale).
    input_scale_thalamus : float = 0.5
        Scale for thalamic input weights W_tin (uniform ±scale).
    retic_scale : float = 0.1
        Scale for reticular couplings W_rt (T→R) and W_rc (C→R).
    lam_leak : float = 0.2
        Cortical leak λ ∈ (0,1].
    alpha_f : float = 0.4
        Fast thalamic rate (EPSC-like filter).
    alpha_s : float = 0.04
        Slow packet rate (≫ burst duration).
    alpha_h : float = 0.01
        Readiness (T-type) recovery rate.
    theta_h : float = 0.3
        Readiness threshold (applied to previous p).
    theta_b : float = 0.4
        Burst depolarization threshold (applied to current p).
    k_b : float = 8.0
        Burst sigmoid steepness.
    alpha_r : float = 0.01
        Reticular trace update rate.
    theta_r : float = 0.3
        Reticular activation threshold.
    gamma_r : float = 0.5
        Strength of reticular inhibition in thalamic pre-drive (eq. 1).
    ridge_alpha : float = 1e-6
        ℓ2 regularisation for ridge read-out.
    use_poly : bool = True
        If True, append element-wise squares and a bias to features.
    feature_mode : {'cortex','ctx_thal','full'} = 'cortex'
        Read-out features:
          'cortex'   → x
          'ctx_thal' → [x ; y]
          'full'     → [x ; y ; b ; h ; p ; q]
    seed : int = 42
        PRNG seed.

    Notes
    -----
    For stability (ESP), W_cc is spectral-scaled to spectral_radius_cc.
    Cross matrices W_ct, W_tc, and reticular W_rt, W_rc are norm-scaled
    by gain_ct, gain_tc, and retic_scale respectively.
    """

    # -----------------------------------------------------------------
    def __init__(
        self,
        cortex_size: int = 800,
        thalamus_size: int = 200,
        input_dim: int = 3,
        spectral_radius_cc: float = 0.9,
        gain_ct: float = 0.3,
        gain_tc: float = 0.3,
        input_scale_cortex: float = 0.5,
        input_scale_thalamus: float = 0.5,
        retic_scale: float = 0.1,
        lam_leak: float = 0.2,
        alpha_f: float = 0.4,
        alpha_s: float = 0.04,
        alpha_h: float = 0.01,
        theta_h: float = 0.3,
        theta_b: float = 0.4,
        k_b: float = 8.0,
        alpha_r: float = 0.01,
        theta_r: float = 0.3,
        gamma_r: float = 0.5,
        ridge_alpha: float = 1e-6,
        use_poly: bool = True,
        feature_mode: str = "cortex",
        seed: int = 42,
    ):
        # --- sanity
        if cortex_size < 1 or thalamus_size < 1:
            raise ValueError("cortex_size and thalamus_size must be ≥ 1")
        if not 0 < lam_leak <= 1:
            raise ValueError("lam_leak must lie in (0,1]")
        if feature_mode not in {"cortex", "ctx_thal", "full"}:
            raise ValueError("feature_mode must be 'cortex', 'ctx_thal', or 'full'")

        self.Nc = cortex_size
        self.Nt = thalamus_size
        self.d_in = input_dim

        self.rho_cc = spectral_radius_cc
        self.gain_ct = gain_ct
        self.gain_tc = gain_tc
        self.in_scale_cx = input_scale_cortex
        self.in_scale_th = input_scale_thalamus
        self.retic_scale = retic_scale

        self.lam = lam_leak
        self.af = alpha_f
        self.as_ = alpha_s
        self.ah = alpha_h
        self.th_h = theta_h
        self.th_b = theta_b
        self.k_b = k_b
        self.ar = alpha_r
        self.th_r = theta_r
        self.gamma_r = gamma_r

        self.ridge_alpha = ridge_alpha
        self.use_poly = use_poly
        self.feature_mode = feature_mode
        self.seed = seed

        rng = np.random.default_rng(seed)

        # --- W_cc: dense Gaussian scaled to spectral_radius_cc
        Wcc_raw = rng.standard_normal((self.Nc, self.Nc)).astype(np.float32)
        eig_est = float(np.max(np.abs(np.linalg.eigvals(Wcc_raw))))
        self.W_cc = (self.rho_cc / (eig_est + 1e-8)) * Wcc_raw

        # --- Cross & input matrices (norm-scaled)
        self.W_ct = (rng.standard_normal((self.Nc, self.Nt)).astype(np.float32))
        self.W_ct *= self.gain_ct / (np.linalg.norm(self.W_ct, 2) + 1e-8)

        self.W_tc = (rng.standard_normal((self.Nt, self.Nc)).astype(np.float32))
        self.W_tc *= self.gain_tc / (np.linalg.norm(self.W_tc, 2) + 1e-8)

        self.W_in = (
            rng.uniform(-1.0, 1.0, size=(self.Nc, self.d_in)) * self.in_scale_cx
        ).astype(np.float32)

        self.W_tin = (
            rng.uniform(-1.0, 1.0, size=(self.Nt, self.d_in)) * self.in_scale_th
        ).astype(np.float32)

        # --- Reticular couplings (small)
        self.W_rt = (rng.standard_normal((self.Nt, self.Nt)).astype(np.float32))
        self.W_rt *= self.retic_scale / (np.linalg.norm(self.W_rt, 2) + 1e-8)

        self.W_rc = (rng.standard_normal((self.Nt, self.Nc)).astype(np.float32))
        self.W_rc *= self.retic_scale / (np.linalg.norm(self.W_rc, 2) + 1e-8)

        # --- runtime state
        self.x = np.zeros(self.Nc, dtype=np.float32)  # cortex
        self.p = np.zeros(self.Nt, dtype=np.float32)  # thalamus fast
        self.q = np.zeros(self.Nt, dtype=np.float32)  # thalamus packet
        self.h = np.zeros(self.Nt, dtype=np.float32)  # readiness
        self.b = np.zeros(self.Nt, dtype=np.float32)  # burst gate
        self.r = np.zeros(self.Nt, dtype=np.float32)  # reticular inhibitory trace
        self.y = np.zeros(self.Nt, dtype=np.float32)  # thalamic output

        # trained read-out
        self.W_out: np.ndarray | None = None

    # -----------------------------------------------------------------
    @staticmethod
    def _sigmoid(z: np.ndarray) -> np.ndarray:
        z = np.clip(z, -20.0, 20.0)
        return 1.0 / (1.0 + np.exp(-z))

    # -----------------------------------------------------------------
    def _features_from_state(self) -> np.ndarray:
        """Assemble read-out features per feature_mode (+optional polynomial lift)."""
        if self.feature_mode == "cortex":
            feat = self.x
        elif self.feature_mode == "ctx_thal":
            feat = np.concatenate([self.x, self.y], axis=0)
        else:  # 'full'
            feat = np.concatenate([self.x, self.y, self.b, self.h, self.p, self.q], axis=0)

        if self.use_poly:
            feat = np.concatenate([feat, feat * feat, [1.0]], axis=0)

        return feat.astype(np.float32)

    # -----------------------------------------------------------------
    def _step(self, u_t: np.ndarray):
        """
        One TCB-ESN update with external input u_t.
        Order of ops respects the equations (uses previous states where specified).
        """
        x_prev = self.x
        p_prev = self.p
        q_prev = self.q
        h_prev = self.h
        r_prev = self.r
        y_prev = self.y

        # --- (1) thalamic pre-drive with reticular inhibition
        s = (self.W_tc @ x_prev) + (self.W_tin @ u_t) - (self.gamma_r * r_prev)

        # --- (2) fast thalamic filter
        p_new = (1.0 - self.af) * p_prev + self.af * np.tanh(s)

        # --- (3) slow packet
        q_new = (1.0 - self.as_) * q_prev + self.as_ * p_new

        # --- (4) readiness from prior hyperpolarization proxy (small p_prev)
        h_new = (1.0 - self.ah) * h_prev + self.ah * self._sigmoid(self.th_h - p_prev)

        # --- (5) burst gate: current depol × readiness
        b_new = self._sigmoid(self.k_b * (p_new - self.th_b)) * h_new

        # --- (6) thalamic output
        y_new = (1.0 - b_new) * p_new + b_new * q_new

        # --- (7) reticular slow inhibition (uses previous y and x)
        r_new = (1.0 - self.ar) * r_prev + self.ar * self._sigmoid(
            (self.W_rt @ y_prev) + (self.W_rc @ x_prev) - self.th_r
        )

        # --- (8) cortical innovation & leak
        pre_c = (self.W_cc @ x_prev) + (self.W_ct @ y_new) + (self.W_in @ u_t)
        xi = np.tanh(pre_c).astype(np.float32)
        x_new = (1.0 - self.lam) * x_prev + self.lam * xi

        # --- commit
        self.p = p_new.astype(np.float32)
        self.q = q_new.astype(np.float32)
        self.h = h_new.astype(np.float32)
        self.b = b_new.astype(np.float32)
        self.y = y_new.astype(np.float32)
        self.r = r_new.astype(np.float32)
        self.x = x_new.astype(np.float32)

    # -----------------------------------------------------------------
    def reset_state(self):
        self.x.fill(0.0)
        self.p.fill(0.0)
        self.q.fill(0.0)
        self.h.fill(0.0)
        self.b.fill(0.0)
        self.r.fill(0.0)
        self.y.fill(0.0)

    # -----------------------------------------------------------------
    # Read-out training (teacher forcing)
    # -----------------------------------------------------------------
    def fit_readout(
        self,
        inputs: np.ndarray,     # shape [T, d_in]
        targets: np.ndarray,    # shape [T, d_out]
        discard: int = 100,
    ):
        T, d_in = inputs.shape
        if d_in != self.d_in:
            raise ValueError("input_dim mismatch")
        if T <= discard:
            raise ValueError("sequence too short for discard period")

        self.reset_state()
        feats = []
        for t in range(T):
            self._step(inputs[t])
            if t >= discard:
                feats.append(self._features_from_state())

        X = np.asarray(feats, dtype=np.float32)     # [T-d, F]
        Y = targets[discard:]                       # [T-d, d_out]

        reg = Ridge(alpha=self.ridge_alpha, fit_intercept=False)
        reg.fit(X, Y)
        self.W_out = reg.coef_.astype(np.float32)

    # -----------------------------------------------------------------
    # Closed-loop autoregressive prediction
    # -----------------------------------------------------------------
    def predict_autoregressive(
        self,
        init_input: np.ndarray,   # shape [d_in]
        n_steps: int,
    ) -> np.ndarray:
        if self.W_out is None:
            raise RuntimeError("Call fit_readout() before prediction")

        d_out = self.W_out.shape[0]
        preds = np.empty((n_steps, d_out), dtype=np.float32)

        #self.reset_state()
        current_u = init_input.astype(np.float32).copy()

        for t in range(n_steps):
            self._step(current_u)
            feat_vec = self._features_from_state()
            y_t = (self.W_out @ feat_vec).astype(np.float32)
            preds[t] = y_t
            # Feedback first d_in outputs as next input (common ESN practice)
            current_u = y_t[: self.d_in]

        return preds
