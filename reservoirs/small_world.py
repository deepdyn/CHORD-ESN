import numpy as np
import networkx as nx
from sklearn.linear_model import Ridge
from numpy.linalg import eigvals


def scale_spectral_radius(W, target_rho):
    """Scale W so that its spectral radius is target_rho."""
    if target_rho <= 0:
        raise ValueError("target_rho must be positive.")
    eigmax = np.max(np.abs(eigvals(W)))
    if eigmax == 0:
        return W  # already zero; nothing to scale
    return (target_rho / eigmax) * W


class SWRes3D_IO:
    """
    Small-World Reservoir ESN for 3D -> 3D prediction with I/O segregation.

    - Reservoir wiring: Watts–Strogatz small-world graph (N, k=degree, p=rewiring_prob).
    - Strict tanh nonlinearity for all nodes.
    - I/O segregation:
        * Only a small subset of reservoir nodes receives input ("input nodes").
        * Readout observes only a (different, distant) subset ("output nodes").
    - Operating point controls:
        * 'spectral_radius' rescales W's eigen-spectrum.
        * 'gain' multiplies the recurrent term at update (lets you probe larger effective gains).
        * 'leaking_rate' is the leaky integrator coefficient.

    Training:
        Quadratic readout over OUTPUT subset: phi = [x_out, x_out^2, 1].

    Shapes:
        inputs:  (T, 3)
        targets: (T, 3)

    Example:
        esn = SWRes3D_IO(reservoir_size=300, num_input_nodes=12, num_output_nodes=12,
                         rewiring_prob=0.1, degree=6, spectral_radius=1.10, gain=1.3,
                         leaking_rate=0.7, ridge_alpha=1e-6, seed=42)
        esn.fit_readout(train_u, train_y, discard=200)
        y_hat = esn.predict_open_loop(test_u)
        y_free = esn.predict_autoregressive(initial_input=test_u[0], num_steps=1000)
    """

    def __init__(self,
                 reservoir_size=300,
                 rewiring_prob=0.1,
                 degree=6,
                 spectral_radius=0.95,
                 gain=1.0,
                 input_scale=1.0,
                 leaking_rate=1.0,
                 ridge_alpha=1e-6,
                 num_input_nodes=12,
                 num_output_nodes=12,
                 io_separation_mode="max",  # "max" or an integer separation
                 seed=42):

        # ------------------ store params ------------------
        self.N = int(reservoir_size)
        self.rewiring_prob = float(rewiring_prob)
        self.degree = int(degree)
        self.spectral_radius = float(spectral_radius)
        self.gain = float(gain)
        self.input_scale = float(input_scale)
        self.alpha = float(leaking_rate)
        self.ridge_alpha = float(ridge_alpha)
        self.num_input_nodes = int(num_input_nodes)
        self.num_output_nodes = int(num_output_nodes)
        self.io_separation_mode = io_separation_mode
        self.seed = int(seed)

        if self.num_input_nodes <= 0 or self.num_output_nodes <= 0:
            raise ValueError("num_input_nodes and num_output_nodes must be positive.")
        if self.num_input_nodes > self.N or self.num_output_nodes > self.N:
            raise ValueError("I/O subset sizes cannot exceed reservoir size.")
        if self.degree <= 0 or self.degree >= self.N:
            raise ValueError("degree must be in {1, 2, ..., N-1}.")

        # ------------------ build small-world reservoir ------------------
        rng = np.random.default_rng(self.seed)
        ws_graph = nx.watts_strogatz_graph(n=self.N, k=self.degree,
                                           p=self.rewiring_prob, seed=self.seed)
        A = nx.to_numpy_array(ws_graph, dtype=float)  # adjacency 0/1
        W_raw = (rng.uniform(-1.0, 1.0, size=A.shape)) * A
        self.W = scale_spectral_radius(W_raw, self.spectral_radius)

        # ------------------ choose I/O subsets (segregated) ------------------
        self.idx_in, self.idx_out = self._make_io_subsets(self.N,
                                                          self.num_input_nodes,
                                                          self.num_output_nodes,
                                                          self.io_separation_mode,
                                                          rng)
        # ------------------ input weights (only on input nodes) ------------------
        # Start with zeros, then fill rows for input subset.
        self.W_in = np.zeros((self.N, 3), dtype=float)
        W_in_rows = (rng.random((self.num_input_nodes, 3)) - 0.5) * 2.0 * self.input_scale
        self.W_in[self.idx_in, :] = W_in_rows

        # ------------------ state & readout ------------------
        self.reset_state()
        self.W_out = None   # shape (3, 2*|O| + 1)

        # Optional: quick operating-point sanity check in infinity norm
        self._operating_point_diagnostic()

    # ---------- I/O subsets ----------
    @staticmethod
    def _block_indices(center, half_width, N):
        """
        Return a contiguous block of size 2*half_width around 'center' on a ring of length N.
        """
        size = 2 * half_width
        if size <= 0:
            return np.array([], dtype=int)
        start = (center - half_width) % N
        return np.array([(start + i) % N for i in range(size)], dtype=int)

    def _make_io_subsets(self, N, m_in, m_out, sep_mode, rng):
        """
        Choose input and output node indices that are far apart on the ring.
        - If sep_mode == 'max': place blocks opposite each other (≈ N/2 apart).
        - If sep_mode is an int: center-to-center ring distance is that integer (mod N).
        We pick compact contiguous blocks (good proxy for spatial segregation).
        """
        # Make input block centered at c_in
        c_in = rng.integers(0, N)
        # Use contiguous blocks (even sizes preferred). If odd, we still make a block of size m.
        half_in = m_in // 2
        idx_in = self._block_indices(c_in, half_in, N)
        if idx_in.size < m_in:
            # add one more on the right for odd sizes
            extra = ((c_in + half_in) % N)
            idx_in = np.unique(np.concatenate([idx_in, np.array([extra])]))

        # Decide output center
        if sep_mode == "max":
            c_out = (c_in + N // 2) % N
        else:
            try:
                d = int(sep_mode)
            except Exception:
                d = N // 2
            c_out = (c_in + d) % N

        half_out = m_out // 2
        idx_out = self._block_indices(c_out, half_out, N)
        if idx_out.size < m_out:
            extra = ((c_out + half_out) % N)
            idx_out = np.unique(np.concatenate([idx_out, np.array([extra])]))

        # Ensure disjoint sets (if they accidentally overlap due to small N / large blocks)
        overlap = np.intersect1d(idx_in, idx_out, assume_unique=False)
        if overlap.size > 0:
            # shift output block until disjoint
            shift = 1
            while overlap.size > 0 and shift < N:
                idx_out = (idx_out + 1) % N
                overlap = np.intersect1d(idx_in, idx_out, assume_unique=False)
                shift += 1

        return idx_in.astype(int), idx_out.astype(int)

    # ---------- core dynamics ----------
    def reset_state(self):
        self.x = np.zeros(self.N, dtype=float)

    def _preact(self, u):
        # Recurrent + input; 'gain' multiplies the recurrent drive.
        return self.gain * (self.W @ self.x) + (self.W_in @ u)

    def _update(self, u):
        # Strict tanh for all nodes; leaky integrator
        pre = self._preact(u)
        x_new = np.tanh(pre)
        self.x = (1.0 - self.alpha) * self.x + self.alpha * x_new

    # ---------- features restricted to OUTPUT subset ----------
    def _phi_out(self):
        xo = self.x[self.idx_out]
        return np.concatenate([xo, xo * xo, np.array([1.0])], axis=0)  # (2*|O| + 1,)

    # ---------- data collection ----------
    def collect_states(self, inputs, discard=100, return_full=False):
        """
        Drive the reservoir and collect states (full) and features (out-subset).
        Returns:
            states_full[discard:], features_out[discard:]
        If return_full=False, 'states_full' is None.
        """
        self.reset_state()
        T = len(inputs)
        feats = np.zeros((T, 2 * len(self.idx_out) + 1), dtype=float)
        states = np.zeros((T, self.N), dtype=float) if return_full else None

        for t, u in enumerate(inputs):
            self._update(u)
            if return_full:
                states[t] = self.x
            feats[t] = self._phi_out()

        if return_full:
            return states[discard:], feats[discard:], states[:discard], feats[:discard]
        else:
            return None, feats[discard:], None, feats[:discard]

    # ---------- training ----------
    def fit_readout(self, train_input, train_target, discard=100):
        """
        Fit ridge regression readout mapping phi_out(x_t) -> y_t (3-D).
        train_input : (T, 3)
        train_target: (T, 3)
        """
        _, feats_out, _, _ = self.collect_states(train_input, discard=discard, return_full=False)
        targets = np.asarray(train_target, dtype=float)[discard:]

        if feats_out.shape[0] != targets.shape[0]:
            raise ValueError("Feature/target length mismatch after discard.")

        ridge = Ridge(alpha=self.ridge_alpha, fit_intercept=False)
        ridge.fit(feats_out, targets)  # coef_: (3, 2*|O|+1)
        self.W_out = ridge.coef_

    # ---------- inference ----------
    def predict_open_loop(self, inputs):
        """
        Open-loop (teacher-forced inputs). Returns (T, 3).
        """
        if self.W_out is None:
            raise RuntimeError("Call fit_readout() before predicting.")

        preds = []
        for u in inputs:
            self._update(u)
            phi = self._phi_out()
            preds.append(self.W_out @ phi)
        return np.asarray(preds, dtype=float)

    def predict_autoregressive(self, initial_input, num_steps):
        """
        Closed-loop / free-run: feed back the 3-D prediction as the next input.
        """
        if self.W_out is None:
            raise RuntimeError("Call fit_readout() before predicting.")

        preds = []
        current = np.asarray(initial_input, dtype=float).copy()
        for _ in range(int(num_steps)):
            self._update(current)
            phi = self._phi_out()
            out = self.W_out @ phi
            preds.append(out)
            current = out
        return np.asarray(preds, dtype=float)

    # ---------- diagnostics ----------
    def _operating_point_diagnostic(self):
        """
        Print a one-line diagnostic about a sufficient (very conservative) contraction condition
        in the infinity norm: (1-alpha) + alpha * gain * ||W||_∞ < 1.
        This is only a heuristic; ESP is more subtle, but it’s a useful quick check.
        """
        W_inf = np.max(np.sum(np.abs(self.W), axis=1))
        lhs = (1.0 - self.alpha) + self.alpha * self.gain * W_inf
        # Only print once per instance:
        self._diag_msg = f"[OP] (1-α) + α*gain*||W||_∞ = {lhs:.3f}  (target < 1.0 for a crude contraction bound)"
        # You can comment out the next line if you prefer silence:
        #print(self._diag_msg)

    # ---------- utilities ----------
    @property
    def input_indices(self):
        return self.idx_in.copy()

    @property
    def output_indices(self):
        return self.idx_out.copy()
