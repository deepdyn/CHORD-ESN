import numpy as np
from sklearn.linear_model import Ridge


class CHORDESN:
    """

    The reservoir state is a triple of discrete differential forms:
      • 0-form  (nodes):   x_t ∈ R^N
      • 1-form  (edges):   y_t ∈ R^M
      • 2-form  (faces):   z_t ∈ R^Q
    with oriented coboundaries d0: C^0→C^1 and d1: C^1→C^2, codifferentials
    δ0 = star0^{-1} d0^T star1,  δ1 = star1^{-1} d1^T star2, and Laplacians
    L0 = δ0 d0, L1 = d0 δ0 + δ1 d1, L2 = d1 δ1.

    Updates (per step):
      ξ^x_t = tanh( W_x x_{t-1} + δ0 y_{t-1} + W_in0 u_t )
      x_t   = (I − heat0 L0)[ (1−λ0) x_{t-1} + λ0 ξ^x_t ]

      ξ^y_t = tanh( W_y y_{t-1} + α d0 x_{t-1} + β δ1 z_{t-1} + W_in1 u_t )
      y_t   = y_{t-1}
              − leak_ex * y_exact(t−1) − leak_co * y_coex(t−1) − leak_ha * y_harm(t−1)
              − heat1 * L1 y_{t-1}
              + λ1 ξ^y_t
        (Hodge decomposition y = y_exact + y_coex + y_harm computed with SPD CG)

      ξ^z_t = tanh( W_z z_{t-1} + γ d1 y_{t-1} + W_in2 u_t )
      z_t   = (I − heat2 L2)[ (1−λ2) z_{t-1} + λ2 ξ^z_t ]

    Only the read-out W_out is trained (ridge regression).

    Parameters
    ----------
    num_nodes : int = 300
        Number of vertices N.
    input_dim : int = 3
        External input dimension.
    # Topology (2-complex)
    edges : list[tuple[int,int]] | None = None
        Optional undirected edge list (i, j) with i != j. If None, an ER graph
        with avg_degree is sampled.
    faces : list[tuple[int,int,int]] | None = None
        Optional list of oriented triangular faces (i, j, k) with boundary
        i→j, j→k, k→i. If None, Q=0 (graph only).
    avg_degree : int = 8
        Used only when edges=None (ER graph).
    # Hodge stars (diagonal, SPD)
    star0_diag : np.ndarray | None = None
    star1_diag : np.ndarray | None = None
    star2_diag : np.ndarray | None = None
        Optional positive diagonal entries for Hodge stars on nodes/edges/faces.
        If None, all-ones are used.
    # Leaks / heat
    lam_node : float = 0.25
    lam_edge : float = 0.25
    lam_face : float = 0.25
    leak_exact : float = 0.40
    leak_coexact : float = 0.40
    leak_harm : float = 0.05
    heat0 : float = 0.05
    heat1 : float = 0.05
    heat2 : float = 0.05
    # Cross-degree couplings
    alpha : float = 0.3   # d0 x → 1-forms
    beta  : float = 0.3   # δ1 z → 1-forms
    gamma : float = 0.3   # d1 y → 2-forms
    # Local mixers (diagonal for speed)
    node_mix_gain : float = 0.3
    edge_mix_gain : float = 0.3
    face_mix_gain : float = 0.3
    # Inputs
    input_scale_node : float = 0.5
    input_scale_edge : float = 0.5
    input_scale_face : float = 0.5
    # Projector (Hodge decomposition) controls
    projector_eps : float = 1e-4
    cg_tol : float = 1e-6
    cg_maxiter : int = 200
    proj_every : int = 5         # recompute y decomposition every k steps
    # Read-out
    ridge_alpha : float = 1e-6
    use_poly : bool = True
    feature_mode : {'node','node_harm','full'} = 'node_harm'
        'node'       → x
        'node_harm'  → [ x ; y_harm ]
        'full'       → [ x ; y_harm ; z ]
    seed : int = 42
    """

    # -----------------------------------------------------------------
    def __init__(
        self,
        num_nodes: int = 300,
        input_dim: int = 3,
        edges: list | None = None,
        faces: list | None = None,
        avg_degree: int = 8,
        star0_diag: np.ndarray | None = None,
        star1_diag: np.ndarray | None = None,
        star2_diag: np.ndarray | None = None,
        lam_node: float = 0.25,
        lam_edge: float = 0.25,
        lam_face: float = 0.25,
        leak_exact: float = 0.40,
        leak_coexact: float = 0.40,
        leak_harm: float = 0.05,
        heat0: float = 0.05,
        heat1: float = 0.05,
        heat2: float = 0.05,
        alpha: float = 0.3,
        beta: float = 0.3,
        gamma: float = 0.3,
        node_mix_gain: float = 0.3,
        edge_mix_gain: float = 0.3,
        face_mix_gain: float = 0.3,
        input_scale_node: float = 0.5,
        input_scale_edge: float = 0.5,
        input_scale_face: float = 0.5,
        projector_eps: float = 1e-4,
        cg_tol: float = 1e-6,
        cg_maxiter: int = 200,
        proj_every: int = 5,
        ridge_alpha: float = 1e-6,
        use_poly: bool = True,
        feature_mode: str = "node_harm",
        seed: int = 42,
    ):
        if num_nodes < 2:
            raise ValueError("num_nodes must be ≥ 2")
        for nm in (lam_node, lam_edge, lam_face):
            if not 0 < nm <= 1:
                raise ValueError("lam_* must be in (0,1]")
        if not (0 <= leak_harm <= min(leak_exact, leak_coexact) < 1):
            raise ValueError("Require 0 ≤ leak_harm ≤ min(leak_exact, leak_coexact) < 1")
        if feature_mode not in {"node", "node_harm", "full"}:
            raise ValueError("feature_mode must be 'node', 'node_harm', or 'full'")

        self.N = int(num_nodes)
        self.d_in = int(input_dim)
        self.avg_degree = int(avg_degree)

        # Hyperparameters
        self.lam0 = float(lam_node)
        self.lam1 = float(lam_edge)
        self.lam2 = float(lam_face)
        self.leak_ex = float(leak_exact)
        self.leak_co = float(leak_coexact)
        self.leak_ha = float(leak_harm)
        self.heat0 = float(heat0)
        self.heat1 = float(heat1)
        self.heat2 = float(heat2)

        self.alpha = float(alpha)
        self.beta = float(beta)
        self.gamma = float(gamma)

        self.node_mix_gain = float(node_mix_gain)
        self.edge_mix_gain = float(edge_mix_gain)
        self.face_mix_gain = float(face_mix_gain)

        self.in_scale_node = float(input_scale_node)
        self.in_scale_edge = float(input_scale_edge)
        self.in_scale_face = float(input_scale_face)

        self.eps_proj = float(projector_eps)
        self.cg_tol = float(cg_tol)
        self.cg_maxiter = int(cg_maxiter)
        self.proj_every = int(proj_every)

        self.ridge_alpha = float(ridge_alpha)
        self.use_poly = bool(use_poly)
        self.feature_mode = feature_mode
        self.seed = int(seed)

        rng = np.random.default_rng(self.seed)

        # -------------------- Build 1-skeleton (edges) --------------------
        if edges is None:
            p = min(1.0, max(0.0, self.avg_degree / max(1, self.N - 1)))
            mask_upper = rng.random((self.N, self.N)) < p
            mask_upper = np.triu(mask_upper, k=1)
            ii, jj = np.where(mask_upper)
            u = ii.astype(np.int32)
            v = jj.astype(np.int32)
        else:
            es = {(min(i, j), max(i, j)) for (i, j) in edges if i != j}
            if not es:
                raise ValueError("Provided edges are empty.")
            u = np.array([e[0] for e in sorted(es)], dtype=np.int32)
            v = np.array([e[1] for e in sorted(es)], dtype=np.int32)

        self.M = len(u)
        if self.M == 0:
            raise ValueError("Graph has no edges; increase avg_degree or provide edges.")

        # Canonical undirected orientation: (edge_u[m] -> edge_v[m]) with u<v
        self.edge_u = np.minimum(u, v).astype(np.int32)
        self.edge_v = np.maximum(u, v).astype(np.int32)

        # Index lookup (min,max) -> m
        idx_map = -np.ones((self.N, self.N), dtype=np.int32)
        for m, (a, b) in enumerate(zip(self.edge_u, self.edge_v)):
            idx_map[a, b] = m
        self._edge_idx_map = idx_map

        # -------------------- Build faces (2-simplices) --------------------
        if faces is None:
            self.Q = 0
            self.face_edges = np.zeros((0, 3), dtype=np.int32)
            self.face_signs = np.zeros((0, 3), dtype=np.int8)
        else:
            F = []
            S = []
            for (i, j, k) in faces:
                i, j, k = int(i), int(j), int(k)
                trip = [(i, j), (j, k), (k, i)]
                e_idx = []
                e_sgn = []
                ok = True
                for a, b in trip:
                    a0, b0 = (a, b) if a < b else (b, a)
                    m = self._edge_idx_map[a0, b0]
                    if m < 0:
                        ok = False
                        break
                    e_idx.append(int(m))
                    e_sgn.append(+1 if a < b else -1)  # +1 if along canonical u<v
                if ok:
                    F.append(e_idx)
                    S.append(e_sgn)
            self.face_edges = np.asarray(F, dtype=np.int32) if F else np.zeros((0, 3), dtype=np.int32)
            self.face_signs = np.asarray(S, dtype=np.int8) if S else np.zeros((0, 3), dtype=np.int8)
            self.Q = int(self.face_edges.shape[0])

        # Edge→faces adjacency for d1^T
        self._edge_to_faces_idx = [[] for _ in range(self.M)]
        self._edge_to_faces_sgn = [[] for _ in range(self.M)]
        for f in range(self.Q):
            for k in range(3):
                e = int(self.face_edges[f, k])
                s = int(self.face_signs[f, k])
                self._edge_to_faces_idx[e].append(f)
                self._edge_to_faces_sgn[e].append(s)

        # -------------------- Hodge stars (diagonal, SPD) --------------------
        self.star0 = (np.ones(self.N, dtype=np.float32) if star0_diag is None
                      else np.asarray(star0_diag, dtype=np.float32).copy())
        self.star1 = (np.ones(self.M, dtype=np.float32) if star1_diag is None
                      else np.asarray(star1_diag, dtype=np.float32).copy())
        self.star2 = (np.ones(self.Q, dtype=np.float32) if star2_diag is None
                      else np.asarray(star2_diag, dtype=np.float32).copy())

        if np.any(self.star0 <= 0) or np.any(self.star1 <= 0) or (self.Q and np.any(self.star2 <= 0)):
            raise ValueError("All Hodge star diagonals must be positive.")

        self.star0_inv = 1.0 / self.star0
        self.star1_inv = 1.0 / self.star1
        self.star2_inv = (1.0 / self.star2) if self.Q else np.zeros(0, dtype=np.float32)

        # -------------------- Local mixers (diagonal for speed) --------------------
        self.w_x = rng.uniform(-self.node_mix_gain, self.node_mix_gain, size=self.N).astype(np.float32)
        self.w_y = rng.uniform(-self.edge_mix_gain, self.edge_mix_gain, size=self.M).astype(np.float32)
        self.w_z = rng.uniform(-self.face_mix_gain, self.face_mix_gain, size=self.Q).astype(np.float32) if self.Q else np.zeros(0, dtype=np.float32)

        # -------------------- Input maps --------------------
        self.W_in0 = (rng.uniform(-1.0, 1.0, size=(self.N, self.d_in)) * self.in_scale_node).astype(np.float32)
        self.W_in1 = (rng.uniform(-1.0, 1.0, size=(self.M, self.d_in)) * self.in_scale_edge).astype(np.float32)
        self.W_in2 = ((rng.uniform(-1.0, 1.0, size=(self.Q, self.d_in)) * self.in_scale_face).astype(np.float32)
                      if self.Q else np.zeros((0, self.d_in), dtype=np.float32))

        # -------------------- Runtime state --------------------
        self.x = np.zeros(self.N, dtype=np.float32)
        self.y = np.zeros(self.M, dtype=np.float32)
        self.z = np.zeros(self.Q, dtype=np.float32) if self.Q else np.zeros(0, dtype=np.float32)

        # cache for Hodge decomposition of y
        self._t = 0
        self._y_exact = np.zeros_like(self.y)
        self._y_coex = np.zeros_like(self.y)
        self._y_harm = np.zeros_like(self.y)
        self._cache_step = -1  # last step when cache was refreshed

        # trained read-out
        self.W_out: np.ndarray | None = None

    # ======================== DEC core operators ========================
    # d0 @ x : edges <- nodes
    def _apply_d0(self, x_vec: np.ndarray) -> np.ndarray:
        return (x_vec[self.edge_v] - x_vec[self.edge_u]).astype(np.float32)

    # d0^T @ y : nodes <- edges
    def _apply_d0T(self, y_vec: np.ndarray) -> np.ndarray:
        out = np.zeros(self.N, dtype=np.float32)
        np.add.at(out, self.edge_u, -y_vec)
        np.add.at(out, self.edge_v, +y_vec)
        return out

    # d1 @ y : faces <- edges
    def _apply_d1(self, y_vec: np.ndarray) -> np.ndarray:
        if self.Q == 0:
            return np.zeros(0, dtype=np.float32)
        # y_faces[f] = sum_k sign[f,k] * y[edge_idx[f,k]]
        return np.sum(self.face_signs * y_vec[self.face_edges], axis=1).astype(np.float32)

    # d1^T @ z : edges <- faces
    def _apply_d1T(self, z_vec: np.ndarray) -> np.ndarray:
        if self.Q == 0:
            return np.zeros(self.M, dtype=np.float32)
        out = np.zeros(self.M, dtype=np.float32)
        # Accumulate contributions from each face-edge
        for k in range(3):
            np.add.at(out, self.face_edges[:, k], self.face_signs[:, k] * z_vec)
        return out

    # δ0 y = star0^{-1} d0^T star1 y
    def _apply_delta0(self, y_vec: np.ndarray) -> np.ndarray:
        tmp = self.star1 * y_vec
        return (self.star0_inv * self._apply_d0T(tmp)).astype(np.float32)

    # δ1 z = star1^{-1} d1^T star2 z
    def _apply_delta1(self, z_vec: np.ndarray) -> np.ndarray:
        if self.Q == 0:
            return np.zeros(self.M, dtype=np.float32)
        tmp = self.star2 * z_vec
        return (self.star1_inv * self._apply_d1T(tmp)).astype(np.float32)

    # Laplacians
    def _apply_L0(self, x_vec: np.ndarray) -> np.ndarray:
        return self._apply_delta0(self._apply_d0(x_vec))

    def _apply_L1(self, y_vec: np.ndarray) -> np.ndarray:
        return (self._apply_d0(self._apply_delta0(y_vec)) +
                self._apply_delta1(self._apply_d1(y_vec))).astype(np.float32)

    def _apply_L2(self, z_vec: np.ndarray) -> np.ndarray:
        if self.Q == 0:
            return np.zeros(0, dtype=np.float32)
        return self._apply_d1(self._apply_delta1(z_vec))

    # ======================== CG solver utilities =======================
    @staticmethod
    def _dot(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.dot(a.ravel().astype(np.float64), b.ravel().astype(np.float64)))

    def _cg(self, apply_A, b: np.ndarray, tol: float, maxiter: int) -> np.ndarray:
        x = np.zeros_like(b, dtype=np.float32)
        r = b - apply_A(x)
        p = r.copy()
        rsold = self._dot(r, r)
        if rsold == 0.0:
            return x
        bnorm = np.sqrt(rsold)
        for _ in range(maxiter):
            Ap = apply_A(p)
            denom = self._dot(p, Ap) + 1e-12
            alpha = rsold / denom
            x = (x + alpha * p).astype(np.float32)
            r = (r - alpha * Ap).astype(np.float32)
            rsnew = self._dot(r, r)
            if np.sqrt(rsnew) <= max(tol * bnorm, 1e-12):
                break
            p = (r + (rsnew / (rsold + 1e-12)) * p).astype(np.float32)
            rsold = rsnew
        return x

    # ================== Hodge decomposition of y (cached) =================
    def _decompose_y(self, y_vec: np.ndarray):
        """
        Compute y_exact, y_coex, y_harm with Tikhonov-regularized projectors:

          Solve (d0^T star1 d0 + ε star0) p = d0^T star1 y,   y_exact = d0 p
          Solve (d1 star1^{-1} d1^T + ε star2) q = d1 y,      y_coex  = star1^{-1} d1^T q
          y_harm = y − y_exact − y_coex
        """
        # Exact component
        rhs_p = self._apply_d0T(self.star1 * y_vec)  # nodes
        def apply_Ap(p):
            return (self._apply_d0T(self.star1 * self._apply_d0(p)) + self.star0 * self.eps_proj * p)
        p = self._cg(apply_Ap, rhs_p, tol=self.cg_tol, maxiter=self.cg_maxiter)
        y_exact = self._apply_d0(p)

        # Coexact component
        if self.Q > 0:
            rhs_q = self._apply_d1(y_vec)  # faces
            def apply_Aq(q):
                return (self._apply_d1(self.star1_inv * self._apply_d1T(q)) + self.star2 * self.eps_proj * q)
            q = self._cg(apply_Aq, rhs_q, tol=self.cg_tol, maxiter=self.cg_maxiter)
            y_coex = (self.star1_inv * self._apply_d1T(q)).astype(np.float32)
        else:
            y_coex = np.zeros_like(y_vec)

        y_harm = (y_vec - y_exact - y_coex).astype(np.float32)
        return y_exact.astype(np.float32), y_coex.astype(np.float32), y_harm

    def _ensure_y_decomp(self):
        if (self._t % self.proj_every == 0) or (self._cache_step != self._t):
            self._y_exact, self._y_coex, self._y_harm = self._decompose_y(self.y)
            self._cache_step = self._t

    # ======================== Feature construction =======================
    def _features_from_state(self) -> np.ndarray:
        feats = [self.x]
        if self.feature_mode in {"node_harm", "full"}:
            self._ensure_y_decomp()
            feats.append(self._y_harm)
        if self.feature_mode == "full":
            feats.append(self.z if self.Q else np.zeros(0, dtype=np.float32))
        feat = np.concatenate(feats, axis=0).astype(np.float32)
        if self.use_poly:
            feat = np.concatenate([feat, feat * feat, [1.0]], axis=0).astype(np.float32)
        return feat

    # ============================ One step ===============================
    def _step(self, u_t: np.ndarray):
        """
        One DEC-ESN update with input u_t (shape [d_in]).
        Order: node innovation+heat → edge innovation+decomp+leaks+heat → face innovation+heat.
        """
        X_prev = self.x
        Y_prev = self.y
        Z_prev = self.z

        # ---- Node channel
        pre_x = (self.w_x * X_prev) + self._apply_delta0(Y_prev) + (self.W_in0 @ u_t)
        xi_x = np.tanh(pre_x).astype(np.float32)
        x_tmp = (1.0 - self.lam0) * X_prev + self.lam0 * xi_x
        X_new = (x_tmp - self.heat0 * self._apply_L0(x_tmp)).astype(np.float32)

        # ---- Edge channel
        term_d0x = self.alpha * self._apply_d0(X_prev)
        term_d1z = self.beta * self._apply_delta1(Z_prev) if self.Q else 0.0
        pre_y = (self.w_y * Y_prev) + term_d0x + term_d1z + (self.W_in1 @ u_t)
        xi_y  = np.tanh(pre_y).astype(np.float32)
        
        y_pre = ((1.0 - self.lam1) * Y_prev + self.lam1 * xi_y).astype(np.float32)
        
        # Hodge decomposition ON y_pre (with cadence); cache components from y_pre
        if (self._t % self.proj_every == 0) or (self._cache_step != self._t):
            y_ex, y_co, y_ha = self._decompose_y(y_pre)
            self._y_exact, self._y_coex, self._y_harm = y_ex, y_co, y_ha
            self._cache_step = self._t
        else:
            y_ex, y_co, y_ha = self._y_exact, self._y_coex, self._y_harm
        
        # Split-leak ON THE COMPONENTS OF y_pre
        y_tilde = (y_pre
                   - self.leak_ex * y_ex
                   - self.leak_co * y_co
                   - self.leak_ha * y_ha).astype(np.float32)
        
        # Heat on the leaked signal: y_t = (I - heat1 L1) y_tilde
        Y_new = (y_tilde - self.heat1 * self._apply_L1(y_tilde)).astype(np.float32)

        # ---- Face channel
        if self.Q:
            pre_z = (self.w_z * Z_prev) + (self.gamma * self._apply_d1(Y_prev)) + (self.W_in2 @ u_t)
            xi_z = np.tanh(pre_z).astype(np.float32)
            z_tmp = (1.0 - self.lam2) * Z_prev + self.lam2 * xi_z
            Z_new = (z_tmp - self.heat2 * self._apply_L2(z_tmp)).astype(np.float32)
        else:
            Z_new = Z_prev  # no faces

        # Commit & advance time
        self.x, self.y, self.z = X_new, Y_new, Z_new
        self._t += 1  # advance AFTER using cached decomposition

    # ============================ API ===================================
    def reset_state(self):
        self.x.fill(0.0)
        self.y.fill(0.0)
        if self.Q:
            self.z.fill(0.0)
        self._t = 0
        self._cache_step = -1

    # Teacher-forced read-out fitting
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
            self._step(inputs[t].astype(np.float32))
            if t >= discard:
                feats.append(self._features_from_state())

        X_feat = np.asarray(feats, dtype=np.float32)  # [T-d, F]
        Y = targets[discard:]                         # [T-d, d_out]

        reg = Ridge(alpha=self.ridge_alpha, fit_intercept=False)
        reg.fit(X_feat, Y)
        self.W_out = reg.coef_.astype(np.float32)

    # Closed-loop autoregressive prediction
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
            # feedback first d_in outputs as next input (common ESN practice)
            current_u = y_t[: self.d_in]

        return preds

    # Open-loop (teacher-forced) prediction
    def predict_open_loop(self, test_input: np.ndarray) -> np.ndarray:
        """
        Run the reservoir in open-loop mode over a provided input sequence and
        produce outputs using the trained read-out. No feedback loop.

        Parameters
        ----------
        test_input : np.ndarray [T, d_in]

        Returns
        -------
        np.ndarray [T, d_out]
        """
        if self.W_out is None:
            raise RuntimeError("Call fit_readout() before prediction")
        if test_input.ndim != 2 or test_input.shape[1] != self.d_in:
            raise ValueError(f"test_input must have shape [T, {self.d_in}]")

        T = test_input.shape[0]
        d_out = self.W_out.shape[0]
        preds = np.empty((T, d_out), dtype=np.float32)

        #self.reset_state()
        for t in range(T):
            self._step(test_input[t].astype(np.float32))
            feat_vec = self._features_from_state()
            preds[t] = (self.W_out @ feat_vec).astype(np.float32)

        return preds



