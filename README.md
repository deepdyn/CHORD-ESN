# CHORD-ESN: Cycle/Harmonic ORthogonal Decomposition Echo-State Networks

> **Topology-anchored long memory for reservoirs.**
> CHORD-ESN embeds the reservoir state as discrete differential forms on a 2-simplicial complex and uses a Hodge projector to separate fast (exact/coexact) transients from **slow harmonic** circulations. The slow channel’s capacity is governed by the first Betti number (b_1); stability follows from non-expansive DEC heat and a small block-contraction check (ESP).

---

## ✨ Highlights

* **Lawful geometry:** cross-degree couplings are *only* via `d0`, `delta0`, `d1`, `delta1` (discrete grad/div/curl/co-grad).
* **Hodge-projected 1-forms:** `y = y_ex + y_co + y_ha`; apply split leaks with `lambda_ha << lambda_ex, lambda_co`.
* **Nonexpansive smoothing:** per-degree heat steps `Hk(tau_k) = I − tau_k * Lk` (SPSD Laplacians).
* **ESP by design:** small 3×3 block Lipschitz matrix `M` with `rho(M) < 1` (Gershgorin-friendly row sums).
* **Linear-time updates:** sparse matvecs; Hodge projector amortized every `T_proj` steps.
* **Interpretable diagnostics:** residuals `d0 x`, `delta0 y`, `d1 y`, `delta1 z`; **Harmonic Energy Fraction (HEF)**; ablation toggles.

---

## 🧠 Method (one-screen summary)

**State:**
`x_t ∈ C^0` (0-forms), `y_t ∈ C^1` (1-forms), `z_t ∈ C^2` (2-forms)

**Innovations**

```text
xi_x(t) = phi( Wx x(t-1) + delta0 y(t-1) + Win0 u(t) )
xi_y(t) = phi( Wy y(t-1) + alpha d0 x(t-1) + beta delta1 z(t-1) + Win1 u(t) )
xi_z(t) = phi( Wz z(t-1) + gamma d1 y(t-1) + Win2 u(t) )
```

**Leaky + heat**

```text
x(t) = H0(tau0) * [ (1 - lambda0) x(t-1) + lambda0 * xi_x(t) ]
y(t) = H1(tau1) * [ (1 - lambda1) y(t-1) + lambda1 * xi_y(t) ]
z(t) = H2(tau2) * [ (1 - lambda2) z(t-1) + lambda2 * xi_z(t) ]

where: Hk(tau) = I - tau * Lk  (nonexpansive because Lk ⪰ 0)
```

**Hodge on `C^1`** (every `T_proj` steps)

```text
# Project y into exact / coexact / harmonic components
# (ε > 0 is a small Tikhonov regularizer; stars are SPD "mass" matrices)

Solve: (d0^T * star1 * d0 + ε * star0) p = d0^T * star1 * y
=> y_ex = d0 * p

Solve: (d1 * star1^{-1} * d1^T + ε * star2) q = d1 * y
=> y_co = star1^{-1} * d1^T * q

y_ha = y - y_ex - y_co

# Split-leak and heat
y_tilde = y_pre
          - lambda_ex * y_ex
          - lambda_co * y_co
          - lambda_ha * y_ha

y(t) = (I - tau1 * L1) * y_tilde
```

**Notes**

* Use a 1-Lipschitz `phi` (e.g., `tanh` or clipped ReLU).
* Cross-degree couplings are restricted to `d0`, `delta0`, `d1`, `delta1` only (no arbitrary dense mixing).
* Keep row sums of `M` strictly `< 1` for an ESP-safe configuration.

---

## 🔧 Installation

```bash
# Python >= 3.10
git clone https://github.com/deepdyn/CHORD-ESN && cd CHORD-ESN
python -m venv .venv && source .venv/bin/activate   # or conda env create ...
pip install -r requirements.txt
```

---

## 🧷 Notes

* **Order matters (C¹):** leaky → Hodge split → split-leak → **heat**.
* **Lawful couplings only:** no arbitrary dense cross-degree matrices.
* **Nonlinearity:** use a 1-Lipschitz activation (`tanh`, clipped ReLU).
* **HEF window:** compute over the valid prediction window (H^\star) (same as VPT(^\star)).

---

## 📚 Citation

If you find CHORD-ESN useful, please cite:

```bibtex
@article{singh2025CHORDESN,
  title   = {Hodge-Projected Echo-State Networks with Topologically Anchored Memory for Chaotic Flows},
  author  = {P. Singh, O. R. Madare, R. Balasubramanian},
  journal = {},
  year    = {2025},
  note    = {Code: CHORD-ESN}
}
```

---

## 🛡️ License

This project is released under the **MIT License**. See `LICENSE`.

---

## 🙏 Acknowledgements

We build on discrete exterior calculus (Hirani; Desbrun et al.), Hodge/graph signal processing (Barbarossa & Sardellitti; Schaub et al.), and the reservoir computing literature (Jaeger; Maass; Lukoševičius & Jaeger). 

---

## 📫 Questions / Issues

Please open a GitHub issue with:

* dataset + config,
* minimal command to reproduce,
* environment info (`python -V`; `pip freeze`),
* logs/tracebacks.

We’re happy to help!
