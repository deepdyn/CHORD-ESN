# CHORD-ESN: Cycle/Harmonic ORthogonal Decomposition Echo-State Networks

> **Topology-anchored long memory for reservoirs.**
> CHORD-ESN embeds the reservoir state as discrete differential forms on a 2-simplicial complex and uses a Hodge projector to separate fast (exact/coexact) transients from **slow harmonic** circulations. The slow channel’s capacity is governed by the first Betti number (b_1); stability follows from non-expansive DEC heat and a small block-contraction check (ESP).

---

## ✨ Highlights

* **Lawful geometry:** cross-degree couplings are *only* via (d_0,\delta_0,d_1,\delta_1) (discrete grad/div/curl/co-grad).
* **Hodge-projected 1-forms:** (y = y_{\text{ex}} + y_{\text{co}} + y_{\text{ha}}); apply split leaks with (\lambda_{\text{ha}}\ll \lambda_{\text{ex}},\lambda_{\text{co}}).
* **Nonexpansive smoothing:** per-degree heat steps (H_k(\tau_k)=I-\tau_k L_k) (SPSD Laplacians).
* **ESP by design:** small 3×3 block Lipschitz matrix (M) with (\rho(M){<}1) (Gershgorin-friendly row sums).
* **Linear-time updates:** sparse matvecs; Hodge projector amortized every (T_{\text{proj}}) steps.
* **Interpretable diagnostics:** residuals ((d_0x,\delta_0y,d_1y,\delta_1z)), **Harmonic Energy Fraction (HEF)**, ablation toggles.

---

## 🧠 Method (one-screen summary)

* **State:** (x_t\in C^0) (0-forms), (y_t\in C^1) (1-forms), (z_t\in C^2) (2-forms).
* **Innovations**
  [
  \begin{aligned}
  \xi^x_t&=\phi(W_xx_{t-1}+\delta_0 y_{t-1}+W^{(0)}*{\text{in}}u_t),\
  \xi^y_t&=\phi(W_yy*{t-1}+\alpha d_0x_{t-1}+\beta \delta_1z_{t-1}+W^{(1)}*{\text{in}}u_t),\
  \xi^z_t&=\phi(W_zz*{t-1}+\gamma d_1y_{t-1}+W^{(2)}_{\text{in}}u_t).
  \end{aligned}
  ]
* **Leaky + heat**
  [
  x_t=H_0(\tau_0)!\big((1-\lambda_0)x_{t-1}+\lambda_0\xi^x_t\big),;;
  y_t=H_1(\tau_1)!\big((1-\lambda_1)y_{t-1}+\lambda_1\xi^y_t\big),;;
  z_t=H_2(\tau_2)!\big((1-\lambda_2)z_{t-1}+\lambda_2\xi^z_t\big).
  ]
* **Hodge on (C^1)** (every (T_{\text{proj}}) steps):
  solve
  ((d_0^\top\star_1 d_0+\varepsilon\star_0)p=d_0^\top\star_1 y\Rightarrow y_{\text{ex}}=d_0p),
  ((d_1\star_1^{-1}d_1^\top+\varepsilon\star_2)q=d_1y\Rightarrow y_{\text{co}}=\star_1^{-1}d_1^\top q),
  set (y_{\text{ha}}=y-y_{\text{ex}}-y_{\text{co}}),
  then ( \tilde y = y^{\text{pre}}-\lambda_{\text{ex}}y_{\text{ex}}-\lambda_{\text{co}}y_{\text{co}}-\lambda_{\text{ha}}y_{\text{ha}}),
  and **heat**: (y_t=(I-\tau_1 L_1)\tilde y).

---

## 🔧 Installation

```bash
# Python >= 3.10
git clone https://github.com/deepdyn/CHORD-ESN && cd CHORD-ESN
python -m venv .venv && source .venv/bin/activate   # or conda env create ...
pip install -r requirements.txt
```

---

## 🧷 Notes & Gotchas

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
