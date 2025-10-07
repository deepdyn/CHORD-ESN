# Geometric Reservoirs

A curated playground for **reservoir-computing architectures** and their
dynamical analysis. Everything is pure NumPy + Scikit-learn (except for gradient based baselines). One can clone, hack, and run the whole repo inside a vanilla Jupyter install.


## ✨ Key Features

* **15 + reservoir classes**

  | Category | Files |
  |----------|-------|
  | Standard / baseline | `cycle.py`, `sparse.py`, `deep_esn.py`, `parallel_esn.py` |
  | Biological | `bei_stp.py`, `short_term_plasticity.py`, `glia_neuron.py` |
  | Geometric | `euclidean.py`, `spherical.py`, `hyper.py` |
  | Others | `mci_esn.py`, `swirl_gated_multicycle.py`, `resonator.py`, … |

* **Analysis utilities**
  * eigen-spectrum plots  
  * linear memory-capacity curves  
  * participation ratio (effective dimensionality)  
  * largest Lyapunov exponent 

* **Reproducible notebooks** in `notebooks/`
* Figures auto-saved in **`figures/`** at user-chosen dpi
* Single-command install: `pip install -r requirements.txt`


## 📂 Repository Layout

```
RESERVOIR-EXPERIMENTS/
├── figures/                 ← output PNGs land here
├── notebooks/
│   └── main.ipynb           ← run all demos
├── reservoirs/
│   ├── cycle.py
│   ├── bei\_stp.py
│   ├── spherical.py
│   ├── hyper.py
│   └── …                    ← many more
├── utils/
│   ├── dynamics.py          ← spectrum, MC, PR, Lyapunov
│   ├── plotting.py          ← nice 2-D / 3-D plots
│   ├── metrics.py           ← NRMSE, VPT, ADev …
│   └── helpers.py           ← misc. math helpers
└── requirements.txt
```

## 🚀 Quick-start

git clone https://github.com/<your-handle>/Reservoir-Experiments.git
cd Reservoir-Experiments
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

jupyter lab   # open notebooks/main.ipynb


`main.ipynb` demonstrates:

1. Training & multi-step prediction on Lorenz-63
2. Batch benchmarking across reservoirs
3. Eigen-spectrum, memory-capacity and participation-ratio plots
   (saved under `figures/`)


## 🛠️ Add Your Own Reservoir

1. Drop `<name>.py` into `reservoirs/` exposing a NumPy weight matrix in
   one of the usual attributes (`W`, `W_res`, `W_base`, `W_nn`, `W0`, …).
2. Provide `_step(u)` **or** `_update(u)` and `reset_state()`.
3. Import it in a notebook; all analysis helpers work out-of-the-box.


## 📊 Dynamical Diagnostics (`utils/dynamics.py`)

| Function                         | Purpose                           | Call                                           |
| -------------------------------- | --------------------------------- | ---------------------------------------------- |
| `plot_eigen_spectrum`            | Scatter eigenvalues + unit circle | `plot_eigen_spectrum(res)`                     |
| `compute_memory_capacity`        | MC spectrum + total MC            | `delays, C, MC = compute_memory_capacity(res)` |
| `plot_memory_capacity`           | Save/plot MC curve                | `plot_memory_capacity(delays, C)`              |
| `estimate_participation_ratio`   | Effective dimensionality          | `pr, eig = estimate_participation_ratio(res)`  |
| `estimate_lyapunov`              | Largest Lyapunov exponent         | `lyap = estimate_lyapunov(res)`                |


## 📑 Citation

If you use this code in academic work, please cite the original
architectures (see docstrings) **and** this repository:

@misc{geometric-reservoirs2025,
  author = {Singh, Pradeep},
  title  = {Geometric Reservoirs: a geometric & bio-inspired ESN toolbox},
  year   = {2025},
  url    = {https://github.com/deepdyn/geometric-reservoirs}
}


## 📝 License

Distributed under the Apache License – see `LICENSE` for details.

Enjoy exploring strange attractors in weird geometries! 🎢


