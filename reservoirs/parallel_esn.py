import numpy as np
from sklearn.linear_model import Ridge
from utils.helpers import scale_spectral_radius

class ParallelESN:
    """
    Parallel Echo State Network (ParallelESN) where k reservoirs run in parallel.
    Each reservoir independently processes the input, and their outputs are combined for the final prediction.
    """

    def __init__(self,
                 n_reservoirs=3,
                 reservoir_size=100,
                 spectral_radius=0.95,
                 input_scale=1.0,
                 leaking_rate=1.0,
                 ridge_alpha=1e-6,
                 input_dim =3,
                 seed=42):
        """
        Parameters:
        - n_reservoirs: Number of parallel reservoirs.
        - reservoir_size: Number of neurons in each reservoir.
        - spectral_radius: Spectral radius for each reservoir's weight matrix.
        - input_scale: Scaling factor for input weights.
        - leaking_rate: Leaking rate for reservoir updates.
        - ridge_alpha: Regularization parameter for ridge regression.
        - seed: Random seed for reproducibility.
        """
        self.n_reservoirs = n_reservoirs
        self.reservoir_size = reservoir_size
        self.spectral_radius = spectral_radius
        self.input_scale = input_scale
        self.leaking_rate = leaking_rate
        self.ridge_alpha = ridge_alpha
        self.d_in = input_dim 
        self.seed = seed

        # Initialize reservoirs, input weights, and states for each reservoir
        self.reservoirs = []
        self.input_weights = []
        self.states = []

        np.random.seed(self.seed)
        for i in range(n_reservoirs):
            W = np.random.randn(reservoir_size, reservoir_size) * 0.1
            W = scale_spectral_radius(W, spectral_radius)
            self.reservoirs.append(W)

            W_in = (np.random.rand(reservoir_size, self.d_in) - 0.5) * 2.0 * input_scale
            self.input_weights.append(W_in)

            self.states.append(np.zeros(reservoir_size))

        # Output weights
        self.W_out = None

    def reset_state(self):
        """
        Reset the states of all reservoirs to zero.
        """
        self.states = [np.zeros(self.reservoir_size) for _ in range(self.n_reservoirs)]

    def _update_reservoir(self, reservoir_idx, u):
        """
        Update a single reservoir.
        """
        pre_activation = self.reservoirs[reservoir_idx] @ self.states[reservoir_idx]
        pre_activation += self.input_weights[reservoir_idx] @ u

        x_new = np.tanh(pre_activation)
        alpha = self.leaking_rate
        self.states[reservoir_idx] = (1.0 - alpha) * self.states[reservoir_idx] + alpha * x_new

    def collect_states(self, inputs, discard=100):
        """
        Collect states from all reservoirs using teacher forcing.
        """
        self.reset_state()
        all_states = []
        for u in inputs:
            combined_states = []
            for reservoir_idx in range(self.n_reservoirs):
                self._update_reservoir(reservoir_idx, u)
                combined_states.append(self.states[reservoir_idx])
            all_states.append(np.concatenate(combined_states))
        all_states = np.array(all_states)
        return all_states[discard:], all_states[:discard]

    def fit_readout(self, train_input, train_target, discard=100):
        """
        Train the readout layer using ridge regression.
        """
        states_use, _ = self.collect_states(train_input, discard=discard)
        targets_use = train_target[discard:]

        reg = Ridge(alpha=self.ridge_alpha, fit_intercept=False)
        reg.fit(states_use, targets_use)
        self.W_out = reg.coef_

    def predict(self, inputs):
        """
        Single-step-ahead inference on test data.
        """
        preds = []
        for u in inputs:
            combined_states = []
            for reservoir_idx in range(self.n_reservoirs):
                self._update_reservoir(reservoir_idx, u)
                combined_states.append(self.states[reservoir_idx])
            x_combined = np.concatenate(combined_states)
            out = self.W_out @ x_combined
            preds.append(out)
        return np.array(preds)

    def predict_autoregressive(self, initial_input, n_steps):
        """
        Fully autoregressive test: feed last output as next input.
        """
        preds = []
        current_in = np.array(initial_input)
        for _ in range(n_steps):
            combined_states = []
            for reservoir_idx in range(self.n_reservoirs):
                self._update_reservoir(reservoir_idx, current_in)
                combined_states.append(self.states[reservoir_idx])
            x_combined = np.concatenate(combined_states)
            out = self.W_out @ x_combined
            preds.append(out)
            current_in = out
        return np.array(preds)