import numpy as np
from sklearn.linear_model import Ridge
from utils.helpers import scale_spectral_radius


class DeepESN3D:
    """
    Deep Echo State Network (DeepESN) for multi-layered reservoir computing.
    Each layer has its own reservoir, and the states are propagated through layers.
    """

    def __init__(self,
                 num_layers=3,
                 reservoir_size=100,
                 spectral_radius=0.95,
                 input_scale=1.0,
                 leaking_rate=1.0,
                 input_dim =3,
                 ridge_alpha=1e-6,
                 seed=42):
        """
        Parameters:
        - num_layers: Number of reservoir layers.
        - reservoir_size: Number of neurons in each reservoir layer.
        """
        self.num_layers = num_layers
        self.reservoir_size = reservoir_size
        self.spectral_radius = spectral_radius
        self.input_scale = input_scale
        self.leaking_rate = leaking_rate
        self.d_in = input_dim 
        self.ridge_alpha = ridge_alpha
        self.seed = seed

        # Initialize reservoirs and input weights for each layer
        self.reservoirs = []
        self.input_weights = []
        self.states = []

        np.random.seed(self.seed)
        for layer in range(num_layers):
            np.random.seed(seed + layer)
            W = np.random.randn(reservoir_size, reservoir_size) * 0.1
            W = scale_spectral_radius(W, spectral_radius)
            self.reservoirs.append(W)

            if layer == 0 :
                W_in = (np.random.rand(reservoir_size, self.d_in) - 0.5) * 2.0 * input_scale
            else:
                W_in = (np.random.rand(reservoir_size, reservoir_size) - 0.5) * 2.0 * input_scale
            self.input_weights.append(W_in)

        self.W_out = None
        self.reset_state()

    def reset_state(self):
        """
        Reset the states of all reservoir layers.
        """
        self.states = [np.zeros(self.reservoir_size) for _ in range(self.num_layers)]

    def _update_layer(self, layer_idx, u):
        """
        Update a single reservoir layer.
        """
        pre_activation = self.reservoirs[layer_idx] @ self.states[layer_idx]
        if layer_idx == 0:
            pre_activation += self.input_weights[layer_idx] @ u
        else:
            pre_activation += self.input_weights[layer_idx] @ self.states[layer_idx - 1]

        x_new = np.tanh(pre_activation)
        alpha = self.leaking_rate
        self.states[layer_idx] = (1.0 - alpha) * self.states[layer_idx] + alpha * x_new

    def collect_states(self, inputs, discard=100):
        self.reset_state()
        all_states = []
        for u in inputs:
            for layer_idx in range(self.num_layers):
                self._update_layer(layer_idx, u)
            all_states.append(np.concatenate(self.states))
        all_states = np.array(all_states)
        return all_states[discard:], all_states[:discard]

    def fit_readout(self, train_input, train_target, discard=100):
        """
        Train the readout layer using ridge regression.
        """
        states_use, _ = self.collect_states(train_input, discard=discard)
        targets_use = train_target[discard:]

        # Augment states with bias
        # X_aug = np.hstack([states_use, np.ones((states_use.shape[0], 1))])  # shape [T-discard, N*L+1]

        # Quadratic readout
        # Build augmented matrix [ x, x^2, 1 ]
        X_list = []
        for s in states_use:
            X_list.append( np.concatenate([s]) )
            # X_list.append( np.concatenate([s, s**2, [1.0]]) )
        X_aug = np.array(X_list)                                    # shape [T-discard, 2N*L+1]

        reg = Ridge(alpha=self.ridge_alpha, fit_intercept=False)
        reg.fit(X_aug, targets_use)
        self.W_out = reg.coef_

    def predict(self, inputs):
        """
        Single-step-ahead inference on test data.
        """
        preds = []
        for u in inputs:
            for layer_idx in range(self.num_layers):
                self._update_layer(layer_idx, u)
            state = np.concatenate(self.states)
            x_aug = np.concatenate([state])
            # x_aug = np.concatenate([state, (state)**2, [1.0]])  # For quadrartic readout
            out = self.W_out @ x_aug
            preds.append(out)
        return np.array(preds)

    def predict_open_loop(self, test_input):
        """
        Open-loop (teacher-forced) prediction:
        feed the TRUE input sequence and emit readout at each step.
        This mirrors `predict(...)` and does NOT reset state.
        """
        if self.W_out is None:
            raise RuntimeError("Call fit_readout() before prediction.")
        return self.predict(test_input)

    def predict_autoregressive(self, initial_input, num_steps):
        """
        Autoregressive multi-step forecasting for num_steps
        """
        preds = []
        current_input = initial_input.copy()

        for _ in range(num_steps):
            for layer_idx in range(self.num_layers):
                self._update_layer(layer_idx, current_input)
            state = np.concatenate(self.states)
            x_aug = np.concatenate([state])
            # x_aug = np.concatenate([state, (state)**2, [1.0]])  # For quadrartic readout
            out = self.W_out @ x_aug
            preds.append(out)
            current_input = out

        return np.array(preds)


import numpy as np
from sklearn.linear_model import Ridge
from utils.helpers import scale_spectral_radius



class DeepESN3D:
    """
    Deep Echo State Network (DeepESN) for multi-layered reservoir computing.
    Each layer has its own reservoir, and the states are propagated through layers.

    Notes
    -----
    • For eigen-spectrum tools: a single combined recurrent matrix is exposed as
      `self.W_res` (and alias `self.W`). It is a block-diagonal matrix whose
      diagonal blocks are the per-layer recurrent matrices. This keeps behavior
      unchanged while letting spectrum code retrieve a square matrix.
    """

    def __init__(self,
                 num_layers=3,
                 reservoir_size=100,
                 spectral_radius=0.95,
                 input_scale=1.0,
                 leaking_rate=1.0,
                 input_dim=3,
                 ridge_alpha=1e-6,
                 seed=42):
        """
        Parameters:
        - num_layers: Number of reservoir layers.
        - reservoir_size: Number of neurons in each reservoir layer.
        """
        self.num_layers = num_layers
        self.reservoir_size = reservoir_size
        self.spectral_radius = spectral_radius
        self.input_scale = input_scale
        self.leaking_rate = leaking_rate
        self.d_in = input_dim
        self.ridge_alpha = ridge_alpha
        self.seed = seed

        # Initialize reservoirs and input weights for each layer
        self.reservoirs = []      # list of (N x N) recurrent matrices per layer
        self.input_weights = []   # list of input / inter-layer weights
        self.states = []          # list of state vectors per layer

        np.random.seed(self.seed)
        for layer in range(num_layers):
            np.random.seed(seed + layer)
            W = np.random.randn(reservoir_size, reservoir_size) * 0.1
            W = scale_spectral_radius(W, spectral_radius)
            self.reservoirs.append(W.astype(np.float32))

            if layer == 0:
                W_in = (np.random.rand(reservoir_size, self.d_in) - 0.5) * 2.0 * input_scale
            else:
                # feed-forward weight from previous layer's state
                W_in = (np.random.rand(reservoir_size, reservoir_size) - 0.5) * 2.0 * input_scale
            self.input_weights.append(W_in.astype(np.float32))

        # ---- Expose a single recurrent matrix for spectrum tools (minimal change)
        # Block-diagonal stack of layer recurrent matrices: diag(W_0, W_1, ..., W_{L-1})
        L, N = self.num_layers, self.reservoir_size
        W_stack = np.zeros((L * N, L * N), dtype=np.float32)
        for j in range(L):
            Wjj = self.reservoirs[j]
            W_stack[j * N:(j + 1) * N, j * N:(j + 1) * N] = Wjj
        self.W_res = W_stack     # spectrum helper will find this
        self.W = self.W_res      # optional alias

        self.W_out = None
        self.reset_state()

    def reset_state(self):
        """
        Reset the states of all reservoir layers.
        """
        self.states = [np.zeros(self.reservoir_size, dtype=np.float32)
                       for _ in range(self.num_layers)]

    def _update_layer(self, layer_idx, u):
        """
        Update a single reservoir layer.
        """
        pre_activation = self.reservoirs[layer_idx] @ self.states[layer_idx]
        if layer_idx == 0:
            pre_activation += self.input_weights[layer_idx] @ u
        else:
            pre_activation += self.input_weights[layer_idx] @ self.states[layer_idx - 1]

        x_new = np.tanh(pre_activation)
        alpha = self.leaking_rate
        self.states[layer_idx] = (1.0 - alpha) * self.states[layer_idx] + alpha * x_new

    def collect_states(self, inputs, discard=100):
        self.reset_state()
        all_states = []
        for u in inputs:
            for layer_idx in range(self.num_layers):
                self._update_layer(layer_idx, u)
            all_states.append(np.concatenate(self.states))
        all_states = np.array(all_states, dtype=np.float32)
        return all_states[discard:], all_states[:discard]

    def fit_readout(self, train_input, train_target, discard=100):
        """
        Train the readout layer using ridge regression.
        """
        states_use, _ = self.collect_states(train_input, discard=discard)
        targets_use = train_target[discard:]

        # Linear readout (switch to quadratic by augmenting features if desired)
        X_aug = states_use

        reg = Ridge(alpha=self.ridge_alpha, fit_intercept=False)
        reg.fit(X_aug, targets_use)
        self.W_out = reg.coef_.astype(np.float32)

    def predict(self, inputs):
        """
        Single-step-ahead inference on test data (open-loop with provided inputs).
        """
        preds = []
        for u in inputs:
            for layer_idx in range(self.num_layers):
                self._update_layer(layer_idx, u)
            state = np.concatenate(self.states)
            x_aug = state
            out = self.W_out @ x_aug
            preds.append(out)
        return np.array(preds, dtype=np.float32)

    def predict_open_loop(self, test_input):
        """
        Open-loop (teacher-forced) prediction:
        feed the TRUE input sequence and emit readout at each step.
        This mirrors `predict(...)` and does NOT reset state.
        """
        if self.W_out is None:
            raise RuntimeError("Call fit_readout() before prediction.")
        return self.predict(test_input)

    def predict_autoregressive(self, initial_input, num_steps):
        """
        Autoregressive multi-step forecasting for num_steps
        """
        if self.W_out is None:
            raise RuntimeError("Call fit_readout() before prediction.")
        preds = []
        current_input = initial_input.astype(np.float32).copy()

        for _ in range(num_steps):
            for layer_idx in range(self.num_layers):
                self._update_layer(layer_idx, current_input)
            state = np.concatenate(self.states)
            x_aug = state
            out = self.W_out @ x_aug
            preds.append(out)
            current_input = out
        return np.array(preds, dtype=np.float32)
