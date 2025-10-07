import numpy as np
from sklearn.linear_model import Ridge
from utils.helpers import scale_spectral_radius, augment_state_with_squares
import networkx as nx      


class CycleReservoir3D:
    """
    Cycle reservoir for 3D->3D single-step, with x^2 readout augmentation.
    """
    def __init__(self,
                 reservoir_size=300,
                 spectral_radius=0.95,
                 input_scale=1.0,
                 leaking_rate=1.0,
                 ridge_alpha=1e-6,
                 seed=42):
        self.reservoir_size = reservoir_size
        self.spectral_radius = spectral_radius
        self.input_scale = input_scale
        self.leaking_rate = leaking_rate
        self.ridge_alpha = ridge_alpha
        self.seed = seed

        np.random.seed(self.seed)
        W = np.zeros((reservoir_size, reservoir_size))
        for i in range(reservoir_size):
            j = (i+1) % reservoir_size
            W[i, j] = 1.0
        W = scale_spectral_radius(W, self.spectral_radius)
        self.W = W

        np.random.seed(self.seed+1)
        self.W_in = (np.random.rand(reservoir_size, 3) - 0.5)*2.0*self.input_scale

        self.W_out = None
        self.x = np.zeros(reservoir_size)

    def reset_state(self):
        self.x = np.zeros(self.reservoir_size)

    def _update(self, u):
        pre_activation = self.W @ self.x + self.W_in @ u
        x_new = np.tanh(pre_activation)
        alpha = self.leaking_rate
        self.x = (1.0 - alpha)*self.x + alpha*x_new

    def collect_states(self, inputs, discard=100):
        self.reset_state()
        states = []
        for val in inputs:
            self._update(val)
            states.append(self.x.copy())
        return np.array(states[discard:]), np.array(states[:discard])

    def fit_readout(self, train_input, train_target, discard=100):
        states_use, _ = self.collect_states(train_input, discard=discard)
        targets_use = train_target[discard:]

        X_list = []
        for s in states_use:
            X_list.append(np.concatenate([s, s**2, [1.0]]))
        X_aug = np.array(X_list)

        reg = Ridge(alpha=self.ridge_alpha, fit_intercept=False)
        reg.fit(X_aug, targets_use)
        self.W_out = reg.coef_

    def predict_autoregressive(self, initial_input, n_steps):
        preds = []
        current_in = np.array(initial_input)
        for _ in range(n_steps):
            self._update(current_in)
            big_x = augment_state_with_squares(self.x)
            out = self.W_out @ big_x
            preds.append(out)
            current_in = out
        return np.array(preds)

    def predict_open_loop(self, test_input):
        """
        Open-loop (teacher-forced) prediction:
        feed the TRUE input sequence and emit readout at each step.
        Note: does NOT reset the state to preserve whatever warm-up you use.
        """
        preds = []
        for true_input in test_input:
            self._update(true_input)
            x_aug = augment_state_with_squares(self.x)
            out = self.W_out @ x_aug
            preds.append(out)
        return np.array(preds)


class BaselineESN3D:
    """
    Dense random ESN for 3D->3D single-step. 
    Teacher forcing for training, autoregressive for testing.
    Uses [x, x^2, 1] in readout.
    """
    def __init__(self,
                 reservoir_size=300,
                 spectral_radius=0.95,
                 input_scale=1.0,
                 leaking_rate=1.0,
                 ridge_alpha=1e-6,
                 seed=42):
        self.reservoir_size = reservoir_size
        self.spectral_radius = spectral_radius
        self.input_scale = input_scale
        self.leaking_rate = leaking_rate
        self.ridge_alpha = ridge_alpha
        self.seed = seed

        np.random.seed(self.seed)
        W = np.random.randn(reservoir_size, reservoir_size)*0.1
        W = scale_spectral_radius(W, self.spectral_radius)
        self.W = W

        np.random.seed(self.seed+1)
        self.W_in = (np.random.rand(reservoir_size, 3) - 0.5)*2.0*self.input_scale

        self.W_out = None
        self.x = np.zeros(reservoir_size)

    def reset_state(self):
        self.x = np.zeros(self.reservoir_size)

    def _update(self, u):
        """Single-step update."""
        pre_activation = self.W @ self.x + self.W_in @ u
        x_new = np.tanh(pre_activation)
        alpha = self.leaking_rate
        self.x = (1.0 - alpha)*self.x + alpha*x_new

    def collect_states(self, inputs, discard=100):
        """Teacher forcing: feed real 3D inputs => gather states."""
        self.reset_state()
        states = []
        for val in inputs:
            self._update(val)
            states.append(self.x.copy())
        states = np.array(states)  # [T, reservoir_size]
        return states[discard:], states[:discard]

    def fit_readout(self, train_input, train_target, discard=100):
        """
        1) collect states with teacher forcing
        2) augment with squares => [T-discard, 2N+1]
        3) train ridge => readout in R^(3 x 2N+1)
        """
        states_use, _ = self.collect_states(train_input, discard=discard)
        targets_use = train_target[discard:]

        X_list = []
        for s in states_use:
            X_list.append(np.concatenate([s, s**2, [1.0]]))
        X_aug = np.array(X_list)

        reg = Ridge(alpha=self.ridge_alpha, fit_intercept=False)
        reg.fit(X_aug, targets_use)
        self.W_out = reg.coef_

    def predict_autoregressive(self, initial_input, n_steps):
        """Fully autoregressive test => feed last output as next input."""
        preds = []
        current_in = np.array(initial_input)
        for _ in range(n_steps):
            self._update(current_in)
            big_x = augment_state_with_squares(self.x)
            out = self.W_out @ big_x
            preds.append(out)
            current_in = out
        return np.array(preds)

    def predict_open_loop(self, test_input):
        """
        Open-loop (teacher-forced) prediction:
        feed the TRUE input sequence and emit readout at each step.
        Note: does NOT reset the state to preserve whatever warm-up you use.
        """
        preds = []
        for true_input in test_input:
            self._update(true_input)
            x_aug = augment_state_with_squares(self.x)
            out = self.W_out @ x_aug
            preds.append(out)
        return np.array(preds)


class CRJRes3D(BaselineESN3D):
    def __init__(self,
                 reservoir_size=300,
                 edge_weight=0.8,
                 jump=10,
                 spectral_radius=0.95,
                 input_scale=1.0,
                 leaking_rate=1.0,
                 ridge_alpha=1e-6,
                 seed=42):

        self.reservoir_size = reservoir_size
        self.edge_weight = edge_weight
        self.jump = jump
        self.spectral_radius = spectral_radius
        self.input_scale = input_scale
        self.leaking_rate = leaking_rate
        self.ridge_alpha = ridge_alpha
        self.seed = seed

        np.random.seed(self.seed)
        W = np.zeros((reservoir_size, reservoir_size))
        for i in range(reservoir_size):
            W[i, (i+1) % reservoir_size] = edge_weight              # Cycle edge
            W[i, (i + self.jump) % reservoir_size] = edge_weight    # Jump edge

        W = scale_spectral_radius(W, self.spectral_radius)
        self.W = W

        np.random.seed(self.seed+100)
        self.W_in = (np.random.rand(self.reservoir_size, 3) - 0.5) * 2.0 * self.input_scale

        self.W_out = None
        self.reset_state()
    # NOTE: inherits predict_open_loop from BaselineESN3D


class SWRes3D(BaselineESN3D):
    """
    Small-World (SW) Reservoir for 3D->3D single-step prediction using the Watts-Strogatz (WS) method.
    """
    def __init__(self,
                 reservoir_size=300,
                 rewiring_prob=0.1,
                 degree=6,
                 spectral_radius=0.95,
                 input_scale=1.0,
                 leaking_rate=1.0,
                 ridge_alpha=1e-6,
                 seed=42):

        self.reservoir_size = reservoir_size
        self.spectral_radius = spectral_radius
        self.rewiring_prob = rewiring_prob
        self.degree = degree
        self.input_scale = input_scale
        self.leaking_rate = leaking_rate
        self.ridge_alpha = ridge_alpha
        self.seed = seed
        self.d_in = 3  # Input dimension is 3 for 3D inputs

        np.random.seed(self.seed)
        ws_graph = nx.watts_strogatz_graph(n=reservoir_size, k=self.degree, p=self.rewiring_prob, seed=self.seed)
        adjacency_matrix = nx.to_numpy_array(ws_graph)

        W = adjacency_matrix * np.random.uniform(-1, 1, (reservoir_size, reservoir_size))
        W = scale_spectral_radius(W, spectral_radius)
        self.W = W

        np.random.seed(self.seed+100)
        self.W_in = (np.random.rand(reservoir_size, self.d_in) - 0.5) * 2.0 * input_scale

        self.W_out = None
        self.reset_state()
    # NOTE: inherits predict_open_loop from BaselineESN3D
