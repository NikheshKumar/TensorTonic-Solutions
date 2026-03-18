import numpy as np

class VanillaRNN:
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        self.hidden_dim = hidden_dim

        # Xavier initialization
        self.W_xh = np.random.randn(hidden_dim, input_dim) * np.sqrt(2.0 / (input_dim + hidden_dim))
        self.W_hh = np.random.randn(hidden_dim, hidden_dim) * np.sqrt(2.0 / (2 * hidden_dim))
        self.W_hy = np.random.randn(output_dim, hidden_dim) * np.sqrt(2.0 / (hidden_dim + output_dim))
        self.b_h = np.zeros(hidden_dim)
        self.b_y = np.zeros(output_dim)

    def forward(self, X: np.ndarray, h_0: np.ndarray = None) -> tuple:
        """
        Forward pass through entire sequence.
        Returns (y_seq, h_final).
        """
        # YOUR CODE HERE

        B, T, N = X.shape

        h_prev = np.random.randn(B, self.hidden_dim) if h_0 is None else h_0
        hidden_states = []
        y_seq = []

        for t in range(T):
          x_t = X[:,t,:]
          h = np.tanh(x_t @ self.W_xh.T + h_prev @ self.W_hh + self.b_h)
          y_t = h@self.W_hy.T + self.b_y

          y_seq.append(y_t)
          h_prev = h
          hidden_states.append(h)
          

        hidden_states = np.stack(hidden_states, axis=1)
        y_seq = np.stack(y_seq, axis=1)
        h_final = h_prev

        return (y_seq, h_final)