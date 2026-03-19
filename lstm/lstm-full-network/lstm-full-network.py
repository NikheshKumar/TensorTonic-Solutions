import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

class LSTM:
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        self.hidden_dim = hidden_dim
        scale = np.sqrt(2.0 / (input_dim + hidden_dim))

        self.W_f = np.random.randn(hidden_dim, hidden_dim + input_dim) * scale
        self.W_i = np.random.randn(hidden_dim, hidden_dim + input_dim) * scale
        self.W_c = np.random.randn(hidden_dim, hidden_dim + input_dim) * scale
        self.W_o = np.random.randn(hidden_dim, hidden_dim + input_dim) * scale
        self.b_f = np.zeros(hidden_dim)
        self.b_i = np.zeros(hidden_dim)
        self.b_c = np.zeros(hidden_dim)
        self.b_o = np.zeros(hidden_dim)

        self.W_y = np.random.randn(output_dim, hidden_dim) * np.sqrt(2.0 / (hidden_dim + output_dim))
        self.b_y = np.zeros(output_dim)

    def forward(self, X: np.ndarray) -> tuple:
        """Forward pass. Returns (y, h_last, C_last)."""
        # YOUR CODE HERE
        B, T, N = X.shape

        h = np.zeros((B, self.hidden_dim))
        C = np.zeros((B, self.hidden_dim))

        y_seq = [] 

        for t in range(T):
            x_t = X[:, t, :]

            concatenated_input = np.concatenate([h, x_t], axis=1)

            f_t = sigmoid(concatenated_input @ self.W_f.T + self.b_f)
            i_t = sigmoid(concatenated_input @ self.W_i.T + self.b_i)
            o_t = sigmoid(concatenated_input @ self.W_o.T + self.b_o)
            C_tilde = np.tanh(concatenated_input @ self.W_c.T + self.b_c)

            C = f_t * C + i_t * C_tilde
            h = o_t * np.tanh(C)

            y = h @ self.W_y.T + self.b_y
            y_seq.append(y)

        y_seq = np.stack(y_seq, axis=1)
      

        return y_seq, h, C