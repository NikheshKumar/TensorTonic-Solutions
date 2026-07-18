## Long Short-Term Memory (LSTM) Cells

- LSTMs were introduced by Hochreiter and Schmidhuber (1997) to address the vanishing gradient problem in standard recurrent neural networks
- The core innovation is the **cell state**, a dedicated memory pathway that runs through time with only multiplicative and additive modifications
- This allows information to persist across many time steps without gradient degradation

## The Vanishing Gradient Problem

- In vanilla RNNs, the hidden state is updated via:

$$
h_t = \tanh(W_{xh} x_t + W_{hh} h_{t-1} + b)
$$

- During backpropagation through time (BPTT), gradients are multiplied by $W_{hh}$ at each step
- If the spectral radius of $W_{hh}$ is less than 1, gradients shrink exponentially
- If greater than 1, gradients explode
- This makes it nearly impossible to learn dependencies spanning more than 10-20 time steps

## LSTM Gate Structure

- The LSTM cell uses three gates (forget, input, output) plus a candidate cell value
- Each gate applies a sigmoid activation, producing values in $[0, 1]$ that act as soft switches

**Forget Gate**: controls what to discard from the cell state

$$
f_t = \sigma(x_t W_{if}^T + b_{if} + h_{t-1} W_{hf}^T + b_{hf})
$$

**Input Gate**: controls what new information to store

$$
i_t = \sigma(x_t W_{ii}^T + b_{ii} + h_{t-1} W_{hi}^T + b_{hi})
$$

**Candidate Cell Value**: proposes new values to add to the state

$$
g_t = \tanh(x_t W_{ig}^T + b_{ig} + h_{t-1} W_{hg}^T + b_{hg})
$$

**Output Gate**: controls what to output from the cell state

$$
o_t = \sigma(x_t W_{io}^T + b_{io} + h_{t-1} W_{ho}^T + b_{ho})
$$

## Cell State and Hidden State Updates

- The cell state update is purely element-wise, with no nonlinear squashing:

$$
c_t = f_t \odot c_{t-1} + i_t \odot g_t
$$

- The hidden state is derived from the cell state, modulated by the output gate:

$$
h_t = o_t \odot \tanh(c_t)
$$

- The gradient of the loss with respect to $c_{t-1}$ includes a direct path through $f_t$
- When $f_t \approx 1$, the gradient flows almost unchanged, solving the vanishing gradient problem
- This is analogous to residual connections in deep feedforward networks

## Parameter Count

- For an LSTM cell with input size $d$ and hidden size $h$:
  - Each gate has two weight matrices: $(h \times d)$ for input and $(h \times h)$ for hidden state
  - Each gate has two bias vectors of size $h$
  - Total per gate: $h \cdot d + h^2 + 2h$ parameters
  - Four gates total: $4(h \cdot d + h^2 + 2h) = 4h(d + h + 2)$ parameters

## Weight Initialization

- Biases are typically initialized to zero
- Weight matrices can be initialized with Xavier/Glorot or simple random normal
- Some practitioners initialize the forget gate bias to a positive value (e.g., 1.0) so the cell retains information by default early in training

## Relationship to GRU

- The Gated Recurrent Unit (GRU) simplifies the LSTM by merging the cell state and hidden state
- GRUs use only two gates (reset and update) instead of three
- LSTMs generally perform comparably to GRUs, but LSTMs are more expressive due to the separate cell state
- For tasks requiring very long-range dependencies, LSTMs often have a slight advantage