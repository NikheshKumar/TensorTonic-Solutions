## Recurrent Neural Networks: The RNN Cell

- A Recurrent Neural Network (RNN) processes sequential data by maintaining a hidden state that carries information from previous time steps
- Unlike feedforward networks, RNNs share the same parameters across all time steps, enabling them to handle sequences of variable length

### The Vanilla RNN Cell

- The simplest form of recurrence, the Elman RNN cell, updates the hidden state at each time step using:

$$
h_t = \tanh(W_{ih} \cdot x_t + b_{ih} + W_{hh} \cdot h_{t-1} + b_{hh})
$$

- Here $x_t \in \mathbb{R}^{d}$ is the input at time $t$, and $h_{t-1} \in \mathbb{R}^{h}$ is the hidden state from the previous step
- $W_{ih} \in \mathbb{R}^{h \times d}$ projects the input into hidden space
- $W_{hh} \in \mathbb{R}^{h \times h}$ projects the previous hidden state into hidden space
- $b_{ih}, b_{hh} \in \mathbb{R}^{h}$ are additive bias terms
- The $\tanh$ activation squashes the output to $[-1, 1]$, preventing unbounded growth

### Why tanh?

- The hyperbolic tangent function is zero-centered, meaning its outputs are distributed around zero
- This is beneficial for gradient flow compared to sigmoid, which outputs only positive values
- The derivative of $\tanh$ is:

$$
\frac{d}{dz}\tanh(z) = 1 - \tanh^2(z)
$$

- This means the gradient is at most 1 (when $z = 0$) and decays toward 0 for large $|z|$

### Information Flow Across Time Steps

- At each step, the RNN cell combines two sources of information:
  - The current input $x_t$, transformed by $W_{ih}$
  - The previous hidden state $h_{t-1}$, transformed by $W_{hh}$
- The hidden state $h_t$ serves as a compressed summary of all inputs seen up to time $t$
- For a sequence of length $T$, the same cell is applied repeatedly:

$$
h_1 = \tanh(W_{ih} x_1 + b_{ih} + W_{hh} h_0 + b_{hh})
$$

$$
h_2 = \tanh(W_{ih} x_2 + b_{ih} + W_{hh} h_1 + b_{hh})
$$

$$
\vdots
$$

$$
h_T = \tanh(W_{ih} x_T + b_{ih} + W_{hh} h_{T-1} + b_{hh})
$$

### The Vanishing Gradient Problem

- During backpropagation through time (BPTT), gradients flow backward through the chain of hidden states
- The gradient at step $t$ depends on a product of Jacobians:

$$
\frac{\partial h_T}{\partial h_t} = \prod_{k=t+1}^{T} \frac{\partial h_k}{\partial h_{k-1}}
$$

- Each factor involves the derivative of $\tanh$ (bounded by 1) multiplied by $W_{hh}$
- If the spectral norm of $W_{hh}$ is less than 1, this product shrinks exponentially, causing gradients to vanish
- If it is greater than 1, gradients can explode
- This limits the vanilla RNN's ability to learn long-range dependencies

### Parameter Sharing

- A key property of the RNN cell is that $W_{ih}$, $W_{hh}$, $b_{ih}$, and $b_{hh}$ are shared across all time steps
- This drastically reduces the number of learnable parameters compared to having separate weights per step
- It also enables the model to generalize to sequences of lengths not seen during training

### Batched Computation

- In practice, inputs are batched: $X \in \mathbb{R}^{B \times d}$ where $B$ is the batch size
- The computation becomes:

$$
H_t = \tanh(X_t W_{ih}^T + b_{ih} + H_{t-1} W_{hh}^T + b_{hh})
$$

- The bias vectors broadcast across the batch dimension automatically
- This enables efficient parallel computation on GPUs

### Connection to LSTMs and GRUs

- The vanilla RNN cell is the foundation for more advanced gated architectures
- LSTMs add input, forget, and output gates along with a separate cell state to mitigate the vanishing gradient problem
- GRUs simplify the gating mechanism to update and reset gates, reducing the parameter count while retaining the benefits of gating
- Understanding the vanilla RNN cell is essential before studying these more complex variants