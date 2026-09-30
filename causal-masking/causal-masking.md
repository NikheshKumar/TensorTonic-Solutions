## Why Causal Masking Exists

In language modeling, the goal is to predict the next token given all previous tokens. At position $t$, the model should only see tokens $1, 2, ..., t-1$ when predicting token $t$.

But Transformers process all positions in parallel. Without intervention, the self-attention mechanism would allow each position to attend to every other position, including future tokens. This is called **information leakage** and it would make the task trivial during training (just copy the answer).

**Causal masking** prevents this by blocking attention to future positions.

---

## The Attention Mechanism

Standard self-attention computes:

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V
$$

where:
- $Q$ (queries), $K$ (keys), $V$ (values) are matrices of shape (sequence_length, d_k)
- $QK^T$ produces a (sequence_length, sequence_length) matrix of attention scores
- Each row $i$ contains scores indicating how much position $i$ attends to every position

Without masking, position 3 can attend to positions 1, 2, 3, 4, 5, ... (all positions).

---

## The Causal Mask

A causal mask is an upper triangular matrix of negative infinity (or very large negative numbers):

$$
M = \begin{bmatrix}
0 & -\infty & -\infty & -\infty \\
0 & 0 & -\infty & -\infty \\
0 & 0 & 0 & -\infty \\
0 & 0 & 0 & 0
\end{bmatrix}
$$

This mask is added to the attention scores before softmax:

$$
\text{Attention} = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}} + M\right) V
$$

---

## How the Mask Works

When we add $-\infty$ to an attention score and then apply softmax:

$$
\text{softmax}([2.0, 1.5, -\infty, -\infty]) = [0.62, 0.38, 0, 0]
$$

The $-\infty$ values become exactly 0 after softmax because $e^{-\infty} = 0$.

**Effect:** Position $i$ can only attend to positions $1, 2, ..., i$. Future positions ($i+1, i+2, ...$) receive zero attention weight.

---

## Visualizing the Mask

For a sequence of length 4, the attention pattern after masking:

**Position 1:** Can attend to [1]

**Position 2:** Can attend to [1, 2]

**Position 3:** Can attend to [1, 2, 3]

**Position 4:** Can attend to [1, 2, 3, 4]

The attention matrix looks like a lower triangle:

$$
\begin{bmatrix}
\checkmark & \times & \times & \times \\
\checkmark & \checkmark & \times & \times \\
\checkmark & \checkmark & \checkmark & \times \\
\checkmark & \checkmark & \checkmark & \checkmark
\end{bmatrix}
$$

---

## Building the Mask

**Step 1:** Create a matrix of ones with shape (seq_len, seq_len)

**Step 2:** Take the upper triangular part (excluding the diagonal)

**Step 3:** Replace 1s with $-\infty$ (or a large negative number like -1e9)

**Step 4:** Replace 0s with 0 (or leave as is)

Alternatively, create a lower triangular matrix of 1s (valid positions) and convert invalid positions to $-\infty$.

**Example for seq_len = 4:**

Upper triangular (what to mask):

$$
\begin{bmatrix}
0 & 1 & 1 & 1 \\
0 & 0 & 1 & 1 \\
0 & 0 & 0 & 1 \\
0 & 0 & 0 & 0
\end{bmatrix}
$$

Multiply by $-\infty$:

$$
\begin{bmatrix}
0 & -\infty & -\infty & -\infty \\
0 & 0 & -\infty & -\infty \\
0 & 0 & 0 & -\infty \\
0 & 0 & 0 & 0
\end{bmatrix}
$$

---

## Causal vs. Bidirectional Attention

**Causal (autoregressive):**
- Each position sees only past and current positions
- Used in: GPT, language modeling, text generation
- The model generates one token at a time, left to right

**Bidirectional:**
- Each position sees all positions
- Used in: BERT, masked language modeling, classification
- The model processes the entire sequence at once

**Encoder-decoder:**
- Encoder uses bidirectional attention (sees full input)
- Decoder uses causal attention (generates output left to right)
- Cross-attention from decoder to encoder is not causally masked

---

## Why Not Just Process Sequentially?

RNNs process sequences one step at a time, naturally preventing future information leakage. Why use parallel processing with masking?

**Efficiency:** Transformers process all positions in parallel during training. This is much faster on GPUs than sequential processing.

**Training vs. Inference:** During training, we know all tokens and can compute losses at all positions simultaneously. During inference, we still generate one token at a time, but training is parallelized.

---

## Implementation Details

**Using -inf vs. large negative:**

True $-\infty$ can cause NaN issues in some frameworks. Using -1e9 or -1e4 is practically equivalent (softmax of -1e9 is essentially 0) and more numerically stable.

**Broadcasting:**

The mask has shape (seq_len, seq_len) but attention scores have shape (batch, heads, seq_len, seq_len). The mask is broadcast across batch and head dimensions.

**Caching:**

The causal mask depends only on sequence length, not on the actual content. It can be precomputed and reused.

**Variable sequence lengths:**

When processing batches with different sequence lengths, combine the causal mask with a padding mask to also ignore padding tokens.

---

## Causal Masking in Multi-Head Attention

The same causal mask is applied to all attention heads. Each head learns different attention patterns, but all are constrained to be causal.

$$
\text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V, \text{mask})
$$

The mask parameter is the same for all heads.

---

## Applications Beyond Language Modeling

**Time series forecasting:**
Predicting future values from past values. Future data points must be masked.

**Audio generation:**
WaveNet and similar models generate audio samples causally.

**Video prediction:**
Generating future frames from past frames.

**Any autoregressive task:**
Wherever the output at step $t$ should depend only on inputs up to step $t$.