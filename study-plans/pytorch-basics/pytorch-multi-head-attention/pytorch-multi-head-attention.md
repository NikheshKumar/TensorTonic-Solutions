## Multi-Head Attention

<span style="font-size: 14px;">Multi-head attention is the core mechanism of the Transformer architecture. Instead of performing a single attention function, it runs multiple attention operations in parallel across different learned subspaces.</span>

### Why Multiple Heads?

<span style="font-size: 14px;">A single attention head can only focus on one type of relationship at a time. Multiple heads allow the model to:</span>

- <span style="font-size: 14px;">Attend to different positions simultaneously</span>
- <span style="font-size: 14px;">Capture different types of relationships (syntactic, semantic, positional)</span>
- <span style="font-size: 14px;">Learn complementary representations in parallel subspaces</span>

### The Algorithm

<span style="font-size: 14px;">Given input queries $Q$, keys $K$, and values $V$, each of shape $(B, S, d_{model})$:</span>

**Step 1: Linear Projections**

<span style="font-size: 14px;">Project each input through learned weight matrices:</span>

$$
Q' = Q W_q, \quad K' = K W_k, \quad V' = V W_v
$$

<span style="font-size: 14px;">where $W_q, W_k, W_v \in \mathbb{R}^{d_{model} \times d_{model}}$.</span>

**Step 2: Split into Heads**

<span style="font-size: 14px;">Reshape each projected tensor from $(B, S, d_{model})$ to $(B, h, S, d_k)$ where $d_k = d_{model} / h$. Each head operates on a $d_k$-dimensional slice.</span>

**Step 3: Scaled Dot-Product Attention**

<span style="font-size: 14px;">For each head $i$:</span>

$$
\text{head}_i = \text{softmax}\!\left(\frac{Q_i K_i^T}{\sqrt{d_k}}\right) V_i
$$

- <span style="font-size: 14px;">$Q_i K_i^T$ computes pairwise similarity scores</span>
- <span style="font-size: 14px;">Division by $\sqrt{d_k}$ prevents large dot products that push softmax into saturated regions</span>
- <span style="font-size: 14px;">Softmax normalizes scores to attention weights that sum to 1</span>
- <span style="font-size: 14px;">Multiplication by $V_i$ produces a weighted combination of values</span>

**Step 4: Concatenate and Project**

$$
\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h) \, W_o
$$

<span style="font-size: 14px;">The heads are concatenated back to $d_{model}$ dimensions, then multiplied by $W_o \in \mathbb{R}^{d_{model} \times d_{model}}$ to mix information across heads.</span>

### Scaling Factor Intuition

<span style="font-size: 14px;">If the entries of $Q_i$ and $K_i$ are independent random variables with mean 0 and variance 1, then $Q_i K_i^T$ has variance $d_k$. Dividing by $\sqrt{d_k}$ restores unit variance, keeping softmax in a region where gradients are well-behaved.</span>

### Computational Complexity

- <span style="font-size: 14px;">The four projections cost $O(B \cdot S \cdot d_{model}^2)$ each</span>
- <span style="font-size: 14px;">Attention per head costs $O(B \cdot S^2 \cdot d_k)$, summed over $h$ heads gives $O(B \cdot S^2 \cdot d_{model})$</span>
- <span style="font-size: 14px;">Total complexity is $O(B \cdot S \cdot d_{model}^2 + B \cdot S^2 \cdot d_{model})$</span>

### Implementation Notes

- <span style="font-size: 14px;">The reshape from $(B, S, d_{model})$ to $(B, h, S, d_k)$ is done via view + transpose, not by slicing</span>
- <span style="font-size: 14px;">After attention, call contiguous() before view() since transpose creates non-contiguous memory</span>
- <span style="font-size: 14px;">$d_{model}$ must be divisible by $h$ so that $d_k$ is an integer</span>