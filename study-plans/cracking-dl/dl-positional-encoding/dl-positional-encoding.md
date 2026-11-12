# <span style="font-size: 20px;">Positional Encodings: Sinusoidal and RoPE</span>

<span style="font-size: 14px;">Transformers process all tokens in parallel through self-attention, which is inherently permutation-equivariant: swapping two tokens in the input swaps the corresponding outputs, with no notion of order. Without explicit position information, a Transformer cannot distinguish "the cat sat on the mat" from "mat the on sat cat the." Position encodings inject this missing sequential structure.</span>

---

## <span style="font-size: 16px;">Why Transformers Need Position Information</span>

<span style="font-size: 14px;">In RNNs and LSTMs, position is implicit: token $t$ is processed after token $t{-}1$, so the hidden state naturally encodes order. Self-attention has no such sequential processing. The attention score between positions $i$ and $j$ is computed as $q_i^T k_j$, which is a function only of the content at those positions, not their positions themselves.</span>

<span style="font-size: 14px;">The solution is to add or inject position information into the token representations before (or during) attention. The two dominant approaches are:</span>

- <span style="font-size: 14px;">**Additive encodings**: add a position vector to the token embedding before feeding into the Transformer (sinusoidal, learned)</span>
- <span style="font-size: 14px;">**Rotary encodings**: modify the query/key vectors inside attention so that the dot product naturally encodes relative position (RoPE)</span>

---

## <span style="font-size: 16px;">Sinusoidal Positional Encoding</span>

<span style="font-size: 14px;">Introduced in "Attention Is All You Need" (Vaswani et al., 2017), sinusoidal PE creates a fixed matrix added to the input embeddings:</span>

$$
\begin{aligned}
PE(\text{pos}, 2i) &= \sin\!\left(\frac{\text{pos}}{10000^{2i/d}}\right) \\[6pt]
PE(\text{pos}, 2i{+}1) &= \cos\!\left(\frac{\text{pos}}{10000^{2i/d}}\right)
\end{aligned}
$$

<span style="font-size: 14px;">**Geometric frequency spectrum**: each dimension pair $(2i, 2i{+}1)$ oscillates at a different frequency. Dimension pair 0 has the highest frequency ($\omega_0 = 1$), while the last pair has the lowest ($\omega_{d/2-1} = 1/10000$). This creates a spectrum of wavelengths from $2\pi$ to $2\pi \cdot 10000$, allowing the model to attend to both local and distant positions.</span>

<span style="font-size: 14px;">**Key property**: for any fixed offset $k$, there exists a linear transformation $M_k$ such that $PE(\text{pos} + k) = M_k \cdot PE(\text{pos})$. This is because sin/cos of a sum can be written as a linear combination of sin/cos of the parts. This lets the model learn to attend to relative positions through linear projections.</span>

<span style="font-size: 14px;">**Implementation**: compute $\omega_i = \exp(-2i \cdot \ln(10000) / d)$ rather than $1 / 10000^{2i/d}$ to avoid numerical issues with large powers. The result is identical but numerically stable.</span>

---

## <span style="font-size: 16px;">Rotary Position Embeddings (RoPE)</span>

<span style="font-size: 14px;">RoPE (Su et al., 2021) is now the standard position encoding in virtually every modern LLM: LLaMA, Mistral, Falcon, GPT-NeoX, and reportedly GPT-4. Instead of adding position vectors to embeddings, RoPE rotates the query and key vectors before computing attention scores.</span>

<span style="font-size: 14px;">**Core idea**: group the $d$-dimensional vector into $d/2$ pairs. For each pair $(x_{2i}, x_{2i+1})$ at position $p$, apply a 2D rotation by angle $\theta_i = p / 10000^{2i/d}$:</span>

$$
\begin{bmatrix} x'_{2i} \\ x'_{2i+1} \end{bmatrix} = \begin{bmatrix} \cos\theta_i & -\sin\theta_i \\ \sin\theta_i & \cos\theta_i \end{bmatrix} \begin{bmatrix} x_{2i} \\ x_{2i+1} \end{bmatrix}
$$

<span style="font-size: 14px;">**Why rotation encodes relative position**: when computing the attention score between positions $m$ and $n$, we get $q_m^T k_n = (R_m x_m)^T (R_n x_n) = x_m^T R_m^T R_n x_n = x_m^T R_{n-m} x_n$. The rotation matrices compose by addition of angles, so the score depends only on the content and the relative distance $n - m$, not the absolute positions.</span>

<span style="font-size: 14px;">**Backward pass**: since rotation matrices are orthogonal ($R^T R = I$), the backward pass simply applies the transpose (inverse) rotation: $\partial L / \partial x = R^T \cdot \partial L / \partial x'$. Concretely:</span>

$$
\frac{\partial L}{\partial x_{2i}} = \frac{\partial L}{\partial x'_{2i}} \cos\theta_i + \frac{\partial L}{\partial x'_{2i+1}} \sin\theta_i
$$

$$
\frac{\partial L}{\partial x_{2i+1}} = -\frac{\partial L}{\partial x'_{2i}} \sin\theta_i + \frac{\partial L}{\partial x'_{2i+1}} \cos\theta_i
$$

---

## <span style="font-size: 16px;">Comparing Position Encoding Methods</span>

- <span style="font-size: 14px;">**Sinusoidal (fixed)**: no learnable parameters, theoretically allows extrapolation to longer sequences than seen in training (though in practice this is limited). Simple to implement. Used in the original Transformer, T5.</span>
- <span style="font-size: 14px;">**Learned absolute**: each position gets a learned embedding vector. More flexible but cannot extrapolate beyond training length. Used in BERT, GPT-2.</span>
- <span style="font-size: 14px;">**RoPE**: encodes relative position directly in the attention computation. Excellent length generalization (especially with techniques like NTK-aware scaling or YaRN). No extra parameters. Now the dominant choice in LLMs.</span>
- <span style="font-size: 14px;">**ALiBi (Press et al., 2022)**: adds a linear bias to attention scores proportional to distance. No position embeddings at all. Good length generalization. Used in BLOOM and some Falcon variants.</span>
- <span style="font-size: 14px;">**Relative position bias**: learned bias per distance bin (e.g., T5 uses bucketed relative positions). Flexible but adds parameters and computation.</span>

<span style="font-size: 14px;">RoPE has largely won the position encoding debate for decoder-only LLMs as of 2025, primarily because it combines relative position encoding with zero additional parameters and strong length generalization.</span>

---


## <span style="font-size: 16px;">Length Generalization</span>

<span style="font-size: 14px;">**The core challenge:** a model trained on sequences of length 1024 must also work at length 4096+ during inference. Absolute learned positions fail completely beyond the training length because the embedding lookup has no entry for unseen positions.</span>

<span style="font-size: 14px;">**Sinusoidal encodings** can theoretically generalize because they are defined for any position, but in practice attention patterns degrade for positions far beyond training range because the model has not learned to attend to those relative distance patterns.</span>

<span style="font-size: 14px;">**RoPE** (Rotary Position Embeddings) generalizes better than absolute encodings because it encodes relative position directly into the attention logits. However, it still degrades beyond the training context. Techniques like NTK-aware scaling, YaRN, and dynamic NTK rescale the frequency basis to extend the effective context window without retraining.</span>

<span style="font-size: 14px;">**ALiBi** (Attention with Linear Biases) adds a position-dependent linear penalty to attention scores, achieving strong length generalization with zero learned parameters for position.</span>


## <span style="font-size: 16px;">Common Interview Follow-ups</span>

- <span style="font-size: 14px;">**Q: Why does RoPE use different frequencies for different dimension pairs?**</span>
  <span style="font-size: 14px;">A: Different frequencies create a multi-scale representation of position. Low-frequency pairs (high dimension indices) encode coarse position information (paragraph-level), while high-frequency pairs (low indices) encode fine-grained position (word-level). This is analogous to how Fourier series represent signals at multiple scales. The model can learn which frequency bands to attend to for different tasks.</span>

- <span style="font-size: 14px;">**Q: How do modern LLMs extend RoPE to longer contexts than seen in training?**</span>
  <span style="font-size: 14px;">A: Several techniques exist. NTK-aware interpolation (Code LLaMA) scales the base frequency from 10000 to a larger value, spreading the frequencies to cover more positions without extrapolation. YaRN combines NTK scaling with attention temperature adjustment. Linear interpolation simply divides position indices by a factor. All exploit the fact that RoPE's rotation angles are continuous functions of position, so interpolating between trained positions is more stable than extrapolating beyond them.</span>

- <span style="font-size: 14px;">**Q: Why is sinusoidal PE not used in modern LLMs?**</span>
  <span style="font-size: 14px;">A: Sinusoidal PE encodes absolute position, which the model must learn to convert to relative position through its attention weights. RoPE directly encodes relative position in the dot product, which is what attention fundamentally computes. Empirically, RoPE achieves better perplexity and length generalization. Also, sinusoidal PE is added once at the input, so deeper layers have increasingly indirect access to position information, whereas RoPE is applied at every attention layer.</span>

- <span style="font-size: 14px;">**Q: What happens to the backward pass through sinusoidal PE?**</span>
  <span style="font-size: 14px;">A: Sinusoidal PE is a fixed constant added to the input embeddings. Since the gradient of a constant is zero, the gradient passes through unchanged: $\partial L / \partial \text{embed} = \partial L / \partial (\text{embed} + PE)$. There is nothing to learn or backpropagate through. This is why sinusoidal PE has no backward mode in this problem.</span>

- <span style="font-size: 14px;">**Q: Can RoPE be applied to the values in attention, not just queries and keys?**</span>
  <span style="font-size: 14px;">A: RoPE is intentionally applied only to Q and K because the attention score $q^T k$ should be position-dependent (to select which tokens to attend to), but the value aggregation should be position-independent (the information retrieved should not be rotated by position). Applying RoPE to values would make the output representation position-dependent in a way that complicates downstream layers.</span>

---