## <span style="font-size: 18px;">Attention Mechanisms in Deep Learning</span>

### <span style="font-size: 16px;">Motivation</span>

<span style="font-size: 14px;">Traditional sequence models like RNNs process tokens sequentially, creating a bottleneck when capturing long-range dependencies. The attention mechanism addresses this by allowing each position in a sequence to directly attend to every other position, regardless of distance.</span>

<span style="font-size: 14px;">Key motivations:</span>

* <span style="font-size: 14px;">Sequential processing in RNNs limits parallelization and creates information bottlenecks</span>
* <span style="font-size: 14px;">Fixed-length context vectors in encoder-decoder models lose information for long sequences</span>
* <span style="font-size: 14px;">Attention provides $O(1)$ path length between any two positions, enabling better gradient flow</span>

### <span style="font-size: 16px;">Query, Key, and Value Abstraction</span>

<span style="font-size: 14px;">The attention mechanism is built on a retrieval analogy:</span>

* <span style="font-size: 14px;">**Query** ($Q$): what information the current position is looking for</span>
* <span style="font-size: 14px;">**Key** ($K$): what information each position advertises about itself</span>
* <span style="font-size: 14px;">**Value** ($V$): the actual content each position provides when attended to</span>

<span style="font-size: 14px;">In self-attention, $Q$, $K$, and $V$ are all derived from the same input sequence via learned linear projections. In cross-attention, $Q$ comes from one sequence while $K$ and $V$ come from another.</span>

### <span style="font-size: 16px;">Scaled Dot-Product Attention</span>

<span style="font-size: 14px;">The core computation involves three steps:</span>

* <span style="font-size: 14px;">Compute raw alignment scores via the dot product $Q K^T$</span>
* <span style="font-size: 14px;">Scale by $\frac{1}{\sqrt{d_k}}$ to prevent softmax saturation</span>
* <span style="font-size: 14px;">Apply softmax to obtain normalized attention weights, then multiply by $V$</span>

<span style="font-size: 14px;">The full formula:</span>

$$
\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{Q K^T}{\sqrt{d_k}}\right) V
$$

### <span style="font-size: 16px;">Why Scaling Matters</span>

<span style="font-size: 14px;">Without scaling, the dot product magnitudes grow proportionally to $d_k$. Consider two random vectors with entries drawn from a standard normal distribution:</span>

* <span style="font-size: 14px;">Their dot product has mean 0 and variance $d_k$</span>
* <span style="font-size: 14px;">For large $d_k$, some dot products become very large in absolute value</span>
* <span style="font-size: 14px;">Softmax maps large inputs to near-zero or near-one outputs, creating vanishing gradients</span>
* <span style="font-size: 14px;">Dividing by $\sqrt{d_k}$ restores the variance to approximately 1</span>

### <span style="font-size: 16px;">Softmax as a Soft Argmax</span>

<span style="font-size: 14px;">The softmax function converts raw scores into a probability distribution:</span>

$$
\text{softmax}(z_i) = \frac{e^{z_i}}{\sum_j e^{z_j}}
$$

* <span style="font-size: 14px;">All output values are positive and sum to 1</span>
* <span style="font-size: 14px;">It acts as a differentiable approximation to argmax</span>
* <span style="font-size: 14px;">Higher scores receive exponentially more weight, creating a "soft" selection</span>
* <span style="font-size: 14px;">Applied along the key dimension so each query produces a valid distribution over keys</span>

### <span style="font-size: 16px;">Multi-Head Attention</span>

<span style="font-size: 14px;">In practice, a single attention function is extended to multiple heads:</span>

* <span style="font-size: 14px;">The model projects $Q$, $K$, $V$ into $h$ different subspaces</span>
* <span style="font-size: 14px;">Each head performs attention independently with dimension $d_k / h$</span>
* <span style="font-size: 14px;">Outputs are concatenated and projected back to the model dimension</span>
* <span style="font-size: 14px;">This allows the model to jointly attend to information from different representation subspaces</span>

### <span style="font-size: 16px;">Computational Complexity</span>

<span style="font-size: 14px;">For sequence length $n$ and dimension $d$:</span>

* <span style="font-size: 14px;">Computing $Q K^T$ requires $O(n^2 d)$ operations</span>
* <span style="font-size: 14px;">Storing the attention weight matrix requires $O(n^2)$ memory per head</span>
* <span style="font-size: 14px;">This quadratic cost is the primary limitation of standard attention</span>
* <span style="font-size: 14px;">Techniques like Flash Attention, sparse attention, and linear attention aim to reduce this cost</span>

### <span style="font-size: 16px;">Attention in the Transformer</span>

<span style="font-size: 14px;">The Transformer architecture uses attention in three distinct ways:</span>

* <span style="font-size: 14px;">**Encoder self-attention**: each token attends to all tokens in the input sequence</span>
* <span style="font-size: 14px;">**Decoder self-attention**: each token attends to previous tokens only (causal masking)</span>
* <span style="font-size: 14px;">**Cross-attention**: decoder tokens attend to encoder outputs</span>

<span style="font-size: 14px;">Combined with residual connections, layer normalization, and feed-forward networks, attention forms the backbone of modern language models, vision transformers, and multimodal systems.</span>