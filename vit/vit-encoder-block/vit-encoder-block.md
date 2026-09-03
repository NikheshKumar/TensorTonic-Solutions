# <span style="font-size: 20px;">ViT Encoder Block</span>

<span style="font-size: 14px;">The encoder block is the core repeating unit of the Vision Transformer (Dosovitskiy et al., 2020). Each block applies multi-head self-attention followed by a feed-forward MLP, with LayerNorm placed before each sub-layer (Pre-LayerNorm) and residual connections after each sub-layer. Unlike GPT's decoder blocks, ViT uses no causal mask, so every patch token attends to every other patch token bidirectionally.</span>

---

## <span style="font-size: 16px;">What It Is</span>

<span style="font-size: 14px;">A ViT encoder block is one layer of the Transformer encoder stack. The input is a sequence of patch embeddings (plus a class token) with shape $(N, D)$, and the output has the same shape. The block performs two sub-operations in sequence: multi-head self-attention (MSA) and a position-wise MLP. Each sub-operation is preceded by LayerNorm and followed by a residual connection.</span>

<span style="font-size: 14px;">The ViT architecture stacks $L$ identical encoder blocks. ViT-Base uses $L = 12$, ViT-Large uses $L = 24$, and ViT-Huge uses $L = 32$. Every block has the same structure but its own learned weights. The design is essentially identical to a GPT-2 decoder block with one critical difference: there is no causal mask in the self-attention.</span>

---

## <span style="font-size: 16px;">Key Equations</span>

<span style="font-size: 14px;">Let $x \in \mathbb{R}^{N \times D}$ be the input, where $N$ is the number of tokens and $D$ is the embedding dimension.</span>

### <span style="font-size: 14px;">Step 1 -- LayerNorm Before Attention</span>

$$
\hat{x} = \text{LayerNorm}(x)
$$

<span style="font-size: 14px;">For each token $x_i \in \mathbb{R}^D$, compute mean $\mu_i = \frac{1}{D}\sum_j x_{ij}$ and variance $\sigma_i^2 = \frac{1}{D}\sum_j (x_{ij} - \mu_i)^2$, then normalize: $\hat{x}_{ij} = \gamma_j \cdot \frac{x_{ij} - \mu_i}{\sqrt{\sigma_i^2 + \epsilon}} + \beta_j$ with learnable $\gamma, \beta \in \mathbb{R}^D$.</span>

### <span style="font-size: 14px;">Step 2 -- Multi-Head Self-Attention</span>

<span style="font-size: 14px;">Project the normalized input into queries, keys, and values:</span>

$$
Q = \hat{x} W_Q, \quad K = \hat{x} W_K, \quad V = \hat{x} W_V
$$

<span style="font-size: 14px;">where $W_Q, W_K, W_V \in \mathbb{R}^{D \times D}$. Reshape into $h$ heads with $d_k = D / h$, then compute scaled dot-product attention per head with no causal mask:</span>

$$
\text{head}_i = \text{softmax}\!\left(\frac{Q_i K_i^T}{\sqrt{d_k}}\right) V_i
$$

<span style="font-size: 14px;">Concatenate all heads and project through $W_O \in \mathbb{R}^{D \times D}$:</span>

$$
\text{MSA}(\hat{x}) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h) \, W_O
$$

### <span style="font-size: 14px;">Step 3 -- First Residual Connection</span>

$$
x' = x + \text{MSA}(\text{LayerNorm}(x))
$$

<span style="font-size: 14px;">The residual adds to the original $x$, not the normalized $\hat{x}$. This is the defining property of Pre-LayerNorm.</span>

### <span style="font-size: 14px;">Step 4 -- LayerNorm Before MLP</span>

$$
\hat{x'} = \text{LayerNorm}(x')
$$

<span style="font-size: 14px;">A second LayerNorm with its own separate $\gamma$ and $\beta$ parameters.</span>

### <span style="font-size: 14px;">Step 5 -- MLP with GELU</span>

$$
\text{MLP}(\hat{x'}) = \text{GELU}(\hat{x'} W_1 + b_1) \, W_2 + b_2
$$

<span style="font-size: 14px;">where $W_1 \in \mathbb{R}^{D \times D_{\text{ff}}}$, $W_2 \in \mathbb{R}^{D_{\text{ff}} \times D}$, and $D_{\text{ff}} = \text{mlp\_ratio} \times D$. The GELU approximation:</span>

$$
\text{GELU}(x) \approx 0.5 \, x \left(1 + \tanh\!\left(\sqrt{\frac{2}{\pi}}\left(x + 0.044715 \, x^3\right)\right)\right)
$$

### <span style="font-size: 14px;">Step 6 -- Second Residual Connection</span>

$$
\text{output} = x' + \text{MLP}(\text{LayerNorm}(x'))
$$

<span style="font-size: 14px;">The complete block in two lines:</span>

$$
x' = x + \text{MSA}(\text{LN}(x)), \qquad \text{output} = x' + \text{MLP}(\text{LN}(x'))
$$

---

## <span style="font-size: 16px;">No Causal Mask</span>

<span style="font-size: 14px;">In autoregressive models like GPT, attention applies a causal mask that sets the upper-triangular entries of the score matrix to $-\infty$ before softmax, preventing each token from attending to future positions. This creates a lower-triangular attention weight matrix essential for left-to-right generation.</span>

<span style="font-size: 14px;">ViT has no such constraint. The input is an image decomposed into a fixed set of patches, all available simultaneously. There is no sequential generation, so every patch token can attend to every other patch token, including the class token. The attention weight matrix is fully dense with no entries masked out.</span>

<span style="font-size: 14px;">This bidirectional attention makes ViT an encoder, not a decoder. The same distinction exists between BERT (bidirectional, no mask) and GPT (unidirectional, causal mask). In implementation, this simply means the attention computation does not add $-\infty$ to any score entries. The softmax receives the full, unmasked score matrix.</span>

<span style="font-size: 14px;">The practical consequence is that each patch's representation depends on every other patch in the image from the very first layer. A patch in the top-left corner can attend to a patch in the bottom-right corner even in the first encoder block. This global receptive field from layer one is a key difference between ViT and convolutional networks, where the receptive field grows gradually with depth through stacked local filters.</span>

---

## <span style="font-size: 16px;">Pre-LayerNorm</span>

<span style="font-size: 14px;">The original Transformer (Vaswani et al., 2017) places LayerNorm after the residual: $\text{output} = \text{LN}(x + \text{SubLayer}(x))$ (Post-LayerNorm). ViT instead uses Pre-LayerNorm: $\text{output} = x + \text{SubLayer}(\text{LN}(x))$, the same arrangement used by GPT-2.</span>

<span style="font-size: 14px;">The advantage is training stability. With Pre-LayerNorm, the residual path is a clean identity connection from the first block's input to the last block's output. Gradients flow through this path without being distorted by normalization layers. With Post-LayerNorm, the normalization sits on the residual path itself, which can cause gradient magnitude issues in deep models. Post-LayerNorm models typically require careful learning rate warmup to avoid early divergence, while Pre-LayerNorm models are more robust to hyperparameter choices.</span>

<span style="font-size: 14px;">Dosovitskiy et al. (2020) explicitly state: "Layernorm (LN) is applied before every block." Each encoder block has two LayerNorm layers with their own learnable parameters. The first normalizes the input before MSA, the second normalizes before the MLP. Neither is applied after the sub-layer or after the residual addition.</span>

---

## <span style="font-size: 16px;">The MLP</span>

<span style="font-size: 14px;">The MLP is a two-layer feed-forward network applied independently to each token. It first projects from $D$ to $D_{\text{ff}} = \text{mlp\_ratio} \times D$ via $W_1$, applies GELU element-wise, then projects back to $D$ via $W_2$. In ViT-Base with $D = 768$ and $\text{mlp\_ratio} = 4$, the hidden dimension is $3072$.</span>

<span style="font-size: 14px;">While attention allows tokens to exchange information by computing weighted combinations, the MLP transforms each token's representation through a nonlinear function. The expansion to $4D$ gives the network enough capacity to learn complex feature transformations, and the contraction back to $D$ keeps the block's output compatible with the next block's input. The $\text{mlp\_ratio}$ hyperparameter controls this expansion factor. The original Transformer uses $D_{\text{ff}} = 4 \times D_{\text{model}}$, and ViT follows this convention exactly.</span>

### <span style="font-size: 14px;">Why GELU and Its Approximation</span>

<span style="font-size: 14px;">GELU applies a smooth probabilistic gate: $\text{GELU}(x) = x \cdot \Phi(x)$ where $\Phi$ is the standard normal CDF. For large positive $x$, the gate approaches 1 (identity). For large negative $x$, it approaches 0 (suppression). Unlike ReLU, GELU never completely kills a neuron since $\Phi(x) > 0$ for all finite $x$.</span>

<span style="font-size: 14px;">The tanh approximation replaces the error function with a polynomial inside tanh. The constant $\sqrt{2/\pi} \approx 0.7979$ arises from the Gaussian CDF's relationship to the error function, and $0.044715$ is a fitted coefficient minimizing maximum absolute error. The approximation is accurate to roughly 4-5 decimal places. In code: $0.5 \cdot x \cdot (1 + \text{tanh}(\sqrt{2/\pi} \cdot (x + 0.044715 \cdot x^3)))$.</span>

---

## <span style="font-size: 16px;">Paper Context</span>

<span style="font-size: 14px;">"An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale" (Dosovitskiy et al., 2020) demonstrates that a pure Transformer applied directly to sequences of image patches can match or exceed convolutional networks on image classification when trained on sufficient data. The paper makes no modifications to the standard Transformer encoder; the novelty is in applying it to vision.</span>

<span style="font-size: 14px;">Three model variants are defined. ViT-Base (ViT-B/16) has 12 blocks, 12 heads, $D = 768$, and $D_{\text{ff}} = 3072$. ViT-Large (ViT-L/16) has 24 blocks, 16 heads, $D = 1024$, and $D_{\text{ff}} = 4096$. ViT-Huge (ViT-H/14) has 32 blocks, 16 heads, $D = 1280$, and $D_{\text{ff}} = 5120$. All use $\text{mlp\_ratio} = 4$.</span>

<span style="font-size: 14px;">A key finding is that ViT underperforms ResNets on mid-size datasets like ImageNet alone but surpasses them when pre-trained on large datasets (JFT-300M or ImageNet-21k). The authors attribute this to the lack of inductive biases: CNNs have built-in translation equivariance and locality that help with small data, but the encoder block's global self-attention allows learning more general representations from large data.</span>

---

## <span style="font-size: 16px;">Numerical Example</span>

<span style="font-size: 14px;">Trace one block with $N = 2$ tokens, $D = 4$, $h = 2$ heads ($d_k = 2$), $\text{mlp\_ratio} = 2$ ($D_{\text{ff}} = 8$). Input:</span>

$$
x = \begin{pmatrix} 1.0 & 0.0 & -1.0 & 2.0 \\ 0.0 & 1.0 & 1.0 & 0.0 \end{pmatrix}
$$

### <span style="font-size: 14px;">Step 1 -- LayerNorm</span>

<span style="font-size: 14px;">Token 1: $\mu = 0.5$, $\sigma^2 = 1.25$. Normalized (with $\gamma = 1, \beta = 0$): $(0.4472, -0.4472, -1.3416, 1.3416)$. Token 2: $\mu = 0.5$, $\sigma^2 = 0.25$. Normalized: $(-1.0, 1.0, 1.0, -1.0)$.</span>

### <span style="font-size: 14px;">Step 2 -- Attention Scores</span>

<span style="font-size: 14px;">After Q/K/V projections and head splitting, suppose head 1 produces scaled scores:</span>

$$
\frac{Q_1 K_1^T}{\sqrt{2}} = \begin{pmatrix} 0.8 & -0.3 \\ -0.3 & 0.6 \end{pmatrix}
$$

<span style="font-size: 14px;">Softmax row 1: $e^{0.8} = 2.2255$, $e^{-0.3} = 0.7408$, sum $= 2.9663$, weights $(0.7504, 0.2496)$. Row 2: $e^{-0.3} = 0.7408$, $e^{0.6} = 1.8221$, sum $= 2.5629$, weights $(0.2891, 0.7109)$. Both tokens attend to both tokens with no entries masked. After concatenating heads and projecting through $W_O$, the MSA output has shape $(2, 4)$.</span>

### <span style="font-size: 14px;">Step 3 -- First Residual</span>

<span style="font-size: 14px;">Suppose MSA output is $\begin{pmatrix} 0.2 & -0.1 & 0.3 & -0.1 \\ -0.1 & 0.2 & -0.1 & 0.1 \end{pmatrix}$. Add to original $x$:</span>

$$
x' = \begin{pmatrix} 1.2 & -0.1 & -0.7 & 1.9 \\ -0.1 & 1.2 & 0.9 & 0.1 \end{pmatrix}
$$

### <span style="font-size: 14px;">Steps 4-5 -- LN then MLP</span>

<span style="font-size: 14px;">Apply second LayerNorm to $x'$, then feed through the MLP. For a single element $z = 0.5$ after the first linear layer, GELU computes: inner $= \sqrt{2/\pi} \cdot (0.5 + 0.044715 \cdot 0.125) = 0.7979 \cdot 0.5056 = 0.4034$, $\tanh(0.4034) = 0.3831$, GELU$(0.5) = 0.25 \cdot 1.3831 = 0.3458$. The 8-dim hidden vector is projected back to 4 dims by $W_2$.</span>

### <span style="font-size: 14px;">Step 6 -- Second Residual</span>

<span style="font-size: 14px;">Add MLP output to $x'$: $\text{output} = x' + \text{MLP}(\text{LN}(x'))$. The output has shape $(2, 4)$, identical to the input, and feeds into the next encoder block.</span>

---

## <span style="font-size: 16px;">How It Differs from a GPT Block</span>

<span style="font-size: 14px;">The ViT encoder block and GPT-2 decoder block share almost identical architecture: Pre-LayerNorm placement, MSA followed by MLP, GELU activation, and residual connections around each sub-layer. The structural formula is the same: $x' = x + \text{Attn}(\text{LN}(x))$, $\text{out} = x' + \text{MLP}(\text{LN}(x'))$.</span>

<span style="font-size: 14px;">The single difference is the causal mask. GPT-2 sets all entries above the diagonal of the score matrix to $-\infty$ before softmax, ensuring position $i$ only attends to positions $j \leq i$. ViT applies no mask at all, producing a fully dense attention weight matrix. This reflects the fundamental distinction: GPT generates tokens autoregressively (left to right) while ViT processes all patches simultaneously for classification.</span>

<span style="font-size: 14px;">In implementation, switching between a ViT encoder block and a GPT decoder block requires only removing or adding the line that constructs and applies the causal mask tensor. Everything else, including weight matrices, MLP, LayerNorm layers, and residual connections, remains structurally identical. The distinction is purely about whether the model is allowed to look at all positions or only past positions.</span>

---

## <span style="font-size: 16px;">Pitfalls</span>

* <span style="font-size: 14px;">**Adding a causal mask by accident.** If you copy attention code from a GPT implementation, it likely includes a causal mask. Applying it in ViT means early patches cannot attend to later patches, breaking the model's ability to reason globally over the image. The class token (typically at position 0) would only attend to itself, making classification impossible. Always verify that no mask is being applied in encoder-style attention.</span>

* <span style="font-size: 14px;">**Wrong GELU approximation.** The tanh approximation has a specific constant $0.044715$ multiplying $x^3$. Using a different constant (such as $0.04471$ from a rounding error), or using $x^2$ instead of $x^3$, produces a numerically different function. The error is small per element but compounds across $D_{\text{ff}} = 3072$ dimensions and $L = 12$ blocks. When the problem specifies the exact approximation formula, use it character for character.</span>

* <span style="font-size: 14px;">**Wrong mlp_ratio.** The default $\text{mlp\_ratio}$ for ViT is $4$, meaning $D_{\text{ff}} = 4D$. A common mistake is using $D_{\text{ff}} = D$ (ratio of 1), which turns the MLP into a simple bottleneck without meaningful expansion. Another mistake is confusing the ratio with an absolute hidden dimension. Always compute $D_{\text{ff}} = \text{mlp\_ratio} \times D$ explicitly from the given parameters.</span>

* <span style="font-size: 14px;">**Forgetting residual connections.** Each sub-layer (MSA and MLP) must add its output to the sub-layer's input via a residual connection. Omitting either residual kills the identity path that enables gradient flow through deep stacks. A model with 12+ blocks and no residual connections will almost certainly fail to train. Remember that the residual adds to the original $x$, not to the normalized $\hat{x}$.</span>

* <span style="font-size: 14px;">**Wrong LayerNorm placement.** Placing LayerNorm after the residual ($\text{LN}(x + \text{SubLayer}(x))$) instead of before ($x + \text{SubLayer}(\text{LN}(x))$) changes training dynamics. ViT uses Pre-LN. Post-LN was the original Transformer's choice but is not what this architecture requires.</span>

* <span style="font-size: 14px;">**Sharing one LayerNorm across both sub-layers.** The block has two separate LayerNorm modules with their own $\gamma$ and $\beta$. Using a single shared instance forces both sub-layers to share normalization parameters, reducing model capacity and producing incorrect outputs.</span>

---