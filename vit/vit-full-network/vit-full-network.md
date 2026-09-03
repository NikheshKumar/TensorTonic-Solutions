# <span style="font-size: 20px;">Complete Vision Transformer</span>

<span style="font-size: 14px;">The Vision Transformer (ViT) applies a standard Transformer encoder directly to sequences of image patches for image classification. Rather than using convolutions, it splits an image into fixed-size patches, linearly embeds each one, and processes the resulting sequence with a Transformer. This is the full end-to-end pipeline: patch embedding, CLS token, position embeddings, $L$ encoder blocks, and a classification head.</span>

---

## <span style="font-size: 16px;">What It Is</span>

<span style="font-size: 14px;">ViT is an image classification architecture that replaces convolutional feature extraction with a Transformer encoder. The input image is divided into non-overlapping patches, each flattened and linearly projected to produce a patch embedding. A learnable **[CLS] token** is prepended, learnable position embeddings are added, and the full sequence passes through $L$ encoder blocks (pre-LN, bidirectional attention, GELU MLP). After the final block, the [CLS] representation is extracted, LayerNormed, and linearly projected to class logits. No softmax is applied.</span>

<span style="font-size: 14px;">The architecture is deliberately minimal. Dosovitskiy et al. made as few modifications to the standard Transformer as possible, proving that a pure Transformer without any convolutional inductive bias can achieve state-of-the-art image classification when trained on sufficient data.</span>

---

## <span style="font-size: 16px;">Key Equations</span>

<span style="font-size: 14px;">**Stage 1 -- Patch embedding.** Split the image into $N$ patches of size $P \times P$. Flatten each patch to length $P^2 \cdot C$, then project:</span>

$$
z_i^{(0)} = x_{\text{patch}_i} \cdot W_E + b_E, \quad i = 1, \ldots, N
$$

<span style="font-size: 14px;">where $W_E \in \mathbb{R}^{(P^2 C) \times D}$ and $D$ is the hidden dimension.</span>

<span style="font-size: 14px;">**Stage 2 -- Prepend [CLS] and add position embeddings:**</span>

$$
z^{(0)} = [z_{\text{cls}};\; z_1^{(0)};\; \ldots;\; z_N^{(0)}] + E_{\text{pos}}
$$

<span style="font-size: 14px;">where $z_{\text{cls}} \in \mathbb{R}^D$ is learnable and $E_{\text{pos}} \in \mathbb{R}^{(N+1) \times D}$.</span>

<span style="font-size: 14px;">**Stage 3 -- $L$ encoder blocks (pre-LN):**</span>

$$
z'^{(\ell)} = \text{MSA}(\text{LN}(z^{(\ell-1)})) + z^{(\ell-1)}
$$

$$
z^{(\ell)} = \text{MLP}(\text{LN}(z'^{(\ell)})) + z'^{(\ell)}
$$

<span style="font-size: 14px;">**Stage 4 -- Classification head:**</span>

$$
y = \text{LN}(z_0^{(L)}) \cdot W_{\text{head}} + b_{\text{head}}
$$

<span style="font-size: 14px;">where $z_0^{(L)}$ is the CLS token after block $L$, and $y \in \mathbb{R}^K$ is raw logits.</span>

---

## <span style="font-size: 16px;">Stage 1: Patch Embedding</span>

<span style="font-size: 14px;">The image has shape $(C, H, W)$. It is divided into a grid of non-overlapping $P \times P$ patches. The number of patches is $N = HW / P^2$. For a 224x224 image with $P = 16$: $N = 196$. Each patch is flattened to length $P^2 C = 768$ (for RGB), then projected through $W_E \in \mathbb{R}^{(P^2 C) \times D}$ with bias.</span>

<span style="font-size: 14px;">Why patches instead of pixels? A 224x224 image has 50,176 pixels. Self-attention is $O(N^2)$, making per-pixel tokenization prohibitive. Patch size 16 reduces this to 196 tokens. Smaller patches give finer resolution but quadratically increase cost. The ViT naming convention reflects this: ViT-B/16 means Base model with patch size 16.</span>

<span style="font-size: 14px;">In practice, this linear projection is equivalent to a 2D convolution with kernel size $P$ and stride $P$. All weights are initialized from $\mathcal{N}(0, 0.02^2)$.</span>

---

## <span style="font-size: 16px;">Stage 2: CLS Token and Position Embeddings</span>

<span style="font-size: 14px;">**The [CLS] token.** A learnable vector $z_{\text{cls}} \in \mathbb{R}^D$ is prepended, increasing sequence length from $N$ to $N + 1$. It has no spatial content and serves as a global aggregation point. Because attention is bidirectional, CLS can attend to all patches and vice versa. After $L$ blocks, the CLS representation encodes the entire image. This follows BERT's design.</span>

<span style="font-size: 14px;">**Position embeddings.** Learnable $E_{\text{pos}} \in \mathbb{R}^{(N+1) \times D}$ are added element-wise. Each position gets its own $D$-dimensional vector, encoding spatial layout: position 1 is the top-left patch, position 2 the next, and so on. Without these, the Transformer treats patches as an unordered set and loses all spatial information.</span>

<span style="font-size: 14px;">The paper found that learned 1D embeddings perform as well as more complex 2D-aware schemes. The authors tested fixed sinusoidal, learned 1D, and learned 2D embeddings, finding no significant accuracy difference. The Transformer learns to infer 2D spatial relationships from 1D indices.</span>

---

## <span style="font-size: 16px;">Stage 3: Transformer Encoder</span>

<span style="font-size: 14px;">The sequence $z^{(0)} \in \mathbb{R}^{(N+1) \times D}$ passes through $L$ identical encoder blocks. Each has two sub-layers: multi-head self-attention (MSA) and a feed-forward MLP, both with pre-LayerNorm and residual connections.</span>

<span style="font-size: 14px;">**Multi-head self-attention.** Input is normalized: $\hat{z} = \text{LN}(z^{(\ell-1)})$. Queries, keys, values are computed, split into $h$ heads of dimension $d_h = D/h$. Each head computes:</span>

$$
\text{head}_j = \text{softmax}\!\left(\frac{Q_j K_j^T}{\sqrt{d_h}}\right) V_j
$$

<span style="font-size: 14px;">Heads are concatenated and projected: $\text{MSA} = [\text{head}_1; \ldots; \text{head}_h] W_O$. **There is no causal mask.** Every token attends to every other token bidirectionally. Image patches have no sequential ordering, so masking future positions would create arbitrary asymmetry.</span>

<span style="font-size: 14px;">**Feed-forward MLP.** After the attention residual, the output is normalized and passed through:</span>

$$
\text{MLP}(x) = \text{GELU}(x W_1 + b_1) W_2 + b_2
$$

<span style="font-size: 14px;">where $W_1 \in \mathbb{R}^{D \times 4D}$ and $W_2 \in \mathbb{R}^{4D \times D}$. The activation is GELU, not ReLU.</span>

<span style="font-size: 14px;">**Pre-LN ordering** means LayerNorm is applied before each sub-layer, not after. This stabilizes training for deeper models because the residual path carries unnormalized gradients directly.</span>

---

## <span style="font-size: 16px;">Stage 4: Classification Head</span>

<span style="font-size: 14px;">Extract the [CLS] token from position 0 of the final block output: $z_0^{(L)} \in \mathbb{R}^D$. Apply LayerNorm, then project:</span>

$$
y = \text{LN}(z_0^{(L)}) \cdot W_{\text{head}} + b_{\text{head}}
$$

<span style="font-size: 14px;">where $W_{\text{head}} \in \mathbb{R}^{D \times K}$ and $K$ is the number of classes. Output is raw logits (no softmax). During training, logits go to cross-entropy loss which applies log-softmax internally.</span>

<span style="font-size: 14px;">During pre-training on large datasets, the paper uses an MLP head with tanh activation. During fine-tuning, this is replaced with a single linear layer. This problem uses the fine-tuning variant: LayerNorm then linear. All head weights are initialized as $\mathcal{N}(0, 0.02^2)$.</span>

---

## <span style="font-size: 16px;">The Data Requirement</span>

<span style="font-size: 14px;">The central empirical finding: pure Transformers need large-scale pre-training to match CNNs. On ImageNet-1K alone (1.3M images), ViT-B/16 underperforms a comparably sized ResNet. The Transformer lacks convolutional inductive biases: translation equivariance and locality. It must learn spatial structure entirely from data.</span>

<span style="font-size: 14px;">As the paper states: "When trained on mid-sized datasets, these models yield modest accuracies. However, the picture changes if trained on larger datasets." On JFT-300M (303M images), ViT-L/16 surpasses the best CNNs on every benchmark.</span>

* <span style="font-size: 14px;">**Small data:** CNN inductive biases act as regularizers, reducing the hypothesis space. ViT must learn these patterns from scratch.</span>
* <span style="font-size: 14px;">**Large data:** Inductive biases become constraints. Self-attention can learn arbitrary spatial relationships that CNNs only capture through many stacked layers. With enough data, flexibility wins.</span>
* <span style="font-size: 14px;">**Compute:** ViT maps directly onto hardware optimized for matrix multiplications. ViT-H/14 reached 88.55% top-1 on ImageNet using fewer TPU-days than competing models.</span>

---

## <span style="font-size: 16px;">Paper Context</span>

<span style="font-size: 14px;">"An Image is Worth 16x16 Words" by Dosovitskiy et al. (2020, Google Brain) showed that a pure Transformer applied directly to image patches can match or exceed convolutional networks at scale.</span>

<span style="font-size: 14px;">Three model configurations:</span>

* <span style="font-size: 14px;">**ViT-Base:** $L = 12$, $D = 768$, $h = 12$ heads, MLP $= 3072$, ~86M params.</span>
* <span style="font-size: 14px;">**ViT-Large:** $L = 24$, $D = 1024$, $h = 16$ heads, MLP $= 4096$, ~307M params.</span>
* <span style="font-size: 14px;">**ViT-Huge:** $L = 32$, $D = 1280$, $h = 16$ heads, MLP $= 5120$, ~632M params.</span>

<span style="font-size: 14px;">The design philosophy was deliberate minimalism. Prior work combined CNN feature extractors with Transformers. Dosovitskiy et al. asked: can a standard Transformer with no convolutions work for vision? The patch embedding is the only vision-specific component. Weight initialization throughout is $\mathcal{N}(0, 0.02^2)$.</span>

---

## <span style="font-size: 16px;">Numerical Example</span>

<span style="font-size: 14px;">Trace a tiny image through all stages. Image: $C = 3$, $H = 4$, $W = 4$, $P = 2$, $D = 4$, $L = 1$ block, $h = 2$ heads ($d_h = 2$), $K = 3$ classes.</span>

<span style="font-size: 14px;">**Stage 1: Patch embedding.** $N = 16 / 4 = 4$ patches. Each flattened to length $4 \cdot 3 = 12$, projected through $W_E \in \mathbb{R}^{12 \times 4}$:</span>

* <span style="font-size: 14px;">$z_1^{(0)} = [0.42, -0.18, 0.31, 0.05]$ (top-left)</span>
* <span style="font-size: 14px;">$z_2^{(0)} = [-0.11, 0.53, 0.27, -0.34]$ (top-right)</span>
* <span style="font-size: 14px;">$z_3^{(0)} = [0.29, 0.14, -0.22, 0.61]$ (bottom-left)</span>
* <span style="font-size: 14px;">$z_4^{(0)} = [0.55, -0.07, 0.48, 0.19]$ (bottom-right)</span>

<span style="font-size: 14px;">**Stage 2: CLS + position.** Prepend $z_{\text{cls}} = [0.10, 0.10, 0.10, 0.10]$. Sequence: 5 tokens. Add $E_{\text{pos}} \in \mathbb{R}^{5 \times 4}$:</span>

* <span style="font-size: 14px;">CLS: $[0.10 + 0.01, 0.10 - 0.02, 0.10 + 0.03, 0.10 - 0.01] = [0.11, 0.08, 0.13, 0.09]$</span>
* <span style="font-size: 14px;">Patch 1: $[0.44, -0.17, 0.30, 0.08]$, patches 2-4 similar.</span>

<span style="font-size: 14px;">**Stage 3: Encoder block.** Focus on CLS token.</span>

<span style="font-size: 14px;">LayerNorm CLS $[0.11, 0.08, 0.13, 0.09]$: mean $= 0.1025$, std $= 0.0192$. Normalized: $[0.39, -1.17, 1.43, -0.65]$.</span>

<span style="font-size: 14px;">MSA: CLS query attends to all 5 positions (no mask). After 2-head attention and $W_O$ projection, output $= [0.06, -0.03, 0.08, -0.02]$. Residual: $[0.17, 0.05, 0.21, 0.07]$.</span>

<span style="font-size: 14px;">MLP: LayerNorm, expand to 8 dims with GELU, compress back to 4. Output $= [0.04, -0.01, 0.05, 0.02]$. Residual: $[0.21, 0.04, 0.26, 0.09]$.</span>

<span style="font-size: 14px;">**Stage 4: Classification head.** Extract CLS: $z_0^{(1)} = [0.21, 0.04, 0.26, 0.09]$. LayerNorm: $h_f = [0.64, -1.17, 1.17, -0.64]$. Project through $W_{\text{head}} \in \mathbb{R}^{4 \times 3}$:</span>

* <span style="font-size: 14px;">Class 0: $0.64(0.3) + (-1.17)(-0.1) + 1.17(0.2) + (-0.64)(0.4) = 0.192 + 0.117 + 0.234 - 0.256 = 0.287$</span>
* <span style="font-size: 14px;">Class 1: $0.64(-0.2) + (-1.17)(0.5) + 1.17(0.1) + (-0.64)(-0.3) = -0.128 - 0.585 + 0.117 + 0.192 = -0.404$</span>
* <span style="font-size: 14px;">Class 2: $0.64(0.1) + (-1.17)(0.3) + 1.17(-0.4) + (-0.64)(0.2) = 0.064 - 0.351 - 0.468 - 0.128 = -0.883$</span>

<span style="font-size: 14px;">Logits: $[0.287, -0.404, -0.883]$. Predicted class: 0. No softmax. The CLS token started with no image content but aggregated all patch information through bidirectional attention.</span>

---

## <span style="font-size: 16px;">ViT's Impact</span>

<span style="font-size: 14px;">ViT proved that Transformers work for vision without any convolutional layers. This sparked a transformation of the field:</span>

* <span style="font-size: 14px;">**DeiT (2021):** Knowledge distillation from a CNN teacher made ViT data-efficient, achieving competitive results with only ImageNet-1K.</span>
* <span style="font-size: 14px;">**Swin Transformer (2021):** Shifted windows for local attention created hierarchical features, enabling dense prediction tasks like detection and segmentation.</span>
* <span style="font-size: 14px;">**MAE (2022):** Masked 75% of patches and trained reconstruction, enabling powerful self-supervised pre-training for ViT.</span>
* <span style="font-size: 14px;">**CLIP (2021):** Paired a ViT image encoder with a text encoder for zero-shot classification from natural language supervision.</span>

---

## <span style="font-size: 16px;">Pitfalls</span>

<span style="font-size: 14px;">**1. Adding a causal mask to self-attention.**</span>

<span style="font-size: 14px;">ViT uses bidirectional attention with no causal mask. Adding one would prevent patches from attending to later positions in the flattened sequence. Since "later" means "further right or down" in the image, a causal mask creates arbitrary asymmetry where top-left patches see less context than bottom-right patches.</span>

<span style="font-size: 14px;">**2. Forgetting position embeddings.**</span>

<span style="font-size: 14px;">Without position embeddings, the Transformer treats patches as an unordered set. Permuting patches would produce identical output. The model cannot learn spatial relationships without positional information.</span>

<span style="font-size: 14px;">**3. Wrong patch size or miscounting patches.**</span>

<span style="font-size: 14px;">$P$ must evenly divide $H$ and $W$. The position embedding matrix has $N + 1$ entries (the +1 is for CLS). Using $P = 14$ on a 224x224 image gives 256 patches instead of 196, requiring a different position embedding size.</span>

<span style="font-size: 14px;">**4. Applying softmax to the output.**</span>

<span style="font-size: 14px;">The forward pass outputs raw logits. Cross-entropy loss applies log-softmax internally. Adding softmax in the model creates double-softmax, compressing gradients toward zero and stopping learning.</span>

<span style="font-size: 14px;">**5. Not extracting the CLS token.**</span>

<span style="font-size: 14px;">The classification head operates only on position 0 (CLS), not the full sequence. Passing all $N + 1$ tokens through the linear head produces a $(N+1) \times K$ output instead of a $K$-dimensional logit vector. Average pooling is a valid alternative but architecturally different from what ViT specifies.</span>

---