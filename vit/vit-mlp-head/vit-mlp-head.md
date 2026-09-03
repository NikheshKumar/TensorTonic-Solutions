# <span style="font-size: 20px;">Classification Head</span>

<span style="font-size: 14px;">The classification head is the final stage of the Vision Transformer (ViT) that converts the encoder's representation into class predictions. Introduced by Dosovitskiy et al. (2020) in "An Image is Worth 16x16 Words," it extracts the CLS token, normalizes it with LayerNorm, and projects it through a single linear layer to produce raw logits.</span>

---

## <span style="font-size: 16px;">What It Is</span>

<span style="font-size: 14px;">The classification head maps the CLS token's representation to class logits. After the image has been split into patches, linearly embedded, augmented with a learnable CLS token, and processed through multiple Transformer encoder blocks, the CLS token at position 0 of the final encoder output carries a global summary of the entire image. The classification head extracts this single vector, normalizes it, and projects it to a vector of dimension $C$ (the number of classes), where each element is the unnormalized score (logit) for the corresponding class.</span>

<span style="font-size: 14px;">The operation is deliberately simple. The encoder has already done the heavy lifting of learning patch interactions through self-attention. The classification head's job is to translate the encoder's internal representation into a format suitable for computing cross-entropy loss during training or argmax during inference. No softmax is applied inside the head -- the loss function handles the softmax internally for numerical stability.</span>

<span style="font-size: 14px;">In this fine-tuning version, the head consists of exactly three sequential operations: extract position 0, apply LayerNorm (normalize without learnable scale or shift), and multiply by a weight matrix $W_{\text{head}} \in \mathbb{R}^{D \times C}$. No bias terms, no activation functions, and no hidden layers.</span>

---

## <span style="font-size: 16px;">Key Equations</span>

<span style="font-size: 14px;">Given the encoder output $z_L \in \mathbb{R}^{B \times (N+1) \times D}$ where $B$ is batch size, $N$ is the number of patches, and $D$ is the hidden dimension, the classification head proceeds in three steps.</span>

<span style="font-size: 14px;">**Step 1 -- Extract the CLS token.** Select position 0 along the sequence dimension:</span>

$$
h_{\text{cls}} = z_L[:, 0, :] \quad \in \mathbb{R}^{B \times D}
$$

<span style="font-size: 14px;">This slicing operation picks out the first token from each example in the batch. The CLS token was prepended at the input and has attended to every patch token through all encoder layers.</span>

<span style="font-size: 14px;">**Step 2 -- Apply LayerNorm.** Normalize across the feature dimension $D$:</span>

$$
\mu = \frac{1}{D} \sum_{i=1}^{D} h_{\text{cls}, i}, \quad \sigma^2 = \frac{1}{D} \sum_{i=1}^{D} (h_{\text{cls}, i} - \mu)^2
$$

$$
\hat{h}_i = \frac{h_{\text{cls}, i} - \mu}{\sqrt{\sigma^2 + \epsilon}}
$$

<span style="font-size: 14px;">where $\epsilon$ is a small constant (typically $10^{-5}$) for numerical stability. In this problem, the LayerNorm uses no learnable scale ($\gamma$) or shift ($\beta$) parameters -- it performs pure statistical normalization only. Each sample in the batch is normalized independently.</span>

<span style="font-size: 14px;">**Step 3 -- Linear projection.** Map from hidden dimension to class logits:</span>

$$
\text{logits} = \hat{h} \cdot W_{\text{head}} \quad \in \mathbb{R}^{B \times C}
$$

<span style="font-size: 14px;">where $W_{\text{head}} \in \mathbb{R}^{D \times C}$ is the projection matrix and $C$ is the number of classes. No bias is added and no softmax is applied.</span>

---

## <span style="font-size: 16px;">Why the CLS Token for Classification</span>

<span style="font-size: 14px;">The CLS token is a learnable embedding prepended to the patch sequence before the encoder. It does not correspond to any spatial region of the image -- it is a purely abstract token whose sole purpose is to accumulate global information through self-attention.</span>

<span style="font-size: 14px;">**Attended to all patches.** In every encoder block, the CLS token computes attention over the full sequence, including all $N$ patch tokens. Through $L$ layers of self-attention, it accumulates multi-hop interactions between patches. By the final layer, it has aggregated information from every spatial position in the image, weighted by learned attention patterns.</span>

<span style="font-size: 14px;">**Consistent position.** The CLS token always occupies position 0 in the sequence. Unlike patch tokens, whose position encodings correspond to spatial locations in the image, the CLS token has a fixed, image-independent position. The classification head always knows exactly where to find the global representation, regardless of image resolution or patch configuration.</span>

<span style="font-size: 14px;">**No spatial bias.** Using any particular patch token for classification would introduce a spatial bias -- the prediction would disproportionately reflect that patch's region. The CLS token avoids this because it starts with no spatial content and builds its representation entirely through attention. Dosovitskiy et al. (2020) note that global average pooling (GAP) over all patch tokens works comparably, but they follow the BERT convention of using a CLS token as the default.</span>

---

## <span style="font-size: 16px;">The LayerNorm Step</span>

<span style="font-size: 14px;">Before the linear projection, the CLS vector is normalized using LayerNorm. This step serves several important purposes in the classification pipeline.</span>

<span style="font-size: 14px;">**Stabilizes input to the classifier.** The encoder output's magnitude can vary significantly across different inputs and training stages. Without normalization, the linear classifier $W_{\text{head}}$ must simultaneously learn class boundaries and handle scale variation. LayerNorm decouples these concerns: it standardizes the scale, letting the classifier focus purely on directional information.</span>

<span style="font-size: 14px;">**Consistent with ViT architecture.** The original ViT applies a final LayerNorm after the last encoder block, before the classification head. This mirrors the GPT-2 convention of placing a final LayerNorm after the transformer stack. Without it, residual connections could cause the output magnitude to grow proportionally with depth.</span>

<span style="font-size: 14px;">**No learnable parameters in this version.** This problem's LayerNorm has no $\gamma$ (scale) or $\beta$ (shift) parameters. It performs pure normalization: subtract the mean, divide by the standard deviation. In practice, the ViT implementation uses a standard LayerNorm with learnable affine parameters, but the core normalization behavior is identical.</span>

<span style="font-size: 14px;">**Per-sample operation.** The normalization statistics are computed independently for each CLS vector in the batch. There is no interaction between batch elements -- each image's representation is normalized using only its own feature statistics. This makes the behavior identical during training and inference.</span>

---

## <span style="font-size: 16px;">Pre-training vs Fine-tuning Head</span>

<span style="font-size: 14px;">Dosovitskiy et al. (2020) explicitly describe two different classification head architectures depending on the training phase. The paper states: "The classification head is implemented by a MLP with one hidden layer at pre-training time and by a single linear layer at fine-tuning time."</span>

<span style="font-size: 14px;">**Pre-training head (MLP).** During pre-training on large-scale datasets like JFT-300M (303 million images, 18,291 classes) or ImageNet-21k (14 million images, 21,843 classes), the classification head uses a two-layer MLP with a GELU activation:</span>

$$
\text{logits} = W_2 \cdot \text{GELU}(W_1 \cdot \hat{h})
$$

<span style="font-size: 14px;">where $W_1 \in \mathbb{R}^{D \times D}$ projects to a hidden representation and $W_2 \in \mathbb{R}^{D \times C_{\text{pretrain}}}$ projects to the pre-training class count. The hidden layer adds nonlinearity, giving the head more capacity for learning with a large number of classes.</span>

<span style="font-size: 14px;">**Fine-tuning head (single linear layer).** When transferring to a downstream task like ImageNet-1k (1,000 classes), the pre-training MLP is discarded and replaced with a single linear layer $\text{logits} = \hat{h} \cdot W_{\text{head}}$ where $W_{\text{head}} \in \mathbb{R}^{D \times C_{\text{finetune}}}$ is randomly initialized. The simpler head is sufficient because the encoder already produces highly structured representations from pre-training.</span>

<span style="font-size: 14px;">**This problem implements the fine-tuning version** -- the single linear projection from $D$ to $C$ with no hidden layer, no bias, and no activation function.</span>

---

## <span style="font-size: 16px;">Paper Context: Dosovitskiy et al. (2020)</span>

<span style="font-size: 14px;">"An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale" demonstrated that a pure Transformer applied directly to sequences of image patches can match or exceed state-of-the-art CNNs when pre-trained on sufficient data.</span>

<span style="font-size: 14px;">**Architecture.** The full ViT pipeline is: (1) split the image into fixed-size patches (16x16 pixels), (2) linearly embed each patch into a $D$-dimensional vector, (3) prepend a learnable CLS token, (4) add learnable position embeddings, (5) process through $L$ Transformer encoder blocks, (6) extract the CLS token, (7) apply LayerNorm, (8) project to class logits. The classification head covers steps 6 through 8.</span>

<span style="font-size: 14px;">**Scale results.** The strongest ViT variants (ViT-Large/16 and ViT-Huge/14) were pre-trained on JFT-300M. At this scale, ViT outperformed Big Transfer (BiT) ResNets on ImageNet, showing that Transformers require more data to compensate for lacking inductive biases like translation equivariance, but match or exceed CNNs when that data is available.</span>

<span style="font-size: 14px;">**Fine-tuning protocol.** The pre-trained MLP head is removed and a zero-initialized $D \times 1000$ linear layer is attached. The model is trained at higher resolution (384x384 vs 224x224 for ViT-Large/16), with position embeddings 2D-interpolated for the increased sequence length. The classification head's simplicity makes this swap trivial.</span>

---

## <span style="font-size: 16px;">Numerical Example</span>

<span style="font-size: 14px;">Consider $B = 2$ images with hidden dimension $D = 4$ and $C = 3$ classes.</span>

<span style="font-size: 14px;">**CLS tokens extracted at position 0:**</span>

$$
h_{\text{cls}} = \begin{pmatrix} 2.0 & -1.0 & 0.0 & 3.0 \\ 1.0 & 1.0 & -1.0 & -1.0 \end{pmatrix}
$$

<span style="font-size: 14px;">**LayerNorm (sample 1).** For $h_1 = [2.0, -1.0, 0.0, 3.0]$:</span>

$$
\mu_1 = \frac{2.0 - 1.0 + 0.0 + 3.0}{4} = 1.0, \quad \sigma_1^2 = \frac{1.0 + 4.0 + 1.0 + 4.0}{4} = 2.5
$$

$$
\hat{h}_1 = \frac{[1.0, ; -2.0, ; -1.0, ; 2.0]}{\sqrt{2.5}} \approx [0.6325, ; -1.2649, ; -0.6325, ; 1.2649]
$$

<span style="font-size: 14px;">**LayerNorm (sample 2).** For $h_2 = [1.0, 1.0, -1.0, -1.0]$:</span>

$$
\mu_2 = 0.0, \quad \sigma_2^2 = 1.0, \quad \hat{h}_2 \approx [1.0, ; 1.0, ; -1.0, ; -1.0]
$$

<span style="font-size: 14px;">Sample 2 already has zero mean and unit variance, so normalization leaves it unchanged.</span>

<span style="font-size: 14px;">**Linear projection** with $W_{\text{head}} \in \mathbb{R}^{4 \times 3}$:</span>

$$
W_{\text{head}} = \begin{pmatrix} 0.5 & -0.2 & 0.1 \\ 0.3 & 0.4 & -0.5 \\ -0.1 & 0.6 & 0.2 \\ 0.7 & -0.3 & 0.8 \end{pmatrix}
$$

<span style="font-size: 14px;">**Sample 1 logits:** $\hat{h}_1 \cdot W_{\text{head}}$:</span>

$$
\text{logit}_{1,1} = (0.6325)(0.5) + (-1.2649)(0.3) + (-0.6325)(-0.1) + (1.2649)(0.7) = 0.886
$$

$$
\text{logit}_{1,2} = (0.6325)(-0.2) + (-1.2649)(0.4) + (-0.6325)(0.6) + (1.2649)(-0.3) = -1.392
$$

$$
\text{logit}_{1,3} = (0.6325)(0.1) + (-1.2649)(-0.5) + (-0.6325)(0.2) + (1.2649)(0.8) = 1.581
$$

<span style="font-size: 14px;">**Sample 2 logits:** $\hat{h}_2 \cdot W_{\text{head}}$:</span>

$$
\text{logit}_{2,1} = 0.5 + 0.3 + 0.1 - 0.7 = 0.2, \quad \text{logit}_{2,2} = -0.2 + 0.4 - 0.6 + 0.3 = -0.1
$$

$$
\text{logit}_{2,3} = 0.1 - 0.5 - 0.2 - 0.8 = -1.4
$$

<span style="font-size: 14px;">**Final output:**</span>

$$
\text{logits} = \begin{pmatrix} 0.886 & -1.392 & 1.581 \\ 0.200 & -0.100 & -1.400 \end{pmatrix}
$$

<span style="font-size: 14px;">Sample 1 is classified as class 3 (highest logit at index 2). Sample 2 is classified as class 1 (highest logit at index 0). These are raw logits -- softmax would give probabilities, but the head does not apply it.</span>

---

## <span style="font-size: 16px;">Connection to BERT's Pooler</span>

<span style="font-size: 14px;">The ViT classification head is directly inspired by BERT's approach to sequence-level classification. Both architectures prepend a CLS token, process it through Transformer encoder layers, and extract it for classification. The differences lie in post-extraction processing.</span>

<span style="font-size: 14px;">**BERT's pooler.** BERT passes the extracted CLS token through a dense layer with $\tanh$ activation: $\text{pooled} = \tanh(W_{\text{pool}} \cdot h_{\text{cls}} + b_{\text{pool}})$. The $\tanh$ squashes output to $[-1, 1]$, providing bounded representations. A separate classification layer then maps to class logits.</span>

<span style="font-size: 14px;">**ViT's head.** ViT applies LayerNorm (not $\tanh$) and projects directly to logits with no intermediate nonlinearity. LayerNorm normalizes to approximately zero mean and unit variance but does not bound the range. The absence of $\tanh$ means no saturation -- the full dynamic range is preserved for the linear classifier.</span>

<span style="font-size: 14px;">**Design rationale.** BERT's pooler was designed for pre-training with next-sentence prediction, where bounded representations help the binary classification objective. ViT's head is designed purely for classification, where cross-entropy loss handles normalization through its internal softmax.</span>

---

## <span style="font-size: 16px;">Pitfalls</span>

* <span style="font-size: 14px;">**Extracting the wrong position.** The CLS token is always at position 0. A common mistake is extracting position $-1$ (the last token) or averaging all positions. Extracting the wrong position produces a patch-specific representation instead of the global summary, leading to spatially-biased predictions.</span>

* <span style="font-size: 14px;">**Applying softmax inside the head.** The classification head returns raw logits, not probabilities. Cross-entropy loss functions apply log-softmax internally for numerical stability. Adding softmax before the loss means it is applied twice, producing a "flattened" distribution that impedes gradient flow and severely slows convergence.</span>

* <span style="font-size: 14px;">**Forgetting LayerNorm before projection.** Skipping normalization means the linear classifier receives activations whose scale depends on input content and encoder depth. This makes the effective learning rate input-dependent, causing unstable decision boundaries across different input magnitudes.</span>

* <span style="font-size: 14px;">**Using learnable LayerNorm parameters when not specified.** This problem specifies LayerNorm without learnable $\gamma$ and $\beta$. Adding learnable affine parameters changes the function's behavior and produces outputs that differ from expected test case values.</span>

* <span style="font-size: 14px;">**Wrong $W_{\text{head}}$ dimensions.** The weight matrix must have shape $(D, C)$ -- hidden dimension to number of classes. Transposing to $(C, D)$ causes a dimension mismatch: $\hat{h} \in \mathbb{R}^{B \times D}$ times $W \in \mathbb{R}^{C \times D}$ fails because inner dimensions $D$ and $C$ do not match.</span>

* <span style="font-size: 14px;">**Adding a bias term to the linear projection.** The fine-tuning classification head uses a weight-only projection with no bias. Adding a bias vector $b \in \mathbb{R}^C$ shifts all logits by a class-specific constant, changing decision boundaries and violating the problem specification.</span>

---