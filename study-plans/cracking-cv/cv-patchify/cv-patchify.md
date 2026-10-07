# <span style="font-size: 20px;">ViT Patchify</span>

<span style="font-size: 14px;">Patchify is the first operation in a Vision Transformer (Dosovitskiy et al., 2020, "An Image Is Worth 16x16 Words"). It cuts an image into a grid of non-overlapping $P \times P$ patches, flattens each patch into a vector, and linearly projects it into a token embedding. This turns a 2D image into a 1D sequence of tokens that a standard Transformer encoder can consume, with no convolutions anywhere in the backbone.</span>

---

## <span style="font-size: 16px;">What It Does</span>

<span style="font-size: 14px;">A Transformer operates on a sequence of token vectors, but an image is a dense grid of pixels. The ViT paper bridges this gap with the simplest possible adapter: chop the image into fixed-size square patches and treat each patch as one token, exactly as words are tokens in NLP. The title of the paper, "An image is worth 16x16 words", refers to the default patch size $P = 16$.</span>

<span style="font-size: 14px;">Before ViT, applying Transformers to images required either restricting attention to local neighborhoods or combining attention with convolutions, because full pixel-level self-attention does not scale. The ViT contribution was to show that a near-standard NLP Transformer, applied to patch tokens, matches or beats convolutional networks once enough pretraining data is available. Patchify is the single piece of glue that makes this possible.</span>

<span style="font-size: 14px;">Given a batch of images of shape $(B, C, H, W)$ with $H \bmod P = W \bmod P = 0$, the operation produces a row-major grid of patches and flattens each one in $(p_1, p_2, c)$ order (row, then column, then channel). This is the well-known einops pattern:</span>

$$
\texttt{b c (h p1) (w p2)} \rightarrow \texttt{b (h w) (p1 p2 c)}
$$

<span style="font-size: 14px;">The result is a tensor of shape $(B, N, P^2 C)$, where $N$ is the number of tokens per image. After flattening, a learned linear layer projects each $P^2 C$ vector to the model dimension $D$, producing the actual patch embeddings fed to the encoder.</span>

---

## <span style="font-size: 16px;">Why Patches Instead of Pixels</span>

<span style="font-size: 14px;">The obvious alternative, treating every pixel as a token, is computationally hopeless. Self-attention has cost $O(N^2 D)$ in the number of tokens $N$. A $224 \times 224$ image has $50{,}176$ pixels, so pixel-level attention would build a $50{,}176 \times 50{,}176$ score matrix, around $2.5$ billion entries per layer per head. Patching at $P = 16$ cuts the token count to $196$, a $256$-fold reduction in $N$ and roughly a $65{,}000$-fold reduction in the attention matrix size.</span>

<span style="font-size: 14px;">Patching also aggregates local pixels before any attention runs, so each token already summarizes a small image region. This makes patches a reasonable atomic unit: large enough to be cheap, small enough that a $16 \times 16$ region of a natural image is still fairly homogeneous. The paper deliberately keeps this front-end trivial so that almost all modeling capacity lives in the generic Transformer, letting the same architecture transfer ideas straight from NLP.</span>

---

## <span style="font-size: 16px;">The Shape Arithmetic</span>

<span style="font-size: 14px;">Two quantities define the token grid. The number of patches along each spatial axis is:</span>

$$
N_h = \frac{H}{P}, \quad N_w = \frac{W}{P}, \quad N = N_h \, N_w
$$

<span style="font-size: 14px;">Each patch contains $P \times P$ pixels across $C$ channels, so its flattened length is:</span>

$$
P^2 C
$$

<span style="font-size: 14px;">For the canonical ViT configuration with a $224 \times 224$ RGB image ($C = 3$) and $P = 16$:</span>

* <span style="font-size: 14px;">$N_h = N_w = 224 / 16 = 14$, so $N = 14 \times 14 = 196$ patch tokens</span>
* <span style="font-size: 14px;">Each flattened patch has length $16^2 \times 3 = 768$</span>
* <span style="font-size: 14px;">The linear projection maps $768 \to D$, and for ViT-Base $D = 768$ too, so the projection is a square matrix</span>

<span style="font-size: 14px;">In the full model a learnable `[CLS]` token is prepended, giving sequence length $N + 1 = 197$, and a learned positional embedding of shape $(N + 1, D)$ is added. Patchify itself produces only the $N$ patch vectors; the class token and positions are added afterwards.</span>

<span style="font-size: 14px;">The arithmetic generalizes cleanly. For ViT-Large with $P = 16$ the token count is unchanged at $196$ because $N$ depends only on image size and patch size, not on $D$; only the embedding width grows to $D = 1024$. For the larger-patch ViT-B/32 variant, $N_h = N_w = 224 / 32 = 7$ gives $N = 49$ tokens, roughly a quarter of the ViT-B/16 sequence, which makes it about four times cheaper in attention at the cost of coarser spatial resolution. This direct, predictable relationship between patch size, token count, and compute is one reason patching is such a convenient design knob.</span>

---

## <span style="font-size: 16px;">Flatten Order Matters</span>

<span style="font-size: 14px;">The flatten order is $(p_1, p_2, c)$: iterate the patch's rows slowest, then columns, then channels fastest. Equivalently, within a patch the pixel at intra-patch row $p_1$ and column $p_2$ for channel $c$ lands at index $p_1 \cdot (P \cdot C) + p_2 \cdot C + c$ in the flattened vector. The patch grid itself is walked in row-major order, top-left patch first and bottom-right patch last, matching `(h w)` in the einops string.</span>

<span style="font-size: 14px;">There are two nested orderings that must not be confused. The **outer** ordering walks over the $N = N_h N_w$ patches and decides which token each patch becomes. The **inner** ordering flattens the $P^2 C$ scalars inside a single patch and decides the internal layout of one token vector. Patchify uses row-major for both, but they are independent choices.</span>

<span style="font-size: 14px;">The exact order is a convention, not a mathematical necessity. The linear projection can in principle absorb any consistent permutation of the inner layout, since it is a dense $P^2 C \to D$ matrix and a fixed permutation of its input columns is equivalent to permuting its rows. But the convention must be consistent between training and inference, and it must match how the positional embeddings are laid out, otherwise the model sees scrambled spatial structure.</span>

<span style="font-size: 14px;">In practice the convolutional implementation fixes this order for you: a `Conv2d` kernel of shape $(D, C, P, P)$ contracts over $(C, p_1, p_2)$, which corresponds to a specific flatten convention. Reference ViT checkpoints are trained with that convention, so reusing pretrained weights requires reproducing it exactly.</span>

---

## <span style="font-size: 16px;">From Flattened Patch to Token Embedding</span>

<span style="font-size: 14px;">Patchify produces the raw flattened vectors; the ViT embedding stage adds a learned linear projection $E \in \mathbb{R}^{(P^2 C) \times D}$ that maps each patch to the model dimension. The paper calls these **patch embeddings** and writes the stage as $z_0 = [x_\text{class}; \, x_p^1 E; \, x_p^2 E; \ldots; \, x_p^N E] + E_\text{pos}$, where $x_p^k$ is the $k$-th flattened patch.</span>

<span style="font-size: 14px;">Three things are stacked here. First, the linear projection $E$ gives every patch a learned $D$-dimensional representation. Second, a single learnable class token $x_\text{class}$ is prepended; its final-layer state is used for classification, mirroring BERT's `[CLS]` token. Third, a learned positional embedding $E_\text{pos} \in \mathbb{R}^{(N+1) \times D}$ is added so the model can tell tokens apart by location. The paper found 1D learned positions work as well as 2D-aware ones, because the model learns the grid structure anyway.</span>

---

## <span style="font-size: 16px;">Equivalence to a Strided Convolution</span>

<span style="font-size: 14px;">Flatten-then-project is mathematically identical to a single 2D convolution with kernel size $P$, stride $P$, $C$ input channels, and $D$ output channels. Because the stride equals the kernel size, the receptive fields tile the image with no overlap and no gaps. Each output spatial location is one patch token, and the $D$ output channels are exactly the projected embedding.</span>

<span style="font-size: 14px;">This is why most efficient ViT implementations use `nn.Conv2d(C, D, kernel_size=P, stride=P)` and then flatten the $(B, D, N_h, N_w)$ output to $(B, N, D)$. The convolution kernel of shape $(D, C, P, P)$ holds exactly the same parameters as the $(P^2 C, D)$ projection matrix, just reshaped. The paper notes this equivalence and frames patch embedding as the only place convolutional inductive bias enters the architecture. The conv form is also faster: a single strided convolution is a highly optimized GPU primitive, whereas an explicit reshape-then-matmul materializes the large intermediate patch tensor and is more memory-bound.</span>

<span style="font-size: 14px;">The deeper point from the paper: ViT deliberately strips out the translation-equivariance and locality priors that CNNs bake in. A CNN slides small filters across the whole image, so it assumes that the same local feature detector is useful everywhere and that nearby pixels matter most. ViT makes neither assumption beyond the single patch-embedding conv. With far less inductive bias, ViT underperforms ResNets when trained on ImageNet-1k alone, but overtakes them when pretrained on very large datasets such as JFT-300M. The data effectively teaches the priors that a CNN gets for free.</span>

<span style="font-size: 14px;">Patchify is where that trade-off begins: it is the minimal, almost prior-free way to feed pixels into a Transformer. The only spatial structure it preserves is which pixels share a patch; everything beyond that, including the relative arrangement of patches, must be learned from positional embeddings and attention. This is exactly why ViT is so data-hungry and why patch size acts as a dial trading locality bias against sequence length.</span>

---

## <span style="font-size: 16px;">Numerical Example</span>

<span style="font-size: 14px;">Take a single grayscale image, $B = 1$, $C = 1$, $H = W = 4$, patch size $P = 2$. The pixel grid is:</span>

$$
\begin{pmatrix} 1 & 2 & 3 & 4 \\ 5 & 6 & 7 & 8 \\ 9 & 10 & 11 & 12 \\ 13 & 14 & 15 & 16 \end{pmatrix}
$$

<span style="font-size: 14px;">Here $N_h = N_w = 4 / 2 = 2$, so $N = 4$ patches, and each flattened patch has length $P^2 C = 4$.</span>

<span style="font-size: 14px;">Walking the patch grid row-major, and flattening each patch in row-then-column order:</span>

* <span style="font-size: 14px;">**Patch (0,0)** top-left: rows 1-2, cols 1-2 = $[1, 2, 5, 6]$</span>
* <span style="font-size: 14px;">**Patch (0,1)** top-right: rows 1-2, cols 3-4 = $[3, 4, 7, 8]$</span>
* <span style="font-size: 14px;">**Patch (1,0)** bottom-left: rows 3-4, cols 1-2 = $[9, 10, 13, 14]$</span>
* <span style="font-size: 14px;">**Patch (1,1)** bottom-right: rows 3-4, cols 3-4 = $[11, 12, 15, 16]$</span>

<span style="font-size: 14px;">The output has shape $(1, 4, 4)$: four tokens, each a length-4 vector. Note how pixel 3 is not adjacent to pixel 2 in the token sequence even though they are adjacent in the image, the model recovers spatial relationships only through positional embeddings and attention.</span>

<span style="font-size: 14px;">To see the conv equivalence concretely, suppose the projection is the identity map ($D = P^2 C = 4$ with $W$ equal to the $4 \times 4$ identity). Then a $\texttt{Conv2d}$ with kernel size 2 and stride 2 whose four filters each pick out one of the four intra-patch positions produces output channels $[1, 2, 5, 6]$ at spatial location $(0,0)$, which is precisely patch $(0,0)$ above. Flattening the conv output from $(1, 4, 2, 2)$ to $(1, 4, 4)$ recovers the same token sequence. The stride-equals-kernel choice guarantees the four $2 \times 2$ receptive fields tile the $4 \times 4$ image with no overlap.</span>

---

## <span style="font-size: 16px;">Variants and Modern Context</span>

* <span style="font-size: 14px;">**Patch size as a speed knob.** Tokens scale as $N = HW / P^2$ and attention cost scales as $O(N^2)$, so halving $P$ quadruples $N$ and increases attention cost roughly sixteen-fold. Smaller patches give finer detail at much higher cost. ViT ships variants like ViT-B/16 and ViT-B/32 named by their patch size.</span>
* <span style="font-size: 14px;">**Overlapping patch stems.** Later works (T2T-ViT, Swin, PVT) found that strictly non-overlapping patches lose fine boundary information, and use overlapping convolutional stems or progressive patch merging to inject locality back in.</span>
* <span style="font-size: 14px;">**MAE masking.** Masked Autoencoders (He et al., 2021) operate directly on these patch tokens, masking 75 percent of them and reconstructing the missing patches, which only makes sense because patchify defines a clean per-token unit.</span>
* <span style="font-size: 14px;">**Hybrid stems.** The ViT paper also tests a hybrid where a CNN produces the feature map and patches are taken from that map instead of raw pixels, useful at smaller data scales.</span>
* <span style="font-size: 14px;">**Flexible resolution.** Because $N = HW / P^2$ depends on input size, changing the test resolution changes the token count. ViT handles this by interpolating the learned positional embeddings to the new grid, a trick that later models like DeiT and DINOv2 rely on for multi-scale evaluation.</span>
* <span style="font-size: 14px;">**Patch dropout and masking.** Because each token corresponds to a clean spatial unit, whole patches can be dropped to speed training (FlexiViT, patch dropout) or masked for self-supervision (MAE, BEiT), something pixel-level tokenization would make far messier.</span>

---

## <span style="font-size: 16px;">Pitfalls</span>

* <span style="font-size: 14px;">**Wrong flatten order.** Flattening as $(c, p_1, p_2)$ (channel-major) instead of $(p_1, p_2, c)$ produces a different vector layout. It will not crash because the length $P^2 C$ is unchanged, but it silently scrambles the projection and breaks any pretrained weights that assumed the other order.</span>
* <span style="font-size: 14px;">**Non-divisible dimensions.** If $H$ or $W$ is not a multiple of $P$, a naive reshape either errors or drops a partial strip of pixels at the bottom or right edge. Always pad or resize so that $H \bmod P = W \bmod P = 0$ before patchifying.</span>
* <span style="font-size: 14px;">**Confusing patch-grid order with intra-patch order.** The outer loop over patches and the inner flatten of pixels within a patch are two separate row-major orderings. Mixing them up, for example walking patches column-major while flattening pixels row-major, corrupts the token sequence in a way that is hard to debug.</span>
* <span style="font-size: 14px;">**Forgetting that positional info is gone after flatten.** Patchify destroys the 2D layout; absolute pixel adjacency no longer implies sequence adjacency. Omitting or mismatching the positional embedding leaves the model with a bag of patches and no sense of where they came from.</span>

---