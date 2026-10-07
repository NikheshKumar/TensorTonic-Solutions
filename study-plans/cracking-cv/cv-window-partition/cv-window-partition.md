# <span style="font-size: 20px;">Window Partition and Reverse</span>

<span style="font-size: 14px;">Window partition is the operation at the heart of the Swin Transformer (Liu et al., 2021, "Swin Transformer: Hierarchical Vision Transformer using Shifted Windows"). It reshapes a feature map into a set of non-overlapping local windows so that self-attention can run inside each window independently, turning the quadratic cost of global attention into a cost that is linear in the number of patches. Its exact inverse, window reverse, stitches the windows back into the original feature map.</span>

---

## <span style="font-size: 16px;">The Problem It Solves</span>

<span style="font-size: 14px;">A plain ViT runs global self-attention: every token attends to every other token, costing $O(N^2)$ in the token count $N$. For a $224 \times 224$ image at patch size 16 that is $196$ tokens, manageable. But Swin targets dense vision tasks like detection and segmentation that need high-resolution feature maps with thousands of tokens, where $O(N^2)$ becomes prohibitive in both time and memory.</span>

<span style="font-size: 14px;">Swin's answer is **local attention**: partition the feature map into fixed-size windows (the paper uses $M = 7$, so $7 \times 7$ patches per window) and run self-attention only among the tokens inside each window. Because window size is constant, the cost per window is constant, and the total cost grows linearly with the number of windows, hence linearly with image area.</span>

<span style="font-size: 14px;">This mirrors a long-standing idea in vision: convolutions are also local, processing a small neighborhood at a time. Swin reintroduces locality as an explicit inductive bias that plain ViT discarded, which is part of why Swin trains well on ImageNet-1k without the massive pretraining ViT needs. The trade-off is that a single window-attention layer can only mix tokens within one window; Swin recovers global context by stacking many layers and alternating window positions, much as a deep CNN grows its receptive field layer by layer.</span>

---

## <span style="font-size: 16px;">Linear vs Quadratic Complexity</span>

<span style="font-size: 14px;">For a feature map with $h \times w$ patch tokens and window side $M$, the paper gives the attention cost of the two schemes. Global multi-head self-attention is quadratic in the number of tokens:</span>

$$
\Omega(\text{MSA}) = 4 h w C^2 + 2 (h w)^2 C
$$

<span style="font-size: 14px;">Window-based attention replaces the quadratic term with one that is linear in $hw$:</span>

$$
\Omega(\text{W-MSA}) = 4 h w C^2 + 2 M^2 h w C
$$

<span style="font-size: 14px;">The expensive $(hw)^2$ term becomes $M^2 hw$. Since $M$ is a small constant (49 tokens per window), the attention term now scales linearly with the spatial size $hw$. This is the key enabler for Swin to act as a general-purpose backbone at high resolution, where a global ViT would run out of memory.</span>

<span style="font-size: 14px;">A concrete sense of scale: a feature map with $56 \times 56 = 3136$ tokens has a global attention matrix of about $9.8$ million entries per head. With $M = 7$ windows there are $64$ windows each with a $49 \times 49$ matrix, totaling about $154{,}000$ entries, a reduction of roughly $64 \times$. Both schemes share the same $4 h w C^2$ projection term, so the saving is entirely in the attention matrix, exactly where global attention blows up. Swin also builds a hierarchy: after each stage it merges $2 \times 2$ neighboring patches, halving $h$ and $w$ and doubling $C$, so the token count shrinks toward the deeper stages much like a CNN's spatial downsampling.</span>

---

## <span style="font-size: 16px;">The Partition Operation</span>

<span style="font-size: 14px;">Given a tensor $x$ of shape $(B, H, W, C)$ where the window size divides both $H$ and $W$, let $n_h = H / \texttt{window\_size}$ and $n_w = W / \texttt{window\_size}$ be the number of windows along each axis. Partition produces a tensor of shape:</span>

$$
(B \cdot n_h n_w, \ \texttt{window\_size}, \ \texttt{window\_size}, \ C)
$$

<span style="font-size: 14px;">The reshape proceeds in two conceptual steps. First view $x$ as $(B, n_h, M, n_w, M, C)$, splitting each spatial axis into a window index and an intra-window offset. Then permute to $(B, n_h, n_w, M, M, C)$ so the two window-grid indices are adjacent, and finally flatten the leading three axes into the batch dimension, giving $(B \cdot n_h n_w, M, M, C)$.</span>

<span style="font-size: 14px;">The reason a permute is required, and not just a reshape, is that the height axis splits into $(n_h, M)$ and the width axis into $(n_w, M)$, but a plain reshape would leave them interleaved as $n_h, M, n_w, M$. Attention needs each window's two intra-window axes ($M, M$) grouped together and separated from the window-grid axes ($n_h, n_w$). The permute physically reorders memory so that the $M \times M$ block of one window is contiguous, which is what lets the subsequent flatten produce clean per-window sequences. This is the single step most often gotten wrong.</span>

<span style="font-size: 14px;">The ordering convention is precise: all windows of batch element $b$ come before any window of batch $b+1$, and within a batch element the windows are laid out row-major over the window grid, top-left window first and bottom-right window last. Each window is a contiguous $M \times M \times C$ block of the original feature map.</span>

<span style="font-size: 14px;">Folding the window grid into the batch dimension is the trick that makes the downstream attention trivial to implement. After partition, the attention layer simply treats the merged leading axis as an ordinary batch: it flattens each window's $M \times M$ tokens into a length-$M^2$ sequence and runs standard multi-head self-attention with no awareness that these are windows at all. All the windowing logic lives entirely in partition and reverse, keeping the attention code identical to a plain ViT block. This separation of concerns is why the rearrange must be exact and order-preserving.</span>

---

## <span style="font-size: 16px;">The Reverse Operation</span>

<span style="font-size: 14px;">Reverse is the exact inverse: given the partitioned tensor of shape $(B \cdot n_h n_w, M, M, C)$ and the original $H, W$, it reconstructs $x$ of shape $(B, H, W, C)$ so that $\texttt{reverse}(\texttt{partition}(x)) = x$ exactly. It undoes the reshape and permute in the opposite order: view the leading axis back as $(B, n_h, n_w, M, M, C)$, permute to $(B, n_h, M, n_w, M, C)$, and collapse the paired axes into $(B, H, W, C)$.</span>

<span style="font-size: 14px;">The reverse is needed because Swin runs window attention, which changes the token values inside each window, and then must place those updated tokens back at their correct spatial locations before the next stage. Without an exact inverse the spatial structure of the feature map would be permanently scrambled after the first attention block.</span>

---

## <span style="font-size: 16px;">Shifted Windows</span>

<span style="font-size: 14px;">Pure window attention has a flaw: tokens never exchange information across window boundaries, so the receptive field is capped at one window. Swin fixes this with **shifted window attention (SW-MSA)** in alternating blocks. Every second block cyclically shifts the feature map by $\lfloor M/2 \rfloor$ pixels before partitioning, so the new windows straddle the previous boundaries, letting information flow between formerly separate windows.</span>

<span style="font-size: 14px;">After attention, the shift is reversed so positions line up again. The shift uses a cyclic roll plus an attention mask to prevent tokens that wrapped around the edge from attending to unrelated content. Alternating regular and shifted blocks gives Swin a growing effective receptive field while keeping every attention computation local and cheap. Window partition and its reverse are the primitives both block types are built on.</span>

<span style="font-size: 14px;">The cyclic-roll trick is itself a clever efficiency choice. A naive shift would create partial windows at the image border, breaking the constant window size that makes the cost analysis hold. Instead Swin rolls the feature map so the leftover strips wrap around to fill complete windows, keeping every window exactly $M \times M$, and uses a precomputed mask to zero out attention between tokens that the roll placed together but that are not actually spatially adjacent. This keeps the number of windows constant between regular and shifted blocks, so batching stays uniform.</span>

---

## <span style="font-size: 16px;">Why an Exact Inverse Matters</span>

<span style="font-size: 14px;">Swin alternates partition, attention, and reverse many times through a deep network. If reverse were even slightly inexact, errors would compound across blocks and stages, and the patch-merging layers that build the hierarchy assume a correctly laid-out feature map. Because both operations are pure index rearrangements with no rounding or arithmetic, the composition is provably the identity, which lets the network treat windowing as a transparent wrapper around attention.</span>

<span style="font-size: 14px;">The exactness also matters for the shifted-window path. The cyclic roll, partition, attention, reverse, and inverse roll must all compose to leave non-attended positions untouched. Any asymmetry between partition and reverse would leak into the masked, shifted computation and silently degrade the cross-window information exchange that shifted windows exist to provide.</span>

<span style="font-size: 14px;">Both operations are also memory-cheap and gradient-friendly. As pure rearranges they allocate a view or a single contiguous copy, carry no learnable parameters, and pass gradients straight through by scattering them back to the original positions. This is what lets Swin insert dozens of partition/reverse pairs through its depth without any training-stability or parameter-count cost.</span>

---

## <span style="font-size: 16px;">Numerical Example</span>

<span style="font-size: 14px;">Take $B = 1$, $H = W = 4$, $C = 1$, window size $M = 2$, so $n_h = n_w = 2$ and there are $4$ windows. Let the feature map be:</span>

$$
\begin{pmatrix} 1 & 2 & 3 & 4 \\ 5 & 6 & 7 & 8 \\ 9 & 10 & 11 & 12 \\ 13 & 14 & 15 & 16 \end{pmatrix}
$$

<span style="font-size: 14px;">Partition walks the $2 \times 2$ window grid row-major, each window a contiguous $2 \times 2$ block:</span>

* <span style="font-size: 14px;">**Window 0** top-left: rows 0-1, cols 0-1 = $\begin{pmatrix} 1 & 2 \\ 5 & 6 \end{pmatrix}$</span>
* <span style="font-size: 14px;">**Window 1** top-right: rows 0-1, cols 2-3 = $\begin{pmatrix} 3 & 4 \\ 7 & 8 \end{pmatrix}$</span>
* <span style="font-size: 14px;">**Window 2** bottom-left: rows 2-3, cols 0-1 = $\begin{pmatrix} 9 & 10 \\ 13 & 14 \end{pmatrix}$</span>
* <span style="font-size: 14px;">**Window 3** bottom-right: rows 2-3, cols 2-3 = $\begin{pmatrix} 11 & 12 \\ 15 & 16 \end{pmatrix}$</span>

<span style="font-size: 14px;">The partitioned tensor has shape $(4, 2, 2, 1)$. Reverse takes these four windows and places each block back at its row-major location, exactly recovering the original $4 \times 4$ map. The round trip is lossless because partition is a pure rearrange with no arithmetic.</span>

<span style="font-size: 14px;">If $B = 2$ instead, the merged batch axis would hold $2 \times 4 = 8$ windows, with image 0's four windows in positions 0 to 3 and image 1's four windows in positions 4 to 7. Reverse must split that axis back as $(B, n_h, n_w)$ in the same order, which is why the batch-major then row-major convention has to be honored on both sides.</span>

<span style="font-size: 14px;">Notice the contrast with patchify: both rearrange a spatial grid into a batch of blocks, but patchify then flattens each block into a single token vector, whereas window partition keeps each window as a small 2D grid of tokens so that attention can run among them. Window 0 here holds the four neighboring values 1, 2, 5, 6 that attention will let interact, while values 1 and 3, separated by a window boundary, cannot interact until a shifted-window block brings them together. Walking the example by hand makes the row-major batch ordering concrete and is the fastest way to catch a permute bug.</span>

---

## <span style="font-size: 16px;">Variants and Context</span>

* <span style="font-size: 14px;">**Patch merging vs window attention.** Windowing keeps the token count fixed within a stage; the separate patch-merging layer between stages concatenates $2 \times 2$ neighbors and reduces resolution. Together they give the pyramid feature maps that detection and segmentation heads expect, unlike ViT's single-scale output.</span>
* <span style="font-size: 14px;">**Swin V2.** Liu et al., 2022 scale Swin to larger models and resolutions, replacing the attention dot product with scaled cosine attention and using log-spaced continuous relative position bias so windows transfer across resolutions.</span>
* <span style="font-size: 14px;">**Relative position bias.** Inside each window Swin adds a learned bias indexed by the relative offset between query and key tokens, so position is encoded by offset within the window rather than by absolute grid coordinate. This is added after partition, on the per-window sequences.</span>
* <span style="font-size: 14px;">**Other local-attention backbones.** Twins, CSWin, and Focal Transformer explore alternative window shapes (cross-shaped, axial, multi-scale) but all rely on the same partition/reverse primitive to localize attention.</span>

---

## <span style="font-size: 16px;">Pitfalls</span>

* <span style="font-size: 14px;">**Wrong permute order in the reshape.** Partition is a view + permute + flatten; getting the permute axis order wrong (for example swapping $n_w$ and $M$) produces a tensor of the correct shape but with pixels from different windows interleaved. It does not crash, and the bug only surfaces as garbage attention much later.</span>
* <span style="font-size: 14px;">**Mismatched window ordering between partition and reverse.** Partition lays windows out batch-major then row-major; reverse must undo that exact order. Using column-major in one and row-major in the other breaks the identity $\texttt{reverse}(\texttt{partition}(x)) = x$ silently, scrambling the feature map after the first block.</span>
* <span style="font-size: 14px;">**Window size not dividing $H$ or $W$.** The reshape assumes $H \bmod M = W \bmod M = 0$. If it does not divide evenly the view fails or drops edge pixels. Swin pads the feature map up to a multiple of $M$ before partitioning and crops after reversing; forgetting the pad/crop loses the border.</span>
* <span style="font-size: 14px;">**Forgetting batch ordering when $B > 1$.** All windows of one image must stay contiguous in the merged batch axis. Interleaving windows across batch elements corrupts which image each window belongs to, so reverse reassembles pixels from the wrong images without any error.</span>

---