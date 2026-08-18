# <span style="font-size: 20px;">KV Compression via Low-Rank Down-Projection</span>

<span style="font-size: 14px;">In standard multi-head attention, the KV cache stores separate key and value vectors for every head at every token position, creating a memory bottleneck that grows linearly with sequence length and number of heads. DeepSeek V3's Multi-head Latent Attention (MLA) eliminates this by projecting the input into a low-dimensional latent vector $c_{kv}$ using a learned down-projection matrix $W_{dkv}$. This compressed representation is cached instead of full K and V tensors, reducing per-token cache memory by up to 32x while preserving the information needed to reconstruct keys and values on the fly.</span>

---

## <span style="font-size: 16px;">What It Is</span>

<span style="font-size: 14px;">KV compression via low-rank down-projection is the first stage of DeepSeek V3's MLA mechanism. The idea is deceptively simple: instead of computing and storing all $n_h$ key-value head vectors (each of dimension $d_h$) for every token, you apply a single linear transformation that maps the input $x$ into a compact latent vector $c_{kv}$ of dimension $d_c$, where $d_c \ll n_h \cdot d_h$.</span>

<span style="font-size: 14px;">Concretely, MLA replaces the traditional per-head key and value projections ($W_K^{(i)}, W_V^{(i)}$ for each head $i$) with one shared down-projection matrix $W_{dkv} \in \mathbb{R}^{d_c \times d_{\text{model}}}$. The output $c_{kv}$ captures a compressed "joint key-value representation" that encodes the information of all heads simultaneously. During attention computation, separate up-projection matrices reconstruct the per-head keys and values from $c_{kv}$, but only $c_{kv}$ itself is stored in the cache.</span>

<span style="font-size: 14px;">This is a **factorized approach**: the standard KV projection is decomposed into a shared down-projection (cached) followed by per-head up-projections (recomputed each time). The memory savings come from persisting only the narrow bottleneck vector.</span>

---

## <span style="font-size: 16px;">Key Equations</span>

<span style="font-size: 14px;">The down-projection that produces the compressed KV latent is:</span>

$$
c_{kv} = x \cdot W_{dkv}^T
$$

<span style="font-size: 14px;">where $x \in \mathbb{R}^{(\text{batch}, \text{seq}, d_{\text{model}})}$ is the input tensor and $W_{dkv} \in \mathbb{R}^{(d_c, d_{\text{model}})}$ is the learned down-projection matrix. The result $c_{kv} \in \mathbb{R}^{(\text{batch}, \text{seq}, d_c)}$ is the compressed latent that replaces the full KV cache.</span>

<span style="font-size: 14px;">The **compression ratio** measures how much smaller the latent is compared to the full KV representation:</span>

$$
\text{compression ratio} = \frac{n_h \cdot d_h}{d_c}
$$

<span style="font-size: 14px;">For DeepSeek V3, $n_h = 128$, $d_h = 128$, and $d_c = 512$, giving a compression ratio of $16384 / 512 = 32$.</span>

<span style="font-size: 14px;">The **per-token memory savings** follow directly. Standard MHA stores $2 \cdot n_h \cdot d_h$ values per token (keys and values across all heads). With MLA, only $d_c$ values are stored:</span>

$$
\text{memory per token (standard)} = 2 \cdot n_h \cdot d_h
$$

$$
\text{memory per token (MLA)} = d_c
$$

$$
\text{reduction factor} = \frac{2 \cdot n_h \cdot d_h}{d_c}
$$

<span style="font-size: 14px;">This gives a $2 \times 16384 / 512 = 64$x reduction in the per-token KV cache footprint. The factor of 2 comes from replacing both keys and values with a single shared latent.</span>

---

## <span style="font-size: 16px;">Why the KV Cache is a Problem</span>

<span style="font-size: 14px;">To understand why this compression matters, consider the memory arithmetic of autoregressive inference. The model must attend to all previous tokens, storing a key and value vector for every token, every head, and every layer. The total KV cache memory scales as:</span>

$$
\text{KV cache} = 2 \cdot n_{\text{layers}} \cdot n_h \cdot d_h \cdot L \cdot B \cdot \text{bytes per element}
$$

<span style="font-size: 14px;">where $L$ is sequence length, $B$ is batch size, and the factor of 2 accounts for both keys and values.</span>

<span style="font-size: 14px;">For a model the size of DeepSeek V3 with 61 layers, 128 attention heads, head dimension 128, and using FP16 (2 bytes per element), the KV cache for a single sequence of 4096 tokens is:</span>

$$
2 \times 61 \times 128 \times 128 \times 4096 \times 2 \text{ bytes} \approx 16.4 \text{ GB}
$$

<span style="font-size: 14px;">That is 16.4 GB for the KV cache of a **single sequence**. For a batch of 8 sequences, you need over 130 GB just for the cache, not counting model weights or activations. This far exceeds the memory required by the model parameters themselves during inference.</span>

<span style="font-size: 14px;">The problem compounds at longer context lengths. At 128K tokens (which DeepSeek V3 supports), the single-sequence KV cache would balloon to over 512 GB under standard MHA, clearly infeasible even on high-end GPU clusters. Beyond raw capacity, the KV cache also creates a bandwidth bottleneck: at every decoding step, the entire cache must be read from GPU memory to compute attention, making inference memory-bound rather than compute-bound. Reducing the cache size directly translates to faster token generation.</span>

---

## <span style="font-size: 16px;">The Low-Rank Compression Idea</span>

<span style="font-size: 14px;">The foundation for KV compression rests on a key observation: the full $n_h \cdot d_h$-dimensional KV representation is heavily redundant. The key-value vectors across heads often lie in a much lower-dimensional subspace, a form of **low-rank structure** that can be exploited.</span>

<span style="font-size: 14px;">From a linear algebra perspective, consider the combined KV projection in standard attention. The input $x \in \mathbb{R}^{d_{\text{model}}}$ is mapped to a concatenated key-value vector of dimension $2 \cdot n_h \cdot d_h$ by the stacked projection matrices. If this mapping has effective rank much less than $2 \cdot n_h \cdot d_h$, then there exists a factorization that preserves the essential information:</span>

$$
W_{KV} \approx W_{\text{up}} \cdot W_{\text{down}}
$$

<span style="font-size: 14px;">where $W_{\text{down}} \in \mathbb{R}^{d_c \times d_{\text{model}}}$ and $W_{\text{up}} \in \mathbb{R}^{(2 \cdot n_h \cdot d_h) \times d_c}$. This is analogous to **bottleneck layers** in autoencoders: the down-projection compresses the input into a compact code, and the up-projection reconstructs the full-dimensional representation. The key insight is that only the compact code needs to be cached.</span>

<span style="font-size: 14px;">Unlike post-hoc methods (quantizing or pruning a trained model's cache), MLA learns the compression jointly during pre-training. The model adapts its representations to be maximally informative within the $d_c$-dimensional bottleneck, yielding better quality than applying compression as an afterthought.</span>

<span style="font-size: 14px;">From an information bottleneck perspective, forcing all key-value information through a $d_c$-dimensional channel incentivizes the model to retain only the most relevant features while discarding noise and redundancy. This acts as regularization in addition to saving memory.</span>

---

## <span style="font-size: 16px;">Paper Context: DeepSeek V3 MLA</span>

<span style="font-size: 14px;">Multi-head Latent Attention was first introduced in DeepSeek-V2 and carried forward into DeepSeek V3. The KV cache is the dominant memory consumer at long sequence lengths, and MLA was the architectural answer to this challenge.</span>

<span style="font-size: 14px;">In DeepSeek V3's configuration, the specific dimensions are:</span>

* <span style="font-size: 14px;">**Model dimension** $d_{\text{model}} = 7168$</span>
* <span style="font-size: 14px;">**Number of attention heads** $n_h = 128$</span>
* <span style="font-size: 14px;">**Per-head dimension** $d_h = 128$</span>
* <span style="font-size: 14px;">**KV compression dimension** $d_c = 512$</span>
* <span style="font-size: 14px;">**Full KV dimension** $n_h \cdot d_h = 16384$</span>
* <span style="font-size: 14px;">**Compression ratio** $16384 / 512 = 32$x</span>

<span style="font-size: 14px;">The MLA pipeline works in stages. First, the input is projected down to $c_{kv} = x \cdot W_{dkv}^T$ (this problem). Then, during attention, $c_{kv}$ is projected back up to reconstruct per-head keys and values via learned up-projection matrices $W_{uk}$ and $W_{uv}$. Queries go through an analogous compression path.</span>

<span style="font-size: 14px;">The paper emphasizes that this design makes the KV cache **independent of the number of heads**. In standard MHA, the cache grows proportionally to $n_h$. In MLA, the cache stores only $c_{kv}$ with fixed dimension $d_c$ regardless of head count. This allows the model to use 128 heads for representational capacity without paying the corresponding memory cost during inference.</span>

<span style="font-size: 14px;">The query side uses analogous compression: $c_q = x \cdot W_{dq}^T$ with its own bottleneck dimension $d_c'$. This does not affect cache size (queries are only needed for the current token) but reduces parameter count and compute.</span>

<span style="font-size: 14px;">DeepSeek V3 also uses **decoupled RoPE** because applying Rotary Position Embeddings inside the latent space would break the compression. Position information flows through a separate small set of dedicated head dimensions carrying RoPE, concatenated with the latent-reconstructed keys.</span>

---

## <span style="font-size: 16px;">Numerical Example</span>

<span style="font-size: 14px;">Consider a simplified model with the following dimensions:</span>

* <span style="font-size: 14px;">**Model dimension** $d_{\text{model}} = 16$</span>
* <span style="font-size: 14px;">**Number of heads** $n_h = 4$</span>
* <span style="font-size: 14px;">**Per-head dimension** $d_h = 4$</span>
* <span style="font-size: 14px;">**Full KV dimension** $n_h \cdot d_h = 16$</span>
* <span style="font-size: 14px;">**Compression dimension** $d_c = 4$</span>

<span style="font-size: 14px;">The compression ratio is $16 / 4 = 4$x. The down-projection matrix $W_{dkv}$ has shape $(d_c, d_{\text{model}}) = (4, 16)$.</span>

<span style="font-size: 14px;">For a single token with input $x \in \mathbb{R}^{16}$, the compressed latent is:</span>

$$
c_{kv} = x \cdot W_{dkv}^T \in \mathbb{R}^{4}
$$

<span style="font-size: 14px;">This 4-dimensional vector replaces what would have been a 16-dimensional key vector plus a 16-dimensional value vector (32 total values) in standard MHA. We store 4 values instead of 32, a reduction factor of $32 / 4 = 8$x (accounting for both K and V).</span>

<span style="font-size: 14px;">Now scale this to a sequence. For a batch of 2 sequences, each of length 1024:</span>

* <span style="font-size: 14px;">**Standard MHA cache**: $2 \times 1024 \times 2 \times 16 = 65536$ values (keys + values, all heads)</span>
* <span style="font-size: 14px;">**MLA cache**: $2 \times 1024 \times 4 = 8192$ values (just $c_{kv}$)</span>
* <span style="font-size: 14px;">**Memory reduction**: $65536 / 8192 = 8$x</span>

<span style="font-size: 14px;">Now scale to DeepSeek V3's actual dimensions with a single 4096-token sequence in FP16:</span>

* <span style="font-size: 14px;">**Standard MHA cache per layer**: $4096 \times 2 \times 128 \times 128 \times 2 \text{ bytes} \approx 256 \text{ MB}$</span>
* <span style="font-size: 14px;">**MLA cache per layer**: $4096 \times 512 \times 2 \text{ bytes} = 4 \text{ MB}$</span>
* <span style="font-size: 14px;">**Per-layer reduction**: $256 / 4 = 64$x</span>

<span style="font-size: 14px;">Across 61 layers, standard MHA cache totals $61 \times 256 \text{ MB} \approx 15.6 \text{ GB}$, while MLA cache is $61 \times 4 \text{ MB} \approx 244 \text{ MB}$. This transforms the KV cache from a dominant memory consumer into minor overhead, enabling larger batch sizes and longer contexts on the same hardware.</span>

---

## <span style="font-size: 16px;">Modern Context: GQA, MQA, and MLA</span>

<span style="font-size: 14px;">KV cache compression is not unique to MLA. Several approaches make different tradeoffs between cache size, model quality, and implementation complexity.</span>

<span style="font-size: 14px;">**Multi-Query Attention (MQA)**, introduced by Shazeer (2019), uses a single shared key-value head across all query heads, reducing the KV cache by a factor of $n_h$. The cache stores $2 \cdot d_h$ values per token per layer. MQA is simple but can degrade quality because all heads share identical key-value representations.</span>

<span style="font-size: 14px;">**Grouped-Query Attention (GQA)**, introduced by Ainslie et al. (2023), groups query heads into $n_g$ groups sharing one KV head each. The cache stores $2 \cdot n_g \cdot d_h$ values per token per layer. Llama 2 70B and Llama 3 use GQA with $n_g = 8$, giving an $n_h / n_g$ reduction while preserving more diversity than MQA.</span>

<span style="font-size: 14px;">**MLA** takes a fundamentally different approach. Instead of reducing the number of KV heads, it projects all KV information into a learned latent space. The key differences are:</span>

* <span style="font-size: 14px;">**Learned compression vs. weight sharing**: MQA/GQA force heads to share weights. MLA learns an optimal compression that distributes information across latent dimensions to best serve all heads.</span>
* <span style="font-size: 14px;">**Decoupled from head count**: MLA's cache size is independent of $n_h$. Adding more heads costs nothing in cache memory. GQA's cache still scales with the number of groups.</span>
* <span style="font-size: 14px;">**Compute tradeoff**: MLA requires extra compute for the up-projections during attention. MQA and GQA have no such overhead since their KV representations are used directly.</span>

<span style="font-size: 14px;">The DeepSeek team reported that MLA matches full MHA performance in their ablations, whereas MQA showed measurable degradation on some benchmarks. MLA achieves comparable or better quality at a fraction of the cache cost.</span>

---

## <span style="font-size: 16px;">Pitfalls</span>

<span style="font-size: 14px;">Common mistakes when implementing KV compression via down-projection:</span>

* <span style="font-size: 14px;">**Setting $d_c$ too small**: if the bottleneck is too aggressive, the latent cannot preserve enough information to reconstruct useful keys and values. Attention patterns degrade and quality drops. DeepSeek V3 chose $d_c = 512$ through ablation, finding it to be the sweet spot where cache savings are large and quality loss is negligible.</span>
* <span style="font-size: 14px;">**Setting $d_c$ too large**: if $d_c$ approaches $n_h \cdot d_h$, the compression becomes trivial and memory savings vanish. The factorization adds computational cost without meaningful cache reduction. A good $d_c$ should be at least an order of magnitude smaller than $n_h \cdot d_h$.</span>
* <span style="font-size: 14px;">**Forgetting the up-projection**: the down-projection alone is only half the mechanism. The compressed $c_{kv}$ must be projected back up to produce per-head key and value vectors for attention. The full pipeline is: down-project to cache, then up-project to compute attention.</span>
* <span style="font-size: 14px;">**Dimension mismatches in the projection matrix**: $W_{dkv}$ has shape $(d_c, d_{\text{model}})$. A common error is transposing the dimensions or confusing $d_c$ with $d_h$. Since the operation is $c_{kv} = x \cdot W_{dkv}^T$, the weight matrix must have $d_c$ rows and $d_{\text{model}}$ columns.</span>
* <span style="font-size: 14px;">**Confusing compression ratio with memory reduction**: the compression ratio $n_h \cdot d_h / d_c$ measures how much smaller the latent is compared to the per-token key dimension. The actual memory reduction is larger because the single latent replaces both keys and values, giving a factor of $2 \cdot n_h \cdot d_h / d_c$. Reporting only the compression ratio understates the true savings by 2x.</span>
* <span style="font-size: 14px;">**Applying RoPE to the compressed latent**: RoPE cannot be naively applied to $c_{kv}$ because the compression mixes information across head dimensions, and RoPE operates on per-head key/query pairs. DeepSeek V3 decouples RoPE into a separate pathway with dedicated dimensions concatenated after the up-projection.</span>
* <span style="font-size: 14px;">**Assuming the compression is lossless**: the down-projection is inherently lossy. Information that cannot be represented in $d_c$ dimensions is discarded. The model learns to minimize the impact on task performance, but the latent is strictly less expressive than the full KV representation.</span>

---