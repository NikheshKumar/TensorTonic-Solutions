# <span style="font-size: 20px;">Sliding Window Causal Mask</span>

<span style="font-size: 14px;">A **sliding window causal mask** restricts each query token to attend only to the previous $W$ keys (plus itself). GPT-OSS applies this mask on every even-indexed attention layer, alternating with full causal attention on odd layers, to cut memory and bias the model toward local context.</span>

---

## <span style="font-size: 16px;">Why Masks Exist in Causal Attention</span>

<span style="font-size: 14px;">Self-attention computes scores $S = QK^T$ for every (query, key) pair. In a causal language model, token $i$ must not see tokens at positions $j > i$ during training, otherwise it would trivially predict the next token by copying from the future. The standard remedy is an additive mask $M$ where $M_{ij} = 0$ for visible positions and $M_{ij} = -\infty$ for hidden positions. Adding $M$ before the softmax turns invisible logits into exact zeros after exponentiation:</span>

$$
\text{Attention}(Q, K, V)_{ij} = \text{softmax}\!\left(\frac{QK^T + M}{\sqrt{d_k}}\right)_{ij} V
$$

<span style="font-size: 14px;">Because $\exp(-\infty) = 0$, the masked positions contribute no probability mass, even when the underlying logit is large. This is the cleanest way to express a hard structural constraint inside a smooth differentiable softmax.</span>

---

## <span style="font-size: 16px;">The Original Causal Mask</span>

<span style="font-size: 14px;">The vanilla GPT-style causal mask blocks every future position. For sequence length $n$, the mask is a strict upper triangle of $-\infty$:</span>

$$
M^{\text{causal}}_{ij} = \begin{cases} 0 & j \le i \\ -\infty & j > i \end{cases}
$$

<span style="font-size: 14px;">In PyTorch this is one line: $\texttt{torch.triu(torch.full((n, n), -inf), diagonal=1)}$. The argument $\texttt{diagonal=1}$ keeps the main diagonal at $0$ (so a token can attend to itself) and sets only the strictly upper triangle to $-\infty$.</span>

* <span style="font-size: 14px;">Visible band per query $i$: $j \in [0, i]$, that is $i + 1$ keys</span>
* <span style="font-size: 14px;">Total FLOPs per layer: $O(n^2 d)$ because every pair is still computed before masking</span>
* <span style="font-size: 14px;">Memory pressure: KV cache grows as $O(n L d)$ where $L$ is the number of layers</span>

---

## <span style="font-size: 16px;">Sliding Window: Locality Plus Memory</span>

<span style="font-size: 14px;">For long contexts (32K, 128K, or longer), full causal attention becomes the dominant cost in both FLOPs and KV-cache size. A **sliding window** restricts each query to the most recent $W$ keys. With $W \ll n$ the per-token compute drops from $O(n)$ to $O(W)$, and only the last $W$ KV entries per layer have to live in memory at inference time.</span>

<span style="font-size: 14px;">The mask becomes:</span>

$$
M^{\text{swa}}_{ij} = \begin{cases} 0 & 0 \le i - j < W \\ -\infty & \text{otherwise} \end{cases}
$$

<span style="font-size: 14px;">Reading the cases carefully:</span>

* <span style="font-size: 14px;">$j > i$ : future, blocked (causal)</span>
* <span style="font-size: 14px;">$i - j \ge W$ : too far in the past, blocked (window)</span>
* <span style="font-size: 14px;">$0 \le i - j < W$ : visible, mask value $0$</span>

<span style="font-size: 14px;">The visible region per query is a band of width $W$ along the main diagonal. Token $i$ attends to positions $\max(0, i - W + 1), \ldots, i$.</span>

---

## <span style="font-size: 16px;">The GPT-OSS Implementation</span>

<span style="font-size: 14px;">GPT-OSS builds this mask in two additive steps that mirror the reference code from $\texttt{openai/gpt-oss}$:</span>

```text
mask = torch.triu(torch.full((n, n), -inf), diagonal=1)        # block future
if window > 0:
    mask = mask + torch.tril(torch.full((n, n), -inf), diagonal=-window)  # block far past
```

<span style="font-size: 14px;">Two important design choices:</span>

* <span style="font-size: 14px;">**$\texttt{window == 0}$ means no window**: when window is zero, the second additive term is skipped and the mask collapses to a plain causal mask. GPT-OSS uses $\texttt{window = 0}$ on odd-indexed layers to recover full attention.</span>
* <span style="font-size: 14px;">**$\texttt{+=}$ instead of $\texttt{torch.maximum}$**: both terms are $-\infty$ in their respective triangles. Adding two $-\infty$ values stays $-\infty$, while adding $-\infty$ and $0$ is $-\infty$, and adding $0$ and $0$ is $0$. In the strict-causal mask the upper triangle and the lower-window triangle never overlap (one is above diagonal $\texttt{diagonal}=1$, the other below diagonal $\texttt{diagonal}=-W$ for $W \ge 1$), so no $-\infty + -\infty$ collision occurs. The visible band stays exactly $0$.</span>

---

## <span style="font-size: 16px;">Alternating Layers: GPT-OSS's Pattern</span>

<span style="font-size: 14px;">GPT-OSS stacks two attention variants in alternation. With $\texttt{sliding\_window = 128}$:</span>

* <span style="font-size: 14px;">**Even layers (idx 0, 2, 4, ...)**: sliding window of 128 tokens. Cheap, local.</span>
* <span style="font-size: 14px;">**Odd layers (idx 1, 3, 5, ...)**: full causal. Expensive, global.</span>

<span style="font-size: 14px;">This is the same alternation pattern used by Mistral-7B's later variants and by Gemma. The motivation: most tokens only need local context, but every other layer keeps a path to the full prefix so the model can still aggregate global information. Memory drops roughly in half compared to full attention at every layer, while quality stays close to the full-attention baseline.</span>

---

## <span style="font-size: 16px;">Interaction with Attention Sinks</span>

<span style="font-size: 14px;">GPT-OSS combines this sliding window with a separate mechanism called **attention sinks**: a learned scalar $s_h$ per head that is appended to the softmax denominator (but not the output). The sink lives entirely inside the softmax normalisation; it is not part of the mask matrix.</span>

* <span style="font-size: 14px;">The sliding-window mask still blocks far-past tokens by setting their score to $-\infty$.</span>
* <span style="font-size: 14px;">The attention sink does NOT live in the visible band, so the sliding window does not need to widen by one to include it.</span>
* <span style="font-size: 14px;">When all near tokens look uninformative, the head can route mass into the sink instead of being forced to assign a confident weight to a noisy neighbour.</span>

<span style="font-size: 14px;">Together, sliding window and attention sinks make the windowed layers stable: locality without "forced confidence" on the local window.</span>

---

## <span style="font-size: 16px;">Worked Example: $n = 4$, $W = 2$</span>

<span style="font-size: 14px;">Pretend the sequence has four tokens and the window is 2. Walk the cases:</span>

* <span style="font-size: 14px;">Row 0 (query at position 0): can see only itself, so column 0 is $0$, columns 1, 2, 3 are $-\infty$ (all future).</span>
* <span style="font-size: 14px;">Row 1: $i - j \in \\{1, 0, -1, -2\\}$ for $j = 0, 1, 2, 3$. Cases $i - j = 0$ and $1$ are visible, others masked.</span>
* <span style="font-size: 14px;">Row 2: $j = 0$ has $i - j = 2 \ge W$ so it is masked; $j = 1, 2$ are visible; $j = 3$ is future, masked.</span>
* <span style="font-size: 14px;">Row 3: similarly, $j = 2, 3$ are visible, $j = 0, 1$ are masked (too old), $j = 4, \ldots$ do not exist.</span>

<span style="font-size: 14px;">The resulting mask:</span>

$$
M = \begin{pmatrix}
0 & -\infty & -\infty & -\infty \\
0 & 0 & -\infty & -\infty \\
-\infty & 0 & 0 & -\infty \\
-\infty & -\infty & 0 & 0
\end{pmatrix}
$$

<span style="font-size: 14px;">The visible band has width $W = 2$. The total number of $0$ entries equals $\sum_{i=0}^{n-1} \min(i + 1, W)$. For $n = 4, W = 2$ this is $1 + 2 + 2 + 2 = 7$.</span>

---

## <span style="font-size: 16px;">Edge Cases</span>

<span style="font-size: 14px;">A correct implementation has to handle a few boundary cases that often trip up first-time authors:</span>

* <span style="font-size: 14px;">**$\texttt{n == 1}$**: any window value yields the $1 \times 1$ matrix $\[[0]\]$. The query at position 0 attends to itself, no future, no past.</span>
* <span style="font-size: 14px;">**$\texttt{window == 0}$**: skip the second additive term entirely. The result is the plain causal mask.</span>
* <span style="font-size: 14px;">**$\texttt{window >= n}$**: the second additive term has no effect (the lower triangle below $k = -W$ is empty when $W \ge n$). The result equals the plain causal mask.</span>
* <span style="font-size: 14px;">**$\texttt{window == n - 1}$**: every below-diagonal entry except the bottom-left corner is visible. Only $M_{n-1, 0}$ is masked among the lower triangle.</span>

---

## <span style="font-size: 16px;">Comparison with Other Architectures</span>

<span style="font-size: 14px;">Sliding window attention is not new with GPT-OSS. The variants worth knowing:</span>

* <span style="font-size: 14px;">**Mistral 7B**: a fixed sliding window of 4096 tokens on every layer. No alternation. Works because the receptive field across $L$ stacked sliding layers is $L \cdot W$, so a 32-layer model with $W = 4096$ effectively reaches $\sim 130\text{K}$ tokens of context.</span>
* <span style="font-size: 14px;">**Longformer (Beltagy et al., 2020)**: a fixed window plus a small set of global tokens that every position can attend to. Used for document QA where a few special tokens (like $\texttt{[CLS]}$) need full reach.</span>
* <span style="font-size: 14px;">**BigBird (Zaheer et al., 2020)**: sliding window plus random plus global attention. Provable sequence-length scaling but rarely used in production.</span>
* <span style="font-size: 14px;">**GPT-OSS / Gemma 2**: alternating sliding and full layers. The full layers handle long-range information; the windowed layers do most of the work cheaply.</span>

<span style="font-size: 14px;">All of these share the same mathematical primitive: an additive $-\infty$ mask whose visible band has finite width.</span>

---

## <span style="font-size: 16px;">Why Build the Mask at All?</span>

<span style="font-size: 14px;">Conceptually the mask is just a way to communicate sparsity to a dense matmul. In production attention kernels (FlashAttention with sliding-window support, vLLM's PagedAttention, xFormers' block-diagonal attention) the kernel takes the window size as an integer and skips the masked tiles entirely, so the actual cost is $O(n W d)$, not $O(n^2 d)$. The mask matrix is the contract between the math (clean softmax over a band) and the kernel (skip the masked tiles).</span>

<span style="font-size: 14px;">Even when the kernel does not skip masked positions (naive PyTorch), the additive $-\infty$ keeps the math correct: those positions contribute nothing to the softmax. Only the FLOPs are wasted.</span>

---

## <span style="font-size: 16px;">Complexity and Memory Analysis</span>

<span style="font-size: 14px;">A windowed layer has materially smaller cost than a full-attention layer at long context. The relevant quantities:</span>

* <span style="font-size: 14px;">**FLOPs per query token**: full attention is $O(n d)$, windowed attention is $O(W d)$. With $W = 128$ and $n = 32\text{K}$, that is a $256\times$ reduction at long context.</span>
* <span style="font-size: 14px;">**KV cache memory**: full attention keeps every key/value pair, so $O(n L d)$ bytes. Windowed layers only need the last $W$ entries, so $O(W L d)$. For half the layers, this cuts cache memory by close to a factor of two at long context.</span>
* <span style="font-size: 14px;">**Training memory**: the attention matrix itself is $n \times n$ for full layers, $n \times W$ for windowed (with the right kernel). At sequence length 8K with full attention, the attention probabilities tensor alone is 64M floats per head; with $W = 256$ it drops to 2M.</span>

<span style="font-size: 14px;">These are not theoretical: they show up as real wall-clock and memory savings in published GPT-OSS benchmarks.</span>

---

## <span style="font-size: 16px;">Effective Receptive Field</span>

<span style="font-size: 14px;">A natural worry: if half the layers only see 128 tokens, can the model handle long context at all? The answer is yes, because the full-attention layers in the alternation cover the entire prefix, and even purely sliding stacks accumulate range.</span>

* <span style="font-size: 14px;">**Stacked sliding layers**: after $L$ layers each with window $W$, the effective receptive field is roughly $L \cdot W$ tokens (information flows one window per layer). Mistral 7B has 32 sliding layers at $W = 4096$, so the receptive field is $\sim 131\text{K}$ tokens despite never using full attention.</span>
* <span style="font-size: 14px;">**Alternating layers**: GPT-OSS gets long-range information immediately from the full layers, so the windowed layers only have to refine local representations. This is closer to a hybrid of a convolution (local) and a transformer (global).</span>
* <span style="font-size: 14px;">**Comparison to RNNs**: a stack of sliding-window attention layers is mathematically reminiscent of a deep dilated convolution. The window plays the role of the kernel size.</span>

---

## <span style="font-size: 16px;">Pitfalls</span>

* <span style="font-size: 14px;">**Off-by-one on the main diagonal.** Using $\texttt{torch.triu(..., diagonal=0)}$ instead of $\texttt{diagonal=1}$ masks the token's own position, which means the query cannot attend to itself. This breaks language modelling completely: row 0 becomes all $-\infty$ and the softmax over it produces NaN.</span>
* <span style="font-size: 14px;">**Wrong sign of the mask value.** Some implementations use $+\infty$ or a large positive number "to penalize". After softmax, $+\infty$ becomes the only attended position, the inverse of what was intended. Always use $-\infty$ (or a large negative like $-10^9$) for additive masks.</span>
* <span style="font-size: 14px;">**Applying the window on the wrong axis.** Off-by-sign errors on the second $\texttt{torch.tril}$ call (using $\texttt{torch.triu}$ by accident, or $\texttt{diagonal=window}$ instead of $\texttt{diagonal=-window}$) produce a transposed or rotated band. Easy to spot in $n = 4$ but quiet in production where most masked entries are correct by coincidence.</span>
* <span style="font-size: 14px;">**Forgetting the $\texttt{window == 0}$ branch.** Adding $\texttt{torch.tril(..., diagonal=0)}$ when $\texttt{window = 0}$ blocks every position including the diagonal, again yielding all-$-\infty$ rows and NaN softmax outputs. The conditional skip is load-bearing.</span>
* <span style="font-size: 14px;">**Confusing causal with bidirectional.** A sliding window over a bidirectional encoder (BERT-style) does not have the $j > i$ constraint: it is symmetric around the diagonal, with width $2W + 1$. Mixing the two conventions silently changes the model from autoregressive to bidirectional.</span>
* <span style="font-size: 14px;">**Assuming $\texttt{(i, j)}$ means $\texttt{(key, query)}$.** Convention in this code: rows are queries, columns are keys. Swapping the two yields the transpose, which masks the wrong half and corrupts every attention head.</span>
* <span style="font-size: 14px;">**Forgetting that window includes the current token.** The visible band has width $W$, which means token $i$ sees positions $i - W + 1$ through $i$ (inclusive), not $i - W$ through $i$. Off-by-one here halves the effective context for the windowed layers.</span>
* <span style="font-size: 14px;">**Using $\texttt{torch.maximum}$ instead of $\texttt{+=}$.** The two are equivalent for the strict causal case (no overlap), but if you ever extend the mask to include a non-disjoint third condition, $\texttt{torch.maximum}$ silently masks "either", while $\texttt{+=}$ produces NaN where two $-\infty$ values collide. Always test with overlapping conditions before changing the combiner.</span>

---
