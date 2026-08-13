# <span style="font-size: 20px;">Global vs Local Attention Layer Routing</span>

<span style="font-size: 14px;">In Gemma 3, not every transformer layer attends to the full sequence. Instead, the architecture **alternates** between local attention layers (which see only a sliding window of nearby tokens) and global attention layers (which attend to every token in the context). A single integer hyperparameter $R$ governs the pattern: every $(R+1)$-th layer is global, and the rest are local. With $R = 5$ in Gemma 3 27B, every 6th layer is global, producing 52 local layers and 10 global layers across the 62-layer stack.</span>

---

## <span style="font-size: 16px;">What It Is</span>

<span style="font-size: 14px;">Global vs local attention layer routing is a layer-level architectural decision that assigns each transformer layer one of two attention modes. A **local attention layer** restricts each token's attention to a fixed-size sliding window of $w$ surrounding tokens. A **global attention layer** allows each token to attend to every other token in the sequence, up to the full context length.</span>

<span style="font-size: 14px;">The assignment is not learned or adaptive. It is a static, deterministic pattern set at architecture design time and baked into the model configuration. Given the layer index $i$ (zero-indexed) and the hyperparameter $R$ (the number of local layers between consecutive global layers), the formula that decides whether layer $i$ is global or local is a single modular arithmetic expression.</span>

<span style="font-size: 14px;">The core motivation is computational efficiency. Full self-attention over a sequence of length $n$ costs $O(n^2)$ per layer. If most layers only attend within a local window of size $w$, their cost drops to $O(n \cdot w)$, which is linear in $n$ for fixed $w$. By reserving global attention for a small fraction of layers, the model can process very long sequences without the quadratic blowup that would occur if every layer were global.</span>

---

## <span style="font-size: 16px;">Key Equations</span>

<span style="font-size: 14px;">The routing decision for a given layer is captured by a simple modular arithmetic formula. Given a zero-indexed layer index $i$ and the hyperparameter $R$, the layer is **global** if and only if:</span>

$$
\text{is\_global}(i) = \bigl((i + 1) \bmod (R + 1)\bigr) == 0
$$

<span style="font-size: 14px;">Equivalently, the layer is **local** when the above condition is false:</span>

$$
\text{is\_local}(i) = \bigl((i + 1) \bmod (R + 1)\bigr) \neq 0
$$

<span style="font-size: 14px;">Breaking the formula down:</span>

* <span style="font-size: 14px;">**$i + 1$:** Shifts from zero-based to one-based indexing so that the pattern starts cleanly. Layer 0 becomes position 1 in the cycle, layer 1 becomes position 2, and so on. This shift ensures that layer 0 is always local, never global.</span>
* <span style="font-size: 14px;">**$R + 1$:** The total cycle length. With $R = 5$, the cycle length is 6, meaning every group of 6 consecutive layers contains exactly 5 local layers followed by 1 global layer.</span>
* <span style="font-size: 14px;">**$\bmod$ check against 0:** When the one-based index is a multiple of the cycle length, that layer is global. All other positions in the cycle are local.</span>

<span style="font-size: 14px;">The overall fraction of layers using global attention is $\frac{1}{R + 1}$, and the fraction using local attention is $\frac{R}{R + 1}$. With $R = 5$, that means roughly 16.7% global and 83.3% local. For a model with $L$ total layers, the exact global count is $\lfloor L / (R+1) \rfloor$, and the local count is $L - \lfloor L / (R+1) \rfloor$.</span>

---

## <span style="font-size: 16px;">Why Mix Local and Global</span>

<span style="font-size: 14px;">The fundamental tension in transformer design is between **receptive field** and **compute cost**. Full self-attention gives every layer access to the entire sequence, but the quadratic cost $O(n^2 \cdot d)$ per layer becomes prohibitive for long contexts. Sliding window attention keeps cost linear in sequence length, but each layer can only see a limited neighborhood.</span>

<span style="font-size: 14px;">Mixing the two resolves this tension. Local attention layers handle the bulk of the processing, capturing syntactic structure, local dependencies, and phrase-level patterns that rarely need long-range context. The periodic global layers then aggregate information from across the entire sequence, enabling long-range reasoning and resolution of dependencies that span thousands of tokens.</span>

<span style="font-size: 14px;">The design also improves memory efficiency during inference with KV caching. Local attention layers only need to cache $w$ key-value pairs per layer (the sliding window size), while global layers cache the full sequence. Since most layers are local, the total KV cache size is dramatically reduced. For Gemma 3 27B with $R = 5$, only 10 out of 62 layers maintain a full-length KV cache, and the remaining 52 layers maintain a window-sized cache.</span>

---

## <span style="font-size: 16px;">The Interleaving Pattern</span>

<span style="font-size: 14px;">The hyperparameter $R$ controls the density of global layers in the stack. The cycle length is always $R + 1$. Within each cycle, the first $R$ layers are local and the last layer is global.</span>

<span style="font-size: 14px;">With $R = 5$, the cycle length is 6. Let us trace layers 0 through 11:</span>

* <span style="font-size: 14px;">**Layer 0:** $(0+1) \bmod 6 = 1 \neq 0$ -- **Local**</span>
* <span style="font-size: 14px;">**Layer 1:** $(1+1) \bmod 6 = 2 \neq 0$ -- **Local**</span>
* <span style="font-size: 14px;">**Layer 2:** $(2+1) \bmod 6 = 3 \neq 0$ -- **Local**</span>
* <span style="font-size: 14px;">**Layer 3:** $(3+1) \bmod 6 = 4 \neq 0$ -- **Local**</span>
* <span style="font-size: 14px;">**Layer 4:** $(4+1) \bmod 6 = 5 \neq 0$ -- **Local**</span>
* <span style="font-size: 14px;">**Layer 5:** $(5+1) \bmod 6 = 0$ -- **Global**. First global layer. Attends to all tokens.</span>
* <span style="font-size: 14px;">**Layer 6:** $(6+1) \bmod 6 = 1 \neq 0$ -- **Local**. Cycle resets.</span>
* <span style="font-size: 14px;">**Layer 7:** $(7+1) \bmod 6 = 2 \neq 0$ -- **Local**</span>
* <span style="font-size: 14px;">**Layer 8:** $(8+1) \bmod 6 = 3 \neq 0$ -- **Local**</span>
* <span style="font-size: 14px;">**Layer 9:** $(9+1) \bmod 6 = 4 \neq 0$ -- **Local**</span>
* <span style="font-size: 14px;">**Layer 10:** $(10+1) \bmod 6 = 5 \neq 0$ -- **Local**</span>
* <span style="font-size: 14px;">**Layer 11:** $(11+1) \bmod 6 = 0$ -- **Global**. Second global layer in the stack.</span>

<span style="font-size: 14px;">**Summary:** Out of these 12 layers, layers 5 and 11 are global (2 layers), and layers 0, 1, 2, 3, 4, 6, 7, 8, 9, 10 are local (10 layers). The pattern is [L, L, L, L, L, G, L, L, L, L, L, G], repeating indefinitely.</span>

<span style="font-size: 14px;">Different values of $R$ produce different mixes:</span>

* <span style="font-size: 14px;">**$R = 1$:** Cycle length 2. Alternating Local, Global. 50% of layers are global.</span>
* <span style="font-size: 14px;">**$R = 3$:** Cycle length 4. Three local then one global. 25% global.</span>
* <span style="font-size: 14px;">**$R = 5$:** Cycle length 6. Five local then one global. 16.7% global. This is the Gemma 3 27B setting.</span>
* <span style="font-size: 14px;">**$R = 7$:** Cycle length 8. Seven local then one global. 12.5% global.</span>

---

## <span style="font-size: 16px;">Paper Context</span>

<span style="font-size: 14px;">Gemma 3, released by the Google DeepMind Gemma Team, introduces this local/global attention interleaving as a central architectural choice. The technique appears across Gemma 3 model sizes but is most prominent in the 27B variant, which has 62 transformer layers.</span>

<span style="font-size: 14px;">With $R = 5$, the 62-layer model distributes as follows. The cycle length is 6. The number of complete cycles is $\lfloor 62 / 6 \rfloor = 10$, leaving $62 - 60 = 2$ layers in a final incomplete cycle. This gives exactly 10 global layers (at indices 5, 11, 17, 23, 29, 35, 41, 47, 53, 59) and 52 local layers. The final two layers (60, 61) are both local because the incomplete cycle does not reach position 6 in the pattern.</span>

<span style="font-size: 14px;">The local attention window size in Gemma 3 is $w = 4096$ tokens. Each local layer attends to the 4096 nearest preceding tokens in the causal setting. Global layers attend to the full context window, which extends to 128K tokens for the 27B model.</span>

<span style="font-size: 14px;">The authors note that this design enables Gemma 3 to handle very long contexts efficiently. If all 62 layers used full attention over 128K tokens, the computational and memory costs would be extreme. By making 52 layers local, the effective cost is far closer to a model with a 4K context, while the 10 global layers still provide long-range connectivity for tasks like multi-document question answering.</span>

<span style="font-size: 14px;">This approach builds on prior work from Mistral (sliding window attention in Mistral 7B) and Longformer (mixed local and global attention for long documents). Gemma 3's contribution is integrating this pattern into a large-scale, general-purpose language model with a clean, formula-driven interleaving scheme governed by a single hyperparameter.</span>

---

## <span style="font-size: 16px;">How Information Propagates</span>

<span style="font-size: 14px;">If most layers are local, how does information from token 0 reach token 50,000? The answer involves both global layers and the cumulative effect of local layers through the residual stream.</span>

<span style="font-size: 14px;">Each local layer has a receptive field of $w$ tokens. After passing through $k$ consecutive local layers, a token's representation has been influenced by information from up to $k \cdot w$ tokens away. With $w = 4096$ and 5 consecutive local layers between each global layer, a token can aggregate information from up to $5 \times 4096 = 20{,}480$ positions away through local layers alone in a single cycle.</span>

<span style="font-size: 14px;">The global layers are the real workhorses for long-range communication. When a global layer fires, every token can attend to every other token. After the first global layer (layer 5), token 0 and token 50,000 can directly interact. The residual stream ensures that information injected by a global layer persists through all subsequent local layers. A global layer at depth 5 injects long-range information, and local layers 6 through 10 can build on that information even though they only attend locally.</span>

<span style="font-size: 14px;">This creates a hierarchical information flow: local layers refine nearby context, global layers periodically synchronize across the full sequence, and residual connections ensure nothing is lost between synchronization points.</span>

---

## <span style="font-size: 16px;">Numerical Example</span>

<span style="font-size: 14px;">Let us work through the complete arithmetic for Gemma 3 27B. The model has $L = 62$ layers (indexed 0 to 61) and $R = 5$. The cycle length is $C = R + 1 = 6$.</span>

<span style="font-size: 14px;">**Count global layers.** $\lfloor L / C \rfloor = \lfloor 62 / 6 \rfloor = 10$ global layers.</span>

<span style="font-size: 14px;">**Count local layers.** $L - 10 = 62 - 10 = 52$ local layers.</span>

<span style="font-size: 14px;">**Identify global layer indices.** A layer $i$ is global when $(i + 1) \bmod 6 = 0$, meaning $i + 1$ is a multiple of 6. The multiples of 6 from 1 to 62 are: 6, 12, 18, 24, 30, 36, 42, 48, 54, 60. Subtracting 1: global layers are at indices **5, 11, 17, 23, 29, 35, 41, 47, 53, 59**.</span>

<span style="font-size: 14px;">**Verify the last layers.** Layer 59: $(59 + 1) \bmod 6 = 60 \bmod 6 = 0$. Global. Layer 60: $(60 + 1) \bmod 6 = 61 \bmod 6 = 1 \neq 0$. Local. Layer 61: $(61 + 1) \bmod 6 = 62 \bmod 6 = 2 \neq 0$. Local. The final two layers form an incomplete cycle.</span>

<span style="font-size: 14px;">**Verify fractions.** Global fraction: $10 / 62 \approx 0.161$ (16.1%). Local fraction: $52 / 62 \approx 0.839$ (83.9%). These are close to the theoretical $1/6 \approx 0.167$ and $5/6 \approx 0.833$, with the small discrepancy caused by the incomplete final cycle.</span>

<span style="font-size: 14px;">**Smaller example.** For a model with $L = 8$ layers and $R = 3$ (cycle length 4):</span>

* <span style="font-size: 14px;">Layer 0: $(0+1) \bmod 4 = 1$ -- Local</span>
* <span style="font-size: 14px;">Layer 1: $(1+1) \bmod 4 = 2$ -- Local</span>
* <span style="font-size: 14px;">Layer 2: $(2+1) \bmod 4 = 3$ -- Local</span>
* <span style="font-size: 14px;">Layer 3: $(3+1) \bmod 4 = 0$ -- **Global**</span>
* <span style="font-size: 14px;">Layer 4: $(4+1) \bmod 4 = 1$ -- Local</span>
* <span style="font-size: 14px;">Layer 5: $(5+1) \bmod 4 = 2$ -- Local</span>
* <span style="font-size: 14px;">Layer 6: $(6+1) \bmod 4 = 3$ -- Local</span>
* <span style="font-size: 14px;">Layer 7: $(7+1) \bmod 4 = 0$ -- **Global**</span>

<span style="font-size: 14px;">Result: 2 global layers (indices 3, 7) and 6 local layers. Global fraction: $2/8 = 25\%$, matching the expected $1/4$.</span>

---

## <span style="font-size: 16px;">Comparison with Other Approaches</span>

* <span style="font-size: 14px;">**Full global attention everywhere (GPT-4, LLaMA):** Every layer attends to the full sequence. Maximum representational power but $O(L \cdot n^2 \cdot d)$ total cost. Impractical for very long contexts without hardware-level optimizations.</span>
* <span style="font-size: 14px;">**Sliding window everywhere (Mistral 7B):** Every layer uses a fixed sliding window. Cost is $O(L \cdot n \cdot w \cdot d)$, linear in $n$. Efficient but distant tokens can only interact indirectly through the residual stream.</span>
* <span style="font-size: 14px;">**Longformer-style mixed attention:** Combines local window attention with task-specific global tokens (e.g., [CLS] attends globally). Flexible but requires designating which tokens are global at input level.</span>
* <span style="font-size: 14px;">**Sparse attention (BigBird):** Uses structured sparsity patterns (random, window, global) combined. Complex masks that are harder to implement efficiently on modern hardware.</span>
* <span style="font-size: 14px;">**Gemma 3 interleaving:** Routing is at the layer level, not the token level. No special tokens needed. Each layer has a uniform attention pattern, making it hardware-friendly and trivially computed from the layer index.</span>

---

## <span style="font-size: 16px;">Pitfalls</span>

* <span style="font-size: 14px;">**Off-by-one in the modular formula:** The formula uses $i + 1$, not $i$. If you forget the $+1$, layer 0 would satisfy $0 \bmod 6 = 0$ and incorrectly be classified as global. The shift to one-based indexing ensures that the first layer in every cycle is always local. This is the most common mistake.</span>
* <span style="font-size: 14px;">**Wrong cycle length:** The cycle length is $R + 1$, not $R$. With $R = 5$, the cycle is 6 layers long, not 5. Using $R$ instead of $R + 1$ as the modulus produces $\lfloor 62 / 5 \rfloor = 12$ global layers instead of 10, and the global layers appear at wrong indices (4, 9, 14, ... instead of 5, 11, 17, ...).</span>
* <span style="font-size: 14px;">**The $R = 0$ edge case:** When $R = 0$, the cycle length is $0 + 1 = 1$. For every layer, $(i + 1) \bmod 1 = 0$, so **every layer is global**. This is mathematically correct (zero local layers between each global layer) but code that assumes at least one local layer exists will break. Handle this case explicitly.</span>
* <span style="font-size: 14px;">**Confusing local attention with no attention:** A local attention layer is not a "skip" layer. It computes full self-attention with query, key, and value projections. The only difference is that the attention mask restricts each token to a sliding window of $w$ tokens. It still performs the complete attention computation within that window.</span>
* <span style="font-size: 14px;">**Ignoring the incomplete final cycle:** When $L$ is not a multiple of $R + 1$, the last few layers form an incomplete cycle containing only local layers. For Gemma 3 27B, layers 60 and 61 are in an incomplete cycle. The last global aggregation happens at layer 59, three layers before the output. Forgetting this leads to incorrect layer count calculations.</span>
* <span style="font-size: 14px;">**Thinking $R$ controls absolute counts:** The hyperparameter $R$ controls the ratio within each cycle, not the total number of global or local layers. The actual counts depend on both $R$ and the model depth $L$. Two models with the same $R$ but different depths will have different numbers of global layers.</span>

---