# <span style="font-size: 20px;">Implement Multi-Head Latent Attention (MLA)</span>

<span style="font-size: 14px;">Multi-head latent attention stores a compressed latent representation for keys and values, then reconstructs the per-head states needed by attention.</span>

---

## <span style="font-size: 16px;">A different axis of compression</span>

<span style="font-size: 14px;">Grouped-query and multi-query attention reduce KV-cache size by sharing entire key/value heads across multiple query heads: fewer independent key and value tensors, but each one is full width. Multi-head latent attention takes an orthogonal approach: keep one independent key and value per query head structurally, but store them in a compressed, low-rank shared representation and reconstruct the full-width versions on demand.</span>

<span style="font-size: 14px;">Where GQA and MQA answer "how many separate key/value heads do we need," MLA answers "how many bits does each token's cached state actually require," compressing the representation itself rather than reducing how many heads have their own representation.</span>

$$
c = X W_{down}
$$

$$
K = c \, W_{up}^{K}
$$

$$
V = c \, W_{up}^{V}
$$

$$
Q = X W_q
$$

---

## <span style="font-size: 16px;">Why the latent is a genuine factorization, not an approximation</span>

<span style="font-size: 14px;">A direct key projection would compute $K = X W_k$ for some learned $(d_{model}, d_{model})$ matrix $W_k$. Multi-head latent attention instead computes $K = (X W_{down}) W_{up}^{K} = X (W_{down} W_{up}^{K})$, which, by associativity of matrix multiplication, is exactly $X$ multiplied by the single combined matrix $W_{down} W_{up}^{K}$.</span>

<span style="font-size: 14px;">This means MLA does not approximate a full-rank key projection; it restricts $K$'s effective projection matrix to have rank at most $d_{latent}$, since $W_{down} W_{up}^{K}$ is a product of a $(d_{model}, d_{latent})$ and a $(d_{latent}, d_{model})$ matrix, which can have rank at most $d_{latent}$.</span>

<span style="font-size: 14px;">When $d_{latent} < d_{model}$, this is a real representational restriction relative to an unconstrained $W_k$, but the two-stage computation itself introduces no numerical error or approximation beyond that rank restriction: given the same $W_{down}$ and $W_{up}^{K}$, computing $K$ through the latent produces bit-for-bit the same result (up to floating-point rounding) as computing it through the single combined matrix.</span>

---

## <span style="font-size: 16px;">Why the same latent produces both keys and values</span>

<span style="font-size: 14px;">$W_{down}$ is shared between the key and value reconstruction paths; only the up-projection matrices $W_{up}^{K}$ and $W_{up}^{V}$ differ. This means a single compressed vector per token carries enough information to reconstruct both a key and a value, using two different learned "read-out" directions from the same underlying compressed representation. This is what makes the cache genuinely smaller rather than merely reorganized: caching the latent once, rather than caching a separately compressed key-latent and value-latent, means the per-token cache entry has size $d_{latent}$, not $2 \cdot d_{latent}$.</span>

---

## <span style="font-size: 16px;">Why queries are not compressed here</span>

<span style="font-size: 14px;">Only keys and values pass through the down-projection and up-projection; queries are projected directly from hidden_states with an ordinary $W_q$. This is a deliberate asymmetry: the query for the current position is computed once per generation step and never needs to be cached or reused across steps, so there is no inference-time memory benefit to compressing it.</span>

<span style="font-size: 14px;">Keys and values, by contrast, must be retained for every previous position for the entire generation, which is exactly why compressing them, and only them, addresses the actual memory bottleneck. Some production variants of this idea also compress the query for parameter-efficiency reasons unrelated to caching, but that is a separate concern from the inference-memory motivation this problem focuses on, which is why it is out of scope here.</span>

---

## <span style="font-size: 16px;">Reconstruction happens once per step, not once per head</span>

<span style="font-size: 14px;">Because the latent has no head dimension, key and value are reconstructed to their full $d_{model}$ width in a single matrix multiply each, and only afterward split into num_heads heads, exactly as in standard multi-head attention.</span>

<span style="font-size: 14px;">This ordering matters: reconstructing per-head keys and values separately, by giving $W_{up}^{K}$ a head dimension and looping, would be mathematically equivalent but implemented far less efficiently than one full-width matrix multiply followed by a reshape, since a single larger matmul is significantly better utilized on real hardware than several smaller ones producing the same total output.</span>

---

## <span style="font-size: 16px;">Inference motivation</span>

<span style="font-size: 14px;">During autoregressive decoding, what a serving system caches directly determines both the memory footprint of long-context generation and the bandwidth cost of reading that cache back at every subsequent step. Caching full-width keys and values, as in standard multi-head attention, means the cache grows with $num\_heads \cdot d_k = d_{model}$ per token, per layer.</span>

<span style="font-size: 14px;">Caching only the latent means the cache grows with $d_{latent}$ per token, per layer, reconstructing the full-width keys and values on demand at attention time rather than storing them directly. When $d_{latent} \ll d_{model}$, this is a substantial reduction in cache memory and bandwidth without discarding any query head's ability to have its own effective key and value, which is the specific limitation multi-query attention accepts and grouped-query attention partially accepts.</span>

<span style="font-size: 14px;">This is why latent compression is a genuinely different tool from head-sharing, and production systems that need both aggressive memory reduction and high per-head representational capacity combine ideas from both families rather than choosing one exclusively.</span>

---

## <span style="font-size: 16px;">Complexity</span>

<span style="font-size: 14px;">Computing the latent costs $O(B \cdot S \cdot d_{model} \cdot d_{latent})$. Each up-projection costs $O(B \cdot S \cdot d_{latent} \cdot d_{model})$, so reconstructing both $K$ and $V$ costs twice that. When $d_{latent} < d_{model}$, the total cost of one down-projection plus two up-projections is less than directly computing two full $(d_{model}, d_{model})$ projections for $K$ and $V$, since $O(d_{model} \cdot d_{latent})$ per projection beats $O(d_{model}^2)$ whenever $d_{latent} < d_{model}$.</span>

<span style="font-size: 14px;">The attention computation itself, after reconstruction, costs $O(B \cdot num\_heads \cdot S^2 \cdot d_k)$, identical to standard multi-head attention, since compression affects the key/value projection cost, not the number of per-head attention computations.</span>

---

## <span style="font-size: 16px;">Memory behavior</span>

<span style="font-size: 14px;">The quantity that matters for inference is not the size of the reconstructed key and value tensors, which match standard multi-head attention's full-width tensors, but the size of latent, which is $O(B \cdot S \cdot d_{latent})$. A real serving system caches only latent across decoding steps and reconstructs key and value fresh at each step's attention computation; the reconstructed tensors are transient, not part of the persistent cache.</span>

<span style="font-size: 14px;">This is the precise sense in which multi-head latent attention reduces KV-cache memory: the compression ratio is $d_{model} / d_{latent}$ relative to caching full-width keys and values, independent of num_heads, unlike grouped-query attention, whose reduction ratio depends on how many query heads share each key/value head.</span>

---

## <span style="font-size: 16px;">Common failure modes</span>

* <span style="font-size: 14px;">Reconstructing $K$ or $V$ through any path other than the latent, such as projecting directly from hidden_states with a separate combined matrix, which may be numerically correct in isolation but does not reflect what an inference system can actually cache and reconstruct.</span>

* <span style="font-size: 14px;">Giving the latent, or the up-projection matrices, a head dimension, which breaks the shared-compression structure that makes the cache smaller than per-head compression would be.</span>

* <span style="font-size: 14px;">Forgetting to return the latent alongside the output, which discards the one piece of information a caching system actually needs to persist.</span>

* <span style="font-size: 14px;">Compressing queries as well as keys and values without being asked to, which changes the problem's scope and its cache-size accounting.</span>

* <span style="font-size: 14px;">Assuming $d_{latent} < d_{model}$ is required for correctness; the computation is well defined for any $1 \leq d_{latent} \leq d_{model}$, and only the memory-saving motivation depends on choosing $d_{latent}$ meaningfully smaller than $d_{model}$.</span>

---

## <span style="font-size: 16px;">Contract and notation</span>

<span style="font-size: 14px;">$X$ denotes hidden states, $C=XW_{down}$ is the cacheable latent tensor, $W_{up}^K$ and $W_{up}^V$ reconstruct keys and values, $h$ is the head count, and $d_c$ is the latent width.</span>

<span style="font-size: 14px;">Hidden states and output have shape $(B,S,d_{model})$; the latent has shape $(B,S,d_c)$; reconstructed keys and values split to $(B,h,S,d_k)$.</span>

---

## <span style="font-size: 16px;">Worked example</span>

<span style="font-size: 14px;">Input: hidden_states.shape = (1, 2, 4), d_latent = 2, num_heads = 1, causal = False</span>

<span style="font-size: 14px;">Output: output tensor of shape (1, 2, 4), e.g. row 0 is [-0.0639, 0.0558, -0.2386, 0.0318]; latent tensor of shape (1, 2, 2), e.g. row 0 is [-0.2309, -0.7171]</span>

<span style="font-size: 14px;">Explanation: The down-projection produces the single compact state that is eligible for caching.</span>

---

## <span style="font-size: 16px;">Paper and system context</span>

<span style="font-size: 14px;">DeepSeek-V2 introduced MLA as a cache-compression mechanism based on low-rank latent states. The complete architecture contains additional details, while this problem isolates the inference-relevant compressed KV path.</span>

<span style="font-size: 14px;">The distinction between persistent and transient state is central. Reconstructed keys and values may exist during a computation, but only the latent must survive between decoding steps, which is where the memory reduction appears.</span>

---

## <span style="font-size: 16px;">Correctness invariants</span>

<span style="font-size: 14px;">The down-projection produces the single compact state that is eligible for caching.</span>

<span style="font-size: 14px;">Independent up-projections reconstruct key and value features from the same latent without changing token order.</span>

<span style="font-size: 14px;">Attention over the reconstructed heads follows the same scaled and causal contract as MHA.</span>

<span style="font-size: 14px;">The latent width may differ from both model width and head width, so shape validation must follow the supplied projection matrices rather than assume equality.</span>

---

## <span style="font-size: 16px;">Validation strategy</span>

<span style="font-size: 14px;">Validation should independently reconstruct keys and values from the returned latent and compare the attention output with a direct reference. It should also verify the latent shape before checking numerical values.</span>

<span style="font-size: 14px;">Causal behavior can be tested by changing a future hidden state and confirming that every earlier output remains fixed even though the shared latent tensor changes at that future position.</span>

---

## <span style="font-size: 16px;">Behavioral contract</span>

<span style="font-size: 14px;">Accept hidden states, query and output projections, one KV down-projection, separate key and value up-projections, a head count, and a causal flag.</span>

<span style="font-size: 14px;">Return both the projected attention output and the compressed latent tensor.</span>

<span style="font-size: 14px;">The latent tensor must have the input batch size and sequence length with the configured latent width.</span>

<span style="font-size: 14px;">Reconstructed keys and values must provide one state for every attention head and token.</span>

<span style="font-size: 14px;">When causal mode is enabled, earlier output positions must not depend on later hidden states.</span>

<span style="font-size: 14px;">The returned attention output must match direct reconstruction from the returned latent tensor.</span>

<span style="font-size: 14px;">Both returned tensors must be finite for valid inputs.</span>

---