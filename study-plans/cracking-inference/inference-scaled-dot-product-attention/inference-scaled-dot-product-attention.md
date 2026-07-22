# <span style="font-size: 20px;">Scaled Dot-Product Attention</span>

<span style="font-size: 14px;">Scaled dot-product attention converts a set of relevance scores into one contextual vector for each query position. The operation is small enough to express in one equation, but its shape rules, masking convention, numerical behavior, and inference cost determine how full transformer systems behave.</span>

---

## <span style="font-size: 16px;">Contract and notation</span>

<span style="font-size: 14px;">The inputs are queries, keys, and values. Queries describe what each output position is looking for. Keys describe what each source position can be matched against. Values carry the information returned when a key receives attention.</span>

$$
Q \in \mathbb{R}^{B \times S_q \times d_k}
$$

$$
K \in \mathbb{R}^{B \times S_k \times d_k}
$$

$$
V \in \mathbb{R}^{B \times S_k \times d_v}
$$

<span style="font-size: 14px;">Here, $B$ is the batch size, $S_q$ is the number of query positions, and $S_k$ is the number of key and value positions. Queries and keys share $d_k$ because their last dimensions participate in a dot product. Values may use a different width $d_v$ because they are aggregated rather than compared.</span>

$$
\operatorname{Attention}(Q,K,V)
=
\operatorname{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}+M\right)V
$$

<span style="font-size: 14px;">The matrix $M$ represents masking. It contributes zero at permitted positions and negative infinity at blocked positions. With no mask, every element of $M$ is zero. The softmax acts over the final dimension, so every query receives a distribution over the $S_k$ key positions.</span>

---

## <span style="font-size: 16px;">From query-key scores to contextual values</span>

<span style="font-size: 14px;">The product $QK^\top$ forms all query-key comparisons in one batched matrix multiplication. The transpose exchanges the sequence and feature axes of the key tensor while leaving the batch axis intact.</span>

$$
QK^\top:
(B,S_q,d_k)(B,d_k,S_k)
\longrightarrow
(B,S_q,S_k)
$$

<span style="font-size: 14px;">Entry $(b,i,j)$ is the dot product between query $i$ and key $j$ in batch item $b$. A larger score means that the learned representations consider that key more relevant to that query. A score alone is not a probability. It can be negative, positive, or much larger than neighboring scores.</span>

<span style="font-size: 14px;">Softmax converts each score row into nonnegative weights that sum to one. The final multiplication uses those weights to form a weighted sum of the value vectors.</span>

$$
A = \operatorname{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}+M\right)
$$

$$
O=AV
$$

$$
(B,S_q,S_k)(B,S_k,d_v)
\longrightarrow
(B,S_q,d_v)
$$

<span style="font-size: 14px;">The output keeps the batch size and query length. Its final width comes from the values. This shape flow is the fastest way to detect an incorrect transpose or a softmax over the wrong axis.</span>

---

## <span style="font-size: 16px;">Why the scale is tied to the key width</span>

<span style="font-size: 14px;">Assume the components of a query and key are independent, centered, and have unit variance. Their dot product is a sum of $d_k$ products.</span>

$$
q \cdot k = \sum_{r=1}^{d_k} q_r k_r
$$

<span style="font-size: 14px;">The variance of this sum grows in proportion to $d_k$, so its standard deviation grows in proportion to $\sqrt{d_k}$. Wider heads therefore produce larger score magnitudes even when the statistical character of each component stays the same.</span>

<span style="font-size: 14px;">Large score gaps push softmax toward nearly one-hot distributions. That makes small score changes have little effect and can create poorly conditioned gradients during training. Dividing by $\sqrt{d_k}$ keeps score magnitudes in a comparable range across head widths.</span>

$$
S = \frac{QK^\top}{\sqrt{d_k}}
$$

<span style="font-size: 14px;">The divisor uses the shared query-key feature dimension. It does not use the sequence length or value width. Those quantities do not determine the variance of the dot product.</span>

---

## <span style="font-size: 16px;">Worked example</span>

<span style="font-size: 14px;">Consider one query, two keys, and two scalar values. Let the query and keys have width four.</span>

$$
q = [1,1,0,0]
$$

$$
k_1=[1,0,0,0]
$$

$$
k_2=[0,2,0,0]
$$

$$
v_1=10
$$

$$
v_2=20
$$

$$
\sqrt{d_k}=2
$$

<span style="font-size: 14px;">The unscaled dot products are $1$ and $2$. Scaling produces scores $0.5$ and $1.0$.</span>

$$
s = \left[\frac{1}{2},\frac{2}{2}\right]=[0.5,1.0]
$$

<span style="font-size: 14px;">Their softmax weights are approximately $0.3775$ and $0.6225$. The output is the corresponding weighted sum.</span>

$$
o=(0.3775)(10)+(0.6225)(20)=16.225
$$

<span style="font-size: 14px;">If the second key is blocked, its score becomes negative infinity before softmax. The resulting weights are $[1,0]$, so the output becomes $10$. Masking changes which values may contribute, not merely how strongly they contribute.</span>

---

## <span style="font-size: 16px;">Boolean and additive masking</span>

<span style="font-size: 14px;">The function accepts a boolean mask in which true means blocked. The mathematical equation uses an additive mask. Converting between the two conventions is direct: replace a blocked score with negative infinity and leave a permitted score unchanged.</span>

$$
M_{ij}=
\begin{cases}
-\infty, & \text{if position }(i,j)\text{ is blocked} \\
0, & \text{otherwise}
\end{cases}
$$

<span style="font-size: 14px;">Masking must affect the scores before softmax. A blocked entry then contributes $e^{-\infty}=0$ to both the numerator and denominator. Its normalized weight is exactly zero, while the remaining weights are normalized over permitted positions.</span>

<span style="font-size: 14px;">Multiplying weights by a mask after softmax is not equivalent. That approach removes some mass after normalization, leaving the surviving weights with a sum below one. Renormalizing afterward would add work and still obscure the intended score-level semantics.</span>

<span style="font-size: 14px;">Masks may broadcast across batch or query dimensions. A shape of $(1,S_q,S_k)$ can share one pattern across a batch. A shape of $(B,1,S_k)$ can apply one key-padding pattern to every query in each batch item. The broadcast result must match $(B,S_q,S_k)$.</span>

<span style="font-size: 14px;">Every query row needs at least one permitted key. A fully blocked row contains only negative infinity. Stable softmax subtracts the row maximum, but the maximum is also negative infinity, so the subtraction is undefined and the row becomes not-a-number values.</span>

---

## <span style="font-size: 16px;">Self-attention and cross-attention</span>

<span style="font-size: 14px;">Self-attention usually has equal query, key, and value sequence lengths because all three projections originate from the same token sequence. Equality is a common use case, not a requirement of the operation.</span>

<span style="font-size: 14px;">Cross-attention uses queries from one sequence and keys and values from another. An encoder-decoder model may query encoder states using decoder states, so $S_q$ and $S_k$ naturally differ. The output still contains one vector per query.</span>

$$
S_q \neq S_k
\quad\Longrightarrow\quad
O \in \mathbb{R}^{B \times S_q \times d_v}
$$

<span style="font-size: 14px;">Only two alignments are mandatory. Query and key feature widths must match, and key and value sequence lengths must match. The first enables dot products. The second lets each attention weight select the value associated with its key.</span>

---

## <span style="font-size: 16px;">Prefill and autoregressive decoding</span>

<span style="font-size: 14px;">During prefill, a language model processes many prompt tokens together. Queries, keys, and values commonly share the prompt length, while a causal mask blocks every query from using future keys. The score matrix is large and usually triangular in meaning even though it is stored as a dense rectangle.</span>

<span style="font-size: 14px;">During autoregressive decoding, the model often creates one new query at a time and attends to all cached keys and values. Then $S_q=1$ while $S_k$ grows with the prompt and generated sequence. This is a cross-attention shape even though the operation belongs to decoder self-attention.</span>

<span style="font-size: 14px;">The distinction matters for inference optimization. Prefill performs a large matrix multiplication with substantial parallelism. Decode performs many thin query-against-cache operations and often becomes limited by reading the key-value cache from memory.</span>

<span style="font-size: 14px;">A correct primitive must therefore support non-square score matrices without special handling by the caller. Assuming $S_q=S_k$ would break the most common cached-decoding shape.</span>

---

## <span style="font-size: 16px;">Numerically stable softmax</span>

<span style="font-size: 14px;">A direct exponential can overflow when a score is large. Stable softmax subtracts the maximum score in each row before exponentiation.</span>

$$
\operatorname{softmax}(x)_i
=
\frac{e^{x_i-m}}{\sum_j e^{x_j-m}}
$$

$$
m=\max_j x_j
$$

<span style="font-size: 14px;">Subtracting the same constant from each entry leaves the ratio unchanged. It makes the largest shifted score zero, so every exponential is at most one. Negative infinity remains negative infinity after subtracting a finite maximum, preserving exact zero weight for blocked entries.</span>

<span style="font-size: 14px;">PyTorch performs this stabilization inside its softmax operation. Calling that operation on the masked scores is safer and clearer than manually forming exponentials and a denominator. The output should remain in 32-bit floating point for this problem, which also gives comfortable precision for the small test tensors.</span>

<span style="font-size: 14px;">Stable softmax prevents overflow, but it does not repair invalid inputs. Non-finite query, key, or value elements can still contaminate scores. A fully blocked row is also invalid because it has no finite maximum and no probability distribution to normalize.</span>

---

## <span style="font-size: 16px;">Connection to transformer papers and kernels</span>

<span style="font-size: 14px;">The Transformer paper, Attention Is All You Need, defines scaled dot-product attention with this same score, scale, softmax, and value aggregation. Multi-head attention applies several learned projections, evaluates this primitive independently for each head, concatenates the outputs, and applies an output projection.</span>

<span style="font-size: 14px;">Grouped-query attention and multi-query attention alter how key and value heads are shared. They do not change the per-head attention equation. Their inference benefit comes from shrinking the key-value cache and reducing memory traffic during decoding.</span>

<span style="font-size: 14px;">FlashAttention also computes the same exact mathematical result. Its contribution is an input-output aware execution order that tiles the score calculation and maintains running softmax statistics. It avoids storing the full score and weight matrices in high-bandwidth memory.</span>

<span style="font-size: 14px;">This separation between semantics and execution is important. A direct implementation is the clearest correctness reference. Optimized kernels may reorder arithmetic, fuse masking and normalization, or use lower precision internally, but they must preserve the observable attention result within an accepted numerical tolerance.</span>

---

## <span style="font-size: 16px;">Time and memory cost</span>

<span style="font-size: 14px;">Forming the score matrix requires one dot product of width $d_k$ for each query-key pair.</span>

$$
T_{QK^\top}=O(BS_qS_kd_k)
$$

<span style="font-size: 14px;">Masking and softmax visit each score once, costing $O(BS_qS_k)$. Combining the attention weights with values costs $O(BS_qS_kd_v)$. The total arithmetic cost is therefore dominated by the two batched matrix multiplications.</span>

$$
T=O\!\left(BS_qS_k(d_k+d_v)\right)
$$

<span style="font-size: 14px;">The direct algorithm materializes scores and weights with shape $(B,S_q,S_k)$. Their memory cost is $O(BS_qS_k)$. When self-attention length doubles, this term grows by roughly four times.</span>

<span style="font-size: 14px;">During one-token decoding, $S_q$ is one, so score storage is linear in the cache length. The repeated cost still grows over a generation because each new query reads the accumulated keys and values. This explains why cache layout, head sharing, and fused kernels matter even when the equation itself remains unchanged.</span>

---

## <span style="font-size: 16px;">Practical pitfalls</span>

* <span style="font-size: 14px;"><strong>Wrong transpose:</strong> Transposing batch or sequence axes can produce an error or silently compare unrelated items. Only the final two key dimensions should exchange places.</span>
* <span style="font-size: 14px;"><strong>Wrong normalization axis:</strong> Softmax belongs over key positions. Normalizing over queries makes each key distribute mass across outputs, which is a different operation.</span>
* <span style="font-size: 14px;"><strong>Reversed mask meaning:</strong> This problem defines true as blocked. Some libraries define true as permitted, so copying a mask without checking its convention can invert attention.</span>
* <span style="font-size: 14px;"><strong>Post-softmax masking:</strong> Removing weights after normalization leaves rows with less than unit mass and changes the scale of the output.</span>

---

## <span style="font-size: 16px;">Summary</span>

<span style="font-size: 14px;">Scaled dot-product attention compares every query with every key, controls score magnitude using $\sqrt{d_k}$, removes blocked positions before normalization, and uses the resulting distribution to combine values. Its output follows the query length and value width. Correct handling of non-square shapes and boolean masks makes the same primitive valid for self-attention, cross-attention, prompt prefill, and cached autoregressive decoding.</span>

---