# <span style="font-size: 20px;">Attention Sinks</span>

<span style="font-size: 14px;">An **attention sink** is a learned per-head scalar that is added to the softmax denominator inside every self-attention layer of GPT-OSS. It gives each head an "escape valve" so attention weights are not forced to sum to 1, fixing a long-standing constraint that the original Transformer softmax imposed on every layer. The mechanism is one of the smallest but most consequential design choices in the GPT-OSS architecture (OpenAI, 2025).</span>

---

## <span style="font-size: 16px;">The Constraint That Softmax Imposes</span>

<span style="font-size: 14px;">Standard scaled-dot-product attention computes weights by applying softmax over the key axis:</span>

$$
w_{i,j} = \frac{\exp(q_i^\top k_j / \sqrt{d_k})}{\sum_{k'=1}^{n_k} \exp(q_i^\top k_{k'} / \sqrt{d_k})}
$$

<span style="font-size: 14px;">By construction, $\sum_j w_{i,j} = 1$ for every query $i$. The output $o_i = \sum_j w_{i,j} v_j$ is therefore always a convex combination of value vectors, the head cannot output zero, and it cannot output something larger in norm than its largest value vector.</span>

<span style="font-size: 14px;">This sounds fine until you realize: **the head has no way to express "I have nothing useful to say here."** Whatever scores come out of $QK^\top$, the softmax will normalize them and the head will produce a non-trivial output. If the most useful behavior at a given position is to do nothing, the head is forced to attend somewhere anyway and contribute to the residual stream.</span>

---

## <span style="font-size: 16px;">How Models Worked Around It (Before Sinks)</span>

<span style="font-size: 14px;">Researchers observed years ago that trained Transformers do something strange: they assign a huge fraction of their attention mass to a few specific positions, often the very first token, or a small set of "register" tokens, even when those tokens carry no semantic information.</span>

* <span style="font-size: 14px;">**Xiao et al. (2023)** named this phenomenon "attention sinks" in **StreamingLLM** and showed that LLaMA-2 dedicates a large share of its attention to positions 0-3 across nearly every head and layer.</span>
* <span style="font-size: 14px;">**Darcet et al. (2023)** observed the same effect in Vision Transformers and proposed **register tokens**, extra learnable tokens prepended to the sequence, to soak up this leftover attention mass.</span>

<span style="font-size: 14px;">In both cases the model was implicitly learning a workaround: it would drive a few key vectors to have a very specific (often very large) magnitude so that the softmax would dump the unused mass onto them. The first token of the sequence became a kind of dustbin. This worked, but it had costs:</span>

* <span style="font-size: 14px;">**It wasted real positions.** The first few tokens of the context window were burned on bookkeeping rather than content.</span>
* <span style="font-size: 14px;">**It broke streaming inference.** If you evicted the first token from the KV cache (as you must for any sliding-window scheme), the model's behavior degraded catastrophically because its "dustbin" was gone.</span>
* <span style="font-size: 14px;">**It was implicit and brittle.** The exact positions used as sinks differed across heads and could shift between fine-tunes.</span>

---

## <span style="font-size: 16px;">The Off-By-One Insight</span>

<span style="font-size: 14px;">Evan Miller's note "Attention is Off By One" (2023) argued the root cause was the softmax itself: forcing weights to sum to 1 is a strange choice when the natural quantity is "how much attention does the head want to pay." The fix is small, add a $+1$ in the denominator:</span>

$$
\text{softmax}_1(z)_j = \frac{\exp(z_j)}{1 + \sum_{k} \exp(z_k)}
$$

<span style="font-size: 14px;">Now the head can output zero attention by making all $z_j$ very negative, no need to find an unused position to dump mass onto. GPT-OSS generalizes this from a fixed $+1$ to a **learned per-head scalar** $\exp(s_h)$.</span>

---

## <span style="font-size: 16px;">The GPT-OSS Formulation</span>

<span style="font-size: 14px;">Let $S \in \mathbb{R}^{H \times n_q \times n_k}$ be the pre-softmax attention scores (after $QK^\top$, scaling, and any causal or sliding-window mask). Let $\mathbf{s} \in \mathbb{R}^H$ be the learned sink vector, one scalar per attention head. The sink-softmax computes:</span>

$$
w_{h,i,j} = \frac{\exp(S_{h,i,j})}{\exp(s_h) + \sum_{k=1}^{n_k} \exp(S_{h,i,k})}
$$

<span style="font-size: 14px;">Key properties of this formula:</span>

* <span style="font-size: 14px;">**The numerator is unchanged**, only real keys contribute weights.</span>
* <span style="font-size: 14px;">**The denominator gains one extra term** $\exp(s_h)$, independent of $i$ and $j$.</span>
* <span style="font-size: 14px;">**Each row sums to** $\sum_j w_{h,i,j} = \frac{\sum_j \exp(S_{h,i,j})}{\exp(s_h) + \sum_j \exp(S_{h,i,j})} < 1$.</span>
* <span style="font-size: 14px;">**The "missing" mass** $1 - \sum_j w_{h,i,j} = \frac{\exp(s_h)}{\exp(s_h) + \sum_j \exp(S_{h,i,j})}$ is the implicit sink weight; it is never written into the output.</span>

<span style="font-size: 14px;">The reference implementation in OpenAI's gpt-oss repository implements this by concatenating the sink as an extra key, running a standard softmax over the combined axis, and slicing the sink column off before multiplying with V:</span>

* <span style="font-size: 14px;">**Append** $\mathbf{s}_h$ as one extra key for every query position: combined shape $(H, n_q, n_k + 1)$.</span>
* <span style="font-size: 14px;">**Softmax** over the last axis as usual.</span>
* <span style="font-size: 14px;">**Slice off the sink column** with `e[..., :-1]` before $W V$.</span>

<span style="font-size: 14px;">This is equivalent to the closed-form expression above but is the cleanest implementation in autodiff frameworks because it reuses the existing softmax primitive.</span>

---

## <span style="font-size: 16px;">Why the Sink Is Per-Head</span>

<span style="font-size: 14px;">Different heads play different roles. Some heads do precise positional matching and want strong, high-confidence attention. Others do broad averaging or only matter for a subset of tokens. A single global sink would force every head into the same "no-op tolerance", instead, each head learns its own $s_h$.</span>

* <span style="font-size: 14px;">A head with $s_h \to -\infty$ behaves exactly like vanilla softmax (sink is negligible).</span>
* <span style="font-size: 14px;">A head with $s_h \to +\infty$ produces near-zero output (the head is effectively turned off).</span>
* <span style="font-size: 14px;">A head with a moderate $s_h$ outputs a partial answer when confident and nothing when not.</span>

<span style="font-size: 14px;">The sink is also independent of query position $i$. This is a deliberate inductive bias: the head's willingness to "say nothing" is a property of the head, not of where in the sequence we are.</span>

---

## <span style="font-size: 16px;">Numerical Example</span>

<span style="font-size: 14px;">Consider one head, one query, three keys, with scores $S = [1.0, 2.0, 0.5]$ and sink $s = 0.0$.</span>

* <span style="font-size: 14px;">**Without sink (vanilla softmax):**</span>
  - <span style="font-size: 14px;">$\exp([1, 2, 0.5]) = [2.72, 7.39, 1.65]$, sum $= 11.76$</span>
  - <span style="font-size: 14px;">$w = [0.231, 0.628, 0.140]$, row sum $= 1.000$</span>

* <span style="font-size: 14px;">**With sink $s = 0.0$ (so $\exp(s) = 1$):**</span>
  - <span style="font-size: 14px;">denominator $= 11.76 + 1.0 = 12.76$</span>
  - <span style="font-size: 14px;">$w = [0.213, 0.579, 0.129]$, row sum $\approx 0.921$</span>

<span style="font-size: 14px;">The sink absorbed about 8% of the attention mass. The relative ratios between the three keys are preserved, but their absolute magnitudes are smaller. This is the head saying "I am partially confident in these keys, but I am leaking some mass."</span>

<span style="font-size: 14px;">Now imagine the head encounters a query where none of the keys are useful. Vanilla softmax must still produce weights summing to 1, it picks the "least bad" key and amplifies it. With a sink, the head can drive every score very negative and let the sink absorb almost all the mass, producing an output close to zero.</span>

---

## <span style="font-size: 16px;">Numerical Stability</span>

<span style="font-size: 14px;">The standard max-subtraction trick must be applied over the **combined** axis (scores + sink), not over the scores alone. A naïve implementation that subtracts $\max(S_{h,i,:})$ but forgets that $s_h$ could be larger will overflow:</span>

$$
m_{h,i} = \max\!\left(\max_j S_{h,i,j},\; s_h\right)
$$

<span style="font-size: 14px;">All exponentials use $m_{h,i}$ as the offset. If $s_h$ ever drifts high during training, missing it from the max calculation will silently produce inf/NaN gradients.</span>

---

## <span style="font-size: 16px;">GPT-OSS Sinks vs StreamingLLM Sinks</span>

<span style="font-size: 14px;">Both mechanisms are called "attention sinks" but they are fundamentally different:</span>

* <span style="font-size: 14px;">**StreamingLLM sinks (Xiao et al. 2023)** are **position-based**. The first $k$ token positions (typically 0-3) are kept in the KV cache permanently and remain attendable from every later query. There is no change to the softmax formula, the sink is a property of which tokens you keep in the cache, not of the math.</span>
* <span style="font-size: 14px;">**GPT-OSS sinks** are **logit-based** and **per-head**. There is no special position; the sink is one learned scalar per head added inside the softmax denominator. The KV cache contains only real tokens.</span>

<span style="font-size: 14px;">The two are not mutually exclusive. GPT-OSS sinks change the math, while StreamingLLM-style position pinning is a separate KV-cache strategy. GPT-OSS chose the math change because it composes cleanly with sliding-window attention and does not require carving out any positions from the context window.</span>

---

## <span style="font-size: 16px;">Composition With Sliding Window</span>

<span style="font-size: 14px;">GPT-OSS alternates layer-wise between full attention and 128-token sliding-window attention. Without sinks, a sliding-window layer near the start of the sequence has very few keys to attend to (sometimes just one), and softmax over a one-element set is trivially 1.0 on that single key, regardless of whether the key is informative.</span>

<span style="font-size: 14px;">Sinks fix this elegantly: even with a single visible key, the head can choose to put most of the mass on the sink and effectively say nothing. The mechanism is independent of the mask, which is why the same sink mechanism is used in every layer regardless of whether that layer is full-attention or sliding-window.</span>

---

## <span style="font-size: 16px;">Effect on Quantization and Inference</span>

<span style="font-size: 14px;">The model card notes that the sink mechanism interacts well with **MXFP4 quantization** of the MoE weights. Without sinks, models tend to produce a small number of activations with huge magnitudes (the dustbin-attention artifacts), which makes low-bit quantization much harder because the quantization grid has to span an enormous range. With sinks, those outlier activations largely disappear, the dynamic range tightens, and 4-bit quantization preserves quality.</span>

<span style="font-size: 14px;">The sinks themselves are stored in bfloat16 in the reference implementation. They are tiny: one scalar per head per layer, so for gpt-oss-120b with 36 layers and 64 heads, the total sink parameter count is $36 \times 64 = 2{,}304$, completely negligible against the 117B total parameters.</span>

---

## <span style="font-size: 16px;">Initialization and Training Behavior</span>

<span style="font-size: 14px;">A natural question is: what should $s_h$ be initialized to, and how does it evolve during training? The reference implementation initializes $\mathbf{s}$ as an empty bfloat16 parameter (filled by the checkpoint loader), but the simplest sensible default is zero, which gives every head a sink weight of $\frac{1}{1 + n_k}$ at initialization, equivalent to inserting one "neutral" key per row.</span>

<span style="font-size: 14px;">During training, $s_h$ is learned through standard backprop:</span>

* <span style="font-size: 14px;">**Gradient flows through the denominator.** Since $s_h$ appears only in the denominator of every weight, increasing $s_h$ decreases every weight in head $h$, decreasing the head's contribution to the output. The gradient signal is therefore "did this head help or hurt?", heads that contribute noisy or harmful output learn large positive $s_h$ and gradually silence themselves.</span>
* <span style="font-size: 14px;">**No collapse to zero.** Unlike a learned gate placed outside the softmax, the sink cannot drive the head exactly to zero in finite training. The head can become very small, but the keys still receive non-zero weight as long as $S_{h,i,j}$ can grow. This is a useful regularizer, silenced heads can re-activate if a new training signal demands it.</span>

---

## <span style="font-size: 16px;">Related Approaches</span>

* <span style="font-size: 14px;">**Softmax-1 (Miller, 2023):** the fixed $+1$ in the denominator. GPT-OSS sinks generalize this to a per-head learned scalar, recovering softmax-1 as the special case $s_h = 0$ for all heads.</span>
* <span style="font-size: 14px;">**Register tokens (Darcet et al. 2023):** extra learnable tokens prepended to the sequence to absorb sink-like attention mass in Vision Transformers. Solves the same problem but at the sequence level rather than the softmax level. Sinks are cheaper (one scalar vs one full token vector) and do not consume positional slots.</span>
* <span style="font-size: 14px;">**StreamingLLM (Xiao et al. 2023):** keep the first $k$ token positions pinned in the KV cache so they remain attendable. A serving-time strategy rather than a model architecture change. GPT-OSS sinks make this unnecessary because the model never needed those positions to be there in the first place.</span>
* <span style="font-size: 14px;">**Learned attention temperature** (e.g. $\tau_h$ per head): a different head-level scalar applied inside the softmax. Temperature controls peakiness but does not give the head a "say nothing" option. Sinks and temperature can coexist.</span>

---

## <span style="font-size: 16px;">Pitfalls</span>

* <span style="font-size: 14px;">**Forgetting to drop the sink column.** If you concatenate the sink for the softmax and then forget to slice it off with `e[..., :-1]` before $W V$, the output shape will be wrong and you will be multiplying the sink "weight" against nothing meaningful. The bug is loud (shape mismatch crash) which is the best kind.</span>
* <span style="font-size: 14px;">**Subtracting the max only over scores.** The max-subtraction for numerical stability must include the sink. If $s_h$ is the largest entry, leaving it out of the max means $\exp(s_h - m)$ can still overflow.</span>
* <span style="font-size: 14px;">**Confusing the sink with a token.** The sink does not consume a position in the sequence and is not stored in the KV cache. It is a learned bias on the softmax denominator, not a key vector. Treating it like a special token (the StreamingLLM interpretation) leads to the wrong mechanism entirely.</span>
* <span style="font-size: 14px;">**Sharing the sink across heads.** Each head has its own sink. A single global scalar shared by all heads removes the per-head expressiveness, heads that need to be "loud" and heads that need to be "silent" can no longer coexist.</span>
* <span style="font-size: 14px;">**Adding the sink to the keys.** A common misreading is to add $s_h$ to every score before softmax, i.e. $S'_{h,i,j} = S_{h,i,j} + s_h$. This shifts every score by the same constant per head, which is a no-op for the softmax, it cancels in the normalization. Sinks must be appended, not added.</span>
* <span style="font-size: 14px;">**Expecting weights to sum to 1.** Downstream code that asserts `weights.sum(dim=-1) == 1` will break. This is by design, the sum is strictly less than 1, and the "missing" mass is what gives the head its escape valve.</span>

---
