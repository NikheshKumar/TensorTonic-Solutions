# <span style="font-size: 20px;">Top-k Sampling</span>

<span style="font-size: 14px;">Top-k sampling is a decoding strategy for autoregressive language models that restricts the next-token selection to only the $k$ highest-scoring candidates. Rather than sampling from the entire vocabulary (risking rare, incoherent tokens) or always picking the single best token (producing repetitive text), top-k narrows the pool to a fixed number of plausible continuations, then samples from that restricted set. Combined with temperature scaling, it provides two knobs to control the tradeoff between creativity and coherence.</span>

---

## <span style="font-size: 16px;">What It Is</span>

<span style="font-size: 14px;">Top-k sampling is restricted random sampling from top candidates. At each decoding step, the model produces a logit (raw score) for every token in its vocabulary. Instead of sampling from the full softmax distribution, top-k keeps only the $k$ tokens with the highest logits, sets all other logits to $-\infty$, then applies softmax to the filtered logits. The result is a probability distribution concentrated entirely on the top $k$ candidates, from which a single token is drawn.</span>

<span style="font-size: 14px;">Language model output distributions are typically very peaked: a small number of tokens carry most of the probability mass, while the long tail of low-probability tokens collectively can cause nonsensical outputs when sampled. By discarding this tail, top-k eliminates the worst failure modes of unrestricted sampling while preserving diversity among plausible continuations.</span>

<span style="font-size: 14px;">Temperature scaling is applied before the top-k filter. Dividing logits by $T$ reshapes the distribution's sharpness, then top-k removes the tail. Temperature controls how spread out the probability mass is among kept tokens, while $k$ controls how many tokens share that mass.</span>

---

## <span style="font-size: 16px;">Key Equations</span>

<span style="font-size: 14px;">Let $z_i$ be the raw logit for token $i$ in a vocabulary of size $V$. The full top-k sampling pipeline has three stages:</span>

<span style="font-size: 14px;">**Step 1: Temperature scaling.** Divide all logits by temperature $T > 0$:</span>

$$
z_i' = \frac{z_i}{T}
$$

<span style="font-size: 14px;">**Step 2: Top-k filtering.** Sort tokens by $z_i'$ in descending order. Let $S_k$ be the set of indices for the top $k$ tokens. Set all other logits to negative infinity:</span>

$$
\tilde{z}_i = \begin{cases} z_i' & \text{if } i \in S_k \\ -\infty & \text{otherwise} \end{cases}
$$

<span style="font-size: 14px;">**Step 3: Softmax and sampling.** Apply softmax to the filtered logits to obtain a valid probability distribution, then sample one token:</span>

$$
p_i = \frac{e^{\tilde{z}_i}}{\sum_{j=1}^{V} e^{\tilde{z}_j}} = \begin{cases} \frac{e^{z_i'}}{\sum_{j \in S_k} e^{z_j'}} & \text{if } i \in S_k \\ 0 & \text{otherwise} \end{cases}
$$

<span style="font-size: 14px;">The second equality holds because $e^{-\infty} = 0$, so tokens outside $S_k$ contribute nothing to either the numerator or the denominator. The softmax effectively renormalizes probability mass over just the top $k$ tokens.</span>

---

## <span style="font-size: 16px;">Temperature Scaling</span>

<span style="font-size: 14px;">Temperature is borrowed from statistical mechanics, where it controls the entropy of a Boltzmann distribution. In the context of language model decoding, temperature reshapes the probability distribution before filtering.</span>

<span style="font-size: 14px;">**$T = 1$ (neutral):** The logits are used as-is. The probability distribution matches whatever the model learned during training. This is the default and serves as the baseline.</span>

<span style="font-size: 14px;">**$T < 1$ (sharpening):** Dividing by a number less than 1 amplifies the logits, making large logits even larger relative to small ones. After softmax, the distribution becomes more peaked. As $T \to 0^+$, the distribution collapses to a point mass on the argmax token (greedy decoding). Lower temperature produces more predictable, conservative text.</span>

<span style="font-size: 14px;">**$T > 1$ (flattening):** Dividing by a number greater than 1 compresses the logits toward zero, reducing the gap between high and low scores. After softmax, probability mass spreads more evenly across candidates. As $T \to \infty$, the distribution approaches uniform. Higher temperature produces more diverse, surprising (and potentially incoherent) text.</span>

<span style="font-size: 14px;">The connection to entropy is direct. Shannon entropy $H = -\sum_i p_i \log p_i$ decreases with low temperature (less randomness) and increases with high temperature (more randomness). Temperature gives continuous control over this spectrum.</span>

<span style="font-size: 14px;">An important subtlety: temperature is applied before top-k filtering, not after. This matters because temperature changes which tokens make it into the top $k$. At $T = 0.5$, the gap between the best and second-best token is doubled. At $T = 2.0$, gaps shrink, so the top-$k$ set may include tokens that would have been negligible at $T = 1$.</span>

---

## <span style="font-size: 16px;">Top-k Filtering</span>

<span style="font-size: 14px;">The filtering step sorts all $V$ logits in descending order, keeps the first $k$, and sets the remaining $V - k$ logits to $-\infty$.</span>

<span style="font-size: 14px;">Why $-\infty$ specifically? After softmax, $e^{-\infty} = 0$, so filtered-out tokens receive exactly zero probability. There is no chance of accidentally sampling a discarded token. Any large negative number (like $-10^9$) would approximate this, but $-\infty$ is the mathematically clean choice.</span>

<span style="font-size: 14px;">The parameter $k$ determines how many candidates survive:</span>

* <span style="font-size: 14px;">**Small $k$ (e.g., 5-10):** Only the most confident predictions survive. Coherent but potentially repetitive.</span>
* <span style="font-size: 14px;">**Large $k$ (e.g., 100-500):** Includes lower-ranked but contextually valid tokens. More diverse but risks incoherence.</span>
* <span style="font-size: 14px;">**$k = 1$:** Degenerates to greedy decoding. No randomness regardless of temperature.</span>
* <span style="font-size: 14px;">**$k = V$:** No filtering. Reduces to pure temperature sampling from the full softmax.</span>

<span style="font-size: 14px;">Creative writing might use $k = 40$ (as in GPT-2 demos), while factual QA might use $k = 5$ or $k = 1$.</span>

---

## <span style="font-size: 16px;">The Sampling Step</span>

<span style="font-size: 14px;">After filtering, the surviving $k$ logits are passed through softmax to get a valid probability distribution. A token is then drawn using the cumulative distribution function (CDF):</span>

* <span style="font-size: 14px;">**Compute probabilities:** Softmax the $k$ surviving logits, yielding $p_1, p_2, \ldots, p_k$ sorted in descending order.</span>
* <span style="font-size: 14px;">**Build the CDF:** $C_j = \sum_{i=1}^{j} p_i$, giving $C_1 = p_1$, $C_2 = p_1 + p_2$, up to $C_k = 1.0$.</span>
* <span style="font-size: 14px;">**Draw a random value:** Generate $r \sim \text{Uniform}[0, 1)$.</span>
* <span style="font-size: 14px;">**Select the token:** Find the smallest $j$ such that $C_j > r$. That token is the output.</span>

<span style="font-size: 14px;">This works because $\Pr(r \in [C_{j-1}, C_j)) = p_j$, so each token is selected with its softmax probability. For deterministic testing, $r$ is fixed rather than random, making the output reproducible given the same logits, temperature, $k$, and random value.</span>

---

## <span style="font-size: 16px;">Paper Context</span>

<span style="font-size: 14px;">Top-k sampling was introduced by Fan et al. in "Hierarchical Neural Story Generation" (2018). The paper addressed a core problem: beam search and greedy decoding produced fluent but dull, repetitive text, while pure sampling from the full softmax frequently went off the rails. Fan et al. proposed a simple fix: truncate the distribution to the $k$ most likely words before sampling. They used $k = 10$ for story generation and found it was both more coherent than full sampling and more interesting than beam search.</span>

<span style="font-size: 14px;">GPT-2 (Radford et al., 2019) brought top-k sampling to widespread attention. OpenAI used $k = 40$ for the generation demos, producing the famous "unicorn" story: a fabricated news article about a herd of unicorns discovered in the Andes, generated entirely by the model. The text was so convincing that it sparked broad discussion about potential misuse of language models, and OpenAI initially withheld the full model citing safety concerns.</span>

<span style="font-size: 14px;">The choice of $k = 40$ was tuned empirically for good output quality across diverse prompts. It became a de facto default in many subsequent implementations.</span>

---

## <span style="font-size: 16px;">Numerical Example</span>

<span style="font-size: 14px;">Consider a vocabulary of $V = 5$ tokens with raw logits:</span>

* <span style="font-size: 14px;">**Token A:** $z_A = 5.0$</span>
* <span style="font-size: 14px;">**Token B:** $z_B = 3.0$</span>
* <span style="font-size: 14px;">**Token C:** $z_C = 1.5$</span>
* <span style="font-size: 14px;">**Token D:** $z_D = 0.5$</span>
* <span style="font-size: 14px;">**Token E:** $z_E = -1.0$</span>

<span style="font-size: 14px;">Parameters: $T = 0.8$, $k = 3$, random value $r = 0.75$.</span>

### <span style="font-size: 14px;">Step 1: Temperature Scaling ($T = 0.8$)</span>

<span style="font-size: 14px;">Divide each logit by $T = 0.8$:</span>

$$
z_A' = 5.0 / 0.8 = 6.25, \quad z_B' = 3.0 / 0.8 = 3.75, \quad z_C' = 1.5 / 0.8 = 1.875
$$

$$
z_D' = 0.5 / 0.8 = 0.625, \quad z_E' = -1.0 / 0.8 = -1.25
$$

<span style="font-size: 14px;">Dividing by $T < 1$ amplified the differences. The gap between $z_A'$ and $z_B'$ is now $2.5$ instead of the original $2.0$.</span>

### <span style="font-size: 14px;">Step 2: Top-k Filtering ($k = 3$)</span>

<span style="font-size: 14px;">The three highest scaled logits are $z_A' = 6.25$, $z_B' = 3.75$, and $z_C' = 1.875$. Set the rest to $-\infty$:</span>

$$
\tilde{z}_A = 6.25, \quad \tilde{z}_B = 3.75, \quad \tilde{z}_C = 1.875, \quad \tilde{z}_D = -\infty, \quad \tilde{z}_E = -\infty
$$

<span style="font-size: 14px;">Tokens D and E are eliminated from consideration. No matter what random value is drawn, they cannot be selected.</span>

### <span style="font-size: 14px;">Step 3: Softmax Over Filtered Logits</span>

<span style="font-size: 14px;">Compute the exponentials of the surviving logits:</span>

$$
e^{6.25} \approx 518.01, \quad e^{3.75} \approx 42.52, \quad e^{1.875} \approx 6.52
$$

<span style="font-size: 14px;">The normalization constant is:</span>

$$
Z = 518.01 + 42.52 + 6.52 = 567.05
$$

<span style="font-size: 14px;">The probabilities are:</span>

$$
p_A = \frac{518.01}{567.05} \approx 0.9135, \quad p_B = \frac{42.52}{567.05} \approx 0.0750, \quad p_C = \frac{6.52}{567.05} \approx 0.0115
$$

<span style="font-size: 14px;">Token A dominates with 91.35% of the mass. The low temperature ($T = 0.8$) has sharpened the distribution compared to $T = 1$.</span>

### <span style="font-size: 14px;">Step 4: Sampling with $r = 0.75$</span>

<span style="font-size: 14px;">Build the cumulative distribution:</span>

$$
C_A = 0.9135, \quad C_B = 0.9135 + 0.0750 = 0.9885, \quad C_C = 0.9885 + 0.0115 = 1.0
$$

<span style="font-size: 14px;">Find the smallest index where $C_j > r = 0.75$. Since $C_A = 0.9135 > 0.75$, the condition is satisfied at the first token. **Token A is selected.**</span>

<span style="font-size: 14px;">If $r = 0.95$, then $C_A = 0.9135 < 0.95$ (fail), $C_B = 0.9885 > 0.95$ (pass), so Token B would be selected. Only $r > 0.9885$ would select Token C.</span>

---

## <span style="font-size: 16px;">Top-k vs Top-p (Nucleus Sampling)</span>

<span style="font-size: 14px;">Top-p sampling, also called nucleus sampling (Holtzman et al., "The Curious Case of Neural Text Degeneration," 2020), was proposed as a direct improvement over top-k. The core insight: a fixed $k$ is a poor fit for the varying shape of model output distributions.</span>

<span style="font-size: 14px;">**Top-k: fixed count.** Always keeps exactly $k$ tokens regardless of how probability mass is distributed. If the model is very confident (one token has 99% probability), top-k still keeps $k - 1$ near-zero-probability tokens. If the model is uncertain, top-k may cut off tokens that carry meaningful mass.</span>

<span style="font-size: 14px;">**Top-p: adaptive count.** Keeps the smallest set of tokens whose cumulative probability exceeds threshold $p$ (e.g., $p = 0.9$). When the model is confident, top-p might keep only 2-3 tokens. When uncertain, it might keep 50 or more. The method adapts to the distribution shape automatically.</span>

<span style="font-size: 14px;">Example: if probabilities are $[0.95, 0.03, 0.01, 0.005, 0.005]$, top-k with $k=3$ keeps 3 tokens, but top-p with $p=0.9$ keeps just 1 (since 0.95 already exceeds 0.9). If probabilities are $[0.15, 0.14, 0.13, 0.12, \ldots]$, top-k still keeps 3 but top-p keeps around 7. Top-p naturally tightens when the model is sure and loosens when it is not.</span>

<span style="font-size: 14px;">Tradeoffs between the two approaches:</span>

* <span style="font-size: 14px;">**Simplicity:** Top-k is simpler to implement and reason about. The parameter $k$ has an obvious meaning (number of candidates), while $p$ requires understanding cumulative probability.</span>
* <span style="font-size: 14px;">**Adaptiveness:** Top-p is strictly more adaptive. It never wastes probability budget on near-zero tokens and never truncates tokens that carry significant mass.</span>
* <span style="font-size: 14px;">**Combination:** In practice, many systems apply both: top-k first (as a hard ceiling on candidates) followed by top-p (to further trim if probability is concentrated). This combines top-k's worst-case bound with top-p's adaptiveness.</span>

---

## <span style="font-size: 16px;">Pitfalls</span>

### <span style="font-size: 14px;">$k = 1$ Is Greedy Decoding</span>

<span style="font-size: 14px;">Setting $k = 1$ reduces top-k sampling to greedy decoding: only the highest-logit token survives, and the softmax of a single element is always 1.0. Temperature becomes irrelevant because there is no choice to make. This is a valid but degenerate case.</span>

### <span style="font-size: 14px;">$k = V$ Is Pure Temperature Sampling</span>

<span style="font-size: 14px;">Setting $k = V$ means no tokens are filtered. The method becomes vanilla temperature sampling from the full softmax. All tail-cutting benefits are lost, and even the lowest-probability tokens in a large vocabulary can produce gibberish when sampled.</span>

### <span style="font-size: 14px;">$T = 0$ Causes Division by Zero</span>

<span style="font-size: 14px;">Temperature scaling divides logits by $T$. If $T = 0$, this is division by zero. Conceptually, $T \to 0^+$ converges to greedy decoding (all mass on the argmax), but the limit cannot be evaluated directly. Implementations must special-case $T = 0$ by returning the argmax token without the division. Failing to handle this produces NaN values that propagate through softmax.</span>

### <span style="font-size: 14px;">Applying Temperature After Top-k Filtering</span>

<span style="font-size: 14px;">Filtering first, then applying temperature, yields different results. While temperature does not change the logit ranking (monotonic transform), it changes the probability distribution within the top $k$, affecting sampling outcomes. The correct order is: scale, filter, softmax.</span>

### <span style="font-size: 14px;">Wrong Cumulative Probability Direction</span>

<span style="font-size: 14px;">When implementing CDF-based sampling, a subtle bug arises from accumulating probabilities in ascending rather than descending order. The cumulative distribution must be built from the highest-probability token to the lowest, and the selected token is the first one where the cumulative sum exceeds $r$. Reversing this direction flips the sampling bias: high-probability tokens get sampled too rarely and low-probability tokens too often.</span>

### <span style="font-size: 14px;">Forgetting Temperature Before Top-k Selection</span>

<span style="font-size: 14px;">Some implementations skip temperature scaling entirely and apply top-k filtering directly to raw logits. This works at $T = 1$ (where scaling is a no-op) but silently produces wrong results for any other temperature. The distribution shape within the top $k$ will not reflect the intended temperature, and the bug is insidious because output may look superficially reasonable while being statistically incorrect.</span>