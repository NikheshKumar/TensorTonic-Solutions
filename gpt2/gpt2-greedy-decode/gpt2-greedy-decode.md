# <span style="font-size: 20px;">Greedy Decoding</span>

<span style="font-size: 14px;">Greedy decoding is the simplest text generation strategy for autoregressive language models. At each time step, the model produces a vector of logits over the vocabulary, and greedy decoding selects the token with the highest logit. That token is appended to the sequence, the extended sequence is fed back into the model, and the process repeats. No randomness, no search -- just the single best-looking token at every step.</span>

<span style="font-size: 14px;">Every more sophisticated decoding strategy -- beam search, top-k sampling, nucleus sampling -- is a deliberate departure from this baseline. Understanding greedy decoding means understanding the autoregressive generation loop itself.</span>

---

## <span style="font-size: 16px;">What It Is</span>

<span style="font-size: 14px;">Greedy decoding is a **deterministic, one-step-lookahead** generation algorithm. At each position, it makes the locally optimal choice: pick whichever token has the highest logit. It never reconsiders past choices and never explores alternatives.</span>

<span style="font-size: 14px;">You start with a prompt (a sequence of token IDs), feed it to the model, get back logits for the next token, take the argmax, and append that token ID to the sequence. Feed the extended sequence back and repeat. After $N$ steps, stop and return the generated sequence.</span>

<span style="font-size: 14px;">The term "greedy" comes from algorithm design: make the best local choice at each step without considering global consequences. Here, that means always picking the highest-scoring next token without asking whether a slightly worse token now might lead to a much better sequence overall.</span>

<span style="font-size: 14px;">In this problem, you use a deterministic lookup table mapping token sequences (as tuples) to logit vectors, isolating decoding logic from the model forward pass. If the current sequence is not in the table, generation stops early.</span>

---

## <span style="font-size: 16px;">Key Equations</span>

<span style="font-size: 14px;">**The core operation -- argmax over logits:**</span>

$$
\texttt{next\_token} = \arg\max_{v \in \mathcal{V}} \; \texttt{logits}[v]
$$

<span style="font-size: 14px;">where $\mathcal{V} = \{0, 1, \ldots, V-1\}$ is the vocabulary and $\texttt{logits} \in \mathbb{R}^V$ is the raw output of the model's final linear layer. Argmax returns the index of the largest element.</span>

<span style="font-size: 14px;">**Logits vs probabilities.** Logits are unnormalized. To convert to probabilities, apply softmax:</span>

$$
P(\text{token} = v \mid \text{context}) = \frac{e^{\texttt{logits}[v]}}{\sum_{j=0}^{V-1} e^{\texttt{logits}[j]}}
$$

<span style="font-size: 14px;">Greedy decoding does not need this. Since softmax is monotonic, the argmax of logits equals the argmax of probabilities. Skipping softmax saves computation.</span>

<span style="font-size: 14px;">**The generation loop.** Let $x_0 = (t_1, t_2, \ldots, t_k)$ be the prompt of $k$ tokens:</span>

$$
t_{k+i} = \arg\max \; f_\theta(x_{i-1}) \quad \text{for } i = 1, 2, \ldots, N
$$

$$
x_i = (t_1, t_2, \ldots, t_{k+i})
$$

<span style="font-size: 14px;">where $f_\theta$ is the language model returning logits for the next position. At each step, the sequence grows by one token.</span>

---

## <span style="font-size: 16px;">The Autoregressive Loop</span>

<span style="font-size: 14px;">Autoregressive generation produces one token at a time, left to right. Each new token depends on all previous tokens. This is how all decoder-only models like GPT-2 generate text. Greedy decoding is one strategy for choosing which token to emit.</span>

<span style="font-size: 14px;">The loop has four phases per iteration:</span>

* <span style="font-size: 14px;">**Feed:** Pass the current sequence into the model (or lookup table) to get logits for the next position.</span>
* <span style="font-size: 14px;">**Select:** Argmax the logits to find the highest-scoring token.</span>
* <span style="font-size: 14px;">**Append:** Add the selected token ID to the sequence.</span>
* <span style="font-size: 14px;">**Repeat:** Continue until $N$ new tokens have been generated or an early stop triggers (end-of-sequence token, missing lookup entry).</span>

<span style="font-size: 14px;">For $N$ new tokens there are $N$ model calls. Each processes the entire sequence. In production, KV caching avoids recomputing attention for earlier positions, but the conceptual loop is the same.</span>

<span style="font-size: 14px;">Order matters: append the token **before** the next model call, because the model needs the full context including the just-generated token to produce correct logits for the next position.</span>

---

## <span style="font-size: 16px;">Why Greedy</span>

<span style="font-size: 14px;">Greedy decoding has properties that make it a natural baseline:</span>

* <span style="font-size: 14px;">**Simplicity:** One argmax per step, one append, one loop. No hyperparameters -- no temperature, top-k, or beam width.</span>
* <span style="font-size: 14px;">**Determinism:** Same prompt and model always produce the same output. Valuable for debugging and reproducibility.</span>
* <span style="font-size: 14px;">**Speed:** No overhead beyond the model call. Beam search maintains multiple hypotheses; sampling may sort logits. Greedy just takes argmax.</span>
* <span style="font-size: 14px;">**No sampling noise:** Every token is the model's single best prediction. No risk of a low-probability token derailing the sequence.</span>

<span style="font-size: 14px;">For tasks with one correct answer (arithmetic, code completion, factual QA), greedy works well because the highest-probability token is usually the right one.</span>

---

## <span style="font-size: 16px;">Greedy vs Sampling</span>

<span style="font-size: 14px;">The fundamental tradeoff is **quality vs diversity**.</span>

<span style="font-size: 14px;">**Greedy decoding** always picks the top token:</span>

* <span style="font-size: 14px;">Deterministic -- run it twice, get the same result.</span>
* <span style="font-size: 14px;">Tends toward safe, repetitive phrasing. If "the" is always the most probable next word, you get "the" every time.</span>
* <span style="font-size: 14px;">Can get stuck in loops: "I think that I think that I think that..." repeats if each "I think that" makes the model predict the same continuation.</span>

<span style="font-size: 14px;">**Sampling** draws from the probability distribution:</span>

$$
\texttt{next\_token} \sim P(\cdot \mid \text{context})
$$

<span style="font-size: 14px;">This introduces randomness. Text is more varied but occasionally incoherent. Filtering strategies constrain the sampling:</span>

* <span style="font-size: 14px;">**Temperature scaling:** Divide logits by $\tau$ before softmax. $\tau < 1$ sharpens (closer to greedy), $\tau > 1$ flattens (more random). At $\tau \to 0$, sampling becomes greedy.</span>
* <span style="font-size: 14px;">**Top-k sampling:** Zero out all but the $k$ highest logits, then sample. Prevents drawing extremely unlikely tokens.</span>
* <span style="font-size: 14px;">**Nucleus (top-p) sampling:** Keep the smallest set of tokens whose cumulative probability exceeds $p$. Adapts dynamically -- fewer candidates when the model is confident, more when uncertain.</span>

<span style="font-size: 14px;">Greedy is the $\tau \to 0$ limit of temperature scaling, equivalently top-k with $k = 1$ or top-p with $p \to 0$. All reduce to greedy at maximum determinism.</span>

---

## <span style="font-size: 16px;">Paper Context</span>

<span style="font-size: 14px;">GPT-2 was introduced in "Language Models are Unsupervised Multitask Learners" (Radford et al., 2019). The paper showed that a large language model trained on diverse internet text could perform many tasks without fine-tuning simply by conditioning on the right prompt.</span>

<span style="font-size: 14px;">For generation experiments, they used greedy decoding as the simplest baseline, noting it produces repetitive text on longer generations. Top-k sampling (with $k = 40$) produced more natural, varied outputs.</span>

<span style="font-size: 14px;">Greedy serves as the reference point: the strategy you try first, whose failure modes motivate better approaches. The paper's insight was that model capacity and data quality matter as much as decoding -- a large model with greedy can outperform a smaller model with sophisticated sampling.</span>

<span style="font-size: 14px;">The generation pipeline -- feed tokens, get logits, select, append, repeat -- is identical regardless of strategy. Only the "select" step changes. Master the loop with greedy, and every other strategy is a drop-in replacement for argmax.</span>

---

## <span style="font-size: 16px;">The KV Cache Concept</span>

<span style="font-size: 14px;">In the autoregressive loop, each model call processes the **entire** sequence. At step $i$, the model sees $k + i$ tokens. Naively, this means recomputing attention for all positions, even though the first $k + i - 1$ tokens have not changed since the last call.</span>

<span style="font-size: 14px;">The KV cache eliminates this redundancy. During attention, each token is projected into key and value vectors. For positions already processed, these are unchanged. The cache stores them so only the new token's K and V need to be computed. The new token's query still attends to all cached keys and values.</span>

<span style="font-size: 14px;">**Without KV cache:** compute K and V for all $k + i$ tokens at step $i$. Total across $N$ steps: $O(N \cdot (k + N) \cdot d)$ for projections, plus quadratic attention.</span>

<span style="font-size: 14px;">**With KV cache:** compute K and V for only the new token at each step. Total: $O(N \cdot d)$.</span>

<span style="font-size: 14px;">The cache does not change the output -- it is a pure optimization. This problem uses a lookup table, so no cache is needed, but the concept is essential for real Transformer generation.</span>

<span style="font-size: 14px;">Memory cost grows linearly with sequence length and layers. For GPT-2 small (12 layers, 12 heads, $d_k = 64$), the KV cache for 1024 tokens requires $2 \times 12 \times 12 \times 1024 \times 64 \approx 18.9\text{M}$ floats -- why long-context generation demands significant GPU memory.</span>

---

## <span style="font-size: 16px;">Numerical Example</span>

<span style="font-size: 14px;">Trace greedy decoding for 3 steps with a vocabulary of 4 tokens: {0: "the", 1: "cat", 2: "sat", 3: "on"}. The prompt is $(1)$ (the token "cat").</span>

<span style="font-size: 14px;">**Step 1.** Current sequence: $(1)$. Look up logits:</span>

$$
\texttt{logits} = [0.2, \; -1.0, \; 2.5, \; 0.8]
$$

<span style="font-size: 14px;">Argmax: index $2$ (logit $2.5$). Append token $2$ ("sat"). Sequence: $(1, 2)$.</span>

<span style="font-size: 14px;">**Step 2.** Current sequence: $(1, 2)$. Look up logits:</span>

$$
\texttt{logits} = [0.1, \; -0.5, \; -2.0, \; 3.1]
$$

<span style="font-size: 14px;">Argmax: index $3$ (logit $3.1$). Append token $3$ ("on"). Sequence: $(1, 2, 3)$.</span>

<span style="font-size: 14px;">**Step 3.** Current sequence: $(1, 2, 3)$. Look up logits:</span>

$$
\texttt{logits} = [4.0, \; 0.3, \; -0.1, \; -1.2]
$$

<span style="font-size: 14px;">Argmax: index $0$ (logit $4.0$). Append token $0$ ("the"). Final sequence: $(1, 2, 3, 0)$ -- "cat sat on the".</span>

<span style="font-size: 14px;">**Key observations:**</span>

* <span style="font-size: 14px;">No softmax was needed. Argmax of logits and argmax of probabilities are identical.</span>
* <span style="font-size: 14px;">Each step uses the **full** accumulated sequence as input, not just the last token. The logits for $(1, 2)$ differ from those for just $(2)$ because context matters.</span>
* <span style="font-size: 14px;">The process is fully deterministic. Same prompt and table always produce the same output.</span>
* <span style="font-size: 14px;">If at step 2 the sequence $(1, 2)$ was not in the table, generation would stop early and return $(1, 2)$.</span>

---

## <span style="font-size: 16px;">When Greedy Fails</span>

<span style="font-size: 14px;">**Repetitive loops.** The most common failure. If the context makes the model predict a token leading back to a similar context, greedy gets trapped. A model that always predicts "dogs" after "I like" produces "I like dogs. I like dogs." forever. Greedy has no mechanism to break out because it does not track what it already generated.</span>

<span style="font-size: 14px;">**Missing globally optimal sequences.** Greedy optimizes locally, not globally. Suppose "Once upon a" has top tokens "time" (logit 5.0) and "dark" (logit 4.8). Greedy picks "time", but "dark and stormy night" might have much higher total sequence probability. The better path requires a suboptimal early choice that greedy never explores.</span>

<span style="font-size: 14px;">**Bland, generic text.** High-probability tokens are common words and safe continuations. Greedy produces grammatically correct but dull text. The model "knows" many plausible continuations but greedy always picks the safest one.</span>

<span style="font-size: 14px;">**Sensitivity to prompt.** With no randomness, small prompt changes cascade into completely different outputs. A single extra space changes the logit distribution, the argmax, and the entire generated sequence.</span>

---

## <span style="font-size: 16px;">Modern Context</span>

<span style="font-size: 14px;">Greedy decoding is rarely used for open-ended generation, but its alternatives are everywhere:</span>

* <span style="font-size: 14px;">**Beam search:** Maintains $B$ hypotheses at each step; only the $B$ best survive. Dominated machine translation but still produces repetitive open-ended text.</span>
* <span style="font-size: 14px;">**Top-k sampling (Fan et al., 2018):** Restricts sampling to the $k$ highest-probability tokens. GPT-2 used $k = 40$.</span>
* <span style="font-size: 14px;">**Nucleus / top-p (Holtzman et al., 2020):** Keeps the smallest token set whose cumulative probability exceeds $p$. Adapts to model confidence at each step.</span>
* <span style="font-size: 14px;">**Temperature:** Controls distribution sharpness. Low approaches greedy; high approaches uniform. Most systems combine temperature with top-p.</span>
* <span style="font-size: 14px;">**Repetition penalties:** Subtract from logits of tokens already in the sequence, directly addressing greedy's repetition problem.</span>
* <span style="font-size: 14px;">**Speculative decoding (Leviathan et al., 2023):** A small draft model generates tokens greedily, then the large model verifies in parallel. Speeds inference 2-3x without changing the output distribution.</span>

<span style="font-size: 14px;">Despite its limitations, greedy remains standard for code completion, structured output (JSON, SQL), and tasks with one correct answer. It is also the fastest method, the default when latency matters more than creativity.</span>

---

## <span style="font-size: 16px;">Pitfalls</span>

<span style="font-size: 14px;">Common implementation mistakes:</span>

* <span style="font-size: 14px;">**Not appending the generated token.** You must append the token before the next model call. Forgetting this means the model sees the same input every iteration and generates the same token $N$ times.</span>
* <span style="font-size: 14px;">**Wrong stop condition.** Stop after exactly $N$ new tokens, or earlier on an early-stop trigger. Off-by-one errors are common: generating $N+1$ or $N-1$ tokens, or counting from the prompt length.</span>
* <span style="font-size: 14px;">**Confusing logits with probabilities.** Logits can be any real number; probabilities are non-negative and sum to 1. Applying softmax before argmax wastes computation since softmax is monotonic.</span>
* <span style="font-size: 14px;">**Using the wrong logits position.** A real Transformer returns logits with shape $(B, T, V)$. For next-token prediction, use the **last** position: $\texttt{logits}[:, -1, :]$. An earlier position predicts a token for that earlier point in the sequence.</span>
* <span style="font-size: 14px;">**KV cache invalidation.** With a cache, pass only the new token on subsequent calls. Passing the full sequence while the cache is active creates duplicate key-value pairs. Use full sequence with no cache, or new token with cache -- never mix.</span>
* <span style="font-size: 14px;">**Mutating the input.** If the prompt is a list and you append in-place, the caller's original is modified. Copy the input before modifying.</span>
* <span style="font-size: 14px;">**Ignoring early stop.** If the current sequence is not in the lookup table, return what has been built so far. Do not crash with a KeyError or continue with stale logits.</span>