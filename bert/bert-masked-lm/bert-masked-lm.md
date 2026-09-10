# <span style="font-size: 20px;">Masked Language Modeling</span>

<span style="font-size: 14px;">Masked Language Modeling (MLM) is the primary pre-training objective of BERT (Devlin et al., 2019). Rather than predicting tokens left-to-right like autoregressive models, MLM randomly masks a subset of input tokens and trains the model to reconstruct them using bidirectional context. This is what allows BERT to attend to both left and right context simultaneously, producing deeply bidirectional representations.</span>

<span style="font-size: 14px;">MLM transforms pre-training into a fill-in-the-blank task. 15% of tokens are selected, corrupted via an 80-10-10 strategy, and the model must recover the original token at each corrupted position. Loss is computed only on masked positions.</span>

---

## <span style="font-size: 16px;">What It Is / What It Does</span>

<span style="font-size: 14px;">MLM randomly selects tokens from the input sequence and replaces them with corrupted versions. The model must predict the original token at each selected position using the surrounding context. Because no causal mask restricts attention, the model can use tokens from both the left and right to make its prediction.</span>

<span style="font-size: 14px;">This is fundamentally different from autoregressive models like GPT, where position $t$ can only encode information from positions $1, 2, \ldots, t-1$. In MLM, masked positions draw from all unmasked positions, both before and after.</span>

<span style="font-size: 14px;">Key properties:</span>

* <span style="font-size: 14px;">**Bidirectional context:** Every token attends to every other token (no causal mask), so masked positions are predicted from full surrounding context.</span>
* <span style="font-size: 14px;">**Partial prediction:** Only 15% of tokens are predicted per forward pass, making each training step less efficient than autoregressive models (which predict all tokens), but producing richer representations per token.</span>
* <span style="font-size: 14px;">**Corruption strategy:** Selected tokens are not simply replaced with a [MASK] symbol. The 80-10-10 scheme is used to reduce the gap between pre-training (which uses [MASK]) and fine-tuning (which never sees [MASK]).</span>

---

## <span style="font-size: 16px;">Key Equations</span>

<span style="font-size: 14px;">Let $\mathbf{x} = (x_1, x_2, \ldots, x_n)$ be the input token sequence of length $n$. Let $\mathcal{M} \subset \{1, 2, \ldots, n\}$ be the set of positions selected for masking, where $|\mathcal{M}| \approx 0.15n$.</span>

<span style="font-size: 14px;">**The 80-10-10 masking strategy.** For each position $i \in \mathcal{M}$, the corrupted token $\tilde{x}_i$ is determined as:</span>

$$
\tilde{x}_i = \begin{cases} \texttt{[MASK]} & \text{with probability } 0.80 \\ x_r \sim \text{Uniform}(\mathcal{V}) & \text{with probability } 0.10 \\ x_i & \text{with probability } 0.10 \end{cases}
$$

<span style="font-size: 14px;">where $\mathcal{V}$ is the full vocabulary and $x_r$ is a token sampled uniformly at random from $\mathcal{V}$.</span>

<span style="font-size: 14px;">**The MLM prediction head.** At each masked position $i \in \mathcal{M}$, the model takes the final hidden state $\mathbf{h}_i \in \mathbb{R}^{H}$ and projects it to vocabulary-sized logits:</span>

$$
\mathbf{z}_i = \mathbf{h}_i \mathbf{W} + \mathbf{b}
$$

<span style="font-size: 14px;">where $\mathbf{W} \in \mathbb{R}^{H \times V}$ is the projection weight matrix, $\mathbf{b} \in \mathbb{R}^{V}$ is the bias vector, $H$ is the hidden size, and $V$ is the vocabulary size.</span>

<span style="font-size: 14px;">**The MLM loss.** Cross-entropy is computed only at masked positions:</span>

$$
\mathcal{L}_{\text{MLM}} = -\frac{1}{|\mathcal{M}|} \sum_{i \in \mathcal{M}} \log \frac{\exp(\mathbf{z}_i[x_i])}{\sum_{v=1}^{V} \exp(\mathbf{z}_i[v])}
$$

<span style="font-size: 14px;">where $\mathbf{z}_i[x_i]$ is the logit corresponding to the true token $x_i$ at position $i$, and the denominator is the softmax normalizer over the full vocabulary.</span>

<span style="font-size: 14px;">Positions not in $\mathcal{M}$ are assigned a label of $-100$, which is the standard PyTorch convention for `ignore_index` in `CrossEntropyLoss`. These positions contribute zero gradient and are excluded from the loss computation entirely.</span>

---

## <span style="font-size: 16px;">Why 80-10-10 (Not 100% [MASK])</span>

<span style="font-size: 14px;">A naive approach would replace every selected token with the [MASK] symbol. This creates a train-test mismatch: during pre-training, the model sees [MASK] tokens everywhere, but during fine-tuning and inference, [MASK] never appears in the input. The model's representations would be optimized for inputs containing [MASK] and degrade on real text.</span>

<span style="font-size: 14px;">The 80-10-10 strategy mitigates this mismatch in three ways:</span>

* <span style="font-size: 14px;">**80% [MASK]:** Gives the model a clear signal that a position needs to be predicted. Without [MASK], the model would not know which positions require reconstruction.</span>
* <span style="font-size: 14px;">**10% random token:** Forces the model to maintain good representations at every position. The model cannot assume a non-[MASK] token is correct -- it must verify consistency against context, adding robustness.</span>
* <span style="font-size: 14px;">**10% unchanged:** Teaches the model that even correct-looking tokens might be prediction targets. This biases representations toward encoding the observed token, which helps at fine-tuning time when [MASK] never appears.</span>

<span style="font-size: 14px;">Devlin et al. chose the 80-10-10 split empirically. Ablations showed the exact ratio matters less than having all three components present. Using 100% [MASK] performed measurably worse on downstream tasks.</span>

---

## <span style="font-size: 16px;">Why 15% Masking Rate</span>

<span style="font-size: 14px;">The masking rate controls a fundamental tradeoff between training efficiency and prediction quality:</span>

* <span style="font-size: 14px;">**Too low (e.g., 5%):** Very few tokens are predicted per forward pass, meaning each training step provides minimal gradient signal. The model would need many more steps to learn the same representations, making pre-training prohibitively slow and expensive.</span>
* <span style="font-size: 14px;">**Too high (e.g., 50%):** Too many tokens are masked, destroying most of the context. The model cannot make meaningful predictions because there is not enough surrounding information to reconstruct the masked tokens. The task becomes closer to random guessing than language understanding.</span>

<span style="font-size: 14px;">At 15%, for a typical BERT sequence of 512 tokens, roughly 77 tokens are selected for prediction per sequence. This provides substantial gradient signal per step while preserving 85% of the context for the model to condition on.</span>

<span style="font-size: 14px;">The 15% rate was determined empirically. MLM models converge slower than autoregressive models (which predict 100% of tokens) because only 15% produce loss. This is the fundamental efficiency cost of bidirectional pre-training: richer representations, but fewer prediction targets per step. Later work found slightly higher rates (20-25%) can work for large models, but 15% remains the standard baseline.</span>

---

## <span style="font-size: 16px;">Paper Context</span>

<span style="font-size: 14px;">The MLM objective was introduced in "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding" (Devlin, Chang, Lee, and Toutanova, 2019). The core claim is that deeply bidirectional representations outperform both left-to-right (GPT) and shallow bidirectional (ELMo) approaches.</span>

<span style="font-size: 14px;">The paper contrasts MLM with GPT's autoregressive objective, where a causal mask restricts position $t$ to attend only to positions $1$ through $t-1$. BERT removes this mask, allowing full bidirectional attention. The tradeoff is that BERT cannot be used directly for text generation.</span>

<span style="font-size: 14px;">Devlin et al. borrowed the MLM concept from the Cloze task (Taylor, 1953). The paper states: "We simply mask some percentage of the input tokens at random, and then predict those masked tokens." This produced representations that set new state-of-the-art across 11 NLP benchmarks.</span>

<span style="font-size: 14px;">The $-100$ label convention is central to implementation. Every position gets a label: masked positions receive the original token ID, unmasked positions receive $-100$. PyTorch's `CrossEntropyLoss` with `ignore_index=-100` skips these positions, contributing zero gradient. This convention is now standard across all MLM implementations.</span>

<span style="font-size: 14px;">BERT was pre-trained on BooksCorpus (800M words) and English Wikipedia (2500M words) for 1M steps with batch size 256. The MLM objective was combined with Next Sentence Prediction (NSP), though later work (RoBERTa) showed NSP to be unnecessary.</span>

---

## <span style="font-size: 16px;">The Prediction Head</span>

<span style="font-size: 14px;">The MLM prediction head maps hidden states to vocabulary logits. In BERT's full implementation, it applies a dense layer ($H \to H$), GELU activation, layer normalization, then a vocabulary projection ($H \to V$). In simplified form:</span>

$$
\mathbf{z}_i = \mathbf{h}_i \mathbf{W} + \mathbf{b}
$$

<span style="font-size: 14px;">where $\mathbf{W} \in \mathbb{R}^{H \times V}$ and $\mathbf{b} \in \mathbb{R}^{V}$.</span>

<span style="font-size: 14px;">**Weight tying.** A common optimization ties $\mathbf{W}$ with the input embedding matrix $\mathbf{E} \in \mathbb{R}^{V \times H}$, setting $\mathbf{W} = \mathbf{E}^\top$. This saves $V \times H$ parameters (roughly 23M for BERT-base) and enforces consistency: tokens with similar embeddings produce similar logits. BERT uses weight tying.</span>

<span style="font-size: 14px;">**Training vs. inference.** During pre-training, the prediction head is applied only at masked positions. During fine-tuning, it is discarded -- only the transformer encoder is kept and a new task-specific head is attached.</span>

---

## <span style="font-size: 16px;">Numerical Example</span>

<span style="font-size: 14px;">Consider a sequence of 10 tokens (using token IDs for clarity):</span>

* <span style="font-size: 14px;">**Input token IDs:** $[101, 2023, 2003, 1037, 6251, 2005, 9543, 4083, 1012, 102]$</span>
* <span style="font-size: 14px;">**Readable form:** [CLS] this is a sentence for masked language . [SEP]</span>

<span style="font-size: 14px;">**Step 1: Select 15% of tokens for masking.**</span>

<span style="font-size: 14px;">$0.15 \times 10 = 1.5$, rounded to 2 tokens. Suppose positions 4 and 6 are selected (0-indexed). Position 4 has token ID 6251 ("sentence") and position 6 has token ID 9543 ("masked").</span>

<span style="font-size: 14px;">Note: positions 0 and 9 are [CLS] and [SEP] -- special tokens are typically excluded from masking candidates, so the actual candidate pool is positions 1-8.</span>

<span style="font-size: 14px;">**Step 2: Apply 80-10-10 for each selected position.**</span>

<span style="font-size: 14px;">For position 4 (token 6251, "sentence"):</span>

* <span style="font-size: 14px;">**Draw:** random value 0.35 -- falls in the 80% range (0.0 to 0.8).</span>
* <span style="font-size: 14px;">**Action:** Replace with [MASK] (token ID 103).</span>

<span style="font-size: 14px;">For position 6 (token 9543, "masked"):</span>

* <span style="font-size: 14px;">**Draw:** random value 0.87 -- falls in the 10% random range (0.8 to 0.9).</span>
* <span style="font-size: 14px;">**Action:** Replace with a random token from the vocabulary. Suppose the random token is 7592 ("playing").</span>

<span style="font-size: 14px;">**Step 3: Construct the three output arrays.**</span>

<span style="font-size: 14px;">**masked_ids** (the corrupted input fed to the model):</span>

$$
[101, 2023, 2003, 1037, \mathbf{103}, 2005, \mathbf{7592}, 4083, 1012, 102]
$$

<span style="font-size: 14px;">Position 4 changed from 6251 to 103 ([MASK]). Position 6 changed from 9543 to 7592 (random token).</span>

<span style="font-size: 14px;">**labels** (targets for loss computation):</span>

$$
[-100, -100, -100, -100, \mathbf{6251}, -100, \mathbf{9543}, -100, -100, -100]
$$

<span style="font-size: 14px;">Only positions 4 and 6 have real labels (the original token IDs). All other positions are $-100$, meaning they are ignored in the cross-entropy loss.</span>

<span style="font-size: 14px;">**mask** (boolean array indicating which positions are prediction targets):</span>

$$
[\text{False}, \text{False}, \text{False}, \text{False}, \mathbf{\text{True}}, \text{False}, \mathbf{\text{True}}, \text{False}, \text{False}, \text{False}]
$$

<span style="font-size: 14px;">**Step 4: Forward pass and loss.**</span>

<span style="font-size: 14px;">The model processes masked_ids through the encoder. At positions 4 and 6, the prediction head computes logits $\mathbf{z}_4$ and $\mathbf{z}_6$. Cross-entropy loss is computed against labels 6251 and 9543 respectively. The MLM loss is the average of these two.</span>

---

## <span style="font-size: 16px;">Modern Context</span>

<span style="font-size: 14px;">MLM was the dominant pre-training objective from 2018 to roughly 2020, but the landscape has shifted significantly since then:</span>

* <span style="font-size: 14px;">**GPT-style next-token prediction:** Autoregressive pre-training scales more effectively because every token produces a loss signal (not just 15%). This approach now dominates large-scale pre-training.</span>
* <span style="font-size: 14px;">**SpanBERT (Joshi et al., 2020):** Masks contiguous spans instead of random tokens, forcing the model to reason about longer-range dependencies. Outperforms BERT on span extraction tasks like question answering.</span>
* <span style="font-size: 14px;">**ELECTRA (Clark et al., 2020):** Replaces MLM with replaced token detection. A small generator produces plausible replacements, and the discriminator classifies every token as real or replaced, receiving gradient at all positions. Significantly more sample-efficient than BERT.</span>
* <span style="font-size: 14px;">**T5 (Raffel et al., 2020):** Uses span corruption with sentinel tokens and an encoder-decoder architecture, combining bidirectional encoding with autoregressive span generation.</span>
* <span style="font-size: 14px;">**RoBERTa (Liu et al., 2019):** Keeps MLM but applies dynamic masking (regenerated each epoch) rather than fixed preprocessing-time masks, improving performance with no architectural changes.</span>

<span style="font-size: 14px;">Despite these advances, MLM remains foundational for understanding bidirectional pre-training and the tradeoffs relative to autoregressive models.</span>

---

## <span style="font-size: 16px;">Common Pitfalls</span>

* <span style="font-size: 14px;">**Computing loss on all positions instead of only masked ones.** If you compute cross-entropy over the entire sequence, unmasked positions dominate the loss and drown out the actual learning signal. Always use `ignore_index=-100` or explicitly index into masked positions.</span>
* <span style="font-size: 14px;">**Wrong masking rate.** Too low means slow convergence; too high destroys context. Stick to 15% unless you have ablation results justifying a different rate.</span>
* <span style="font-size: 14px;">**Forgetting the 10% random and 10% unchanged.** Replacing all selected tokens with [MASK] produces worse downstream performance due to train-test mismatch. Always implement the full 80-10-10 strategy.</span>
* <span style="font-size: 14px;">**Using [MASK] at inference or fine-tuning.** The [MASK] token exists only for pre-training. Including it in fine-tuning inputs causes unpredictable behavior.</span>
* <span style="font-size: 14px;">**Label -100 not handled properly.** If your loss function does not support `ignore_index`, the model tries to predict a nonexistent token, causing numerical errors or crashes.</span>
* <span style="font-size: 14px;">**Masking special tokens.** [CLS], [SEP], and padding tokens should never be masked. Exclude them from the candidate pool before selecting the 15%.</span>
* <span style="font-size: 14px;">**Static vs. dynamic masking.** Original BERT used static masking (fixed during preprocessing). RoBERTa showed dynamic masking (regenerated each epoch) improves performance.</span>

---