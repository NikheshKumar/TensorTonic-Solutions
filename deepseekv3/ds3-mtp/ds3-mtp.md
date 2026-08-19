# <span style="font-size: 20px;">Multi-Token Prediction Head</span>

<span style="font-size: 14px;">Standard language models predict one token at a time: given hidden state $h$, compute $\text{logits} = h \cdot W_{\text{head}}^T$ for only the next token. Multi-Token Prediction (MTP) predicts $D$ future tokens simultaneously. Each depth $d$ has its own projection $W_{\text{proj}}^{(d)}$ that transforms $h$ into a depth-specific representation, which passes through a shared LM head $W_{\text{head}}$ to produce logits. Results stack into shape $(B, S, D, V)$: depth 0 predicts the next token, depth 1 predicts two ahead, and so on.</span>

---

## <span style="font-size: 16px;">What It Is</span>

<span style="font-size: 14px;">In a standard autoregressive model, each forward pass at position $t$ predicts exactly one token: position $t+1$. The final hidden state passes through a single linear layer (the LM head) to produce logits over the vocabulary. MTP changes this by predicting $D$ future tokens at once: positions $t+1, t+2, \ldots, t+D$.</span>

<span style="font-size: 14px;">Instead of one projection path, MTP introduces $D$ parallel paths. Each depth $d \in \{0, \ldots, D-1\}$ has its own projection matrix $W_{\text{proj}}^{(d)} \in \mathbb{R}^{d_{\text{model}} \times d_{\text{model}}}$ that transforms the hidden state into a depth-specific representation. All $D$ representations are independently mapped to vocabulary logits through the same shared $W_{\text{head}} \in \mathbb{R}^{V \times d_{\text{model}}}$.</span>

<span style="font-size: 14px;">Different future positions require different information from the hidden state. What matters for predicting the immediate next word differs from what matters for predicting two words ahead. The per-depth projections learn to extract the right features for each horizon, while the shared head ensures consistent vocabulary mapping across all depths.</span>

---

## <span style="font-size: 16px;">Key Equations</span>

<span style="font-size: 14px;">Let $h \in \mathbb{R}^{B \times S \times d_{\text{model}}}$ be the final hidden states from the Transformer backbone, $D$ the number of prediction depths, and $V$ the vocabulary size.</span>

### <span style="font-size: 14px;">Per-depth projection</span>

<span style="font-size: 14px;">For each depth $d \in \{0, 1, \ldots, D-1\}$:</span>

$$
h_d = h \cdot (W_{\text{proj}}^{(d)})^T
$$

<span style="font-size: 14px;">where $W_{\text{proj}}^{(d)} \in \mathbb{R}^{d_{\text{model}} \times d_{\text{model}}}$ is a learned matrix specific to depth $d$, producing $h_d \in \mathbb{R}^{B \times S \times d_{\text{model}}}$.</span>

### <span style="font-size: 14px;">Shared LM head</span>

$$
\text{logits}_d = h_d \cdot W_{\text{head}}^T
$$

<span style="font-size: 14px;">where $W_{\text{head}} \in \mathbb{R}^{V \times d_{\text{model}}}$ is the shared language model head, producing $\text{logits}_d \in \mathbb{R}^{B \times S \times V}$.</span>

### <span style="font-size: 14px;">Stacking</span>

$$
\text{logits} = \text{stack}(\text{logits}_0, \text{logits}_1, \ldots, \text{logits}_{D-1}, \text{dim}=2)
$$

<span style="font-size: 14px;">This produces shape $(B, S, D, V)$. At position $s$, the slice $\text{logits}[:, s, d, :]$ gives the distribution over tokens at position $s + d + 1$.</span>

### <span style="font-size: 14px;">Combined form</span>

$$
\text{logits}[:, :, d, :] = (h \cdot (W_{\text{proj}}^{(d)})^T) \cdot W_{\text{head}}^T \quad \text{for } d = 0, 1, \ldots, D-1
$$

<span style="font-size: 14px;">The total learnable parameters added by MTP: $D \times d_{\text{model}}^2$ (for the projection matrices). The LM head $W_{\text{head}}$ is not additional -- it is the same head already used for standard next-token prediction.</span>

---

## <span style="font-size: 16px;">Why Multi-Token Prediction</span>

### <span style="font-size: 14px;">Better representation learning during training</span>

<span style="font-size: 14px;">The standard next-token objective only requires the hidden state to be useful for one step ahead. MTP forces it to be useful for $D$ steps simultaneously, acting as an auxiliary training signal that enriches backbone representations. Even when the model cannot confidently predict three tokens ahead, the attempt encourages the backbone to encode more abstract, planning-oriented features rather than purely local patterns.</span>

### <span style="font-size: 14px;">Speculative decoding at inference</span>

<span style="font-size: 14px;">During standard autoregressive inference, each token requires a full forward pass. MTP enables speculative decoding: the deeper predictions (depths 1 through $D-1$) serve as draft tokens that can be verified in parallel. If the drafts match what the model would have generated through standard decoding, multiple tokens are accepted in a single step, yielding significant speedup without any loss in output quality.</span>

### <span style="font-size: 14px;">Connection to planning and lookahead</span>

<span style="font-size: 14px;">Language generation requires planning ahead -- the choice of the next word depends on where the sentence is going, not just what came before. MTP encourages hidden representations to encode future trajectory information, analogous to lookahead in search algorithms.</span>

---

## <span style="font-size: 16px;">The Architecture</span>

### <span style="font-size: 14px;">D separate projection matrices, one shared LM head</span>

<span style="font-size: 14px;">The MTP module sits on top of the Transformer backbone. The backbone produces hidden states $h \in \mathbb{R}^{B \times S \times d_{\text{model}}}$, and MTP applies $D$ independent linear projections to create $D$ different views. Each $W_{\text{proj}}^{(d)}$ is a square $d_{\text{model}} \times d_{\text{model}}$ matrix with its own learned parameters -- not shared across depths.</span>

### <span style="font-size: 14px;">Why share the LM head</span>

<span style="font-size: 14px;">All depths share a single $W_{\text{head}} \in \mathbb{R}^{V \times d_{\text{model}}}$:</span>

* <span style="font-size: 14px;">**Parameter efficiency:** The LM head is large ($V \times d_{\text{model}}$, where $V$ can be 100K+). Having $D$ separate heads multiplies this cost by $D$. Sharing keeps overhead at just $D \times d_{\text{model}}^2$.</span>
* <span style="font-size: 14px;">**Consistent vocabulary mapping:** A shared head ensures the same direction in embedding space means the same token regardless of depth.</span>
* <span style="font-size: 14px;">**Responsibility separation:** Projections handle "what to extract for this depth" while the head handles "how to map features to tokens."</span>

---

## <span style="font-size: 16px;">Training vs. Inference</span>

### <span style="font-size: 14px;">Training: all depths provide supervision</span>

<span style="font-size: 14px;">During training, each depth $d$ produces logits compared against the ground-truth token at position $t + d + 1$. The training loss is a weighted sum of cross-entropy losses:</span>

$$
\mathcal{L}_{\text{MTP}} = \sum_{d=0}^{D-1} \lambda_d \cdot \mathcal{L}_{\text{CE}}(\text{logits}_d, y_{t+d+1})
$$

<span style="font-size: 14px;">Typically $\lambda_0 = 1.0$ (the standard next-token loss) and deeper depths use smaller weights, ensuring primary prediction quality is not degraded. All $D$ depth predictions share the same backbone hidden states, so gradients from all depths flow back into the Transformer layers.</span>

### <span style="font-size: 14px;">Inference: depth 0 is standard, other depths enable speculation</span>

<span style="font-size: 14px;">At inference, depth 0 behaves identically to standard next-token prediction. You can ignore the other depths entirely and the model produces the same output as a standard autoregressive model. However, depths 1 through $D-1$ can be used for speculative decoding: generate draft tokens, verify them, and commit multiple tokens at once if they match, reducing forward passes by up to a factor of $D$.</span>

---

## <span style="font-size: 16px;">Paper Context: DeepSeek V3</span>

<span style="font-size: 14px;">DeepSeek V3 uses MTP as one of its key architectural innovations. The model employs $D = 1$ additional prediction depth beyond the standard next-token head during training, predicting both the next token and the token after that (2 total targets per position).</span>

<span style="font-size: 14px;">In DeepSeek V3, the MTP module is more sophisticated than a bare projection. Each depth uses a small additional Transformer layer on top of the backbone output before the shared LM head. The depth-$d$ module takes the backbone hidden state, combines it with the embedding of the token at position $t + d$ (the token the previous depth targeted), and processes this through a dedicated block before applying the shared head.</span>

<span style="font-size: 14px;">The MTP loss uses $\lambda = 0.3$ for the additional depth: $\mathcal{L} = \mathcal{L}_{\text{CE}}^{(0)} + 0.3 \cdot \mathcal{L}_{\text{CE}}^{(1)}$. DeepSeek V3 reports MTP improves both training efficiency and final model quality, with modest additional cost. At inference, MTP modules can be discarded (standard decoding) or used for speculative decoding.</span>

---

## <span style="font-size: 16px;">Speculative Decoding Connection</span>

### <span style="font-size: 14px;">Using deeper predictions as draft tokens</span>

<span style="font-size: 14px;">Speculative decoding generates candidate tokens cheaply and verifies them in parallel. Without MTP, this requires a separate draft model. With MTP, drafts come for free from deeper prediction depths. At position $t$, the MTP head produces predictions for $t+1$ (depth 0), $t+2$ (depth 1), ..., $t+D$ (depth $D-1$). Depth 0 is the primary prediction; the rest serve as speculative drafts.</span>

### <span style="font-size: 14px;">Verification with a standard forward pass</span>

<span style="font-size: 14px;">To verify drafts, the model runs a single forward pass on the sequence extended with all $D-1$ draft tokens. This produces depth-0 logits at each new position. If the depth-0 prediction at position $t+k$ matches the draft from depth $k-1$, the draft is accepted. Verification proceeds left to right; the first mismatch causes rejection of that token and all subsequent drafts.</span>

### <span style="font-size: 14px;">Speedup characteristics</span>

<span style="font-size: 14px;">Speedup depends on draft acceptance rate. For predictable text (code, formulaic language), deeper predictions are often correct, yielding near-$D$x speedup. For unpredictable text, acceptance drops, but worst-case overhead is just the cost of unused MTP projections -- small relative to the backbone. The key advantage over external draft models: MTP drafts come from the same model that verifies them, so the draft distribution naturally matches the target.</span>

---

## <span style="font-size: 16px;">Numerical Example</span>

<span style="font-size: 14px;">Trace through MTP with $d_{\text{model}} = 4$, vocabulary size $V = 6$, and $D = 3$ prediction depths, processing a single token.</span>

### <span style="font-size: 14px;">Setup</span>

$$
h = [0.5, \; -1.0, \; 0.8, \; 0.3]
$$

<span style="font-size: 14px;">Three projection matrices, each near-identity with one off-diagonal entry that mixes a different pair of dimensions:</span>

* <span style="font-size: 14px;">$W_{\text{proj}}^{(0)}$: identity plus 0.5 at row 0, col 2 (mixes dim 2 into dim 0)</span>
* <span style="font-size: 14px;">$W_{\text{proj}}^{(1)}$: identity plus 0.5 at row 1, col 3 (mixes dim 3 into dim 1)</span>
* <span style="font-size: 14px;">$W_{\text{proj}}^{(2)}$: identity plus 0.5 at row 2, col 0 (mixes dim 0 into dim 2)</span>

### <span style="font-size: 14px;">Depth 0 projection</span>

<span style="font-size: 14px;">$h_0 = h \cdot (W_{\text{proj}}^{(0)})^T$. The transpose places 0.5 at row 0, col 2, so dim 0 gains $0.5 \times h[2]$:</span>

$$
h_0 = [0.5 + 0.5 \times 0.8, \; -1.0, \; 0.8, \; 0.3] = [0.90, \; -1.00, \; 0.80, \; 0.30]
$$

### <span style="font-size: 14px;">Depth 1 projection</span>

<span style="font-size: 14px;">$h_1 = h \cdot (W_{\text{proj}}^{(1)})^T$. Dim 1 gains $0.5 \times h[3]$:</span>

$$
h_1 = [0.50, \; -1.0 + 0.5 \times 0.3, \; 0.80, \; 0.30] = [0.50, \; -0.85, \; 0.80, \; 0.30]
$$

### <span style="font-size: 14px;">Depth 2 projection</span>

<span style="font-size: 14px;">$h_2 = h \cdot (W_{\text{proj}}^{(2)})^T$. Dim 2 gains $0.5 \times h[0]$:</span>

$$
h_2 = [0.50, \; -1.00, \; 0.8 + 0.5 \times 0.5, \; 0.30] = [0.50, \; -1.00, \; 1.05, \; 0.30]
$$

### <span style="font-size: 14px;">Shared LM head</span>

<span style="font-size: 14px;">$W_{\text{head}} \in \mathbb{R}^{6 \times 4}$: rows 0-3 are one-hot (selecting individual dims), row 4 sums dims 0+1, row 5 sums dims 2+3.</span>

$$
\text{logits}_0 = [0.90, \; -1.00, \; 0.80, \; 0.30, \; -0.10, \; 1.10]
$$

$$
\text{logits}_1 = [0.50, \; -0.85, \; 0.80, \; 0.30, \; -0.35, \; 1.10]
$$

$$
\text{logits}_2 = [0.50, \; -1.00, \; 1.05, \; 0.30, \; -0.50, \; 1.35]
$$

### <span style="font-size: 14px;">Stacking and interpretation</span>

<span style="font-size: 14px;">Stack along dim=2 to get shape $(1, 1, 3, 6)$. Taking argmax per depth:</span>

* <span style="font-size: 14px;">**Depth 0** (next token): $[0.90, -1.00, 0.80, 0.30, -0.10, 1.10]$ -- argmax = token 5</span>
* <span style="font-size: 14px;">**Depth 1** (2 ahead): $[0.50, -0.85, 0.80, 0.30, -0.35, 1.10]$ -- argmax = token 5</span>
* <span style="font-size: 14px;">**Depth 2** (3 ahead): $[0.50, -1.00, 1.05, 0.30, -0.50, 1.35]$ -- argmax = token 5</span>

<span style="font-size: 14px;">All depths predict token 5, but distributions differ. Depth 0 has 0.90 for token 0 as runner-up; depth 2 has 1.05 for token 2. The projections shifted $h$ differently, causing different relative rankings even though the same LM head was applied.</span>

---

## <span style="font-size: 16px;">Pitfalls</span>

### <span style="font-size: 14px;">1. Not sharing the LM head across depths</span>

<span style="font-size: 14px;">Creating $D$ separate LM heads dramatically increases parameter count (each head is $V \times d_{\text{model}}$) and breaks consistent vocabulary mapping across depths. The projection matrices handle depth-specific transformations; the head provides a single shared mapping. Separate heads also make speculative decoding less effective due to inconsistent distributions.</span>

### <span style="font-size: 14px;">2. Wrong stacking dimension</span>

<span style="font-size: 14px;">The output must be $(B, S, D, V)$ -- batch, sequence, depth, vocabulary. A frequent error is stacking along the wrong axis, producing $(B, D, S, V)$ or $(B, S, V, D)$. This swaps depth with sequence or vocabulary dimensions, causing the loss to compare logits against wrong target tokens. Depth must be dimension 2 (0-indexed), between sequence and vocabulary.</span>

### <span style="font-size: 14px;">3. Confusing depth index with sequence position</span>

<span style="font-size: 14px;">Depth $d$ at sequence position $s$ predicts the token at position $s + d + 1$. A common mistake is assuming depth $d$ predicts position $s + d$ (off by one). The target for $\text{logits}[:, s, d, :]$ is the token at position $s + d + 1$. Getting this wrong means every depth trains against the wrong target, and depth-0 loss no longer matches the standard next-token loss.</span>

### <span style="font-size: 14px;">4. MTP loss weighting</span>

<span style="font-size: 14px;">Loss weights $\lambda_d$ for deeper predictions should be smaller than $\lambda_0 = 1.0$. Equal weights overemphasize harder, noisier deep predictions and degrade primary next-token accuracy. DeepSeek V3 uses $\lambda_1 = 0.3$. Too-large weights pull backbone representations toward long-range speculative features at the expense of precise next-token features. Zero weights for $d > 0$ mean projection matrices learn nothing useful.</span>

### <span style="font-size: 14px;">5. Applying MTP projections before the final backbone layer</span>

<span style="font-size: 14px;">MTP projections must be applied to the final hidden states, after all Transformer layers and any final normalization. Applying them to intermediate outputs means projections operate on features still being refined, reducing prediction quality for all depths.</span>

### <span style="font-size: 14px;">6. Forgetting to handle sequence boundaries</span>

<span style="font-size: 14px;">At the end of a sequence, deeper predictions extend beyond available ground-truth tokens. For length $S$, depth $d$ at position $s$ targets $s + d + 1$, exceeding $S$ when $s > S - d - 2$. These positions must be masked out of the loss. Failing to mask causes index-out-of-bounds errors or silently wraps around to wrong targets.</span>