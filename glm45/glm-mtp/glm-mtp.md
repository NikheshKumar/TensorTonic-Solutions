# <span style="font-size: 20px;">Multi-Token Prediction Head</span>

<span style="font-size: 14px;">The Multi-Token Prediction (MTP) head is a lightweight auxiliary module attached to a large language model that learns to predict the token at position $t+2$ given the main model's last hidden state at position $t$ and the embedding of the token at position $t+1$. GLM-4.5 adopts this design (originating in DeepSeek-V3) primarily to enable fast **speculative decoding** in production serving stacks such as vLLM and SGLang.</span>

---

## <span style="font-size: 16px;">Why MTP Exists</span>

<span style="font-size: 14px;">Autoregressive decoding is sequential: every token requires a full forward pass through the base model. For a 100B+ parameter MoE model this is the dominant cost in serving. Speculative decoding accelerates generation by having a small drafter propose several tokens at once, then verifying them in parallel with one base-model forward pass.</span>

<span style="font-size: 14px;">MTP supplies the drafter for free during training. The base model already produces a strong hidden state at each position. By tacking on a small head that consumes that hidden state plus the next token's embedding, we get a length-2 lookahead at a fraction of the cost of running a separate small model.</span>

* <span style="font-size: 14px;">**Speed:** verified tokens per base-model call jumps from 1 to roughly 1.5 to 2.0 in practice, depending on acceptance rate</span>
* <span style="font-size: 14px;">**Training signal:** the MTP loss acts as an auxiliary objective that pushes hidden states to encode information about more than the immediate next token</span>
* <span style="font-size: 14px;">**Cheap to run:** one extra RMSNorm pair, one projection, one transformer block, and a shared lm head</span>

---

## <span style="font-size: 16px;">Origin in DeepSeek-V3</span>

<span style="font-size: 14px;">The MTP head was introduced in the DeepSeek-V3 paper as a training auxiliary that predicts the second-next token at every position. DeepSeek-V3 reported that MTP not only enables speculative decoding but also yields a small main-task quality bump because the additional supervision densifies the gradient signal across positions.</span>

<span style="font-size: 14px;">GLM-4.5 inherits the architecture almost verbatim. The vLLM file `vllm/model_executor/models/glm4_moe_mtp.py` and the SGLang `glm4_moe.py` both contain a single MTP block per checkpoint, with these components in order:</span>

* <span style="font-size: 14px;">Two separate RMSNorms applied to the previous hidden state and to the embedding of the next token</span>
* <span style="font-size: 14px;">A projection that maps the concatenation back from $2H$ to $H$ dimensions</span>
* <span style="font-size: 14px;">A standard transformer block (attention + MoE FFN) for additional mixing</span>
* <span style="font-size: 14px;">A final RMSNorm before the shared lm head</span>

<span style="font-size: 14px;">This problem scopes down the implementation to the projection, the norms, and the lm head. The inner transformer block is omitted here because it is identical to a regular GLM-4.5 decoder block and is covered by block-level problems. Concentrating on the MTP-specific glue makes the algebra easy to read and the test cases tractable.</span>

---

## <span style="font-size: 16px;">Inputs and Outputs</span>

<span style="font-size: 14px;">The function takes seven tensors and an epsilon:</span>

* <span style="font-size: 14px;">$h_{prev}$ of shape $(n, H)$: the main model's last hidden state at positions $t_1, t_2, \ldots, t_n$</span>
* <span style="font-size: 14px;">$e_{next}$ of shape $(n, H)$: the embedding of the token at position $t_i + 1$ (i.e. the verified next token)</span>
* <span style="font-size: 14px;">$\gamma_h, \gamma_e$: RMSNorm scales applied to $h_{prev}$ and $e_{next}$ respectively, each of shape $(H,)$</span>
* <span style="font-size: 14px;">$W_{proj}$ of shape $(2H, H)$: down-projects the concatenation back to the model dimension</span>
* <span style="font-size: 14px;">$\gamma_{final}$ of shape $(H,)$: RMSNorm scale before the lm head</span>
* <span style="font-size: 14px;">$W_{lm}$ of shape $(H, V)$: the lm head, typically tied to the input embedding matrix</span>
* <span style="font-size: 14px;">$\epsilon$ for numerical stability inside RMSNorm</span>

<span style="font-size: 14px;">The output is a logit tensor of shape $(n, V)$ predicting the token at position $t_i + 2$ for each $i$.</span>

---

## <span style="font-size: 16px;">Key Equations</span>

<span style="font-size: 14px;">RMSNorm without the bias term, as used throughout LLaMA, DeepSeek-V3, and GLM-4.5:</span>

$$
\text{RMS}(x, \gamma) = \gamma \odot \frac{x}{\sqrt{\frac{1}{H}\sum_{i=1}^{H} x_i^2 + \epsilon}}
$$

<span style="font-size: 14px;">The MTP head, top to bottom:</span>

$$
\tilde h = \text{RMS}(h_{prev}, \gamma_h), \qquad \tilde e = \text{RMS}(e_{next}, \gamma_e)
$$

$$
z = [\tilde h \,\Vert\, \tilde e] \cdot W_{proj} \in \mathbb{R}^{n \times H}
$$

$$
\hat z = \text{RMS}(z, \gamma_{final})
$$

$$
\text{logits} = \hat z \cdot W_{lm} \in \mathbb{R}^{n \times V}
$$

<span style="font-size: 14px;">where $\Vert$ denotes concatenation along the feature dimension.</span>

---

## <span style="font-size: 16px;">Why Two Separate Norms</span>

<span style="font-size: 14px;">A natural first instinct is to concatenate $h_{prev}$ and $e_{next}$ first and then normalize the joint vector. GLM-4.5 (and DeepSeek-V3) explicitly does not do this. There are two reasons:</span>

* <span style="font-size: 14px;">**Different scales.** $h_{prev}$ comes out of a deep transformer stack and tends to have a large, paper-trained-distribution magnitude. $e_{next}$ comes from the embedding table and lives on a different scale. Concatenating before norming would let one signal dominate the joint statistics</span>
* <span style="font-size: 14px;">**Independent learned gains.** Two separate $\gamma$ vectors let the network rescale each stream on its own terms. This is the same reason cross-attention uses independent norms for queries and keys/values in some architectures</span>

<span style="font-size: 14px;">After normalization, both streams sit on roughly the same scale and the projection $W_{proj}$ is free to learn a useful mixture without first fighting a magnitude imbalance.</span>

---

## <span style="font-size: 16px;">Why Project from 2H back to H</span>

<span style="font-size: 14px;">The projection $W_{proj} \in \mathbb{R}^{2H \times H}$ is the single learned mixer of the two streams. After projection the tensor lives in the same shape as a standard hidden state, which lets the downstream transformer block (omitted in this problem) be a vanilla GLM-4.5 decoder block with no architectural surgery.</span>

<span style="font-size: 14px;">Concretely the projection can be read as a block-row matrix:</span>

$$
W_{proj} = \begin{pmatrix} W_h \\ W_e \end{pmatrix}, \quad W_h, W_e \in \mathbb{R}^{H \times H}
$$

<span style="font-size: 14px;">so that</span>

$$
z = \tilde h \, W_h + \tilde e \, W_e
$$

<span style="font-size: 14px;">i.e. a learned weighted sum of two normed streams. The factored form (concatenation then a single matmul) is more efficient on accelerator kernels than two separate matmuls plus an add.</span>

---

## <span style="font-size: 16px;">Training Objective</span>

<span style="font-size: 14px;">During training, for each position $t$ in the sequence the MTP head receives:</span>

* <span style="font-size: 14px;">$h_t$, the base model's last hidden state at $t$</span>
* <span style="font-size: 14px;">$e_{t+1}$, the embedding of the token actually present at $t+1$ in the training sequence</span>

<span style="font-size: 14px;">and is asked to predict the token at $t+2$. The loss is standard token-level cross-entropy:</span>

$$
\mathcal{L}_{MTP} = -\frac{1}{N} \sum_{t=1}^{N-2} \log p_\theta(x_{t+2} \mid h_t, e_{t+1})
$$

<span style="font-size: 14px;">GLM-4.5 trains this jointly with the standard next-token cross-entropy, with a small weight (often around $0.1$ to $0.3$) on the MTP loss so that it does not overwhelm the primary objective.</span>

---

## <span style="font-size: 16px;">Speculative Decoding at Inference</span>

<span style="font-size: 14px;">At inference the head plays a different role. The vLLM and SGLang implementations both run the MTP head once after the main forward pass to obtain a draft for position $t+2$. The high-level loop:</span>

* <span style="font-size: 14px;">Run the base model on the current prefix, take the last hidden state $h_{last}$</span>
* <span style="font-size: 14px;">Greedy-sample the next token from the base logits, call it $y_{next}$</span>
* <span style="font-size: 14px;">Look up the embedding $e(y_{next})$ and feed $(h_{last}, e(y_{next}))$ into the MTP head</span>
* <span style="font-size: 14px;">Sample $y_{next+1}$ from the MTP logits as the drafted token</span>
* <span style="font-size: 14px;">In the next base-model step, verify $y_{next+1}$ by checking it against the base model's actual distribution. Accept if it matches the sample under a rejection-sampling rule; otherwise correct</span>

<span style="font-size: 14px;">When acceptance succeeds the model emits two tokens per base-model forward pass instead of one, roughly doubling decode throughput in compute-bound serving.</span>

---

## <span style="font-size: 16px;">Numerical Example</span>

<span style="font-size: 14px;">Let $n = 1$, $H = 2$, $V = 3$, $\epsilon = 10^{-5}$. Take:</span>

* <span style="font-size: 14px;">$h_{prev} = [1.0, 2.0]$</span>
* <span style="font-size: 14px;">$e_{next} = [0.5, 0.0]$</span>
* <span style="font-size: 14px;">$\gamma_h = \gamma_e = \gamma_{final} = [1, 1]$</span>
* <span style="font-size: 14px;">$W_{proj} = I_4 \in \mathbb{R}^{4 \times 2}$ truncated: take $W_{proj}[:,0] = [1, 0, 1, 0]$, $W_{proj}[:,1] = [0, 1, 0, 1]$</span>
* <span style="font-size: 14px;">$W_{lm} = \begin{pmatrix} 1 & 0 & 1 \\ 0 & 1 & 1 \end{pmatrix}$</span>

<span style="font-size: 14px;">Step 1: RMS of $h_{prev}$: $\sqrt{(1 + 4)/2} = \sqrt{2.5} \approx 1.5811$. So $\tilde h \approx [0.6325, 1.2649]$.</span>

<span style="font-size: 14px;">Step 2: RMS of $e_{next}$: $\sqrt{(0.25 + 0)/2} \approx 0.3536$. So $\tilde e \approx [1.4142, 0]$.</span>

<span style="font-size: 14px;">Step 3: concat to $[0.6325, 1.2649, 1.4142, 0]$, project: $z_0 = 0.6325 + 1.4142 = 2.0467$, $z_1 = 1.2649 + 0 = 1.2649$. So $z \approx [2.0467, 1.2649]$.</span>

<span style="font-size: 14px;">Step 4: RMS of $z$: $\sqrt{(4.189 + 1.600)/2} \approx 1.7012$. So $\hat z \approx [1.2031, 0.7435]$.</span>

<span style="font-size: 14px;">Step 5: logits $= \hat z \cdot W_{lm} = [1.2031, 0.7435, 1.9465]$.</span>

<span style="font-size: 14px;">The largest logit is the third class, so the head predicts token id $2$ as the speculation for position $t+2$.</span>

---

## <span style="font-size: 16px;">Comparison with Plain Next-Token Prediction</span>

* <span style="font-size: 14px;">**Standard lm head:** $\hat z = \text{RMS}(h_{prev})$, $\text{logits} = \hat z W_{lm}$. Predicts only the token at $t+1$</span>
* <span style="font-size: 14px;">**MTP head:** same shape, but the input is a learned mixture of $h_{prev}$ and the actual $t+1$ embedding, predicting the token at $t+2$</span>

<span style="font-size: 14px;">The extra information channel (the verified next token's embedding) is what makes the MTP prediction reliable. Without it, predicting $t+2$ from $h_{prev}$ alone collapses to "marginalize over the unknown next token", which is much harder.</span>

---

## <span style="font-size: 16px;">Connection to Other Drafting Strategies</span>

<span style="font-size: 14px;">MTP is one of several drafting strategies for speculative decoding. It is useful to place it in the broader landscape:</span>

* <span style="font-size: 14px;">**Separate draft model:** train a small auxiliary LM (e.g. a 1B-parameter model alongside a 70B target). Strongest accept rates, highest memory and training cost. Examples: medusa-style draft trees, EAGLE</span>
* <span style="font-size: 14px;">**Self-speculation:** reuse a subset of the target model's layers as a faster drafter (e.g. early-exit decoding). Cheap but the drafter and target share representations, capping the gain</span>
* <span style="font-size: 14px;">**MTP head:** a single transformer block conditioned on the target's last hidden state plus the verified next-token embedding. Cheaper than a separate draft model and uses the target's own representations as ground truth</span>

<span style="font-size: 14px;">GLM-4.5 picks MTP because it shares the embedding table and the lm head with the main model, which keeps the extra parameter count under one percent of the total, while still giving accept rates competitive with much larger drafters in production.</span>

---

## <span style="font-size: 16px;">GLM-4.5 Specifics</span>

<span style="font-size: 14px;">A few details that matter for matching the reference checkpoint:</span>

* <span style="font-size: 14px;">**Weight tying.** $W_{lm}$ is shared with the input embedding table, just like in the main model. This keeps the parameter count of the MTP module small (effectively one projection plus three RMSNorm gains plus a transformer block)</span>
* <span style="font-size: 14px;">**RMSNorm, not LayerNorm.** Consistent with the rest of the GLM-4.5 stack. No mean subtraction, no learned bias</span>
* <span style="font-size: 14px;">**Single MTP layer.** Unlike some variants that stack two MTP heads to predict $t+2$ and $t+3$, GLM-4.5 ships exactly one head and predicts a single extra token</span>
* <span style="font-size: 14px;">**Inner transformer block is MoE.** When present, the block is a full GLM-4.5 MoE decoder block, including routing. This problem omits that block to keep the focus on the MTP-specific algebra</span>

---

## <span style="font-size: 16px;">Pitfalls</span>

* <span style="font-size: 14px;">**Norming the concatenation instead of each stream separately.** Concatenating $h_{prev}$ and $e_{next}$ first and applying a single RMSNorm to the joint $2H$ vector throws away the independent-gain property and ties the scales of the two streams together. The two-norm design is deliberate.</span>
* <span style="font-size: 14px;">**Skipping the final RMSNorm before the lm head.** RMSNorm before the lm head stabilizes the logit scale at training time and is what the trained $W_{lm}$ expects to see. Dropping it makes the logits explode or collapse depending on the magnitude of $z$.</span>
* <span style="font-size: 14px;">**Swapping the order of $h$ and $e$ in the concatenation.** The projection $W_{proj}$ is trained with a specific row ordering (h on top, e on bottom). Swapping the halves applies the wrong learned weights to each stream and silently corrupts the logits.</span>
* <span style="font-size: 14px;">**Using $4H$ or $H$ for the projection input.** $W_{proj}$ is exactly $(2H, H)$: the input has $2H$ features (concatenation of two $H$-dim vectors) and the output has $H$ features (matching the model dimension). Misreading the shape and using $4H$ or $H$ is a common copy-paste error.</span>
* <span style="font-size: 14px;">**Predicting $t+1$ instead of $t+2$.** The MTP head sees the actual $e_{next}$ embedding for position $t+1$, so the only useful thing it can predict is $t+2$. Training it to predict $t+1$ collapses to "copy the input embedding through the projection" and learns nothing useful.</span>
* <span style="font-size: 14px;">**Forgetting the lm head matmul.** The output of the MTP block is a hidden state, not logits. The final $\hat z \cdot W_{lm}$ matmul is what turns the hidden state into a distribution over the vocabulary. Returning $\hat z$ directly gives the wrong shape $(n, H)$ instead of $(n, V)$.</span>
* <span style="font-size: 14px;">**Mismatched epsilon between the three norms.** GLM-4.5 uses the same $\epsilon$ for all three RMSNorms in the MTP module. Using different epsilons (or floor/ceiling values like $10^{-12}$ that mismatch the trained checkpoint) silently shifts the output magnitudes.</span>

---
