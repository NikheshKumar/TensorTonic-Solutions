# <span style="font-size: 20px;">YaRN Scaled RoPE</span>

<span style="font-size: 14px;">YaRN (Yet another RoPE extensioN, Peng et al. 2023) is a context-extension recipe for Rotary Position Embeddings. It blends position interpolation on low-frequency dimensions with extrapolation on high-frequency dimensions, then applies a temperature on the attention logits. GPT-OSS uses YaRN to push a model pretrained at 4096 tokens to a 128K context.</span>

---

## <span style="font-size: 16px;">RoPE in One Paragraph</span>

<span style="font-size: 14px;">Rotary Position Embedding (Su et al. 2021) injects absolute position into queries and keys by rotating consecutive coordinate pairs of every head by an angle proportional to position. For head dimension $d$ and base $\theta$ (typically $10000$), the angular velocity of the $i$-th pair ($i = 0, 1, \ldots, d/2-1$) is:</span>

$$
\omega_i = \theta^{-2i/d}
$$

<span style="font-size: 14px;">The token at position $m$ has its $i$-th pair rotated by $m \omega_i$. Two important properties follow from this construction:</span>

* <span style="font-size: 14px;">**Relative phase**: the dot product between a query at position $m$ and a key at position $n$ depends only on $m - n$, so RoPE encodes relative position.</span>
* <span style="font-size: 14px;">**Frequency spectrum**: $\omega_0 = 1$ (highest frequency, wavelength $2\pi$) down to $\omega_{d/2-1} \approx \theta^{-1}$ (lowest frequency, wavelength $2\pi\theta$). Low-frequency dimensions encode coarse position; high-frequency dimensions encode fine position.</span>

---

## <span style="font-size: 16px;">Why Plain RoPE Breaks at Long Contexts</span>

<span style="font-size: 14px;">A model pretrained at sequence length $L_{\text{train}}$ has only seen rotation angles up to $L_{\text{train}} \cdot \omega_i$ for each frequency. When inference extends to $L_{\text{test}} \gg L_{\text{train}}$, three failure modes appear:</span>

* <span style="font-size: 14px;">**Distribution shift on low-frequency dims.** Wavelength $2\pi/\omega_i$ may exceed $L_{\text{train}}$ so the model never observed a full rotation on those dims. Pushing them to $L_{\text{test}}$ produces unfamiliar angles and the attention logits behave erratically.</span>
* <span style="font-size: 14px;">**High-frequency aliasing under naive interpolation.** Naive Position Interpolation (Chen et al. 2023) divides all positions by $s = L_{\text{test}}/L_{\text{train}}$. That fixes the low-frequency dims but squashes the high-frequency dims, so the model loses the fine-grained position signal it was relying on.</span>
* <span style="font-size: 14px;">**Attention entropy drift.** After any interpolation, the typical magnitude of $q^\top k$ shrinks because rotations are smaller, which makes the softmax softer (higher entropy) than at training time.</span>

<span style="font-size: 14px;">YaRN addresses all three by treating low- and high-frequency dims differently (NTK-by-parts blending) and by rescaling the logits (concentration).</span>

---

## <span style="font-size: 16px;">NTK-by-parts Blending</span>

<span style="font-size: 14px;">Two candidate inverse frequencies are computed for every dim:</span>

$$
\omega^{\text{interp}}_i = \frac{1}{s} \cdot \theta^{-2i/d}, \qquad \omega^{\text{extrap}}_i = \theta^{-2i/d}
$$

<span style="font-size: 14px;">Interpolation divides by the scaling factor $s$ (good for low-frequency, long-wavelength dims). Extrapolation keeps the original frequency (good for high-frequency, short-wavelength dims). YaRN then chooses a per-dim blend weight by looking at how the wavelength of dim $i$ compares to $L_{\text{train}}$.</span>

<span style="font-size: 14px;">The wavelength of dim $i$ is $\lambda_i = 2\pi \theta^{2i/d}$. Define the "rotations per training context" $r_i = L_{\text{train}} / \lambda_i$. NTK-by-parts says:</span>

* <span style="font-size: 14px;">If $r_i \ge \beta$ (dim completes many rotations within $L_{\text{train}}$): pure extrapolation. The model has seen enough variation in this dim that extending position is safe.</span>
* <span style="font-size: 14px;">If $r_i \le \alpha$ (dim never completes one rotation within $L_{\text{train}}$): pure interpolation. The model has never seen a full cycle, so divide angles by $s$.</span>
* <span style="font-size: 14px;">Between $\alpha$ and $\beta$: linear ramp.</span>

<span style="font-size: 14px;">Inverting $r_i = L_{\text{train}} / (2\pi \theta^{2i/d})$ for $i$ gives the cutoff indices:</span>

$$
\text{low} = \frac{d}{2} \cdot \frac{\ln\!\left(L_{\text{train}} / (\beta \cdot 2\pi)\right)}{\ln \theta}, \quad \text{high} = \frac{d}{2} \cdot \frac{\ln\!\left(L_{\text{train}} / (\alpha \cdot 2\pi)\right)}{\ln \theta}
$$

<span style="font-size: 14px;">By construction, $\text{low} < \text{high}$ because $\alpha < \beta$. The ramp and mask are:</span>

$$
\text{ramp}_i = \frac{i - \text{low}}{\text{high} - \text{low}}, \qquad \text{mask}_i = 1 - \text{clip}(\text{ramp}_i, 0, 1)
$$

<span style="font-size: 14px;">$\text{mask}_i$ is the extrapolation weight: 1 below $\text{low}$ (pure extrapolation), 0 above $\text{high}$ (pure interpolation), linear in between. The final inverse frequency is the convex combination:</span>

$$
\omega_i^{\text{YaRN}} = (1 - \text{mask}_i) \cdot \omega^{\text{interp}}_i + \text{mask}_i \cdot \omega^{\text{extrap}}_i
$$

<span style="font-size: 14px;">Note the sign convention. Low index $i$ corresponds to high-frequency dims (short wavelength, many rotations per $L_{\text{train}}$), so the mask is 1 there. High index $i$ corresponds to low-frequency dims (long wavelength), so the mask is 0 there. The naming "low" and "high" in the reference code refers to the index axis, not the frequency axis, which is a common source of confusion.</span>

---

## <span style="font-size: 16px;">Concentration (Attention Temperature)</span>

<span style="font-size: 14px;">After interpolation the rotation magnitudes shrink, so $q^\top k$ shrinks, and softmax becomes flatter than at training time. YaRN compensates by multiplying the pre-softmax logits by a scalar:</span>

$$
\text{concentration}(s) = 0.1 \cdot \ln(s) + 1
$$

<span style="font-size: 14px;">The $0.1$ constant is empirical, from the YaRN paper. The motivation comes from analysing the variance of $q^\top k$ before and after rescaling angles by $1/s$; the log scaling fits the observed variance ratio over a wide range of $s$. GPT-OSS uses this exact value verbatim.</span>

<span style="font-size: 14px;">When $s = 32$, $\text{concentration} \approx 1.347$. The downstream attention layer applies the standard $1/\sqrt{d_k}$ scaling, then multiplies by $\text{concentration}$. Equivalently, $1/\sqrt{d_k}$ is replaced by $\text{concentration}/\sqrt{d_k}$.</span>

---

## <span style="font-size: 16px;">GPT-OSS Settings</span>

<span style="font-size: 14px;">GPT-OSS ships with the following YaRN configuration (see `gpt_oss/torch/model.py`):</span>

* <span style="font-size: 14px;">**Base $\theta$**: $150000$ (much larger than LLaMA's $10000$, which already pushes wavelengths longer and softens the need for aggressive interpolation).</span>
* <span style="font-size: 14px;">**Initial context length**: $4096$.</span>
* <span style="font-size: 14px;">**Scaling factor $s$**: $32$, giving a target context of $4096 \times 32 = 131072$ tokens ($128$K).</span>
* <span style="font-size: 14px;">**$\alpha = 1.0$, $\beta = 32.0$**.</span>
* <span style="font-size: 14px;">**Head dim**: $64$.</span>

<span style="font-size: 14px;">Plugging into the cutoff formulas with $d/2 = 32$, $L_{\text{train}} = 4096$, $\theta = 150000$:</span>

$$
\text{low} \approx 32 \cdot \frac{\ln(4096 / (32 \cdot 2\pi))}{\ln 150000} \approx 32 \cdot \frac{\ln 20.37}{\ln 150000} \approx 32 \cdot 0.253 \approx 8.10
$$

$$
\text{high} \approx 32 \cdot \frac{\ln(4096 / (1 \cdot 2\pi))}{\ln 150000} \approx 32 \cdot \frac{\ln 651.9}{\ln 150000} \approx 32 \cdot 0.544 \approx 17.4
$$

<span style="font-size: 14px;">So dims $0$ to $8$ are pure extrapolation, dims $18$ to $31$ are pure interpolation, dims $9$ to $17$ are the linear blend. The assertion $0 < \text{low} < \text{high} < d/2 - 1$ in the reference implementation guards against pathological hyperparameter choices that would put the cutoffs outside the index range.</span>

---

## <span style="font-size: 16px;">Comparison with Other RoPE Extensions</span>

* <span style="font-size: 14px;">**Vanilla RoPE**: no extension. Performance collapses sharply once the test context exceeds $L_{\text{train}}$, because low-frequency dims see unseen angles.</span>
* <span style="font-size: 14px;">**Position Interpolation (PI, Chen et al. 2023)**: divide all positions by $s$. Equivalent to YaRN with $\text{mask} = 0$ everywhere. Fixes low-frequency dims but loses high-frequency precision; works well up to about $4 \times$ context.</span>
* <span style="font-size: 14px;">**NTK-aware (bloc97 2023)**: keep PI on low-frequency dims, extrapolate on high-frequency dims, but use a non-blending hard switch. A precursor to YaRN.</span>
* <span style="font-size: 14px;">**YaRN**: NTK-aware with a smooth ramp plus the concentration temperature. The smooth ramp avoids the discontinuity at the cutoff, and the temperature restores the softmax sharpness. Empirically wins for $s \ge 8$.</span>
* <span style="font-size: 14px;">**Longrope (Microsoft 2024)**: learns per-dim scaling factors via search instead of a fixed ramp. Strictly more flexible than YaRN, at the cost of an extra calibration step.</span>

---

## <span style="font-size: 16px;">Worked Example: head_dim=8, scaling_factor=4, base=10000</span>

<span style="font-size: 14px;">$d/2 = 4$, $L_{\text{train}} = 4096$, $\alpha = 1$, $\beta = 32$, $\theta = 10000$.</span>

<span style="font-size: 14px;">**Step 1. Base frequency table.**</span>

$$
\texttt{freq} = 10000^{[0, 2, 4, 6]/8} = [1, 10, 100, 1000]
$$

<span style="font-size: 14px;">So extrapolation values are $1/\texttt{freq} = [1, 0.1, 0.01, 0.001]$.</span>

<span style="font-size: 14px;">**Step 2. Concentration.**</span>

$$
0.1 \ln 4 + 1 = 0.1 \cdot 1.3863 + 1 = 1.1386
$$

<span style="font-size: 14px;">**Step 3. Cutoffs.** $\ln 10000 = 9.2103$.</span>

$$
\text{low} = 4 \cdot \frac{\ln(4096/(32 \cdot 2\pi))}{9.2103} = 4 \cdot \frac{\ln 20.37}{9.2103} = 4 \cdot 0.327 = 1.31
$$

$$
\text{high} = 4 \cdot \frac{\ln(4096/(1 \cdot 2\pi))}{9.2103} = 4 \cdot \frac{\ln 651.9}{9.2103} = 4 \cdot 0.704 = 2.82
$$

<span style="font-size: 14px;">**Step 4. Ramp and mask.** $\texttt{ramp} = ([0,1,2,3] - 1.31) / (2.82 - 1.31) = [-0.87, -0.21, 0.46, 1.12]$. Clipping to $[0, 1]$ gives $[0, 0, 0.46, 1]$. Then $\texttt{mask} = 1 - \texttt{clip} = [1, 1, 0.54, 0]$.</span>

<span style="font-size: 14px;">**Step 5. Blend.** Interpolation is $1/(4 \cdot \texttt{freq}) = [0.25, 0.025, 0.0025, 0.00025]$. Extrapolation is $[1, 0.1, 0.01, 0.001]$.</span>

* <span style="font-size: 14px;">Dim 0: $\texttt{mask}=1$, output $= 1 \cdot 1 + 0 \cdot 0.25 = 1$.</span>
* <span style="font-size: 14px;">Dim 1: $\texttt{mask}=1$, output $= 1 \cdot 0.1 + 0 \cdot 0.025 = 0.1$.</span>
* <span style="font-size: 14px;">Dim 2: $\texttt{mask}=0.54$, output $= 0.54 \cdot 0.01 + 0.46 \cdot 0.0025 = 0.0054 + 0.00115 \approx 0.006557$.</span>
* <span style="font-size: 14px;">Dim 3: $\texttt{mask}=0$, output $= 0 \cdot 0.001 + 1 \cdot 0.00025 = 0.00025$.</span>

<span style="font-size: 14px;">Final result: $(\texttt{concentration}, \texttt{inv\_freq}) = (1.1386, [1, 0.1, 0.006557, 0.00025])$, matching the public test case exactly.</span>

---

## <span style="font-size: 16px;">Connecting to Real Models</span>

* <span style="font-size: 14px;">**GPT-OSS** (OpenAI, 2024): $L_{\text{train}} = 4096$, $s = 32$, target $128$K. YaRN with $\theta = 150000$.</span>
* <span style="font-size: 14px;">**Qwen 2** (Alibaba, 2024): also uses YaRN; $\theta = 1000000$ and various $s$ depending on variant.</span>
* <span style="font-size: 14px;">**LLaMA 3.1**: uses a YaRN-like long-context recipe (the official paper calls it "frequency-dependent scaling").</span>
* <span style="font-size: 14px;">**Mistral and DeepSeek**: variations of NTK-aware or learned scaling; the YaRN reference is widely cited.</span>

---

## <span style="font-size: 16px;">Implementation Notes</span>

<span style="font-size: 14px;">The reference implementation in `gpt_oss/torch/model.py` precomputes $(\texttt{concentration}, \texttt{inv\_freq})$ once at model load time. The returned $\texttt{inv\_freq}$ is then combined with a positions vector $\texttt{pos} \in \{0, 1, \ldots, L-1\}$ to build the per-position angle table:</span>

$$
\texttt{angles}[m, i] = m \cdot \texttt{inv\_freq}[i]
$$

<span style="font-size: 14px;">From the angles, the standard $\cos$ and $\sin$ tables are produced and applied to query and key pairs. The $\texttt{concentration}$ scalar is stored separately and multiplied into the attention logit scaling factor inside the attention module.</span>

<span style="font-size: 14px;">A subtle point: since YaRN modifies the inverse frequencies (not the positions themselves), the change is invisible to any code that treats RoPE as a black box. Swapping vanilla RoPE for YaRN requires only replacing the precomputed $\texttt{inv\_freq}$ table and multiplying the logits by $\texttt{concentration}$. No changes to attention shape, masking, or KV cache are needed.</span>

---

## <span style="font-size: 16px;">Pitfalls</span>

* <span style="font-size: 14px;">**Confusing low and high.** $\text{low}$ and $\text{high}$ are cutoffs along the dimension index, not along the frequency. Low index = high frequency = mask $\to 1$ = extrapolation. Swapping the roles of $\alpha$ and $\beta$, or of low and high, silently produces wrong inverse frequencies that still type-check.</span>
* <span style="font-size: 14px;">**Forgetting the clip.** Without $\text{clip}(\text{ramp}, 0, 1)$, the mask becomes negative below $\text{low}$ and greater than 1 above $\text{high}$. The final blend then extrapolates beyond either endpoint, including negative inverse frequencies near the corners.</span>
* <span style="font-size: 14px;">**Wrong concentration constant.** The factor is $0.1$, not $0.5$ or $\ln 10$. A wrong constant changes the attention logit scale and the model's effective temperature, which usually shows up as either oversharp or oversoft attention at long context.</span>
* <span style="font-size: 14px;">**Skipping the $s \le 1$ branch.** When no extension is requested, return vanilla RoPE without computing cutoffs. The cutoff formula divides by $\ln \theta$ and depends on $\ln s$ via concentration; the YaRN math still works mathematically but produces a concentration of $1 + 0$ only when $s = 1$ exactly, and the cutoff-bound assertion can fail for tiny $L_{\text{train}}$.</span>
* <span style="font-size: 14px;">**Indexing the freq table with the wrong stride.** $\texttt{freq[i]} = \theta^{2i/d}$ uses $\texttt{torch.arange(0, d, 2) / d}$, not $\texttt{torch.arange(d/2) / (d/2)}$. The two expressions look similar but differ for $d/2 = 1$ and edge dims. Use the stride-2 version to match the reference.</span>
* <span style="font-size: 14px;">**Treating concentration as a multiplier on inv_freq.** Concentration scales the attention logits, not the rotation angles. Multiplying inv_freq by concentration is a common bug; the correct location is in the attention layer's scaling factor.</span>
* <span style="font-size: 14px;">**Hard-coding $\alpha, \beta$.** The reference defaults are $\alpha = 1, \beta = 32$ for GPT-OSS, but Qwen 2 and Llama 3.1 use different values. Keep them as parameters and pass through.</span>
* <span style="font-size: 14px;">**Assertion failure on tiny head_dim.** The reference implementation asserts $0 < \text{low} < \text{high} < d/2 - 1$. For very small $d$ (say $d = 4$) and large $\beta$, this can fail. In practice $d$ is at least $32$, so the assertion holds.</span>

---
