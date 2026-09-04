# <span style="font-size: 20px;">RMSNorm</span>

<span style="font-size: 14px;">Root Mean Square Layer Normalization (Zhang and Sennrich, 2019) is the normalization primitive used throughout GLM-4.5. It rescales each token's hidden vector by the inverse root mean square of its components and then applies a learned per-channel gain. Compared to standard LayerNorm, it skips the mean centering step, which makes it faster, simpler, and empirically just as effective in modern decoder-only transformers.</span>

---

## <span style="font-size: 16px;">Why GLM-4.5 Uses RMSNorm</span>

<span style="font-size: 14px;">GLM-4.5 is a Mixture-of-Experts transformer with 355B total and 32B active parameters. At that scale, normalization layers are evaluated trillions of times during training and inference, so the per-layer cost matters. The GLM-4.5 technical report follows the LLaMA family in adopting RMSNorm on the residual stream because:</span>

* <span style="font-size: 14px;">**Lower FLOPs.** RMSNorm drops the mean subtraction and the variance computation, replacing them with a single mean-of-squares reduction. For hidden size $d$, that is one pass over the vector instead of two.</span>
* <span style="font-size: 14px;">**Cleaner gradients.** Without the recentering term, the backward pass has fewer dependencies between channels, which simplifies fused kernels and helps stability for very deep stacks.</span>
* <span style="font-size: 14px;">**Empirical parity.** Across architectures (LLaMA, Mistral, GLM, Qwen, DeepSeek), RMSNorm matches LayerNorm in final loss while being measurably faster. The GLM team did not see a quality regression when switching.</span>

<span style="font-size: 14px;">Inside each GLM-4.5 block, RMSNorm appears three times: before attention (pre-norm), before the FFN or MoE router, and as the final norm before the language modeling head. The same primitive is reused everywhere, so an efficient implementation pays dividends across the whole network.</span>

---

## <span style="font-size: 16px;">The Formula</span>

<span style="font-size: 14px;">Given a hidden vector $x \in \mathbb{R}^d$ at a single token position, a learned gain vector $\gamma \in \mathbb{R}^d$, and a small constant $\epsilon$ for numerical stability, RMSNorm computes:</span>

$$
\text{RMSNorm}(x)_i = \frac{x_i}{\sqrt{\frac{1}{d}\sum_{j=1}^{d} x_j^2 + \epsilon}} \cdot \gamma_i
$$

<span style="font-size: 14px;">Equivalently, define the root mean square statistic:</span>

$$
\text{RMS}(x) = \sqrt{\frac{1}{d}\sum_{j=1}^{d} x_j^2 + \epsilon}
$$

<span style="font-size: 14px;">Then the operator is just</span>

$$
\text{RMSNorm}(x) = \gamma \odot \frac{x}{\text{RMS}(x)}
$$

<span style="font-size: 14px;">where $\odot$ is the elementwise product. The statistic is shared across all $d$ channels of a token, so the normalization is applied along the last axis of the activation tensor and broadcast across any batch or sequence dimensions in front.</span>

---

## <span style="font-size: 16px;">Comparison With LayerNorm</span>

<span style="font-size: 14px;">LayerNorm (Ba et al., 2016) is the older primitive that RMSNorm replaces. The two differ in exactly one step:</span>

* <span style="font-size: 14px;">**LayerNorm** centers and scales: $\text{LN}(x) = \gamma \odot (x - \mu) / \sqrt{\sigma^2 + \epsilon} + \beta$, where $\mu$ is the mean across the feature axis and $\sigma^2$ is the variance.</span>
* <span style="font-size: 14px;">**RMSNorm** scales only: $\text{RMSNorm}(x) = \gamma \odot x / \text{RMS}(x)$. No mean subtraction, no learned bias $\beta$.</span>

<span style="font-size: 14px;">Zhang and Sennrich (2019) argued that the success of LayerNorm is driven by its re-scaling invariance rather than its re-centering invariance. They tested this by ablating the mean centering and found no quality loss across translation and language modeling benchmarks. Modern LLMs took this as license to drop the recentering term and pocket the speed savings.</span>

<span style="font-size: 14px;">There is also no $\beta$ shift in standard RMSNorm. The learned gain $\gamma$ is sufficient to recover any required scale, and an additive bias would interact poorly with the residual stream's expected zero-mean activations.</span>

---

## <span style="font-size: 16px;">Why Drop The Mean?</span>

<span style="font-size: 14px;">The mean of a hidden activation across $d$ channels has a specific role: it removes any constant DC offset. In practice, three observations make this term unnecessary:</span>

* <span style="font-size: 14px;">**Residual streams are already centered.** After the first few blocks, the running mean of the hidden state is close to zero by symmetry, so subtracting it is mostly a no-op.</span>
* <span style="font-size: 14px;">**The next linear layer absorbs any offset.** Any constant added to all channels propagates into a single scalar per output unit of the next $W x$ multiplication, which the bias term of that layer can easily learn to cancel.</span>
* <span style="font-size: 14px;">**Re-scaling is what stabilizes training.** Empirically, the magnitude of activations is the dimension that drifts during training, not the mean. Controlling the RMS is enough to keep gradients well behaved.</span>

<span style="font-size: 14px;">Removing the mean also saves one global reduction per token and removes the centering bias from the backward pass, which simplifies the gradient formula.</span>

---

## <span style="font-size: 16px;">Mechanics Step By Step</span>

<span style="font-size: 14px;">1. **Square.** Compute $x_j^2$ for every channel along the last axis of the input tensor.</span>

<span style="font-size: 14px;">2. **Mean of squares.** Average across the last axis with $\texttt{keepdim=True}$ so the resulting statistic broadcasts back over $x$.</span>

<span style="font-size: 14px;">3. **Add epsilon and take the square root.** This produces $\text{RMS}(x)$ with $\epsilon$ inside the radical, guaranteeing the denominator is strictly positive even when $x$ is all zeros.</span>

<span style="font-size: 14px;">4. **Divide.** Compute $x / \text{RMS}(x)$. The division broadcasts along the last axis.</span>

<span style="font-size: 14px;">5. **Apply gain.** Multiply elementwise by $\gamma$. Because $\gamma$ has shape $(d,)$, this also broadcasts cleanly over any leading batch and sequence dimensions.</span>

<span style="font-size: 14px;">All five steps are pointwise or single-axis reductions, so the operator is trivially parallel across token positions and across the batch.</span>

---

## <span style="font-size: 16px;">Role In The GLM-4.5 Block</span>

<span style="font-size: 14px;">GLM-4.5 uses a pre-norm transformer layout. A single block looks like:</span>

* <span style="font-size: 14px;">**Pre-attention RMSNorm.** Normalizes the residual stream before projecting into queries, keys, and values for grouped-query attention with partial RoPE.</span>
* <span style="font-size: 14px;">**Attention residual.** The attention output is added back to the residual.</span>
* <span style="font-size: 14px;">**Pre-MoE RMSNorm.** Normalizes the residual again before the MoE router. The router scores experts, top-$k$ experts are activated, and their weighted outputs are summed.</span>
* <span style="font-size: 14px;">**MoE residual.** The MoE output is added back to the residual.</span>

<span style="font-size: 14px;">After the final block, one more RMSNorm is applied to the residual stream before the LM head projects to vocabulary logits. Each RMSNorm has its own independent $\gamma$ parameter, all initialized to ones and learned during training. The number of RMSNorm parameters in GLM-4.5 is therefore roughly $2 L d + d$ for $L$ layers of hidden size $d$, which is negligible compared to the attention and FFN weights.</span>

---

## <span style="font-size: 16px;">Worked Example (d = 4)</span>

<span style="font-size: 14px;">Let $x = [1, 2, 3, 4]$, $\gamma = [1, 1, 1, 1]$, $\epsilon = 10^{-5}$.</span>

<span style="font-size: 14px;">1. **Squares**: $[1, 4, 9, 16]$.</span>

<span style="font-size: 14px;">2. **Mean of squares**: $(1 + 4 + 9 + 16) / 4 = 30 / 4 = 7.5$.</span>

<span style="font-size: 14px;">3. **RMS**: $\sqrt{7.5 + 10^{-5}} \approx 2.7386$.</span>

<span style="font-size: 14px;">4. **Divide**: $x / \text{RMS} \approx [0.3651, 0.7303, 1.0954, 1.4606]$.</span>

<span style="font-size: 14px;">5. **Apply gain**: with $\gamma = 1$, the output equals the previous step.</span>

<span style="font-size: 14px;">Notice that the squared norm of the output is $0.3651^2 + 0.7303^2 + 1.0954^2 + 1.4606^2 \approx 3.9999$, which gives an RMS of $\sqrt{3.9999 / 4} = 0.99999$. After RMSNorm with unit gain, every token vector has RMS very close to 1, which is exactly the invariant the operator enforces.</span>

---

## <span style="font-size: 16px;">RMSNorm vs DeepNorm vs Sandwich Norm</span>

<span style="font-size: 14px;">Several norm variants compete for use in modern transformers:</span>

* <span style="font-size: 14px;">**LayerNorm + Post-Norm.** The original Vaswani et al. (2017) recipe. Applies LayerNorm after the residual addition. Trains poorly at depth without careful warmup and is rarely used in current LLMs.</span>
* <span style="font-size: 14px;">**LayerNorm + Pre-Norm.** Applies LayerNorm before each sublayer. Stabilizes deep stacks but is more expensive than RMSNorm.</span>
* <span style="font-size: 14px;">**RMSNorm + Pre-Norm.** The GLM-4.5, LLaMA, Mistral, and DeepSeek choice. Pre-norm placement plus the cheaper RMS statistic.</span>
* <span style="font-size: 14px;">**DeepNorm** (Wang et al., 2022). Rescales the residual branch by a constant $\alpha > 1$ and applies LayerNorm post-residual. Designed for 1000-layer models. Not adopted by GLM-4.5.</span>
* <span style="font-size: 14px;">**Sandwich Norm.** Adds an extra LayerNorm after each sublayer for extra stability. Used in some Chinese LLM lineages but not in GLM-4.5.</span>

<span style="font-size: 14px;">GLM-4.5 picks the simplest stable recipe: RMSNorm with pre-norm placement on a residual stream, mirroring LLaMA's design. The technical report attributes most of the training stability gains to MoE routing tweaks and learning rate schedules, not to the norm choice itself.</span>

---

## <span style="font-size: 16px;">Numerical Stability And Epsilon</span>

<span style="font-size: 14px;">The $\epsilon$ inside the square root has one job: prevent division by zero when the input vector is the zero vector. With $x = \mathbf{0}$, the mean of squares is zero and the unprotected denominator would be exactly zero, producing NaNs that propagate through the rest of the network.</span>

<span style="font-size: 14px;">Standard values:</span>

* <span style="font-size: 14px;">**$\epsilon = 10^{-5}$.** Default for LLaMA, GLM-4.5, and the original Zhang and Sennrich paper. Large enough to keep the denominator away from zero in float16 and bfloat16.</span>
* <span style="font-size: 14px;">**$\epsilon = 10^{-6}$.** Used by some FP32-friendly implementations when the activations stay well away from zero.</span>

<span style="font-size: 14px;">The $\epsilon$ must be added inside the radical, not outside. The form $\sqrt{m + \epsilon}$ has a non-zero floor for the denominator. The variant $\sqrt{m} + \epsilon$ has the same floor but a different gradient and is not the standard.</span>

---

## <span style="font-size: 16px;">Complexity</span>

<span style="font-size: 14px;">For a tensor of shape $(B, T, d)$:</span>

* <span style="font-size: 14px;">**Time**: $O(B \cdot T \cdot d)$. One pass to compute squares, one reduction along the last axis, one elementwise division, one elementwise multiplication.</span>
* <span style="font-size: 14px;">**Memory**: $O(1)$ extra beyond the output tensor. The RMS statistic has shape $(B, T, 1)$ and is negligible compared to the activation itself.</span>
* <span style="font-size: 14px;">**Parameters**: $d$ floats for $\gamma$. Initialized to all ones so that the operator is effectively identity at the start of training, then learned.</span>

<span style="font-size: 14px;">Modern attention kernels (Triton, CUDA) fuse RMSNorm into the surrounding linear layers when possible. The fused variant avoids materializing the normalized activations in HBM and is one of the easiest wins in inference engines like vLLM and SGLang, which both serve GLM-4.5.</span>

---

## <span style="font-size: 16px;">Pitfalls</span>

* <span style="font-size: 14px;">**Reducing over the wrong axis.** RMSNorm normalizes per token across the feature axis, which is the last axis of $x$. Reducing over the batch axis (axis 0) or the sequence axis (axis 1) breaks the invariant: every token mixes with every other token and the model collapses. Always use $\texttt{dim=-1}$ with $\texttt{keepdim=True}$.</span>

* <span style="font-size: 14px;">**Using variance instead of mean of squares.** PyTorch's $\texttt{var}$ subtracts the mean first, so $\texttt{x.var(dim=-1)}$ is not $\frac{1}{d}\sum x_j^2$. RMSNorm needs the raw mean-of-squares, which is $\texttt{(x**2).mean(dim=-1)}$. Substituting variance silently turns the operator into LayerNorm minus the bias and gives wrong results on every test that has non-zero mean activations.</span>

* <span style="font-size: 14px;">**Forgetting to add $\epsilon$ before the square root.** With $\epsilon = 0$ and $x = \mathbf{0}$, the operator produces NaN. Even with non-zero $x$, small inputs combined with float16 underflow can push the denominator to zero. The fix is always $\sqrt{\text{mean}(x^2) + \epsilon}$, never $\sqrt{\text{mean}(x^2)} + \epsilon$.</span>

* <span style="font-size: 14px;">**Skipping the square root.** Dividing by the mean of squares (no $\sqrt{\cdot}$) produces an output whose magnitude scales as $1 / \|x\|^2$ rather than $1 / \|x\|$. Activations shrink dramatically for inputs with even moderate magnitude. The square root is what gives RMSNorm its scale-invariance property.</span>

* <span style="font-size: 14px;">**Dropping the gain $\gamma$.** Without the learned per-channel scale, every layer's output is forced to have unit RMS in every channel. That is too restrictive: the model needs different channels to have different magnitudes to encode different concepts. The $\gamma$ vector is initialized to all ones but quickly diverges across channels during training.</span>

* <span style="font-size: 14px;">**Initializing $\gamma$ to zero.** Zero initialization makes the RMSNorm output the zero vector regardless of input, which kills the residual stream's gradient signal at the very first step. Always initialize $\gamma$ to ones so the operator starts as a normalize-only no-op.</span>

* <span style="font-size: 14px;">**Casting $\gamma$ to a different dtype than $x$ in mixed-precision training.** The $\gamma$ parameter is often kept in fp32 master weights while activations run in bf16. The multiplication must broadcast the cast cleanly. PyTorch handles this with $\texttt{torch.as\_tensor}$, but custom kernels can silently demote to fp16 and lose precision.</span>

* <span style="font-size: 14px;">**Computing the RMS in low precision.** The intermediate sum of squares grows quadratically with magnitude. Computing it in fp16 can overflow when activations spike. Production implementations (including GLM-4.5's inference path) cast to fp32 for the RMS computation and cast back to bf16 only at the end.</span>

---
