# <span style="font-size: 20px;">Residual Weight Scaling</span>

<span style="font-size: 14px;">Residual weight scaling is an initialization trick introduced in GPT-2 (Radford et al., 2019) that scales the output projection weights of each residual sub-layer by $1/\sqrt{N}$, where $N$ is the total number of residual layers. Without this scaling, activations in the residual stream grow with depth and can explode in deep networks. With it, the variance of the residual stream stays bounded regardless of how many layers the model has.</span>

---

## <span style="font-size: 16px;">What It Is</span>

<span style="font-size: 14px;">Residual weight scaling is a modification to weight initialization, not to the architecture or the forward pass equations. At initialization time, you multiply the weights of specific layers by $1/\sqrt{N}$ so that their initial output contributions are smaller. During training, these weights are free to grow, but the scaled starting point prevents the very first forward passes from producing enormous activations that destabilize optimization.</span>

<span style="font-size: 14px;">The technique targets only the **output projections** of residual sub-layers. In a Transformer, each block contains two residual connections: one around the self-attention sub-layer and one around the feed-forward network (FFN). The output projection is the final linear layer in each sub-layer -- $W_O$ in attention and $W_2$ in the FFN -- whose output gets added directly to the residual stream. These are the only layers that get scaled.</span>

<span style="font-size: 14px;">Every other weight matrix ($W_Q$, $W_K$, $W_V$, $W_1$, embedding weights, LayerNorm parameters) is initialized normally, without the $1/\sqrt{N}$ factor.</span>

---

## <span style="font-size: 16px;">Key Equations</span>

<span style="font-size: 14px;">**Scaled weight initialization.** Given a weight matrix $W$ initialized by any standard scheme (e.g., Xavier normal), the scaled version is:</span>

$$
W_{\text{scaled}} = \frac{W}{\sqrt{N}}
$$

<span style="font-size: 14px;">where $N$ is the number of residual layers. In GPT-2, each Transformer block contributes two residual connections (attention + FFN), so a model with $L$ blocks has $N = 2L$.</span>

<span style="font-size: 14px;">**Residual forward pass.** Each residual block computes:</span>

$$
x_{i+1} = x_i + f_i(x_i)
$$

<span style="font-size: 14px;">where $x_i$ is the residual stream at layer $i$ and $f_i$ is the sub-layer function. In the simplest case where $f_i(x) = W_i \cdot x$:</span>

$$
x_{i+1} = x_i + W_i \cdot x_i
$$

<span style="font-size: 14px;">With scaling applied to $W_i$:</span>

$$
x_{i+1} = x_i + \frac{W_i}{\sqrt{N}} \cdot x_i
$$

<span style="font-size: 14px;">**L2 norm.** The L2 norm of a vector $x \in \mathbb{R}^d$ measures activation magnitude:</span>

$$
\|x\|_2 = \sqrt{\sum_{j=1}^{d} x_j^2}
$$

<span style="font-size: 14px;">By comparing $\|x_N\|_2$ with and without scaling, you can directly observe the stabilization effect.</span>

---

## <span style="font-size: 16px;">The Activation Explosion Problem</span>

<span style="font-size: 14px;">Consider what happens in a deep residual network without scaling. Each residual layer adds a contribution:</span>

$$
x_{i+1} = x_i + W_i \cdot x_i = (I + W_i) \cdot x_i
$$

<span style="font-size: 14px;">At initialization, $W_i$ has random entries from a zero-mean distribution with variance $\sigma^2$. With Xavier init ($\sigma^2 \approx 1/d$), the product $W_i \cdot x_i$ has roughly the same magnitude as $x_i$. Each layer adds a contribution of roughly the same size as the current stream.</span>

<span style="font-size: 14px;">After $N$ residual additions, variance accumulates. Each layer independently contributes $\sim \sigma^2$ of variance, so:</span>

$$
\text{Var}(x_N) \approx \text{Var}(x_0) + N \cdot \sigma^2
$$

<span style="font-size: 14px;">The **L2 norm grows as $O(\sqrt{N})$**. For $N = 96$ residual layers (GPT-2 XL), the activation magnitude at the final layer is roughly $\sqrt{96} \approx 9.8$ times larger than the input. This creates several problems:</span>

* <span style="font-size: 14px;">**Gradient instability.** Large activations produce large gradients, causing updates that overshoot and destabilize training.</span>
* <span style="font-size: 14px;">**Numerical overflow.** In FP16 training (max value ~65,504), exploding activations can overflow to infinity.</span>
* <span style="font-size: 14px;">**LayerNorm saturation.** Extremely large pre-norm values create vanishing gradients through the normalization.</span>
* <span style="font-size: 14px;">**Loss spikes.** Unstable logits from the first forward passes cause sudden loss jumps that can permanently derail training.</span>

---

## <span style="font-size: 16px;">Why 1/sqrt(N)</span>

<span style="font-size: 14px;">The factor $1/\sqrt{N}$ is the precise correction that keeps variance constant regardless of depth. Without scaling, after $N$ layers:</span>

$$
\text{Var}(x_N) = \text{Var}(x_0) + N \cdot \sigma^2
$$

<span style="font-size: 14px;">Now apply scaling: each layer's contribution is divided by $\sqrt{N}$, giving variance $\frac{\sigma^2}{N}$ per layer. After $N$ layers:</span>

$$
\text{Var}(x_N) = \text{Var}(x_0) + N \cdot \frac{\sigma^2}{N} = \text{Var}(x_0) + \sigma^2
$$

<span style="font-size: 14px;">The depth factor $N$ cancels perfectly. The final variance is $\text{Var}(x_0) + \sigma^2$ regardless of whether the model has 12 or 96 residual layers. **Scaling by $1/\sqrt{N}$ converts a depth-dependent variance into a depth-independent one.**</span>

<span style="font-size: 14px;">Why not $1/N$? Each layer's contribution would have variance $\sigma^2 / N^2$, totaling $\sigma^2/N$ after $N$ layers. As $N$ grows, the residual layers contribute almost nothing -- the network behaves like a very shallow model with deep layers that barely modify the stream.</span>

<span style="font-size: 14px;">Why not $1/\sqrt{2N}$? GPT-2 defines $N$ as total residual connections ($N = 2L$), so the factor is already $1/\sqrt{2L}$. Using $\sqrt{2N}$ on top of that double-counts. If your code defines $N$ as blocks instead, use $1/\sqrt{2N}$; if $N$ counts residual connections, use $1/\sqrt{N}$.</span>

---

## <span style="font-size: 16px;">Paper Context</span>

<span style="font-size: 14px;">The technique comes from "Language Models are Unsupervised Multitask Learners" (Radford et al., 2019). The paper states:</span>

<span style="font-size: 14px;">*"A modified initialization which accounts for the accumulation on the residual path with model depth is used. We scale the weights of residual layers at initialization by a factor of $1/\sqrt{N}$ where $N$ is the number of residual layers."*</span>

<span style="font-size: 14px;">GPT-2 ranges from 12 blocks (Small) to 48 blocks (XL), corresponding to $N$ from 24 to 96:</span>

* <span style="font-size: 14px;">**GPT-2 Small:** 12 blocks, $N = 24$, scale factor $= 1/\sqrt{24} \approx 0.2041$</span>
* <span style="font-size: 14px;">**GPT-2 Medium:** 24 blocks, $N = 48$, scale factor $= 1/\sqrt{48} \approx 0.1443$</span>
* <span style="font-size: 14px;">**GPT-2 Large:** 36 blocks, $N = 72$, scale factor $= 1/\sqrt{72} \approx 0.1179$</span>
* <span style="font-size: 14px;">**GPT-2 XL:** 48 blocks, $N = 96$, scale factor $= 1/\sqrt{96} \approx 0.1021$</span>

<span style="font-size: 14px;">The scale factor shrinks with depth. GPT-2 XL's output projections start at roughly 10% of their normal initialized magnitude, allowing a 48-block model to train stably from the first gradient step.</span>

<span style="font-size: 14px;">The scaling targets exactly two weight matrices per block: the attention output projection $W_O$ and the FFN second linear layer $W_2$. All other matrices ($W_Q$, $W_K$, $W_V$, $W_1$, embedding, LayerNorm) are unmodified.</span>

<span style="font-size: 14px;">GPT-2 uses pre-LayerNorm (applying LayerNorm before each sub-layer rather than after), which interacts favorably with residual scaling. Pre-norm ensures the sub-layer receives normalized input, and the scaled output projection controls the magnitude of what gets added back. This combination gives GPT-2 much better training stability than the original post-norm Transformer.</span>

---

## <span style="font-size: 16px;">The Forward Comparison</span>

<span style="font-size: 14px;">The most direct way to see the effect of residual weight scaling is to run the same input through $N$ residual blocks twice -- once with standard weights and once with scaled weights -- then compare the L2 norm of the output.</span>

* <span style="font-size: 14px;">**Input:** A fixed vector $x_0 \in \mathbb{R}^d$ and $N$ weight matrices $W_0, W_1, \ldots, W_{N-1}$, each $(d, d)$.</span>
* <span style="font-size: 14px;">**Unscaled forward:** Apply $x_{i+1} = x_i + W_i \cdot x_i$ for all layers. Record $\|x_N\|_2$.</span>
* <span style="font-size: 14px;">**Scaled forward:** Apply $x_{i+1} = x_i + \frac{W_i}{\sqrt{N}} \cdot x_i$ for all layers. Record $\|x_N\|_2$.</span>
* <span style="font-size: 14px;">**Compare:** The unscaled norm will be significantly larger, and the gap grows with $N$.</span>

<span style="font-size: 14px;">This comparison strips away the complexity of a real Transformer (attention, nonlinearities, normalization) to isolate the core phenomenon: linear residual additions accumulate variance, and $1/\sqrt{N}$ scaling corrects it. The forward comparison returns both L2 norms rounded to 4 decimal places.</span>

---

## <span style="font-size: 16px;">Numerical Example</span>

<span style="font-size: 14px;">Trace through $N = 4$ residual layers with dimension $d = 3$.</span>

<span style="font-size: 14px;">**Input:** $x_0 = [1.0, 0.5, -0.5]$, $\|x_0\|_2 = \sqrt{1.5} \approx 1.2247$.</span>

<span style="font-size: 14px;">**Weight matrices:**</span>

$$
W_0 = \begin{pmatrix} 0.3 & -0.1 & 0.2 \\ 0.1 & 0.4 & -0.2 \\ -0.2 & 0.1 & 0.3 \end{pmatrix}, \quad W_1 = \begin{pmatrix} -0.1 & 0.3 & 0.1 \\ 0.2 & -0.2 & 0.4 \\ 0.1 & 0.2 & -0.1 \end{pmatrix}
$$

$$
W_2 = \begin{pmatrix} 0.2 & 0.1 & -0.3 \\ -0.1 & 0.3 & 0.2 \\ 0.3 & -0.2 & 0.1 \end{pmatrix}, \quad W_3 = \begin{pmatrix} 0.1 & -0.2 & 0.3 \\ 0.3 & 0.1 & -0.1 \\ -0.1 & 0.4 & 0.2 \end{pmatrix}
$$

<span style="font-size: 14px;">**Unscaled forward pass:**</span>

<span style="font-size: 14px;">**Layer 0:** $W_0 \cdot x_0 = [0.15, 0.4, -0.3]$. $x_1 = x_0 + W_0 x_0 = [1.15, 0.9, -0.8]$, $\|x_1\|_2 \approx 1.6651$.</span>

<span style="font-size: 14px;">**Layer 1:** $W_1 \cdot x_1 = [0.075, -0.27, 0.375]$. $x_2 = [1.225, 0.63, -0.425]$, $\|x_2\|_2 \approx 1.4416$.</span>

<span style="font-size: 14px;">**Layer 2:** $W_2 \cdot x_2 = [0.4355, -0.0185, 0.199]$. $x_3 = [1.6605, 0.6115, -0.226]$, $\|x_3\|_2 \approx 1.7839$.</span>

<span style="font-size: 14px;">**Layer 3:** $W_3 \cdot x_3 = [-0.024, 0.582, 0.0333]$. $x_4 = [1.6365, 1.1935, -0.1927]$, $\|x_4\|_2 \approx 2.0346$.</span>

<span style="font-size: 14px;">**Unscaled result:** Norm grew from $1.2247$ to $2.0346$ -- a $66\%$ increase over 4 layers.</span>

<span style="font-size: 14px;">**Scaled forward pass ($1/\sqrt{4} = 0.5$):**</span>

<span style="font-size: 14px;">Each $W_i$ is halved before computing the residual addition. The matrix-vector products change because each layer receives a different $x_i$ from the scaled path.</span>

<span style="font-size: 14px;">**Layer 0:** Scaled $W_0 x_0 / 2 = [0.075, 0.2, -0.15]$. $x_1 = [1.075, 0.7, -0.65]$, $\|x_1\|_2 \approx 1.4381$.</span>

<span style="font-size: 14px;">**Layer 1:** $W_1 \cdot x_1 / 2 = [0.0188, -0.0925, 0.1563]$. $x_2 = [1.0938, 0.6075, -0.4937]$, $\|x_2\|_2 \approx 1.3451$.</span>

<span style="font-size: 14px;">**Layer 2:** $W_2 \cdot x_2 / 2 = [0.2139, -0.0129, 0.0786]$. $x_3 = [1.3077, 0.5946, -0.4151]$, $\|x_3\|_2 \approx 1.4953$.</span>

<span style="font-size: 14px;">**Layer 3:** $W_3 \cdot x_3 / 2 = [-0.0563, 0.2467, 0.012]$. $x_4 = [1.2514, 0.8413, -0.4031]$, $\|x_4\|_2 \approx 1.5609$.</span>

<span style="font-size: 14px;">**Scaled result:** Norm grew from $1.2247$ to $1.5609$ -- only a $27\%$ increase, compared to $66\%$ without scaling.</span>

<span style="font-size: 14px;">**Summary:** At every layer, the scaled norm stays closer to the input norm. With 4 layers the gap is moderate ($2.0346$ vs. $1.5609$). At $N = 96$ (GPT-2 XL), the unscaled norm would be roughly $10\times$ the scaled norm -- the difference between a trainable and an unstable model.</span>

---

## <span style="font-size: 16px;">Connection to Other Scaling Methods</span>

<span style="font-size: 14px;">**Xavier/Glorot initialization** (Glorot & Bengio, 2010) sets weight variance to $2/(d_{\text{in}} + d_{\text{out}})$ to preserve variance across a single linear layer. It solves the width problem but not depth. GPT-2's $1/\sqrt{N}$ is applied on top of Xavier -- first initialize normally, then multiply by $1/\sqrt{N}$.</span>

<span style="font-size: 14px;">**He/Kaiming initialization** (He et al., 2015) extends Xavier for ReLU networks using variance $2/d_{\text{in}}$. Like Xavier, it addresses width but not residual depth. The $1/\sqrt{N}$ factor is an orthogonal depth correction applied on top of either base scheme.</span>

<span style="font-size: 14px;">**Fixup initialization** (Zhang et al., 2019) eliminates LayerNorm/BatchNorm entirely by initializing the last layer of each residual branch to zero, so blocks start as identity functions. GPT-2 takes a middle ground -- scaled but non-zero initialization preserves signal flow while controlling variance growth.</span>

<span style="font-size: 14px;">**DeepNet** (Wang et al., 2022) extends the idea to very deep Transformers (up to 1000 layers) with a dual-factor scheme: upscaling the residual connection by $\alpha$ and downscaling initialization by $\beta$, both depth-dependent. GPT-2's $1/\sqrt{N}$ is a special case of their general framework.</span>

<span style="font-size: 14px;">**muP (Maximal Update Parametrization)** (Yang & Hu, 2021) adjusts learning rates, initialization, and multipliers jointly so optimal hyperparameters transfer across model widths. GPT-2's $1/\sqrt{N}$ can be seen as a depth-axis analogue of what muP does for width.</span>

---

## <span style="font-size: 16px;">Pitfalls</span>

<span style="font-size: 14px;">**1. Scaling all weight matrices, not just residual projections.**</span>

<span style="font-size: 14px;">Only the output projections ($W_O$ in attention, $W_2$ in FFN) should be scaled. Scaling $W_Q$, $W_K$, $W_V$, or $W_1$ changes sub-layer internals without addressing the residual accumulation problem. Scaling the embedding matrix shrinks all activations from the start, degrading representation quality.</span>

<span style="font-size: 14px;">**2. Using the wrong value of $N$.**</span>

<span style="font-size: 14px;">$N$ counts residual layers, not Transformer blocks. A 12-block GPT-2 model has $N = 24$ (2 residual connections per block). Using $N = 12$ gives $1/\sqrt{12} \approx 0.2887$ instead of $1/\sqrt{24} \approx 0.2041$ -- weights start 41% too large.</span>

<span style="font-size: 14px;">**3. Confusing $\sqrt{N}$ with $\sqrt{2N}$.**</span>

<span style="font-size: 14px;">If $N$ counts blocks, use $1/\sqrt{2N}$. If $N$ counts residual connections, use $1/\sqrt{N}$. Mixing conventions underestimates the correction by $\sqrt{2}$.</span>

<span style="font-size: 14px;">**4. Applying scaling at inference time rather than baking it into initialization.**</span>

<span style="font-size: 14px;">The $1/\sqrt{N}$ factor is a one-time initialization operation. Multiply the weight tensor once when creating the model, then train with the modified weights. Do not divide outputs by $\sqrt{N}$ during the forward pass -- weights diverge during training, so the scaling must be embedded in the starting point.</span>

<span style="font-size: 14px;">**5. Forgetting to scale when increasing model depth.**</span>

<span style="font-size: 14px;">The $1/\sqrt{N}$ factor must change when you change the number of layers. Hardcoding a scale factor from one configuration and reusing it for a different depth defeats the purpose.</span>

<span style="font-size: 14px;">**6. Over-attributing stability to scaling alone.**</span>

<span style="font-size: 14px;">Residual weight scaling works together with LayerNorm and learning rate tuning. Scaling without LayerNorm helps but does not fully prevent activation growth, because the variance analysis assumes independent layer contributions -- correlations between layers can still amplify the signal.</span>

---