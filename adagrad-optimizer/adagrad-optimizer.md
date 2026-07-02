## The Problem: One Learning Rate for All Parameters

Vanilla gradient descent uses a single learning rate for every parameter in the model:

$$
w_t = w_{t-1} - \eta \cdot g_t
$$

Every weight, every bias, every embedding entry gets multiplied by the same $\eta$. This is a problem because different parameters have very different gradient characteristics:

- **Weights connected to frequent features** (like common words in NLP, or frequently active pixels): these get large gradients on most mini-batches. A learning rate that is good for these parameters may be too large, causing oscillation.
- **Weights connected to rare features** (like uncommon words, or rarely active inputs): these get nonzero gradients only occasionally. They barely move because most of the time their gradient is zero. A learning rate that is good for these parameters needs to be much larger.

A single $\eta$ cannot satisfy both groups. If you set it for the frequent features, rare features learn too slowly. If you set it for the rare features, frequent features oscillate.

---

## The Core Idea: Divide by Past Gradient Size

AdaGrad (Adaptive Gradient Algorithm, Duchi et al., 2011) gives each parameter its own effective learning rate based on a simple principle: **parameters that have received large gradients in the past should get smaller learning rates now.**

It keeps a running **accumulator** $G_t$ for each parameter:

$$
G_t = G_{t-1} + g_t^2
$$

- $G$ starts at 0
- At each step, the squared gradient $g_t^2$ is added (element-wise)
- $G_t$ is the **sum of all past squared gradients** up to step $t$
- For a parameter that always gets gradient 1.0: after 100 steps, $G = 100$
- For a parameter that gets gradient 1.0 only 5 times: after 100 steps, $G = 5$

Then the update rule divides by the square root of $G_t$:

$$
w_t = w_{t-1} - \frac{\eta}{\sqrt{G_t + \epsilon}} \cdot g_t
$$

- $\epsilon$ (typically $10^{-8}$) prevents division by zero when $G_t = 0$
- The denominator $\sqrt{G_t}$ is large for frequently-updated parameters, making their effective learning rate small
- The denominator is small for rarely-updated parameters, making their effective learning rate large

---

## A Step-by-Step Example

Parameters: $w = [1.0, 2.0]$, accumulator: $G = [0.0, 0.0]$, gradient: $g = [2.0, 0.1]$, $\eta = 0.5$

**Update the accumulator** (element-wise):
- $G_1 = 0 + (2.0)^2 = 4.0$
- $G_2 = 0 + (0.1)^2 = 0.01$

**Compute effective learning rates**:
- For parameter 1: $\frac{\eta}{\sqrt{G_1 + \epsilon}} = \frac{0.5}{\sqrt{4.0}} = \frac{0.5}{2.0} = 0.25$
- For parameter 2: $\frac{\eta}{\sqrt{G_2 + \epsilon}} = \frac{0.5}{\sqrt{0.01}} = \frac{0.5}{0.1} = 5.0$

**Update parameters**:
- $w_1 = 1.0 - 0.25 \times 2.0 = 1.0 - 0.5 = 0.5$
- $w_2 = 2.0 - 5.0 \times 0.1 = 2.0 - 0.5 = 1.5$

Both parameters moved by the same amount (0.5), even though parameter 1's gradient was 20x larger! AdaGrad automatically equalized the step sizes.

Now suppose we do another step with gradient $g = [2.0, 0.0]$:

**Update accumulator**:
- $G_1 = 4.0 + 4.0 = 8.0$
- $G_2 = 0.01 + 0.0 = 0.01$ (unchanged, gradient was zero)

**Effective learning rates**:
- For parameter 1: $\frac{0.5}{\sqrt{8.0}} \approx 0.177$ (decreased from 0.25)
- For parameter 2: $\frac{0.5}{\sqrt{0.01}} = 5.0$ (unchanged)

Parameter 1's learning rate has shrunk because it keeps getting large gradients. Parameter 2's rate stayed the same because it did not receive a gradient this step. This is exactly the adaptive behavior we want.

---

## The Monotonic Decay Problem

The accumulator $G_t$ is a **sum**. It only grows. It never decreases. This means the effective learning rate $\frac{\eta}{\sqrt{G_t}}$ only decreases over time.

After enough steps:

- $G_t$ becomes very large for every parameter (even rare ones eventually accumulate enough)
- The effective learning rate approaches zero
- The model essentially stops learning

This is a fundamental issue:

- **For convex optimization**: the decaying learning rate is actually a feature. Theory says you want the learning rate to decrease as $O(1/\sqrt{t})$ for optimal convergence, and that is exactly what AdaGrad provides. It has provably optimal convergence rates for convex problems.
- **For deep learning (non-convex)**: the loss surface is complex. You might need to make large updates late in training (e.g., to escape a saddle point or navigate a new region). AdaGrad's permanently decaying rate prevents this.

This is why RMSProp and Adam replaced AdaGrad for most deep learning tasks. They use a **decaying average** instead of a sum, so the effective learning rate can recover.

---

## Where AdaGrad Excels

Despite the decay problem, AdaGrad is the best choice in specific settings:

- **Sparse features**: NLP tasks with bag-of-words representations, where most features are zero. The word "the" gets gradients every batch, but the word "serendipity" gets gradients rarely. AdaGrad automatically gives "serendipity" a much larger learning rate, so it learns efficiently from its few gradient signals.
- **Recommender systems**: user-item interaction matrices are extremely sparse. Popular items have many interactions (many gradients), niche items have few. AdaGrad naturally balances them.
- **Convex problems**: for logistic regression, SVMs, and other convex losses, AdaGrad's convergence rate is theoretically optimal.
- **Online learning**: settings where data arrives one example at a time and you need to learn from each example immediately.

---

## AdaGrad's Legacy

Even though AdaGrad is rarely used directly in modern deep learning, every major optimizer builds on its idea:

- **RMSProp**: replaces AdaGrad's sum with a decaying average (fixes the decay problem)
- **Adam**: adds momentum on top of RMSProp's adaptive rates
- **AdamW**: fixes Adam's weight decay interaction
- **Nadam**: adds Nesterov momentum to Adam

Understanding AdaGrad is understanding the foundation of the entire adaptive optimizer family.