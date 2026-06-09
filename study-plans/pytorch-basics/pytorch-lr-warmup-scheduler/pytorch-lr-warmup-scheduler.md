## Learning Rate Schedules in Deep Learning

- The learning rate is one of the most important hyperparameters in training neural networks
- It controls how large a step the optimizer takes when updating model weights
- A fixed learning rate often leads to suboptimal training: too high causes divergence, too low causes slow convergence
- Learning rate schedules dynamically adjust the rate during training to balance fast initial progress with fine-grained convergence later

## Why Warmup Matters

- At the start of training, model weights are randomly initialized and gradients can be noisy and large
- Using a high learning rate immediately can cause unstable updates that push the model into poor regions of the loss landscape
- Warmup addresses this by starting with a very small learning rate and gradually increasing it
- This gives the optimizer time to estimate reasonable gradient statistics before making large updates
- Warmup is especially critical for large batch training, where gradient noise is lower but individual updates have outsized impact
- Transformer-based architectures (BERT, GPT, ViT) almost universally rely on warmup for stable training

## Linear Warmup

- The simplest and most common warmup strategy increases the learning rate linearly from near-zero to the target base rate
- Over $warmup\_steps$ steps, the rate at step $t$ is:

$$
lr_t = base\_lr \times \frac{t + 1}{warmup\_steps}
$$

- The "+1" offset ensures the very first step uses a small but non-zero learning rate
- At step $warmup\_steps - 1$, the rate reaches exactly $base\_lr$, creating a smooth handoff to the decay phase
- Linear warmup is preferred over exponential or polynomial warmup due to its simplicity and consistent empirical performance

## Cosine Decay

- After warmup, the learning rate should decrease to allow the model to settle into a minimum
- Cosine annealing provides a smooth, non-linear decay that starts slowly, accelerates in the middle, and slows again near the end
- The formula for the decay phase is:

$$
lr_t = base\_lr \times 0.5 \times \left(1 + \cos\left(\pi \times \frac{t - warmup\_steps}{total\_steps - warmup\_steps}\right)\right)
$$

- At the start of decay, progress is 0, so $\cos(0) = 1$ and the rate equals $base\_lr$
- At the end of training, progress is 1, so $\cos(\pi) = -1$ and the rate reaches 0
- The gradual slowdown near the end is beneficial: the model makes increasingly fine adjustments as it approaches convergence
- Compared to step decay (abrupt drops at fixed epochs), cosine decay avoids sudden destabilization of training dynamics

## Combining Warmup and Cosine Decay

- The warmup-cosine schedule is a two-phase approach that combines the benefits of both strategies
- Phase 1 (warmup): linear ramp from near-zero to base_lr over the first few steps
- Phase 2 (decay): smooth cosine annealing from base_lr down to zero over the remaining steps
- The transition between phases is continuous, meaning there is no discontinuity in the learning rate curve
- This combination has become the standard schedule for training transformers, vision models, and many other architectures

## Practical Considerations

- Typical warmup duration is 1 to 10 percent of total training steps
- Too short a warmup may not sufficiently stabilize early training
- Too long a warmup wastes training budget on suboptimal learning rates
- The base learning rate should be tuned in conjunction with the warmup duration
- Some variants add a minimum learning rate floor instead of decaying all the way to zero
- Other variants include cosine annealing with restarts, where the schedule repeats multiple times during training
- The schedule is computed per optimization step, not per epoch, to maintain consistency across different dataset sizes

## Historical Context

- Cosine annealing was introduced in the SGDR paper (Loshchilov and Hutter, 2017) for stochastic gradient descent with warm restarts
- Linear warmup gained prominence with the original Transformer paper (Vaswani et al., 2017), which used warmup with inverse square root decay
- The combination of linear warmup and cosine decay became widespread with BERT (Devlin et al., 2019) and subsequent language model training recipes
- Modern frameworks like PyTorch provide built-in schedulers such as CosineAnnealingLR and LinearLR that can be chained using SequentialLR to implement this pattern