# Estimate Critical Batch Size

Doubling the batch size halves the number of training steps only up to a point. Beyond it, extra examples per step stop buying fewer steps, and the additional compute is mostly wasted. The heart of this problem is estimating that turning point, the critical batch size, from a sweep of training runs that each record how many steps they needed to reach a target loss.

## Why bigger batches help, then stop helping

Each optimizer step uses a gradient estimated from a batch of examples. With a small batch the estimate is noisy, so the optimizer zigzags and needs many steps. A larger batch averages out more noise, which allows bigger, more reliable steps, and the number of steps to reach a given loss falls.

Once the batch is large enough that noise is no longer the main obstacle, adding more examples barely improves the gradient. The step count levels off at some minimum, while the number of examples processed keeps growing. Large-scale training often aims near the batch size where this trade-off turns, because a run there needs only about twice the minimum number of steps while using only about twice the minimum data, a balanced compromise between wall-clock time and compute.

## Every run trades steps for examples

For sweep point $i$ with batch size $B_i$ that reached the target in $S_i$ steps, the number of training examples it consumed is

$$
E_i = B_i S_i
$$

Small batches use few examples but many steps, and large batches use few steps but many examples, so no single run minimizes both.

Some runs in a sweep may never reach the target, for example because they diverged or ran out of budget. Those points have no meaningful step or example count, so they are left out of the calculation.

## Dividing the two minima

Let $S_{\min}$ be the fewest steps any reached run needed and $E_{\min}$ the fewest examples any reached run consumed. These two minima usually come from opposite ends of the sweep. Their ratio is the estimate

$$
B_{\text{crit}} = \frac{E_{\min}}{S_{\min}}
$$

This comes from the empirical model of McCandlish et al. (2018). In that model $S_{\min}$ and $E_{\min}$ are limits that an infinitely large or infinitely small batch would reach, and the steps $S$ and examples $E$ of any one run trade off as

$$
\left( \frac{S}{S_{\min}} - 1 \right) \left( \frac{E}{E_{\min}} - 1 \right) = 1
$$

The balanced point on this curve is the run that needs twice the minimum steps and twice the minimum examples. Its batch size is $2E_{\min} / 2S_{\min} = E_{\min} / S_{\min}$, which is why the ratio of the two minima is called the critical batch size. Below it, larger batches save steps almost for free, and above it they cost examples without saving many steps.

A finite sweep only observes values above the true limits, so its smallest observed counts serve as estimates of them. Real sweeps, like the made-up one below, need not follow the curve exactly.

## A sweep with one failed run

Take batch sizes 8, 32, 128 and 256, where the first three reached the target in 1000, 400 and 300 steps and the run at 256 did not reach it.

The examples consumed are $8 \times 1000 = 8000$, $32 \times 400 = 12{,}800$ and $128 \times 300 = 38{,}400$, and the failed run has no value. The fewest steps are 300, from batch size 128, and the fewest examples are 8000, from batch size 8. The estimate is

$$
B_{\text{crit}} = \frac{8000}{300} \approx 26.67
$$

The run at batch size 32 sits close to it, using 1.6 times the minimum examples and about 1.3 times the minimum steps. Going to 128 cut steps by only a further quarter while tripling the data used.

## How critical batch size changes during training

The critical batch size is not fixed for a model. It tends to grow as the loss falls, because the true gradient shrinks relative to the noise in each example's gradient, so more examples are needed to average the noise away. This is one reason large training runs often increase the batch size as training goes on.

## Common mistakes to avoid

- **Taking both minima from the same run.** The fewest steps and the fewest examples are found separately and usually come from different batch sizes.
- **Including runs that never reached the target.** Unreached points are excluded from both minima.
- **Treating a failed run's step budget as its step count.** A diverged run never reached the target, so its step count says nothing about how fast that batch size learns.
- **Dividing steps by examples.** The result is the reciprocal of a batch size, a tiny number with units of steps per example.
- **Using the largest batch as the answer.** The largest batch minimizes steps but usually wastes data.
