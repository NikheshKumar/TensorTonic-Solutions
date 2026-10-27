# Diagnose muP Width Invariants

The maximal-update parameterization (muP) is a way of scaling initialization and learning rates with model width so that hyperparameters tuned on a small model still work on a large one. This works only if activations and their per-step changes stay the same size as width grows. The heart of this problem is measuring that property with a log-log fit for each quantity and checking whether its slope is close enough to zero.

## Why width breaks hyperparameter transfer

In the usual setup every layer shares one learning rate. Each output of a layer sums over $w$ inputs, so when every weight moves by a similar amount, the change in that sum grows with $w$, and the same learning rate makes wider models take larger effective steps. A learning rate that is ideal at width 256 can then be far too large at width 4096, so every new model size needs its own expensive hyperparameter search.

muP was introduced by Yang and Hu in Tensor Programs IV (2021), and Yang et al. (2022) showed in Tensor Programs V that it lets hyperparameters transfer across widths. It chooses width-dependent multipliers for initialization, learning rates and outputs so that two things stay roughly constant as width grows. Each activation keeps the same typical size, and the change in each activation after one optimizer step also keeps the same typical size. Under those conditions the best learning rate found on a narrow model transfers to a wide one.

## Measuring with root mean square

The typical size of an activation tensor is measured by its root mean square (RMS), the square root of the mean of its squared entries. The diagnostic takes two series measured at several widths $w_i$. The first is the activation RMS, and the second is the RMS of the change in activations after one update.

A practical muP setup is checked by training a few steps at several widths and plotting these values, sometimes called a coordinate check. A correct setup gives flat lines, while a bug shows up as values that grow or shrink with width.

## Fitting a slope in log space

If a quantity behaves like $y \propto w^b$, it is a straight line on a log-log plot with slope $b$. For each series, an ordinary least-squares line is fitted in natural logs:

$$
\log y_i = a + b \log w_i
$$

where $y_i$ is the measured RMS at width $w_i$, $a$ is the intercept and $b$ is the slope. A slope of 0 means the quantity is width-invariant, a slope of 0.5 means it grows like $\sqrt{w}$, and a negative slope means it shrinks. The slope is the same in any log base, but the intercept is not, so the natural log matters for the intercept.

Measurements are noisy, so a series counts as width-invariant when $|b|$ is at most a chosen tolerance $\tau$. The two series are judged separately, since a setup can get one right and the other wrong.

The ratio of the largest to smallest measured value, $\max_i y_i / \min_i y_i$, catches spread that a slope can hide, such as one width with a spike while the fitted line stays flat.

## A diagnosis with one bug

Take widths 256, 1024, 4096 and 16,384, activation RMS values 0.5, 1, 2 and 4, change RMS values 0.1, 0.104, 0.098 and 0.101, and a tolerance of 0.05.

The activation RMS doubles every time the width grows fourfold, so its fitted slope is

$$
b = \frac{\log 2}{\log 4} = 0.5
$$

with intercept about $-3.466$ and a max-to-min ratio of 8. The slope is far above 0.05, so the activations are not width-invariant. They grow like $\sqrt{w}$, which is what happens, for example, when initialization variance is not scaled down with the number of inputs to each unit.

The change RMS fits a slope of about $-0.0021$, with intercept about $-2.279$ and ratio about 1.061. That slope is well within the tolerance, so the updates are width-invariant. The diagnosis suggests the forward scale, such as the initialization or a width multiplier, is wrong, while the per-step update size already behaves as muP intends.

## Common mistakes to avoid

- **Fitting on raw values.** The slope comes from a line between the natural logs of width and RMS.
- **Testing the ratio against the tolerance.** The invariance decision uses the fitted slope, and the ratio is a separate measure of spread.
- **Checking only the slope's sign.** Large negative slopes fail the test just like large positive ones.
- **Combining the two series.** Activations and their changes are fitted and judged independently.
- **Using log base 10.** The slope is unchanged, but the intercept comes out different from the natural-log fit.
