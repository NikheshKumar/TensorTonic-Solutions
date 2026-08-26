# SiTU-GLU

SiTU-GLU is a gated feed-forward activation designed to behave like a familiar smooth GLU near ordinary values while preventing very large activations. It applies a smooth cap to both multiplicative branches, so neither branch can make their product grow without limit.

## Begin with a gated activation

A gated linear unit has two learned projections of the same input. One branch decides what should pass, while the other carries transformed content. The two branches are multiplied element by element.

SwiGLU uses a Swish-style gate branch and a linear up branch. This works well in many transformers, but both branches can grow with large positive inputs. Their product can therefore become very large, especially inside a deep routed expert path.

SiTU-GLU keeps the same general structure while smoothly limiting the two branches:

$$
\left[\beta_1\tanh\left(\frac{g}{\beta_1}\right)\odot\operatorname{sigmoid}(g)\right]
\odot
\left[\beta_2\tanh\left(\frac{u}{\beta_2}\right)\right]
$$

Here $g$ is the gate projection, $u$ is the up projection, and the positive values $\beta_1$ and $\beta_2$ control the caps of the two branches.

## Why scaled tanh is useful

The expression $\beta\tanh(x/\beta)$ has two helpful behaviors.

Near zero, tanh is approximately its input, so

$$
\beta\tanh(x/\beta) \approx x
$$

This means SiTU-GLU stays close to the uncapped activation for ordinary small values. At large magnitude, tanh approaches either $1$ or $-1$, so the scaled result approaches either $\beta$ or $-\beta$. The transition is smooth rather than a hard clipping boundary.

The sigmoid remains on the gate branch. For a large negative gate value, sigmoid moves toward zero, preserving the vanishing negative behavior of Swish. For a large positive gate value, sigmoid approaches one while the scaled tanh supplies the cap.

## The two caps are independent

The gate branch uses $\beta_1$, and the up branch uses $\beta_2$. Do not reuse one cap for both unless the arguments actually contain the same number.

Because the absolute gate branch is below $\beta_1$ and the absolute up branch is below $\beta_2$, the absolute product is bounded by $\beta_1\beta_2$. Kimi K3 uses gate and up caps of $4$ and $25$, giving a coordinate-wise bound of $100$.

This bound is the purpose of the operation, but it should not distract from the implementation. Project the input twice, transform each branch exactly as stated, and multiply the results element by element.

## A scalar example

Suppose both projected values are $2$, with gate cap $4$ and up cap $25$.

The gate branch is

$$
4\tanh(2/4)\operatorname{sigmoid}(2)
$$

which is approximately $4(0.4621)(0.8808)=1.628$. The up branch is

$$
25\tanh(2/25)
$$

which is approximately $1.996$. Their product is about $3.25$.

For values near two, the caps have only a modest effect. If the projections become extremely large, the branches approach their fixed limits instead of continuing to grow. This is the balance SiTU-GLU is designed to provide.

## Applying the projections

The supplied projection tensors map the input's final feature dimension to the expert feature dimension. Use the same matrix orientation described by the prompt. Leading dimensions such as batch and sequence are simply carried through, so the activation is applied independently at every token position.

No reduction is involved. Every output coordinate comes from the matching gate and up coordinates after their projections and nonlinearities.

## Implementation order

- Apply the gate projection and up projection to the input.
- Divide the gate projection by its cap, apply tanh, and multiply back by that cap.
- Multiply the capped gate branch by sigmoid of the original gate projection.
- Smoothly cap the up projection with its own cap.
- Multiply the two transformed branches element by element and return the result.

## Common mistakes to avoid

- **Applying sigmoid to the up branch.** Sigmoid belongs only to the gate projection.
- **Using hard clipping.** The task requires scaled tanh, which has different values and gradients.
- **Sharing the cap values.** Gate and up caps are independent.
- **Applying tanh before dividing by the cap.** The correct form is $\beta\tanh(x/\beta)$.
- **Multiplying before the nonlinearities.** Transform each branch first, then combine them.
