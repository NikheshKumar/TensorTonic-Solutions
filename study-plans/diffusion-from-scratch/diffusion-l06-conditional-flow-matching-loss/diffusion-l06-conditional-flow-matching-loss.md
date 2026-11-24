# Conditional Flow Matching Loss

Flow matching teaches a model how to move a point, little by little, between random noise and realistic data. Think of a pile of sand where every grain is given its own direction and speed, so that the whole pile slowly reshapes into a sculpture. A generator that knows these directions can start from noise and follow them to new data, but nobody knows the right motion for the whole crowd of points in advance. The surprise is that straight lines between single data and noise samples are enough to learn it. This page builds the idea from one point moving along one line and ends with the batch loss for a small velocity model that you implement in this problem.

## Moving one point at constant speed

Start with a single number. A data value $x_{\text{data}} = 2$ and a noise value $x_{\text{noise}} = -1$ are joined by a straight line, and a time $t$ between 0 and 1 says how far along the line a point has travelled. In this plan time 0 is the data end and time 1 is the noise end, so the point at time $t$ is

$$
x_t = (1 - t)\, x_{\text{data}} + t\, x_{\text{noise}} .
$$

At $t = 0$ this gives the data value 2, at $t = 1$ it gives the noise value $-1$, and at $t = 0.3$ it gives $0.7 \cdot 2 + 0.3 \cdot (-1) = 1.1$. The point moves the same distance in every equal slice of time, so its speed is constant.

The velocity is the derivative of the position with respect to time.

1. The derivative of $(1 - t)\, x_{\text{data}}$ with respect to $t$ is $-x_{\text{data}}$.
2. The derivative of $t\, x_{\text{noise}}$ with respect to $t$ is $x_{\text{noise}}$.
3. Adding them gives the velocity $u = x_{\text{noise}} - x_{\text{data}}$, which does not depend on $t$.

In the example the velocity is $-1 - 2 = -3$. The point travels three units to the left over one unit of time, which matches the distance from 2 to $-1$.

## A velocity field moves a whole cloud of points

A generator has to move many points at once, and they all start from different places. A velocity field $v(x, t)$ assigns a velocity to every position $x$ at every time $t$, and the points follow it through the ordinary differential equation

$$
\frac{dx}{dt} = v(x, t) .
$$

If the field is chosen well, a cloud of points that starts as Gaussian noise at $t = 1$ and follows the field backwards in time ends at $t = 0$ distributed exactly like the data. Sampling then needs only an ODE solver. The simplest solver is the Euler step $x \leftarrow x - \Delta t \, v(x, t)$, where the minus sign appears because time runs from 1 down towards 0. With the exact velocity $-3$ from the example, a single Euler step of size 1 carries the noise value $-1$ to $-1 - 1 \cdot (-3) = 2$, the data value.

Continuous normalizing flows used the same picture earlier, but they trained the field by simulating the ODE during training, which is very slow. Flow matching avoids simulation entirely.

## The field for the whole dataset is unknown

The ideal loss would compare a model $v_\theta(x, t)$ with the true field of a flow that carries noise to data. That field is defined by an average. Many data and noise pairs can have straight paths that pass through the same point at the same time, and the true velocity there is the average of all their velocities. Computing it requires the whole dataset at every point, so regressing onto it directly is impossible.

## Training on single pairs is enough

Lipman et al. (2023) and Liu et al. (2023) showed that it is enough to regress onto the velocity of one pair at a time. Draw a data sample, a noise sample and a time, build the point $x_t$ on their line, and ask the model for the velocity $x_{\text{noise}} - x_{\text{data}}$ at that point. The loss is

$$
\mathcal L = \mathbb E_{t,\, x_{\text{data}},\, x_{\text{noise}}} \left\lVert v_\theta(x_t, t) - (x_{\text{noise}} - x_{\text{data}}) \right\rVert^2 .
$$

A small example shows why this works. Pair A joins data 1 to noise $-1$, and pair B joins data $-1$ to noise 1. At $t = 0.5$ both paths pass through 0, with velocities $-2$ and $+2$. A model sees the same input from both pairs and can give only one answer $c$. The total squared error $(c + 2)^2 + (c - 2)^2$ is smallest at $c = 0$, the average of the two velocities.

This holds in general, because the minimizer of a squared error is the average of the targets. A model trained on single-pair targets therefore learns the average velocity at each point, which is exactly the true field. Lipman et al. proved that the gradient of this conditional loss equals the gradient of the intractable one.

## Data at time zero, noise at time one

This problem follows the convention of Esser et al. (2024) and of many diffusion codebases, with data at $t = 0$ and noise at $t = 1$. The target then points from data towards noise, and sampling runs backwards in time as described above.

Lipman et al. and Liu et al. use the opposite convention, with noise at $t = 0$ and data at $t = 1$, which flips the sign of the target. Either convention works on its own. Mixing them, for example interpolating one way and building the target the other way, trains a velocity that points the wrong way and gives a sampler that walks away from the data.

## The toy velocity model and the batch loss

This problem replaces the network with a linear model that sees both the position and the time. The point $x_t$ is a row of length D, $W$ is a weight of shape $(D, D)$, $w_{\text{time}}$ is the time weight of shape $(D)$ and $b$ is a bias of shape $(D)$.

$$
v = x_t W + t\, w_{\text{time}} + b
$$

The time term lets the predicted velocity change along the path, which a real network does through a timestep embedding. The per-sample loss $\ell$ is the mean of the squared error over the D features, and the batch loss is the mean of $\ell$ over the B samples. Averaging over features keeps the loss on the same scale whatever D is.

In a batch, x_data and x_noise have shape $(B, D)$ and t has shape $(B)$, one time per row. Each time must scale every feature of its own row, so the $B$ values have to line up with the batch axis before they meet a $(B, D)$ tensor. The same alignment applies to the time term of the model.

## A two-feature worked example

Use $x_{\text{data}} = (1.5, -0.5)$, $x_{\text{noise}} = (-0.5, 1.0)$ and $t = 0.4$, with $W$ having rows $(0.2, 0.0)$ and $(0.1, -0.3)$, $w_{\text{time}} = (0.5, 1.0)$ and $b = (0.0, -0.2)$.

1. The point on the path is $x_t = 0.6 \cdot (1.5, -0.5) + 0.4 \cdot (-0.5, 1.0) = (0.7, 0.1)$.
2. The target velocity is $(-0.5, 1.0) - (1.5, -0.5) = (-2.0, 1.5)$.
3. The product $x_t W$ is $(0.7 \cdot 0.2 + 0.1 \cdot 0.1,\ 0.7 \cdot 0 + 0.1 \cdot (-0.3)) = (0.15, -0.03)$.
4. Adding $0.4 \cdot (0.5, 1.0) = (0.2, 0.4)$ and the bias gives the prediction $v = (0.35, 0.17)$.
5. The error is $(2.35, -1.33)$, the squared errors are $5.5225$ and $1.7689$, and their mean is $\ell = 3.6457$.

The model is badly wrong on the first feature, where it predicts a small positive velocity while the path moves quickly in the negative direction. Training would push $W$, $w_{\text{time}}$ and $b$ to close that gap across many pairs.

## How this relates to diffusion

Straight-path flow matching is closely related to diffusion. The interpolation is a forward noising process whose signal factor is $1 - t$ and whose noise factor is $t$, and the velocity target is a fixed linear combination of the noise and the clean sample. A network that predicts the velocity can therefore be converted into one that predicts either of them. Esser et al. scaled this objective to large text to image models.

The loss involves only interpolation and squared errors, so it is numerically benign. The endpoints $t = 0$ and $t = 1$ are valid inputs, where $x_t$ equals the data or the noise exactly.

## Common mistakes to avoid

- **Flipping the target to data minus noise.** With data at time 0 the path moves towards the noise, and the reversed target trains a field that pushes the sampler away from the data.
- **Swapping the interpolation weights.** Writing $t\, x_{\text{data}} + (1 - t)\, x_{\text{noise}}$ places the data at time 1, which contradicts both the target and the time given to the model.
- **Evaluating the model on the clean data.** The velocity must be predicted at the interpolated point $x_t$, since that is the only kind of point the sampler will ever see.
- **Summing the squared error over the features.** The per-sample loss is a mean over D, so a sum scales every value by the feature count.
- **Broadcasting t along the wrong axis.** A $(B)$ time vector combined with a $(B, D)$ tensor without a reshape either fails or pairs times with features instead of samples.
- **Transposing the weight matrix.** The model multiplies the row $x_t$ by $W$ on the right, and $W^\top$ gives a different velocity whenever $W$ is not symmetric.