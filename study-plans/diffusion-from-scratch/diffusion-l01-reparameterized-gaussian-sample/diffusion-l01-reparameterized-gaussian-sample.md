# Reparameterized Gaussian Sample

Picture drawing a random height from a large population. You could ask for a height directly, or you could draw a standard score first and then shift it by the average height and stretch it by the spread. The two routes give samples with the same distribution, but only the second one is made of plain arithmetic that a neural network can differentiate. Diffusion models use this second route every time they mix noise into data, and this page builds it step by step, from one number with a known spread to a whole array of samples described by a mean and a log variance, which is what you implement here.

## Why a random draw blocks gradients

Suppose a network outputs a mean $\mu$ and a variance $\sigma^2$, a sample $z$ is drawn from the Gaussian $\mathcal N(\mu, \sigma^2)$ with that mean and variance, and a loss is computed from $z$. Training needs the derivative of the loss with respect to $\mu$ and $\sigma^2$, so that gradient descent knows how to move them.

If $z$ comes from a function that simply returns a random number for the given parameters, there is no formula that links $z$ to $\mu$ and $\sigma^2$. Automatic differentiation records arithmetic operations, and a black-box random draw is not one of them, so the gradient path stops at the sample.

One workaround, the score function estimator, uses the gradient of the log density instead of the gradient of the sample. It gives the right answer on average, but its estimates vary so much from batch to batch that training becomes slow and noisy. Kingma and Welling (2014) pointed out that Gaussians allow a much better option.

## Shifting and stretching one standard draw

Start with the simplest Gaussian of all. Let $\epsilon$ be one number drawn from the standard Gaussian $\mathcal N(0, 1)$, which has mean 0 and variance 1. Multiplying a random number by a constant multiplies its standard deviation by that constant, and adding a constant moves its mean by that constant. So if $\sigma$ is a standard deviation,

$$
z = \mu + \sigma \epsilon
$$

has mean $\mu$ and variance $\sigma^2$. Take $\mu = 3$ and $\sigma = 2$. A noise draw of $\epsilon = 0.5$ gives $z = 3 + 2 \cdot 0.5 = 4$, and a draw of $\epsilon = -1$ gives $z = 1$. Every standard draw is mapped to the target Gaussian by the same shift and stretch.

The important change is where the randomness lives. It sits entirely in $\epsilon$, and $\epsilon$ does not depend on any parameter. Once $\epsilon$ has been drawn, $z$ is an ordinary deterministic function of $\mu$ and $\sigma$.

## The gradients become simple

Because $z$ is now plain arithmetic, its derivatives can be read off directly.

$$
\frac{\partial z}{\partial \mu} = 1, \qquad \frac{\partial z}{\partial \sigma} = \epsilon
$$

Moving the mean moves the sample by the same amount, and widening the Gaussian pushes the sample further in the direction its noise already pointed. In the example above with $\epsilon = 0.5$, increasing $\sigma$ from 2 to 2.1 moves the sample from 4 to 4.05, which matches a derivative of 0.5.

Backpropagation can now pass through the sample into whatever produced $\mu$ and $\sigma$. This substitution is called the reparameterization trick, because it rewrites the same random quantity with a parameterization that gradients can flow through.

## From one coordinate to a diagonal Gaussian

Real data has many coordinates, such as the pixels of an image or the entries of a latent vector. A diagonal Gaussian in $D$ dimensions treats every coordinate as an independent one-dimensional Gaussian with its own mean and its own variance. Its mean $\mu$ and standard deviation $\sigma$ are vectors of length $D$, and the noise $\epsilon$ is a vector of $D$ independent standard Gaussian values.

The shift and stretch is then applied coordinate by coordinate. Writing $\odot$ for elementwise multiplication, the sample is $z = \mu + \sigma \odot \epsilon$. No coordinate affects any other, so the whole operation is a single elementwise expression.

## Why networks predict the log variance

A variance must be positive, but the output of a linear layer can be any real number. Networks therefore predict the natural logarithm of the variance, $\log \sigma^2$, which can take any real value. Its exponential is always positive, so every possible output describes a valid Gaussian.

The log scale also makes tiny and huge variances equally easy to express. A variance of $10^{-4}$ corresponds to a log variance of about $-9.2103$, and a variance of 100 corresponds to about $4.6052$, so neither needs an extreme activation.

Turning a log variance into a standard deviation takes one step. Exponentiating recovers the variance, $\sigma^2 = \exp(\log \sigma^2)$, and a square root halves the exponent.

$$
\sigma = \exp\!\left(\tfrac{1}{2} \log \sigma^2\right)
$$

Computing the standard deviation in this form never builds the variance as an intermediate value, so a very small or very large variance cannot underflow or overflow before the square root is taken. The gradient with respect to the log variance follows from the chain rule, $\partial z / \partial \log \sigma^2 = \tfrac{1}{2} \sigma \epsilon$.

## A worked example with a log variance

Take one coordinate with mean $\mu = -1$ and variance 9, so the log variance is $\log 9 \approx 2.1972$. Half of that is about $1.0986$, and its exponential gives the standard deviation $\sigma = 3$.

With the noise value $\epsilon = -0.4$, the sample is $z = -1 + 3 \cdot (-0.4) = -2.2$. The gradient with respect to the mean is 1, and the gradient with respect to the log variance is $\tfrac{1}{2} \cdot 3 \cdot (-0.4) = -0.6$. A slightly larger log variance would push the sample further below the mean, which agrees with the negative sign.

Now extend it to two coordinates. Let the means be $0.5$ and $-1$ and the variances be 4 and $0.01$, so the log variances are about $1.3863$ and $-4.6052$ and the standard deviations are 2 and $0.1$. A noise vector of $(1, -2)$ gives the sample $(0.5 + 2 \cdot 1, \ -1 + 0.1 \cdot (-2)) = (2.5, -1.2)$. The wide first coordinate moves far from its mean, while the narrow second coordinate barely moves.

## Broadcasting one Gaussian over many draws

Often one set of parameters is paired with many noise draws, for example to estimate an expected loss by averaging over $N$ samples. If $\mu$ and $\log \sigma^2$ have shape $(D,)$ and the noise has shape $(N, D)$, broadcasting lines up the last axes and produces $N$ samples of shape $(N, D)$ in one expression.

Continuing the two-coordinate example, a second noise row of $(-0.5, 3)$ gives the sample $(-0.5, -0.7)$. Stacking both noise rows into a $(2, 2)$ array returns both samples together. The same broadcasting rule also covers the opposite case, where a batch of $N$ different Gaussians is paired with noise of the same $(N, D)$ shape.

## How diffusion reuses the trick

The forward process of a diffusion model defines the noisy sample at timestep $t$ as a Gaussian, $q(x_t \mid x_0) = \mathcal N(\sqrt{\bar\alpha_t}\, x_0, (1 - \bar\alpha_t) I)$. Here $x_0$ is the clean data, $\bar\alpha_t$ is the fraction of the signal variance that survives up to step $t$, and $I$ is the identity matrix, so every coordinate gets the same variance. Ho et al. (2020) never sample this Gaussian with a black-box call. They write

$$
x_t = \sqrt{\bar\alpha_t}\, x_0 + \sqrt{1 - \bar\alpha_t}\, \epsilon
$$

which is the reparameterization formula with mean $\sqrt{\bar\alpha_t}\, x_0$ and standard deviation $\sqrt{1 - \bar\alpha_t}$.

Keeping $\epsilon$ as an explicit input is what makes the usual training loss possible. The noise that produced $x_t$ is known exactly, so the network can be trained to predict it. With $\bar\alpha_t = 0.81$, the log variance is $\log 0.19 \approx -1.6607$, the standard deviation is about $0.4359$, and a pixel with $x_0 = 1.5$ and $\epsilon = 0.5$ becomes $0.9 \cdot 1.5 + 0.4359 \cdot 0.5 \approx 1.5679$. The sampler later uses the same form again, adding fresh noise scaled by a known standard deviation at each reverse step.

## Numerical range of the log variance

A log variance of $-16$ gives a standard deviation of about $3.35 \times 10^{-4}$, so the sample is essentially the mean. A log variance of $5$ gives a standard deviation of about $12.18$. Computing $\exp(\tfrac{1}{2} \log \sigma^2)$ directly keeps both ends accurate. Models that predict log variances often clamp them to a range like this, so that a single exploding output cannot produce infinite samples.

## Common mistakes to avoid

- **Using the exponential of the log variance as the scale.** That multiplies the noise by the variance instead of the standard deviation, so wide Gaussians spread far too much and narrow ones far too little.
- **Flipping the sign of the exponent.** The factor $\exp(-\tfrac{1}{2} \log \sigma^2)$ is one over the standard deviation, so wide Gaussians produce narrow samples and the other way round.
- **Taking the square root of the log variance.** The log variance can be negative, and its square root is either undefined or a number with no meaning as a spread.
- **Writing into the noise array in place.** The same noise is often reused as a regression target, and overwriting it silently corrupts that later use.
- **Drawing fresh noise inside the function.** A hidden random call makes the result unrepeatable and breaks any training objective that needs the exact noise that was used.
- **Reshaping the inputs instead of broadcasting them.** Forcing every array to one fixed shape fails when a single mean of shape (D,) is paired with a batch of noise.