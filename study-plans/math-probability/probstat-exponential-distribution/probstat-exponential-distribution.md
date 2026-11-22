## <span style="font-size: 20px;">Exponential Distribution</span>

The exponential distribution is the continuous probability distribution that models the time between events in a Poisson process. It is the simplest continuous distribution for modeling waiting times and durations.

### Probability Density Function

A random variable $T \sim \text{Exponential}(\lambda)$ has the PDF:

$$f(t) = \lambda e^{-\lambda t}, \quad t \geq 0$$

where $\lambda > 0$ is the **rate parameter** (the average number of events per unit time). The PDF starts at $\lambda$ when $t=0$ and decays exponentially toward zero. Higher $\lambda$ means events occur more frequently, so the distribution is concentrated near zero.

### Cumulative Distribution Function

The CDF gives the probability that the event occurs by time $t$:

$$F(t) = P(T \leq t) = 1 - e^{-\lambda t}$$

The CDF increases from 0 toward 1 as $t \to \infty$. At $t = 1/\lambda$ (the mean), $F(1/\lambda) = 1 - e^{-1} \approx 0.632$, meaning about 63.2% of events occur before the mean waiting time.

### Survival Function

The **survival function** gives the probability of surviving (no event) beyond time $t$:

$$S(t) = P(T > t) = 1 - F(t) = e^{-\lambda t}$$

This is simply the complement of the CDF. The name comes from reliability engineering and medical statistics, where $S(t)$ represents the probability that a component or patient survives past time $t$.

### Mean and Variance

$$E[T] = \frac{1}{\lambda} \qquad \text{Var}(T) = \frac{1}{\lambda^2}$$

The mean waiting time is the reciprocal of the rate. If events occur at rate $\lambda = 2$ per hour, the average wait is $1/2$ hour = 30 minutes. The standard deviation equals the mean ($\sigma = 1/\lambda = \mu$), a unique property of the exponential distribution.

### The Memoryless Property

The exponential distribution is the **only** continuous distribution with the memoryless property:

$$P(T > s + t \mid T > s) = P(T > t)$$

This means that having already waited $s$ time units gives no information about how much longer you will wait. The remaining wait time has the same distribution as the original. Proof:

$$P(T > s + t \mid T > s) = \frac{P(T > s + t)}{P(T > s)} = \frac{e^{-\lambda(s+t)}}{e^{-\lambda s}} = e^{-\lambda t} = P(T > t)$$

This is both a strength (mathematical convenience) and a limitation (many real processes have memory).

### Connection to the Poisson Process

The exponential and Poisson distributions are deeply connected through the Poisson process:

- If events arrive according to a Poisson process with rate $\lambda$, the number of events in time $t$ follows $\text{Poisson}(\lambda t)$.
- The time between consecutive events follows $\text{Exponential}(\lambda)$.
- The time until the $k$-th event follows $\text{Gamma}(k, \lambda)$.

### Hazard Rate

The **hazard rate** (or failure rate) is:

$$h(t) = \frac{f(t)}{S(t)} = \lambda$$

For the exponential distribution, the hazard rate is constant - the risk of failure does not change over time. This corresponds to the memoryless property. In contrast, the Weibull distribution generalizes the exponential by allowing increasing or decreasing hazard rates.

### Applications in Machine Learning

**Survival analysis**: Modeling time-to-event data (customer churn, equipment failure, disease progression). The exponential distribution serves as a baseline model, and more flexible models like Cox proportional hazards extend it by allowing covariate-dependent rates.

**Modeling time between events**: Web request inter-arrival times, user session durations, and time between anomalies are often modeled with exponential distributions. Deviations from the exponential model can signal non-stationarity or dependencies.

**System reliability**: In distributed ML systems, component failure times are often modeled as exponential. The probability that a system of $n$ independent components all survive past time $t$ is $e^{-n\lambda t}$, showing that system reliability decreases with the number of components.

**Queueing models**: Request processing in ML serving systems uses exponential service times. The M/M/1 queue (Poisson arrivals, exponential service) gives closed-form results for latency, throughput, and queue length.