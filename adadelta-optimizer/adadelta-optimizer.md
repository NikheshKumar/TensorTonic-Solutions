## The Learning Rate Problem

Every optimizer we have seen so far requires you to set a **learning rate** $\eta$. This is arguably the most important and most frustrating hyperparameter in deep learning:

- **Too large**: training diverges, loss spikes to infinity or NaN
- **Too small**: training crawls, taking forever to converge
- **Depends on the problem**: the right $\eta$ for one architecture on one dataset can be completely wrong for another
- **Changes during training**: the ideal $\eta$ at the start (when you want big steps) is different from the ideal $\eta$ at the end (when you want to fine-tune)

Practitioners spend enormous effort tuning learning rates: grid search, learning rate finders, warmup schedules, decay schedules. What if we could eliminate the learning rate entirely?

---

## AdaDelta: No Learning Rate Required

AdaDelta (Zeiler, 2012) is designed to do exactly this. It computes the step size automatically by maintaining two running statistics, and the learning rate $\eta$ does not appear anywhere in its update formula.

The key idea: **use the ratio of the root-mean-square of recent parameter updates to the root-mean-square of recent gradients as the step size.**

---

## The Two Running Averages

AdaDelta maintains two exponentially decaying averages:

**1. Running average of squared gradients** $E[g^2]_t$:

$$
E[g^2]_t = \rho \cdot E[g^2]_{t-1} + (1 - \rho) \cdot g_t^2
$$

- This is identical to what RMSProp uses
- $\rho$ is the decay rate (typically 0.9 or 0.95)
- It tracks "how large have recent gradients been?"
- The root-mean-square is $\text{RMS}[g]_t = \sqrt{E[g^2]_t + \epsilon}$

**2. Running average of squared parameter updates** $E[\Delta w^2]_t$:

$$
E[\Delta w^2]_t = \rho \cdot E[\Delta w^2]_{t-1} + (1 - \rho) \cdot (\Delta w_t)^2
$$

- This tracks "how large have recent parameter updates been?"
- $\Delta w_t$ is the actual change applied to the parameter at step $t$
- The root-mean-square is $\text{RMS}[\Delta w]_t = \sqrt{E[\Delta w^2]_t + \epsilon}$
- This second accumulator is what makes AdaDelta unique

---

## The Update Rule

The parameter change at step $t$ is:

$$
\Delta w_t = -\frac{\text{RMS}[\Delta w]_{t-1}}{\text{RMS}[g]_t} \cdot g_t
$$

Expanding the RMS terms:

$$
\Delta w_t = -\frac{\sqrt{E[\Delta w^2]_{t-1} + \epsilon}}{\sqrt{E[g^2]_t + \epsilon}} \cdot g_t
$$

Then apply:

$$
w_t = w_{t-1} + \Delta w_t
$$

And update the second accumulator:

$$
E[\Delta w^2]_t = \rho \cdot E[\Delta w^2]_{t-1} + (1 - \rho) \cdot (\Delta w_t)^2
$$

Notice: **there is no learning rate $\eta$ anywhere.** The step size is entirely determined by the ratio of past update sizes to current gradient sizes.

---

## Why the Ratio Makes Sense

The numerator $\text{RMS}[\Delta w]_{t-1}$ captures "how much have we been changing this parameter?"

- If recent updates have been large, the numerator is large, allowing bigger steps
- If recent updates have been small, the numerator is small, producing smaller steps
- This creates a **self-calibrating** feedback loop

The denominator $\text{RMS}[g]_t$ captures "how large are the gradients?" (same as RMSProp)

- If gradients are large, the denominator is large, making steps smaller
- If gradients are small, the denominator is small, making steps larger
- This provides the **adaptive learning rate** behavior

Together, the ratio acts as an automatically-tuned learning rate that adjusts based on the training dynamics.

---

## The Bootstrap Issue

At $t = 0$, both accumulators are initialized to zero:
- $E[g^2]_0 = 0$
- $E[\Delta w^2]_0 = 0$

For the first update:
- The denominator updates immediately: $E[g^2]_1 = (1-\rho) g_1^2$ (nonzero)
- But the numerator uses $E[\Delta w^2]_0 = 0$ (because no updates have happened yet)
- So $\text{RMS}[\Delta w]_0 = \sqrt{0 + \epsilon} = \sqrt{\epsilon}$

The first update is approximately:

$$
\Delta w_1 \approx -\frac{\sqrt{\epsilon}}{\sqrt{(1-\rho) g_1^2 + \epsilon}} \cdot g_1
$$

This is very small (controlled by $\epsilon$, which is typically $10^{-6}$). AdaDelta starts with extremely cautious steps and gradually increases step size as the update accumulator fills in. This provides a natural **warmup** behavior.

---

## A Detailed Example

Parameters: $w = [2.0]$, accumulators: $E[g^2] = [0]$, $E[\Delta w^2] = [0]$, $\rho = 0.9$, $\epsilon = 10^{-6}$

**Step 1** with gradient $g = [1.0]$:

1. Update gradient accumulator:
   - $E[g^2] = 0.9 \times 0 + 0.1 \times 1.0 = 0.1$

2. Compute update:
   - Numerator: $\sqrt{E[\Delta w^2] + \epsilon} = \sqrt{0 + 10^{-6}} = 0.001$
   - Denominator: $\sqrt{E[g^2] + \epsilon} = \sqrt{0.1 + 10^{-6}} \approx 0.3162$
   - $\Delta w = -\frac{0.001}{0.3162} \times 1.0 \approx -0.00316$

3. Apply update:
   - $w = 2.0 + (-0.00316) = 1.99684$

4. Update parameter change accumulator:
   - $E[\Delta w^2] = 0.9 \times 0 + 0.1 \times (0.00316)^2 \approx 0.000001$

The first step is very small because the numerator starts from $\sqrt{\epsilon}$. Over subsequent steps, as $E[\Delta w^2]$ accumulates, the steps grow larger.

**Step 2** with gradient $g = [0.8]$:

1. $E[g^2] = 0.9 \times 0.1 + 0.1 \times 0.64 = 0.154$
2. Numerator: $\sqrt{0.000001 + 10^{-6}} \approx 0.00141$
3. Denominator: $\sqrt{0.154 + 10^{-6}} \approx 0.3924$
4. $\Delta w = -\frac{0.00141}{0.3924} \times 0.8 \approx -0.00288$

The steps are gradually increasing as the update accumulator builds up.

---

## Choosing Rho

$\rho$ is the only hyperparameter (besides $\epsilon$):

- **$\rho = 0.9$**: effective window of ~10 steps. The default in most implementations.
- **$\rho = 0.95$**: window of ~20 steps. Smoother estimates, slower adaptation.
- **$\rho = 0.99$**: window of ~100 steps. Very smooth, very slow to adapt.
- **Higher $\rho$** means slower warmup (the update accumulator fills in more slowly)

---

## AdaDelta vs. RMSProp vs. Adam

How AdaDelta compares to other adaptive optimizers:

- **RMSProp**: uses the same gradient accumulator, but requires a learning rate $\eta$. No momentum. Simpler to understand and tune.
- **AdaDelta**: replaces the learning rate with the update accumulator ratio. No $\eta$ to tune. But harder to control: you cannot just "increase the learning rate" if training is slow.
- **Adam**: uses both momentum and adaptive rates, requires $\eta$, includes bias correction. More features, more hyperparameters, generally more effective.

AdaDelta's main advantage is eliminating $\eta$. Its main disadvantage is limited control over step size. In practice, being able to tune $\eta$ (as in Adam) usually outweighs the convenience of not having one.

---

## Where AdaDelta Shows Up

- **Speech recognition**: AdaDelta was popular in early deep speech models before Adam became dominant
- **Natural language processing**: used in some early NLP deep learning work
- **When hyperparameter tuning is expensive**: if you truly cannot afford to tune a learning rate (e.g., very long training runs), AdaDelta provides a reasonable hands-off approach
- **Keras and TensorFlow**: available as `tf.keras.optimizers.Adadelta`
- **Pedagogically**: AdaDelta is important for understanding the idea that step sizes can be derived from training dynamics rather than set manually