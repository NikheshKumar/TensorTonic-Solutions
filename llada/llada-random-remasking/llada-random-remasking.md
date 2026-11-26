# Random Remasking

Random remasking is the sampler that follows LLaDA's theory most closely. It turns a fully masked response into text by running the forward masking process in reverse, one time step at a time, with a mask predictor that guesses every masked token at once. This problem implements Algorithm 4 of Nie et al., 2025, where each fresh guess is sent back to the mask with probability $s/t$.

## Running the masking process backward

Let $x_0$ be a clean sequence of $L$ tokens and $M$ the mask token. LLaDA's forward process at time $t \in [0, 1]$ masks each token independently with probability $t$ and keeps it with probability $1 - t$ (Equation 8). This is the key difference from continuous diffusion such as DDPM, which adds Gaussian noise and trains a network to predict that noise, while LLaDA replaces tokens with a mask, predicts the clean tokens with a softmax, and needs no timestep input.

The reverse transition from time $t$ to an earlier time $s < t$ factorizes over positions (Equations 9 and 10). A position that is already a real token stays as it is. A masked position stays masked with probability $s/t$, and with probability $(t-s)/t$ it becomes a real token drawn from the predictor's distribution over clean tokens. Algorithm 4 makes this concrete with $N$ equal steps from $t = 1$ down to $t = 0$, where step $k$ moves from $t_k = (N-k)/N$ to $s_k = (N-k-1)/N$.

## Why the ratio s over t keeps the schedule on track

The ratio is what keeps sampling consistent with training. Suppose a fraction $t$ of the response is still masked at time $t$. Each masked position survives the step with probability $s/t$, so the expected masked fraction after the step is $t \cdot s/t = s$. The sampler therefore visits states whose masked fraction matches the forward process at every time, which is exactly the distribution of states the predictor saw during training.

The ratio $\rho_k = s_k/t_k$ simplifies to $(N-k-1)/(N-k)$. It starts near one when many tokens are masked and only a few should be revealed, and it falls to zero at the last step, so the final step always reveals whatever is left. Because the remask decision is random, the number of tokens revealed per step varies, and some steps reveal nothing.

## What this exercise fixes

The paper uses greedy prediction in Algorithm 4, so each guess is the argmax of the logits and no temperature is involved. The exercise supplies the predictor's logits for every step as a fixed script and supplies the uniform draws that decide each remask, which makes the sampler deterministic without changing the algorithm. The official generate.py has a random remasking option that draws a random confidence per position and reveals a fixed number of tokens per step. That option is a different algorithm with a deterministic reveal count, and this exercise implements Algorithm 4 as written instead.

The paper's main results use low-confidence remasking, which Table 9 shows outperforms random remasking. Random remasking remains the reference process that Equation 10 describes.

## Three steps on a two-token response

Take $N = 3$ and $L = 2$, so the remask probabilities are $\rho_0 = 2/3$, $\rho_1 = 1/2$ and $\rho_2 = 0$. Write the mask as $M$ and start from $(M, M)$.

- At step 0 the predictions are $(2, 1)$ and the draws are $(0.80, 0.40)$. Position 0 has $0.80 \ge 2/3$ and becomes 2, while position 1 has $0.40 < 2/3$ and returns to the mask. The state is $(2, M)$.
- At step 1 the predictions are $(0, 3)$ and the draws are $(0.10, 0.30)$. Position 0 is already revealed, so its new prediction 0 and its low draw are ignored. Position 1 has $0.30 < 1/2$ and stays masked, so the state is still $(2, M)$.
- At step 2 the predictions are $(1, 0)$ and the remask probability is zero, so position 1 becomes 0 and the final response is $(2, 0)$.

The trajectory has four rows, $(M, M)$, $(2, M)$, $(2, M)$ and $(2, 0)$, one more than the number of steps.

## Common mistakes to avoid

- **Remasking with the wrong probability.** A masked token stays masked with probability $s/t$ and is revealed with probability $(t-s)/t$, and swapping the two reveals almost everything at the first step.
- **Swapping the two times.** The ratio $t/s$ exceeds one and divides by zero at the last step, so masks survive to the end.
- **Counting steps from the wrong end.** The first step starts at $t = 1$, and a ratio that starts at zero reveals every token immediately.
- **Overwriting revealed tokens.** A token revealed earlier is fixed, so later predictions at that position must be ignored.
- **Remasking revealed tokens.** Only positions that are still masked take part in the random draw, because Equation 10 gives a real token probability one of staying unchanged.
- **Using the code's random confidence option.** Revealing a fixed number of tokens per step ranked by random scores is a different sampler from Algorithm 4.
- **Dropping the starting row.** The trajectory begins with the fully masked response, so it holds one more state than there are steps.