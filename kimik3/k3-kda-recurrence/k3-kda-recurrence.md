# KDA Recurrence

Kimi Delta Attention, or KDA, reads a sequence through a running memory matrix. Instead of keeping every earlier key and value, it updates one fixed-size state after each token. The important idea in this problem is that the state can forget old information, correct information associated with a key, and write new information before the current query reads from it.

## What the state remembers

Think of the state as a small associative memory. A key describes where information belongs, a value is the information being stored, and a query asks the memory what it currently knows. The matrix state connects key features to value features, so multiplying the updated state by a query produces a value-like output.

KDA processes tokens in sequence order because every update depends on the state left by the previous token. For the current token, the update is

$$
S_t = \left(I-\beta_t k_t k_t^{\mathsf T}\right)\operatorname{Diag}(\alpha_t)S_{t-1} + \beta_t k_t v_t^{\mathsf T}
$$

This formula has three understandable parts:

- **Decay:** the retention vector $\alpha_t$ reduces selected channels of the old state.
- **Erase:** the factor involving $k_t k_t^{\mathsf T}$ removes part of what the state currently associates with this key.
- **Write:** the outer product $k_t v_t^{\mathsf T}$ stores the new value at the location described by the key.

The erase and write terms use the same strength $\beta_t$. A small value makes the update cautious, while a value near one makes the new token change the memory strongly.

## Why KDA decays channels separately

A single scalar forget gate would retain or fade every key feature by the same amount. KDA instead computes one retention value for each key channel. This allows some parts of the memory to remain stable while others change quickly.

The decay logits are converted to retention values with

$$
\alpha_t = \exp\left(g_{\min}\operatorname{sigmoid}(z_t)\right)
$$

Since $g_{\min}$ is negative and sigmoid returns a value between zero and one, every retention value lies between $\exp(g_{\min})$ and one. In Kimi K3, the fixed lower log-decay bound prevents the decay from becoming arbitrarily extreme. For this exercise, the practical consequence is simple: apply sigmoid first, multiply by the negative bound, then exponentiate element by element.

## Read after writing

The current output is read from $S_t$, not from $S_{t-1}$:

$$
\widetilde{o}_t = S_t^{\mathsf T}q_t
$$

That ordering means a token can contribute to its own result. If the implementation reads first and updates afterward, every output is shifted by one step and the first token cannot use its own value.

The raw readout is normalized independently within each head. RMS normalization divides a head vector by the square root of its mean squared value plus a small epsilon. This controls scale without subtracting the mean. A sigmoid gate then decides, channel by channel, how much of the normalized readout should pass. Finally, the heads are joined and the supplied output projection mixes their channels into model width.

## A one-dimensional example

Use a single head with one key channel and one value channel. Let the previous state be $2$, retention be $0.5$, key be $1$, value be $4$, and write strength be $0.25$.

First decay the old state:

$$
0.5 \times 2 = 1
$$

The erase factor is $1-0.25\times1\times1=0.75$, so the retained part becomes $0.75$. The new write is $0.25\times1\times4=1$. The updated state is therefore $1.75$.

If the query is $2$, the raw readout is $1.75\times2=3.5$. This example shows why the update is more than ordinary accumulation: part of the old association is deliberately erased before the new one is written.

## Implementation order

- Convert every decay logit into a channel-wise retention value.
- Start from the supplied initial state and iterate over sequence positions.
- For each batch item and head, decay the old state, apply the erase term, and add the write outer product.
- Read the current query from the newly updated state.
- RMS-normalize each head readout, apply the sigmoid output gate, join the heads, and apply the output projection.
- Stack the per-token outputs and return them with the final state.

## Common mistakes to avoid

- **Reading the previous state.** The problem requires the query to use the state after the current update.
- **Using one decay value per head.** Decay is channel-wise and must match the key features.
- **Reversing the outer product.** The state maps key width to value width, so the write must follow $k_t v_t^{\mathsf T}$.
- **Normalizing across heads.** RMS normalization is independent for each head's value channels.
- **Mutating the initial state.** Build a working state without changing the supplied tensor.
