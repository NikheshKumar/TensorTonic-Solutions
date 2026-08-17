# <span style="font-size: 20px;">Token + Positional Embedding</span>

<span style="font-size: 14px;">Token and positional embeddings are the two learned lookup tables that form the input representation in GPT-2 (Radford et al., 2019). Every token entering the model is converted into a dense vector by summing its token embedding (encoding word identity) and its positional embedding (encoding sequential position). This combined representation is the only input the Transformer decoder stack ever sees.</span>

---

## <span style="font-size: 16px;">What It Is</span>

<span style="font-size: 14px;">Language models operate on discrete token IDs, but Transformer layers require continuous vector representations. The embedding layer bridges this gap. GPT-2 uses two separate embedding tables, both learned during pre-training:</span>

* <span style="font-size: 14px;">**Token embedding table ($W_e$):** Maps each vocabulary index to a dense vector. Shape: $V \times d_{\text{model}}$, where $V$ is the vocabulary size and $d_{\text{model}}$ is the hidden dimension.</span>
* <span style="font-size: 14px;">**Positional embedding table ($W_p$):** Maps each sequence position to a dense vector. Shape: $L \times d_{\text{model}}$, where $L$ is the maximum context length.</span>

<span style="font-size: 14px;">For each token in the input sequence, the model performs two table lookups and sums the results element-wise. The token embedding captures what the token means; the positional embedding captures where the token sits in the sequence. Their sum gives the Transformer a single vector per position that encodes both identity and order.</span>

<span style="font-size: 14px;">Without positional embeddings, the Transformer is entirely permutation-invariant -- self-attention treats inputs as a set, not a sequence. "The dog bit the man" and "the man bit the dog" would produce identical representations. The positional embedding breaks this symmetry.</span>

---

## <span style="font-size: 16px;">Key Equations</span>

<span style="font-size: 14px;">Given an input sequence of token IDs $[t_0, t_1, \ldots, t_{n-1}]$ where each $t_i \in \{0, 1, \ldots, V-1\}$, the embedding for position $i$ is:</span>

$$
h_i^{(0)} = W_e[t_i] + W_p[i]
$$

<span style="font-size: 14px;">where:</span>

* <span style="font-size: 14px;">$W_e \in \mathbb{R}^{V \times d_{\text{model}}}$ is the token embedding table. The lookup $W_e[t_i]$ selects row $t_i$, returning a vector in $\mathbb{R}^{d_{\text{model}}}$.</span>
* <span style="font-size: 14px;">$W_p \in \mathbb{R}^{L \times d_{\text{model}}}$ is the positional embedding table. The lookup $W_p[i]$ selects row $i$, returning a vector in $\mathbb{R}^{d_{\text{model}}}$.</span>
* <span style="font-size: 14px;">$h_i^{(0)} \in \mathbb{R}^{d_{\text{model}}}$ is the input to the first Transformer block at position $i$.</span>

<span style="font-size: 14px;">For the entire sequence, the result is a matrix $H^{(0)} \in \mathbb{R}^{n \times d_{\text{model}}}$ where each row $i$ is the sum of the token embedding for $t_i$ and the positional embedding for position $i$. Both lookups are simple row selections from their respective tables -- no matrix multiplication, no learned projection. The addition is strictly element-wise.</span>

---

## <span style="font-size: 16px;">Token Embeddings</span>

<span style="font-size: 14px;">The token embedding table is the largest single parameter matrix in GPT-2. It maps each token in the BPE vocabulary to a dense vector of dimension $d_{\text{model}}$. For GPT-2 Small, the vocabulary has 50,257 tokens (50,000 BPE merges plus 256 byte-level tokens plus one end-of-text token) and $d_{\text{model}} = 768$, giving $50{,}257 \times 768 = 38{,}597{,}376$ parameters.</span>

<span style="font-size: 14px;">The token embedding is a lookup, not a matrix multiplication. Given token ID $t_i$, the operation $W_e[t_i]$ selects a single row from the table. Gradients flow only to the selected row during backpropagation. Before training, the table is initialized randomly. During pre-training, each row becomes a dense vector capturing semantic and syntactic properties of its corresponding token.</span>

<span style="font-size: 14px;">GPT-2 ties the token embedding table to the output projection matrix. The final layer of the model projects hidden states back to vocabulary logits by multiplying with $W_e^T$. This weight tying halves the parameter cost of the vocabulary mapping and acts as a regularizer, forcing the input and output representations to share the same vector space.</span>

---

## <span style="font-size: 16px;">Positional Embeddings</span>

<span style="font-size: 14px;">Self-attention is inherently order-agnostic -- permuting the input tokens permutes the outputs identically, but the attention weights themselves do not change. GPT-2 injects positional information through a learned positional embedding table $W_p \in \mathbb{R}^{1024 \times d_{\text{model}}}$, where 1024 is the maximum context length. Each position $i \in \{0, 1, \ldots, 1023\}$ has its own trainable vector. This is fundamentally different from the original Transformer (Vaswani et al., 2017), which used fixed sinusoidal functions.</span>

### <span style="font-size: 14px;">Learned vs. Sinusoidal Positional Encodings</span>

<span style="font-size: 14px;">The original Transformer defines positional encodings using sine and cosine functions at different frequencies:</span>

$$
PE_{(pos, 2k)} = \sin\!\left(\frac{pos}{10000^{2k/d_{\text{model}}}}\right), \quad PE_{(pos, 2k+1)} = \cos\!\left(\frac{pos}{10000^{2k/d_{\text{model}}}}\right)
$$

<span style="font-size: 14px;">These sinusoidal encodings are deterministic and require no learned parameters. Because relative position shifts correspond to linear transformations of the sinusoidal vectors, the model can potentially generalize to unseen sequence lengths. GPT-2 instead uses fully learned positional embeddings. Each of the 1024 position vectors is an independent parameter vector trained alongside all other model weights:</span>

* <span style="font-size: 14px;">**Greater expressiveness:** Learned embeddings can represent arbitrary position-dependent patterns, not just smooth frequency-based functions. If certain positions require qualitatively different representations (e.g., the very first token vs. mid-sequence tokens), learned embeddings can accommodate this.</span>
* <span style="font-size: 14px;">**Fixed maximum length:** The embedding table has exactly $L$ rows. Positions beyond $L - 1$ have no corresponding embedding, making it impossible to process sequences longer than $L$ without modification. GPT-2's hard limit is 1024 tokens.</span>
* <span style="font-size: 14px;">**Empirical parity:** Vaswani et al. (2017) noted that learned and sinusoidal positional encodings produced "nearly identical results" on their translation benchmarks. GPT-2 chose learned embeddings with no degradation in language modeling quality.</span>
* <span style="font-size: 14px;">**Parameter cost:** The positional embedding table adds $L \times d_{\text{model}}$ parameters. For GPT-2 Small, this is $1024 \times 768 = 786{,}432$ parameters -- about 0.5% of total model parameters. The cost is negligible.</span>

---

## <span style="font-size: 16px;">Why Addition, Not Concatenation</span>

<span style="font-size: 14px;">Concatenation would produce a vector of dimension $2 \times d_{\text{model}}$, preserving both signals without interference. However, this would double the hidden dimension entering the Transformer stack. Every subsequent weight matrix -- $W^Q$, $W^K$, $W^V$, and both feed-forward layers -- would need twice the input width. This roughly doubles the parameter count and quadruples attention compute (since cost scales as $O(n^2 \cdot d)$).</span>

<span style="font-size: 14px;">Addition works because the embedding space is high-dimensional. In $d_{\text{model}} = 768$ dimensions, two random vectors are nearly orthogonal with high probability. The token and positional embeddings occupy largely non-overlapping subspaces of $\mathbb{R}^{d_{\text{model}}}$, making their sum nearly as informative as concatenation. The attention layers' linear projections can learn to disentangle the two signals.</span>

---

## <span style="font-size: 16px;">Paper Context</span>

<span style="font-size: 14px;">Radford et al. describe the embedding layer in "Language Models are Unsupervised Multitask Learners" (2019), which introduced GPT-2. The input representation is the sum of token embeddings and position embeddings, with no segment embeddings (unlike BERT, which adds a third embedding to distinguish sentence pairs).</span>

<span style="font-size: 14px;">GPT-2 uses byte-level Byte Pair Encoding (BPE) as its tokenization scheme. The vocabulary contains 50,257 entries: 256 base byte tokens, 50,000 BPE merges, and a single end-of-text token. The byte-level approach ensures that any Unicode string can be tokenized without [UNK] tokens. The maximum context length of 1024 tokens doubled GPT-1's 512-token limit, contributing to improved performance across language modeling benchmarks.</span>

<span style="font-size: 14px;">The paper reports four model sizes (Small through XL, from 117M to 1.5B parameters), all sharing the same embedding structure. GPT-2 Small uses $d_{\text{model}} = 768$ with 12 layers and 12 heads; GPT-2 XL uses $d_{\text{model}} = 1600$ with 48 layers and 25 heads. In all variants, the positional embedding table has 1024 rows and $d_{\text{model}}$ columns. The positional embedding parameter count scales linearly with $d_{\text{model}}$: from 786K (Small) to 1.6M (XL) -- a tiny fraction of total parameters in every case.</span>

---

## <span style="font-size: 16px;">Numerical Example</span>

<span style="font-size: 14px;">Consider a toy example with vocabulary size $V = 5$, embedding dimension $d_{\text{model}} = 4$, and maximum context length $L = 4$. The input sequence has 3 token IDs: $[2, 0, 4]$.</span>

<span style="font-size: 14px;">**Token embedding table** $W_e \in \mathbb{R}^{5 \times 4}$:</span>

$$
W_e = \begin{pmatrix} 0.1 & 0.3 & -0.2 & 0.5 \\ -0.4 & 0.2 & 0.7 & -0.1 \\ 0.6 & -0.5 & 0.1 & 0.3 \\ 0.0 & 0.8 & -0.3 & 0.2 \\ -0.2 & 0.4 & 0.5 & -0.6 \end{pmatrix}
$$

<span style="font-size: 14px;">**Positional embedding table** $W_p \in \mathbb{R}^{4 \times 4}$:</span>

$$
W_p = \begin{pmatrix} 0.05 & 0.02 & -0.01 & 0.03 \\ -0.03 & 0.06 & 0.04 & -0.02 \\ 0.01 & -0.04 & 0.07 & 0.05 \\ 0.08 & 0.01 & -0.05 & 0.02 \end{pmatrix}
$$

<span style="font-size: 14px;">**Position 0, token ID = 2:**</span>

<span style="font-size: 14px;">Token lookup: $W_e[2] = [0.6, -0.5, 0.1, 0.3]$</span>

<span style="font-size: 14px;">Position lookup: $W_p[0] = [0.05, 0.02, -0.01, 0.03]$</span>

$$
h_0^{(0)} = [0.6 + 0.05, \; -0.5 + 0.02, \; 0.1 + (-0.01), \; 0.3 + 0.03] = [0.65, -0.48, 0.09, 0.33]
$$

<span style="font-size: 14px;">**Position 1, token ID = 0:**</span>

<span style="font-size: 14px;">Token lookup: $W_e[0] = [0.1, 0.3, -0.2, 0.5]$</span>

<span style="font-size: 14px;">Position lookup: $W_p[1] = [-0.03, 0.06, 0.04, -0.02]$</span>

$$
h_1^{(0)} = [0.1 + (-0.03), \; 0.3 + 0.06, \; -0.2 + 0.04, \; 0.5 + (-0.02)] = [0.07, 0.36, -0.16, 0.48]
$$

<span style="font-size: 14px;">**Position 2, token ID = 4:**</span>

<span style="font-size: 14px;">Token lookup: $W_e[4] = [-0.2, 0.4, 0.5, -0.6]$</span>

<span style="font-size: 14px;">Position lookup: $W_p[2] = [0.01, -0.04, 0.07, 0.05]$</span>

$$
h_2^{(0)} = [-0.2 + 0.01, \; 0.4 + (-0.04), \; 0.5 + 0.07, \; -0.6 + 0.05] = [-0.19, 0.36, 0.57, -0.55]
$$

<span style="font-size: 14px;">Notice that the same token ID at a different position would produce a different vector because the positional embedding contribution changes. If token ID 2 appeared at position 1 instead of position 0, the result would be $[0.6 + (-0.03), -0.5 + 0.06, 0.1 + 0.04, 0.3 + (-0.02)] = [0.57, -0.44, 0.14, 0.28]$ -- different from the $[0.65, -0.48, 0.09, 0.33]$ computed above.</span>

---

## <span style="font-size: 16px;">Modern Context</span>

<span style="font-size: 14px;">The GPT-2 approach of learned absolute positional embeddings was the dominant paradigm through GPT-3 (Brown et al., 2020). However, the landscape of positional encoding has shifted substantially since then, driven by the desire for longer context windows and better length generalization.</span>

* <span style="font-size: 14px;">**Rotary Position Embeddings (RoPE)** (Su et al., 2021) encode position by rotating query and key vectors rather than adding a position vector to the input. RoPE operates inside the attention mechanism, not at the embedding layer, and naturally encodes relative distances between tokens. It is used in LLaMA, LLaMA 2, Mistral, and most open-source models released after 2022.</span>
* <span style="font-size: 14px;">**ALiBi (Attention with Linear Biases)** (Press et al., 2022) eliminates positional embeddings entirely. It adds a position-dependent linear bias directly to the attention scores: tokens farther apart receive a larger negative bias. ALiBi was adopted by BLOOM and MPT. It requires zero additional parameters for positional encoding.</span>
* <span style="font-size: 14px;">**Relative position encodings** (Shaw et al., 2018; Dai et al., 2019) add learned biases based on the distance between two tokens rather than their absolute positions. Transformer-XL and T5 use variants of this approach.</span>
* <span style="font-size: 14px;">**No position at embedding time:** In all of the above modern approaches, position information is injected inside the attention layers rather than at the embedding layer. The initial input representation consists only of token embeddings, with no positional vector added before the first Transformer block. This is a fundamental architectural shift from GPT-2.</span>

<span style="font-size: 14px;">Despite these advances, learned absolute positional embeddings remain relevant. They are simple, effective at fixed context lengths, and understanding the lookup-and-add mechanism is foundational because every modern alternative defines itself in contrast to it.</span>

---

## <span style="font-size: 16px;">Pitfalls</span>

* <span style="font-size: 14px;">**Position index off-by-one errors.** Positions are 0-indexed: the first token maps to $W_p[0]$, not $W_p[1]$. An off-by-one error shifts every token's positional embedding by one row, which means every position gets the wrong positional signal. This does not crash -- the model produces outputs of the correct shape -- but the results are silently wrong because every token is represented as if it were one position later (or earlier) than its true location.</span>
* <span style="font-size: 14px;">**Exceeding the maximum context length.** GPT-2's positional embedding table has exactly 1024 rows. Attempting to embed a token at position 1024 or beyond triggers an index-out-of-bounds error. During inference, position indices must never exceed $L - 1 = 1023$. Some implementations silently clamp positions, which avoids crashes but produces incorrect representations because the model was never trained with clamped positions.</span>
* <span style="font-size: 14px;">**Forgetting to add the positional embedding.** If only the token embedding is passed to the Transformer, the model loses all positional information. Self-attention becomes a bag-of-tokens operation where word order is invisible. The model will still generate text, but it will lack coherent ordering because the attention mechanism has no position signal to work with.</span>
* <span style="font-size: 14px;">**Mismatched table dimensions.** Both embedding tables must produce vectors of the same dimension $d_{\text{model}}$ because their outputs are summed element-wise. If $W_e$ has shape $(V, 768)$ and $W_p$ has shape $(L, 512)$, the addition is impossible and will raise a shape mismatch error. This mistake arises when configuring model hyperparameters incorrectly or when adapting a pre-trained model to a different architecture.</span>
* <span style="font-size: 14px;">**Confusing position embeddings with position encodings.** In GPT-2, positional representations are learned parameters (embeddings) stored in a lookup table. In the original Transformer, they are deterministic sinusoidal functions (encodings) computed on the fly. The terms are often used interchangeably, but they refer to fundamentally different mechanisms -- one has trainable parameters, the other does not.</span>
* <span style="font-size: 14px;">**Ignoring weight tying between input and output.** GPT-2 shares $W_e$ with the final language modeling head. If the implementation creates two separate copies, the model has nearly 40M extra untied parameters, changing training dynamics. When loading pre-trained weights, the output projection must point to the same parameter tensor as the input embedding.</span>

---