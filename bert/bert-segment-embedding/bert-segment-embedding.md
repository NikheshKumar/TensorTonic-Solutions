# <span style="font-size: 20px;">Segment Embeddings</span>

<span style="font-size: 14px;">Segment embeddings are one of three embedding components that BERT (Devlin et al., 2019) sums to produce each token's input representation. While token embeddings encode word identity and position embeddings encode sequential order, segment embeddings tell the model which sentence a token belongs to. This is essential because many NLP tasks -- natural language inference, question answering, paraphrase detection -- require reasoning about pairs of sentences simultaneously.</span>

<span style="font-size: 14px;">The segment embedding is the simplest of the three: a lookup table with just two rows, one for sentence A (segment 0) and one for sentence B (segment 1). Every token receives one of these two vectors based on which sentence it belongs to.</span>

---

## <span style="font-size: 16px;">What It Is / What It Does</span>

<span style="font-size: 14px;">BERT constructs each token's input representation by summing three separate embeddings:</span>

* <span style="font-size: 14px;">**Token embedding:** Maps the token's vocabulary index to a dense vector. Lookup table shape: (vocab_size, hidden_size).</span>
* <span style="font-size: 14px;">**Position embedding:** Maps the token's absolute position in the sequence to a dense vector. Lookup table shape: (max_position_embeddings, hidden_size). These are learned, not sinusoidal.</span>
* <span style="font-size: 14px;">**Segment embedding:** Maps a binary indicator (0 or 1) to a dense vector. Lookup table shape: (2, hidden_size). Sentence A tokens get segment 0, sentence B tokens get segment 1.</span>

<span style="font-size: 14px;">The final input to the Transformer encoder stack is the element-wise sum of all three vectors. This sum is then passed through LayerNorm and dropout before entering the first Transformer block.</span>

<span style="font-size: 14px;">Segment embeddings exist because BERT was designed from the start for sentence-pair tasks. Tasks like natural language inference require the model to compare two sentences, so the model needs a mechanism to distinguish which tokens belong to which sentence. The segment embedding provides this signal at the embedding level, before any self-attention computation occurs.</span>

---

## <span style="font-size: 16px;">Key Equations</span>

<span style="font-size: 14px;">For a token at position $p$ with vocabulary index $t$ and segment label $s \in \{0, 1\}$, the input representation $E$ is:</span>

$$
E = E_{\text{token}}[t] + E_{\text{position}}[p] + E_{\text{segment}}[s]
$$

<span style="font-size: 14px;">Each term is a lookup into a learned embedding table:</span>

* <span style="font-size: 14px;">**$E_{\text{token}} \in \mathbb{R}^{V \times H}$:** Token embedding table, where $V$ is the vocabulary size (30,522 for BERT-base with WordPiece) and $H$ is the hidden size (768 for BERT-base). The lookup $E_{\text{token}}[t]$ returns a vector in $\mathbb{R}^H$.</span>
* <span style="font-size: 14px;">**$E_{\text{position}} \in \mathbb{R}^{L \times H}$:** Position embedding table, where $L$ is the maximum sequence length (512 for BERT). The lookup $E_{\text{position}}[p]$ returns a vector in $\mathbb{R}^H$.</span>
* <span style="font-size: 14px;">**$E_{\text{segment}} \in \mathbb{R}^{2 \times H}$:** Segment embedding table with exactly two rows. The lookup $E_{\text{segment}}[s]$ returns a vector in $\mathbb{R}^H$.</span>

<span style="font-size: 14px;">The segment label $s$ is determined by the token's membership:</span>

$$
s_i = \begin{cases} 0 & \text{if token } i \text{ belongs to sentence A (including [CLS] and first [SEP])} \\ 1 & \text{if token } i \text{ belongs to sentence B (including second [SEP])} \end{cases}
$$

<span style="font-size: 14px;">All three embeddings must have the same dimensionality $H$ because they are summed element-wise, not concatenated. Concatenation would triple the hidden dimension, which is not what BERT does.</span>

<span style="font-size: 14px;">The total number of learnable parameters in the segment embedding table is $2 \times H$. For BERT-base ($H = 768$), this is just 1,536 parameters -- trivial compared to the token embedding table's $30{,}522 \times 768 = 23{,}440{,}896$ parameters.</span>

---

## <span style="font-size: 16px;">Why BERT Needs Segment Embeddings</span>

<span style="font-size: 14px;">Many important NLP tasks are inherently about relationships between two text sequences:</span>

* <span style="font-size: 14px;">**Natural language inference (NLI):** Given a premise and a hypothesis, classify as entailment, contradiction, or neutral. The model must know which sentence is the premise and which is the hypothesis.</span>
* <span style="font-size: 14px;">**Question answering (QA):** Given a question and a passage, find the answer span in the passage. The model must distinguish question tokens from passage tokens.</span>
* <span style="font-size: 14px;">**Paraphrase detection:** Given two sentences, determine if they express the same meaning. The model must compare one sentence against the other.</span>
* <span style="font-size: 14px;">**Sentence similarity:** Given two sentences, compute a similarity score. Again, the model needs to know which tokens belong to which sentence.</span>

<span style="font-size: 14px;">Without segment embeddings, the model would rely solely on the [SEP] token to infer sentence boundaries. But [SEP] only marks a boundary at one position -- it does not propagate sentence membership to every token. Segment embeddings solve this by injecting membership directly into every token's representation at the input level, giving the Transformer a clean, unambiguous signal from the very first layer.</span>

---

## <span style="font-size: 16px;">Input Format</span>

<span style="font-size: 14px;">BERT's input for a sentence pair follows a strict format:</span>

$$
[\text{CLS}] \; w_1^A \; w_2^A \; \ldots \; w_m^A \; [\text{SEP}] \; w_1^B \; w_2^B \; \ldots \; w_n^B \; [\text{SEP}]
$$

<span style="font-size: 14px;">The corresponding segment IDs are:</span>

$$
\underbrace{0 \; 0 \; 0 \; \ldots \; 0 \; 0}_{\text{[CLS] + sentence A + [SEP]}} \; \underbrace{1 \; 1 \; 1 \; \ldots \; 1 \; 1}_{\text{sentence B + [SEP]}}
$$

<span style="font-size: 14px;">Key details about the input format:</span>

* <span style="font-size: 14px;">**[CLS] gets segment 0.** It is always part of sentence A's segment. Its final hidden state is used as the aggregate sequence representation for classification tasks.</span>
* <span style="font-size: 14px;">**First [SEP] gets segment 0.** The separator after sentence A belongs to sentence A's segment.</span>
* <span style="font-size: 14px;">**Second [SEP] gets segment 1.** The separator after sentence B belongs to sentence B's segment.</span>
* <span style="font-size: 14px;">**Position IDs are global.** Positions run from 0 to $m + n + 2$ (total tokens minus 1), spanning both sentences continuously. They do not reset at the sentence boundary.</span>

<span style="font-size: 14px;">For single-sentence tasks (sentiment analysis, named entity recognition), only sentence A is used:</span>

$$
[\text{CLS}] \; w_1 \; w_2 \; \ldots \; w_m \; [\text{SEP}]
$$

<span style="font-size: 14px;">All segment IDs are 0 in the single-sentence case. The segment B embedding row ($E_{\text{segment}}[1]$) is simply never accessed.</span>

---

## <span style="font-size: 16px;">Paper Context</span>

<span style="font-size: 14px;">Devlin et al. describe the input representation in Section 3 of "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding" (2019). The paper states: "We differentiate the sentences in two ways. First, we separate them with a special token ([SEP]). Second, we add a learned embedding to every token indicating whether it belongs to sentence A or sentence B."</span>

<span style="font-size: 14px;">The segment embedding was motivated by BERT's Next Sentence Prediction (NSP) pre-training objective. BERT receives sentence pairs and predicts whether sentence B actually follows sentence A in the corpus, or is randomly sampled. This binary classification uses the [CLS] token's final representation, so the model needs segment embeddings to distinguish the two sentences.</span>

<span style="font-size: 14px;">BERT uses learned position embeddings, not the sinusoidal encodings from the original Transformer (Vaswani et al., 2017). The original Transformer defined $\text{PE}(pos, 2i) = \sin(pos / 10000^{2i/d})$ and $\text{PE}(pos, 2i+1) = \cos(pos / 10000^{2i/d})$. BERT replaces these with a fully learned position embedding table, trading extrapolation ability for a representation optimized during pre-training. This limits BERT to sequences of at most 512 tokens.</span>

<span style="font-size: 14px;">The paper notes that "sentence" in BERT's context is a misnomer -- a "sentence" can be any contiguous span of text, not necessarily a linguistic sentence. Each pre-training input consists of two "segments" that may contain multiple actual sentences. This is why "segment embedding" is more accurate than "sentence embedding."</span>

---

## <span style="font-size: 16px;">Numerical Example</span>

<span style="font-size: 14px;">Consider the sentence pair: "The cat sat" (sentence A) and "It slept" (sentence B). After WordPiece tokenization, assume the tokens and their vocabulary indices are:</span>

* <span style="font-size: 14px;">**[CLS]:** token_id = 101</span>
* <span style="font-size: 14px;">**The:** token_id = 1996</span>
* <span style="font-size: 14px;">**cat:** token_id = 4937</span>
* <span style="font-size: 14px;">**sat:** token_id = 2938</span>
* <span style="font-size: 14px;">**[SEP]:** token_id = 102</span>
* <span style="font-size: 14px;">**It:** token_id = 2009</span>
* <span style="font-size: 14px;">**slept:** token_id = 14195</span>
* <span style="font-size: 14px;">**[SEP]:** token_id = 102</span>

<span style="font-size: 14px;">The three ID sequences for this input are:</span>

* <span style="font-size: 14px;">**token_ids:** [101, 1996, 4937, 2938, 102, 2009, 14195, 102]</span>
* <span style="font-size: 14px;">**position_ids:** [0, 1, 2, 3, 4, 5, 6, 7]</span>
* <span style="font-size: 14px;">**segment_ids:** [0, 0, 0, 0, 0, 1, 1, 1]</span>

<span style="font-size: 14px;">Now consider the embedding lookups for the token "cat" at position 2 (segment A) and the token "slept" at position 6 (segment B). Assume a tiny hidden_size $H = 4$ for illustration.</span>

<span style="font-size: 14px;">For "cat" (token_id=4937, position=2, segment=0):</span>

$$
E_{\text{token}}[4937] = [0.12, -0.45, 0.78, 0.33]
$$

$$
E_{\text{position}}[2] = [0.05, 0.11, -0.03, 0.22]
$$

$$
E_{\text{segment}}[0] = [0.08, -0.14, 0.06, 0.19]
$$

$$
E_{\text{cat}} = [0.12 + 0.05 + 0.08, \; -0.45 + 0.11 + (-0.14), \; 0.78 + (-0.03) + 0.06, \; 0.33 + 0.22 + 0.19]
$$

$$
E_{\text{cat}} = [0.25, \; -0.48, \; 0.81, \; 0.74]
$$

<span style="font-size: 14px;">For "slept" (token_id=14195, position=6, segment=1):</span>

$$
E_{\text{token}}[14195] = [-0.31, 0.56, 0.14, -0.27]
$$

$$
E_{\text{position}}[6] = [0.18, -0.07, 0.29, 0.03]
$$

$$
E_{\text{segment}}[1] = [-0.11, 0.22, -0.09, 0.15]
$$

$$
E_{\text{slept}} = [-0.31 + 0.18 + (-0.11), \; 0.56 + (-0.07) + 0.22, \; 0.14 + 0.29 + (-0.09), \; -0.27 + 0.03 + 0.15]
$$

$$
E_{\text{slept}} = [-0.24, \; 0.71, \; 0.34, \; -0.09]
$$

<span style="font-size: 14px;">"cat" received $E_{\text{segment}}[0] = [0.08, -0.14, 0.06, 0.19]$ while "slept" received $E_{\text{segment}}[1] = [-0.11, 0.22, -0.09, 0.15]$. These differ, so even before self-attention begins, the model has a signal that these tokens belong to different sentences.</span>

<span style="font-size: 14px;">Every sentence A token gets the same $E_{\text{segment}}[0]$ added; every sentence B token gets $E_{\text{segment}}[1]$. This uniform additive signal is what the Transformer uses to learn sentence-aware representations.</span>

---

## <span style="font-size: 16px;">Modern Context</span>

<span style="font-size: 14px;">Since BERT's publication, several architectural decisions around embeddings have evolved significantly.</span>

<span style="font-size: 14px;">**Position embeddings: learned to rotary.** Modern models like LLaMA and Mistral use Rotary Position Embeddings (RoPE, Su et al., 2021), which encode relative position by rotating query and key vectors. RoPE captures relative distances between tokens and can extrapolate to unseen sequence lengths. BERT's learned positions are limited to max_position_embeddings = 512.</span>

<span style="font-size: 14px;">**Segment embeddings: sometimes unnecessary.** RoBERTa (Liu et al., 2019) demonstrated that the Next Sentence Prediction objective -- the primary reason segment embeddings exist -- actually hurts downstream performance. RoBERTa removed NSP and set all segment IDs to 0 during pre-training, making the segment embedding a constant additive bias absorbed by other parameters. This showed that sentence membership can be learned implicitly through other mechanisms.</span>

<span style="font-size: 14px;">**Decoder-only models do not use segment embeddings.** GPT-2, GPT-3, LLaMA, Mistral, and other autoregressive models process a single continuous token sequence with no concept of sentence A vs B. Multi-turn distinctions are encoded through special tokens and formatting in the text itself, not a separate embedding table.</span>

<span style="font-size: 14px;">**ALBERT's factored embeddings.** ALBERT (Lan et al., 2020) factored the token embedding into two smaller matrices ($V \times E$ then $E \times H$ with $E \ll H$), reducing parameters dramatically. It retained segment embeddings.</span>

<span style="font-size: 14px;">**XLNet's relative segments.** XLNet (Yang et al., 2019) replaced BERT-style absolute segment embeddings with relative segment encodings, computing attention biases based on whether two tokens are in the same or different segments rather than assigning a fixed vector per segment.</span>

---

## <span style="font-size: 16px;">Pitfalls</span>

### <span style="font-size: 14px;">Wrong Segment IDs for Single-Sentence Input</span>

<span style="font-size: 14px;">When fine-tuning on single-sentence tasks, all segment IDs must be 0. Using segment 1 would add an unexpected bias vector to every token, degrading performance. Most tokenizer implementations default to all-zeros, but custom preprocessing pipelines may not.</span>

### <span style="font-size: 14px;">Forgetting That [CLS] Gets Segment 0</span>

<span style="font-size: 14px;">The [CLS] token always receives segment ID 0. During pre-training, the [CLS] representation feeds the NSP classification head. If [CLS] were given segment 1, the pre-trained head would see an input distribution it was never trained on, and downstream fine-tuning on classification tasks would start from a poor initialization.</span>

### <span style="font-size: 14px;">Position ID Off-by-One Errors</span>

<span style="font-size: 14px;">Position IDs run continuously from 0 across both sentences. They do not reset at sentence B's start. [CLS] is position 0, sentence A occupies 1 to $m$, first [SEP] is $m+1$, sentence B starts at $m+2$, second [SEP] at $m+n+2$. Resetting positions at the boundary would reuse embeddings and confuse the model.</span>

### <span style="font-size: 14px;">Mixing Up Embedding Table Dimensions</span>

<span style="font-size: 14px;">The three embedding tables have very different first dimensions but identical second dimensions:</span>

* <span style="font-size: 14px;">**Token:** (30522, 768) -- 30,522 vocabulary entries</span>
* <span style="font-size: 14px;">**Position:** (512, 768) -- 512 possible positions</span>
* <span style="font-size: 14px;">**Segment:** (2, 768) -- 2 possible segments</span>

<span style="font-size: 14px;">A common confusion is stating the shape as (768, 2). In PyTorch's nn.Embedding, the convention is (num_embeddings, embedding_dim), so the first dimension is always the number of discrete entries. Transposing would produce incorrect lookups.</span>

### <span style="font-size: 14px;">Concatenating Instead of Summing</span>

<span style="font-size: 14px;">BERT sums the three embeddings element-wise, producing a vector of dimension $H$. It does not concatenate them into a vector of dimension $3H$. Summing preserves the hidden dimension expected by the Transformer layers, while concatenation would produce an input three times too wide and fail at the first attention layer.</span>

### <span style="font-size: 14px;">Assuming More Than Two Segments</span>

<span style="font-size: 14px;">BERT's segment embedding table has exactly 2 rows. It cannot handle three or more segments without modification. Workarounds include concatenating extra text into one segment, adding rows (requires re-training), or using separator tokens without segment distinction.</span>

---