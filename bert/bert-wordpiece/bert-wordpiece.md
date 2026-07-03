# <span style="font-size: 20px;">WordPiece Tokenization</span>

<span style="font-size: 14px;">WordPiece is a subword tokenization algorithm that splits text into subword units drawn from a fixed vocabulary. Introduced by Schuster and Nakajima (2012) for Japanese and Korean segmentation, it was later adopted as the core tokenizer for BERT (Devlin et al., 2018). The algorithm uses greedy longest-match-first: given a word, it finds the longest prefix in the vocabulary, emits it, and continues from where it left off.</span>

<span style="font-size: 14px;">In BERT, WordPiece operates on a 30,000-token vocabulary. Every input sentence is first split into whitespace-delimited words, then each word is broken into WordPiece tokens. The first subword token retains its original form; all subsequent subword tokens within the same word are prefixed with `##` to signal continuation. If the algorithm cannot segment a word at all, the entire word is replaced by `[UNK]`.</span>

---

## <span style="font-size: 16px;">What It Is / What It Does</span>

<span style="font-size: 14px;">Tokenization converts raw text into discrete units a model can process. Three broad strategies exist:</span>

* <span style="font-size: 14px;">**Word-level tokenization:** Each whitespace-separated word becomes one token. This creates enormous vocabularies (English alone has hundreds of thousands of word forms) and causes out-of-vocabulary (OOV) failures on unseen words.</span>
* <span style="font-size: 14px;">**Character-level tokenization:** Each character becomes one token. The vocabulary is tiny and OOV is eliminated, but sequences become extremely long, making self-attention quadratically expensive. Individual characters carry minimal semantic information.</span>
* <span style="font-size: 14px;">**Subword tokenization:** Words are split into pieces that balance frequency and coverage. Common words stay whole ("the", "is"), while rare words are decomposed into meaningful sub-units ("un", "##happi", "##ness"). The vocabulary stays compact and OOV is handled gracefully.</span>

<span style="font-size: 14px;">WordPiece maintains a fixed-size vocabulary of subword units covering all training text. Frequent words appear as single tokens; rare words decompose into known subwords carrying partial semantics. This is why BERT handles misspellings, neologisms, and morphological variants without [UNK].</span>

<span style="font-size: 14px;">The `##` prefix disambiguates word-initial tokens from continuations. "un" starting "unhappy" is token `un`, while "un" mid-word would be `##un`. This lets the model reconstruct word boundaries from a flat token sequence.</span>

---

## <span style="font-size: 16px;">Key Equations</span>

<span style="font-size: 14px;">WordPiece tokenization at inference time is defined by a greedy algorithm rather than a closed-form equation. The core logic can be expressed formally as follows.</span>

<span style="font-size: 14px;">Let $w = c_1 c_2 \dots c_n$ be a word of $n$ characters, and let $V$ be the vocabulary. Define the tokenization function $\text{WordPiece}(w, V)$:</span>

<span style="font-size: 14px;">**Initialize:**</span>

$$
\text{tokens} = [\,], \quad \text{start} = 0
$$

<span style="font-size: 14px;">**While** $\text{start} < n$:</span>

$$
\text{end} = n, \quad \text{cur\_substr} = \text{None}
$$

<span style="font-size: 14px;">**While** $\text{start} < \text{end}$:</span>

$$
\text{substr} = \begin{cases} w[\text{start}:\text{end}] & \text{if } \text{start} = 0 \\ \texttt{\#\#} + w[\text{start}:\text{end}] & \text{if } \text{start} > 0 \end{cases}
$$

$$
\text{if } \text{substr} \in V: \quad \text{cur\_substr} = \text{substr}, \quad \text{break}
$$

$$
\text{end} = \text{end} - 1
$$

<span style="font-size: 14px;">**If** $\text{cur\_substr}$ is None: return $[\texttt{[UNK]}]$</span>

$$
\text{tokens.append}(\text{cur\_substr}), \quad \text{start} = \text{end}
$$

<span style="font-size: 14px;">**Return** tokens</span>

<span style="font-size: 14px;">The key insight is greedy longest-match: at each position, try the longest possible substring first and shrink until a vocabulary match is found. This is $O(n^2)$ worst case for a word of length $n$, though in practice words are short and vocabulary lookups use hash sets for $O(1)$ membership testing.</span>

<span style="font-size: 14px;">The vocabulary construction phase (offline, before training) uses a different criterion. WordPiece merges the token pair $(x, y)$ that maximizes:</span>

$$
\text{score}(x, y) = \frac{P(xy)}{P(x) \cdot P(y)}
$$

<span style="font-size: 14px;">where $P$ denotes corpus frequency. This is pointwise mutual information -- it favors merging pairs that co-occur more than expected by chance, not merely pairs that are frequent in absolute terms.</span>

---

## <span style="font-size: 16px;">The Algorithm Step by Step</span>

<span style="font-size: 14px;">The WordPiece tokenization algorithm for a single word proceeds as follows:</span>

* <span style="font-size: 14px;">**Step 1 -- Set the start pointer to position 0.** The first token emitted will not have the `##` prefix.</span>
* <span style="font-size: 14px;">**Step 2 -- Set the end pointer to the word length.** The candidate substring is `word[start:end]`. We try the longest possible substring first.</span>
* <span style="font-size: 14px;">**Step 3 -- Check if the candidate is in the vocabulary.** If `start > 0`, prepend `##` before checking. If found, emit it and go to Step 5.</span>
* <span style="font-size: 14px;">**Step 4 -- Shrink the candidate.** Decrement `end` by 1 and repeat Step 3. If `end` reaches `start` without finding a match, return `[UNK]` for the entire word.</span>
* <span style="font-size: 14px;">**Step 5 -- Advance the start pointer.** Set `start = end`. If `start` has reached the end of the word, we are done. Otherwise, go to Step 2.</span>

<span style="font-size: 14px;">Critical details:</span>

* <span style="font-size: 14px;">**The `##` prefix is only added when `start > 0`.** The first subword of every word is unprefixed. All continuation subwords carry `##`.</span>
* <span style="font-size: 14px;">**Failure is all-or-nothing.** If at any position the algorithm cannot find even a single-character match, the entire original word becomes `[UNK]`. Partial results are discarded.</span>
* <span style="font-size: 14px;">**Greedy does not guarantee optimal segmentation.** The longest match at position $i$ may prevent a globally better segmentation. WordPiece accepts this tradeoff for speed.</span>
* <span style="font-size: 14px;">**The vocabulary must contain all individual characters from training data.** If even one character is missing, any word containing it becomes `[UNK]`.</span>

---

## <span style="font-size: 16px;">Paper Context</span>

<span style="font-size: 14px;">The original BERT paper (Devlin et al., 2018) states: "We use WordPiece embeddings with a 30,000 token vocabulary." Here is how WordPiece fits into BERT:</span>

<span style="font-size: 14px;">**Vocabulary construction.** The 30K vocabulary is built offline before training. Starting from individual characters, the algorithm iteratively merges token pairs that maximize corpus likelihood. This differs from BPE, which merges the most frequent pair. The process repeats until 30,000 tokens are reached.</span>

<span style="font-size: 14px;">**Special tokens.** BERT's vocabulary includes five special tokens:</span>

* <span style="font-size: 14px;">**[CLS]:** Prepended to every input. Its final hidden state serves as the sequence representation for classification.</span>
* <span style="font-size: 14px;">**[SEP]:** Inserted between sentence pairs and at the end. Marks sentence boundaries.</span>
* <span style="font-size: 14px;">**[PAD]:** Fills sequences to uniform length. Masked out in attention.</span>
* <span style="font-size: 14px;">**[MASK]:** Replaces tokens during Masked Language Model pre-training. The model predicts the original token from context.</span>
* <span style="font-size: 14px;">**[UNK]:** Represents words that cannot be segmented into known subwords.</span>

<span style="font-size: 14px;">**Cased vs. uncased models.** The uncased model lowercases text and strips accents, reducing vocabulary pressure but losing case info. The cased model preserves casing, important for NER where "Apple" (company) differs from "apple" (fruit).</span>

<span style="font-size: 14px;">**Input representation.** After tokenization, each token receives three embeddings summed element-wise:</span>

* <span style="font-size: 14px;">**Token embedding:** Learned vector for each vocabulary entry (30,000 vectors).</span>
* <span style="font-size: 14px;">**Segment embedding:** Indicates Sentence A vs. Sentence B (two vectors).</span>
* <span style="font-size: 14px;">**Position embedding:** Encodes absolute position (up to 512 vectors).</span>

<span style="font-size: 14px;">**Maximum sequence length.** BERT supports up to 512 WordPiece tokens including [CLS] and [SEP]. Subword expansion means the effective word-level context window is shorter than 512.</span>

---

## <span style="font-size: 16px;">WordPiece vs BPE vs Unigram</span>

<span style="font-size: 14px;">Three subword tokenization algorithms dominate modern NLP. They share the goal of decomposing text into subword units but differ in how they build vocabularies and segment at inference.</span>

* <span style="font-size: 14px;">**BPE (GPT, GPT-2, GPT-3, RoBERTa):** Build bottom-up by merging the most frequent adjacent pair. Inference replays merges in learned order. Deterministic.</span>
* <span style="font-size: 14px;">**WordPiece (BERT, DistilBERT, ELECTRA):** Build bottom-up by merging the pair that maximizes $P(xy) / (P(x) \cdot P(y))$ -- likelihood gain rather than raw frequency. Inference uses greedy longest-match on the final vocabulary; merge history is discarded. Deterministic.</span>
* <span style="font-size: 14px;">**Unigram (T5, XLNet, ALBERT, mBART):** Build top-down by starting with a large vocabulary and pruning tokens whose removal least decreases corpus likelihood. Inference finds the optimal segmentation via Viterbi. Can sample multiple segmentations for subword regularization.</span>

<span style="font-size: 14px;">**Key distinctions:**</span>

* <span style="font-size: 14px;">**Build direction:** BPE and WordPiece merge bottom-up. Unigram prunes top-down.</span>
* <span style="font-size: 14px;">**Merge/prune criterion:** BPE uses frequency. WordPiece uses likelihood ratio. Unigram uses likelihood loss on removal.</span>
* <span style="font-size: 14px;">**Inference:** BPE replays merges. WordPiece does greedy longest-match. Unigram uses Viterbi or sampling.</span>

---

## <span style="font-size: 16px;">Numerical Example</span>

<span style="font-size: 14px;">Consider tokenizing "unhappiness" with vocabulary $V = \{$"un", "happy", "##happi", "##happiness", "##ness", "##i", "##n", "##e", "##s", "a", "h", "i", "n", "p", "s", "u", "e", ...$\}$</span>

<span style="font-size: 14px;">**Iteration 1 (start=0):**</span>

* <span style="font-size: 14px;">**Try** word[0:11] = "unhappiness". Not in $V$.</span>
* <span style="font-size: 14px;">**Try** word[0:10] through word[0:3]. None in $V$.</span>
* <span style="font-size: 14px;">**Try** word[0:2] = "un". Found! Emit "un". Set start=2.</span>

<span style="font-size: 14px;">**Iteration 2 (start=2):**</span>

* <span style="font-size: 14px;">**Try** "##" + word[2:11] = "##happiness". Found! Emit "##happiness". Set start=11.</span>

<span style="font-size: 14px;">**start=11 = word length. Result:** ["un", "##happiness"]</span>

<span style="font-size: 14px;">Now with $V_2$ where "##happiness" is absent but "##happi" and "##ness" are present:</span>

<span style="font-size: 14px;">**Iteration 1 (start=0):**</span>

* <span style="font-size: 14px;">**Try** word[0:11] through word[0:3]. None found.</span>
* <span style="font-size: 14px;">**Try** word[0:2] = "un". Found! Emit "un". Set start=2.</span>

<span style="font-size: 14px;">**Iteration 2 (start=2):**</span>

* <span style="font-size: 14px;">**Try** "##" + word[2:11] through "##" + word[2:8]. None in $V_2$.</span>
* <span style="font-size: 14px;">**Try** "##" + word[2:7] = "##happi". Found! Emit "##happi". Set start=7.</span>

<span style="font-size: 14px;">**Iteration 3 (start=7):**</span>

* <span style="font-size: 14px;">**Try** "##" + word[7:11] = "##ness". Found! Emit "##ness". Set start=11.</span>

<span style="font-size: 14px;">**Result:** ["un", "##happi", "##ness"]</span>

<span style="font-size: 14px;">The same word produces two tokens with $V$ and three with $V_2$. Vocabulary composition directly determines segmentation granularity. In both cases the algorithm greedily takes the longest available match at each position.</span>

<span style="font-size: 14px;">**The [UNK] case.** If we tokenize "cafe" with an accent on the final character but that accented character is absent from the vocabulary, the algorithm fails at that position and the entire word becomes [UNK]. The unaccented "cafe" tokenizes fine -- this is why BERT's uncased model strips accents.</span>

---

## <span style="font-size: 16px;">Modern Context</span>

<span style="font-size: 14px;">WordPiece was state-of-the-art when BERT was published in 2018, but tokenization has evolved significantly.</span>

<span style="font-size: 14px;">**SentencePiece (Kudo, 2018).** A language-independent library that treats input as a raw byte stream, removing the need for whitespace-based pre-tokenization. It implements both BPE and Unigram, making it applicable to languages without word boundaries (Chinese, Japanese, Thai). Used by T5, ALBERT, XLNet, and mBART.</span>

<span style="font-size: 14px;">**Byte-level BPE (Radford et al., 2019).** GPT-2 introduced BPE on UTF-8 bytes instead of Unicode characters. The base vocabulary is 256 byte values, so any text can be encoded without [UNK]. GPT-3 and GPT-4 use the same approach via tiktoken, a fast Rust-based tokenizer with a 100K vocabulary (cl100k_base).</span>

<span style="font-size: 14px;">**HuggingFace Tokenizers (2020).** A Rust-based library with fast implementations of WordPiece, BPE, and Unigram behind a unified API. Makes BERT tokenization microsecond-fast.</span>

<span style="font-size: 14px;">**Vocabulary size trends.** BERT uses 30K. GPT-2 uses 50K. T5 uses 32K. GPT-4 uses 100K. Larger vocabularies reduce sequence length but increase embedding table size.</span>

<span style="font-size: 14px;">**Multilingual challenges.** Multilingual BERT uses a shared 110K vocabulary across 104 languages. Non-Latin scripts fragment heavily, increasing sequence length -- a known issue called "fertility" disparity.</span>

<span style="font-size: 14px;">**Byte-level models.** ByT5 and MegaByte operate directly on raw bytes, bypassing tokenization but producing much longer sequences.</span>

---

## <span style="font-size: 16px;">Pitfalls and Common Mistakes</span>

* <span style="font-size: 14px;">**Greedy is not optimal.** Consider a vocabulary with "a", "ab", "##b", "##bc", "##c". For "abc", greedy matches "ab" + "##c", but "a" + "##bc" might be semantically better. Unigram avoids this via Viterbi.</span>

* <span style="font-size: 14px;">**A single unknown character kills the entire word.** If the vocabulary lacks a character, the algorithm fails at that position and returns [UNK] for the whole word. All other valid characters are lost.</span>

* <span style="font-size: 14px;">**Case sensitivity matters.** BERT's cased model treats "The" and "the" as different tokens. The uncased model lowercases everything, but then "US" (country) and "us" (pronoun) become identical.</span>

* <span style="font-size: 14px;">**The `##` prefix must be handled consistently.** Forgetting `##` for continuations, or prepending it for the first token, produces vocabulary misses leading to spurious [UNK] or embedding lookup errors.</span>

* <span style="font-size: 14px;">**Vocabulary must contain single characters as fallback.** Without them the algorithm has no base case and any uncovered character causes [UNK].</span>

* <span style="font-size: 14px;">**Tokenization is not fully reversible.** Whitespace between words is lost during pre-tokenization. The token sequence alone does not encode where spaces existed.</span>

* <span style="font-size: 14px;">**Sequence length explosion.** Rare words, URLs, and code fragments into many tokens, rapidly consuming the 512-token budget.</span>

* <span style="font-size: 14px;">**Pre-tokenization assumptions.** WordPiece assumes whitespace-delimited input. Languages without whitespace (Chinese, Japanese) need special handling -- BERT inserts spaces around each CJK character.</span>

* <span style="font-size: 14px;">**Domain vocabulary mismatch.** Terms absent from the pre-training vocabulary fragment excessively during fine-tuning, degrading domain-specific performance.</span>

---