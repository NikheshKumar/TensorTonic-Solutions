# <span style="font-size: 20px;">BPE Training</span>

<span style="font-size: 14px;">Byte Pair Encoding (BPE) is a subword tokenization algorithm that learns a vocabulary by iteratively merging the most frequent adjacent pair of tokens. Originally a data compression technique (Gage, 1994), it was adapted for NLP by Sennrich et al. (2016) and became the tokenization backbone of GPT-2 (Radford et al., 2019), where it operates on raw bytes rather than Unicode characters.</span>

---

## <span style="font-size: 16px;">What It Is</span>

<span style="font-size: 14px;">BPE training is a vocabulary construction algorithm. Given a corpus of text and a target vocabulary size $V$, it starts with a base vocabulary of individual tokens and repeatedly merges the most frequent adjacent pair into a single new token. Each merge adds one entry to the vocabulary and reduces the total token count of the corpus. The process stops when the vocabulary reaches size $V$.</span>

<span style="font-size: 14px;">The output of BPE training is twofold: a vocabulary mapping token strings to integer IDs, and an ordered list of merge rules. At inference time, tokenization replays these merges in learned order to segment new text. Common words become single tokens; rare words decompose into subword pieces carrying partial meaning.</span>

<span style="font-size: 14px;">Frequency drives the merges. Pairs that co-occur most often get merged first, so common sequences like "th", "he", "in" become single tokens early, while rare sequences stay decomposed. This naturally balances vocabulary size against sequence length.</span>

---

## <span style="font-size: 16px;">Key Equations</span>

<span style="font-size: 14px;">BPE training is algorithmic rather than equation-driven, but the core operations can be expressed formally.</span>

<span style="font-size: 14px;">**Pair counting.** Given a token sequence $T = [t_0, t_1, \ldots, t_{n-1}]$, count every adjacent pair:</span>

$$
\text{count}(a, b) = \sum_{i=0}^{n-2} \mathbb{1}[t_i = a \;\text{and}\; t_{i+1} = b]
$$

<span style="font-size: 14px;">**Best pair selection.** Find the pair with the highest count:</span>

$$
(a^*, b^*) = \arg\max_{(a,b)} \text{count}(a, b)
$$

<span style="font-size: 14px;">When multiple pairs share the maximum count, a deterministic tie-breaking rule is applied (discussed in detail below).</span>

<span style="font-size: 14px;">**Merge operation.** Replace every occurrence of $(a^*, b^*)$ in the token sequence with a new token $c$:</span>

$$
\text{new\_id}(c) = |\text{vocab}|
$$

<span style="font-size: 14px;">where $|\text{vocab}|$ is the current vocabulary size before the merge.</span>

<span style="font-size: 14px;">**Vocabulary expansion.** After each merge, the vocabulary grows by one:</span>

$$
\text{vocab} \leftarrow \text{vocab} \cup \{c\}, \quad |\text{vocab}| \leftarrow |\text{vocab}| + 1
$$

<span style="font-size: 14px;">The process repeats until $|\text{vocab}|$ equals the target size $V$. If the base vocabulary has $B$ tokens, exactly $V - B$ merge operations are performed.</span>

---

## <span style="font-size: 16px;">The BPE Algorithm Step by Step</span>

<span style="font-size: 14px;">The complete BPE training procedure works as follows:</span>

* <span style="font-size: 14px;">**Step 1 -- Initialize the token sequence.** Convert the entire training corpus into a sequence of base tokens. Each base token has a unique integer ID. In standard BPE, base tokens are characters; in byte-level BPE, they are the 256 possible byte values (IDs 0 through 255).</span>
* <span style="font-size: 14px;">**Step 2 -- Count all adjacent pairs.** Scan the token sequence left to right. For every position $i$, record the pair $(t_i, t_{i+1})$ and increment its count.</span>
* <span style="font-size: 14px;">**Step 3 -- Find the most frequent pair.** Identify the pair $(a^*, b^*)$ with the highest count. If there is a tie, apply the tie-breaking rule: select the pair with the smallest first token ID; if still tied, the smallest second token ID.</span>
* <span style="font-size: 14px;">**Step 4 -- Create a new token.** Assign the new merged token an ID equal to the current vocabulary size. For example, if the vocabulary currently has 258 tokens (IDs 0-257), the new token gets ID 258. Add it to the vocabulary.</span>
* <span style="font-size: 14px;">**Step 5 -- Merge in the token sequence.** Scan the token sequence left to right. Whenever the pair $(a^*, b^*)$ is found at positions $i$ and $i+1$, replace both with the new merged token. After merging at position $i$, the next scan position is $i$ (not $i+1$), because the token previously at $i+2$ has shifted to $i+1$ and may form a new match.</span>
* <span style="font-size: 14px;">**Step 6 -- Repeat.** Go back to Step 2. Continue until the vocabulary reaches the target size $V$.</span>

<span style="font-size: 14px;">Each iteration reduces the length of the token sequence (every merge replaces two tokens with one) while growing the vocabulary by exactly one entry. After $V - B$ iterations, training is complete.</span>

---

## <span style="font-size: 16px;">Byte-Level BPE</span>

<span style="font-size: 14px;">Classical BPE operates on characters: the base vocabulary contains every unique character in the training corpus. Any character absent from training produces an unknown token at inference. Unicode has over 143,000 characters, so covering all as base tokens is impractical.</span>

<span style="font-size: 14px;">Radford et al. (2019) solved this in GPT-2 by starting from bytes instead of characters. Every piece of text, in any language, is ultimately a sequence of bytes with values 0 through 255. By using these 256 byte values as the base vocabulary, byte-level BPE encodes any input without unknown tokens. This was GPT-2's key tokenization innovation.</span>

<span style="font-size: 14px;">The base vocabulary is exactly 256 tokens, with IDs 0 through 255. Merged tokens start at ID 256. A target of 50,257 tokens (GPT-2) means 50,001 merge operations on top of the 256 byte tokens, plus special tokens.</span>

<span style="font-size: 14px;">Byte-level BPE can tokenize any UTF-8 string. A Chinese character encoded as three UTF-8 bytes starts as three byte tokens, and common characters get merged into single tokens through training. Rare characters stay as constituent bytes. There is no `[UNK]` token and no need for language-specific pre-processing.</span>

---

## <span style="font-size: 16px;">Tie-Breaking</span>

<span style="font-size: 14px;">When two or more pairs share the same maximum frequency count, the algorithm must select one deterministically. Without a consistent rule, different implementations learn different vocabularies from the same corpus.</span>

* <span style="font-size: 14px;">**Primary key: smallest first token ID.** Among all pairs tied at the maximum count, prefer the pair whose first (left) token has the smallest integer ID.</span>
* <span style="font-size: 14px;">**Secondary key: smallest second token ID.** If multiple pairs share both the maximum count and the same first token ID, prefer the pair whose second (right) token has the smallest integer ID.</span>

<span style="font-size: 14px;">Formally, given tied pairs $\{(a_1, b_1), (a_2, b_2), \ldots\}$, select the pair $(a_i, b_i)$ that minimizes $(a_i, b_i)$ under lexicographic ordering of integer pairs. This means $(3, 7)$ beats $(3, 9)$, which beats $(5, 1)$.</span>

<span style="font-size: 14px;">This rule is critical for correctness. A different tie-breaking choice at any step changes which merge happens, which changes all subsequent pair counts, cascading into a completely different vocabulary.</span>

---

## <span style="font-size: 16px;">The Merge Scan</span>

<span style="font-size: 14px;">The merge operation -- replacing all occurrences of the best pair -- has subtle mechanics that are easy to get wrong. The scan always proceeds left to right.</span>

<span style="font-size: 14px;">**Left-to-right scanning.** Starting at position 0, examine $(t_i, t_{i+1})$. If they match the target pair, replace them with the new token. The sequence shrinks by one at that position.</span>

<span style="font-size: 14px;">**After merging, resume from position $i$, not $i+1$.** After merging at position $i$, the token previously at $i+2$ shifts to $i+1$. The scan must check whether the new token at $i$ and the shifted token at $i+1$ also form the target pair. Jumping to $i+1$ skips this check and misses valid merges.</span>

<span style="font-size: 14px;">**Overlapping pairs.** Consider $[a, a, a]$ with best pair $(a, a)$. Left-to-right scanning merges positions 0-1 into the new token, leaving $[\text{new}, a]$. Only one merge happens, not two. The pair at original positions 1-2 was consumed.</span>

<span style="font-size: 14px;">**Non-overlapping example.** Given $[a, b, a, b, a, b]$ and pair $(a, b)$: merge at 0 gives $[c, a, b, a, b]$, merge at 1 gives $[c, c, a, b]$, merge at 2 gives $[c, c, c]$. Three merges from six tokens.</span>

---

## <span style="font-size: 16px;">Paper Context</span>

<span style="font-size: 14px;">The GPT-2 paper (Radford et al., "Language Models are Unsupervised Multitask Learners", 2019) introduced byte-level BPE to create a general-purpose language model handling any text without preprocessing or language-specific rules.</span>

<span style="font-size: 14px;">GPT-2's vocabulary has 50,257 tokens: 256 byte tokens, 50,000 merges, and one special end-of-text token. The vocabulary was trained on WebText, a corpus of approximately 8 million web pages (40 GB of text). A regex-based pre-tokenization step splits text on whitespace and punctuation boundaries before BPE, preventing merges across word boundaries.</span>

<span style="font-size: 14px;">The paper notes that byte-level BPE "only requires a base vocabulary of size 256" and "is capable of encoding any string," eliminating unknown tokens entirely. This departed from BERT's WordPiece, which produces `[UNK]` for characters outside its training distribution. GPT-2's approach was inherited by GPT-3 (Brown et al., 2020) and powers GPT-4's tokenizer (cl100k_base, 100,277 tokens) via tiktoken.</span>

---

## <span style="font-size: 16px;">Numerical Example</span>

<span style="font-size: 14px;">Consider the text "aaabbc" with a base vocabulary of individual characters and a target of 3 merge iterations.</span>

<span style="font-size: 14px;">**Initial state:**</span>

* <span style="font-size: 14px;">**Token sequence:** $[a, a, a, b, b, c]$ using IDs $[0, 0, 0, 1, 1, 2]$</span>
* <span style="font-size: 14px;">**Vocabulary:** $\{a: 0, \; b: 1, \; c: 2\}$ (size 3)</span>

<span style="font-size: 14px;">**Merge 1:**</span>

* <span style="font-size: 14px;">**Pair counts:** $(0,0)$: 2, $(0,1)$: 1, $(1,1)$: 1, $(1,2)$: 1</span>
* <span style="font-size: 14px;">**Best pair:** $(0, 0)$ with count 2. No tie-breaking needed.</span>
* <span style="font-size: 14px;">**New token:** "aa" gets ID 3.</span>
* <span style="font-size: 14px;">**Merge scan:** Position 0: $[0, 0]$ match, merge to 3. Resume from 0: $[3, 0]$ no match, advance. Position 1: $[0, 1]$ no match. Continue through remaining positions, no more matches.</span>
* <span style="font-size: 14px;">**Result:** $[3, 0, 1, 1, 2]$ = $[\text{aa}, a, b, b, c]$. Vocab size: 4.</span>

<span style="font-size: 14px;">**Merge 2:**</span>

* <span style="font-size: 14px;">**Pair counts:** $(3,0)$: 1, $(0,1)$: 1, $(1,1)$: 1, $(1,2)$: 1</span>
* <span style="font-size: 14px;">**Best pair:** Four-way tie at count 1. Tie-breaking: smallest first ID is 0, giving pair $(0,1)$. Select $(0, 1)$.</span>
* <span style="font-size: 14px;">**New token:** "ab" gets ID 4.</span>
* <span style="font-size: 14px;">**Merge scan:** Position 0: $[3, 0]$ no match. Position 1: $[0, 1]$ match, merge to 4. Resume from 1: $[4, 1]$ no match. Position 2: $[1, 2]$ no match.</span>
* <span style="font-size: 14px;">**Result:** $[3, 4, 1, 2]$ = $[\text{aa}, \text{ab}, b, c]$. Vocab size: 5.</span>

<span style="font-size: 14px;">**Merge 3:**</span>

* <span style="font-size: 14px;">**Pair counts:** $(3,4)$: 1, $(4,1)$: 1, $(1,2)$: 1</span>
* <span style="font-size: 14px;">**Best pair:** Three-way tie at count 1. Smallest first ID is 1 from pair $(1,2)$. Select $(1, 2)$.</span>
* <span style="font-size: 14px;">**New token:** "bc" gets ID 5.</span>
* <span style="font-size: 14px;">**Merge scan:** Position 0: $[3, 4]$ no match. Position 1: $[4, 1]$ no match. Position 2: $[1, 2]$ match, merge to 5.</span>
* <span style="font-size: 14px;">**Result:** $[3, 4, 5]$ = $[\text{aa}, \text{ab}, \text{bc}]$. Vocab size: 6.</span>

<span style="font-size: 14px;">After 3 merges, the 6-character input is represented by 3 tokens. The merge list is: $(0,0) \to 3$, $(0,1) \to 4$, $(1,2) \to 5$. At inference time, these merges are replayed in this exact order to tokenize new text.</span>

---

## <span style="font-size: 16px;">Modern Context</span>

<span style="font-size: 14px;">BPE has become the dominant tokenization algorithm in modern language models, though implementations have evolved considerably since GPT-2.</span>

<span style="font-size: 14px;">**tiktoken (OpenAI, 2022).** OpenAI's open-source tokenizer library, written in Rust with Python bindings. The GPT-4 tokenizer (cl100k_base) uses 100,277 tokens trained on a larger, more multilingual corpus. It is much faster than the original Python tokenizer because it uses a pre-compiled token-to-rank mapping rather than iterative merge replay.</span>

<span style="font-size: 14px;">**SentencePiece (Kudo and Richardson, 2018).** A language-independent tokenizer that treats input as a raw byte stream, eliminating whitespace-based pre-tokenization. It implements both BPE and Unigram and is used by LLaMA, Mistral, and T5.</span>

<span style="font-size: 14px;">**Vocabulary size trends.** GPT-2 uses 50,257 tokens. LLaMA uses 32,000. GPT-4 uses 100,277. Larger vocabularies compress text more aggressively but increase the embedding table size.</span>

<span style="font-size: 14px;">**BPE vs. alternatives.** WordPiece (BERT) selects merges by likelihood ratio rather than frequency. Unigram (T5, XLNet) starts with a large vocabulary and prunes tokens that least reduce corpus likelihood. Despite these alternatives, BPE remains dominant due to its simplicity and the GPT family's strong results.</span>

---

## <span style="font-size: 16px;">Pitfalls</span>

* <span style="font-size: 14px;">**Wrong tie-breaking order.** Using the wrong rule or none at all means a dictionary's arbitrary iteration order picks the winner among tied pairs, producing non-deterministic results across platforms. The correct rule: smallest first token ID, then smallest second. Getting this wrong at merge $k$ changes the entire vocabulary from that point.</span>

* <span style="font-size: 14px;">**Wrong scan position after merge.** After merging at position $i$, the next check must start from $i$, not $i+1$. Advancing to $i+1$ skips checking whether the new token at $i$ pairs with the shifted token at $i+1$. This causes missed merges, but the bug only manifests when the token following a merged pair also forms a valid match.</span>

* <span style="font-size: 14px;">**Off-by-one in new token IDs.** New token IDs must start at the current vocabulary size and increment by one per merge. A common mistake is starting at $B + 1$ instead of $B$ (where $B$ is the base vocabulary size), or incrementing the ID before adding to the vocabulary. With 256 byte tokens, the first merged token must be ID 256, the second 257, and so on.</span>

* <span style="font-size: 14px;">**Not handling overlapping merges correctly.** When the best pair is $(a, a)$ and the sequence contains $[a, a, a]$, left-to-right scanning merges positions 0-1, leaving $[\text{new}, a]$. Only one merge happens, not two. Processing all pairs simultaneously produces incorrect results.</span>

* <span style="font-size: 14px;">**Confusing BPE training with BPE encoding.** Training builds the vocabulary and merge list from a corpus by counting pairs globally. Encoding applies learned merges to new text in learned order. Mixing the two, such as re-counting frequencies during encoding, produces wrong tokenizations.</span>

* <span style="font-size: 14px;">**Forgetting that merge order matters at inference.** BPE encoding must replay merges in the exact order learned during training. Applying merge 47 before merge 12 produces different tokenizations because earlier merges create tokens that later merges depend on. The merge list is an ordered sequence, not a set.</span>

* <span style="font-size: 14px;">**Assuming byte values map to printable characters.** In byte-level BPE, IDs 0-255 are raw byte values, not ASCII characters. GPT-2 maps bytes to printable Unicode for display, but this is cosmetic. Confusing the display mapping with actual byte values leads to encoding errors.</span>

---