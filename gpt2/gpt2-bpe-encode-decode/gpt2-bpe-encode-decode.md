# <span style="font-size: 20px;">BPE Encode and Decode</span>

<span style="font-size: 14px;">Byte Pair Encoding (BPE) is the tokenization algorithm used in GPT-2. Unlike word-level tokenizers, GPT-2's BPE operates at the byte level: every input string is first converted to a sequence of raw UTF-8 bytes, then a fixed, ordered list of merge rules is applied to progressively combine adjacent byte pairs into larger tokens. Decoding is the exact reverse: map each token ID back to its byte string, concatenate, and decode the resulting bytes as UTF-8. This process is lossless, guaranteeing that decode(encode(text)) always recovers the original text.</span>

---

## <span style="font-size: 16px;">What It Is</span>

<span style="font-size: 14px;">BPE tokenization in GPT-2 is a two-phase process: encoding and decoding. The encoding phase takes arbitrary text and converts it into a sequence of integer token IDs by applying a learned vocabulary of merge rules. The decoding phase reverses this mapping, recovering the original text from token IDs.</span>

<span style="font-size: 14px;">The critical insight is that BPE merges are learned during a separate training phase (on a large corpus), but at inference time we only apply them. The merge list is fixed and ordered by frequency: merge rule 0 was the most common byte pair in the training corpus, merge rule 1 was the next most common, and so on. When encoding new text, we apply these rules in exactly this priority order.</span>

<span style="font-size: 14px;">GPT-2's specific innovation was making BPE byte-level rather than character-level. Traditional BPE operates on Unicode characters, but GPT-2 treats the input as a sequence of raw bytes (values 0-255). This means the base vocabulary is always exactly 256 entries (one per possible byte value), and the model can represent any Unicode string, any binary sequence, even malformed text, without ever encountering an unknown token.</span>

<span style="font-size: 14px;">The merge rules then build up larger tokens from these byte-level building blocks. If the vocabulary has $V$ total entries, the first 256 are the individual bytes, and entries 256 through $V - 1$ are the merged tokens, each created by concatenating two existing tokens according to the corresponding merge rule.</span>

---

## <span style="font-size: 16px;">Key Equations</span>

<span style="font-size: 14px;">BPE is algorithmic rather than algebraic, but the core operations can be stated precisely.</span>

<span style="font-size: 14px;">**Encoding** maps text to token IDs:</span>

$$
\text{encode}(s) : \text{string} \rightarrow [t_0, t_1, \ldots, t_{m-1}]
$$

<span style="font-size: 14px;">where each $t_i$ is an integer token ID from the vocabulary.</span>

<span style="font-size: 14px;">**Decoding** maps token IDs back to text:</span>

$$
\text{decode}([t_0, t_1, \ldots, t_{m-1}]) = \text{utf8\_decode}\!\left(\bigoplus_{i=0}^{m-1} \text{vocab}[t_i]\right)
$$

<span style="font-size: 14px;">where $\text{vocab}[t_i]$ returns the byte string associated with token $t_i$, and $\bigoplus$ denotes byte-string concatenation.</span>

<span style="font-size: 14px;">**Merge operation:** Given a sequence of tokens $[a_0, a_1, \ldots, a_{k-1}]$ and a merge rule $(p, q) \rightarrow r$, find the leftmost adjacent pair where $a_i = p$ and $a_{i+1} = q$, and replace them with $r$:</span>

$$
[\ldots, a_{i-1}, \underbrace{p, q}_{\text{merged}}, a_{i+2}, \ldots] \;\longrightarrow\; [\ldots, a_{i-1}, r, a_{i+2}, \ldots]
$$

<span style="font-size: 14px;">This merge is applied repeatedly (left to right, rescanning from the start) for each merge rule, and the merge rules themselves are applied in priority order from rule 0 to the final rule.</span>

---

## <span style="font-size: 16px;">Encoding Step by Step</span>

<span style="font-size: 14px;">The encoding algorithm proceeds in three stages:</span>

<span style="font-size: 14px;">**Stage 1: Text to bytes.** Convert the input string to its UTF-8 byte representation. Each byte becomes an initial token, using token IDs 0-255 that directly correspond to byte values. An ASCII character like `'h'` maps to byte 104, so it starts as token ID 104. A multi-byte character like `'e'` with an accent would map to two or three initial byte tokens.</span>

<span style="font-size: 14px;">**Stage 2: Apply merge rules in order.** Iterate through the merge list from the highest-priority rule (index 0) to the lowest. For each merge rule $(p, q) \rightarrow r$:</span>

* <span style="font-size: 14px;">Scan the current token sequence from left to right.</span>
* <span style="font-size: 14px;">Whenever you find adjacent tokens $p$ and $q$, replace them with the merged token $r$.</span>
* <span style="font-size: 14px;">After each replacement, rescan from the beginning of the sequence for this same rule. The rescan is necessary because a merge can create new adjacent pairs that also match the current rule.</span>
* <span style="font-size: 14px;">Once no more matches exist for this rule, move on to the next rule in the list.</span>

<span style="font-size: 14px;">**Stage 3: Return IDs.** After all merge rules have been applied, the remaining token sequence is the encoded output. Each element is a token ID from the vocabulary.</span>

<span style="font-size: 14px;">The ordering guarantee is essential: higher-priority merges (more frequent in training data) are applied first. This means common pairs like `(t, h)` forming `th` get merged before rarer combinations. If you applied merges in a different order, you would get a different (incorrect) tokenization.</span>

<span style="font-size: 14px;">The rescan-from-start behavior for each rule is also critical. Consider the token sequence `[a, b, a, b, a, b]` with a merge rule $(a, b) \rightarrow ab$. The first pass merges positions 0-1 to get $[ab, a, b, a, b]$. Rescanning finds positions 1-2 matching, yielding $[ab, ab, a, b]$. Another rescan merges positions 2-3: $[ab, ab, ab]$. Only then does scanning find no more matches, and we move to the next rule.</span>

---

## <span style="font-size: 16px;">Decoding</span>

<span style="font-size: 14px;">Decoding is considerably simpler than encoding. Given a list of token IDs, the algorithm is:</span>

* <span style="font-size: 14px;">**Step 1:** For each token ID $t_i$, look up its corresponding byte string in the vocabulary. For base tokens (IDs 0-255), this is simply the single byte with that value. For merged tokens (IDs 256 and above), this is the byte string formed by all the merges that created it.</span>
* <span style="font-size: 14px;">**Step 2:** Concatenate all byte strings in order to form a single byte sequence.</span>
* <span style="font-size: 14px;">**Step 3:** Decode the byte sequence as UTF-8 to produce the output string.</span>

<span style="font-size: 14px;">The vocabulary lookup table is built at initialization time. Each merged token's byte string is computed recursively: if merge rule $i$ combines tokens $p$ and $q$ into token $r$, then $\text{vocab}[r] = \text{vocab}[p] \,\|\, \text{vocab}[q]$, where $\|$ is byte concatenation. Since merges reference only previously defined tokens, this recursion always terminates.</span>

<span style="font-size: 14px;">Decoding requires no knowledge of the merge rules or their ordering. The vocabulary table alone is sufficient, making decoding an $O(n)$ operation where $n$ is the number of tokens.</span>

---

## <span style="font-size: 16px;">The Round-Trip Property</span>

<span style="font-size: 14px;">A fundamental guarantee of BPE is that $\text{decode}(\text{encode}(s)) = s$ for any valid UTF-8 string $s$. This round-trip property holds because:</span>

* <span style="font-size: 14px;">**Encoding is lossless.** The initial conversion to UTF-8 bytes is a bijection (every string has exactly one UTF-8 byte representation). The merge operations only group adjacent bytes together; they never discard, reorder, or alter any byte values. The merged token's byte string is always exactly the concatenation of its constituent bytes.</span>
* <span style="font-size: 14px;">**Decoding recovers the bytes.** Each token ID maps to a unique byte string, and concatenating them in order reconstructs the exact byte sequence that was originally produced from the input text. UTF-8 decoding then recovers the original string.</span>
* <span style="font-size: 14px;">**No information is lost at any stage.** The token boundaries are the only thing that changes during encoding. The actual byte content is preserved perfectly. Decoding simply erases those token boundaries by concatenation, restoring the original flat byte sequence.</span>

<span style="font-size: 14px;">Note that encoding is not a bijection in the other direction: different merge-rule orderings could produce different token sequences for the same text. But any valid tokenization, when decoded, will always produce the same original text, because the underlying bytes are identical regardless of how they are grouped.</span>

---

## <span style="font-size: 16px;">Paper Context</span>

<span style="font-size: 14px;">GPT-2's byte-level BPE was introduced in "Language Models are Unsupervised Multitask Learners" (Radford et al., 2019). The key design decisions were:</span>

* <span style="font-size: 14px;">**Byte-level base vocabulary.** Earlier BPE implementations (Sennrich et al., 2016) operated on Unicode characters and required a pre-tokenization step to handle unknown characters. GPT-2 instead starts from raw bytes, eliminating the possibility of out-of-vocabulary tokens entirely. Any byte sequence can be represented, including any language, emoji, code, or binary data.</span>
* <span style="font-size: 14px;">**Vocabulary size of 50,257.** The base 256 byte tokens plus 50,000 merge rules, plus one special end-of-text token. This provides a balance between compression efficiency (fewer tokens per text) and vocabulary overhead.</span>
* <span style="font-size: 14px;">**Merge rules learned from training data.** During the vocabulary training phase (separate from model training), BPE iteratively counts all adjacent token pairs in the corpus, selects the most frequent pair, merges it into a new token, and repeats. After 50,000 iterations, the merge list is frozen and used for all subsequent tokenization.</span>
* <span style="font-size: 14px;">**Pre-tokenization with regex.** Before applying BPE, GPT-2 splits the input using a regex pattern that separates words, contractions, numbers, and whitespace. BPE merges never cross these pre-token boundaries. This prevents merges like `(d, space)` from absorbing whitespace into word tokens and ensures more linguistically coherent tokenization.</span>
* <span style="font-size: 14px;">**Applied at inference, not trained.** The merge rules are fixed artifacts from vocabulary training. During language model training and inference, the tokenizer simply applies these rules deterministically. The language model never modifies or learns new merges.</span>

---

## <span style="font-size: 16px;">Numerical Example</span>

<span style="font-size: 14px;">Let us trace the complete encode-decode cycle for the string `"hello"` using a simplified merge list. Assume the following ordered merge rules:</span>

* <span style="font-size: 14px;">Rule 0: $(104, 101) \rightarrow 256$ -- merges bytes for `h` and `e` into token 256 representing `"he"`</span>
* <span style="font-size: 14px;">Rule 1: $(108, 108) \rightarrow 257$ -- merges two `l` bytes into token 257 representing `"ll"`</span>
* <span style="font-size: 14px;">Rule 2: $(256, 257) \rightarrow 258$ -- merges tokens `"he"` and `"ll"` into token 258 representing `"hell"`</span>

<span style="font-size: 14px;">**Encoding:**</span>

<span style="font-size: 14px;">**Step 1 -- Text to bytes.** Convert `"hello"` to UTF-8 bytes. Since all characters are ASCII, each maps to one byte:</span>

$$
\text{"hello"} \rightarrow [104, 101, 108, 108, 111]
$$

<span style="font-size: 14px;">These byte values correspond to: `h`=104, `e`=101, `l`=108, `l`=108, `o`=111.</span>

<span style="font-size: 14px;">**Step 2 -- Apply Rule 0: $(104, 101) \rightarrow 256$.** Scan for adjacent pair (104, 101). Found at positions 0-1. Merge:</span>

$$
[104, 101, 108, 108, 111] \rightarrow [256, 108, 108, 111]
$$

<span style="font-size: 14px;">Rescan from start for Rule 0: no more (104, 101) pairs. Move to Rule 1.</span>

<span style="font-size: 14px;">**Step 3 -- Apply Rule 1: $(108, 108) \rightarrow 257$.** Scan for adjacent pair (108, 108). Found at positions 1-2. Merge:</span>

$$
[256, 108, 108, 111] \rightarrow [256, 257, 111]
$$

<span style="font-size: 14px;">Rescan from start for Rule 1: no more (108, 108) pairs. Move to Rule 2.</span>

<span style="font-size: 14px;">**Step 4 -- Apply Rule 2: $(256, 257) \rightarrow 258$.** Scan for adjacent pair (256, 257). Found at positions 0-1. Merge:</span>

$$
[256, 257, 111] \rightarrow [258, 111]
$$

<span style="font-size: 14px;">Rescan from start for Rule 2: no more (256, 257) pairs. No more rules to apply.</span>

<span style="font-size: 14px;">**Final encoding:** $[258, 111]$ -- two tokens representing `"hell"` and `"o"`.</span>

<span style="font-size: 14px;">**Decoding:**</span>

<span style="font-size: 14px;">**Step 1 -- Look up byte strings.** From the vocabulary:</span>

* <span style="font-size: 14px;">Token 258 maps to bytes $[104, 101, 108, 108]$ (the byte string for `"hell"`).</span>
* <span style="font-size: 14px;">Token 111 maps to bytes $[111]$ (the byte for `"o"`).</span>

<span style="font-size: 14px;">**Step 2 -- Concatenate.** Join the byte strings: $[104, 101, 108, 108] \,\|\, [111] = [104, 101, 108, 108, 111]$.</span>

<span style="font-size: 14px;">**Step 3 -- Decode UTF-8.** The byte sequence $[104, 101, 108, 108, 111]$ decodes to `"hello"`.</span>

<span style="font-size: 14px;">**Round-trip verified:** $\text{decode}(\text{encode}(\text{"hello"})) = \text{"hello"}$.</span>

---

## <span style="font-size: 16px;">Pitfalls</span>

<span style="font-size: 14px;">Several implementation errors commonly break BPE encoding or produce incorrect tokenizations:</span>

* <span style="font-size: 14px;">**Applying merges out of priority order.** The merge list is ordered by training frequency. Applying a lower-priority merge before a higher-priority one produces a different tokenization. For example, if rule 5 merges $(a, b)$ and rule 2 merges $(b, c)$, applying rule 5 first on the sequence $[a, b, c]$ yields $[ab, c]$, but the correct behavior is to apply rule 2 first to get $[a, bc]$. The priority ordering is the core contract of BPE encoding.</span>
* <span style="font-size: 14px;">**Not rescanning from the start after a merge.** After merging a pair, the resulting token might form a new matchable pair with its neighbors. Failing to rescan means missing valid merges. Consider $[a, b, a, b]$ with rule $(a, b) \rightarrow c$: a single left-to-right pass yields $[c, a, b]$ and stops, but correct behavior rescans and produces $[c, c]$.</span>
* <span style="font-size: 14px;">**Wrong vocabulary lookup during decode.** Each merged token's byte string must be the concatenation of its constituent tokens' byte strings, resolved recursively to base bytes. If the lookup table is built incorrectly (for example, storing the token IDs instead of the resolved bytes), decoding will produce garbage.</span>
* <span style="font-size: 14px;">**Confusing byte-level with character-level.** GPT-2 BPE operates on bytes, not Unicode characters. The character `"n"` with a tilde is UTF-8 bytes $[195, 177]$, which are two separate initial tokens, not one. Treating multi-byte characters as single tokens produces incorrect results and breaks the round-trip property for non-ASCII text.</span>
* <span style="font-size: 14px;">**Ignoring pre-tokenization boundaries.** GPT-2 splits text with a regex before applying BPE. Merges must not cross these boundaries. If you apply BPE to the entire raw string without pre-tokenization, merges can span word boundaries (for example, merging the last byte of one word with the space before the next), producing tokens that differ from GPT-2's actual output.</span>
* <span style="font-size: 14px;">**Off-by-one in token IDs.** The first 256 token IDs (0-255) are reserved for individual bytes. Merge rule $i$ creates token ID $256 + i$. Misaligning this mapping (for example, starting merged IDs at 255 or 257) corrupts both encoding and decoding.</span>
* <span style="font-size: 14px;">**Assuming merges reduce token count by exactly one each.** A single merge rule can trigger multiple replacements across the sequence if the target pair appears more than once. After applying one rule, the token count may decrease by 2 or more. Algorithms that assume exactly one merge per rule application will terminate too early.</span>