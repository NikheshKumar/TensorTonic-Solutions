# <span style="font-size: 20px;">Subword Tokenization</span>

<span style="font-size: 14px;">Subword tokenization is the strategy behind every modern language model's tokenizer (BERT, GPT, T5). Instead of treating whole words or individual characters as tokens, it breaks words into variable-length pieces that balance vocabulary size with coverage of unseen words.</span>

---

## <span style="font-size: 16px;">The Problem with Word-Level Tokenization</span>

* <span style="font-size: 14px;">A fixed vocabulary cannot handle every word. Rare or novel words become an unknown token</span> `[UNK]`
* <span style="font-size: 14px;">Inflections create redundancy: "play", "plays", "played", "playing" are four separate vocabulary entries that share a common root</span>
* <span style="font-size: 14px;">Vocabulary size grows without bound as the corpus expands</span>
* <span style="font-size: 14px;">Subword tokenization solves all three issues: "playing" becomes ["play", "##ing"], reusing the root and suffix across many words</span>

---

## <span style="font-size: 16px;">WordPiece Tokenization</span>

<span style="font-size: 14px;">WordPiece, introduced by Schuster and Nakajima (2012) and popularized by BERT, uses a greedy longest-match algorithm to segment words into subword tokens from a learned vocabulary.</span>

### <span style="font-size: 14px;">The Algorithm</span>

* <span style="font-size: 14px;">Start at the beginning of the word</span>
* <span style="font-size: 14px;">Find the longest substring starting at the current position that exists in the vocabulary</span>
* <span style="font-size: 14px;">The first piece has no prefix; all continuation pieces are prefixed with "##" to mark them as non-initial</span>
* <span style="font-size: 14px;">Advance past the matched substring and repeat</span>
* <span style="font-size: 14px;">If no match is found, fall back to a single character</span>

### <span style="font-size: 14px;">Example</span>

<span style="font-size: 14px;">Given the word "unhappy" and a vocabulary containing "un" and "##happy":</span>

* <span style="font-size: 14px;">Position 0: try "unhappy", "unhapp", ..., "un" (match). Emit "un"</span>
* <span style="font-size: 14px;">Position 2: try "##happy" (match). Emit "##happy"</span>
* <span style="font-size: 14px;">Result: ["un", "##happy"]</span>

---

## <span style="font-size: 16px;">The "##" Prefix Convention</span>

<span style="font-size: 14px;">The "##" prefix serves a critical purpose: it distinguishes word-initial tokens from continuation tokens.</span>

* <span style="font-size: 14px;">"un" as a first piece means the word starts with "un"</span>
* <span style="font-size: 14px;">"##happy" means "happy" is a continuation of the previous piece, not a standalone word</span>
* <span style="font-size: 14px;">This distinction matters for reconstruction: given ["un", "##happy"], the model knows to join them into "unhappy" rather than treating them as two separate words</span>
* <span style="font-size: 14px;">Different tokenizers use different prefixes: BERT uses "##", SentencePiece uses a leading space marker</span>

---

## <span style="font-size: 16px;">Greedy vs Optimal Segmentation</span>

* <span style="font-size: 14px;">**Greedy longest-match** (this problem): always picks the longest possible token at each position. Fast and simple</span>
* <span style="font-size: 14px;">**Optimal segmentation** (Unigram model): considers all possible segmentations and picks the one with the highest probability. Used by SentencePiece</span>
* <span style="font-size: 14px;">Greedy is not always optimal. For the word "abc" with vocabulary {"ab", "##c", "a", "##bc"}, greedy picks ["ab", "##c"] but ["a", "##bc"] might be preferred if "##bc" is more frequent</span>
* <span style="font-size: 14px;">In practice, greedy works well because the vocabulary is trained to make greedy segmentations reasonable</span>

---

## <span style="font-size: 16px;">Subword Tokenization Methods</span>

* <span style="font-size: 14px;">**BPE (Byte Pair Encoding):** iteratively merges the most frequent adjacent pair of tokens. Used by GPT-2, GPT-3, GPT-4. Covered in the next problem</span>
* <span style="font-size: 14px;">**WordPiece:** similar to BPE but selects merges based on likelihood rather than frequency. Used by BERT</span>
* <span style="font-size: 14px;">**Unigram:** starts with a large vocabulary and prunes tokens that least reduce the likelihood. Used by T5, XLNet via SentencePiece</span>
* <span style="font-size: 14px;">All three produce similar results; the key insight they share is that a compact vocabulary of subword pieces can represent any word</span>

---

## <span style="font-size: 16px;">Common Interview Follow-ups</span>

* <span style="font-size: 14px;">**Why not character-level tokenization?** Character-level creates very long sequences (a 10-word sentence might be 50+ tokens), making attention computation expensive since self-attention is</span> $O(n^2)$<span style="font-size: 14px;">. Subwords keep sequences short while maintaining full coverage</span>
* <span style="font-size: 14px;">**How is the vocabulary learned?** The vocabulary is built during a training phase on a large corpus. BPE starts with characters and merges frequent pairs. WordPiece and Unigram use statistical criteria. This is covered in the BPE Algorithm and Tokenizer Training problems</span>
* <span style="font-size: 14px;">**What happens with completely unknown characters?** If a character is not in the vocabulary at all (e.g., a rare Unicode symbol), some tokenizers map it to a special</span> `[UNK]` <span style="font-size: 14px;">token. Byte-level BPE (used by GPT-2+) avoids this by operating on bytes rather than characters, guaranteeing full coverage</span>
* <span style="font-size: 14px;">**Why does BERT use WordPiece while GPT uses BPE?** Historical choices. Both produce similar quality tokenizations. The main practical difference is the merge selection criterion during vocabulary training</span>

---