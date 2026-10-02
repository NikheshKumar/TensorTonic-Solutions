# PagedAttention Block Allocation

A KV cache grows as a model processes more tokens. Reserving one large continuous memory region for every sequence is awkward because the final sequence length is usually unknown. A short request may reserve far more space than it uses, while a growing request may need a larger region than it originally received.

Paged storage divides cache memory into fixed-size physical blocks. A sequence sees its tokens as one logical order, but the physical blocks holding those tokens can be scattered through the available memory pool.

This problem focuses only on allocation. Given sequence lengths, a block size, and an ordered free list, it builds the logical-to-physical block table needed by each sequence.

## Logical order and physical location

Imagine a sequence with ten tokens and blocks that each hold four tokens. Logically, its cache has three blocks:

- logical block 0 holds tokens 0 through 3,
- logical block 1 holds tokens 4 through 7,
- logical block 2 holds tokens 8 and 9, with two unused slots.

Those three logical blocks could map to physical blocks 7, 2, and 11. Their physical IDs are neither contiguous nor sorted, but the block table preserves the logical order:

$$
[7,2,11]
$$

When the system needs logical block 1, it looks up the second entry and finds physical block 2. The mapping provides continuity to the sequence without requiring continuous physical storage.

## How many blocks a sequence needs

A sequence of length $L_i$ with block capacity $P$ needs

$$
n_i=\left\lceil\frac{L_i}{P}\right\rceil
$$

Ceiling is important because even one leftover token requires another physical block.

With a block size of four:

- length 4 needs one block,
- length 5 needs two blocks,
- length 8 needs two blocks,
- length 9 needs three blocks.

For positive integers, ceiling division can also be understood as adding enough room for a possible remainder. An exact multiple does not create an extra block, while any nonzero remainder does.

The final block may be only partly filled. All earlier blocks for that sequence are full.

## A complete allocation example

Suppose the sequence lengths are $[8,3,5]$, the block size is 4, and the free physical IDs are

$$
[7,2,11,5,9,13]
$$

The sequences need two, one, and two blocks respectively, for a total of five.

Allocation follows the supplied free-list order:

- the first sequence receives physical blocks 7 and 2,
- the second receives physical block 11,
- the third receives physical blocks 5 and 9.

The widest row needs two entries, so the padded block table is

$$
\begin{bmatrix}
7 & 2\\
11 & -1\\
5 & 9
\end{bmatrix}
$$

The used-block counts are $[2,1,2]$, and the remaining free list is $[13]$.

The value $-1$ is padding. It is not a physical block assignment and must never be removed from the free pool.

## Preserve the caller’s free-list order

Physical IDs are distinct non-negative integers, but they may be fragmented, unsorted, and non-contiguous. Their supplied order defines allocation priority.

If the free list is $[20,4,17]$, the first allocated block is 20. Sorting it into $[4,17,20]$ would change observable behavior and violate the contract.

A single cursor through the free list is enough. Each sequence takes the next $n_i$ IDs, and the cursor advances. After all assignments, the suffix beginning at the cursor is the remaining free list.

This also guarantees that no block is assigned twice. Every allocation consumes a different position from the free list.

## Check capacity before assigning anything

The total number of required blocks is

$$
N=\sum_i n_i
$$

Allocation can succeed only when

$$
N\leq |F|
$$

where $F$ is the free-block list.

The total must be checked before building any sequence assignment. If the sequences need six blocks and only five are free, the operation raises the required error immediately.

This all-or-nothing behavior prevents partial allocation. Returning blocks for the first few sequences and failing on the last would leave the caller uncertain about which blocks were consumed. Although this exercise uses ordinary input and output values rather than a mutable allocator, it models the atomic decision expected from a safe allocation step.

Exactly enough blocks is a valid case. The allocation succeeds and returns an empty remaining free list.

## Why rows need padding

Different sequence lengths produce different block counts. A rectangular tensor cannot have a two-entry first row, a one-entry second row, and a three-entry third row without padding.

The table width is the largest block count among the sequences:

$$
W=\max_i n_i
$$

Each row places its real physical IDs first in logical order, then fills unused columns with $-1$.

The separate used-block count tells the consumer where the real portion of each row ends. Reading all columns as physical IDs would incorrectly treat padding as a block.

## What this exercise does not simulate

PagedAttention also requires attention computation that follows block tables to locate keys and values. Real serving systems manage block release, sharing, reference counts, and scheduling as requests arrive and finish.

Those behaviors are outside this function. It does not write token data, run attention, reclaim blocks, or choose an eviction policy. It performs one deterministic allocation from a supplied free pool.

Keeping that boundary clear prevents extra policies from changing the expected table and remaining IDs.

## Complexity and memory

Let $R$ be the number of sequences and $N$ the total blocks allocated. Calculating block needs costs $O(R)$. Assigning IDs and constructing real table entries costs $O(N)$.

The padded table contains $R\times W$ entries, so its output memory is $O(RW)$. Used counts require $O(R)$, and the remaining list contains $|F|-N$ entries.

The function does not allocate KV tensors themselves. It assigns identifiers representing fixed-capacity physical blocks.

Smaller blocks can reduce waste in the partially filled final block, while also making block tables longer. That general tradeoff motivates fixed-block design, but this problem treats the block size as an input and does not optimize it.

## Common mistakes to avoid

- Using floor division leaves a sequence with no space for its final partial block.
- Adding one unconditionally overallocates sequences whose lengths exactly divide the block size.
- Sorting the free IDs changes the caller-defined allocation order.
- Resetting the free-list cursor for each sequence assigns the same physical IDs more than once.
- Checking capacity during assignment permits a partial result before failure.
- Storing $-1$ among used physical IDs confuses padding with allocation.
- Removing too many IDs from the remaining list loses unallocated blocks.
- Requiring physical IDs to be contiguous rejects valid fragmented memory.

The central idea is the separation between logical sequence order and physical storage. Fixed-size blocks provide capacity, the free list determines which physical blocks are used, and the block table reconnects scattered storage into each sequence’s logical cache.
