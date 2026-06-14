# <span style="font-size: 20px;">Dropout (Inverted Scaling)</span>

<span style="font-size: 14px;">Dropout is two kernels wearing the same name. The training-time variant generates a fresh Bernoulli mask on the device, usually with a Philox-style counter-based RNG inside the kernel, and applies it on the fly. The inference-time variant takes an externally-provided mask, multiplies it through, and applies inverted scaling. This problem is the second variant, deliberately: it isolates the launch geometry, the elementwise pattern, and the tail-mask discipline without dragging in an on-device RNG. Once the inference form is solid, the training form is a one-line substitution of $\texttt{tl.rand}$ thresholded against $1 - p$ for the externally-loaded mask.</span>

---

## <span style="font-size: 16px;">The Operation</span>

<span style="font-size: 14px;">Given a flat input $x \in \mathbb{R}^{N}$, a precomputed $0/1$ mask of the same length, and a drop probability $p \in [0, 1)$:</span>

$$
\texttt{out}[i] = x[i] \cdot \texttt{mask}[i] \cdot \frac{1}{1 - p}, \quad 0 \le i < N
$$

<span style="font-size: 14px;">Three tensors in, one out, all the same length, all the same dtype. The kernel reads two values per output element and writes one. The mask is interpreted as fp32 in this problem (with values exactly $0.0$ or $1.0$); in production the mask is sometimes packed as uint8 or even bit-packed, but the externally-provided fp32 form keeps the kernel single-purpose.</span>

---

## <span style="font-size: 16px;">Program Decomposition</span>

<span style="font-size: 14px;">Every output element depends on exactly one element of $x$ and one of the mask. This is a pure **pointwise map**: no reduction, no cross-program communication, no shared memory, no synchronization. The launch grid is $\lceil N / \texttt{BLOCK\_SIZE} \rceil$ one-dimensional programs. Each program reads its index from $\texttt{tl.program\_id(0)}$ and owns a contiguous tile of $\texttt{BLOCK\_SIZE}$ consecutive elements at offsets $\texttt{pid} \cdot \texttt{BLOCK\_SIZE} + \texttt{tl.arange}(0, \texttt{BLOCK\_SIZE})$.</span>

<span style="font-size: 14px;">No two programs touch the same output element. Programs can execute in any order, concurrently, with no barriers and no atomics. The kernel is embarrassingly parallel at the program level - structurally identical to vector add, with an extra load (the mask) and an extra multiply (the inverted scale).</span>

<span style="font-size: 14px;">The training-time variant changes nothing about this decomposition. The only difference is that the mask load is replaced by a per-lane RNG call seeded from $\texttt{pid}$ and the lane offset, which produces deterministic-per-seed Bernoulli samples without ever materializing the mask in HBM. The launch grid is the same, the tile shape is the same, the mask discipline is the same.</span>

<span style="font-size: 14px;">The fact that the inference variant takes an externally-provided mask is also what makes this kernel testable in a deterministic harness. The same input $x$ and the same mask must produce the same output bitwise, which would not hold if the mask were sampled inside the kernel from a launch-dependent seed. Decoupling the mask from the kernel is therefore both a teaching choice and a testing affordance.</span>

---

## <span style="font-size: 16px;">Tile Shape and Masking</span>

<span style="font-size: 14px;">$\texttt{BLOCK\_SIZE} = 1024$ is declared $\texttt{tl.constexpr}$. The same considerations as vector add apply: a power of two so the compiler picks wide vector loads, large enough that launch overhead is amortized over real work, small enough that the register footprint stays inside the per-program budget. There is no reduction to size, no inner $K$ loop to pipeline, and the choice is essentially fixed.</span>

<span style="font-size: 14px;">Because $\texttt{BLOCK\_SIZE}$ is fixed at compile time but $N$ is a runtime value, the last program almost always overshoots. For $N = 1000$ with $\texttt{BLOCK\_SIZE} = 1024$, the single program covers offsets $0..1023$, of which lanes $1000..1023$ are past the end. The mask $\texttt{in\_bounds} = \texttt{offsets} < n$ disables those lanes on every $\texttt{tl.load}$ (so $x$ and the dropout mask do not read past their buffers) and on the $\texttt{tl.store}$ (so $\texttt{out}$ does not write past its buffer).</span>

<span style="font-size: 14px;">There is a vocabulary collision worth naming explicitly here: the kernel has both a **dropout mask** (the externally-provided $0/1$ tensor that decides which elements survive) and a **tile mask** (the boolean predicate that disables out-of-range lanes). They are unrelated. The dropout mask is data; the tile mask is a Triton correctness construct. Confusing them is the second-most-common bug in dropout kernels after forgetting the inverted scale.</span>

---

## <span style="font-size: 16px;">Memory Hierarchy and Reuse</span>

<span style="font-size: 14px;">Each output element comes from one HBM load of $x$, one HBM load of the dropout mask, and one HBM store of $\texttt{out}$. Total per element: $4 + 4 = 8$ bytes read, $4$ bytes written, $12$ bytes moved per element. Nothing is reused. The tile of $x$ that one program holds is consumed by the multiply chain and discarded; no other program will ever query it.</span>

<span style="font-size: 14px;">There is no shared memory in the kernel. The compiler does not stage anything into SRAM because the access pattern is contiguous and the data is consumed immediately. Both loads use $\texttt{other} = 0.0$ on the masked lanes: a zero in either factor produces a zero in the product, so masked lanes do not leak garbage even before the store mask is checked.</span>

<span style="font-size: 14px;">The single hardware feature that matters is **coalesced** HBM access. Contiguous lane offsets within a tile let the compiler emit each load as a small number of wide vector instructions instead of $\texttt{BLOCK\_SIZE}$ separate transactions. For $\texttt{BLOCK\_SIZE} = 1024$ fp32 lanes, the tile is $4096$ bytes, which the compiler lowers to roughly $32$ HBM transactions of $128$ bytes each per tensor. Strided access would multiply the transaction count by the stride and tank effective bandwidth; the contiguous offsets here trivially avoid this.</span>

<span style="font-size: 14px;">Both loads use the same offsets, so the compiler can sometimes issue them as part of the same warp-level memory request stream and hide the second load's latency behind the first. The author writes two separate $\texttt{tl.load}$ calls and the codegen schedules them; there is no manual prefetching or explicit overlap to write.</span>

---

## <span style="font-size: 16px;">Memory-Bound vs Compute-Bound</span>

<span style="font-size: 14px;">Per output element the kernel does roughly $2$ FLOPs: one multiply by the mask and one multiply by the inverted scale. (The scale $1 / (1 - p)$ is a per-kernel scalar computed once before the loads, so it does not contribute meaningfully to the FLOP count.) Bytes moved are $12$ per element. Arithmetic intensity:</span>

$$
\frac{2 \text{ FLOPs}}{12 \text{ bytes}} \approx 0.17 \text{ FLOPs/byte}
$$

<span style="font-size: 14px;">That puts dropout firmly in **memory-bound** territory, only slightly above vector add ($\approx 0.08$) and well below the roofline crossover. The kernel runs at HBM bandwidth on any modern accelerator, and the only optimizations that change runtime are the ones that reduce bytes moved.</span>

<span style="font-size: 14px;">The most realistic byte-saving optimization is to bit-pack the dropout mask. Storing one bit per element instead of one fp32 cuts the mask's HBM traffic by $32\times$ and reduces the total to $4 + 0.125 + 4 = 8.125$ bytes per element. That changes the intensity to $\approx 0.25$ FLOPs/byte and gives back roughly a third of the kernel's runtime in the bandwidth-bound regime. This problem keeps the fp32 mask form because it isolates the launch geometry; a real production kernel would bit-pack.</span>

---

## <span style="font-size: 16px;">Compiler-Handled vs Author-Handled</span>

<span style="font-size: 14px;">**Author chooses:** the grid ($\lceil N / \texttt{BLOCK\_SIZE} \rceil$ programs), the constexpr block size, the offset arithmetic ($\texttt{pid} \cdot \texttt{BLOCK\_SIZE} + \texttt{tl.arange}$), the tile mask $\texttt{offsets} < n$, the $\texttt{other} = 0.0$ on both loads, and the fusion of the mask multiply and the scale multiply into one register expression. The choice to apply inverted scaling (multiplying by $1 / (1 - p)$ on the kept activations) rather than at-inference scaling (multiplying by $1 - p$ on every activation) is also the author's; it is what lets the deployed model run with no scaling at all once dropout is disabled.</span>

<span style="font-size: 14px;">**Compiler handles:** lowering $\texttt{tl.load}$ and $\texttt{tl.store}$ to wide PTX memory instructions of the right vector width, allocating registers for the two tiles, scheduling the two loads to overlap, picking the warp partition of the tile inside the program, and emitting the final machine code. The author never names a warp, never declares a coalescing rule, never inserts a barrier - the kernel has no synchronization to insert.</span>

<span style="font-size: 14px;">For the training-time variant, the boundary shifts slightly. The author additionally chooses the RNG seed scheme (typically a per-program seed derived from a global counter plus the program ID) and the threshold $\texttt{tl.rand}(\dots) > p$, but the compiler still handles the actual Philox state evolution under $\texttt{tl.rand}$. The result is a kernel that produces deterministic-per-seed dropout masks without ever touching HBM for the mask tensor.</span>

---

## <span style="font-size: 16px;">Naive vs Optimized</span>

<span style="font-size: 14px;">A naive PyTorch eager implementation is three kernels: one to multiply $x$ by the mask, one to multiply by the inverted scale, and one to write the result back. (In practice eager mode fuses some of these via the elementwise scheduler, but the worst case is three round-trips.) The Triton form does it in one.</span>

<span style="font-size: 14px;">More interestingly, the inference-time variant exists at all only because the training-time variant uses inverted scaling. If dropout scaled at inference time (multiplying every activation by $1 - p$ during deployment), the inference form would have one fewer multiply per element but the deployed model would carry a permanent scale operation through every forward pass. Inverted scaling pushes the cost to training and lets inference run the original network unmodified. That choice is purely a kernel-economics decision about where to spend the FLOPs, and it is upstream of any specific implementation.</span>

<span style="font-size: 14px;">A real optimization worth naming: the fused form where dropout is combined with whatever produced $x$. Activation followed by dropout (GELU + dropout, for example) is canonically a fused kernel that reads $x$ once, computes the activation in registers, multiplies by the dropout mask in registers, and writes back. The standalone dropout kernel here is the unfused baseline; the fused form removes one HBM round-trip on $x$ and is the form actually deployed in transformer training stacks.</span>

<span style="font-size: 14px;">A second axis is mask precision. An fp32 mask with values exactly $0.0$ or $1.0$ wastes most of the storage bandwidth: a single bit per element would carry the same information. Production kernels often pack the mask as uint8 (one byte per element, still $4\times$ smaller than fp32) or as a bitmap ($32\times$ smaller). The kernel structure is unchanged; only the load's dtype and the cast to fp32 before the multiply change.</span>

---

## <span style="font-size: 16px;">Worked Example</span>

<span style="font-size: 14px;">Take $N = 6$, $\texttt{BLOCK\_SIZE} = 4$, $p = 0.5$ (so the scale is $1 / 0.5 = 2$), $x = [1, 2, 3, 4, 5, 6]$, and the externally-provided dropout mask $[1, 0, 1, 0, 1, 0]$. The launch grid is $\lceil 6 / 4 \rceil = 2$ programs.</span>

<span style="font-size: 14px;">**Program 0** ($\texttt{pid} = 0$): offsets are $[0, 1, 2, 3]$, tile mask is $[T, T, T, T]$. Load $x = [1, 2, 3, 4]$ and dropout mask $[1, 0, 1, 0]$. Compute lane-wise $x \cdot \texttt{mask} \cdot 2 = [2, 0, 6, 0]$, and store into $\texttt{out}[0..3]$.</span>

<span style="font-size: 14px;">**Program 1** ($\texttt{pid} = 1$): offsets are $[4, 5, 6, 7]$, tile mask is $[T, T, F, F]$. Load $x$ at the two in-bounds lanes ($x[4] = 5, x[5] = 6$) and dropout mask ($[1, 0]$); the two out-of-bounds lanes return $0.0$ via $\texttt{other} = 0.0$. The product is $[10, 0, 0, 0]$, but the store mask gates lanes $2$ and $3$ off, so only $\texttt{out}[4] = 10$ and $\texttt{out}[5] = 0$ are written.</span>

<span style="font-size: 14px;">The final output is $[2, 0, 6, 0, 10, 0]$. The kept elements have been scaled by $2$; the dropped elements are zero. The expected value over the dropout distribution is the original $x$, which is the whole point of inverted scaling. If the kernel skipped the scale, the kept elements would be $[1, 0, 3, 0, 5, 0]$ and the expected value would be $0.5 \cdot x$, biasing every downstream activation by a factor of $1 - p$ at inference time.</span>

---

## <span style="font-size: 16px;">Pitfalls</span>

* <span style="font-size: 14px;">**Confusing the dropout mask with the tile mask.** They are unrelated. The dropout mask is data with semantic meaning; the tile mask is a Triton correctness predicate. A common bug is to combine them into one boolean and lose the inverted scale; another is to apply the dropout mask to the store and the tile mask to the load, leaving lanes with stale values.</span>
* <span style="font-size: 14px;">**Forgetting the inverted scale.** Multiplying by the mask alone produces the right zero pattern but the wrong expected value. The model trains to expect the inverted-scaled activations, and skipping the $1 / (1 - p)$ factor leaves the network unable to compensate for the dropped elements at inference time.</span>
* <span style="font-size: 14px;">**Dividing by $1 - p$ inside the inner loop.** The scale is a per-kernel scalar; computing it once before the loads and multiplying by the precomputed reciprocal is materially cheaper than computing $1 / (1 - p)$ on every lane every step. The Triton compiler will usually hoist this on its own, but writing it explicitly is the safer pattern.</span>
* <span style="font-size: 14px;">**Treating the kernel as compute-bound.** Dropout sits next to vector add on the roofline. Adding more arithmetic per element (an activation, a bias, an L1 penalty) is essentially free up to a point because the kernel is dominated by HBM traffic; this is the motivation for fusing activation + dropout into one kernel rather than treating them as separable steps.</span>

---