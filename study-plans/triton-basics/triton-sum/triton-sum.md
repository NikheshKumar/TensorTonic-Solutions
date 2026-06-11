# <span style="font-size: 20px;">Sum Reduction</span>

<span style="font-size: 14px;">The reduction collapses an $N$-element vector into a single scalar, and on a GPU that scalar has to be assembled from contributions made in parallel by many independent programs. Triton's idiom is a clean two-level decomposition: each program reduces its own tile in registers with $\texttt{tl.sum}$, then a single $\texttt{tl.atomic\_add}$ per program combines the per-program partials into one output cell. The whole story is bandwidth and atomic contention, with the actual arithmetic almost free.</span>

---

## <span style="font-size: 16px;">The Operation</span>

<span style="font-size: 14px;">Given a contiguous float tensor $x \in \mathbb{R}^{N}$, the kernel writes the scalar sum into a length-1 output:</span>

$$
\texttt{out}[0] = \sum_{i=0}^{N-1} x[i]
$$

<span style="font-size: 14px;">The output buffer lives in HBM and must be pre-zeroed before launch, because the kernel reaches it through an atomic add rather than a single store.</span>

---

## <span style="font-size: 16px;">Program Decomposition</span>

<span style="font-size: 14px;">The launch grid is one-dimensional with $\lceil N / \texttt{BLOCK\_SIZE} \rceil$ **programs**. Each program is identified by $\texttt{tl.program\_id(0)}$, owns the contiguous tile $\texttt{offs} = p \cdot \texttt{BLOCK\_SIZE} + \texttt{tl.arange}(0, \texttt{BLOCK\_SIZE})$, and is responsible for exactly two things: reducing its tile to a single scalar in registers, and atomically adding that scalar into $\texttt{out}[0]$.</span>

<span style="font-size: 14px;">This is the canonical **per-array reduction** pattern. Unlike a pure map, two programs are not independent at the output level: they both write to the same scalar address. The decomposition makes that contention as cheap as it can be by ensuring each program contributes exactly one atomic, never $\texttt{BLOCK\_SIZE}$ of them. The grid can scale to thousands of programs and the output still sees only one atomic per program.</span>

<span style="font-size: 14px;">The contract with the host matters here. Because $\texttt{tl.atomic\_add}$ adds to whatever already sits at the destination, the launcher must zero $\texttt{out}$ before each call. Skipping that step is not a kernel bug but a launch-protocol bug, and it produces drift across repeated invocations rather than a single deterministic failure.</span>

---

## <span style="font-size: 16px;">Tile Shape and Masking</span>

<span style="font-size: 14px;">$\texttt{BLOCK\_SIZE} = 1024$ is the standard choice for a 1D reduction: a power of two so the compiler can pick wide vector loads and lower the in-tile reduction to a balanced $\log_2(\texttt{BLOCK\_SIZE}) = 10$-step tree, large enough that the per-program atomic overhead is amortized over real work, small enough to keep the register footprint moderate. The value is declared $\texttt{tl.constexpr}$, which is what lets the compiler size registers and unroll the reduction tree at compile time.</span>

<span style="font-size: 14px;">The mask is $\texttt{mask} = \texttt{offs} < N$ on the $\texttt{tl.load}$, paired with $\texttt{other=0.0}$ on the same call. The sentinel value is operation-specific: for a sum, masked lanes must contribute zero. Loading them with any other value pollutes $\texttt{tl.sum}$, and because the pollution is a constant per masked lane, it scales with how badly the final program overshoots. The kernel never calls $\texttt{tl.store}$, so the mask is only needed on the load side.</span>

---

## <span style="font-size: 16px;">Memory Hierarchy and Reuse</span>

<span style="font-size: 14px;">Each program loads its $\texttt{BLOCK\_SIZE}$-lane tile from HBM into registers, computes $\texttt{tl.sum(x, axis=0)}$ entirely in registers, and emits one atomic to HBM. There is no reuse: every input byte is read exactly once across the whole launch, and no two programs ever touch the same input lane. The compiler does not stage anything into shared memory for the in-tile reduction, since the reduction operates on a register-resident tile.</span>

<span style="font-size: 14px;">The interesting memory traffic is the atomic. Every program issues exactly one atomic add to the same address in HBM, and the hardware serializes those contributions through the L2 atomic unit. For a grid of $G$ programs, the atomic path costs $G$ serialized round trips to that one cache line; for the data path it costs $4 N / G$ bytes per program, fully parallel. The kernel is balanced when the atomic stream is dwarfed by the data stream, which is exactly why each program reduces its tile first instead of emitting one atomic per lane. Per-lane atomics would replace $G$ serialized round trips with $N$ of them, an enormous regression.</span>

<span style="font-size: 14px;">The L2 cache earns its keep on the atomic line: that single cache line stays hot in L2 for the duration of the launch, so each atomic is an L2 round trip rather than an HBM round trip. The input data, by contrast, is streamed cold from HBM, since no two programs share a tile and a tile that was loaded is never queried again. A useful concrete number: at $N = 10^6$ and $\texttt{BLOCK\_SIZE} = 1024$, the kernel reads $4$ MB of input ($\approx 4$ microseconds at $1$ TB/s of HBM bandwidth) and issues $\approx 1000$ atomics into L2. Atomic latency on modern hardware is in the tens of nanoseconds even when uncontended, and serializes under contention; a thousand contended atomics through L2 is on the order of microseconds, in the same ballpark as the data stream. As $G$ scales past tens of thousands of programs, the atomic stream starts to dominate, which is when a two-stage kernel begins to pay back.</span>

---

## <span style="font-size: 16px;">Memory-Bound vs Compute-Bound</span>

<span style="font-size: 14px;">Per element, the kernel reads $4$ bytes (one fp32) and performs one addition. The **arithmetic intensity** is therefore</span>

$$
\frac{1 \text{ FLOP}}{4 \text{ bytes}} = 0.25 \text{ FLOPs/byte}
$$

<span style="font-size: 14px;">That places the kernel solidly on the memory-bound side of the roofline. Modern accelerators run at tens of TFLOPs of fp32 and a few TB/s of HBM bandwidth, putting the crossover point at roughly $10$ FLOPs/byte. Sum reduction is a factor of $40\times$ below that line; the achievable runtime is set by how fast the kernel can stream the input through HBM, not by how fast the SMs can add. The atomic combine is the only place compute matters at all, and only because it serializes.</span>

<span style="font-size: 14px;">An interesting wrinkle is **floating-point associativity**. The mathematical sum is commutative and associative; the float32 sum is neither. The kernel reduces lanes inside one tile in a tree shape chosen by the compiler, and combines per-program partials in whatever order the L2 atomic unit happens to commit them. Two runs of the same kernel on the same input are not bit-exact, and a serial CPU sum almost never matches the GPU sum to the last ULP. This is a systems concern, not a bug: tolerances must be set to accommodate it. The harness widens to combined $\texttt{atol=1e-2, rtol=1e-2}$ at $N = 10^6$ for exactly this reason.</span>

<span style="font-size: 14px;">The error model is worth one more sentence. A naive serial sum of $N$ fp32 values has worst-case error proportional to $N \cdot \varepsilon$ where $\varepsilon \approx 1.2 \times 10^{-7}$. A pairwise tree reduction has error proportional to $\log_2(N) \cdot \varepsilon$, asymptotically much better. The Triton kernel here is effectively a tree of depth $\log_2(\texttt{BLOCK\_SIZE})$ inside each program plus a flat sum across $G$ programs at the atomic, so its error grows like $(\log_2(\texttt{BLOCK\_SIZE}) + G) \cdot \varepsilon$ in the worst case. For $N = 10^6$ and $G \approx 10^3$, that is much smaller than a serial CPU sum's $N \cdot \varepsilon$, so the GPU sum is often more accurate than the reference, not less. The mismatch in the last few ULPs goes both ways.</span>

---

## <span style="font-size: 16px;">Compiler-Handled vs Author-Handled</span>

<span style="font-size: 14px;">**Author chooses:** the grid shape ($\lceil N / \texttt{BLOCK\_SIZE} \rceil$ programs), the constexpr block size ($1024$), the mask predicate, the $\texttt{other=0.0}$ sentinel, the two-level decomposition (in-tile $\texttt{tl.sum}$ then one $\texttt{tl.atomic\_add}$ per program), and the host-side pre-zeroing protocol. None of those are things the compiler can infer from the source.</span>

<span style="font-size: 14px;">**Compiler handles:** lowering $\texttt{tl.load}$ to coalesced wide HBM transactions, lowering $\texttt{tl.sum}$ to a balanced reduction tree across the warps that internally shard the tile, scheduling the load and the reduction so that one can overlap with the other, and emitting the atomic as a single PTX $\texttt{atom.global.add.f32}$ instruction. The author never writes a shuffle, never declares shared memory for the reduction tree, never inserts a barrier between the per-warp partials and the cross-warp combine. In CUDA, those are the three or four things that make a reduction kernel hard to get right; Triton hides them behind the single call to $\texttt{tl.sum}$.</span>

<span style="font-size: 14px;">It is worth saying explicitly what the compiler does inside $\texttt{tl.sum}$ for a $\texttt{BLOCK\_SIZE} = 1024$ tile with the default $\texttt{num\_warps} = 4$. The tile is sharded $256$ lanes per warp. Each warp first reduces its $256$ lanes using register shuffle instructions in $\log_2(32) = 5$ steps within each $32$-lane warp, producing one partial per warp. The four warp partials are then exchanged through a small staging in shared memory, with the necessary barrier inserted by the compiler, and combined into the final scalar. The whole sequence is $\log_2(\texttt{BLOCK\_SIZE}) = 10$ logical steps. The author wrote zero lines about warps, shuffles, or barriers; the call to $\texttt{tl.sum(x, axis=0)}$ encodes all of it, and the compiler picks the lowering that matches the target architecture.</span>

---

## <span style="font-size: 16px;">Naive vs Optimized</span>

<span style="font-size: 14px;">The naive Triton sum is one atomic per lane: $\texttt{tl.atomic\_add(out\_ptr, x)}$ over the whole tile. This is correct but disastrous. It funnels every element of $x$ through a serialized atomic into one address, replacing fully parallel HBM bandwidth with single-address contention and dropping effective throughput by orders of magnitude. The optimization is to do the in-tile reduction first ($\texttt{tl.sum}$ reduces $\texttt{BLOCK\_SIZE}$ values to one in registers in a $\log_2(\texttt{BLOCK\_SIZE})$-step tree) and then atomic-add exactly one scalar per program. Atomic traffic drops from $N$ contributions to $\lceil N / \texttt{BLOCK\_SIZE} \rceil$, a factor-of-$1024$ reduction at the default block size.</span>

<span style="font-size: 14px;">The next step up, when atomic contention still bites at very large $G$, is a two-stage kernel: each program writes its partial to a per-program scratch slot with a plain $\texttt{tl.store}$ (no atomic), and a second kernel reduces the scratch buffer. That trades atomic contention for a second pass through a small intermediate buffer of size $G$. With $G \approx 10^4$, the scratch buffer is $40$ KB, easily resident in L2 for the second pass, so the second-pass cost is negligible. The single-stage form wins when $G$ is in the low thousands; the two-stage form wins when $G$ grows by another order of magnitude. The current problem size sits in the first regime, so the simpler version is correct.</span>

<span style="font-size: 14px;">A separate axis of optimization is the accumulator dtype. The kernel here works entirely in fp32, but if the input were bf16 or fp16, the accumulator would still need to be fp32 to avoid losing precision in the in-tile reduction. The $\log_2(\texttt{BLOCK\_SIZE}) = 10$-step reduction tree compounds rounding error at every step, and $10$ steps in fp16 are enough to drift in the third significant digit even for well-conditioned inputs. Casting to the storage dtype only on the final atomic add (or never, if the output is fp32) is the standard pattern, and it costs nothing because the cast happens once per program rather than once per lane.</span>

---

## <span style="font-size: 16px;">Worked Example</span>

<span style="font-size: 14px;">Take $N = 6$, $x = [1, 2, 3, 4, 5, 6]$, and $\texttt{BLOCK\_SIZE} = 4$. The launch grid is $\lceil 6 / 4 \rceil = 2$ programs, and $\texttt{out}[0]$ starts at $0.0$ after $\texttt{out.zero\_()}$.</span>

<span style="font-size: 14px;">**Program 0** ($\texttt{pid} = 0$): the tile of offsets is $[0, 1, 2, 3]$. The mask $\texttt{offs} < 6$ is $[\texttt{T}, \texttt{T}, \texttt{T}, \texttt{T}]$, all lanes active. The load fetches $x = [1, 2, 3, 4]$, $\texttt{tl.sum}$ reduces in two steps in registers (pairs sum to $[3, 7]$, then to $10$), and one $\texttt{tl.atomic\_add}$ adds $10$ to $\texttt{out}[0]$.</span>

<span style="font-size: 14px;">**Program 1** ($\texttt{pid} = 1$): the tile of offsets is $[4, 5, 6, 7]$. The mask is $[\texttt{T}, \texttt{T}, \texttt{F}, \texttt{F}]$, with the $\texttt{other=0.0}$ sentinel filling the masked lanes. The load yields $[5, 6, 0, 0]$, $\texttt{tl.sum}$ produces $11$, and one $\texttt{tl.atomic\_add}$ adds $11$ to $\texttt{out}[0]$.</span>

<span style="font-size: 14px;">The two atomics commit in some hardware-chosen order, but the final value is $0 + 10 + 11 = 21 = \sum x$ regardless. Note that across $G$ programs, only $G$ atomics ever hit the output cell, never $N$. That is the whole optimization.</span>

---

## <span style="font-size: 16px;">Pitfalls</span>

* <span style="font-size: 14px;">**Skipping $\texttt{out.zero\_()}$ before launch.** $\texttt{tl.atomic\_add}$ adds to whatever is already at the destination, so a stale value from the previous launch corrupts the result. The first invocation may happen to work if the buffer was freshly allocated and zero-initialized; subsequent invocations drift, which is the worst possible failure mode because it depends on call order.</span>
* <span style="font-size: 14px;">**Missing $\texttt{other=0.0}$ on the masked load.** Masked lanes return implementation-defined values that flow into $\texttt{tl.sum}$ and poison the partial sum. The sentinel is the only correct way to make masked lanes contribute the additive identity.</span>
* <span style="font-size: 14px;">**One atomic per lane instead of per program.** Replaces the in-tile reduction with a serialized stream of $N$ atomics into a single address. The kernel still produces the right answer eventually, but at a small fraction of the achievable bandwidth.</span>
* <span style="font-size: 14px;">**Expecting bit-exact agreement with serial CPU sums.** Parallel float32 sums reorder additions in two places: inside the tile reduction tree, and across programs at the atomic. Set tolerances with both $\texttt{atol}$ and $\texttt{rtol}$ that scale with $N$.</span>

---