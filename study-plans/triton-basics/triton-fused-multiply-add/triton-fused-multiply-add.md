# <span style="font-size: 20px;">Fused Multiply-Add</span>

<span style="font-size: 14px;">Fused multiply-add scales one vector and accumulates a second into the output in a single pass. It is a **pointwise map** with exactly one more arithmetic operation per lane than vector addition, and the canonical demonstration of why fusion is the first lever a Triton author reaches for. The kernel reads two tensors, writes one, and folds a scalar multiply and a vector add into a single hardware instruction per lane.</span>

---

## <span style="font-size: 16px;">The Operation</span>

<span style="font-size: 14px;">Given a Python float $a$ and two contiguous float tensors $x, y \in \mathbb{R}^{N}$, the kernel writes</span>

$$
\texttt{out}[i] = a \cdot x[i] + y[i], \quad 0 \le i < N
$$

<span style="font-size: 14px;">All three tensors are 1D, the same length $N$, the same dtype (`torch.float32`), and resident in HBM. The scalar $a$ is passed by value through the kernel signature; it does not need a pointer or a load. The launcher allocates $\texttt{out}$ and the kernel fills it in place.</span>

---

## <span style="font-size: 16px;">Program Decomposition</span>

<span style="font-size: 14px;">The launch grid is one-dimensional, $\lceil N / \texttt{BLOCK\_SIZE} \rceil$ **programs** wide. Each program is identified by $\texttt{tl.program\_id(0)}$ and owns one contiguous tile of $\texttt{BLOCK\_SIZE}$ consecutive lanes. The tile of offsets is $\texttt{offs} = \texttt{pid} \cdot \texttt{BLOCK\_SIZE} + \texttt{tl.arange}(0, \texttt{BLOCK\_SIZE})$, which yields contiguous addresses into both $x$ and $y$.</span>

<span style="font-size: 14px;">No program touches another program's output. There is no reduction, no shared accumulator, no atomic. The parallel pattern is a **map**, identical in structure to vector add. The only new beat at the program level is that the scalar $a$ rides into the kernel through the launch signature and broadcasts across every lane of the tile, costing one register and zero loads.</span>

---

## <span style="font-size: 16px;">Tile Shape and Masking</span>

<span style="font-size: 14px;">$\texttt{BLOCK\_SIZE}$ here is a $\texttt{tl.constexpr}$ fixed at $1024$. The compile-time value lets the compiler size registers, unroll the load-multiply-add-store sequence, and emit a small number of wide PTX memory instructions. Power-of-two block sizes let the compiler pick the widest legal vector load; a non-power-of-two value forces scalar fallback on the trailing lanes of every transaction.</span>

<span style="font-size: 14px;">Whenever $N$ is not a multiple of $1024$, the final program overshoots. The mask $\texttt{mask} = \texttt{offs} < N$ disables out-of-range lanes on every $\texttt{tl.load}$ and $\texttt{tl.store}$. Both inputs share the same mask because the offsets are identical for $x$ and $y$; one mask, two loads, one store, all gated. Mask discipline here is purely about correctness: an unmasked load past the input buffer reads whatever bytes the allocator happened to leave behind, and an unmasked store past the output corrupts memory the kernel does not own.</span>

---

## <span style="font-size: 16px;">Memory Hierarchy and Reuse</span>

<span style="font-size: 14px;">Per output element, the kernel moves $4$ bytes of $x$, $4$ bytes of $y$, and $4$ bytes of $\texttt{out}$ - $12$ bytes total, identical to vector add. Nothing is reused across programs and nothing is reused across tiles, so SRAM never enters the picture. The two input tiles and the output tile live in registers between load and store, and the compiler does not stage anything into shared memory because there is nothing for shared memory to do.</span>

<span style="font-size: 14px;">The single memory-system property the kernel needs is **coalesced** HBM access. The lane offsets inside each tile are contiguous, so the compiler lowers each $\texttt{tl.load}$ to a small number of wide transactions instead of $1024$ separate ones. A $\texttt{BLOCK\_SIZE} = 1024$ tile of fp32 is $4096$ bytes per tensor: roughly $32$ transactions of $128$ B each per input. The author writes contiguous $\texttt{tl.arange}$ offsets; the compiler picks the transaction width. CUDA authors achieve the same effect by lining up thread indices with element indices and reading the GPU vendor's coalescing rules by hand.</span>

<span style="font-size: 14px;">The L2 cache is incidental. With no reuse between tiles, no line is touched twice. The kernel runs at HBM speed, and the L2's only contribution is that the prefetched line still under the read head when the multiply issues does not have to round-trip to HBM again. Any benchmark reporting bandwidth above the device's HBM peak is measuring a warm cache or an input small enough to live entirely in L2.</span>

---

## <span style="font-size: 16px;">Memory-Bound vs Compute-Bound</span>

<span style="font-size: 14px;">Per output element the kernel performs one multiply and one add, fused into one FMA. The **arithmetic intensity** is</span>

$$
\frac{2 \text{ FLOPs}}{12 \text{ bytes}} \approx 0.17 \text{ FLOPs/byte}
$$

<span style="font-size: 14px;">That is twice the intensity of vector add ($\approx 0.083$) and still orders of magnitude below the roofline crossover, which sits around $10$ FLOPs/byte on a modern accelerator. The kernel is **memory-bound** by a wide margin: doubling the arithmetic per element does not move the runtime, because the bottleneck is HBM bandwidth, not throughput on the FMA units. This is the most concrete statement of why fusion matters at the bottom of the roofline: an FMA does in one kernel what a separate scale and add would do in two, and the second kernel costs another round-trip through HBM that the FMA does not pay.</span>

<span style="font-size: 14px;">Sized end-to-end, two separate kernels for $\texttt{tmp} = a \cdot x$ followed by $\texttt{out} = \texttt{tmp} + y$ move $20$ bytes per output element ($x$ read, $\texttt{tmp}$ written, $\texttt{tmp}$ read again, $y$ read, $\texttt{out}$ written). The fused kernel moves $12$. The arithmetic is unchanged; the HBM traffic drops by a factor of $20/12 \approx 1.67$, and on a memory-bound kernel that ratio is roughly the runtime savings. Fusion at the lowest level of the stack pays exactly this much, and the same principle scales up to FlashAttention, which fuses softmax and the score matmul to avoid materializing an $N \times N$ buffer in HBM.</span>

<span style="font-size: 14px;">Anchoring against the rest of the foundations: vector add sits at $0.083$ FLOPs/byte, FMA at $0.17$, ReLU at $\approx 0.08$ (one comparison per element), and the activations that call $\texttt{tl.exp}$ or $\texttt{tl.math.erf}$ at roughly $0.4\text{-}0.8$ FLOPs/byte depending on how the special-function-unit op is counted. Every one of these kernels is memory-bound and the runtime ranking between them is set almost entirely by how many bytes per element move through HBM, not by how much arithmetic happens once the bytes are in registers.</span>

---

## <span style="font-size: 16px;">Compiler-Handled vs Author-Handled</span>

<span style="font-size: 14px;">The author chooses the grid ($\lceil N / \texttt{BLOCK\_SIZE} \rceil$ programs in 1D), the constexpr block size, the offset arithmetic, the mask predicate, and the entry-function signature. Critically the author also chooses to write $a \cdot x + y$ as a single Triton expression rather than two: that decision is what gives the compiler a chance to fuse, and is the only authorial knob in this kernel that affects which hardware instructions land in the PTX.</span>

<span style="font-size: 14px;">The compiler lowers the single expression $a \cdot x + y$ to one hardware FMA instruction per lane on NVIDIA targets (FFMA for fp32, HFMA2 for fp16 pairs). The intermediate product $a \cdot x$ is never written back to a register; it stays inside the FMA pipeline and is consumed directly by the add. This is both faster (one instruction instead of two) and slightly more accurate (one rounding step instead of two, because the unrounded full-precision product feeds directly into the add). The author never names an FMA, never controls register pressure, and never hand-picks vector widths. The constexpr block size is the only authorial input the compiler needs to decide everything below the tile level.</span>

<span style="font-size: 14px;">The numerical point is small but worth pinning. In IEEE-754 fp32 a separate multiply followed by a separate add rounds the product to the nearest representable fp32 value before the add, losing whichever low-order bits the multiplier produced beyond the $24$-bit mantissa. The FMA carries the full-precision product through to the add and rounds exactly once at the end. For a single FMA the difference is at most one ulp; aggregated across long reductions it is the reason matmul accumulators are declared $\texttt{tl.float32}$ even when the inputs are fp16. Here the kernel only does one FMA per lane, so the accuracy gain is one ulp at most, but the same lowering path is what makes high-precision accumulation cheap in every kernel that follows.</span>

<span style="font-size: 14px;">The warp story is the same as vector add. Inside one program, the $1024$-lane tile is internally sharded by the compiler across $\texttt{num\_warps}$ (default $4$) groups of $256$ lanes each, with each warp issuing its own load and FMA in parallel. While one warp is waiting on an HBM response, the others can issue arithmetic, which is how the kernel hides hundreds of cycles of HBM latency without the author writing a single line about warps.</span>

---

## <span style="font-size: 16px;">Naive vs Optimized</span>

<span style="font-size: 14px;">The single-kernel form above is already the optimized form. The naive version is splitting it into two kernels: one for $\texttt{tmp} = a \cdot x$ and one for $\texttt{out} = \texttt{tmp} + y$. The split costs an extra HBM write of $\texttt{tmp}$ and an extra HBM read of $\texttt{tmp}$, raising the per-element traffic from $12$ to $20$ bytes. On a memory-bound kernel that is a $\approx 1.67\times$ slowdown for zero functional benefit.</span>

<span style="font-size: 14px;">The kernel can be tuned further by adjusting $\texttt{BLOCK\_SIZE}$ against the input size. A larger block ($4096$) cuts the program count by $4\times$ and lets the compiler emit wider vector loads; useful when $N$ is in the hundreds of millions and launch overhead is dwarfed by HBM traffic. A smaller block ($256$) raises the program count and helps saturate the device on small $N$, where one $1024$-element program per streaming multiprocessor is not enough to fill the pipeline. Both adjustments live in the single-digit-percent range because the kernel is already running near peak HBM bandwidth.</span>

<span style="font-size: 14px;">The much larger optimization is upstream and downstream fusion: rolling the FMA into the kernel that produced $x$, or into the one that consumes $\texttt{out}$. Each fusion removes one round-trip through HBM, which on a memory-bound chain compounds linearly. This is the lesson that scales from vector add to FlashAttention.</span>

---

## <span style="font-size: 16px;">Worked Example</span>

<span style="font-size: 14px;">Take $N = 6$, $a = 2.5$, $\texttt{BLOCK\_SIZE} = 4$, $x = [1, 2, 3, 4, 5, 6]$, $y = [10, 20, 30, 40, 50, 60]$. The launch grid is $\lceil 6 / 4 \rceil = 2$ programs.</span>

<span style="font-size: 14px;">**Program 0** ($\texttt{pid} = 0$): the offsets are $[0, 1, 2, 3]$, the mask is $[\texttt{T}, \texttt{T}, \texttt{T}, \texttt{T}]$. Loads $x[0..3] = [1, 2, 3, 4]$ and $y[0..3] = [10, 20, 30, 40]$. The expression $a \cdot x + y$ broadcasts $a = 2.5$ across every lane: the tile of products is $[2.5, 5, 7.5, 10]$ and the FMA folds the add in, producing $[12.5, 25, 37.5, 50]$. Stores into $\texttt{out}[0..3]$.</span>

<span style="font-size: 14px;">**Program 1** ($\texttt{pid} = 1$): the offsets are $[4, 5, 6, 7]$, the mask is $[\texttt{T}, \texttt{T}, \texttt{F}, \texttt{F}]$. Loads $x[4..5] = [5, 6]$ and $y[4..5] = [50, 60]$ for the two live lanes; the masked lanes load whatever the $\texttt{other}$ value provides (zero by default) and that value flows harmlessly through the FMA because the matching store mask discards the lane. Writes $\texttt{out}[4] = 62.5$ and $\texttt{out}[5] = 75$. Slots $6$ and $7$ never exist and are never touched.</span>

<span style="font-size: 14px;">Both programs run concurrently; nothing program $1$ does depends on anything program $0$ produces, and the FMA arithmetic on each tile is the same one hardware instruction per lane that a hand-written CUDA kernel would have emitted.</span>

<span style="font-size: 14px;">To count the HBM traffic the kernel paid for this $N = 6$ launch: each program issues two $\texttt{tl.load}$ calls and one $\texttt{tl.store}$, gated by the mask. For program $0$ all four lanes are live, so $4 \cdot 4 = 16$ bytes per tensor flow per call: $32$ bytes of input, $16$ bytes of output. For program $1$ only two lanes are live, so $8$ bytes of input per tensor and $8$ bytes of output. Total HBM traffic across both programs is $48 + 24 + 16 + 8 = 96$ bytes ($x$ + $y$ + $\texttt{out}_0$ + $\texttt{out}_1$). The arithmetic budget is $6$ FMAs, one per output lane, exactly. Run the two-kernel split instead and the arithmetic is the same $6$ FMAs but the traffic is $96 + 24 + 24 = 144$ bytes - the extra $48$ bytes are the round-trip through $\texttt{tmp}$.</span>

---

## <span style="font-size: 16px;">Pitfalls</span>

* <span style="font-size: 14px;">**Splitting the expression in two.** Writing $\texttt{tmp} = a \cdot x$ on one line and $\texttt{out} = \texttt{tmp} + y$ on the next gives the compiler more freedom but no obligation to fuse the two into a single FMA. Triton's lowering for $a \cdot x + y$ as a single expression is the reliable path to one FFMA per lane. The numerical and performance difference is small per element but real, and it propagates as kernels get larger.</span>
* <span style="font-size: 14px;">**Treating the scalar as a tensor.** $a$ is a Python float, passed by value into the kernel signature. Calling $\texttt{tl.load}$ on it, or trying to read it through a pointer, is a type error the harness surfaces at launch. The scalar broadcasts across every lane of the tile without any extra plumbing.</span>
* <span style="font-size: 14px;">**Forgetting the tail mask.** $\texttt{BLOCK\_SIZE} = 1024$ almost always overshoots $N$. Without $\texttt{mask} = \texttt{offs} < N$ on every $\texttt{tl.load}$ and $\texttt{tl.store}$, the kernel reads garbage past the input and writes past the output. Hidden test sizes at $N = 257$ and $N = 1025$ catch this immediately.</span>
* <span style="font-size: 14px;">**Block size not constexpr.** A runtime $\texttt{BLOCK\_SIZE}$ prevents the compiler from unrolling the load-FMA-store sequence and picking the widest legal vector load. The kernel compiles but loses most of its speed and may fall back to a scalar inner loop.</span>

---