# <span style="font-size: 20px;">Max Reduction</span>

<span style="font-size: 14px;">Computing the maximum of an array is a reduction, but its lower bound is different from a sum: the max has a natural identity element ($-\infty$) that lets masked lanes be discarded by the comparison rather than added to the result. This kernel uses the simplest possible Triton reduction shape: a single program that loads the entire array into a register tile, calls $\texttt{tl.max}$ once, and stores one scalar. It is also the building block for fused softmax and logsumexp, where the same row-wide max is the first step of a numerical-stability pipeline.</span>

---

## <span style="font-size: 16px;">The Operation</span>

<span style="font-size: 14px;">Given a contiguous float tensor $x \in \mathbb{R}^{N}$, the kernel writes the scalar maximum into a length-1 output:</span>

$$
\texttt{out}[0] = \max_{0 \le i < N} x[i]
$$

<span style="font-size: 14px;">The output buffer lives in HBM. Unlike the sum kernel, no pre-zeroing is required: the kernel writes the result with a plain $\texttt{tl.store}$, not an atomic.</span>

---

## <span style="font-size: 16px;">Program Decomposition</span>

<span style="font-size: 14px;">The launch grid is exactly one program. That single **program** holds the entire array as a $\texttt{BLOCK\_SIZE}$-lane register tile, where $\texttt{BLOCK\_SIZE} = \texttt{triton.next\_power\_of\_2}(N)$. The reduction collapses the tile to one scalar with $\texttt{tl.max(x, axis=0)}$, and the result is stored to $\texttt{out}[0]$.</span>

<span style="font-size: 14px;">This is the **whole-array reduction in one tile** pattern. It is the cleanest possible decomposition when the input fits in a single Triton block, because it eliminates the cross-program combine entirely. No atomics, no second-stage kernel, no scratch buffers. The price is that the pattern caps at the largest $\texttt{BLOCK\_SIZE}$ the hardware can hold in registers, typically $65536$ lanes. For arrays larger than that, the kernel would have to switch to a two-stage form: each program reduces its tile and writes the partial to a scratch buffer, and a second kernel reduces the scratch buffer to a final scalar.</span>

<span style="font-size: 14px;">Triton also exposes $\texttt{tl.atomic\_max}$ as an alternative cross-program combine, but it operates on integer types and requires a float-to-int reinterpret for fp32 values that preserves ordering only for non-negative inputs. The atomic path is therefore awkward for general floats; the two-stage scratch form is the cleaner generalization, and for the current problem size the single-program form is simpler still.</span>

---

## <span style="font-size: 16px;">Tile Shape and Masking</span>

<span style="font-size: 14px;">$\texttt{BLOCK\_SIZE}$ is computed at launch time by the host as $\texttt{triton.next\_power\_of\_2}(N)$ and passed as a $\texttt{tl.constexpr}$ argument. Power-of-two block extents are a requirement of $\texttt{tl.max}$ (and every other Triton tile reduction), not a stylistic preference: the compiler needs that property to lower the reduction to a balanced tree of register-shuffle instructions. A runtime-shaped block would force a scalar fallback for the last partial group and disable most of the optimization.</span>

<span style="font-size: 14px;">The tile of offsets is $\texttt{offs} = \texttt{tl.arange}(0, \texttt{BLOCK\_SIZE})$, paired with $\texttt{mask} = \texttt{offs} < N$. The mask is critical for correctness because $\texttt{BLOCK\_SIZE}$ overshoots $N$ whenever $N$ is not already a power of two. The sentinel value passed to $\texttt{tl.load}$ is $\texttt{other} = -\texttt{float('inf')}$, the additive identity for max under the IEEE-754 ordering rules. Masked lanes load $-\infty$, lose every subsequent $\texttt{tl.max}$ comparison, and cannot corrupt the result.</span>

<span style="font-size: 14px;">The sentinel choice is operation-specific and is one of the few places where the author has to encode the semantics of the reduction by hand. A sum uses $\texttt{other=0.0}$; a max uses $-\infty$; a product would use $1.0$; a logical AND would use $\texttt{True}$. Picking the wrong sentinel is a silent correctness bug. For max in particular, the common failure mode is using $\texttt{other=0.0}$ on an all-negative input: every masked lane suddenly becomes a candidate maximum, and the kernel returns $0$ instead of the true negative maximum.</span>

---

## <span style="font-size: 16px;">Memory Hierarchy and Reuse</span>

<span style="font-size: 14px;">The single program loads the whole input from HBM into a register tile in one logical $\texttt{tl.load}$, runs $\texttt{tl.max}$ in registers, and emits one $\texttt{tl.store}$ of a single fp32 back to HBM. Reuse is zero, by the same argument that applied to vector addition: each input byte is loaded once and discarded after it contributes to the reduction. Nothing is staged into shared memory by the author, although the compiler does stage the cross-warp partials into shared memory when reducing the four warps' contributions into the final scalar.</span>

<span style="font-size: 14px;">The compiler emits the $\texttt{tl.load}$ as a small number of wide coalesced HBM transactions, because the offsets inside the tile are contiguous. For a $4096$-lane fp32 tile, that is $16$ KB of input, or roughly $128$ transactions of $128$ B each at the HBM granularity. The kernel completes in the time it takes to stream that much data through HBM, which on a modern accelerator is well under a microsecond, plus the kernel launch overhead which dominates for arrays this small.</span>

<span style="font-size: 14px;">An important consequence of the single-program design is that the kernel does not saturate the device for typical $N$. One Triton program lands on one streaming multiprocessor, leaves all the others idle, and pulls just enough data through HBM to fill its tile. For $N$ small enough to fit in one block, this is fine: the launch overhead is the limiting factor, not the per-SM bandwidth. For larger $N$, where the input no longer fits in a single block, the kernel must be redecomposed into many programs, which is the regime where the two-stage scratch form starts to matter.</span>

<span style="font-size: 14px;">A useful number for context: kernel launch overhead on a modern GPU is on the order of $5$ to $10$ microseconds, while pulling $16$ KB through HBM at $1$ TB/s takes about $16$ nanoseconds. The arithmetic, even with the $\log_2(\texttt{BLOCK\_SIZE})$-step reduction tree, is dwarfed by both. For small $N$, max reduction is effectively a launch-overhead test, and shaving the kernel cost matters less than batching multiple reductions into one launch (for instance, when computing the max of many rows independently, which is exactly what fused softmax does).</span>

---

## <span style="font-size: 16px;">Memory-Bound vs Compute-Bound</span>

<span style="font-size: 14px;">Per element, the kernel reads $4$ bytes and performs one comparison. **Arithmetic intensity** is on the order of</span>

$$
\frac{1 \text{ compare}}{4 \text{ bytes}} = 0.25 \text{ ops/byte}
$$

<span style="font-size: 14px;">which places the kernel firmly on the memory-bound side of the roofline. The crossover for fp32 compute on modern hardware is around $10$ FLOPs per byte, two orders of magnitude above what max reduction does. Like vector addition and like sum reduction, the runtime is governed by HBM bandwidth and launch overhead, not by arithmetic.</span>

<span style="font-size: 14px;">Unlike sum, max is bit-exact under reordering. The comparison is associative in the exact, infinite-precision sense, so any order of pairwise reductions produces the same result. There is no analogue of the floating-point-associativity problem that complicates parallel sum: the compiler can reduce in any tree shape, the hardware can commit partials in any order, and the answer is identical to the serial $\max$ down to the last bit. This is one of the few cases where the parallel and serial implementations are guaranteed to match exactly, which is why the test harness uses a tight $\texttt{atol=1e-3}$ rather than the widened tolerance used for sum.</span>

---

## <span style="font-size: 16px;">Compiler-Handled vs Author-Handled</span>

<span style="font-size: 14px;">**Author chooses:** the grid of one program, the constexpr $\texttt{BLOCK\_SIZE} = \texttt{next\_power\_of\_2}(N)$, the mask predicate, the $-\infty$ sentinel for masked lanes, the use of a plain $\texttt{tl.store}$ instead of an atomic, and the implicit choice not to bother with a two-stage form. The first four are correctness-critical; the last two are design choices that hold only as long as the input fits in one block.</span>

<span style="font-size: 14px;">**Compiler handles:** lowering $\texttt{tl.load}$ to coalesced HBM transactions of the appropriate vector width, lowering $\texttt{tl.max}$ to a balanced tree of warp-shuffle compare-and-select instructions (five steps within each warp), staging the per-warp partials into shared memory and emitting the necessary barrier before the cross-warp combine, and lowering the final $\texttt{tl.store}$ to a single fp32 write. The author never picks a vector width, never names a warp, never declares shared memory, and never inserts a barrier. In CUDA, writing the same kernel by hand requires explicit warp-shuffle code, a small shared-memory staging buffer, and a $\texttt{\_\_syncthreads}$ before the final reduction step; in Triton, the author writes $\texttt{tl.max(x, axis=0)}$ and the compiler emits all of it.</span>

<span style="font-size: 14px;">The default $\texttt{num\_warps}$ for a tile this size is $4$. The compiler shards the $\texttt{BLOCK\_SIZE}$-lane tile evenly across those four warps, runs the in-warp shuffle reduction in parallel, then combines four scalar partials through shared memory. The author can override $\texttt{num\_warps}$ via $\texttt{@triton.autotune}$ if the default loses against a wider sharding, but for a single-program reduction at moderate $\texttt{BLOCK\_SIZE}$ the default is essentially always correct.</span>

---

## <span style="font-size: 16px;">Naive vs Optimized</span>

<span style="font-size: 14px;">For an array that fits in one tile, the kernel above is already the optimized form. The only meaningful axis of variation is the response to growing $N$. When $N$ exceeds the maximum $\texttt{BLOCK\_SIZE}$ that the hardware can sustain (typically $65536$ lanes, around $256$ KB of fp32 data), the single-program form becomes infeasible and the kernel must redecompose into many programs. The clean redecomposition is a two-stage scratch form: stage one launches $\lceil N / \texttt{BLOCK\_SIZE} \rceil$ programs, each of which reduces its tile with $\texttt{tl.max}$ and writes its partial to scratch slot $\texttt{pid}$ with a plain $\texttt{tl.store}$; stage two launches a single program that loads the scratch buffer and reduces it the same way. There are no atomics in either stage.</span>

<span style="font-size: 14px;">A second axis is using $\texttt{tl.atomic\_max}$ for cross-program combine, which avoids the second-stage kernel. The catch is that $\texttt{tl.atomic\_max}$ on the GPU is defined for integer types, and applying it to fp32 requires a sign-preserving int reinterpret that flips bits to maintain monotonicity (positive floats stay monotone under reinterpret-as-int, but negative floats invert; flipping the sign bit and the body conditionally restores ordering). The reinterpret is correct but adds complexity, and the atomic still serializes on a single output address across $G$ programs. On contemporary GPUs the two-stage scratch form is simpler to read and equally fast in practice. The current problem stays in the single-program regime so neither extension is needed.</span>

<span style="font-size: 14px;">A useful sanity check on this scale: at $\texttt{BLOCK\_SIZE} = 65536$ the in-tile reduction depth is $\log_2(65536) = 16$ steps, all of them warp shuffles and one short shared-memory hop. The total in-tile work is well under a microsecond on a modern accelerator. Everything beyond that point is HBM bandwidth.</span>

---

## <span style="font-size: 16px;">Worked Example</span>

<span style="font-size: 14px;">Take $N = 5$ and $x = [-3, 1, -7, 2, 0]$. The host computes $\texttt{BLOCK\_SIZE} = \texttt{next\_power\_of\_2}(5) = 8$ and launches a grid of one program.</span>

<span style="font-size: 14px;">The single program builds $\texttt{offs} = [0, 1, 2, 3, 4, 5, 6, 7]$ and $\texttt{mask} = [\texttt{T}, \texttt{T}, \texttt{T}, \texttt{T}, \texttt{T}, \texttt{F}, \texttt{F}, \texttt{F}]$. The $\texttt{tl.load}$ with $\texttt{other} = -\infty$ produces the lane vector $[-3, 1, -7, 2, 0, -\infty, -\infty, -\infty]$. The reduction then unfolds in $\log_2(8) = 3$ tree steps inside registers: pairs reduce to $[1, 2, 0, -\infty]$, then to $[2, 0]$, then to $[2]$. The masked $-\infty$ lanes lose every comparison and never contribute. The final $\texttt{tl.store}$ writes $2$ into $\texttt{out}[0]$.</span>

<span style="font-size: 14px;">Now contrast with the broken sentinel: if the load had used $\texttt{other=0.0}$, the lane vector would have been $[-3, 1, -7, 2, 0, 0, 0, 0]$ and the reduction would still produce $2$ in this case, because $2 > 0$. The bug stays hidden until the input is all-negative. With $x = [-3, -1, -7, -2, -5]$, the correct max is $-1$, but the broken kernel would see $[-3, -1, -7, -2, -5, 0, 0, 0]$ and return $0$. This is exactly why the test harness includes an $\texttt{all\_negative}$ pattern: it surfaces the sentinel bug that would otherwise pass most natural inputs.</span>

---

## <span style="font-size: 16px;">Pitfalls</span>

* <span style="font-size: 14px;">**Using $\texttt{other=0.0}$ instead of $-\infty$ on the masked load.** Silently breaks for any input whose true maximum is negative. The kernel returns $0$ from the masked lanes instead of the real maximum. The fix is to match the sentinel to the reduction: $-\infty$ for max, $+\infty$ for min, $0$ for sum, $1$ for product.</span>
* <span style="font-size: 14px;">**Non-power-of-two $\texttt{BLOCK\_SIZE}$.** $\texttt{tl.max}$ requires power-of-two tile extents to lower its balanced reduction tree. Pass $N$ through $\texttt{triton.next\_power\_of\_2}$ at the host before declaring it as $\texttt{tl.constexpr}$. A non-power-of-two block will compile but fall back to scalar code and lose most of its speed.</span>
* <span style="font-size: 14px;">**Reaching for $\texttt{tl.atomic\_max}$ on fp32.** The atomic-max instruction is integer-typed; using it on floats requires a sign-aware reinterpret. For floats, the cleaner cross-program combine is a two-stage scratch kernel, not the atomic.</span>
* <span style="font-size: 14px;">**Assuming the single-program form scales arbitrarily.** A tile of $\texttt{BLOCK\_SIZE} = 2^{20}$ does not fit in registers on any current GPU. When $N$ outgrows the maximum sustainable block size, switch to the two-stage form. The current problem stays inside the single-block regime; future ones will not.</span>

---