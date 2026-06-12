# <span style="font-size: 20px;">Fused Mean and Variance</span>

<span style="font-size: 14px;">Computing the population mean and variance of a tensor in two separate passes is the obvious algorithm; computing both in a single pass is the right one on any bandwidth-bound device. The fusion turns on a small algebraic identity, $\sigma^2 = E[x^2] - E[x]^2$, and the systems lesson is that an identity worth one line of math saves an entire round trip through HBM. This is the canonical Triton example of a kernel that emits more than one statistic from the same loaded tile, and the cleanest demonstration of why a per-array reduction does not have to mean a per-statistic reduction.</span>

---

## <span style="font-size: 16px;">The Operation</span>

<span style="font-size: 14px;">Given a contiguous float tensor $x \in \mathbb{R}^{N}$, the kernel writes the scalar mean and population variance into two length-1 outputs:</span>

$$
\mu = \frac{1}{N} \sum_{i=0}^{N-1} x[i], \qquad \sigma^2 = \frac{1}{N} \sum_{i=0}^{N-1} x[i]^2 - \mu^2
$$

<span style="font-size: 14px;">The identity rewrites the variance as $E[x^2] - E[x]^2$, exposing two reductions that share the same input: $\sum x$ and $\sum x^2$. Both are accumulated into scratch scalars by the kernel, and the host divides by $N$ and applies the identity once after launch.</span>

---

## <span style="font-size: 16px;">Program Decomposition</span>

<span style="font-size: 14px;">The launch grid is one-dimensional with $\lceil N / \texttt{BLOCK\_SIZE} \rceil$ **programs**, the same shape as a plain sum reduction. Each program owns the contiguous tile $\texttt{offs} = p \cdot \texttt{BLOCK\_SIZE} + \texttt{tl.arange}(0, \texttt{BLOCK\_SIZE})$ and is responsible for two in-tile reductions: $\texttt{block\_sum} = \texttt{tl.sum(x, axis=0)}$ and $\texttt{block\_sumsq} = \texttt{tl.sum(x * x, axis=0)}$. It then emits two atomic adds, one into each scratch buffer.</span>

<span style="font-size: 14px;">The decomposition is **two-statistic per-array reduction**. The grid is identical to a single-statistic reduction, but each program now contributes two atomics instead of one, into two distinct addresses. Critically, the two reductions share the same loaded tile: the input is read from HBM exactly once across the whole kernel, even though two reductions are computed. This is the property that makes the fusion worth the few extra lines of code.</span>

<span style="font-size: 14px;">The host contract has two parts. First, both scratch buffers must be pre-zeroed because the kernel reaches them through atomics. Second, the host finalizes the result after the kernel returns by dividing each scratch scalar by $N$ and applying $\sigma^2 = E[x^2] - \mu^2$. Doing the finalize on the host costs essentially nothing (two divisions and one subtraction) and keeps the kernel small.</span>

---

## <span style="font-size: 16px;">Tile Shape and Masking</span>

<span style="font-size: 14px;">$\texttt{BLOCK\_SIZE} = 1024$, declared $\texttt{tl.constexpr}$, the same standard choice as the sum kernel. The compiler uses the constexpr value to size the register tile, pick a vector load width, and unroll both reduction trees at compile time. The two reductions inside one program share the same tile in registers, so the additional reduction tree for $\sum x^2$ does not cost an additional HBM load; it costs only the in-register multiply and the second tree.</span>

<span style="font-size: 14px;">The mask is $\texttt{mask} = \texttt{offs} < N$ on the $\texttt{tl.load}$, paired with $\texttt{other=0.0}$. The same sentinel works for both reductions: masked lanes load $0$, contribute $0$ to $\sum x$ as the additive identity, and contribute $0^2 = 0$ to $\sum x^2$ as well. This is one of the lucky cases where the same sentinel is the identity for both statistics; it would not be true if one of the reductions were a max, in which case the kernel would have to load the tile twice with different sentinels or use a more careful conditional select.</span>

---

## <span style="font-size: 16px;">Memory Hierarchy and Reuse</span>

<span style="font-size: 14px;">Each program loads its $\texttt{BLOCK\_SIZE}$-lane tile from HBM into registers exactly once, computes both reductions in registers, and emits two atomic adds to two distinct HBM addresses. The reuse story is the whole point of the fusion: a single load feeds both statistics. A naive two-kernel implementation would launch two reduction kernels back to back, each loading $x$ from HBM, for a total of $2 \cdot 4 N = 8 N$ bytes of input traffic. The fused kernel does $4 N$ bytes. The HBM-traffic ratio is exactly $2 \times$, which on a memory-bound reduction translates almost linearly into a $2 \times$ speedup.</span>

<span style="font-size: 14px;">A useful concrete number: at $N = 10^6$, the fused kernel reads $4$ MB and writes $8$ bytes (two fp32 scratch scalars, eventually). The naive two-kernel version reads $8$ MB. At $1$ TB/s of HBM bandwidth, the fused version finishes its data pass in about $4$ microseconds and the naive version in about $8$ microseconds, before either pays its launch overhead. The launch overhead matters for tiny $N$ because the naive version pays it twice, which is another reason fusion wins even before the bandwidth math. For even larger $N$ (say $10^8$), the ratio holds, and the absolute savings become hundreds of microseconds, which adds up across an inference or training step that calls reductions thousands of times.</span>

<span style="font-size: 14px;">The atomic traffic doubles relative to a single-statistic reduction, because each program now emits two atomics instead of one. Both atomics still go through L2, both target addresses stay hot in L2 throughout the launch, and they are independent (different cache lines), so they do not interfere with each other. For a grid of $G$ programs, the kernel emits $2G$ atomics, which is well within what L2 absorbs at the relevant $G$. The atomic stream is still well below the data stream in absolute cost.</span>

<span style="font-size: 14px;">No shared memory is staged by the author. The compiler does stage cross-warp partials into shared memory for each $\texttt{tl.sum}$, exactly as it does for a single-statistic reduction, but the author writes nothing about that.</span>

---

## <span style="font-size: 16px;">Memory-Bound vs Compute-Bound</span>

<span style="font-size: 14px;">Per element, the kernel reads $4$ bytes and performs three FLOPs: one multiply for $x \cdot x$, one add into the $\sum x$ tree, and one add into the $\sum x^2$ tree. **Arithmetic intensity** is</span>

$$
\frac{3 \text{ FLOPs}}{4 \text{ bytes}} = 0.75 \text{ FLOPs/byte}
$$

<span style="font-size: 14px;">That is three times the intensity of a plain sum, but still well under the $\approx 10$ FLOPs/byte roofline crossover for fp32 on modern hardware. The kernel is firmly **memory-bound**, and the fusion does not change that. What it does change is which bandwidth ceiling the kernel approaches: the fused kernel sits at roughly the same throughput as a plain sum (one pass over $x$), while the naive two-kernel form sits at half that.</span>

<span style="font-size: 14px;">The variance identity has a known **numerical caveat** that is worth stating clearly: when $\mu^2$ is close in magnitude to $E[x^2]$, the subtraction $E[x^2] - \mu^2$ loses precision through catastrophic cancellation. This happens when $\sigma^2 \ll \mu^2$, that is, on data whose values are tightly clustered around a large mean. The fp32 representation of $E[x^2]$ and $\mu^2$ each carry seven significant digits; if they agree to four digits, the variance is computed with only three significant digits of precision. The test harness avoids the worst of this by using centered or modestly ranged patterns, and widens the tolerance to combined $\texttt{atol=1e-2, rtol=1e-2}$ to absorb the residual error from sum-of-squares accumulation at $N = 10^6$.</span>

<span style="font-size: 14px;">The production-grade alternative is **Welford's algorithm**, which maintains a running mean and a running sum of squared deviations from that mean, updating both with a numerically stable recurrence. Welford is one pass like the identity form, but the recurrence does not parallelize cleanly across tiles (combining two Welford partials requires a careful blend that uses the count from each tile). For a coding problem at this level, the identity form is the right tradeoff: it is simple, it is fast, and the inputs are well-conditioned enough for fp32 to hold the answer to the required tolerance.</span>

---

## <span style="font-size: 16px;">Compiler-Handled vs Author-Handled</span>

<span style="font-size: 14px;">**Author chooses:** the grid shape, the constexpr block size, the mask predicate, the $\texttt{other=0.0}$ sentinel (which happens to work for both statistics), the two in-tile reductions sharing a single load, the two atomic adds into separate scratch buffers, the host-side pre-zeroing of both scratches, and the host-side finalize ($\mu = \sum x / N$, $\sigma^2 = \sum x^2 / N - \mu^2$). The decision to use the identity instead of a two-pass formulation is the most important author choice in the entire kernel; it is the difference between a memory-bound kernel that reads $x$ once and one that reads it twice.</span>

<span style="font-size: 14px;">**Compiler handles:** lowering the single $\texttt{tl.load}$ to coalesced wide HBM transactions, lowering each $\texttt{tl.sum}$ to a balanced warp-shuffle reduction tree with the necessary shared-memory staging for cross-warp combines, scheduling the two reduction trees so that they can share the load and overlap with each other, and emitting both atomic adds as single PTX instructions. The author never names a warp, never declares shared memory, never inserts a barrier between the two reductions. The compiler also handles dead-code elimination across the two reductions if one of them happens to be unused.</span>

<span style="font-size: 14px;">A subtle compiler benefit is that the two $\texttt{tl.sum}$ calls share register pressure intelligently. The intermediate values for $\sum x$ and $\sum x^2$ are short-lived, so the compiler can reuse the same registers for the reduction tree of each. The author writes two separate $\texttt{tl.sum}$ calls; the compiler folds them into a tile-fused pass that does not double the register footprint.</span>

---

## <span style="font-size: 16px;">Naive vs Optimized</span>

<span style="font-size: 14px;">The naive implementation is two kernels: one $\texttt{sum\_kernel}$ to produce $\sum x$ and a second $\texttt{sumsq\_kernel}$ to produce $\sum x^2$. Each reads $x$ from HBM, each emits its own grid of atomics, each pays its own launch overhead. The combined kernel reads $x$ once, emits two atomics per program, and pays one launch overhead. The HBM-traffic ratio is $2 \times$, the launch-overhead ratio is also $2 \times$, and on a bandwidth-bound kernel both compound into a roughly $2 \times$ speedup at any non-trivial $N$.</span>

<span style="font-size: 14px;">A second axis of variation is the choice of identity. The Welford form mentioned above is the higher-precision alternative, at the cost of a tile-combine that depends on the per-tile count. A third axis is keeping the finalize on the device instead of the host: the host divisions and the final subtract could be done in a tiny one-program finalize kernel, but at $N \gg 1$ the cost of those three host-side operations is negligible compared with the main kernel, so the host form is strictly simpler with no measurable downside.</span>

<span style="font-size: 14px;">For inputs with extreme magnitudes or near-singular variance, the right next step is not to optimize further but to switch algorithms: Welford for tight precision, or a two-pass formulation that computes $\mu$ exactly and then $\sum (x - \mu)^2$ in a second pass. The two-pass version doubles HBM traffic in exchange for catastrophic-cancellation-free variance, and the choice between fused-identity and two-pass is a precision-versus-bandwidth tradeoff that the author must make based on the data distribution.</span>

---

## <span style="font-size: 16px;">Worked Example</span>

<span style="font-size: 14px;">Take $N = 5$, $x = [1, 2, 3, 4, 5]$, and $\texttt{BLOCK\_SIZE} = 4$. The host zeros both scratch buffers and launches a grid of $\lceil 5 / 4 \rceil = 2$ programs.</span>

<span style="font-size: 14px;">**Program 0** ($\texttt{pid} = 0$): offsets $[0, 1, 2, 3]$, mask all true, loads $[1, 2, 3, 4]$. The first reduction gives $\texttt{block\_sum} = 1 + 2 + 3 + 4 = 10$. The second gives $\texttt{block\_sumsq} = 1 + 4 + 9 + 16 = 30$. Two atomic adds: $\texttt{sum\_buf} \mathrel{+}= 10$, $\texttt{sumsq\_buf} \mathrel{+}= 30$.</span>

<span style="font-size: 14px;">**Program 1** ($\texttt{pid} = 1$): offsets $[4, 5, 6, 7]$, mask $[\texttt{T}, \texttt{F}, \texttt{F}, \texttt{F}]$, $\texttt{other=0.0}$, loads $[5, 0, 0, 0]$. $\texttt{block\_sum} = 5$, $\texttt{block\_sumsq} = 25$. Two atomic adds: $\texttt{sum\_buf} \mathrel{+}= 5$, $\texttt{sumsq\_buf} \mathrel{+}= 25$.</span>

<span style="font-size: 14px;">After the kernel returns, $\texttt{sum\_buf} = 15$ and $\texttt{sumsq\_buf} = 55$. The host divides: $\mu = 15 / 5 = 3$, $E[x^2] = 55 / 5 = 11$, and $\sigma^2 = 11 - 9 = 2$. The reference values are $\mu = 3$ and $\sigma^2 = 2$, matching exactly because the integers fit cleanly in fp32 mantissas. Notice the load happens once per program, the kernel reads $x$ a total of one time across the launch, and both statistics fall out of the same loaded tile. This is the entire systems argument for the fusion.</span>

---

## <span style="font-size: 16px;">Pitfalls</span>

* <span style="font-size: 14px;">**Two separate kernels for sum and sumsq.** The most common mistake is writing a clean reduction kernel and calling it twice, once for $\sum x$ and once for $\sum x^2$. This doubles HBM traffic and pays two launch overheads. The whole point of the kernel is to fuse, not to be tidy.</span>
* <span style="font-size: 14px;">**Forgetting to zero either scratch buffer.** Both $\texttt{sum\_buf}$ and $\texttt{sumsq\_buf}$ are reached through atomics. Skipping the zero on either one produces drift on second and later invocations, which is the worst failure mode because the first call passes.</span>
* <span style="font-size: 14px;">**Computing $\sigma^2$ in the kernel and atomic-adding it.** Variance is not linear in the input: $\text{Var}(A \cup B) \ne \text{Var}(A) + \text{Var}(B)$. Per-program variances cannot be summed across programs. Only $\sum x$ and $\sum x^2$ are linear in the input; the identity must be applied on the host after the cross-program combine is complete.</span>
* <span style="font-size: 14px;">**Tight tolerance for variance at large $N$.** The sum-of-squares accumulator drifts with $N$, and the $E[x^2] - \mu^2$ subtraction can lose precision through cancellation. The test harness uses combined $\texttt{atol=1e-2, rtol=1e-2}$ for the variance check at large $N$ for both reasons. Production code on near-constant data should switch to Welford.</span>

---