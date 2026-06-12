# <span style="font-size: 20px;">L2 Norm</span>

<span style="font-size: 14px;">The L2 norm of a vector is a sum-of-squares followed by a single square root, and the systems angle is that fusion turns the square into a free in-register operation while the reduction shape is identical to a plain sum. The kernel reads $x$ once, squares lanes in registers before they enter the reduction tree, atomic-adds the per-program partials into a scratch scalar, and lets the host take the square root on the resulting single fp32 value. The square root never appears inside the kernel because $\sqrt{\sum a_i}$ is not the sum of $\sqrt{a_i}$, and the only correct place to apply a non-linear final step is after the cross-program combine has completed.</span>

---

## <span style="font-size: 16px;">The Operation</span>

<span style="font-size: 14px;">Given a contiguous float tensor $x \in \mathbb{R}^{N}$, the kernel writes the Euclidean norm into a length-1 output:</span>

$$
\texttt{out}[0] = \sqrt{\sum_{i=0}^{N-1} x[i]^2}
$$

<span style="font-size: 14px;">The interior of the kernel is a sum-of-squares reduction. The square root is the only step that does not happen inside a Triton program; the host applies it once on the cross-program-combined scalar after the kernel returns.</span>

---

## <span style="font-size: 16px;">Program Decomposition</span>

<span style="font-size: 14px;">The launch grid is one-dimensional with $\lceil N / \texttt{BLOCK\_SIZE} \rceil$ **programs**, the same shape as a plain sum. Each program owns the contiguous tile $\texttt{offs} = p \cdot \texttt{BLOCK\_SIZE} + \texttt{tl.arange}(0, \texttt{BLOCK\_SIZE})$ and reduces it with $\texttt{tl.sum(x * x, axis=0)}$, where the multiply is fused into the reduction call rather than being a separate pass. One $\texttt{tl.atomic\_add}$ per program combines the per-program partials into a scalar scratch buffer.</span>

<span style="font-size: 14px;">The pattern is **per-array reduction with an in-tile fused transform**. The transform here is the square; in other kernels it could be an exponential (logsumexp), an absolute value (L1 norm), or a logical predicate (count of non-zeros). The grid shape and the atomic combine are identical across all of them; what changes is the operator applied to each lane before the reduction tree consumes it. Triton expresses this fusion through ordinary Python-arithmetic on a tile: $\texttt{x * x}$ is a lane-wise multiply that produces a new tile, which $\texttt{tl.sum}$ then collapses.</span>

<span style="font-size: 14px;">The launch contract has two parts. First, the scratch buffer must be allocated by the host and zeroed (the kernel does this implicitly by allocating with $\texttt{torch.zeros}$ inside $\texttt{solve}$). Second, after the kernel returns, the host calls $\texttt{torch.sqrt}$ on the scratch scalar and writes the result into the user-visible $\texttt{out}$ buffer. The square root is one fp32 op on the host side, immeasurably cheap.</span>

---

## <span style="font-size: 16px;">Tile Shape and Masking</span>

<span style="font-size: 14px;">$\texttt{BLOCK\_SIZE} = 1024$, declared $\texttt{tl.constexpr}$. The choice is identical to the plain sum kernel: a power of two for vectorized loads and a $\log_2(\texttt{BLOCK\_SIZE}) = 10$-step balanced reduction tree, large enough to amortize the per-program atomic, small enough to fit comfortably in registers.</span>

<span style="font-size: 14px;">The mask is $\texttt{mask} = \texttt{offs} < N$ on the $\texttt{tl.load}$, paired with $\texttt{other=0.0}$. The sentinel works correctly for the squared sum because $0^2 = 0$: masked lanes load $0$, are squared to $0$ before entering the reduction tree, and contribute the additive identity. A naive concern is that some other sentinel might be needed; the math says no, and the simple choice is both correct and the most economical.</span>

<span style="font-size: 14px;">A subtler concern is that an extreme sentinel could itself overflow once squared. If the author chose $\texttt{other} = 10^{20}$ (any large finite float), squaring it would produce $10^{40}$, well past the fp32 maximum of about $3.4 \times 10^{38}$, and the reduction would saturate to $+\infty$. The clean fix is the obvious one already in use: $\texttt{other=0.0}$ never overflows, and the masked lanes contribute nothing. This is a small example of the kind of attention that fused in-tile transforms demand: the load sentinel must be consistent not just with the additive identity of the reduction but also with the dynamic range of the operator applied before the reduction. The same caution would matter in a hypothetical $\sum \exp(x)$ kernel, where $\texttt{other=0.0}$ would contribute $\exp(0) = 1$ per masked lane and inflate the sum; that kernel needs $\texttt{other} = -\infty$ so that $\exp$ of the sentinel is $0$. The systems rule is to push the sentinel through the in-tile transform mentally before committing to it.</span>

---

## <span style="font-size: 16px;">Memory Hierarchy and Reuse</span>

<span style="font-size: 14px;">Each program loads its $\texttt{BLOCK\_SIZE}$-lane tile from HBM into registers exactly once, squares it in registers, reduces it in registers, and emits one atomic add to a single HBM address. The square never materializes in HBM; it lives only as a transient tile inside the reduction expression. The kernel reads $4 N$ bytes from HBM and writes essentially nothing back, which is the bandwidth lower bound for any L2 norm.</span>

<span style="font-size: 14px;">A worse implementation would be to launch one kernel that writes a squared copy of $x$ back to HBM, and a second kernel that sums it. That naive version moves $8 N + 4 N = 12 N$ bytes (read $x$, write $x^2$, then read $x^2$), three times the bandwidth of the fused kernel. The fusion is a $3 \times$ HBM-traffic win on a memory-bound kernel, which translates almost directly into a $3 \times$ speedup. The lesson generalizes: any time an elementwise transform precedes a reduction, fold the transform into the same pass.</span>

<span style="font-size: 14px;">The atomic story is the same as in plain sum: $G$ atomics into one cache line, all serialized through L2, with the cache line staying hot for the duration of the launch. The output tile is a single fp32 scratch scalar that the host eventually reads and square-roots. No shared memory is used by the author; the compiler stages the cross-warp partials of $\texttt{tl.sum}$ in shared memory exactly as it does for a plain sum.</span>

<span style="font-size: 14px;">A useful number: at $N = 10^6$ and $\texttt{BLOCK\_SIZE} = 1024$, the kernel reads $4$ MB ($\approx 4$ microseconds at $1$ TB/s) and issues $\approx 1000$ atomics into L2. The atomic stream is small enough that its serialization cost is dwarfed by the data stream, even though the atomics are sequentially committed.</span>

---

## <span style="font-size: 16px;">Memory-Bound vs Compute-Bound</span>

<span style="font-size: 14px;">Per element, the kernel reads $4$ bytes and performs two FLOPs: one multiply ($x \cdot x$) and one add into the reduction tree. **Arithmetic intensity** is</span>

$$
\frac{2 \text{ FLOPs}}{4 \text{ bytes}} = 0.5 \text{ FLOPs/byte}
$$

<span style="font-size: 14px;">This is double the intensity of a plain sum but still far under the roofline crossover of $\approx 10$ FLOPs/byte for fp32. The kernel is **memory-bound**; the multiply is free in the same way that any in-register operation is free on a bandwidth-limited kernel. The runtime is set by how fast the input streams through HBM, plus the kernel launch overhead for small $N$. The host-side square root costs one fp32 op on the CPU side, negligible against everything else.</span>

<span style="font-size: 14px;">Floating-point accuracy is worth a paragraph. The reduction is a sum of squared fp32 values, which means the partial sum can grow much faster than a plain sum of the same length: $\sum x_i^2$ has expected magnitude $N \cdot \mathbb{E}[x^2]$, while $\sum x_i$ has expected magnitude $N \cdot \mathbb{E}[x]$ which is often near zero for centered data. The larger accumulator value means rounding error compounds more aggressively, and the typical error grows like $N \cdot \varepsilon$ in the worst case (or $\log_2(N) \cdot \varepsilon$ for the tree-reduced form). The test harness widens the tolerance to combined $\texttt{atol=1e-2, rtol=1e-2}$ specifically for the large-$N$ cases. For the all-zeros input, the answer is exactly $0$ and the comparison is exact; for ramped inputs in $[-1, 1]$, the accumulated sum stays under $N$ in magnitude and fp32 holds the result to the required tolerance.</span>

---

## <span style="font-size: 16px;">Compiler-Handled vs Author-Handled</span>

<span style="font-size: 14px;">**Author chooses:** the grid shape, the constexpr block size, the mask predicate, the $\texttt{other=0.0}$ sentinel, the in-tile fusion of the square into the reduction expression ($\texttt{tl.sum(x * x, axis=0)}$), the one-atomic-per-program combine into a scalar scratch buffer, and the deliberate decision to do the final square root on the host. None of these can be inferred from the source.</span>

<span style="font-size: 14px;">**Compiler handles:** lowering $\texttt{tl.load}$ to coalesced wide HBM transactions, lowering the lane-wise $\texttt{x * x}$ multiply to PTX FMA-style instructions, lowering $\texttt{tl.sum}$ to a warp-shuffle reduction tree with shared-memory staging for the cross-warp combine, scheduling the multiply and the reduction so they overlap with the load, and emitting the atomic as a single PTX $\texttt{atom.global.add.f32}$. The author writes no shuffle, no shared-memory declaration, no barrier. The same call to $\texttt{tl.sum}$ that does a plain sum does this kernel's reduction; only the operand changed.</span>

<span style="font-size: 14px;">The author choice about the host-side square root deserves one line of justification. Calling $\texttt{tl.sqrt}$ inside the kernel before the atomic would compute $\sqrt{\sum x^2}$ per program, then sum those program-level square roots, which is mathematically wrong because $\sqrt{a} + \sqrt{b} \ne \sqrt{a + b}$. The square root has to be applied exactly once on the final combined sum, and the simplest place to do that is the host. A small final-stage kernel could do the same on the device, but for one fp32 op it would cost a full kernel launch overhead, which is strictly worse.</span>

---

## <span style="font-size: 16px;">Naive vs Optimized</span>

<span style="font-size: 14px;">The naive implementation is two kernels: one that writes a squared copy of $x$ to a temporary $N$-element buffer in HBM, and a second that sums the temporary. This is $3 \times$ the HBM traffic and $2 \times$ the launch overhead, with no compensating gain. The fused single-pass kernel folds the multiply into the same load-reduce expression and reads $x$ exactly once.</span>

<span style="font-size: 14px;">A second naive choice is to call $\texttt{tl.sqrt}$ inside the kernel before the atomic, which is wrong as discussed. A subtler version of the same mistake is to take $\sqrt{\cdot}$ of the in-tile reduction $\texttt{block\_sumsq}$ and atomic-add the program-level square root: the kernel runs, atomics are well-formed, and the answer is silently incorrect. The harness catches this if it ever passes for the trivial single-program case but fails as soon as $G > 1$.</span>

<span style="font-size: 14px;">A third axis is precision: for very long $x$ with non-trivial magnitudes, the sum-of-squares can overflow fp32 even when individual squared lanes do not. The mitigation in production code is to scale $x$ by an estimated $\max |x|$ before squaring, accumulate in scaled space, and unscale after the square root. The current test harness bounds the input range so this scaling is not necessary, but the technique is standard in linear-algebra libraries and is worth knowing about as the next step.</span>

<span style="font-size: 14px;">At very large $G$, the cross-program atomic stream becomes a bottleneck and a two-stage kernel (per-program partials to scratch, then a single-program reduce) takes over, exactly as it does for plain sum. At the current problem size the single-stage atomic form is the right choice.</span>

---

## <span style="font-size: 16px;">Worked Example</span>

<span style="font-size: 14px;">Take $N = 5$, $x = [1, 2, 2, 1, 0]$, and $\texttt{BLOCK\_SIZE} = 4$. The host allocates and zeros $\texttt{sumsq\_buf}$, then launches a grid of $\lceil 5 / 4 \rceil = 2$ programs.</span>

<span style="font-size: 14px;">**Program 0** ($\texttt{pid} = 0$): offsets $[0, 1, 2, 3]$, mask all true, loads $[1, 2, 2, 1]$. The lane-wise square in registers produces $[1, 4, 4, 1]$, and $\texttt{tl.sum}$ collapses that to $10$. One atomic add: $\texttt{sumsq\_buf} \mathrel{+}= 10$.</span>

<span style="font-size: 14px;">**Program 1** ($\texttt{pid} = 1$): offsets $[4, 5, 6, 7]$, mask $[\texttt{T}, \texttt{F}, \texttt{F}, \texttt{F}]$, $\texttt{other=0.0}$, loads $[0, 0, 0, 0]$ (the real lane $4$ value of $0$ plus three masked zeros). Squared in registers, the tile is still $[0, 0, 0, 0]$. $\texttt{tl.sum} = 0$. One atomic add: $\texttt{sumsq\_buf} \mathrel{+}= 0$.</span>

<span style="font-size: 14px;">After the kernel returns, $\texttt{sumsq\_buf} = 10$. The host takes $\sqrt{10} \approx 3.1623$ and writes it into $\texttt{out}[0]$. Compare with the bare math: $\sqrt{1 + 4 + 4 + 1 + 0} = \sqrt{10}$, matching exactly. The systems point is that the kernel saw the input once, squared it in registers, and emitted two atomics into a single scratch scalar. No intermediate buffer was ever allocated, and no square root ran inside any program.</span>

---

## <span style="font-size: 16px;">Pitfalls</span>

* <span style="font-size: 14px;">**Squaring after the reduction.** Writing $\texttt{tl.sum(x) * tl.sum(x)}$ is the wrong identity: it computes $(\sum x)^2$, which is not $\sum x^2$. The correct expression is $\texttt{tl.sum(x * x, axis=0)}$, with the square folded into the reduction operand.</span>
* <span style="font-size: 14px;">**Calling $\texttt{tl.sqrt}$ inside the kernel before the atomic.** $\sqrt{a} + \sqrt{b} \ne \sqrt{a + b}$, so taking the square root per program and summing the square roots gives a different number than the L2 norm. The square root must be the final op, applied once on the fully combined sum.</span>
* <span style="font-size: 14px;">**Skipping $\texttt{other=0.0}$ or using a non-zero sentinel.** Undefined or extreme tail-lane values get squared into the reduction. The clean choice is $0$, which is both the additive identity and immune to overflow under the squaring transform.</span>
* <span style="font-size: 14px;">**Reusing a scratch buffer without re-zeroing it.** Atomics accumulate stale state. Always allocate the scratch via $\texttt{torch.zeros}$ inside $\texttt{solve}$, or explicitly $\texttt{zero\_()}$ it before each launch. Same failure mode as plain sum: the first call works, later calls drift.</span>

---