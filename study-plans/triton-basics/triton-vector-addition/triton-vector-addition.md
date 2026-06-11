# <span style="font-size: 20px;">Vector Addition</span>

<span style="font-size: 14px;">Elementwise vector addition is the canonical **pointwise map**: every output element depends on exactly one element of each input and on nothing else. In Triton, this is the simplest possible kernel and the cleanest demonstration of the tile-and-mask model. There is no reduction, no cross-program communication, no shared memory, and no synchronization. The whole kernel is a single triplet of load, add, store at the tile level, repeated across the launch grid.</span>

---

## <span style="font-size: 16px;">The Operation</span>

<span style="font-size: 14px;">Given two contiguous float tensors $x, y \in \mathbb{R}^{N}$, the kernel writes their elementwise sum into a pre-allocated output $\texttt{out} \in \mathbb{R}^{N}$:</span>

$$
\texttt{out}[i] = x[i] + y[i], \quad 0 \le i < N
$$

<span style="font-size: 14px;">All three tensors are 1D, the same length $N$, the same dtype (`torch.float32`), and live in HBM on the GPU. The launcher allocates $\texttt{out}$; the kernel writes into it in place and returns nothing.</span>

---

## <span style="font-size: 16px;">Program Decomposition</span>

<span style="font-size: 14px;">The launch grid is one-dimensional, with $\lceil N / \texttt{BLOCK\_SIZE} \rceil$ **programs**. Each program is identified by $\texttt{tl.program\_id(0)}$, an integer in $[0, \texttt{cdiv}(N, \texttt{BLOCK\_SIZE}))$, and owns a single contiguous tile of $\texttt{BLOCK\_SIZE}$ consecutive elements of the input. The starting offset for program $p$ is $p \cdot \texttt{BLOCK\_SIZE}$, and the lane-wise offsets inside the program form the tile $\texttt{offs} = p \cdot \texttt{BLOCK\_SIZE} + \texttt{tl.arange}(0, \texttt{BLOCK\_SIZE})$.</span>

<span style="font-size: 14px;">No two programs touch the same output element, and no program reads or writes any tensor element another program touches. The kernel is **embarrassingly parallel at the program level**: programs can execute in any order, concurrently, with no barriers and no atomics. This is the easiest possible parallel pattern, and it is why vector add is the "hello world" of GPU kernel writing across every framework.</span>

---

## <span style="font-size: 16px;">Tile Shape and Masking</span>

<span style="font-size: 14px;">Triton tile shapes are compile-time constants, declared with $\texttt{tl.constexpr}$. The standard choice for a 1D map is $\texttt{BLOCK\_SIZE} = 1024$: a power of two so the compiler can pick wide vector loads, large enough that the per-program launch overhead is amortized over real work, small enough that the register footprint stays well inside what the hardware exposes per program. The compiler uses the constexpr value to size registers, unroll the inner sequence, and emit the right PTX vector instructions.</span>

<span style="font-size: 14px;">Because $\texttt{BLOCK\_SIZE}$ is fixed at compile time but $N$ is a runtime value, the last program almost always overshoots: if $N = 1{,}000{,}000$ and $\texttt{BLOCK\_SIZE} = 1024$, the final program covers offsets $999{,}424 \dots 1{,}000{,}447$, of which the last $448$ lanes are past the end of the buffer. The mask $\texttt{mask} = \texttt{offs} < N$ disables those out-of-range lanes on every $\texttt{tl.load}$ (so the kernel does not read garbage) and on every $\texttt{tl.store}$ (so the kernel does not write past the output). The mask is a **correctness** tool, not a performance one: without it, the kernel touches memory it does not own and produces undefined behavior.</span>

---

## <span style="font-size: 16px;">Memory Hierarchy and Reuse</span>

<span style="font-size: 14px;">Vector addition has the simplest possible memory pattern: every input element is loaded from HBM exactly once, every output element is stored to HBM exactly once, and **nothing is reused**. The tile that one program holds is consumed and discarded; no other program will ever ask for those bytes again. As a consequence, the kernel does not stage anything into SRAM and does not need shared memory at all. The compiler leaves the operands in registers between load, add, and store.</span>

<span style="font-size: 14px;">The single thing the kernel does need from the hardware is **coalesced** HBM access. When the lane offsets inside a tile are contiguous, as they are here ($p \cdot \texttt{BLOCK\_SIZE} + 0, 1, 2, \dots$), the compiler emits the load as a small number of wide memory transactions instead of $\texttt{BLOCK\_SIZE}$ separate ones. This is the difference between hitting peak HBM bandwidth and getting a fraction of it. Triton authors do not write coalescing rules by hand the way CUDA authors do; they get coalescing for free by expressing contiguous tiles and trusting the compiler to lower the load to the right vector width.</span>

<span style="font-size: 14px;">A useful number for ground truth: modern accelerators serve HBM in transactions of $32$ or $128$ bytes. A $\texttt{BLOCK\_SIZE} = 1024$ tile of fp32 inputs is $4096$ bytes per tensor, or $32$ transactions of $128$ B each when the access is perfectly coalesced. If the access pattern were strided by $4$ (every fourth element), the same $1024$ logical loads would cost $4\times$ more transactions, since each transaction now carries useful data for only one of every four lanes. Vector addition trivially avoids this because every lane is exactly one fp32 offset from its neighbor; the value of writing in Triton is that the load syntax does not have to express this — the contiguous $\texttt{tl.arange}$ offsets carry the intent and the compiler chooses the transaction width.</span>

<span style="font-size: 14px;">The L2 cache is incidental to this kernel. With no reuse, no two programs ask for the same line, and a line that the L2 happens to capture during one program's load is never queried again. The cache is not hurting (its presence is free) but it is not helping either — the kernel runs at HBM speed, not L2 speed, and any benchmark that reports faster numbers than peak HBM bandwidth is almost certainly measuring a warm cache or a tensor small enough to fit in L2 entirely.</span>

---

## <span style="font-size: 16px;">Memory-Bound vs Compute-Bound</span>

<span style="font-size: 14px;">Per output element, the kernel reads $4 + 4 = 8$ bytes from HBM (one fp32 from $x$, one from $y$), writes $4$ bytes back, and performs exactly one fused multiply-add-style operation (one addition). The **arithmetic intensity** is therefore</span>

$$
\frac{1 \text{ FLOP}}{12 \text{ bytes}} \approx 0.083 \text{ FLOPs/byte}
$$

<span style="font-size: 14px;">That sits orders of magnitude under the roofline's memory ceiling on any modern accelerator (peak fp32 throughput is typically tens of TFLOPs while HBM bandwidth is a few TB/s, putting the crossover point at roughly $10$ FLOPs/byte). Vector addition is firmly **memory-bound**, and the only optimizations that change its runtime are the ones that affect effective bandwidth: contiguous access for coalescing, enough programs to saturate the streaming multiprocessors, and a block size large enough that launch overhead is negligible. No amount of arithmetic cleverness can help, because there is essentially no arithmetic.</span>

<span style="font-size: 14px;">It is useful to anchor this number against the rest of the Triton track. Pointwise activations (ReLU, GELU, SiLU) sit right next to vector add at $\approx 0.1$ FLOPs/byte and are equally memory-bound. Per-row reductions like fused softmax raise the intensity slightly because each input element participates in a max, an exp, and a normalize — perhaps $\approx 0.3$ FLOPs/byte — but stay memory-bound. Tiled matmul is the kernel where intensity becomes a tuning knob: each loaded operand tile of size $\texttt{BLOCK\_K}$ is reused across $\texttt{BLOCK\_M}$ or $\texttt{BLOCK\_N}$ output lanes, so intensity scales as $\Theta(\texttt{BLOCK\_K})$ and crosses into compute-bound territory once $\texttt{BLOCK\_K}$ is in the tens. Vector addition is the floor of this spectrum, and every later kernel in the curriculum is calibrated against it.</span>

---

## <span style="font-size: 16px;">Compiler-Handled vs Author-Handled</span>

<span style="font-size: 14px;">Triton's pitch is that the compiler hides the parts of GPU programming that are mechanical and error-prone, while leaving the parts that require kernel-level intent in the author's hands. Vector addition is small enough to enumerate both sides exactly.</span>

<span style="font-size: 14px;">**Author chooses:** the grid shape ($\lceil N / \texttt{BLOCK\_SIZE} \rceil$ programs in 1D), the constexpr block size, the offset arithmetic ($p \cdot \texttt{BLOCK\_SIZE} + \texttt{tl.arange}$), the mask predicate ($\texttt{offs} < N$), and the entry-function signature. These are the kernel-design decisions: nothing else in the compiler can pick the right block size or know that the program ID indexes into a 1D tile.</span>

<span style="font-size: 14px;">**Compiler handles:** lowering the tile-level $\texttt{tl.load}$ and $\texttt{tl.store}$ to wide PTX memory instructions of the right vector width, allocating registers for the tile, deciding how the tile is sharded across warps inside the program, inserting any pipelining, and emitting the actual machine code. The author never names a warp, never writes a coalescing rule, never declares shared memory, never inserts a synchronization barrier. For a map kernel there is nothing to synchronize.</span>

<span style="font-size: 14px;">The warp story is worth one explicit line because it is where Triton most clearly diverges from CUDA. Inside one program, the $\texttt{BLOCK\_SIZE} = 1024$ tile is internally sharded by the compiler across $\texttt{num\_warps}$ warps (default $4$), with each warp owning $256$ lanes of the tile. Different warps load different slices of HBM in parallel, and while one warp is stalled waiting on a load to return, the others can issue arithmetic — the same **latency-hiding** mechanism CUDA authors achieve by hand with high occupancy. In Triton the author does not pick warp counts unless they autotune over $\texttt{num\_warps}$; the default is tuned to be sensible for the common case.</span>

---

## <span style="font-size: 16px;">Naive vs Optimized</span>

<span style="font-size: 14px;">For vector addition, the canonical kernel described above is already the optimized form: bandwidth-bound code with coalesced access and minimal launch overhead. The interesting variants tune the block size against the work distribution. A larger block (for example, $\texttt{BLOCK\_SIZE} = 4096$) reduces the program count by $4\times$, which can help when launch overhead matters relative to the per-program work, and lets the compiler emit wider vector loads. A smaller block raises the program count, which can help saturate the device on small $N$ where one $1024$-element program per SM is not enough to fill the pipeline.</span>

<span style="font-size: 14px;">In both directions the impact is in the single-digit-percent range on real hardware, because the kernel is already running at near-peak HBM bandwidth. The much larger optimization is fusion: combining the add with whatever produced $x$ and $y$ (or whatever consumes $\texttt{out}$) into a single kernel removes one round-trip through HBM. That is the lesson that scales from vector add up to FlashAttention.</span>

<span style="font-size: 14px;">A useful back-of-the-envelope: kernel launch overhead on a modern GPU is on the order of $5\!-\!10$ microseconds. A $1$M-element fp32 vector add moves $12$ MB through HBM (two inputs read, one output written), which at $1$ TB/s of bandwidth takes about $12$ microseconds. Launch overhead is therefore a comparable fraction of total runtime for small $N$, which is precisely the regime where increasing $\texttt{BLOCK\_SIZE}$ to drop program count pays back. For $N$ in the hundreds of millions, both numbers become irrelevant and the kernel simply runs at HBM bandwidth.</span>

---

## <span style="font-size: 16px;">Worked Example</span>

<span style="font-size: 14px;">Take $N = 6$ and $\texttt{BLOCK\_SIZE} = 4$. The launch grid is $\lceil 6 / 4 \rceil = 2$ programs.</span>

<span style="font-size: 14px;">**Program 0** ($\texttt{pid} = 0$): the starting offset is $0$, and the tile of offsets is $\texttt{offs} = [0, 1, 2, 3]$. The mask $\texttt{offs} < 6$ evaluates to $[\texttt{T}, \texttt{T}, \texttt{T}, \texttt{T}]$, so all four lanes are active. The program loads $x[0..3]$ and $y[0..3]$, adds them lane-wise in registers, and stores the result into $\texttt{out}[0..3]$.</span>

<span style="font-size: 14px;">**Program 1** ($\texttt{pid} = 1$): the starting offset is $4$, and the tile of offsets is $\texttt{offs} = [4, 5, 6, 7]$. The mask $\texttt{offs} < 6$ evaluates to $[\texttt{T}, \texttt{T}, \texttt{F}, \texttt{F}]$. Lanes $0$ and $1$ load $x[4], x[5]$ and $y[4], y[5]$; lanes $2$ and $3$ are masked off and their loaded values are whatever the $\texttt{other}$ argument provides (irrelevant, since the same mask gates the store). The program writes only $\texttt{out}[4]$ and $\texttt{out}[5]$. The hypothetical $\texttt{out}[6]$ and $\texttt{out}[7]$ slots, which do not exist, are never touched.</span>

<span style="font-size: 14px;">Both programs execute concurrently. Program 1 does not have to wait for Program 0 because nothing it does depends on what Program 0 produces. The full result $\texttt{out}[0..5]$ is the union of what the two programs wrote, with the masked lanes in Program 1 silently skipped.</span>

---

## <span style="font-size: 16px;">Pitfalls</span>

* <span style="font-size: 14px;">**Forgetting the tail mask.** The last program's tile almost always overshoots $N$. Without $\texttt{mask} = \texttt{offs} < N$ guarding every $\texttt{tl.load}$ and $\texttt{tl.store}$, the kernel reads garbage from past the input buffer and writes past the output buffer. This is a correctness bug, not a performance one; small $N$ may even appear to work because the buffer happens to be padded.</span>
* <span style="font-size: 14px;">**Block size not declared $\texttt{tl.constexpr}$.** A runtime block size prevents the compiler from sizing registers, unrolling the load-add-store sequence, and picking the right vector width. The kernel will still compile but will lose most of its speed and may even fall back to a scalar loop.</span>
* <span style="font-size: 14px;">**Choosing a non-power-of-two block size.** The compiler can vectorize cleanly when the block size is a power of two; non-power-of-two values force scalar fallbacks for partial vectors. Standard values are $128, 256, 512, 1024, 2048, 4096$, selected against the input size and the target hardware.</span>
* <span style="font-size: 14px;">**Treating the kernel as compute-bound.** Vector addition is at the absolute floor of the roofline. Adding more arithmetic per element (a scale, a bias, an activation) is essentially free up to a point because the kernel is dominated by HBM traffic. This is the motivation for fused kernels (vector add + ReLU, fused FMA, fused bias + activation): they pay for themselves by removing extra round-trips through HBM, not by doing the arithmetic faster.</span>

---