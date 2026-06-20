# <span style="font-size: 20px;">Swish</span>

<span style="font-size: 14px;">Swish gates each value by its own sigmoid, $\text{output}[i] = \text{input}[i] \cdot \sigma(\text{input}[i]) = \text{input}[i] / (1 + e^{-\text{input}[i]})$. It is an **embarrassingly parallel map**: every output depends on exactly one input at the same index, with zero communication between threads. The systems angle is that swish reuses the sigmoid's transcendental `expf` and then folds in one multiply by the input itself, so its profile sits between sigmoid and a plain copy - transcendental, but still firmly bandwidth-limited at scale.</span>

---

## <span style="font-size: 16px;">The Operation</span>

<span style="font-size: 14px;">For an index $i$ in $[0, N)$, the kernel evaluates:</span>

$$
\text{output}[i] = \frac{\text{input}[i]}{1 + e^{-\text{input}[i]}}
$$

<span style="font-size: 14px;">Input and output are contiguous, row-major buffers of $N$ 32-bit floats in device (global) memory. Output element $i$ reads only input element $i$ and writes only output element $i$. Note the input value is used twice in one thread - once as the gate argument inside the sigmoid and once as the multiplicand outside - but it is loaded once into a register, not read twice from memory.</span>

---

## <span style="font-size: 16px;">Parallelization Strategy</span>

<span style="font-size: 14px;">Because the $N$ outputs are mutually independent, the natural decomposition is **one thread per element**. A one-dimensional grid of one-dimensional blocks covers the array, and each thread reconstructs its global position:</span>

$$
\text{idx} = \text{blockIdx.x} \times \text{blockDim.x} + \text{threadIdx.x}
$$

<span style="font-size: 14px;">A block size of 256 threads is conventional: it is a multiple of the 32-lane **warp** so no lanes are wasted, it gives the scheduler many warps per block for latency hiding, and many such blocks fit on one **SM (Streaming Multiprocessor)**, keeping **occupancy** high. The grid needs $\lceil N / 256 \rceil$ blocks.</span>

<span style="font-size: 14px;">The body is guarded by `if (idx < N)` because rounding the grid up leaves surplus tail threads; without the check they read and write past the buffers. There is no `__syncthreads()` and no shared state - the whole computation is one flat wave of independent work.</span>

---

## <span style="font-size: 16px;">Memory Hierarchy and Access Pattern</span>

<span style="font-size: 14px;">The kernel touches two global arrays per element: it loads `input[idx]` once and stores `output[idx]` once. The "reuse" of the input value within the formula is **register reuse**, not memory reuse - the loaded value sits in a register and is consumed twice by the arithmetic, costing nothing extra in global traffic. There is no cross-thread reuse, so nothing belongs in `__shared__` memory; a map loads each datum exactly once.</span>

<span style="font-size: 14px;">The access pattern is ideal for **coalescing**: the 32 threads of a warp hold consecutive `idx` values, so they read 32 consecutive addresses of `input` and write 32 consecutive addresses of `output`. The memory controller serves each warp-wide request in the minimum number of transactions, delivering near-peak effective bandwidth. This unit-stride layout is the best case the hardware offers.</span>

---

## <span style="font-size: 16px;">Memory-Bound or Compute-Bound?</span>

<span style="font-size: 14px;">Per element the kernel moves 8 bytes - one 4-byte load and one 4-byte store - and performs a negate, the transcendental `expf`, an add, a reciprocal, and a final multiply by the input. Counting roughly a dozen-plus effective FLOPs, the **arithmetic intensity** is about:</span>

$$
\frac{\sim 13 \text{ FLOP}}{8 \text{ bytes}} \approx 1.6 \text{ FLOP/byte}
$$

<span style="font-size: 14px;">That is marginally above sigmoid because of the extra gating multiply, but the ridge point of the **roofline** sits in the tens of FLOPs per byte, so at $\approx 1.6$ swish stays well below it: **memory-bound at scale**. The one extra multiply over sigmoid is essentially free - it overlaps with the same outstanding memory transactions - and with enough warps in flight the DRAM bandwidth, not the arithmetic, sets the runtime.</span>

---

## <span style="font-size: 16px;">Hardware Utilization and Latency Hiding</span>

<span style="font-size: 14px;">A global-memory load costs hundreds of cycles; the SFU `expf` plus the surrounding multiplies cost a few. The GPU hides the dominant memory latency with **massive multithreading**: when a warp stalls on its load of `input`, the SM scheduler switches to another resident warp. High occupancy keeps the memory pipeline saturated, and the transcendental work fills the gaps while other warps wait on DRAM. Swish has no data-dependent branch - the same `expf`-reciprocal-multiply sequence runs for positive and negative inputs alike - so every active lane takes the same path and there is no **warp divergence**.</span>

---

## <span style="font-size: 16px;">Naive vs Optimized</span>

<span style="font-size: 14px;">The naive kernel computes the formula literally with `expf`. Two refinements approach the bandwidth ceiling:</span>

<span style="font-size: 14px;">1. **Fuse the sigmoid**: computing $x / (1 + \texttt{expf}(-x))$ directly, rather than calling a separate sigmoid routine and then multiplying, keeps the intermediate in a register and issues one fused sequence. The fast-math `__expf` trades a few low-order bits for higher SFU throughput, which only matters at low occupancy since the kernel is memory-bound.</span>

<span style="font-size: 14px;">2. **Approach the bandwidth ceiling**: a **grid-stride loop** lets a fixed grid handle any $N$ and amortize launch overhead, while vectorized `float4` loads move 16 bytes per instruction and apply swish componentwise, issuing fewer, wider transactions.</span>

<span style="font-size: 14px;">Both sit on top of an already memory-limited pipeline. For a transcendental map the bandwidth roofline is still the wall; fused, faster math only helps until memory becomes the constraint again.</span>

---

## <span style="font-size: 16px;">Worked Example</span>

<span style="font-size: 14px;">Take $N = 6$ with a block size of 4. The grid needs $\lceil 6 / 4 \rceil = 2$ blocks, for 8 threads total - two more than there are elements.</span>

* <span style="font-size: 14px;">**Block 0** (`blockIdx.x = 0`): threads compute `idx` $= 0, 1, 2, 3$ and write `output[0..3]`.</span>
* <span style="font-size: 14px;">**Block 1** (`blockIdx.x = 1`): threads compute `idx` $= 4, 5, 6, 7$. Indices $4$ and $5$ write `output[4..5]`; indices $6$ and $7$ fail `idx < 6` and exit.</span>

<span style="font-size: 14px;">With `input` $= [0, 1, -1, \ldots]$, thread 0 computes $0 \cdot \sigma(0) = 0$, thread 1 computes $1 \cdot \sigma(1) = 1 \cdot 0.731 \approx 0.731$, and thread 2 computes $-1 \cdot \sigma(-1) = -1 \cdot 0.269 \approx -0.269$. Each lane loads its input once, holds it in a register through both the sigmoid and the gating multiply, and runs the identical instruction sequence - so the warp never diverges and no value is reloaded.</span>

---

## <span style="font-size: 16px;">Pitfalls</span>

* <span style="font-size: 14px;">**Reloading the input.** The value is used twice in the formula; keep it in a register after one load rather than indexing `input[idx]` again, which a careless implementation can turn into two global reads.</span>
* <span style="font-size: 14px;">**Treating it as compute-bound.** At $\approx 1.6$ FLOP/byte the extra gating multiply over sigmoid is free; coalescing and occupancy set the runtime, not the arithmetic.</span>
* <span style="font-size: 14px;">**Omitting the bounds check.** When $N$ is not a multiple of the block size the grid rounds up; without `if (idx < N)` the tail threads read and write out of bounds.</span>
* <span style="font-size: 14px;">**Breaking coalescing.** Strided or misaligned indexing fragments a warp's 32 requests into many transactions and collapses effective bandwidth.</span>

---