# <span style="font-size: 20px;">Dropout</span>

<span style="font-size: 14px;">Dropout zeroes a random subset of activations during training and rescales the survivors so the expected value is preserved. With a precomputed mask it reduces to a pure pointwise **map**: $\text{output}[i] = \text{input}[i] \cdot \text{mask}[i] / (1 - p)$, every output depending on exactly one input element with zero communication between threads. The systems-interesting fact is what is missing: the randomness has been lifted out of the kernel entirely, so this is a deterministic, branch-free, bandwidth-bound element-wise pass.</span>

---

## <span style="font-size: 16px;">The Operation</span>

<span style="font-size: 14px;">For an index $i$ in $[0, N)$, a 0/1 mask buffer, and a scalar drop probability $p$:</span>

$$
\text{output}[i] = \text{input}[i] \cdot \text{mask}[i] \cdot \frac{1}{1 - p}
$$

<span style="font-size: 14px;">`input`, `mask`, and `output` are contiguous, row-major buffers of $N$ elements in device global memory. `mask[i]` is `0` or `1`; multiplying by it keeps or kills the element. `p` is a single broadcast scalar passed by value, and the inverse-keep factor $1/(1-p)$ is the same for every thread. There is no structure beyond the index: output $i$ reads input $i$ and mask $i$ and writes output $i$, with nothing shared or reused.</span>

---

## <span style="font-size: 16px;">Parallelization Strategy</span>

<span style="font-size: 14px;">Because the $N$ outputs are mutually independent, the decomposition is **one thread per output element**. A one-dimensional grid of one-dimensional blocks covers the array, and each thread reconstructs its global position from its block and lane coordinates:</span>

$$
\text{idx} = \text{blockIdx.x} \times \text{blockDim.x} + \text{threadIdx.x}
$$

<span style="font-size: 14px;">A block size of 256 is the conventional choice: a multiple of the 32-lane **warp** so no lanes are wasted, large enough to give the scheduler many warps for latency hiding, and small enough that several blocks fit per **SM (Streaming Multiprocessor)** to keep **occupancy** high. The grid needs $\lceil N / 256 \rceil$ blocks, so the tail block usually owns more threads than there are remaining elements; those surplus threads are guarded out by `if (idx < N)`. Without that bounds check the tail lanes read and write past the buffers.</span>

<span style="font-size: 14px;">The scalar $1/(1-p)$ is computed once per thread into a register and reused for that thread's single element. There is no `__syncthreads()` and no shared state: the kernel is one flat wave of independent work, structurally identical to the serial loop with the hardware supplying the index.</span>

---

## <span style="font-size: 16px;">Why the Mask Is a Buffer, Not On-Device RNG</span>

<span style="font-size: 14px;">The defining design decision is that the mask arrives as a precomputed 0/1 buffer rather than being sampled inside the kernel. This keeps the kernel **pure and deterministic**: given the same inputs it always produces the same output, which is exactly what unit tests and gradient checks need. There is no per-thread random state to initialize, advance, or store.</span>

<span style="font-size: 14px;">The alternative - generating randomness on device with `cuRAND` - forces every thread to carry and update a generator state, adds register pressure that lowers occupancy, and makes the result depend on launch geometry. By precomputing the mask the kernel sidesteps all of that. It also keeps the work **branch-free**: instead of `if (keep)` it multiplies by `mask[i]`, so every lane in a warp executes the identical instruction stream and there is zero **warp divergence**. Selection becomes arithmetic, not control flow.</span>

---

## <span style="font-size: 16px;">Memory Hierarchy and Access Pattern</span>

<span style="font-size: 14px;">Per element the kernel loads `input[idx]`, loads `mask[idx]`, and stores `output[idx]`. Each datum is used exactly once, so there is nothing to cache in `__shared__` memory and nothing to keep in registers beyond the scalar factor and the running product. Shared memory exists to enable reuse; a map has no reuse, so adding it would be pure overhead.</span>

<span style="font-size: 14px;">The access pattern is ideal for **coalescing**: the 32 threads of a warp hold consecutive `idx` values, so they read 32 consecutive addresses of `input` and of `mask` and write 32 consecutive addresses of `output`. The memory controller serves each warp-wide request in the minimum number of transactions, delivering near-peak effective bandwidth. The only subtlety is the mask's element width: if the mask is stored as 4-byte floats it doubles the load traffic versus a packed 1-byte representation, which matters because this kernel is bandwidth-limited.</span>

---

## <span style="font-size: 16px;">Memory-Bound or Compute-Bound?</span>

<span style="font-size: 14px;">Assuming float input, mask, and output, the kernel moves 12 bytes per element - two 4-byte loads and one 4-byte store - and performs one multiply by the mask and one by the scale factor. Its **arithmetic intensity** is about:</span>

$$
\frac{2 \text{ FLOP}}{12 \text{ bytes}} \approx 0.17 \text{ FLOP/byte}
$$

<span style="font-size: 14px;">On the **roofline** model a kernel is compute-bound only above the ridge point, which sits in the tens of FLOPs per byte on modern hardware. At $0.17$ this is two orders of magnitude below that line: **deeply memory-bound**. The two multipliers sit idle almost the entire time, waiting for operands to arrive from DRAM. Consequently the only levers that change runtime are bandwidth-side: coalesced access (already optimal), enough warps in flight to hide latency, and minimizing the mask's byte width. Arithmetic cleverness buys nothing because there is barely any arithmetic.</span>

---

## <span style="font-size: 16px;">Hardware Utilization and Latency Hiding</span>

<span style="font-size: 14px;">A global load costs hundreds of cycles. The GPU hides that not with caches but with **massive multithreading**: when a warp issues its loads and stalls, the SM scheduler switches to another resident warp that is ready. With high occupancy there is always other work to issue and the memory pipeline stays saturated. Because the kernel has no divergent branches, no synchronization, and trivial register use, occupancy is limited only by launching enough blocks - automatic for the large $N$ that dropout layers see. The precomputed-mask design directly helps here: no `cuRAND` state means low register pressure and therefore more resident warps.</span>

<span style="font-size: 14px;">The contrast with an on-device RNG kernel is instructive. A `cuRAND` generator state is tens of bytes of per-thread register or local-memory footprint, and the **SM** has a fixed register file shared by all resident warps. Heavier per-thread state means fewer warps fit, fewer warps means less latency hiding, and a memory-bound kernel that cannot hide latency stalls. Pushing the randomness into a precomputed buffer keeps each thread's state down to a couple of registers, so the SM stays packed with warps and the memory pipeline stays fed. The mask buffer trades a one-time generation pass for a leaner, faster, deterministic inner kernel.</span>

---

## <span style="font-size: 16px;">Naive vs Optimized</span>

<span style="font-size: 14px;">The one-thread-per-element kernel is already near-optimal because the work is bandwidth-bound and coalesced. The remaining headroom comes from moving more bytes per instruction, not from cleverer math:</span>

<span style="font-size: 14px;">1. **Vectorized loads**: reinterpreting the arrays as `float4` lets each thread move 16 bytes per load and store instead of 4, issuing fewer, wider transactions and nudging the kernel toward the bandwidth ceiling.</span>

<span style="font-size: 14px;">2. **Grid-stride loop**: launch a fixed device-sized grid and let each thread stride through the array in steps of `blockDim.x * gridDim.x`. One configuration then handles any $N$ and amortizes launch overhead.</span>

<span style="font-size: 14px;">3. **Fusion**: because dropout is a trivial map, it is a prime candidate to be fused into the preceding kernel (an activation, for example) so the activation's output never round-trips through global memory just to be masked. Fusion removes a full read-write pass, the single largest possible speedup for a memory-bound element-wise op.</span>

---

## <span style="font-size: 16px;">Worked Example</span>

<span style="font-size: 14px;">Take $N = 6$, block size 4, and $p = 0.5$ so the keep factor is $1/(1 - 0.5) = 2$. The grid needs $\lceil 6 / 4 \rceil = 2$ blocks, eight threads in total.</span>

* <span style="font-size: 14px;">**Block 0** (`blockIdx.x = 0`): threads compute `idx` $= 0, 1, 2, 3$, all pass `idx < 6`, and write `output[0..3]`.</span>
* <span style="font-size: 14px;">**Block 1** (`blockIdx.x = 1`): threads compute `idx` $= 4, 5, 6, 7$; indices $4$ and $5$ pass and write `output[4]` and `output[5]`; indices $6$ and $7$ fail `idx < 6` and exit.</span>

<span style="font-size: 14px;">With `input` $= [1, 2, 3, 4, 5, 6]$ and `mask` $= [1, 0, 1, 1, 0, 1]$, the six active threads independently produce `output` $= [2, 0, 6, 8, 0, 12]$ - each kept element doubled, each dropped element zeroed. Every lane ran the identical multiply-multiply sequence; the zeros came from data, not from a branch, so no warp diverged.</span>

---

## <span style="font-size: 16px;">Pitfalls</span>

* <span style="font-size: 14px;">**Sampling randomness inside the kernel.** Rolling per-thread `cuRAND` state breaks determinism, adds register pressure that cuts occupancy, and ties the result to launch geometry; the precomputed mask exists precisely to avoid this.</span>
* <span style="font-size: 14px;">**Branching on the mask.** Writing `if (mask[i]) ...` instead of multiplying serializes a warp whenever lanes disagree; multiply by `mask[i]` to stay branchless.</span>
* <span style="font-size: 14px;">**Forgetting the $1/(1-p)$ scale.** Dropping the keep factor changes the expected activation magnitude and silently shifts the layer's statistics; the rescale is what makes inference and training consistent.</span>
* <span style="font-size: 14px;">**Omitting the bounds check.** When $N$ is not a multiple of the block size the tail block has lanes with `idx >= N`; without `if (idx < N)` they read and write out of bounds.</span>

---