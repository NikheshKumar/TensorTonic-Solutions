# <span style="font-size: 20px;">1D Convolution</span>

<span style="font-size: 14px;">A 1D convolution slides a small filter along an input signal, producing one output per position from a weighted sum of a local window. It is the canonical **stencil**: every output reads a contiguous neighbourhood of the input, and adjacent outputs overlap heavily in the inputs they touch. That overlap - input **reuse** - is the entire systems story, because it is what separates a naive kernel that rereads DRAM from a fast one that loads each element once.</span>

---

## <span style="font-size: 16px;">The Operation</span>

<span style="font-size: 14px;">For an input of length $N$ and a filter of length $kN$, valid padding and stride 1, the output has length $N - kN + 1$. Using cross-correlation (no kernel flip), output position $i$ is:</span>

$$
\text{output}[i] = \sum_{a=0}^{kN-1} \text{input}[i+a] \cdot \text{kernel}[a]
$$

<span style="font-size: 14px;">All buffers are contiguous, row-major float arrays in device global memory. The filter is small and read-only, shared by every output. Output $i$ reads the window `input[i .. i+kN-1]`, so output $i+1$ reuses all but one of those elements. Valid padding means no edge values are synthesized: the output is strictly smaller than the input, and every window stays inside bounds, so there is no boundary special-casing on the read side.</span>

---

## <span style="font-size: 16px;">Parallelization Strategy</span>

<span style="font-size: 14px;">The outputs are mutually independent, so the decomposition is **one thread per output element** over a one-dimensional grid. Each thread reconstructs its output index and runs the inner `kN`-step accumulation locally:</span>

$$
\text{idx} = \text{blockIdx.x} \times \text{blockDim.x} + \text{threadIdx.x}
$$

<span style="font-size: 14px;">A block size of 256 is conventional: a multiple of the 32-lane **warp** so no lanes are wasted, and large enough to give the scheduler many warps per **SM (Streaming Multiprocessor)** for latency hiding. The grid needs $\lceil (N - kN + 1) / 256 \rceil$ blocks, and a bounds check `if (idx < N - kN + 1)` guards the tail threads in the rounded-up final block from reading past the valid output range.</span>

<span style="font-size: 14px;">The inner accumulation loop has a fixed trip count `kN` that is identical for every active thread, so a naive kernel has no **warp divergence**: all lanes march through the same window offsets in lockstep. The only branch is the bounds check at the top, and a small fixed `kN` is a natural candidate for `#pragma unroll`, turning the loop into a flat sequence of fused multiply-adds with no loop overhead and the filter offsets resolved at compile time.</span>

---

## <span style="font-size: 16px;">Memory Hierarchy and Access Pattern</span>

<span style="font-size: 14px;">A naive thread issues `kN` global loads of `input` plus `kN` loads of `kernel`, then one store. The killer is reuse: thread $i$ and thread $i+1$ load windows that overlap in `kN-1` positions, so across a block of 256 threads with a width-`kN` filter, each input element is fetched from DRAM roughly `kN` times. The arithmetic is trivial; the redundant global traffic is the bottleneck.</span>

<span style="font-size: 14px;">Two parts of the hierarchy fix this. The filter is tiny and every thread reads the same `kN` weights, so it belongs in **constant memory**, which is cached and broadcasts a single address to all lanes of a warp in one access. The overlapping input is staged once into `__shared__` memory: a block loads a tile covering its outputs plus a **halo** of `kN-1` extra elements on the trailing edge, after which every window read hits shared memory instead of DRAM.</span>

<span style="font-size: 14px;">Coalescing is favourable for the cooperative load. When the threads of a block read consecutive input addresses to fill the shared tile, the 32 lanes of a warp hit 32 consecutive words and the loads are fully **coalesced**. The naive per-thread window reads are also contiguous within each thread but redundant across threads, which is exactly what the tile eliminates.</span>

<span style="font-size: 14px;">Shared memory sits in the **memory hierarchy** roughly an order of magnitude lower in latency than global memory and is explicitly managed per block, acting as a user-controlled L1. Constant memory is a separate read-only path with its own cache; its broadcast behaviour is ideal for the filter precisely because all lanes want the same weight at the same step, so a single cache line feeds the whole warp.</span>

---

## Memory-bound or compute-bound for small $kN$?

<span style="font-size: 14px;">Per output, the kernel performs `kN` multiply-adds ($2 \cdot kN$ FLOPs) against the bytes it moves. A naive kernel moves about `kN` input loads plus the store, so its **arithmetic intensity** is roughly $2 \cdot kN$ FLOPs per $4(kN+1)$ bytes, near $0.5$ FLOP/byte and far under the **roofline** ridge point of tens of FLOPs per byte. For the small filters typical here, 1D convolution is firmly **memory-bound**: redundant input traffic, not the multiplier, sets the runtime.</span>

<span style="font-size: 14px;">Tiling raises intensity by collapsing the redundant loads to one per element, but with a small `kN` the reuse factor is small and the kernel stays bandwidth-limited. The optimization target is therefore effective bandwidth, not FLOP count.</span>

---

## <span style="font-size: 16px;">Hardware Utilization</span>

<span style="font-size: 14px;">Because each output is independent and the inner loop is uniform, occupancy is the main utilization lever for the naive form: enough resident warps per SM to keep issuing loads while others wait on DRAM. A global load costs hundreds of cycles, and the GPU hides that not with caches but by switching to another ready warp, so launching enough blocks keeps the memory pipeline saturated.</span>

<span style="font-size: 14px;">The tiled form adds a `__syncthreads()` barrier between the cooperative load and the window reads, ensuring every thread sees the full tile and halo before computing. That barrier is the only synchronization, and the shared tile is small (block width plus `kN-1` floats), so it costs little occupancy.</span>

<span style="font-size: 14px;">A subtlety in the tile load is the halo: a block of $T$ threads must populate $T + kN - 1$ tile slots, so the trailing `kN-1` elements need extra loads. A common scheme has the first `kN-1` threads issue a second strided load for the halo, which is a brief, contained source of divergence at load time but does not touch the uniform compute loop. The cost is negligible against the global traffic it removes.</span>

---

## <span style="font-size: 16px;">Naive vs Optimized</span>

<span style="font-size: 14px;">The naive kernel is one thread per output, each independently reading its `kN`-element window from global memory. Correct, but it refetches every input about `kN` times and reads the filter from global memory on every step.</span>

<span style="font-size: 14px;">1. **Constant-memory filter**: place the `kN` weights in constant memory so every warp broadcasts each weight in one cached access instead of a global load, removing the filter traffic entirely.</span>

<span style="font-size: 14px;">2. **Shared-memory tile with halo**: each block cooperatively loads its output span plus a trailing `kN-1` halo into `__shared__` memory once, then every thread reads its window from shared memory. This reduces global input traffic from about `kN` reads per element to one, the central win for any stencil.</span>

<span style="font-size: 14px;">A further refinement notes that for very small `kN` the constant-memory filter alone, combined with the hardware L1/L2 caches absorbing the overlapping reads, can rival an explicit shared tile, so the tile pays off most clearly as `kN` and the block width grow. The systems lesson is constant: identify the reuse, then choose the cheapest place in the hierarchy to capture it.</span>

---

## <span style="font-size: 16px;">Worked Example</span>

<span style="font-size: 14px;">Take $N = 6$, $kN = 3$, so the output length is $6 - 3 + 1 = 4$. Output $0$ reads `input[0,1,2]`, output $1$ reads `input[1,2,3]`, output $2$ reads `input[2,3,4]`, output $3$ reads `input[3,4,5]`.</span>

* <span style="font-size: 14px;">**Overlap:** outputs $0$ and $1$ share `input[1,2]` - they differ by exactly one element, the `kN-1 = 2` reuse.</span>
* <span style="font-size: 14px;">**Naive traffic:** four threads issue $4 \times 3 = 12$ input loads for an array of only 6 elements, so the interior values are each fetched up to three times.</span>

<span style="font-size: 14px;">A block-wide shared tile loads `input[0..5]` once (the 4-output span plus a 2-element halo) and serves all 12 window reads from shared memory, cutting global input traffic from 12 loads to 6. The reuse factor here is $12/6 = 2$; for a longer block of $T$ outputs the tile loads $T + kN - 1$ elements to satisfy $T \cdot kN$ window reads, so the factor approaches `kN` as $T$ grows.</span>

---

## <span style="font-size: 16px;">Pitfalls</span>

* <span style="font-size: 14px;">**Forgetting the halo.** A shared tile sized to only the block's outputs omits the trailing `kN-1` inputs the last threads need, producing wrong results at the tile boundary.</span>
* <span style="font-size: 14px;">**Missing `__syncthreads()` after the tile load.** Reading the shared window before every lane has finished writing the tile is a race that yields nondeterministic output.</span>
* <span style="font-size: 14px;">**Out-of-bounds from the bounds check.** Omitting `if (idx < N - kN + 1)` lets tail threads in the rounded-up final block read and write past the valid output range.</span>
* <span style="font-size: 14px;">**Filter in global memory.** Leaving the weights in global memory adds `kN` redundant loads per thread; constant memory broadcasts them once per warp.</span>

---