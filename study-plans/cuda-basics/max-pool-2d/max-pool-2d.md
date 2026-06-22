# <span style="font-size: 20px;">2D Max Pooling</span>

<span style="font-size: 14px;">A 2D max pooling layer slides a small window over an image and emits the maximum of each window, downsampling the spatial dimensions. It is a **stencil** crossed with a **windowed reduction**: every output reads a `kH x kW` neighbourhood like a convolution, but it combines those values with a running `max` rather than a weighted sum. The stride controls how much adjacent windows overlap, which in turn controls how much input **reuse** there is to capture.</span>

---

## <span style="font-size: 16px;">The Operation</span>

<span style="font-size: 14px;">For an input of shape $H \times W$, a window of shape $kH \times kW$ with stride $(sH, sW)$, the output has shape $((H - kH)/sH + 1) \times ((W - kW)/sW + 1)$. Output position $(i, j)$ is the maximum over its window:</span>

$$
\text{output}[i,j] = \max_{0 \le a < kH,\; 0 \le b < kW} \text{input}[i \cdot sH + a,\; j \cdot sW + b]
$$

<span style="font-size: 14px;">All buffers are contiguous row-major arrays in device global memory, so element $(r, c)$ lives at offset `r * W + c`. There are no learned weights: the only operands are the input window and the running maximum, whose identity element is `-FLT_MAX`. Unlike convolution this is a reduction, so the combiner `max` is associative and commutative, meaning the window can be folded in any order without changing the result - a property a parallel implementation can exploit freely.</span>

---

## <span style="font-size: 16px;">Parallelization Strategy</span>

<span style="font-size: 14px;">Outputs are mutually independent, so the decomposition is **one thread per output pixel** over a two-dimensional grid of 2D blocks. Each thread maps its block and lane coordinates to an output row and column, then anchors its window in the input by multiplying by the stride:</span>

$$
\text{col} = \text{blockIdx.x} \cdot \text{blockDim.x} + \text{threadIdx.x}, \quad \text{row} = \text{blockIdx.y} \cdot \text{blockDim.y} + \text{threadIdx.y}
$$

<span style="font-size: 14px;">The thread's window starts at input `(row * sH, col * sW)`. A 16x16 block (256 threads) is conventional: a multiple of the 32-lane **warp**, a 2D shape matching the 2D access, and enough warps per **SM (Streaming Multiprocessor)** for latency hiding. A bounds check `if (row < outH && col < outW)` guards the rounded-up edge blocks.</span>

<span style="font-size: 14px;">Each thread initializes its accumulator to `-FLT_MAX` and folds in every window element with `fmaxf`. Because `fmaxf` is a branchless intrinsic, the reduction carries no per-element `if`, so a naive kernel has no **warp divergence** in the inner loop - every lane executes the same `kH * kW` comparisons in lockstep.</span>

---

## <span style="font-size: 16px;">Memory Hierarchy and Access Pattern</span>

<span style="font-size: 14px;">A naive thread issues `kH * kW` global loads and one store, with no filter traffic at all. How much those loads overlap depends entirely on the stride. When $sH < kH$ or $sW < kW$ the windows overlap and adjacent outputs reuse input, exactly the stencil reuse that motivates a shared tile. When the stride equals the window size the windows tile the input disjointly, every input is read once, and there is no reuse to capture - the common $2 \times 2$ stride-2 pooling case sits here.</span>

<span style="font-size: 14px;">For the overlapping case the fix is a **shared-memory input tile with a halo**, exactly as in 2D convolution: a block of outputs depends on an input region of `(blockH-1)*sH + kH` rows by `(blockW-1)*sW + kW` columns, loaded into `__shared__` memory once, after which every window `max` reads from shared memory. There is no constant memory here because there is no filter; the only read-only operand is the input itself.</span>

<span style="font-size: 14px;">**Coalescing** governs the global loads. With stride 1 the windows of neighbouring lanes are nearly contiguous and warp-wide reads hit consecutive addresses. With a larger stride the per-thread window anchors are spread by `sW` columns, so the warp's loads become strided and fragment into more transactions - the stride that downsamples the output also degrades the access pattern. Staging through a coalesced shared-tile load sidesteps this when reuse exists.</span>

<span style="font-size: 14px;">Shared memory sits roughly an order of magnitude lower in latency than global memory and is user-managed per block, acting as an explicit L1. Whether it is worth using is dictated by the stride: the overlap factor between adjacent windows is what a shared tile converts from repeated DRAM reads into single reads, so the same kernel structure can be either tile-worthy or pointless depending purely on the stride-to-window ratio.</span>

---

## Memory-bound or compute-bound?

<span style="font-size: 14px;">Per output the kernel does `kH * kW` comparisons - which are not even counted as FLOPs - against `kH * kW` loads and one store. The **arithmetic intensity** is essentially zero useful FLOPs per byte, well under any **roofline** ridge point: max pooling is **deeply memory-bound**, even more so than convolution because there is no multiply-add to amortize. The runtime is set purely by how fast the input bytes move.</span>

<span style="font-size: 14px;">This makes the stride the decisive performance variable. A large stride reads fewer total input bytes (less overlap, smaller effective footprint) and writes a smaller output, so it is cheaper; a stride below the window size reads overlapping input and benefits from a shared tile. Either way the lever is effective bandwidth, never arithmetic.</span>

---

## <span style="font-size: 16px;">Hardware Utilization</span>

<span style="font-size: 14px;">A naive kernel has no synchronization and no shared state, so it is one flat wave of independent reductions; occupancy from enough resident warps is the only lever, hiding the hundreds-of-cycles DRAM latency by switching to ready warps while others wait. The tiled form adds one `__syncthreads()` between filling the shared tile and reading windows, and a small per-block tile that does not throttle **occupancy**.</span>

<span style="font-size: 14px;">The `fmaxf` reduction is branchless, so the inner loop never diverges. Divergence can only appear at boundaries when the output size formula leaves a partial window, but valid sizing keeps windows full. The identity `-FLT_MAX` guarantees the first comparison always adopts a real value, so no special-casing of the first element is needed and the loop stays uniform.</span>

<span style="font-size: 14px;">Because pooling carries no learned weights and no multiply, it is often fused with the preceding convolution or activation so the intermediate never round-trips to global memory. As a standalone kernel its only job is to move input bytes and emit a smaller output, so the whole utilization picture reduces to keeping enough warps in flight to saturate the memory pipeline.</span>

---

## <span style="font-size: 16px;">Naive vs Optimized</span>

<span style="font-size: 14px;">The naive kernel is one thread per output, each independently reading its `kH x kW` window from global memory and folding it with `fmaxf`. Correct, and for non-overlapping strides it is already near-optimal because each input is read exactly once.</span>

<span style="font-size: 14px;">1. **Shared-memory tile with halo** (overlapping strides only): when windows overlap, a block loads its input footprint into `__shared__` memory once with coalesced reads, then every thread takes its `max` from shared memory, cutting redundant global loads. With non-overlapping strides this tile gives no benefit and is skipped.</span>

<span style="font-size: 14px;">2. **Branchless reduction and unrolling**: keep the fold in `fmaxf` rather than an `if`, and `#pragma unroll` the fixed `kH * kW` loop so the comparisons become a flat sequence with no loop overhead and no divergence.</span>

---

## <span style="font-size: 16px;">Worked Example</span>

<span style="font-size: 14px;">Take a $4 \times 4$ input, a $2 \times 2$ window, stride $(2, 2)$. The output is $((4-2)/2 + 1) \times ((4-2)/2 + 1) = 2 \times 2$. The four windows anchor at input $(0,0), (0,2), (2,0), (2,2)$.</span>

* <span style="font-size: 14px;">**No overlap:** stride equals window size, so the four $2 \times 2$ windows tile the input disjointly - each of the 16 inputs is read exactly once across all four threads.</span>
* <span style="font-size: 14px;">**Reduction:** output $(0,0)$ folds `input[0,0], input[0,1], input[1,0], input[1,1]` through `fmaxf` starting from `-FLT_MAX`, emitting their maximum.</span>

<span style="font-size: 14px;">Now drop the stride to $(1,1)$: the output grows to $3 \times 3$, windows overlap by one row or column, and interior inputs are read up to four times. That overlap is what a shared tile would amortize, and it is created entirely by choosing a stride smaller than the window.</span>

---

## <span style="font-size: 16px;">Pitfalls</span>

* <span style="font-size: 14px;">**Wrong accumulator identity.** Initializing the running max to `0` instead of `-FLT_MAX` silently clamps all-negative windows to zero; the identity for `max` over floats must be `-FLT_MAX`.</span>
* <span style="font-size: 14px;">**Strided uncoalesced loads.** A large stride spreads each warp's window anchors, fragmenting global reads into many transactions; stage through a coalesced shared-tile load when reuse exists.</span>
* <span style="font-size: 14px;">**Tiling when there is no reuse.** Building a shared tile for non-overlapping strides adds a `__syncthreads()` and load overhead for zero benefit, since each input is already read once.</span>
* <span style="font-size: 14px;">**Out-of-bounds from the bounds check.** Omitting `if (row < outH && col < outW)` lets edge-block threads anchor windows past the valid output region and read or write out of bounds.</span>

---