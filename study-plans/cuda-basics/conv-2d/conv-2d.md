# <span style="font-size: 20px;">2D Convolution</span>

<span style="font-size: 14px;">A 2D convolution slides a small two-dimensional filter across an image, producing one output per spatial position from a weighted sum over a rectangular window. It is the canonical **stencil** in two dimensions: every output reads a `kH x kW` neighbourhood of the input, and a block of adjacent outputs overlaps in almost all of the input they touch. That two-dimensional **reuse** is the systems story, and it is far stronger than in 1D because overlap accumulates along both axes.</span>

---

## <span style="font-size: 16px;">The Operation</span>

<span style="font-size: 14px;">For an input of shape $H \times W$, a filter of shape $kH \times kW$, valid padding and stride 1, the output has shape $(H - kH + 1) \times (W - kW + 1)$. Using cross-correlation (no flip), output position $(i, j)$ is:</span>

$$
\text{output}[i,j] = \sum_{a=0}^{kH-1} \sum_{b=0}^{kW-1} \text{input}[i+a, j+b] \cdot \text{kernel}[a,b]
$$

<span style="font-size: 14px;">All buffers are contiguous row-major arrays in device global memory, so element $(r, c)$ of a width-$W$ image lives at linear offset `r * W + c`. The filter is small and read-only, shared by every output. Output $(i,j)$ reads the `kH x kW` block anchored at $(i,j)$. Valid padding means the output is strictly smaller than the input and every window stays in bounds, so the read side needs no boundary special-casing.</span>

---

## <span style="font-size: 16px;">Parallelization Strategy</span>

<span style="font-size: 14px;">Outputs are mutually independent, so the decomposition is **one thread per output pixel** over a two-dimensional grid of 2D blocks. Each thread maps its block and lane coordinates to an output row and column:</span>

$$
\text{col} = \text{blockIdx.x} \times \text{blockDim.x} + \text{threadIdx.x}, \quad \text{row} = \text{blockIdx.y} \times \text{blockDim.y} + \text{threadIdx.y}
$$

<span style="font-size: 14px;">A 16x16 block (256 threads) is conventional: it is a multiple of the 32-lane **warp**, the 2D shape matches the 2D access pattern, and it gives many warps per **SM (Streaming Multiprocessor)** for latency hiding. The grid tiles the output with $\lceil (W-kW+1)/16 \rceil$ by $\lceil (H-kH+1)/16 \rceil$ blocks. A bounds check `if (row < outH && col < outW)` guards the threads in the rounded-up edge blocks.</span>

<span style="font-size: 14px;">Within a warp, the fastest-varying coordinate is `threadIdx.x`, so the 32 lanes span 32 consecutive columns of one output row. Their input reads therefore march across consecutive columns of the input as well, which is what makes coalescing possible.</span>

---

## <span style="font-size: 16px;">Memory Hierarchy and Access Pattern</span>

<span style="font-size: 14px;">A naive thread issues `kH * kW` global loads of `input` plus `kH * kW` loads of `kernel`, then one store. The reuse is large: two horizontally adjacent outputs share `kH * (kW-1)` input elements, and two vertically adjacent outputs share `(kH-1) * kW`. Across a block of 256 outputs every interior input element is fetched roughly `kH * kW` times from DRAM. The arithmetic is cheap; the redundant global traffic dominates.</span>

<span style="font-size: 14px;">The fix is a **shared-memory input tile with a halo**. A 16x16 block of outputs depends on an input region of `(16 + kH - 1) x (16 + kW - 1)`, the output footprint plus a halo of `kH-1` rows and `kW-1` columns. The block cooperatively loads that whole region into `__shared__` memory once, then every thread reads its `kH x kW` window from shared memory. This amortizes each global input load across all the outputs whose windows overlap it, the central win of any 2D stencil.</span>

<span style="font-size: 14px;">The small filter is read-only and identical for every thread, so it belongs in **constant memory**, whose cache broadcasts one weight to all lanes of a warp in a single access. Filling the shared tile is **coalesced**: consecutive lanes in `threadIdx.x` read consecutive input columns, so each warp-wide row load hits contiguous addresses and the memory controller serves it in the minimum number of transactions.</span>

<span style="font-size: 14px;">Shared memory sits roughly an order of magnitude lower in latency than global memory and is user-managed per block, acting as an explicit L1. The decisive difference from 1D is dimensionality: in 1D the halo is a 1D strip of `kN-1` extra elements, but in 2D the halo wraps the tile on all sides, so the reuse a single shared tile captures grows with the product `kH * kW` rather than just `kW`. That is why 2D convolution rewards tiling far more than 1D.</span>

---

## Memory-bound or compute-bound for small filters?

<span style="font-size: 14px;">Per output the kernel does `kH * kW` multiply-adds against the bytes it moves. A naive kernel moves about `kH * kW` input loads per output, giving an **arithmetic intensity** near $0.5$ FLOP/byte, far below the **roofline** ridge point of tens of FLOPs per byte. For the small filters typical here, 2D convolution is **memory-bound**: redundant input traffic sets the runtime.</span>

<span style="font-size: 14px;">Tiling raises intensity because each loaded element now feeds up to `kH * kW` window reads from shared memory. The reuse factor grows with the tile area relative to its halo, so a 16x16 tile with a 3x3 filter captures most of the available reuse. Even so, with small filters the kernel remains close to bandwidth-limited, so effective bandwidth, not FLOP count, is the optimization target.</span>

---

## <span style="font-size: 16px;">Hardware Utilization</span>

<span style="font-size: 14px;">The tiled kernel needs one `__syncthreads()` barrier between filling the shared tile and reading windows from it, so every lane sees the complete tile and halo before computing. That is the only synchronization. The shared tile costs `(16 + kH - 1) * (16 + kW - 1)` floats per block; for small filters this is a few KB, well within the tens of KB an SM offers, so it does not throttle **occupancy**.</span>

<span style="font-size: 14px;">The compute loop is a uniform `kH * kW` nest with no data-dependent branching, so there is no **warp divergence** in the accumulation. The only branchy code is the bounds check and the halo-load logic, both confined to the tile-loading phase. Occupancy from enough resident warps remains the lever that hides the hundreds-of-cycles DRAM latency of the initial tile load.</span>

<span style="font-size: 14px;">Filling a `(16 + kH - 1) x (16 + kW - 1)` tile with only 256 threads means most threads load one element and a subset load the halo, commonly via a strided second pass so the block covers more tile slots than it has threads. That loading phase has brief divergence and is the one place coalescing can slip, but it runs once per block and is dwarfed by the global traffic it eliminates.</span>

---

## <span style="font-size: 16px;">Naive vs Optimized</span>

<span style="font-size: 14px;">The naive kernel is one thread per output, each independently reading its `kH x kW` window from global memory and the filter from global memory. Correct, but it refetches every interior input about `kH * kW` times.</span>

<span style="font-size: 14px;">1. **Constant-memory filter**: place the `kH * kW` weights in constant memory so each warp broadcasts a weight in one cached access instead of a global load, removing all filter traffic.</span>

<span style="font-size: 14px;">2. **Shared-memory tile with halo**: each block loads its `(16 + kH - 1) x (16 + kW - 1)` input region into `__shared__` memory once, then every thread reads its window from shared memory. This cuts global input traffic from about `kH * kW` reads per element to one - the reuse factor approaches the filter area as the tile grows.</span>

<span style="font-size: 14px;">The tile dimension is itself a tuning knob: a larger tile improves the area-to-halo ratio and so the reuse, but consumes more shared memory and can cap occupancy. The systems discipline is to identify the 2D reuse first, then size the tile so the captured reuse and the surviving warp count are jointly maximized.</span>

---

## <span style="font-size: 16px;">Worked Example</span>

<span style="font-size: 14px;">Take a $4 \times 4$ input with a $3 \times 3$ filter, valid padding, stride 1. The output is $(4-3+1) \times (4-3+1) = 2 \times 2$. The four outputs anchor at $(0,0), (0,1), (1,0), (1,1)$.</span>

* <span style="font-size: 14px;">**Overlap:** output $(0,0)$ reads `input[0..2, 0..2]` and output $(0,1)$ reads `input[0..2, 1..3]`; they share the $3 \times 2 = 6$ overlapping columns - reuse along the horizontal axis.</span>
* <span style="font-size: 14px;">**Naive traffic:** four threads issue $4 \times 9 = 36$ input loads for an image of only 16 elements, so interior pixels are fetched up to four times.</span>

<span style="font-size: 14px;">A block-wide shared tile loads all 16 input elements once (the $2 \times 2$ output footprint plus a 2-row, 2-column halo, which here is the whole image) and serves all 36 window reads from shared memory, cutting global input traffic from 36 loads to 16.</span>

---

## <span style="font-size: 16px;">Pitfalls</span>

* <span style="font-size: 14px;">**Forgetting the halo.** Sizing the shared tile to only the block's output footprint omits the `kH-1` halo rows and `kW-1` halo columns the edge threads need, giving wrong results at the tile border.</span>
* <span style="font-size: 14px;">**Missing `__syncthreads()` after the tile load.** Reading a window before every lane has finished writing the shared tile is a race that yields nondeterministic output.</span>
* <span style="font-size: 14px;">**Uncoalesced tile loads.** Loading the tile column-major, or with each thread fetching its own scattered window directly, breaks coalescing and multiplies transactions; fill the tile row-major so lanes hit consecutive columns.</span>
* <span style="font-size: 14px;">**Out-of-bounds from the bounds check.** Omitting `if (row < outH && col < outW)` lets edge-block threads read and write past the valid output region.</span>

---