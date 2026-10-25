# <span style="font-size: 20px;">3D Convolution</span>

<span style="font-size: 14px;">A 3D convolution slides a small volumetric filter through a 3D input, producing one output per voxel from a weighted sum over a cuboidal window. It is the canonical **stencil** in three dimensions: every output reads a `kD x kH x kW` neighbourhood, and adjacent outputs overlap in nearly all of the input they touch. That three-dimensional **reuse** is the systems story, and it is enormous, but capturing it pressures registers, shared memory, and **occupancy** far harder than the 1D or 2D cases.</span>

---

## <span style="font-size: 16px;">The Operation</span>

<span style="font-size: 14px;">For an input of shape $D \times H \times W$, a filter of shape $kD \times kH \times kW$, valid padding and stride 1, the output has shape $(D - kD + 1) \times (H - kH + 1) \times (W - kW + 1)$. Using cross-correlation (no flip), output voxel $(d, i, j)$ is:</span>

$$
\text{output}[d,i,j] = \sum_{a=0}^{kD-1} \sum_{b=0}^{kH-1} \sum_{c=0}^{kW-1} \text{input}[d+a, i+b, j+c] \cdot \text{kernel}[a,b,c]
$$

<span style="font-size: 14px;">All buffers are contiguous row-major arrays in device global memory, so voxel $(z, y, x)$ lives at linear offset `(z * H + y) * W + x`. The filter is small, read-only, and shared by every output. Valid padding keeps every window in bounds, so the read side needs no boundary handling.</span>

---

## <span style="font-size: 16px;">Parallelization Strategy</span>

<span style="font-size: 14px;">Outputs are mutually independent, so the decomposition is **one thread per output voxel** over a three-dimensional grid of 3D blocks. Each thread maps its block and lane coordinates to a depth, row, and column:</span>

$$
x = \text{blockIdx.x} \cdot \text{blockDim.x} + \text{threadIdx.x}, \quad y = \text{blockIdx.y} \cdot \text{blockDim.y} + \text{threadIdx.y}
$$

<span style="font-size: 14px;">with the depth coordinate built the same way from the `z` dimension. An 8x8x8 block is conventional: 512 threads is a multiple of the 32-lane **warp**, the cubic shape matches the cubic access pattern, and it fits the per-SM thread budget. The grid tiles the output with $\lceil \text{outW}/8 \rceil$, $\lceil \text{outH}/8 \rceil$, $\lceil \text{outD}/8 \rceil$ blocks, and a three-way bounds check guards the rounded-up edge blocks.</span>

<span style="font-size: 14px;">The fastest-varying coordinate is `threadIdx.x`, so the 32 lanes of a warp span consecutive columns within one $(d, i)$ row. Their input reads march across consecutive `x` addresses, which is what makes the inner-axis loads coalesce. With an 8-wide block edge, one warp of 32 lanes already spans four full `x`-rows, so warp membership crosses the row boundary - a detail that matters for both coalescing and how the halo load is partitioned.</span>

---

## <span style="font-size: 16px;">Memory Hierarchy and Access Pattern</span>

<span style="font-size: 14px;">A naive thread issues `kD * kH * kW` global loads of `input` plus the same number of filter loads, then one store. The reuse is the largest of the stencil family: neighbours along any of the three axes share an entire `kD x kH x kW` face minus one slice. Across an 8x8x8 block every interior voxel is fetched on the order of `kD * kH * kW` times from DRAM. The triple-nested accumulation is cheap arithmetic; the redundant volumetric traffic dominates.</span>

<span style="font-size: 14px;">The fix is again a **shared-memory tile with a halo**, now a 3D brick. An 8x8x8 block of outputs depends on an input region of `(8 + kD - 1) x (8 + kH - 1) x (8 + kW - 1)`, the output cube plus a halo on every face. The block cooperatively loads that brick into `__shared__` memory once, then every thread reads its window from shared memory, amortizing each global load across all overlapping output windows.</span>

<span style="font-size: 14px;">The small filter is read-only and identical per thread, so it belongs in **constant memory**, whose cache broadcasts one weight to a whole warp per access. Filling the brick is **coalesced** when consecutive lanes read consecutive `x` addresses of one row, so each warp-wide load hits contiguous global memory. Shared memory sits about an order of magnitude lower in latency than global and is user-managed per block.</span>

<span style="font-size: 14px;">The reuse scaling is what sets 3D apart. In 1D the halo is a 1D strip and reuse tops out near `kN`; in 2D it wraps the tile on four sides and reuse approaches `kH * kW`; in 3D the halo wraps the brick on all six faces and the available reuse approaches the full filter volume `kD * kH * kW`. The reuse grows fastest of the family, but so does the shared footprint needed to capture it, which is the central tension of this kernel.</span>

---

## Memory-bound or compute-bound?

<span style="font-size: 14px;">Per output the kernel does `kD * kH * kW` multiply-adds against the bytes it moves. A naive kernel moves about that many input loads per output, giving an **arithmetic intensity** near $0.5$ FLOP/byte, far below the **roofline** ridge point of tens of FLOPs per byte. For the small filters here, 3D convolution is **memory-bound**: redundant volumetric traffic sets the runtime.</span>

<span style="font-size: 14px;">Tiling raises intensity because each loaded voxel feeds up to `kD * kH * kW` window reads from shared memory, the highest reuse potential of the stencil family. The catch is that the 3D brick plus halo is large, and the triple-nested loop holds many live values, so register and shared-memory pressure can cap occupancy before the full reuse is realized. The optimization frontier is therefore reuse versus occupancy, not FLOP count.</span>

---

## <span style="font-size: 16px;">Hardware Utilization</span>

<span style="font-size: 14px;">The tiled kernel needs one `__syncthreads()` between filling the shared brick and reading windows, so every lane sees the complete brick and halo first. The shared brick costs `(8 + kD - 1) * (8 + kH - 1) * (8 + kW - 1)` floats per block, which grows cubically with block edge and the halo; even a modest filter can consume a large fraction of an SM's tens of KB, directly limiting how many blocks co-reside and therefore **occupancy**.</span>

<span style="font-size: 14px;">The triple-nested accumulation also keeps more state live per thread, raising register usage; high register pressure caps active warps the same way. Because occupancy is what hides the hundreds-of-cycles DRAM latency of the brick load, 3D convolution is the case where capturing all the reuse and keeping enough warps resident are in direct tension. The compute loop itself is uniform, so there is no **warp divergence** outside the bounds check and halo load.</span>

---

## <span style="font-size: 16px;">Naive vs Optimized</span>

<span style="font-size: 14px;">The naive kernel is one thread per output voxel, each independently reading its `kD x kH x kW` window and the filter from global memory. Correct, but it refetches every interior voxel on the order of `kD * kH * kW` times.</span>

<span style="font-size: 14px;">1. **Constant-memory filter**: place the weights in constant memory so each warp broadcasts a weight in one cached access instead of a global load, removing all filter traffic.</span>

<span style="font-size: 14px;">2. **Shared-memory brick with halo**: each block loads its `(8 + kD - 1) x (8 + kH - 1) x (8 + kW - 1)` input region into `__shared__` memory once, then every thread reads its window from shared memory, cutting global input traffic from about `kD * kH * kW` reads per voxel toward one - bounded in practice by how large a brick fits without crushing occupancy.</span>

<span style="font-size: 14px;">3. **2.5D tiling**: when the full brick is too large for shared memory, tile only a 2D plane and stream along the depth axis, keeping a sliding window of `kD` planes resident. This trades some reuse for a much smaller footprint and is the standard escape hatch when a true 3D brick caps occupancy too hard.</span>

---

## <span style="font-size: 16px;">Worked Example</span>

<span style="font-size: 14px;">Take a $3 \times 3 \times 3$ input with a $2 \times 2 \times 2$ filter, valid padding, stride 1. The output is $2 \times 2 \times 2 = 8$ voxels, each summing a $2 \times 2 \times 2 = 8$-element window.</span>

* <span style="font-size: 14px;">**Overlap:** output $(0,0,0)$ reads `input[0..1, 0..1, 0..1]` and its `x`-neighbour $(0,0,1)$ reads `input[0..1, 0..1, 1..2]`; they share a $2 \times 2 = 4$-element face along the depth-row plane.</span>
* <span style="font-size: 14px;">**Naive traffic:** eight threads issue $8 \times 8 = 64$ input loads for a volume of only 27 voxels, so interior voxels are fetched several times each.</span>

<span style="font-size: 14px;">A block-wide shared brick loads all 27 input voxels once (here the whole volume) and serves all 64 window reads from shared memory, cutting global input traffic from 64 loads to 27, a reuse factor of about $2.4$. With larger volumes and the same tiny filter the factor climbs toward the filter volume $8$, which is exactly the leverage a brick is meant to extract.</span>

---

## <span style="font-size: 16px;">Pitfalls</span>

* <span style="font-size: 14px;">**Shared-memory brick crushing occupancy.** A `(8 + kD - 1)^3`-scale brick can exhaust an SM's shared memory, leaving too few resident warps to hide the brick-load latency; shrink the block or tile a 2D plane and stream the depth axis.</span>
* <span style="font-size: 14px;">**Forgetting the 3D halo.** Sizing the brick to only the output cube omits the halo on the depth, row, or column faces the edge threads need, giving wrong results at the brick border.</span>
* <span style="font-size: 14px;">**Missing `__syncthreads()` after the brick load.** Reading a window before every lane finishes writing the shared brick is a race that yields nondeterministic output.</span>
* <span style="font-size: 14px;">**Integer overflow in index math.** The linearization `(z * H + y) * W + x` can overflow 32-bit `int` for large volumes; use `size_t` for the offset.</span>

---