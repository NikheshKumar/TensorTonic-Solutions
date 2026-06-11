# <span style="font-size: 20px;">GELU</span>

<span style="font-size: 14px;">GELU is the smooth, probabilistically motivated activation that has displaced ReLU in modern Transformer feedforward blocks. It is a **pointwise map** with a single transcendental call per lane, and the cleanest demonstration in the foundations track of why even a meaningful jump in arithmetic per element does not move a pointwise kernel off the memory-bound floor of the roofline.</span>

---

## <span style="font-size: 16px;">The Operation</span>

<span style="font-size: 14px;">The exact GELU form is</span>

$$
\texttt{out}[i] = \tfrac{1}{2}\, x[i]\, \Big(1 + \mathrm{erf}\!\Big(\tfrac{x[i]}{\sqrt{2}}\Big)\Big)
$$

<span style="font-size: 14px;">Both $x$ and $\texttt{out}$ are contiguous 1D fp32 tensors of length $N$. The $\mathrm{erf}$ call is the new beat compared to ReLU and FMA: it is the Gauss error function, a transcendental that lowers to a hardware special-function-unit op on NVIDIA targets (the CUDA libdevice $\texttt{\_\_nv\_erff}$ intrinsic). The constant $1/\sqrt{2} \approx 0.7071068$ is multiplied into the argument so the kernel never has to divide.</span>

---

## <span style="font-size: 16px;">Program Decomposition</span>

<span style="font-size: 14px;">The kernel reuses the 1D map skeleton: $\lceil N / \texttt{BLOCK\_SIZE} \rceil$ **programs** in a one-dimensional launch grid, each program identified by $\texttt{tl.program\_id(0)}$, each owning one contiguous tile of $\texttt{BLOCK\_SIZE}$ lanes. The tile of offsets is $\texttt{offs} = \texttt{pid} \cdot \texttt{BLOCK\_SIZE} + \texttt{tl.arange}(0, \texttt{BLOCK\_SIZE})$, and there is no inter-program communication of any kind.</span>

<span style="font-size: 14px;">The parallel pattern is identical to vector add and ReLU. What changes is the body of the kernel: instead of one $\max$ or one FMA, the program issues one multiply, one $\mathrm{erf}$ call, one add, one multiply, and one final multiply per lane. That is five arithmetic operations plus the $\mathrm{erf}$ instead of vector add's one. The interesting question is whether that jump is enough to pull the kernel toward compute-bound. The answer below is no, by a wide margin.</span>

---

## <span style="font-size: 16px;">Tile Shape and Masking</span>

<span style="font-size: 14px;">The kernel pins the block shape at compile time with $\texttt{BLOCK\_SIZE} = 1024$, declared $\texttt{tl.constexpr}$. The compiler uses the constexpr to size registers, unroll the body, and emit wide PTX vector loads and stores. The $\mathrm{erf}$ instruction is one of the more expensive lane-wise ops on the hardware, but it is still issued one lane at a time inside the tile, so the constexpr only matters for memory codegen, not for the transcendental itself.</span>

<span style="font-size: 14px;">The tail mask $\texttt{mask} = \texttt{offs} < N$ gates both $\texttt{tl.load}$ and $\texttt{tl.store}$ to the live lanes. There is a small subtlety with $\mathrm{erf}$ on masked lanes: even if the masked lane's loaded value is whatever $\texttt{other}$ provided (zero by default), the $\mathrm{erf}$ call still runs on the lane and burns its cycle. That cost is uniform across the tile and does not affect correctness because the matching store mask drops the lane's output. Mask discipline is the same as every other map kernel: every $\texttt{tl.load}$, every $\texttt{tl.store}$, on the same predicate.</span>

---

## <span style="font-size: 16px;">Memory Hierarchy and Reuse</span>

<span style="font-size: 14px;">Per output element the kernel moves exactly the same $4$ bytes in and $4$ bytes out as ReLU. Eight bytes of HBM traffic per element, zero reuse across tiles, zero reuse across programs. The kernel does not stage anything into SRAM and never asks the same byte twice. The compiler does not allocate shared memory for the kernel because there is nothing for shared memory to do; the tile lives in registers from the load through the $\mathrm{erf}$ call through the trailing multiplies through the store.</span>

<span style="font-size: 14px;">The contiguous lane offsets give the compiler a clean coalescing pattern. A $\texttt{BLOCK\_SIZE} = 1024$ tile of fp32 is $4096$ bytes, served by an accelerator as roughly $32$ transactions of $128$ B when the access is perfectly coalesced. The Triton author writes the contiguous $\texttt{tl.arange}$ offsets and the compiler picks the transaction width. A CUDA author writing the same kernel would have to align thread indices with element indices by hand and read the GPU vendor's coalescing rules to confirm; the Triton author gets coalescing for free as a consequence of expressing the load at the tile level.</span>

<span style="font-size: 14px;">The cost of the $\mathrm{erf}$ instruction is worth one paragraph on its own. The special-function unit (SFU) on a modern NVIDIA accelerator handles transcendentals like $\texttt{erf}$, $\texttt{exp}$, $\texttt{log}$, $\texttt{sin}$, and $\texttt{cos}$ at a throughput of roughly one operation per cycle per SFU, with a single SFU per warp. That is much slower per cycle than the FMA pipeline (which retires one FMA per lane per cycle across the whole warp), but it is still wildly fast relative to HBM latency, which is hundreds of cycles per load. The $\mathrm{erf}$ call adds maybe $10\text{-}30$ cycles of work per lane on top of the FMA stream, against an HBM round-trip of $400\text{-}600$ cycles. The transcendental therefore does not move the kernel off the memory-bound side of the roofline.</span>

<span style="font-size: 14px;">A practical way to picture the latency hiding: the compiler issues the $\texttt{tl.load}$ for one tile, then while that load is in flight it issues the $\texttt{tl.load}$ for the next program's tile (across warps inside the same program) and starts the $\mathrm{erf}$ chain on whatever data has already returned. The HBM round-trip and the SFU chain overlap. By the time the SFU has retired the $\mathrm{erf}$ for the first batch of lanes, the load for the next batch has often returned, and the kernel never actually stalls on HBM as long as there are enough warps in flight. This is the same latency-hiding mechanism that makes high-occupancy CUDA kernels fast, and Triton's compiler arranges it without the author writing a line about warps or occupancy.</span>

<span style="font-size: 14px;">The L2 cache is incidental, exactly as for every other map kernel. With no inter-tile reuse, no line is touched twice, and the L2's only role is to absorb the asynchronous latency between when the load issues and when the line returns from HBM. The kernel runs at HBM bandwidth, not L2 bandwidth. The one exception worth flagging: if GELU is run on a tensor small enough to fit entirely in L2 (a few MB on modern hardware), the second invocation reads from L2 rather than HBM and clocks faster than the steady-state HBM rate. That regime is not representative of the production case, where the activation tensor sits in HBM and is dwarfed by the parameter tensors that share the same cache budget.</span>

---

## <span style="font-size: 16px;">Memory-Bound vs Compute-Bound</span>

<span style="font-size: 14px;">Per output element the kernel performs one multiply ($x \cdot 0.7071068$), one $\mathrm{erf}$, one add ($1 + \mathrm{erf}(\cdot)$), one multiply ($x \cdot (1 + \mathrm{erf}(\cdot))$), and one multiply ($0.5 \cdot \cdot$). Counting the $\mathrm{erf}$ as roughly $5\text{-}10$ FLOP-equivalent operations (the SFU implements it as a small polynomial), the **arithmetic intensity** is</span>

$$
\frac{\approx 9 \text{ ops}}{8 \text{ bytes}} \approx 1.1 \text{ ops/byte}
$$

<span style="font-size: 14px;">That is roughly $13\times$ the intensity of vector add and $\approx 10\times$ the intensity of ReLU, but still an order of magnitude under the roofline crossover at around $10$ ops/byte. The kernel is **memory-bound** by a comfortable margin: the $\mathrm{erf}$ on every lane is cheap relative to the HBM round-trip for $8$ bytes per element, and the runtime is set by bandwidth, not by SFU throughput. A $1$M-element fp32 GELU moves $8$ MB through HBM, which at $1$ TB/s is $8$ microseconds, and the $\mathrm{erf}$ work runs in parallel with the next tile's load.</span>

<span style="font-size: 14px;">The exact-vs-approximate variant is worth one note. The tanh approximation $0.5 x (1 + \tanh(\sqrt{2/\pi} (x + 0.044715 x^3)))$ is sometimes faster on hardware without a fast $\mathrm{erf}$, because $\tanh$ can be expressed as one $\exp$ and arithmetic. On modern NVIDIA accelerators the SFU handles $\texttt{erf}$ and $\texttt{exp}$ at the same throughput, so the exact form is no slower than the approximation and slightly more accurate. PyTorch's default $\texttt{F.gelu}$ is the exact form for this reason; the tanh approximation survives mainly as a historical artifact from before $\texttt{erf}$ was fast on GPUs, and from a handful of model checkpoints that were trained against it and need bit-compatible inference.</span>

---

## <span style="font-size: 16px;">Compiler-Handled vs Author-Handled</span>

<span style="font-size: 14px;">The author chooses the grid, the constexpr block size, the offset arithmetic, the tail mask, and the form of the expression: $0.5 \cdot x \cdot (1 + \mathrm{erf}(x \cdot \texttt{inv\_sqrt2}))$ rather than $0.5 \cdot x \cdot (1 + \mathrm{erf}(x / \sqrt{2}))$. The second form is identical mathematically but introduces a reciprocal instruction per lane (DIV is materially more expensive than MUL on most hardware), so the author pre-multiplies the constant. Constant folding cannot rescue the second form because $\sqrt{2}$ is not a value the compiler will divide into the expression on its own.</span>

<span style="font-size: 14px;">The compiler lowers $\texttt{tl.math.erf}$ to the libdevice $\texttt{\_\_nv\_erff}$ intrinsic on NVIDIA targets. That intrinsic is a polynomial approximation of $\mathrm{erf}$ accurate to better than $1$ ulp, executed on the SFU. The compiler also fuses the surrounding chain of multiplies and adds into the FMA pipeline ($1 + \mathrm{erf}(\cdot)$ is an FMA against $1$ and the constant $0.5$; the trailing $x \cdot 0.5 \cdot (1 + \mathrm{erf}(\cdot))$ chain folds into two FMAs). The author writes the expression in mathematical form and the compiler picks which units retire it. None of the warp-level work, shared memory, or barrier insertion is the author's problem.</span>

<span style="font-size: 14px;">Inside one program the $1024$-lane tile is internally sharded across $\texttt{num\_warps}$ (default $4$) groups of $256$ lanes each. The SFU executes one $\mathrm{erf}$ per cycle per warp, so the four warps issue their $\mathrm{erf}$ calls in parallel across the four SFUs the SM exposes (or serialize across fewer SFUs on older hardware). The compiler picks the warp count; the author can override it through autotune if a profile shows the default is wrong.</span>

---

## <span style="font-size: 16px;">Naive vs Optimized</span>

<span style="font-size: 14px;">The kernel above is already the optimized standalone form. The naive variant is dividing rather than pre-multiplying by $1/\sqrt{2}$, which introduces a reciprocal per lane that the SFU pipeline retires more slowly than a MUL. A naive author might also compute $\mathrm{erf}(x) \cdot 0.5 + 0.5$ and then multiply by $x$, which is correct but loses one FMA opportunity compared to the canonical form. The numerical and performance differences are small per element but real.</span>

<span style="font-size: 14px;">The much larger optimization, as with ReLU, is fusion with the matmul whose output GELU consumes. The Transformer feedforward block is $\texttt{up}(x) \to \mathrm{GELU} \to \texttt{down}$, and the standard production form fuses $\mathrm{GELU}$ into the $\texttt{up}$ matmul epilogue. The matmul accumulator already holds the pre-activation in registers; running the $\mathrm{erf}$, the two multiplies, and the add on the accumulator before storing removes the intermediate HBM round-trip entirely. The arithmetic cost is unchanged; the HBM traffic for the activation drops to zero.</span>

---

## <span style="font-size: 16px;">Worked Example</span>

<span style="font-size: 14px;">Take $N = 4$, $\texttt{BLOCK\_SIZE} = 4$, $x = [-1, 0, 1, 2]$. One program covers the whole input.</span>

<span style="font-size: 14px;">The offsets are $[0, 1, 2, 3]$, mask $[\texttt{T}, \texttt{T}, \texttt{T}, \texttt{T}]$. The tile loads $x = [-1, 0, 1, 2]$. The multiply by $\texttt{inv\_sqrt2} \approx 0.7071068$ produces $[-0.7071, 0, 0.7071, 1.4142]$. The $\texttt{tl.math.erf}$ call returns approximately $[-0.6827, 0, 0.6827, 0.9545]$ - the well-known values of $\mathrm{erf}$ at $\pm 1/\sqrt{2}$ and $\sqrt{2}$. Adding $1$ gives $[0.3173, 1, 1.6827, 1.9545]$. Multiplying lane-wise by $x$ gives $[-0.3173, 0, 1.6827, 3.9090]$. Multiplying by $0.5$ gives the final $\texttt{out} = [-0.1587, 0, 0.8413, 1.9545]$.</span>

<span style="font-size: 14px;">Those values match $\texttt{torch.nn.functional.gelu}([-1, 0, 1, 2])$ to within fp32 precision. The lane at $x = 0$ produces exactly $0$ because both $x$ and $\mathrm{erf}(0)$ are zero, the lane at $x = -1$ produces a small negative value (the smooth tail GELU keeps but ReLU clips), and the lane at $x = 2$ produces nearly $x$ itself because $\mathrm{erf}(\sqrt{2})$ is already close to $1$. The whole tile passes through one straight-line sequence of instructions: one load, one MUL, one $\mathrm{erf}$, two FMAs, one MUL, one store.</span>

---

## <span style="font-size: 16px;">Pitfalls</span>

* <span style="font-size: 14px;">**Choosing the wrong variant.** The tanh approximation differs from the exact $\mathrm{erf}$ form by up to $10^{-3}$ near $|x| = 1$. The reference is $\texttt{torch.nn.functional.gelu}$ with default $\texttt{approximate='none'}$, which is the exact form. Picking $\texttt{approximate='tanh'}$ in the reference, or the tanh formula in the kernel, makes the tests pass or fail inconsistently.</span>
* <span style="font-size: 14px;">**Dividing by $\sqrt{2}$ instead of multiplying.** $\texttt{x / 1.4142}$ generates a reciprocal per lane, which the SFU retires more slowly than the MUL by $0.7071$. The numerical result is the same to within fp32 precision; the runtime difference is small but real. Pre-multiplying by the constant is the idiomatic form.</span>
* <span style="font-size: 14px;">**$\texttt{tl.math.erf}$ unavailable on old Triton.** On Triton versions before the $\texttt{math}$ namespace was stable, the call lives at $\texttt{from triton.language.extra import libdevice; libdevice.erf(...)}$. The newer $\texttt{tl.math.erf}$ is the supported path on current installs.</span>
* <span style="font-size: 14px;">**Forgetting the tail mask.** Hidden test sizes at $N = 257$ and $N = 1025$ overshoot the $1024$-lane block in the last program. Without $\texttt{mask}$ on both load and store, the kernel reads garbage past the input (which $\mathrm{erf}$ then maps to garbage smoothly) and writes that garbage past the output.</span>

---