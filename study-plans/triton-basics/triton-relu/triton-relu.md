# <span style="font-size: 20px;">ReLU</span>

<span style="font-size: 14px;">ReLU clamps every negative input to zero and passes positives through unchanged. It is a **pure pointwise map** with no cross-lane dependency, no reduction, and no shared memory. The systems interest is in how the clamp is expressed: a single $\texttt{tl.maximum}$ over the tile produces branchless code, in contrast to the per-thread $\texttt{if}$ a naive CUDA author would write, which the GPU resolves by serializing diverged paths inside a warp.</span>

---

## <span style="font-size: 16px;">The Operation</span>

<span style="font-size: 14px;">For each input lane the kernel computes</span>

$$
\texttt{out}[i] = \max(x[i], 0), \quad 0 \le i < N
$$

<span style="font-size: 14px;">Both $x$ and $\texttt{out}$ are contiguous 1D fp32 tensors of length $N$, allocated in HBM. The launcher allocates $\texttt{out}$ and the kernel writes into it in place. There is exactly one operand, one output, and one threshold constant.</span>

---

## <span style="font-size: 16px;">Program Decomposition</span>

<span style="font-size: 14px;">The launch grid is $\lceil N / \texttt{BLOCK\_SIZE} \rceil$ **programs** in 1D. Each program owns one contiguous tile of $\texttt{BLOCK\_SIZE}$ lanes, addressed by $\texttt{offs} = \texttt{pid} \cdot \texttt{BLOCK\_SIZE} + \texttt{tl.arange}(0, \texttt{BLOCK\_SIZE})$. No program touches another program's output, no program needs to know what any other program loaded, and the kernel is embarrassingly parallel at the program level.</span>

<span style="font-size: 14px;">The parallel pattern is identical to vector addition: a 1D map across a 1D tensor. The only structural difference is one fewer input load, because ReLU is unary. Once the tile is in registers, the operation reduces to one lane-wise comparison-and-select, which on modern hardware is a single instruction (FMAX on NVIDIA, MAX.F32 on AMD).</span>

---

## <span style="font-size: 16px;">Tile Shape and Masking</span>

<span style="font-size: 14px;">The tile is fixed at compile time at $\texttt{BLOCK\_SIZE} = 1024$, declared $\texttt{tl.constexpr}$. The constexpr value lets the compiler allocate registers for a $1024$-lane fp32 tile, unroll the load-clamp-store sequence, and emit wide vector loads. Power-of-two block sizes let the compiler pick the widest legal vector width; a non-power-of-two value falls back to scalar handling on the trailing lanes.</span>

<span style="font-size: 14px;">The tail mask $\texttt{mask} = \texttt{offs} < N$ disables the lanes that overshoot when $N$ is not divisible by $1024$. ReLU has a quiet alignment property here: even an unmasked load that pulled garbage past the buffer would feed harmless arithmetic into the clamp, but the matching $\texttt{tl.store}$ would write that garbage past the output. The mask therefore protects the store; the read protection on $\texttt{tl.load}$ is the second-order benefit. Mask discipline is required on every load and store even when the arithmetic in between is innocuous.</span>

---

## <span style="font-size: 16px;">Memory Hierarchy and Reuse</span>

<span style="font-size: 14px;">Per output element the kernel moves $4$ bytes of input and $4$ bytes of output through HBM. Nothing is reused: the tile each program holds is consumed and thrown away, no other program will ask for those bytes, and the compiler never stages anything into SRAM because the kernel never asks twice for the same data. Operands live in registers from $\texttt{tl.load}$ through the clamp through $\texttt{tl.store}$ without ever touching shared memory.</span>

<span style="font-size: 14px;">The single memory-system property the kernel needs is **coalesced** HBM access. The lane offsets inside one tile are contiguous, so the compiler lowers $\texttt{tl.load}$ to a small number of wide transactions. A $\texttt{BLOCK\_SIZE} = 1024$ tile of fp32 is $4096$ bytes, which a modern accelerator serves as $32$ transactions of $128$ B each when the access is perfectly coalesced. Triton authors do not write coalescing rules by hand; the contiguous $\texttt{tl.arange}$ pattern carries the intent and the compiler picks the transaction width.</span>

<span style="font-size: 14px;">The L2 cache is incidental. ReLU has no inter-tile reuse, so any line the L2 captures during one program's load is never queried again. The L2's only role is to absorb the asynchronous latency between when the load issues and when the line returns from HBM, which the compiler is already hiding by interleaving warps inside the program. The kernel runs at HBM speed and not a hair faster.</span>

---

## <span style="font-size: 16px;">Memory-Bound vs Compute-Bound</span>

<span style="font-size: 14px;">Per output element the kernel reads $4$ bytes, performs one $\max$ operation, and writes $4$ bytes. The **arithmetic intensity** is</span>

$$
\frac{1 \text{ op}}{8 \text{ bytes}} \approx 0.125 \text{ ops/byte}
$$

<span style="font-size: 14px;">That sits at roughly the same memory-bound floor as vector add (counting the clamp as one op against the $\max$ unit, not the FMA pipeline). Peak fp32 throughput on a modern accelerator is in the tens of TFLOPs and HBM bandwidth is in the low TB/s, putting the roofline crossover around $10$ ops/byte. ReLU is at $0.125$, two orders of magnitude inside the memory-bound region. No amount of arithmetic cleverness can change the runtime; only effective bandwidth matters.</span>

<span style="font-size: 14px;">The practical consequence is that ReLU done as its own kernel is almost pure HBM bandwidth: a $1$M-element fp32 ReLU moves $8$ MB through HBM, which at $1$ TB/s of bandwidth is roughly $8$ microseconds, comparable to kernel launch overhead. That math is the whole reason ReLU is almost never run as its own kernel in production training stacks - it is fused into the preceding matmul or the following add, eliminating its HBM round-trip entirely. The standalone kernel is a teaching example; the production kernel is an epilogue.</span>

<span style="font-size: 14px;">The intensity numbers across foundations stack predictably: vector add at $0.083$ FLOPs/byte, FMA at $0.17$, ReLU at $0.125$, GELU at roughly $0.4\text{-}0.6$ once the $\texttt{erf}$ is counted, SiLU at roughly $0.4$. Every one of these kernels is firmly memory-bound and the relative runtimes track HBM traffic almost linearly. The exact ordering between them is a question of how many bytes per element each kernel moves, not how many arithmetic operations each kernel performs once the bytes are in registers. This is the recurring lesson of the foundations track and the reason every later kernel in the curriculum starts by counting bytes.</span>

---

## <span style="font-size: 16px;">Compiler-Handled vs Author-Handled</span>

<span style="font-size: 14px;">The author chooses the grid shape, the constexpr block size, the offset arithmetic, the mask predicate, and the form of the clamp ($\texttt{tl.maximum}(x, 0.0)$ rather than $\texttt{tl.where}(x > 0, x, 0.0)$, which is semantically identical but generates an intermediate boolean tile). The author's central choice in this kernel is to express the clamp as a tile-level $\max$ rather than as a per-lane $\texttt{if}$ statement, because the latter would force the compiler to emit divergent control flow.</span>

<span style="font-size: 14px;">This is the cleanest place in the Foundations track to contrast Triton with CUDA. A naive CUDA author might write $\texttt{if (x[i] > 0) out[i] = x[i]; else out[i] = 0;}$ inside the kernel, and the hardware would resolve this by **warp divergence**: the $32$ lanes of a warp issue both sides of the branch, masking off the lanes that did not take each path, and serializing the two paths in time. Even for a clamp where one side is trivial, the divergence machinery has overhead. The Triton author writes $\texttt{tl.maximum}(x, 0.0)$, the compiler lowers it to a single FMAX instruction per lane, and there is no divergence to manage because there is no branch in the lowered code at all. The whole tile passes through one straight-line sequence of instructions.</span>

<span style="font-size: 14px;">The compiler also handles the warp story exactly as it does for vector add. Inside one program, the $1024$-lane tile is internally sharded across $\texttt{num\_warps}$ (default $4$) groups of $256$ lanes each. While one warp is waiting on an HBM load to return, the others issue their FMAX in parallel, hiding the load latency. The author never writes about warps, never invokes $\texttt{\_\_syncthreads}$, and never declares shared memory, because the kernel has nothing for any of them to do.</span>

<span style="font-size: 14px;">The broader pattern this kernel demonstrates: Triton's tile-level operators are the language the compiler wants the author to speak in. Every $\texttt{tl.maximum}$, $\texttt{tl.minimum}$, $\texttt{tl.where}$, $\texttt{tl.exp}$, $\texttt{tl.sigmoid}$, and arithmetic operator is a tile-level primitive that lowers to one straight-line instruction per lane. Per-lane Python conditionals are not the same thing. The compiler treats tile-level operators as opaque vectorizable units and fuses them into long FMA-and-FMAX sequences; it treats per-lane conditionals as control flow it has to lower into masked predicate execution. The first kind composes; the second kind blocks composition. The author's job is to keep the kernel in the first category.</span>

<span style="font-size: 14px;">Concretely on this kernel, $\texttt{tl.maximum}$ lowers to the FMAX instruction on NVIDIA (or its bf16 / fp16 variants), which is a one-cycle operation issued per lane. The FMAX unit and the FMA unit share the same scoreboard, so issuing a clamp does not contend with the surrounding multiply-add traffic - useful when ReLU is fused into a matmul epilogue, because the clamp slots into the existing FMA stream without stalling it.</span>

---

## <span style="font-size: 16px;">Naive vs Optimized</span>

<span style="font-size: 14px;">The kernel above is already the optimized form for a standalone ReLU. The naive form in spirit is the CUDA-style per-thread $\texttt{if}$ over each element, which surfaces in Triton as $\texttt{tl.where}(x > 0, x, 0.0)$: correct, but it generates a boolean intermediate and a select instead of a single FMAX. The runtime difference is small (one extra register, one extra cycle per lane), but the Triton-idiomatic form is shorter and emits cleaner PTX.</span>

<span style="font-size: 14px;">The much larger optimization is fusion. ReLU is the canonical activation to fuse into the matmul whose output it consumes: the matmul epilogue writes $W x + b$ into registers, applies the clamp, and writes the final result to HBM in one pass, never materializing the pre-activation tensor. That single decision drops the HBM traffic for a transformer feedforward block by roughly a third, depending on what else is fused along with it. The standalone ReLU kernel is the building block; the fused matmul-with-ReLU epilogue is the lesson.</span>

<span style="font-size: 14px;">The HBM math for the fusion is direct. Standalone ReLU on a tensor of $M$ elements moves $8M$ bytes through HBM ($4M$ in, $4M$ out). If the producing kernel was a matmul that wrote its $M$-element output to HBM and the consuming kernel was a ReLU that read it back, the total inter-kernel traffic is $8M$ bytes that exist only to materialize the pre-activation. Fusing the clamp into the matmul epilogue removes those $8M$ bytes entirely: the matmul accumulator already holds the value in a register, the FMAX runs against the scalar zero, and the post-activation value is what gets stored. The kernel-launch overhead for the separate ReLU also vanishes. On a memory-bound stack this is the single largest lever the foundations track teaches.</span>

---

## <span style="font-size: 16px;">Worked Example</span>

<span style="font-size: 14px;">Take $N = 6$, $x = [-2, -1, 0, 1, 2, 3]$, $\texttt{BLOCK\_SIZE} = 4$. The launch grid is $\lceil 6 / 4 \rceil = 2$ programs.</span>

<span style="font-size: 14px;">**Program 0** ($\texttt{pid} = 0$): the offsets are $[0, 1, 2, 3]$, the mask is $[\texttt{T}, \texttt{T}, \texttt{T}, \texttt{T}]$. Loads $x[0..3] = [-2, -1, 0, 1]$. $\texttt{tl.maximum}$ against $0.0$ produces $[0, 0, 0, 1]$ in one FMAX per lane. The lane holding $x = 0$ writes $0$ (the comparison treats zero as non-negative, matching the IEEE max for $+0$). Stores into $\texttt{out}[0..3]$.</span>

<span style="font-size: 14px;">**Program 1** ($\texttt{pid} = 1$): the offsets are $[4, 5, 6, 7]$, the mask is $[\texttt{T}, \texttt{T}, \texttt{F}, \texttt{F}]$. Loads $x[4..5] = [2, 3]$ for the two live lanes; the masked lanes load whatever the $\texttt{other}$ value gives (zero by default) and the FMAX produces zero for them, which is harmless because the matching store mask discards the lane anyway. Writes $\texttt{out}[4] = 2$ and $\texttt{out}[5] = 3$. Slots $6$ and $7$ never exist and are never touched.</span>

<span style="font-size: 14px;">Both programs execute concurrently. Inside each program, the four lanes of the tile run as one straight-line sequence of instructions: one load, one FMAX, one store. There is no per-lane branch and no warp-level divergence resolution, because the source code never named a conditional in the first place.</span>

<span style="font-size: 14px;">Counting the HBM traffic for this $N = 6$ launch: program $0$ loads $4 \cdot 4 = 16$ bytes and stores $16$ bytes. Program $1$ loads $2 \cdot 4 = 8$ bytes of live input and stores $8$ bytes of live output. The whole kernel moves $48$ bytes for $6$ output elements, exactly $8$ bytes per element. The arithmetic budget is $6$ FMAX operations, one per live lane, plus two dead FMAX operations on the masked lanes of program $1$ that the store mask discards. The dead FMAX cost two cycles total; the HBM traffic took most of the wall-clock time.</span>

---

## <span style="font-size: 16px;">Pitfalls</span>

* <span style="font-size: 14px;">**Using a Python $\texttt{if}$ inside the kernel.** Writing $\texttt{if x > 0: ... else: ...}$ on a tile is either rejected by the Triton compiler or rewritten into something equivalent to $\texttt{tl.where}$, with no performance benefit. The idiomatic form is $\texttt{tl.maximum}(x, 0.0)$, which the compiler lowers to a single FMAX per lane.</span>
* <span style="font-size: 14px;">**Wrong scalar literal.** Writing $\texttt{tl.maximum}(x, 0)$ with an integer instead of $\texttt{0.0}$ can promote the tile dtype to int and silently break the output. The float literal is required when the input is fp32.</span>
* <span style="font-size: 14px;">**Forgetting the tail mask.** Hidden test sizes at $N = 257$ and $N = 1025$ overshoot $\texttt{BLOCK\_SIZE} = 1024$ in the last program by $767$ lanes. Without $\texttt{mask}$ on both the load and the store, the kernel writes past the output buffer and the test compares against scrambled bytes.</span>
* <span style="font-size: 14px;">**Block size not constexpr.** A runtime $\texttt{BLOCK\_SIZE}$ prevents the compiler from sizing registers, unrolling the load-FMAX-store sequence, and picking the widest legal vector load. The kernel will still compile but it loses most of its bandwidth headroom.</span>

---