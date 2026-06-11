# <span style="font-size: 20px;">SiLU</span>

<span style="font-size: 14px;">SiLU (also called Swish) is the activation that drives LLaMA, Mistral, and most modern Transformer feedforward gates. It is a **pointwise map** with one transcendental call per lane, structurally identical to GELU but cheaper: one $\texttt{tl.exp}$ instead of one $\mathrm{erf}$, folded into the algebraic identity $x \cdot \sigma(x) = x / (1 + e^{-x})$ so the kernel can express the whole activation in a single fused line.</span>

---

## <span style="font-size: 16px;">The Operation</span>

<span style="font-size: 14px;">For each input lane the kernel computes</span>

$$
\texttt{out}[i] = x[i] \cdot \sigma(x[i]) = \frac{x[i]}{1 + e^{-x[i]}}
$$

<span style="font-size: 14px;">Both $x$ and $\texttt{out}$ are contiguous 1D fp32 tensors of length $N$, allocated in HBM. The identity on the right hand side folds the sigmoid and the trailing multiply into a single rational expression, avoiding the cost of materializing $\sigma(x)$ as an intermediate.</span>

---

## <span style="font-size: 16px;">Program Decomposition</span>

<span style="font-size: 14px;">The launch grid is $\lceil N / \texttt{BLOCK\_SIZE} \rceil$ **programs** in 1D. Each program is identified by $\texttt{tl.program\_id(0)}$ and owns one contiguous tile of $\texttt{BLOCK\_SIZE}$ lanes. The offsets are $\texttt{offs} = \texttt{pid} \cdot \texttt{BLOCK\_SIZE} + \texttt{tl.arange}(0, \texttt{BLOCK\_SIZE})$ and the access pattern is unit-strided, exactly as for vector add, ReLU, and GELU.</span>

<span style="font-size: 14px;">The parallel pattern is a map. No reduction, no cross-program communication, no atomic. The body of the kernel does one $\texttt{tl.exp}(-x)$, one add ($1 + \cdot$), one division ($x / \cdot$), and one store. Five operations counting the load and store, three of them arithmetic, one of them a transcendental.</span>

---

## <span style="font-size: 16px;">Tile Shape and Masking</span>

<span style="font-size: 14px;">$\texttt{BLOCK\_SIZE} = 1024$ is declared $\texttt{tl.constexpr}$. The compiler uses the value to size registers for a $1024$-lane fp32 tile, unroll the load-exp-add-div-store sequence, and emit wide PTX vector loads and stores. Power-of-two block sizes give the compiler clean vectorization; a non-power-of-two value falls back to scalar handling on the trailing partial vector.</span>

<span style="font-size: 14px;">The tail mask $\texttt{mask} = \texttt{offs} < N$ gates every $\texttt{tl.load}$ and every $\texttt{tl.store}$ to live lanes. SiLU has a small numerical subtlety on the masked lanes: $\texttt{tl.exp}(-x)$ with $x = 0$ (the default $\texttt{other}$ value) returns $1$, the add gives $2$, the division gives $0/2 = 0$ - all finite. So a missing mask would still produce finite values on the masked lanes, but the store would then write those values past the output buffer. Mask discipline protects the store, exactly as in ReLU.</span>

---

## <span style="font-size: 16px;">Memory Hierarchy and Reuse</span>

<span style="font-size: 14px;">Per output element the kernel moves $4$ bytes of input and $4$ bytes of output through HBM, $8$ bytes total. Zero reuse across tiles, zero reuse across programs. The compiler does not stage anything into SRAM and does not allocate shared memory, because the kernel has nothing to share. The tile lives in registers from load through exp through divide through store.</span>

<span style="font-size: 14px;">Contiguous lane offsets give the compiler a coalesced access pattern. A $1024$-lane fp32 tile is $4096$ bytes per tensor, served as roughly $32$ transactions of $128$ B each on a modern accelerator when the access is perfectly coalesced. The L2 cache is incidental: no inter-tile reuse, no line touched twice, and the kernel runs at HBM bandwidth.</span>

<span style="font-size: 14px;">The cost of $\texttt{tl.exp}$ is worth pinning. On NVIDIA the call lowers to the libdevice $\texttt{\_\_expf}$ intrinsic, which is a hardware-fast operation on the special-function unit (SFU): roughly one $\texttt{exp}$ per cycle per SFU per warp. Combined with the division (the only DIV in the kernel), the arithmetic cost per element is on the order of $20\text{-}40$ cycles, against an HBM round-trip of $400\text{-}600$ cycles per cache line. The transcendental and the divide are entirely hidden behind the load latency as long as there are enough warps in flight to keep the SFU busy.</span>

<span style="font-size: 14px;">A useful comparison: in CUDA, an author worrying about the cost of $\exp$ would either accept the libdevice call, replace it with $\texttt{\_\_expf}$ (the fast intrinsic) explicitly, or hand-write a polynomial approximation. Triton's $\texttt{tl.exp}$ already lowers to $\texttt{\_\_expf}$, so the fast path is the default; no extra annotation is required to get hardware-speed transcendentals. The author writes $\texttt{tl.exp}$ and the compiler picks the intrinsic.</span>

---

## <span style="font-size: 16px;">Memory-Bound vs Compute-Bound</span>

<span style="font-size: 14px;">Per output element the kernel performs one negation, one $\exp$, one add, one division, and one multiply (the $x \cdot \sigma(x)$ fused into the rational form). Counting $\exp$ as roughly $4\text{-}5$ FLOP-equivalent operations on the SFU, the **arithmetic intensity** is</span>

$$
\frac{\approx 7 \text{ ops}}{8 \text{ bytes}} \approx 0.9 \text{ ops/byte}
$$

<span style="font-size: 14px;">That is about an order of magnitude higher than vector add ($0.083$) and about the same as GELU minus the cost difference between $\exp$ and $\mathrm{erf}$ (SiLU is slightly cheaper). The roofline crossover sits around $10$ ops/byte on a modern accelerator, so SiLU is still **memory-bound** by an order of magnitude. The single $\exp$ does not pull the kernel toward compute-bound; the runtime is set by HBM bandwidth, not by SFU throughput.</span>

<span style="font-size: 14px;">A useful ground-truth: a $1$M-element fp32 SiLU moves $8$ MB through HBM, which at $1$ TB/s of bandwidth is roughly $8$ microseconds. The arithmetic budget for the same tensor is $\approx 7$M operations, which at the SFU rate of one $\exp$ per cycle per warp across enough warps to fill the SMs is on the order of $1\text{-}2$ microseconds. The HBM time dominates the SFU time by $4\text{-}8\times$, which is exactly the asymptotic statement of memory-bound.</span>

<span style="font-size: 14px;">Sized against the foundations: ReLU at $0.125$ ops/byte, vector add at $0.083$, FMA at $0.17$, SiLU at $0.9$, GELU at $1.1$. SiLU is the second-highest intensity activation in the section, still an order of magnitude under the roofline crossover. Every kernel in this list runs at HBM bandwidth in practice, and the ordering of their wall-clock runtimes tracks bytes-per-element rather than ops-per-element.</span>

---

## <span style="font-size: 16px;">Compiler-Handled vs Author-Handled</span>

<span style="font-size: 14px;">The author chooses the grid, the constexpr block size, the offset arithmetic, the tail mask, and the algebraic form: $x / (1 + \texttt{tl.exp}(-x))$ rather than $x \cdot \texttt{tl.sigmoid}(x)$ or the explicit two-line form $\texttt{s} = 1.0 / (1.0 + \texttt{tl.exp}(-x))$ followed by $\texttt{out} = x \cdot s$. The fused rational form gives the compiler the cleanest expression to lower into one FMA-and-DIV sequence per lane; the two-line form introduces one extra temporary register and one extra multiply that the compiler may or may not eliminate.</span>

<span style="font-size: 14px;">The compiler lowers $\texttt{tl.exp}$ to the $\texttt{\_\_expf}$ intrinsic, the $1 + e^{-x}$ to one FMA against the constant $1$, and the trailing $x / (\cdot)$ to one DIV instruction per lane. The whole body is a single straight-line sequence in PTX: negate, exp, FMA, DIV, store. The compiler also handles warp sharding ($\texttt{num\_warps}$ default $4$ inside one program), latency hiding (one warp issues exp while another warp's load is in flight), and vector width selection for the load and store. None of that is the author's problem.</span>

<span style="font-size: 14px;">There is a small numerical question worth flagging on the boundary between author and compiler. The fused rational $x / (1 + e^{-x})$ is exact for moderate $|x|$ but begins to differ from $x \cdot \sigma(x)$ near the extremes of fp32 precision: for very large negative $x$, $e^{-x}$ overflows and the division returns $0$ (correct in the limit); for very large positive $x$, $e^{-x}$ underflows to $0$ and the division returns $x$ (also correct in the limit). The constraint $|x| \le 50$ in the test inputs keeps the kernel away from both edges, so no stable rewrite is needed. The author still chooses the rational form deliberately because it has cleaner edge behavior than the explicit sigmoid in fp16.</span>

---

## <span style="font-size: 16px;">Naive vs Optimized</span>

<span style="font-size: 14px;">The kernel above is the optimized standalone form. A naive author might compute the sigmoid explicitly: $\texttt{s} = \texttt{tl.sigmoid}(x)$ followed by $\texttt{out} = x \cdot s$. That is correct and reads cleanly, but it introduces one extra named tile in registers and one extra multiply that the compiler has to chase down. The rational form $x / (1 + \texttt{tl.exp}(-x))$ produces the same PTX with one fewer named intermediate.</span>

<span style="font-size: 14px;">The dominant optimization, as with ReLU and GELU, is fusion. SiLU is the activation inside the SwiGLU feedforward of LLaMA and Mistral: $\texttt{down}(\mathrm{SiLU}(\texttt{gate}(x)) \cdot \texttt{up}(x))$. In production the SiLU runs inside the matmul epilogue for $\texttt{gate}$, the multiply against $\texttt{up}(x)$ folds into the next matmul's load, and the standalone SiLU kernel never appears. The standalone kernel is a teaching example and a building block for the harness; the production kernel is an epilogue.</span>

<span style="font-size: 14px;">A second-order optimization is autotuning $\texttt{BLOCK\_SIZE}$ against the input length. For $N$ in the hundreds of millions, larger blocks ($2048$, $4096$) cut the program count and let the compiler emit wider vector loads; for $N$ in the tens of thousands, smaller blocks ($256$, $512$) raise the program count and help saturate the device. The runtime difference is in the single-digit-percent range because the kernel is already running near peak HBM bandwidth, and an autotune sweep is not justified for a kernel this small. The lesson is that the block size is a knob; the curriculum exercises it later on matmul and softmax where the knob actually moves.</span>

---

## <span style="font-size: 16px;">Worked Example</span>

<span style="font-size: 14px;">Take $N = 5$, $\texttt{BLOCK\_SIZE} = 4$, $x = [-2, -1, 0, 1, 2]$. The launch grid is $\lceil 5 / 4 \rceil = 2$ programs.</span>

<span style="font-size: 14px;">**Program 0** ($\texttt{pid} = 0$): offsets $[0, 1, 2, 3]$, mask $[\texttt{T}, \texttt{T}, \texttt{T}, \texttt{T}]$. Loads $x = [-2, -1, 0, 1]$. The negation gives $[2, 1, 0, -1]$. $\texttt{tl.exp}$ gives $[e^{2}, e^{1}, e^{0}, e^{-1}] \approx [7.389, 2.718, 1, 0.368]$. Adding $1$ gives $[8.389, 3.718, 2, 1.368]$. Dividing $x$ by these denominators lane-wise gives $[-2/8.389, -1/3.718, 0/2, 1/1.368] \approx [-0.2384, -0.2689, 0, 0.7311]$. Stores into $\texttt{out}[0..3]$.</span>

<span style="font-size: 14px;">**Program 1** ($\texttt{pid} = 1$): offsets $[4, 5, 6, 7]$, mask $[\texttt{T}, \texttt{F}, \texttt{F}, \texttt{F}]$. Loads $x[4] = 2$ for the one live lane; the masked lanes load whatever $\texttt{other}$ gave (zero by default) and the rational arithmetic produces zero for them, which is harmless because the matching store mask drops them. The live lane computes $-x = -2$, $\exp(-2) \approx 0.1353$, denominator $\approx 1.1353$, output $\approx 2 / 1.1353 = 1.7616$. Writes $\texttt{out}[4] = 1.7616$ and leaves slots $5\text{-}7$ untouched (they do not exist in the output buffer).</span>

<span style="font-size: 14px;">The full result $\texttt{out} \approx [-0.2384, -0.2689, 0, 0.7311, 1.7616]$ matches $\texttt{torch.nn.functional.silu}([-2, -1, 0, 1, 2])$ to within fp32 precision. The lane at $x = 0$ produces exactly zero because the numerator is zero. The lane at $x = -1$ produces a small negative value (SiLU has a smooth negative tail that bottoms out around $x \approx -1.28$ at $\approx -0.278$, which is where the gradient is largest in magnitude), and the lane at $x = 2$ produces $\approx 88\%$ of $x$ because the sigmoid has saturated near $1$. The whole tile passes through one straight-line PTX sequence: load, negate, exp, FMA-with-$1$, DIV, store.</span>

<span style="font-size: 14px;">Counting the HBM traffic: program $0$ loads $4 \cdot 4 = 16$ bytes and stores $16$ bytes for a total of $32$ bytes; program $1$ loads $4$ bytes (one live lane) and stores $4$ bytes for a total of $8$ bytes. The whole kernel moves $40$ bytes for $5$ output elements, exactly $8$ bytes per element. The SFU budget is $5$ $\exp$ calls (one per output lane plus the dead lanes that share the same SFU cycle), which the SFU retires in roughly $5$ cycles per warp - negligible against the HBM round-trip for the load.</span>

<span style="font-size: 14px;">One more concrete comparison: if SiLU were run as the two-line explicit form ($\texttt{s = 1.0 / (1.0 + tl.exp(-x))}$ followed by $\texttt{out = x \cdot s}$), the arithmetic budget would gain one extra MUL per lane and the compiler might or might not eliminate the named $\texttt{s}$ tile. The HBM traffic is unchanged at $8$ bytes per element because $\texttt{s}$ never leaves registers. The runtime difference is in the noise; the readability difference is the point. The Triton-idiomatic form is the rational expression on one line.</span>

---

## <span style="font-size: 16px;">Pitfalls</span>

* <span style="font-size: 14px;">**Sign error on the exponent.** Writing $\texttt{tl.exp}(x)$ instead of $\texttt{tl.exp}(-x)$ produces $x / (1 + e^{x})$, which is not SiLU. The lane at $x = 1$ would output $\approx 0.269$ instead of $\approx 0.731$. The error is silent unless the test exercises both positive and negative inputs, which the foundation tests do.</span>
* <span style="font-size: 14px;">**Computing the sigmoid explicitly.** $\texttt{s = tl.sigmoid(x); out = x \cdot s}$ is correct but introduces one extra named intermediate and may emit one extra multiply depending on the compiler's pass ordering. The fused rational form is shorter and reliably lowers to the minimum PTX.</span>
* <span style="font-size: 14px;">**Forgetting the tail mask.** Hidden test sizes at $N = 257$ and $N = 1025$ overshoot the $1024$-lane block in the last program. Without $\texttt{mask}$ on both the load and the store, the kernel reads garbage past the input (which the rational form happily evaluates) and writes that garbage past the output buffer.</span>
* <span style="font-size: 14px;">**Treating SiLU as compute-bound.** The kernel has the highest arithmetic intensity of the foundation activations ($\approx 0.9$ ops/byte) but is still firmly memory-bound. Adding more arithmetic per element is free up to a point; the only optimization that moves the runtime is removing HBM round-trips by fusing with the matmul that produces $x$ or the multiply that consumes $\texttt{out}$.</span>

---