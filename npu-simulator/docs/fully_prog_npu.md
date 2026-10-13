# Fully Programmable NPU — Instruction Set Reference

A complete, conceptual instruction set for a **fully programmable NPU**: an embedded control core plus a tensor/vector compute engine that **runs arbitrary kernels** rather than a fixed menu of operations.

The defining property of a *fully programmable* NPU is that the compute engine itself fetches and executes its own instruction stream — with general vector primitives, per-lane predication, engine-side control flow, and a free register/memory model. This lets software express operations the hardware designer never anticipated (e.g., a brand-new activation), which a fixed-function accelerator cannot.

The ISA has three layers:

- **Layer 1 — Control core ISA** (Sections 1–3): scalar instructions for host-side orchestration, loops, addresses, launching kernels. RISC-V-style.
- **Layer 2 — Tensor extension** (Sections 4–8): high-throughput matrix engine for the common dense-linear-algebra fast path.
- **Layer 3 — Programmable vector core** (Sections 9–13): the general-purpose engine ISA that makes the NPU *fully programmable* — primitives, masking, engine control flow, and the kernel model.

Mnemonics and encodings are illustrative (a teaching reference), not tied to any vendor.

---

## 1. Control Core — Integer / Arithmetic

Scalar instructions for control flow, address generation, loop counters.

| Mnemonic | Operands | Purpose |
|----------|----------|---------|
| `ADD`  | rd, rs1, rs2 | rd = rs1 + rs2 |
| `SUB`  | rd, rs1, rs2 | rd = rs1 − rs2 |
| `ADDI` | rd, rs1, imm | rd = rs1 + immediate |
| `MUL`  | rd, rs1, rs2 | rd = rs1 × rs2 (index/stride math) |
| `AND` / `OR` / `XOR` | rd, rs1, rs2 | Bitwise logic |
| `SLL` / `SRL` | rd, rs1, rs2 | Shift left / right logical |
| `SLT`  | rd, rs1, rs2 | Set if less than |

---

## 2. Control Core — Memory (Scalar)

| Mnemonic | Operands | Purpose |
|----------|----------|---------|
| `LW` / `LH` / `LB` | rd, imm(rs1) | Load word / halfword / byte |
| `SW` / `SH` / `SB` | rs2, imm(rs1) | Store word / halfword / byte |

---

## 3. Control Core — Control Flow & Kernel Launch

| Mnemonic | Operands | Purpose |
|----------|----------|---------|
| `BEQ` / `BNE` | rs1, rs2, label | Branch if equal / not equal |
| `BLT` / `BGE` | rs1, rs2, label | Branch if less / greater-equal |
| `JAL`  | rd, label | Jump and link (call) |
| `JALR` | rd, rs1, imm | Jump register (return / indirect) |
| `KLAUNCH` | kaddr, grid, args | **Launch a kernel** on the engine (program address, iteration grid, arg pointer) |
| `KWAIT` | — | Wait for launched kernels to complete |

---

## 4. Tensor Engine — Configuration

Fast-path configuration for dense matrix ops.

| Mnemonic | Operands | Purpose |
|----------|----------|---------|
| `NPU.CFG.SHAPE` | M, K, N | Set tile dimensions |
| `NPU.CFG.DTYPE` | in_type, acc_type | Input precision + accumulator precision |
| `NPU.CFG.STRIDE`| rs1, rs2 | Row/column strides for tile addressing |
| `NPU.CFG.MODE`  | imm | Engine mode (matmul / conv mapping) |
| `NPU.CFG.RND`   | imm | Rounding / saturation policy |

---

## 5. Tensor Engine — Compute (Dense Fast Path)

One instruction performs a whole tile of multiply-accumulates. Kept as a hardwired fast path even on a programmable NPU, because dense matmul is the dominant workload and a fixed unit is far more efficient than building it from vector primitives.

| Mnemonic | Operands | Purpose |
|----------|----------|---------|
| `NPU.MATMUL` | dt, a, b | dt = a × b (tile, overwrite) |
| `NPU.MMA`    | dt, a, b | dt = dt + a × b (tiled reduction over K) |
| `NPU.DOT`    | dt, a, b | Vector dot product |
| `NPU.CONV`   | dt, a, w | 2-D convolution tile |
| `NPU.ACC.CLR`| dt | Clear accumulator tile |

---

## 6. Tensor Engine — Data Movement

| Mnemonic | Operands | Purpose |
|----------|----------|---------|
| `NPU.LOAD`   | treg, addr | Load tile into a tile register |
| `NPU.STORE`  | treg, addr | Store tile register to memory |
| `NPU.LOAD.W` | wreg, addr | Load weights into the array |
| `NPU.DMA.IN` | dst, src, size | Bulk DMA: main memory → scratchpad |
| `NPU.DMA.OUT`| dst, src, size | Bulk DMA: scratchpad → main memory |
| `NPU.BCAST`  | treg, rs1 | Broadcast scalar across a tile |

---

## 7. Tensor Engine — Activation (Hardwired Common Cases)

Optional accelerated versions of the most common activations. On a fully programmable NPU these are *conveniences*; anything not listed is built from Section 10 vector primitives.

| Mnemonic | Operands | Purpose |
|----------|----------|---------|
| `NPU.RELU`    | dt, src | max(0, x) |
| `NPU.QUANT`   | dt, src, scale | int32 accumulator → int8 |
| `NPU.DEQUANT` | dt, src, scale | int → float |
| `NPU.SCALE`   | dt, src, rs1 | Multiply tile by scalar |
| `NPU.POOL`    | dt, src | Pooling reduction |

---

## 8. Tensor Engine — Synchronization

| Mnemonic | Operands | Purpose |
|----------|----------|---------|
| `NPU.FENCE`  | — | Order NPU operations |
| `NPU.WAIT`   | — | Stall control core until engine drains |
| `NPU.BARRIER`| — | Barrier between data movement and compute |
| `NPU.STATUS` | rd | Read engine status into scalar register |

---

# Programmable Vector Core

**Sections 9–13 are what make the NPU fully programmable.** The engine has its own program counter, vector register file, and instruction stream. Kernels written against these primitives can implement arbitrary tensor algorithms — including operations not anticipated by the hardware designer.

## 9. Vector Core — Registers & Memory Model

| Resource | Description |
|----------|-------------|
| `v0`–`v31` | Vector registers (each holds a full vector of lanes) |
| `m0`–`m7`  | Mask/predicate registers (1 bit per lane) |
| `s0`–`s31` | Scalar registers inside the engine (loop counters, addresses) |
| `VL` | Vector-length register — number of active lanes (enables length-agnostic kernels) |
| Scratchpad | Engine-local addressable memory the kernel reads/writes freely |

| Mnemonic | Operands | Purpose |
|----------|----------|---------|
| `VSETVL` | rd, rs1 | Set active vector length from a requested element count |
| `VLOAD`  | vd, addr | Load vector from scratchpad/memory |
| `VSTORE` | vs, addr | Store vector to scratchpad/memory |
| `VGATHER`| vd, base, vidx | Gather: load lanes from data-dependent addresses |
| `VSCATTER`| vs, base, vidx | Scatter: store lanes to data-dependent addresses |
| `VMOV`   | vd, vs | Copy vector register |
| `VBCAST` | vd, rs1 | Broadcast scalar into every lane |

---

## 10. Vector Core — Compute Primitives

The general-purpose math from which arbitrary operations are composed.

| Mnemonic | Operands | Purpose |
|----------|----------|---------|
| `VADD` / `VSUB` | vd, vs1, vs2 | Elementwise add / subtract |
| `VMUL` / `VDIV` | vd, vs1, vs2 | Elementwise multiply / divide |
| `VMAX` / `VMIN` | vd, vs1, vs2 | Elementwise max / min (ReLU = `VMAX(x,0)`) |
| `VFMA` | vd, vs1, vs2 | Fused multiply-add: vd = vd + vs1×vs2 |
| `VCMP` | md, vs1, vs2 | Compare lanes → mask register |
| `VSEL` | vd, vs1, vs2, m | Per-lane select by mask (branchless conditionals) |
| `VEXP` / `VLOG` | vd, vs | Exp / log (build softmax, GELU, layernorm) |
| `VRSQRT` / `VSQRT` | vd, vs | (Reciprocal) square root (normalization) |
| `VRECIP` | vd, vs | Reciprocal |
| `VCVT` | vd, vs | Convert dtype (int8 ↔ fp16 ↔ fp32) |
| `VABS` / `VNEG` | vd, vs | Absolute value / negate |

---

## 11. Vector Core — Reductions

| Mnemonic | Operands | Purpose |
|----------|----------|---------|
| `VREDSUM` | rd, vs | Sum all lanes → scalar |
| `VREDMAX` | rd, vs | Max across lanes (e.g., softmax stability) |
| `VREDMIN` | rd, vs | Min across lanes |
| `VREDAND` / `VREDOR` | rd, vs | Logical reduction |
| `VSCAN`   | vd, vs | Prefix-sum (cumulative) scan |

---

## 12. Vector Core — Predication & Control Flow (Engine-Side)

What lets the engine run *real algorithms* with data-dependent behavior, not just one fixed blast of math.

| Mnemonic | Operands | Purpose |
|----------|----------|---------|
| `VMASK.SET`  | md, vs | Build a predicate mask from a condition |
| `VMASK.AND` / `VMASK.OR` | md, m1, m2 | Combine masks |
| `VMASK.NOT`  | md, m1 | Invert mask |
| `@m <op>`    | (prefix) | Predicated execution — op applies only to lanes where mask = 1 |
| `BR`   | label | Engine branch (unconditional) |
| `BRZ` / `BRNZ` | rs, label | Branch if scalar zero / non-zero |
| `LOOP` | rs, label | Hardware loop with trip count in scalar reg |
| `KEXIT`| — | End kernel; signal completion to host |

---

## 13. Vector Core — Kernel Program Model

A *kernel* is a small program loaded into engine instruction memory and launched by the host with `KLAUNCH`. The engine runs one kernel instance per grid point (SIMT-style), each with its own lane ID, enabling data-parallel execution over large tensors.

| Concept | Mechanism |
|---------|-----------|
| Kernel entry | Program address passed to `KLAUNCH` |
| Iteration space | `grid` argument → engine spawns instances over it |
| Per-instance ID | Readable into a scalar reg (`s`) for addressing |
| Arguments | Pointer block (`args`) holding tensor base addresses, shapes, scales |
| Completion | `KEXIT` per instance; host observes via `KWAIT` |
| Shared memory | Scratchpad regions visible across instances in a group |

---

## 14. Worked Example — Softmax (a kernel you *write*, not a built-in)

Softmax isn't in the fixed op menu — on a fully programmable NPU you build it from primitives. This is the test of full programmability: an operation the hardware never hardwired.

```
; kernel softmax(x[N] -> y[N]), launched per row
softmax:
  VSETVL   t0, N            ; length-agnostic over the row
  VLOAD    v0, x            ; v0 = input row
  VREDMAX  s1, v0           ; s1 = max(x)   (numerical stability)
  VBCAST   v1, s1           ; v1 = max broadcast to all lanes
  VSUB     v0, v0, v1       ; v0 = x - max
  VEXP     v0, v0           ; v0 = exp(x - max)
  VREDSUM  s2, v0           ; s2 = sum of exps
  VRECIP   s3, s2           ; s3 = 1 / sum
  VBCAST   v1, s3
  VMUL     v0, v0, v1       ; v0 = exp(.) / sum
  VSTORE   v0, y            ; write result
  KEXIT
```

Built entirely from `VREDMAX`, `VEXP`, `VREDSUM`, `VRECIP`, `VMUL` — none of which is a "softmax instruction." That composability is full programmability.

Host side (control core) launches it:

```
KLAUNCH  softmax, grid=rows, args=ptr   ; one instance per row
KWAIT                                    ; wait for completion
```

---

## 15. What Makes This NPU Fully Programmable

A fully programmable engine needs **all** of the following together — any one alone is not enough:

1. **General compute primitives** (Section 10) — arbitrary elementwise math, comparisons, transcendentals, conversions; new ops built from these, not selected from a fixed list.
2. **Engine-side control flow & predication** (Section 12) — branches, hardware loops, per-lane masking for data-dependent behavior.
3. **Free register/memory model** (Section 9) — general vector registers plus addressable scratchpad with gather/scatter, not fixed tile slots.
4. **A kernel program model** (Section 13) — the engine fetches and runs its own instruction stream; software loads kernels and launches them.
5. **A compiler/toolchain** — practically essential: an LLVM backend or kernel language that turns arbitrary code into the engine's instruction stream. Without it, the hardware is programmable only in theory.

**The one-line test:** a fixed-function engine answers *"which built-in op do you want?"*; this engine answers *"give me any program over tensors and I'll run it"* (see the softmax kernel above).

---

## 16. Design Notes

- **Hybrid is normal.** Keeping the hardwired tensor fast path (Sections 4–8) alongside the programmable vector core (Sections 9–13) is standard: dense matmul stays cheap on the fixed unit, while everything else runs as kernels. This is roughly the GPU model (tensor cores + general SIMT cores).
- **Cost of programmability.** Full programmability costs silicon area and energy versus a fixed-function design — you pay for flexibility. It's a design choice, not strictly better.
- **Precision.** Inputs typically int8/bf16/fp16; accumulators wider (int32/fp32). `VCVT` handles conversions inside kernels.
- **Length-agnostic kernels.** `VSETVL`/`VL` let one kernel run across different vector widths without recompilation (the RISC-V Vector model).
- **Authoritative sources for the real thing.** The control core mirrors the RISC-V base ISA; the vector core mirrors the RISC-V Vector (RVV) extension. For a real implementation, those official specs are the ground truth — the mnemonics here are a teaching aid.
