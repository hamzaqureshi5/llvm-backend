# NPU Vector Core ISA — Documentation

This directory is the reference manual for the NPU Vector Core instruction set. It is
structured as a textbook: each chapter builds on the previous one. The document covers
the full ISA from first principles through working kernel examples and binary encoding.

---

## Prerequisites

- **Basic computer architecture**: instruction sets, register files, memory hierarchy
  (cache vs. DRAM). Familiarity with RISC-style ISAs (RISC-V, ARM) is helpful but not
  required.
- **Floating-point basics**: IEEE 754 single (f32) and double (f64) representation. You
  should know what a mantissa and exponent are, and what "overflow to infinity" means.
- **C**: all examples are in C. Understanding struct initialization with designated
  initializers (`{ .field = value }`) is important for reading kernel code.
- **Neural networks** (Chapter 10 only): a high-level understanding of what a forward
  pass is, what activation functions and normalization layers do. No backprop knowledge
  needed.

---

## How to Use This Book

**If you are writing a kernel:** Start with Chapter 4 (data movement), Chapter 5 (scalar
helpers), Chapter 6 (compute), and Chapter 7 (reductions). Then read Chapter 10 for
complete worked examples of the most common patterns.

**If you are implementing a backend or toolchain:** Read Chapter 8 (predication), Chapter
9 (gather/scatter), and especially Chapter 11 (binary encoding). Chapter 11 documents the
exact byte layout of every instruction word and the LLVM backend pipeline.

**If you are integrating the NPU into a system or writing a driver:** Read Chapter 12
(kernel driver) for the DRAM heap allocator, command ring, and firmware loop internals.
Then read Chapter 13 (runtime API) for the CUDA-style async programming model, ELF
module loader, and arg-copy pool mechanism.

**If you are debugging a kernel:** The register traces in Chapter 10 show expected state
after each instruction. Cross-reference with the opcode semantics table at the end of
this README.

**If you are evaluating the ISA for a new operation:** Consult the Quick Reference table
below. If no single instruction covers your case, Chapter 10 shows how to compose
primitives (e.g., mean via VREDSUM + scalar multiply).

---

## Chapter Table of Contents

| Chapter | File | Title | What You Will Learn |
|---------|------|-------|---------------------|
| 1 | [ch01-introduction.md](ch01-introduction.md) | Introduction | NPU architecture overview; three ISA layers; why three engines; SIMT execution model |
| 2 | [ch02-register-file.md](ch02-register-file.md) | The Register File | Vector regs v0–v31, engine scalars s0–s31, mask regs m0–m7; types and widths; kernel-entry wiring of s0 and s1 |
| 3 | [ch03-memory-model.md](ch03-memory-model.md) | Memory Model | Two-address-space design (DRAM vs SRAM scratchpad); SCRATCH_BASE; latency model (100 vs 1 cycle); DMA |
| 4 | [ch04-data-movement.md](ch04-data-movement.md) | Data Movement | VLOAD, VSTORE, VGATHER, VSCATTER, VMOV, VBCAST; addressing modes; use_base; lane iteration |
| 5 | [ch05-scalar-helpers.md](ch05-scalar-helpers.md) | Engine Scalar Helpers | SLI, SFI, SADD, SADDI, SSUB, SMUL, SMULI, SLW, SLF, SRECIP, KID; building SIMT slice addresses |
| 6 | ch06-compute.md | Compute Primitives | VADD, VSUB, VMUL, VDIV, VMAX, VMIN, VFMA, VEXP, VLOG, VSQRT, VRSQRT, VRECIP, VABS, VNEG, VCVT, VSCAN |
| 7 | [ch07-reductions.md](ch07-reductions.md) | Reductions | VREDSUM, VREDMAX, VREDMIN, VREDAND, VREDOR; tree-reduction internals; scalar result in SReg |
| 8 | ch08-predication.md | Predication and Masking | VCMP (all six comparators), VMASK_SET, VMASK_AND, VMASK_OR, VMASK_NOT, VSEL; @mN prefix; use_mask on loads/stores |
| 9 | ch09-control-flow.md | Control Flow | VSETVL; BR, BRZ, BRNZ, LOOP; structured loop patterns; KEXIT |
| **10** | **[ch10-worked-examples.md](ch10-worked-examples.md)** | **Worked Examples** | **ReLU, softmax, layer norm, dot product, multi-iteration loop, predicated threshold — full kernels with line-by-line traces** |
| **11** | **[ch11-binary-encoding.md](ch11-binary-encoding.md)** | **Binary Encoding** | **32-byte instruction word layout; encode/decode; three forms of an instruction; ELF output; relocations; LLVM backend pipeline** |
| **12** | **[ch12-kernel-driver.md](ch12-kernel-driver.md)** | **The Kernel Driver** | **Driver internals: DRAM heap allocator (first-fit, split, coalesce), command ring buffer, firmware loop, DMA handlers, GEMM micro-programs, fence management** |
| **13** | **[ch13-runtime-api.md](ch13-runtime-api.md)** | **The Runtime API** | **CUDA-style async stack: npuMalloc/Memcpy/LaunchKernel/Synchronize; arg-copy pool; ELF module loader; error handling; complete annotated matmul example** |

---

## Machine Sizing Constants

These constants appear throughout the documentation and source code. They are defined in
`npu-simulator/include/npu.h`.

```
NPU_MEM_SIZE    = 16 MB          Off-chip DRAM size
NPU_SCRATCH_SIZE = 256 KB        On-chip SRAM scratchpad
SCRATCH_BASE    = 0x80000000     High bit distinguishes SRAM from DRAM
NPU_DRAM_LATENCY = 100 cycles    Cost per VLOAD/VSTORE to DRAM
NPU_SRAM_LATENCY = 1  cycle      Cost per VLOAD/VSTORE to SRAM
NPU_DMA_BW      = 32 bytes/cycle DMA throughput (T_DMA_IN/OUT)
VLEN_MAX        = 256            Maximum lanes in a vector register
HW_LANES        = 16             Physical SIMD width (cycle model)
NVREG           = 32             v0..v31
NMREG           = 8              m0..m7
NSREG           = 32             s0..s31
NCREG           = 32             x0..x31 (control core)
NTREG           = 8              t0..t7 (tensor engine)
MAX_TILE        = 64             Maximum tile edge (64×64 floats)
NPU_WORD_BYTES  = 32             Fixed instruction size (all engines)
```

---

## Quick Reference: Complete Vector-Core ISA

### Register Summary

```
v0..v31   Vector registers. Each holds vl floats (vl ≤ VLEN_MAX=256).
           All lanes active when use_mask=0. Zeroed at kernel entry.

s0..s31   Engine scalar registers. Union: float .f / int64 .i.
           s0 = instance id (set by hardware, read via V_KID)
           s1 = args base pointer (set by hardware)
           s2..s31 = available for kernel use. Zeroed at kernel entry.

m0..m7    Mask registers. One bit per lane (0=inactive, 1=active).
           Used by: @mN predication, VSEL, VCMP, VMASK_*.
```

### Complete Opcode Table

All vector opcodes (`vop_t`) with enum values, operand fields, and one-line semantics.

**Legend:**
- `vd/vs1/vs2/vs` = vector register operand fields
- `rd/rs/rs1/rs2` = engine scalar register operand fields
- `md/m1/m2` = mask register operand fields
- `vl` = current vector length (set by VSETVL)
- `[l]` = applies to lane l for l in 0..vl-1

---

#### Group 1 — Registers and Memory (opcodes 0–6)

| Val | Mnemonic | Operands | One-Line Semantics |
|-----|----------|----------|--------------------|
| 0 | `V_VSETVL` | `rd, rs1` or `rd, imm` | `vl = min(s[rs1].i, VLEN_MAX); s[rd].i = vl` |
| 1 | `V_VLOAD` | `vd, rs, imm, use_base` | `v[vd][l] = float[addr + 4*l]` where `addr = s[rs]+imm` or `imm` |
| 2 | `V_VSTORE` | `vs, rs, imm, use_base` | `float[addr + 4*l] = v[vs][l]` for active lanes |
| 3 | `V_VGATHER` | `vd, vs1, rs, imm` | `v[vd][l] = float[addr + 4*(int32)v[vs1][l]]` |
| 4 | `V_VSCATTER` | `vs, vs1, rs, imm` | `float[addr + 4*(int32)v[vs1][l]] = v[vs][l]` |
| 5 | `V_VMOV` | `vd, vs` | `v[vd][l] = v[vs][l]` for active lanes |
| 6 | `V_VBCAST` | `vd, rs` | `v[vd][l] = s[rs].f` for all active lanes |

---

#### Group 2 — Engine Scalar Helpers (opcodes 7–17)

| Val | Mnemonic | Operands | One-Line Semantics |
|-----|----------|----------|--------------------|
| 7 | `V_SLI` | `rd, imm` | `s[rd].i = imm` (load integer immediate) |
| 8 | `V_SFI` | `rd, fimm` | `s[rd].f = (float)fimm` (load float immediate, encoded as double) |
| 9 | `V_SADD` | `rd, rs1, rs2` | `s[rd].i = s[rs1].i + s[rs2].i` |
| 10 | `V_SADDI` | `rd, rs1, imm` | `s[rd].i = s[rs1].i + imm` |
| 11 | `V_SSUB` | `rd, rs1, rs2` | `s[rd].i = s[rs1].i - s[rs2].i` |
| 12 | `V_SMUL` | `rd, rs1, rs2` | `s[rd].i = s[rs1].i * s[rs2].i` |
| 13 | `V_SMULI` | `rd, rs1, imm` | `s[rd].i = s[rs1].i * imm` |
| 14 | `V_SLW` | `rd, rs1, imm` | `s[rd].i = sign_extend32(int32_at(s[rs1]+imm))` (load int32 from memory) |
| 15 | `V_SLF` | `rd, rs1, imm` | `s[rd].f = float_at(s[rs1]+imm)` (load f32 from memory) |
| 16 | `V_SRECIP` | `rd, rs1` | `s[rd].f = 1.0f / s[rs1].f` |
| 17 | `V_KID` | `rd` | `s[rd].i = current grid instance id` (same as s0 at entry) |

---

#### Group 3 — Compute Primitives (opcodes 18–34)

| Val | Mnemonic | Operands | One-Line Semantics |
|-----|----------|----------|--------------------|
| 18 | `V_VADD` | `vd, vs1, vs2` | `v[vd][l] = v[vs1][l] + v[vs2][l]` |
| 19 | `V_VSUB` | `vd, vs1, vs2` | `v[vd][l] = v[vs1][l] - v[vs2][l]` |
| 20 | `V_VMUL` | `vd, vs1, vs2` | `v[vd][l] = v[vs1][l] * v[vs2][l]` |
| 21 | `V_VDIV` | `vd, vs1, vs2` | `v[vd][l] = v[vs1][l] / v[vs2][l]` |
| 22 | `V_VMAX` | `vd, vs1, vs2` | `v[vd][l] = fmaxf(v[vs1][l], v[vs2][l])` |
| 23 | `V_VMIN` | `vd, vs1, vs2` | `v[vd][l] = fminf(v[vs1][l], v[vs2][l])` |
| 24 | `V_VFMA` | `vd, vs1, vs2` | `v[vd][l] += v[vs1][l] * v[vs2][l]` (destructive accumulate) |
| 25 | `V_VCMP` | `md, vs1, vs2, cmp` | `m[md][l] = (v[vs1][l] CMP v[vs2][l])` — see cmpkind table |
| 26 | `V_VSEL` | `vd, vs1, vs2, mask` | `v[vd][l] = m[mask][l] ? v[vs1][l] : v[vs2][l]` |
| 27 | `V_VEXP` | `vd, vs` | `v[vd][l] = expf(v[vs][l])` |
| 28 | `V_VLOG` | `vd, vs` | `v[vd][l] = logf(v[vs][l])` |
| 29 | `V_VRSQRT` | `vd, vs` | `v[vd][l] = 1.0f / sqrtf(v[vs][l])` |
| 30 | `V_VSQRT` | `vd, vs` | `v[vd][l] = sqrtf(v[vs][l])` |
| 31 | `V_VRECIP` | `vd, vs` | `v[vd][l] = 1.0f / v[vs][l]` |
| 32 | `V_VCVT` | `vd, vs, cvt` | Convert v[vs][l] per cvtmode → v[vd][l] — see cvtmode table |
| 33 | `V_VABS` | `vd, vs` | `v[vd][l] = fabsf(v[vs][l])` |
| 34 | `V_VNEG` | `vd, vs` | `v[vd][l] = -v[vs][l]` |

---

#### Group 4 — Reductions (opcodes 35–40)

| Val | Mnemonic | Operands | One-Line Semantics |
|-----|----------|----------|--------------------|
| 35 | `V_VREDSUM` | `rd, vs` | `s[rd].f = sum(v[vs][0..vl-1])` |
| 36 | `V_VREDMAX` | `rd, vs` | `s[rd].f = max(v[vs][0..vl-1])` |
| 37 | `V_VREDMIN` | `rd, vs` | `s[rd].f = min(v[vs][0..vl-1])` |
| 38 | `V_VREDAND` | `rd, vs` | `s[rd].i = bitwise-AND of (int64)v[vs][l] for l in 0..vl-1` |
| 39 | `V_VREDOR` | `rd, vs` | `s[rd].i = bitwise-OR of (int64)v[vs][l] for l in 0..vl-1` |
| 40 | `V_VSCAN` | `vd, vs` | `v[vd][l] = sum(v[vs][0..l])` (inclusive prefix sum) |

---

#### Group 5 — Predication and Masking (opcodes 41–44)

| Val | Mnemonic | Operands | One-Line Semantics |
|-----|----------|----------|--------------------|
| 41 | `V_VMASK_SET` | `md, vs` | `m[md][l] = (v[vs][l] != 0.0f)` |
| 42 | `V_VMASK_AND` | `md, m1, m2` | `m[md][l] = m[m1][l] & m[m2][l]` |
| 43 | `V_VMASK_OR` | `md, m1, m2` | `m[md][l] = m[m1][l] \| m[m2][l]` |
| 44 | `V_VMASK_NOT` | `md, m1` | `m[md][l] = !m[m1][l]` |

---

#### Group 6 — Control Flow (opcodes 45–49)

| Val | Mnemonic | Operands | One-Line Semantics |
|-----|----------|----------|--------------------|
| 45 | `V_BR` | `imm` | `pc = imm` (unconditional branch to instruction index) |
| 46 | `V_BRZ` | `rs, imm` | `if s[rs].i == 0: pc = imm` |
| 47 | `V_BRNZ` | `rs, imm` | `if s[rs].i != 0: pc = imm` |
| 48 | `V_LOOP` | `rs, imm` | `s[rs].i -= 1; if s[rs].i > 0: pc = imm` |
| 49 | `V_KEXIT` | (none) | terminate this kernel instance |

---

### Auxiliary Enumerations

**`cmpkind_t` — used by V_VCMP `.cmp` field:**

| Value | Name | Meaning |
|-------|------|---------|
| 0 | `CMP_LT` | less than |
| 1 | `CMP_LE` | less than or equal |
| 2 | `CMP_EQ` | equal |
| 3 | `CMP_NE` | not equal |
| 4 | `CMP_GT` | greater than |
| 5 | `CMP_GE` | greater than or equal |

**`cvtmode_t` — used by V_VCVT `.cvt` field:**

| Value | Name | Semantics |
|-------|------|-----------|
| 0 | `CVT_F32` | identity (no-op, pass through as float) |
| 1 | `CVT_I8` | round to nearest, clamp to [-128, 127], store as float |
| 2 | `CVT_I32` | round to nearest integer, store as float |

---

### Cycle Cost Summary

```
Instruction class                         Cost (cycles)
───────────────────────────────────────────────────────
Scalar ops: SLI, SFI, SADD, SADDI,
  SSUB, SMUL, SMULI, SLW, SLF,
  SRECIP, KID                             1
VSETVL, BR, BRZ, BRNZ, LOOP, KEXIT       1
Elementwise vector: VADD, VSUB, VMUL,
  VMAX, VMIN, VMOV, VBCAST, VABS,
  VNEG, VCMP, VSEL, VMASK_*, VCVT        passes (= ceil(vl/16))
VFMA                                      passes
Transcendentals: VEXP, VLOG, VSQRT,
  VRSQRT, VRECIP, VDIV                   passes × 4
Reductions: VREDSUM, VREDMAX, VREDMIN,
  VREDAND, VREDOR, VSCAN                 passes + 4
Gather/scatter: VGATHER, VSCATTER        vl cycles
VLOAD / VSTORE (SRAM, high bit set)      passes + 1
VLOAD / VSTORE (DRAM, high bit clear)    passes + 100

passes = ceil(vl / HW_LANES) = ceil(vl / 16), minimum 1
```

---

### Common Patterns

```
Pattern                     Instructions            Notes
─────────────────────────────────────────────────────────────────────────
Load scalar float from args  V_SLF rd, rs1, imm     load f32 at s[rs1]+imm
Load float constant          V_SFI rd, fimm          scalar float literal
Scalar to all lanes          V_VBCAST vd, rs         broadcast s[rs].f
Scalar reciprocal            V_SRECIP rd, rs1        1/s[rs1].f
1/sqrt(x)                    V_VRSQRT vd, vs         faster than VSQRT+VRECIP
Max-normalize (softmax)      VREDMAX + VBCAST + VSUB subtract max before exp
Intra-lane dot product       VFMA + VREDSUM          FMA accumulates, sum collapses
Counted loop                 SLI (count) + LOOP      decrement-and-branch at end
Conditional update           VCMP + VSEL             mux between two vectors
Zero vector                  SFI(0.0) + VBCAST       broadcast zero
Stride through array         SADDI rd,rs1,imm (×N)   byte offset += chunk_bytes
SIMT slice address           KID + SMULI + SADD      id × stride_bytes + base
```

---

*This documentation covers the functional ISA as implemented in `npu-simulator/`. Cycle
counts are rough estimates from the simulator's cycle model, not silicon measurements.
See `npu-simulator/docs/fully_prog_npu.md` for the ISA reference and
`npu-simulator/docs/llvm_lowering.md` for the compiler design sketch.*
