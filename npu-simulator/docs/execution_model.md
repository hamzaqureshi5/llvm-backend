# NPU Execution Model — SIMD vs SIMT vs SPMD

This document explains which execution model each NPU layer actually uses,
and why the distinction matters when writing kernels.

---

## The Three Models

**SIMD** (Single Instruction, Multiple Data):
- One program counter for the entire execution unit
- One instruction operates on N data lanes simultaneously
- All lanes execute the same instruction at every step
- Control flow divergence is the **programmer's problem** — handled explicitly
  with mask registers
- Examples: x86 AVX-512, ARM NEON, RISC-V V (RVV)

**SIMT** (Single Instruction, Multiple Threads):
- Multiple logical threads, each with their own conceptual PC and register file
- Threads execute in lockstep (a "warp") when they agree on the same instruction
- Divergent branches resolved by **hardware** — serializes both paths, masks
  inactive threads
- Programmer writes scalar code per thread; warp mechanics are invisible
- Examples: CUDA warps, AMD GCN wavefronts

**SPMD** (Single Program, Multiple Data):
- Many independent instances run the same program on different data
- Each instance is completely independent — no shared execution unit
- No warp, no wavefront, no hardware synchronization between instances
- Examples: OpenCL NDRange, CUDA from the C programmer's perspective

---

## Layer 1 — Scalar

Plain scalar CPU. One instruction, one datum, one thread. No parallelism.

---

## Layer 2 — Matrix (not SIMD/SIMT/SPMD)

The tensor engine operates on whole tiles (up to 64×64 floats) per instruction.
This is neither SIMD (no explicit lanes) nor SIMT (no threads). Closest analogy:
Intel AMX — a matrix co-processor that multiplies tiles atomically.

---

## Layer 3 — Vector Core: SIMD within an Instance

The proof is in `run_vector_instance()` in `npu.c`. There is **one program
counter** per kernel instance:

```c
int pc = 0;   // ONE pc for all lanes — not per-lane

while (pc < k->len) {
    const vinst *I = &k->code[pc];

    case V_VADD:
        for (int l = 0; l < vl; l++)
            if (ACTIVE(l))
                n->v[I->vd][l] = n->v[I->vs1][l] + n->v[I->vs2][l];
        break;

    case V_BRZ:
        if (n->s[I->rs].i == 0) { pc = (int)I->imm; continue; }
        break;
    pc++;
}
```

Branches (`V_BR`, `V_BRZ`, `V_BRNZ`) move the single `pc` for the entire
instance. No lane can independently take a different path. This is SIMD.

**Divergence is explicit and manual.** When lanes need different behavior,
the programmer uses mask registers:

```
; lanes where v0[l] > 0 take path A, others take path B

VCMP  m0, v0, v_zero, GT    m0[l] = (v0[l] > 0) ? 1 : 0

VADD  @m0, v2, v0, v1       runs only where m0[l] == 1  (path A)

VMASK_NOT  m1, m0
VSUB  @m1, v2, v0, v1       runs only where m0[l] == 0  (path B)
```

In SIMT (CUDA) you write a plain `if` — the hardware handles the rest.
On the NPU you write masks explicitly. This is the definitive signature of
**predicated SIMD**, not SIMT.

---

## Layer 3 — Grid Launch: SPMD

`C_KLAUNCH` launches `grid` instances. Each instance gets its own fresh
register file (`s0`–`s31`, `v0`–`v31`, `m0`–`m7` all zeroed) and runs the
same kernel program independently.

```c
// npu.c — instances run sequentially in the functional simulator
for (int64_t g = 0; g < grid; g++) {
    memset(n->v, 0, ...);
    memset(n->s, 0, ...);
    n->s[0].i = g;           // s0 = instance id
    n->s[1].i = args_ptr;    // s1 = args address
    run_vector_instance(...);
    // instance 0 completes entirely before instance 1 starts
}
```

**Why this is not SIMT:**

| | NPU grid | CUDA warp |
|---|---|---|
| Threads share execution unit? | No | Yes (32 threads/warp) |
| Hardware divergence handling? | No | Yes |
| Register files coexist? | No (one at a time) | Yes (all 32 simultaneously) |
| `__syncwarp()` needed? | Never | Sometimes |

Each instance starts with a completely fresh register snapshot. They are fully
independent. This is SPMD — the programming model of CUDA from the C
programmer's perspective, but without SIMT execution underneath.

---

## Summary Table

| Layer | Engine | Execution model | Comparable to |
|---|---|---|---|
| Layer 1 | Control core | Scalar | RISC-V scalar |
| Layer 2 | Tensor engine | Matrix tile ops | Intel AMX |
| Layer 3 (within instance) | Vector core | **SIMD** — one PC, vl lanes, explicit masks | RISC-V V, AVX-512 |
| Layer 3 (grid) | KLAUNCH instances | **SPMD** — independent instances | OpenCL NDRange |

---

## Practical Implications

| Question | Answer |
|---|---|
| Can two lanes take different branches? | No — branches move the whole instance's PC |
| How do I handle if/else per lane? | Use `VCMP` + mask registers (`@m` prefix) |
| Do instances share registers? | No — each starts fresh, no shared state |
| Can instances communicate? | Only through DRAM/SRAM — not registers |
| Is there a `__syncthreads()`? | No — use `BARRIER` for cross-launch sync |
| Does warp divergence affect me? | No — there are no warps |
