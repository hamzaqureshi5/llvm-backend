# Chapter 4: Data Movement Instructions (Section 9)

> **Series context.** Chapter 3 established the memory model: a flat 32-bit
> space split at bit 31 into 16 MB DRAM (100 cycles/access) and 256 KB SRAM
> (1 cycle/access). This chapter covers the seven instructions in **Section 9
> of the vector core ISA** that move data between memory and registers: the
> foundational set that every vector kernel begins with.

---

## 4.0 Section 9 at a Glance

Section 9 of the vector ISA defines the register file and the memory model
(treated in Chapter 3) plus the instructions that actually move data. The full
set:

| Opcode enum   | Mnemonic   | Category              | What it does                        |
|---------------|------------|-----------------------|-------------------------------------|
| `V_VSETVL`   | `VSETVL`   | Vector length control | Set number of active lanes, `VL`    |
| `V_VLOAD`    | `VLOAD`    | Vector memory         | Consecutive load from memory        |
| `V_VSTORE`   | `VSTORE`   | Vector memory         | Consecutive store to memory         |
| `V_VGATHER`  | `VGATHER`  | Vector memory         | Gather: per-lane indirect load      |
| `V_VSCATTER` | `VSCATTER` | Vector memory         | Scatter: per-lane indirect store    |
| `V_VMOV`     | `VMOV`     | Register              | Vector-register-to-register copy    |
| `V_VBCAST`   | `VBCAST`   | Register              | Scalar-to-all-lanes broadcast       |

All of these operate on the vector register file (`v0`–`v31`), the scalar
register file (`s0`–`s31`), and the current vector length `VL`. The scalar
helper instructions (`SLW`, `SLF`, `SLI`, `SFI`, `SADD`, `SMUL`, etc.) are the
engine-side ISA support for computing addresses; they are covered at the end of
this chapter since they appear in every realistic kernel.

---

## 4.1 VSETVL — Set Vector Length

### 4.1.1 What It Does

```
VSETVL rd, imm          ; VL = imm;   s[rd] = old VL
VSETVL rd, rs1          ; VL = s[rs1]; s[rd] = old VL
```

`VSETVL` controls how many lanes are active in every subsequent vector
instruction. It writes the **new** VL to the hardware VL register, and saves the
**old** VL into scalar register `rd` (so a caller can save/restore VL across a
subroutine call if needed). If the requested length exceeds `VLEN_MAX` (256), it
is clamped to 256.

This is the first instruction that nearly every kernel emits, before any vector
operation can produce meaningful results.

### 4.1.2 Why It Exists

The NPU vector core is **length-agnostic** in the style of RISC-V Vector (RVV).
Rather than hardwiring a fixed SIMD width (like SSE's 128-bit or AVX-512's
512-bit), the ISA separates the description of an operation ("add these
vectors") from the physical width of the hardware ("we have 16 physical lanes").
`VSETVL` is the mechanism for negotiating that separation at runtime.

The benefits:

1. **One compiled kernel works for any input length.** A softmax kernel compiled
   once for `VLEN_MAX = 256` works correctly on a row of length 16, 64, or 256
   without recompilation. The loop adjusts `VL` for the tail.

2. **Hardware portability.** A future NPU with 32 physical lanes instead of 16
   runs the same binary; `VSETVL` returns the hardware-supported VL, which the
   software uses to tile its loop.

3. **Tail handling.** When a tensor dimension is not a multiple of the physical
   lane width, the final iteration sets `VL` to the remainder so that only
   the valid lanes participate.

### 4.1.3 Instruction Fields

In the `vinst` struct:

```c
/* VSETVL: set vector length */
{
    .op     = V_VSETVL,
    .rd     = 2,          /* s2 receives old VL */
    .rs1    = 0,          /* if use_base=1, read requested length from s[rs1] */
    .imm    = 16,         /* if rs1=0 and use_base=0, use imm directly        */
    .use_base = 0,        /* 0 => use imm; 1 => use s[rs1]                    */
}
```

Simulator implementation (from `npu.c`):

```c
case V_VSETVL: {
    int64_t req = I->use_base ? n->s[I->rs1].i
                              : (I->rs1 ? n->s[I->rs1].i : I->imm);
    if (req > VLEN_MAX) req = VLEN_MAX;
    if (req < 0) req = 0;
    n->vl = (int)req;
    n->s[I->rd].i = n->vl;    /* write new VL (post-clamp) to rd */
    break;
}
```

Cycle cost: **1 cycle** (scalar control operation, no data movement).

### 4.1.4 Assembly Syntax

```asm
VSETVL s2, 16       ; VL = 16, old VL → s2
VSETVL s8, s4       ; VL = s4, old VL → s8
```

### 4.1.5 Usage Patterns

**Pattern A — Fixed length for a fixed-size kernel:**

```c
/* Softmax over exactly 16 elements */
{ .op=V_VSETVL, .rd=2, .imm=16 }
```

After this, all vector instructions operate on 16 lanes. `v0[0..15]` are active;
`v0[16..255]` are inert (their values are preserved but never written by masked
ops or read by reductions).

**Pattern B — Dynamic length from argument:**

```c
/* Kernel receives N (the row length) as arg s4 */
{ .op=V_SLW, .rd=4, .use_base=1, .rs=1, .imm=8 },   /* s4 = N from arg block */
{ .op=V_VSETVL, .rd=8, .rs1=4, .use_base=1 },         /* VL = N               */
```

**Pattern C — Tail loop for N not divisible by 64:**

A kernel processing N floats in 64-lane chunks uses the following loop pattern
in assembly:

```asm
; s2 = N (total elements), s3 = base addr
loop:
  VSETVL   s8, s2          ; VL = min(s2, VLEN_MAX) — clamped at hardware max
  VLOAD    v0, 0(s3)        ; load VL floats
  ... compute v0 ...
  VSTORE   v0, 0(s3)        ; store VL floats
  SADDI    s3, s3, 256      ; advance pointer by 64 * 4 bytes
  SSUB     s2, s2, s8       ; N_remaining -= VL
  BRNZ     s2, loop         ; if any left, continue
  KEXIT
```

On the final iteration, `s2` will be 1–63, `VSETVL` clamps to that value, and
only those lanes participate. The same binary handles any N without a separate
tail epilogue.

**Pattern D — Three-way unroll for 48 elements:**

```c
/* Unrolled: three iterations of vl=16 for a 48-element kernel */
{ .op=V_VSETVL, .rd=0, .imm=16 },          /* pass 1 */
{ .op=V_VLOAD,  .vd=0, .use_base=1, .rs=2, .imm=0   },
... compute ...
{ .op=V_VSTORE, .vs=0, .use_base=1, .rs=4, .imm=0   },

{ .op=V_VLOAD,  .vd=0, .use_base=1, .rs=2, .imm=64  }, /* pass 2, VL still 16 */
... compute ...
{ .op=V_VSTORE, .vs=0, .use_base=1, .rs=4, .imm=64  },

{ .op=V_VLOAD,  .vd=0, .use_base=1, .rs=2, .imm=128 }, /* pass 3 */
... compute ...
{ .op=V_VSTORE, .vs=0, .use_base=1, .rs=4, .imm=128 },
```

`VSETVL` is called once; `VL` remains 16 for all three passes.

### 4.1.6 The VL State

`VL` is a hardware register, part of `npu_t`:

```c
typedef struct {
    ...
    float   v[NVREG][VLEN_MAX];   /* v0..v31, each VLEN_MAX lanes wide     */
    sreg_t  s[NSREG];             /* s0..s31 engine scalar registers        */
    int     vl;                   /* current vector length                  */
    ...
} npu_t;
```

`VL` is reset to 0 at the start of every kernel instance (`run_vector_instance`
clears the whole engine state including `n->vl = 0`). A kernel that forgets to
call `VSETVL` will silently execute all vector operations on 0 lanes — producing
no results and no error. This is a common programming mistake; the simulator's
trace mode (`n->trace = 1`) prints `vl=` on each instruction to catch it.

---

## 4.2 VLOAD — Vector Load from Memory

### 4.2.1 What It Does

```
VLOAD vd, s[rs]+imm
```

Loads `VL` consecutive `float32` values from memory starting at address
`s[rs]+imm` (or just `imm` if `use_base=0`) into vector register `vd`.

Formally: `vd[l] = mem_f32[addr + l*4]` for `l = 0 .. VL-1`.

### 4.2.2 Instruction Fields

```c
{
    .op       = V_VLOAD,
    .vd       = 0,         /* destination vector register v0              */
    .use_base = 1,         /* address = s[rs] + imm                       */
    .rs       = 2,         /* base address register s2                    */
    .imm      = 0,         /* byte offset from base                       */
}
```

Simulator implementation:

```c
case V_VLOAD:
    n->stats.vector_cycles += mem_latency(addr);   /* DRAM=100, SRAM=1 */
    for (int l = 0; l < vl; l++)
        n->v[I->vd][l] = ld_f(n, addr + 4u * l);
    break;
```

Key details:
- Each lane accesses a **distinct** 4-byte float: lane 0 reads `addr`, lane 1
  reads `addr+4`, ..., lane `vl-1` reads `addr + (vl-1)*4`.
- The cycle cost is `mem_latency(addr)` added once for the whole vector (the
  hardware issues a burst transfer, not one transaction per lane). On real
  hardware this would be a cache-line or burst-mode fetch.
- There is **no predicate mask** on `VLOAD` in the simulator — the mask affects
  only compute and store. All `VL` lanes are always loaded.

### 4.2.3 Data Flow Diagram

```
  Memory (SRAM or DRAM starting at addr):
  ┌──────┬──────┬──────┬──────┬──────┬──────┬──────┬──────┐
  │ f0   │ f1   │ f2   │ f3   │ f4   │ f5   │ f6   │ f7   │  (VL=8)
  │addr+0│addr+4│addr+8│addr+C│addr+10│addr+14│addr+18│addr+1C│
  └──┬───┴──┬───┴──┬───┴──┬───┴──┬────┴──┬────┴──┬────┴──┬────┘
     │      │      │      │      │        │        │        │
     ▼      ▼      ▼      ▼      ▼        ▼        ▼        ▼
  v0[ 0 ][ 1 ][ 2 ][ 3 ][ 4 ][ 5 ][ 6 ][ 7 ]  ← vector register lanes
  └────────────────────────────────────────────┘
                  vl * 4 = 32 bytes total
```

### 4.2.4 Assembly Syntax and Examples

```asm
VLOAD v0, 0(s2)      ; v0 = mem[s2 .. s2+vl*4-1], use_base=1
VLOAD v1, 256(s2)    ; v1 = mem[s2+256 .. s2+256+vl*4-1]
VLOAD v2, 0x80001000 ; absolute SRAM address, use_base=0
```

**Example 1: Loading two operand vectors for VADD**

```c
/* Kernel: c[i] = a[i] + b[i]
 * s2 = base of A, s3 = base of B, s4 = base of C, s5 = N */
{ .op=V_VSETVL, .rd=6, .rs1=5, .use_base=1 },      /* VL = N         */
{ .op=V_VLOAD,  .vd=0, .use_base=1, .rs=2 },        /* v0 = A[0..N-1] */
{ .op=V_VLOAD,  .vd=1, .use_base=1, .rs=3 },        /* v1 = B[0..N-1] */
{ .op=V_VADD,   .vd=2, .vs1=0, .vs2=1 },            /* v2 = v0 + v1   */
{ .op=V_VSTORE, .vs=2, .use_base=1, .rs=4 },        /* C = v2         */
```

**Example 2: Loading successive rows (stride via scalar arithmetic)**

```c
/* Load row i of a matrix with row stride STRIDE bytes */
/* s2 = base, s0 = row index, s5 = STRIDE */
{ .op=V_SMUL,  .rd=7, .rs1=0, .rs2=5 },             /* s7 = i * STRIDE */
{ .op=V_SADD,  .rd=8, .rs1=2, .rs2=7 },             /* s8 = base + offset */
{ .op=V_VLOAD, .vd=0, .use_base=1, .rs=8 },          /* v0 = row i */
```

### 4.2.5 Performance Considerations

- **SRAM vs. DRAM**: as established in Chapter 3, always prefer loading from
  SRAM. Stage data via `T_DMA_IN` before launching the kernel.
- **VL matters for cycle modeling**: the simulator charges `mem_latency` once
  per `VLOAD`, independent of VL. On real hardware, wide vectors require wider
  buses or more cycles. With 16 physical lanes (`HW_LANES=16`) and VL=256, the
  hardware would need 16 passes across the physical SRAM ports.
- **No scatter/gather**: `VLOAD` is contiguous only. For non-contiguous patterns,
  use `VGATHER` (Section 4.5), which costs one cycle per lane.

---

## 4.3 VSTORE — Vector Store to Memory

### 4.3.1 What It Does

```
VSTORE s[rs]+imm, vs
```

Stores `VL` consecutive `float32` values from vector register `vs` to memory
starting at address `s[rs]+imm`. This is the exact inverse of `VLOAD`.

Formally: `mem_f32[addr + l*4] = vs[l]` for `l = 0 .. VL-1`, subject to mask.

### 4.3.2 Instruction Fields

```c
{
    .op       = V_VSTORE,
    .vs       = 2,         /* source vector register v2             */
    .use_base = 1,
    .rs       = 4,         /* address = s4 + imm                   */
    .imm      = 0,
    .use_mask = 0,         /* 0 = store all VL lanes                */
    .mask     = 0,         /* mask reg (only used if use_mask=1)    */
}
```

Simulator implementation:

```c
case V_VSTORE:
    n->stats.vector_cycles += mem_latency(addr);
    for (int l = 0; l < vl; l++)
        if (ACTIVE(l)) st_f(n, addr + 4u * l, n->v[I->vs][l]);
    break;
```

The `ACTIVE(l)` macro checks the predicate mask if `use_mask=1`. This is the
mechanism for predicated stores: only lanes where the mask bit is 1 write to
memory. Lanes where the mask is 0 leave the memory location unchanged — a key
property for branchless conditional writes.

### 4.3.3 Data Flow Diagram

```
  v2 (source register):
  [ r0 ][ r1 ][ r2 ][ r3 ][ r4 ][ r5 ][ r6 ][ r7 ]   (VL=8)
     │      │      │      │      │      │      │      │
     ▼      ▼      ▼      ▼      ▼      ▼      ▼      ▼
  ┌──────┬──────┬──────┬──────┬──────┬──────┬──────┬──────┐
  │ r0   │ r1   │ r2   │ r3   │ r4   │ r5   │ r6   │ r7   │  memory
  │addr+0│addr+4│addr+8│addr+C│...   │      │      │      │
  └──────┴──────┴──────┴──────┴──────┴──────┴──────┴──────┘

  With mask active (m0 = [1,1,0,1,0,0,1,1]):
  [ r0 ][ r1 ][ ?? ][ r3 ][ ?? ][ ?? ][ r6 ][ r7 ]  ← only marked lanes write
```

### 4.3.4 Predicated Store Pattern

A common pattern in ML inference is writing results only to lanes that passed a
filter condition (e.g., non-padding positions, active attention heads):

```c
/* v0 = results; m0 = validity mask (from prior VCMP)
 * Only write lanes where m0[l] = 1 */
{ .op=V_VSTORE, .vs=0, .use_base=1, .rs=4, .use_mask=1, .mask=0 }
```

Memory locations corresponding to mask=0 lanes are untouched. Combined with
`VSEL` (Chapter 5), this lets the vector core express conditional writes without
a branch.

### 4.3.5 The VADD Demo: Complete Kernel with VLOAD and VSTORE

From Demo 1 of `examples.c`, the full vector-add kernel:

```c
static const vinst vadd_kernel[] = {
    /* Prologue: read argument block from s1 (args base) */
    { .op=V_SLW, .rd=2, .use_base=1, .rs=1, .imm=0  }, /* s2 = A ptr */
    { .op=V_SLW, .rd=3, .use_base=1, .rs=1, .imm=4  }, /* s3 = B ptr */
    { .op=V_SLW, .rd=4, .use_base=1, .rs=1, .imm=8  }, /* s4 = C ptr */
    { .op=V_SLW, .rd=5, .use_base=1, .rs=1, .imm=12 }, /* s5 = N     */
    /* Body */
    { .op=V_VSETVL, .rd=6, .rs1=5, .use_base=1 },       /* VL = N    */
    { .op=V_VLOAD,  .vd=0, .use_base=1, .rs=2 },         /* v0 = A    */
    { .op=V_VLOAD,  .vd=1, .use_base=1, .rs=3 },         /* v1 = B    */
    { .op=V_VADD,   .vd=2, .vs1=0, .vs2=1 },             /* v2 = A+B  */
    { .op=V_VSTORE, .vs=2, .use_base=1, .rs=4 },          /* C = v2   */
    { .op=V_KEXIT },
};
```

This is the minimal complete kernel: prologue reads args, sets VL, loads two
vectors, computes, stores one result, exits. Every real kernel extends this
structure.

---

## 4.4 VGATHER — Indirect Vector Load (Per-Lane Addresses)

### 4.4.1 What It Does

```
VGATHER vd, vs_idx, addr_base
vd[l] = mem_f32[addr + vs_idx[l]*4]    for l = 0 .. VL-1
```

`VGATHER` is a generalized load where each lane provides its own index into a
base array. Lane `l` reads from `addr + (int32)vs_idx[l] * 4`. The result is a
vector where each lane comes from a **different** memory location.

Simulator implementation:

```c
case V_VGATHER:
    for (int l = 0; l < vl; l++) {
        int32_t idx = (int32_t)n->v[I->vs1][l];      /* per-lane index */
        n->v[I->vd][l] = ld_f(n, addr + 4u * (uint32_t)idx);
    }
    break;
```

Note that `vs_idx[l]` is interpreted as an `int32` (allowing negative offsets),
then cast to `uint32_t` for the address calculation.

### 4.4.2 Why It Exists

Dense ML computation rarely needs gather — matmul, conv, layernorm, and softmax
all have stride-1 or stride-k access patterns that `VLOAD` handles efficiently.
`VGATHER` exists for the cases that fundamentally cannot be expressed as a
contiguous load:

1. **Embedding lookup**: Given a vector of token IDs (integer indices into a
   vocabulary embedding table), load the scalar embedding value for each token.
   Each lane has a different ID; each reads from a different row of the embedding
   matrix.

2. **Sparse attention patterns**: In non-uniform sparse attention, each query
   position attends to a different set of key positions. The key positions are
   stored as indices; gather reads the key vectors.

3. **Quantization index tables**: Lookup-table dequantization maps each 4-bit
   or 8-bit index to a float scale via a per-index table.

4. **Histogram accumulation**: (Paired with VSCATTER) Count occurrences: for
   each element, gather the current count at its bucket address, increment, then
   scatter back.

### 4.4.3 Instruction Fields

```c
{
    .op       = V_VGATHER,
    .vd       = 0,         /* destination vector register             */
    .vs1      = 1,         /* index vector: v1[l] = per-lane index    */
    .use_base = 1,
    .rs       = 2,         /* base address register s2                */
    .imm      = 0,         /* byte offset added to base               */
}
```

The effective address for lane `l` is:
```
addr_base = s[rs] + imm
lane_addr[l] = addr_base + (int32_t)v[vs1][l] * 4
```

### 4.4.4 Data Flow Diagram

```
  vs1 (index vector, VL=4):
  [ 100 ][ 256 ][ 42 ][ 512 ]     (four different indices)
       │         │        │         │
       │   ┌─────┘  ┌─────┘  ┌─────┘
       ▼   ▼        ▼        ▼
  Memory (base = s2):
  ┌─────────────────────────────────────────────────────────────┐
  │  ... [addr+42*4] ...  [addr+100*4] ... [addr+256*4] ...     │
  │         ▲                  ▲                ▲                │
  │  f2    │            f100  │        f256    │         f512   │
  └─────┬──┘──────────────────┴────────────────┴───────────────┘
        │
        ▼ (the four values gathered, placed in their source lanes)

  vd (result, VL=4):
  [ f100 ][ f256 ][ f42 ][ f512 ]

  Lane 0 got mem[base + 100*4]
  Lane 1 got mem[base + 256*4]
  Lane 2 got mem[base +  42*4]
  Lane 3 got mem[base + 512*4]
```

The key visual: the indices are **arbitrary and unordered**. The memory
accesses are to `VL` distinct addresses with no regularity. In the extreme case
every lane reads from a completely different cache line.

### 4.4.5 Cycle Cost

From the simulator's cycle model:

```c
case V_VGATHER: case V_VSCATTER:
    return (uint64_t)(vl > 0 ? vl : 1);   /* ~1 lane per cycle */
```

`VGATHER` costs **VL cycles** — one cycle per lane — versus the constant cost
of `VLOAD` (1–2 cycles for the burst). This reflects that hardware must issue
`VL` independent random-access transactions to memory, which cannot easily be
pipelined or parallelized (unless the hardware has a full scatter-gather unit,
which most NPUs do not).

For VL=64, `VGATHER` costs 64 cycles minimum at SRAM; for DRAM it costs
64 × 100 = 6400 cycles. This is a 64× degradation versus a `VLOAD` of the same
amount of data from DRAM (which costs 100 cycles for the burst). The programmer
must therefore:

- Avoid `VGATHER` in the inner loop of performance-critical kernels if possible.
- If unavoidable, move the gather table into SRAM before the kernel runs, to pay
  SRAM cost (64 cycles) instead of DRAM cost (6400 cycles).
- Consider whether the access pattern can be rewritten as a structured layout
  followed by a contiguous `VLOAD`.

### 4.4.6 Embedding Lookup Example

A 4-lane embedding lookup (vocabulary size 1024, embedding dim = 1):

```c
/* Input: v1 = [token_id_0, token_id_1, token_id_2, token_id_3]  (floats ← CVT from int) */
/* Table: SRAM[base] holds the embedding value for each token ID  */

{ .op=V_VSETVL,  .rd=0, .imm=4 },
/* Load token IDs (from int32 input tensor), already stored as float in v1 */
{ .op=V_VLOAD,   .vd=1, .use_base=1, .rs=2 },         /* v1 = token IDs    */
/* Gather embedding values: vd[l] = emb_table[token_id[l]] */
{ .op=V_VGATHER, .vd=0, .vs1=1, .use_base=1, .rs=3 }, /* v0 = embeddings   */
/* v0 now holds [emb[token_0], emb[token_1], emb[token_2], emb[token_3]] */
```

---

## 4.5 VSCATTER — Indirect Vector Store (Per-Lane Addresses)

### 4.5.1 What It Does

```
VSCATTER s[rs]+imm, vs_idx, vs
mem_f32[addr + vs_idx[l]*4] = vs[l]    for l = 0 .. VL-1
```

`VSCATTER` is the exact inverse of `VGATHER`: each lane stores to a **different**
address determined by the per-lane index in `vs1`. Lane `l` writes `vs[l]` to
`addr + (int32_t)vs1[l] * 4`.

Simulator implementation:

```c
case V_VSCATTER:
    for (int l = 0; l < vl; l++) {
        int32_t idx = (int32_t)n->v[I->vs1][l];
        st_f(n, addr + 4u * (uint32_t)idx, n->v[I->vs][l]);
    }
    break;
```

### 4.5.2 Instruction Fields

```c
{
    .op       = V_VSCATTER,
    .vs       = 2,         /* source vector register v2 (data to write)     */
    .vs1      = 1,         /* index vector: v1[l] = per-lane destination idx */
    .use_base = 1,
    .rs       = 3,         /* base address register s3                       */
    .imm      = 0,
}
```

### 4.5.3 Data Flow Diagram

```
  vs (data to write, VL=4):
  [  d0  ][  d1  ][  d2  ][  d3  ]    four values from four lanes

  vs1 (index vector, VL=4):
  [ 200  ][ 50  ][ 300  ][ 10  ]      each lane's destination index

  Combined addresses:
  Lane 0: base + 200*4  ← d0
  Lane 1: base +  50*4  ← d1
  Lane 2: base + 300*4  ← d2
  Lane 3: base +  10*4  ← d3

  Memory (base = s3):
  ...[ d3 ]...[ d1 ]...[ d0 ]...[ d2 ]...
      ↑         ↑         ↑         ↑
   idx=10    idx=50    idx=200   idx=300
```

### 4.5.4 When to Use VSCATTER

1. **Histogram / frequency count** (with atomic accumulation on hardware; in the
   simulator, each lane simply overwrites): bin index vector → scatter counts.

2. **Non-contiguous output writes**: when the output layout has a stride or
   permutation that cannot be expressed as a single contiguous block.

3. **Sparse update**: applying gradients or corrections to non-consecutive
   positions in a parameter vector.

4. **Permutation tables**: rearranging a vector according to a data-dependent
   permutation (`vd[perm[l]] = vs[l]`).

### 4.5.5 Hazard Warning: Write Conflicts

If two lanes write to the **same** destination index, the result is the value
from whichever lane executes last in the sequential simulator loop. On real
hardware with a parallel scatter unit, this is a write conflict whose resolution
is typically undefined (or a hardware fault). The programmer must ensure that
the index vector has **no duplicate entries** in correctness-critical use cases.

```
; UNSAFE: lanes 0 and 2 both target index 5
vs1 = [ 5 ][ 3 ][ 5 ][ 7 ]   ; CONFLICT at idx=5
```

The simulator will silently write the lane-2 value (it runs last), but real
hardware may produce either value, or neither.

### 4.5.6 Cycle Cost

Same as `VGATHER`: **VL cycles** (one random write per lane). The same
advice about SRAM staging applies.

---

## 4.6 VMOV — Vector Register-to-Register Copy

### 4.6.1 What It Does

```
VMOV vd, vs
vd[l] = vs[l]    for l = 0 .. VL-1  (subject to mask)
```

Copies `VL` lanes from source vector register `vs` to destination vector
register `vd`. Respects the predicate mask if `use_mask=1`.

### 4.6.2 Instruction Fields

```c
{
    .op       = V_VMOV,
    .vd       = 3,        /* destination v3                    */
    .vs       = 0,        /* source v0                         */
    .use_mask = 0,        /* 0 = copy all VL lanes             */
    .mask     = 0,
}
```

Simulator implementation:

```c
case V_VMOV:
    for (int l = 0; l < vl; l++)
        if (ACTIVE(l)) n->v[I->vd][l] = n->v[I->vs][l];
    break;
```

Cycle cost: one pass of `passes = ceil(vl / HW_LANES)` cycles (same as any
elementwise op).

### 4.6.3 Why It Exists — Use Cases

**Register renaming / temporary preservation.** The compiler or programmer may
need to preserve a vector value across a sequence of operations that would
clobber it. `VMOV` is the register copy:

```c
/* Save v0 before it is used as scratch in an intermediate computation */
{ .op=V_VMOV, .vd=7, .vs=0 },   /* v7 = v0 (save) */
... use v0 as scratch ...
{ .op=V_VMOV, .vd=0, .vs=7 },   /* v0 = v7 (restore) */
```

**Predicated select (partial copy).** With `use_mask=1`, `VMOV` copies only the
lanes where the mask is 1, leaving unmasked lanes in `vd` unchanged. This is
subtly different from `VSEL` (which always writes all lanes from one of two
sources): `VMOV @m` is an in-place conditional update.

```c
/* v1[l] = v0[l] where m0[l]=1; v1[l] unchanged where m0[l]=0 */
{ .op=V_VMOV, .vd=1, .vs=0, .use_mask=1, .mask=0 }
```

**Layout after reduction.** After a reduction (`VREDSUM` etc.) produces a
scalar in `s[rd]`, `VBCAST` spreads it to a vector. Occasionally you have the
inverse need: copy an already-computed vector to a fresh register before
`VBCAST` overwrites the source register.

### 4.6.4 Assembly Syntax

```asm
VMOV v3, v0         ; v3 = v0 (all lanes)
VMOV @m0 v3, v0     ; v3 = v0 only where m0=1 (predicated)
```

---

## 4.7 VBCAST — Broadcast Scalar to All Lanes

### 4.7.1 What It Does

```
VBCAST vd, s[rs]
vd[l] = s[rs].f    for l = 0 .. VL-1  (subject to mask)
```

Takes a single scalar value from engine scalar register `s[rs]` (interpreted as
`float`) and replicates it into every lane of vector register `vd`. All `VL`
lanes receive the same value.

### 4.7.2 Instruction Fields

```c
{
    .op       = V_VBCAST,
    .vd       = 2,        /* destination vector register v2         */
    .rs       = 4,        /* source scalar register s4              */
    .use_mask = 0,        /* broadcast to all VL lanes              */
}
```

Simulator implementation:

```c
case V_VBCAST: {
    float s = n->s[I->rs].f;
    for (int l = 0; l < vl; l++)
        if (ACTIVE(l)) n->v[I->vd][l] = s;
    break;
}
```

Cycle cost: one pass of `passes = ceil(vl / HW_LANES)` cycles.

### 4.7.3 Why It Exists — The Scalar-Vector Bridge

Almost every vector kernel needs to apply a scalar parameter uniformly across
all lanes: multiply every element by a scale factor, subtract a mean, add a
bias, compare against a threshold. Without `VBCAST`, each of these would
require loading the scalar into all lanes of a vector one lane at a time.
`VBCAST` makes this a single instruction.

`VBCAST` is the **scalar-to-vector bridge**: it converts a scalar quantity
(computed by reductions, loaded from DRAM, or passed as an argument) into a
vector form that can participate in elementwise vector operations.

In LLVM IR terms, `VBCAST` corresponds to the idiom:

```llvm
%sv = insertelement <64 x float> poison, float %s, i32 0
%bcast = shufflevector <64 x float> %sv, <64 x float> poison,
                       <64 x i32> zeroinitializer
```

The `shufflevector` with a zeroinitializer splat is the direct IR equivalent,
and the NPU backend's instruction selector would emit `V_VBCAST` for it (see
`llvm_lowering.md`).

### 4.7.4 Data Flow Diagram

```
  s3 = 0.25  (single scalar value, float)

  VBCAST v2, s3  (VL = 8)

  Before:
  v2: [ ?? ][ ?? ][ ?? ][ ?? ][ ?? ][ ?? ][ ?? ][ ?? ]

  After:
  v2: [0.25][0.25][0.25][0.25][0.25][0.25][0.25][0.25]
        ↑     ↑     ↑     ↑     ↑     ↑     ↑     ↑
      lane0  lane1  lane2  lane3  lane4  lane5  lane6  lane7

  The same value s3=0.25 was written to every active lane.
```

### 4.7.5 Use Case 1: Scaling (Multiply by a Constant)

```c
/* Divide all elements by a denominator s4 */
{ .op=V_SRECIP, .rd=5, .rs1=4 },         /* s5 = 1 / s4          */
{ .op=V_VBCAST, .vd=2, .rs=5 },           /* v2[l] = 1/s4         */
{ .op=V_VMUL,   .vd=0, .vs1=0, .vs2=2 }, /* v0 = v0 * (1/s4)     */
```

### 4.7.6 Use Case 2: Zero Vector Initialization

```c
/* v1 = all zeros (for use as a comparison base or initial accumulator) */
{ .op=V_SFI,    .rd=6, .fimm=0.0 },       /* s6 = 0.0             */
{ .op=V_VBCAST, .vd=1, .rs=6 },            /* v1[l] = 0.0          */
```

### 4.7.7 Use Case 3: Threshold Comparison

```c
/* Compute ReLU: y = max(x, 0) via VMAX with zero broadcast */
{ .op=V_SFI,    .rd=7, .fimm=0.0 },        /* s7 = 0.0f            */
{ .op=V_VBCAST, .vd=3, .rs=7 },             /* v3[l] = 0.0f         */
{ .op=V_VMAX,   .vd=0, .vs1=0, .vs2=3 },   /* v0 = max(v0, 0)      */
```

### 4.7.8 Use Case 4: Softmax Normalization — The Canonical Example

The softmax computation requires `VBCAST` at two critical points: broadcasting
the per-row maximum (for numerical stability) and broadcasting the reciprocal
sum (for normalization). Here is the complete softmax kernel from Demo 2 of
`examples.c`, annotated to show every `VBCAST` use:

```c
static const vinst softmax_kernel[] = {
    /* --- Prologue: read argument block --- */
    { .op=V_SLW, .rd=2, .use_base=1, .rs=1, .imm=0 },  /* s2 = x base  */
    { .op=V_SLW, .rd=3, .use_base=1, .rs=1, .imm=4 },  /* s3 = y base  */
    { .op=V_SLW, .rd=4, .use_base=1, .rs=1, .imm=8 },  /* s4 = N       */

    /* --- Compute per-instance row pointer --- */
    { .op=V_SMUL,  .rd=5, .rs1=0, .rs2=4 },             /* s5 = id * N  */
    { .op=V_SMULI, .rd=5, .rs1=5, .imm=4 },             /* s5 *= 4 (bytes) */
    { .op=V_SADD,  .rd=6, .rs1=2, .rs2=5 },             /* s6 = x row ptr */
    { .op=V_SADD,  .rd=7, .rs1=3, .rs2=5 },             /* s7 = y row ptr */

    /* --- Body --- */
    { .op=V_VSETVL, .rd=8, .rs1=4, .use_base=1 },       /* VL = N         */
    { .op=V_VLOAD,  .vd=0, .use_base=1, .rs=6 },        /* v0 = x[row]    */

    /* Step 1: max for numerical stability */
    { .op=V_VREDMAX, .rd=9, .vs=0 },                    /* s9 = max(v0)   */
    { .op=V_VBCAST,  .vd=1, .rs=9 },                    /* v1 = max, broadcast */
    { .op=V_VSUB,    .vd=0, .vs1=0, .vs2=1 },           /* v0 = x - max   */
    { .op=V_VEXP,    .vd=0, .vs=0 },                    /* v0 = exp(x-max)*/

    /* Step 2: normalize */
    { .op=V_VREDSUM, .rd=10, .vs=0 },                   /* s10 = sum(exp) */
    { .op=V_SRECIP,  .rd=11, .rs1=10 },                 /* s11 = 1/sum    */
    { .op=V_VBCAST,  .vd=1, .rs=11 },                   /* v1 = 1/sum, broadcast */
    { .op=V_VMUL,    .vd=0, .vs1=0, .vs2=1 },           /* v0 = exp / sum */
    { .op=V_VSTORE,  .vs=0, .use_base=1, .rs=7 },       /* y = v0         */
    { .op=V_KEXIT },
};
```

The two `VBCAST` instructions are at the heart of the algorithm:

```
VREDMAX s9, v0       ; scalar: s9 = max over all lanes
VBCAST  v1, s9       ; vector: v1[l] = max (same for all l)
VSUB    v0, v0, v1   ; vector - vector: elementwise subtraction

VREDSUM s10, v0      ; scalar: s10 = sum of exp values
SRECIP  s11, s10     ; scalar: s11 = 1/sum
VBCAST  v1, s11      ; vector: v1[l] = 1/sum (same for all l)
VMUL    v0, v0, v1   ; elementwise normalize
```

The pattern is: **reduce** (vector → scalar), **transform scalar** (scalar →
scalar), **broadcast** (scalar → vector), **combine** (vector op vector).
This reduction-broadcast-combine idiom appears in layer normalization, batch
normalization, attention score computation, and many other ML kernels.

### 4.7.9 VBCAST vs. Loading a Constant from Memory

An alternative to `VBCAST` is to store a constant in memory and load it with
`VLOAD` — but this wastes memory bandwidth for something that is inherently a
scalar. `VBCAST` is strictly better: it reads a scalar register (zero-cost
internal wire) and replicates it to lanes in `ceil(VL/HW_LANES)` cycles. No
memory transaction occurs.

---

## 4.8 The Engine Scalar Helpers

The seven data-movement instructions above operate on vector registers and
memory. But before a `VLOAD` can load from the right address, the kernel must
compute that address. The NPU vector core includes a small set of scalar helper
instructions for exactly this purpose. They are not strictly "data movement" in
the vector sense, but they appear in every realistic kernel's prologue and loop
machinery.

### 4.8.1 Complete Scalar Helper Table

| Mnemonic | Operation                    | Purpose                          |
|----------|------------------------------|----------------------------------|
| `SLI`    | `s[rd] = imm`                | Load integer immediate into scalar |
| `SFI`    | `s[rd] = (float)fimm`        | Load float immediate into scalar  |
| `SADD`   | `s[rd] = s[rs1] + s[rs2]`    | Integer add (address arithmetic) |
| `SADDI`  | `s[rd] = s[rs1] + imm`       | Add immediate (stride advance)   |
| `SSUB`   | `s[rd] = s[rs1] - s[rs2]`    | Integer subtract (loop counter)  |
| `SMUL`   | `s[rd] = s[rs1] * s[rs2]`    | Integer multiply (index scaling) |
| `SMULI`  | `s[rd] = s[rs1] * imm`       | Multiply by immediate (sizeof)   |
| `SLW`    | `s[rd] = mem_i32[s[rs]+imm]` | Load 32-bit word from memory     |
| `SLF`    | `s[rd] = mem_f32[s[rs]+imm]` | Load float from memory           |
| `SRECIP` | `s[rd] = 1.0f / s[rs1]`      | Float reciprocal (for normalization) |
| `KID`    | `s[rd] = instance_id`        | Read instance ID (alias for s0)  |

All scalar helpers are single-cycle operations (they operate on the
scalar-register file and do not touch vector lanes). `SLW` and `SLF` do pay
`mem_latency` for their memory access.

### 4.8.2 Address Computation Pattern

The archetypal address computation for SIMT (one instance per row of a matrix):

```
; s0 = instance id (row index)
; s1 = args base (in SRAM)
; Goal: compute s6 = base + (row * stride_bytes)

SLW   s2, 0(s1)     ; s2 = tensor base address (from arg block)
SLW   s3, 4(s1)     ; s3 = N (row length)
SMUL  s4, s0, s3    ; s4 = instance_id * N  (element offset)
SMULI s4, s4, 4     ; s4 *= sizeof(float) = 4  (byte offset)
SADD  s6, s2, s4    ; s6 = base + byte_offset  (row pointer)
```

### 4.8.3 Loop Counter Pattern

For a kernel that iterates over chunks of a long vector:

```asm
; s2 = N (total elements), s3 = data pointer, s5 = VL
SLI    s5, 64         ; chunk size
loop:
  VSETVL  s8, s5      ; VL = min(s5, VLEN_MAX)
  VLOAD   v0, 0(s3)
  ... compute ...
  VSTORE  v0, 0(s3)
  SADDI   s3, s3, 256 ; advance pointer: 64 floats * 4 bytes = 256 bytes
  SSUB    s2, s2, s5  ; remaining -= chunk
  BRNZ    s2, loop
  KEXIT
```

`SADDI`, `SSUB`, and `BRNZ` together implement a counted loop without any
vector operations. They are cheap (1 cycle each) and execute on the scalar-side
of the vector engine while the vector functional units are idle between memory
transactions.

### 4.8.4 SRECIP — Float Reciprocal for Normalization

`SRECIP` is a specialized scalar operation: `s[rd] = 1.0f / s[rs1]`. It exists
because the normalization pattern (divide by a sum or count) is ubiquitous in
ML and is the only float operation that the scalar side needs to perform.
Everything else (exp, log, sqrt, elementwise math) is done on the vector side.

```c
/* Compute 1/N and broadcast for averaging */
{ .op=V_VREDSUM, .rd=10, .vs=0 },      /* s10 = sum of v0           */
{ .op=V_SRECIP,  .rd=11, .rs1=10 },    /* s11 = 1/sum               */
{ .op=V_VBCAST,  .vd=2,  .rs=11  },    /* v2[l] = 1/sum             */
{ .op=V_VMUL,    .vd=0, .vs1=0, .vs2=2 }, /* v0 = v0 / sum         */
```

---

## 4.9 Combining the Instructions: Complete Worked Examples

### 4.9.1 Example: Attention Score Normalization (Scaled Dot-Product)

Given a query vector q and key vectors k0..k3, compute attention weights via
softmax of dot products. This combines scalar loads, VBCAST, VMUL, VREDSUM,
and SRECIP:

```c
/* Assume:
 *   s2 = base of q (VL floats, in SRAM)
 *   s3 = base of K matrix (VL * 4 floats per key, in SRAM)
 *   s4 = VL (= head_dim)
 *   s5 = scale factor = 1/sqrt(head_dim), pre-computed by control core
 *   s6 = output base for attention weights
 */
{ .op=V_VSETVL,  .rd=0, .rs1=4, .use_base=1 },    /* VL = head_dim      */

/* Load query */
{ .op=V_VLOAD,   .vd=0, .use_base=1, .rs=2 },      /* v0 = q             */

/* Compute dot(q, k0): load key 0, elementwise mul, sum */
{ .op=V_VLOAD,   .vd=1, .use_base=1, .rs=3, .imm=0 },    /* v1 = k0      */
{ .op=V_VMUL,    .vd=2, .vs1=0, .vs2=1 },                 /* v2 = q*k0    */
{ .op=V_VREDSUM, .rd=10, .vs=2 },                          /* s10 = dot    */

/* Compute dot(q, k1), dot(q,k2), dot(q,k3) similarly ... */
/* (storing results in s11, s12, s13) */

/* Scale: scores = dots * (1/sqrt(d_k)) */
/* (s5 already holds scale = 1/sqrt(head_dim), loaded from arg block) */
/* Collect the four scores into v3 via VSETVL 4 + individual lane writes:  */
/* (In practice, compile loop unroll or use the tensor engine dot product)  */

/* Softmax the four attention scores */
{ .op=V_VSETVL,  .rd=0, .imm=4 },    /* now VL=4 for the 4 scores */
/* ... VBCAST scale, VMUL to scale, then VEXP, VREDSUM, SRECIP, VBCAST, VMUL ... */
```

This illustrates the alternation between `VL=head_dim` (for loading q and k)
and `VL=4` (for the softmax over 4 scores). `VSETVL` makes this switching cost
nothing.

### 4.9.2 Example: Layer Normalization

Layer normalization computes `(x - mean) / (std + eps)` elementwise. It needs
`VLOAD`, `VREDSUM`, `SRECIP`, `VBCAST`, `VSUB`, `VMUL`, and `VSTORE`:

```asm
; s2 = input ptr (SRAM), s3 = output ptr (SRAM), s4 = N, s5 = eps
VSETVL  s0, s4            ; VL = N

; Step 1: compute mean
VLOAD   v0, 0(s2)         ; v0 = x
VREDSUM s6, v0            ; s6 = sum(x)
SRECIP  s7, s4            ; s7 = 1/N   (integer N loaded, used as float)
SFI     s8, ...           ; (or multiply by pre-stored 1/N)
SMULI   s6, s6, s7        ; BUG: this is integer — need float arithmetic
; (in practice: SRECIP gives 1/N as float; multiply sum * (1/N) using scalar VMUL trick)
; Workaround: use VL=1 VMUL with broadcast, then VREDSUM:
VBCAST  v1, s7            ; v1[l] = 1/N
VMUL    v2, v0, v1         ; v2[l] = x[l] / N  (partial mean contribution)
VREDSUM s6, v2             ; s6 = sum(x[l]/N) = mean

; Step 2: subtract mean
VBCAST  v1, s6             ; v1[l] = mean
VSUB    v0, v0, v1         ; v0 = x - mean

; Step 3: compute variance
VMUL    v2, v0, v0         ; v2 = (x - mean)^2
VREDSUM s8, v2             ; s8 = sum((x-mean)^2)
VBCAST  v3, s7             ; v3[l] = 1/N
VMUL    v2, v2, v3         ; ... (scale by 1/N)
VREDSUM s8, v2             ; s8 = variance
; ... add eps, VSQRT, SRECIP for stddev ...

; Step 4: normalize
VBCAST  v3, s_inv_std      ; v3[l] = 1/std
VMUL    v0, v0, v3         ; v0 = (x - mean) / std
VSTORE  v0, 0(s3)          ; store result
KEXIT
```

Even in this simplified sketch, the interleaving of `VBCAST` (scalar-to-vector)
and `VREDSUM` (vector-to-scalar) is the defining rhythm of the computation.

### 4.9.3 Example: Predicated Update with VGATHER + VSCATTER

Histogram update: for each input lane `l`, increment `count[input[l]]` by 1.

```asm
; v0 = indices (integer-valued floats, converted via VCVT)
; s2 = base of count array in SRAM
VSETVL   s0, 16            ; VL = 16
VLOAD    v0, 0(s2_input)   ; load 16 input values
VCVT     v0, v0            ; convert to integer indices (rounds to nearest int8/int32)
VGATHER  v1, v0, s3        ; v1[l] = count[input[l]]
SFI      s4, 1.0           ; s4 = 1.0f
VBCAST   v2, s4            ; v2[l] = 1.0
VADD     v1, v1, v2        ; v1[l] = count[input[l]] + 1
VSCATTER v1, v0, s3        ; count[input[l]] = v1[l]
KEXIT
```

This pattern appears in token frequency counting, softmax denominator
pre-computation for sparse attention, and similar non-contiguous update
workloads. Note the write-conflict hazard: if two lanes have the same index,
only one increment will stick. For correctness this requires ensuring unique
indices per kernel instance, or serializing conflicting lanes via predication.

---

## 4.10 Instruction Summary Reference

The table below consolidates the full Section 9 instruction set with simulator
fields and cycle costs for quick reference.

### 4.10.1 Core Section 9 Instructions

| Mnemonic    | Key `vinst` Fields                     | Simulator Behavior                        | Cycle Cost                 |
|-------------|----------------------------------------|-------------------------------------------|----------------------------|
| `VSETVL`    | `.rd`, `.rs1` or `.imm`, `.use_base`   | `n->vl = clamp(req, 0, VLEN_MAX); s[rd] = new vl` | 1              |
| `VLOAD`     | `.vd`, `.rs`, `.imm`, `.use_base`      | `v[vd][l] = mem[addr + l*4]`             | `mem_latency(addr)` + `ceil(vl/HW_LANES)` |
| `VSTORE`    | `.vs`, `.rs`, `.imm`, `.use_base`, `.use_mask`, `.mask` | `mem[addr+l*4] = v[vs][l]` where `ACTIVE(l)` | `mem_latency(addr)` + `ceil(vl/HW_LANES)` |
| `VGATHER`   | `.vd`, `.vs1` (indices), `.rs`, `.imm`, `.use_base` | `v[vd][l] = mem[addr + v[vs1][l]*4]`   | `vl` (one per lane)        |
| `VSCATTER`  | `.vs` (data), `.vs1` (indices), `.rs`, `.imm`, `.use_base` | `mem[addr + v[vs1][l]*4] = v[vs][l]`  | `vl` (one per lane)        |
| `VMOV`      | `.vd`, `.vs`, `.use_mask`, `.mask`     | `v[vd][l] = v[vs][l]` where `ACTIVE(l)` | `ceil(vl/HW_LANES)`        |
| `VBCAST`    | `.vd`, `.rs`, `.use_mask`, `.mask`     | `v[vd][l] = s[rs].f` where `ACTIVE(l)` | `ceil(vl/HW_LANES)`        |

### 4.10.2 Scalar Helper Instructions

| Mnemonic  | Key Fields                   | Operation                      | Cycle |
|-----------|------------------------------|--------------------------------|-------|
| `SLI`     | `.rd`, `.imm`                | `s[rd].i = imm`                | 1     |
| `SFI`     | `.rd`, `.fimm`               | `s[rd].f = (float)fimm`        | 1     |
| `SADD`    | `.rd`, `.rs1`, `.rs2`        | `s[rd].i = s[rs1].i + s[rs2].i` | 1   |
| `SADDI`   | `.rd`, `.rs1`, `.imm`        | `s[rd].i = s[rs1].i + imm`    | 1     |
| `SSUB`    | `.rd`, `.rs1`, `.rs2`        | `s[rd].i = s[rs1].i - s[rs2].i` | 1   |
| `SMUL`    | `.rd`, `.rs1`, `.rs2`        | `s[rd].i = s[rs1].i * s[rs2].i` | 1   |
| `SMULI`   | `.rd`, `.rs1`, `.imm`        | `s[rd].i = s[rs1].i * imm`    | 1     |
| `SLW`     | `.rd`, `.rs`, `.imm`, `.use_base` | `s[rd].i = mem_i32[s[rs]+imm]` | 1 + `mem_latency` |
| `SLF`     | `.rd`, `.rs`, `.imm`, `.use_base` | `s[rd].f = mem_f32[s[rs]+imm]` | 1 + `mem_latency` |
| `SRECIP`  | `.rd`, `.rs1`                | `s[rd].f = 1.0f / s[rs1].f`   | 1     |
| `KID`     | `.rd`                        | `s[rd].i = instance_id`        | 1     |

### 4.10.3 Address Modes Summary

| `use_base` value | Effective address formula    | Typical use                           |
|------------------|------------------------------|---------------------------------------|
| 0                | `(uint32_t)imm`              | Fixed compile-time address            |
| 1                | `(uint32_t)(s[rs].i + imm)`  | Dynamic base + fixed field offset     |

---

## 4.11 Common Pitfalls and Debugging Tips

### 4.11.1 Forgetting VSETVL

The most common mistake: writing a kernel that begins immediately with `VLOAD`
without setting VL first. VL is 0 at kernel entry; `VLOAD` with VL=0 loads 0
lanes and writes nothing to the vector register. Subsequent instructions operate
on stale/zero data with no error.

**Debug:** Enable the simulator's trace mode (`n->trace = 1`). Every instruction
prints its VL at execution time. If you see `vl=0` on your `VLOAD`, the problem
is a missing `VSETVL`.

### 4.11.2 DRAM vs. SRAM Addresses in Arg Block

A kernel receives addresses in its arg block at `s1`. If those addresses are
DRAM addresses (bit 31 = 0), every `VLOAD` costs 100 cycles. If they are SRAM
addresses (bit 31 = 1), every `VLOAD` costs 1 cycle. The address encoding is
the only thing that changes; the kernel binary is identical.

**Debug:** Check `npu_print_stats()` after running. If `vector_cycles` is much
larger than expected, the kernel is probably loading from DRAM. Confirm that the
arg block contains `SCRATCH_BASE | offset` addresses, not bare DRAM addresses.

### 4.11.3 VGATHER with Duplicate Indices

As noted in Section 4.5.5, `VSCATTER` to duplicate indices produces undefined
behavior on parallel hardware. In the simulator it silently uses the last-lane
value. Add an assertion or sort-and-unique pass on the index vector before
scatter if uniqueness cannot be guaranteed by construction.

### 4.11.4 VSETVL Clamping — Requesting More Than VLEN_MAX

Requesting `VSETVL s0, 512` on a hardware with `VLEN_MAX=256` silently clamps
to 256. The returned value in `s0` will be 256, not 512. A kernel that uses the
request value (512) rather than the returned value (s0 = 256) will have a
stride/count bug. Always use the **returned** VL in subsequent pointer arithmetic:

```asm
VSETVL s8, s4           ; s8 = actual VL (clamped), not s4!
VLOAD  v0, 0(s2)        ; loads s8 lanes (the clamped value)
SMULI  s9, s8, 4        ; advance by s8*4 bytes (use returned VL)
SADD   s2, s2, s9       ; pointer += actual_vl * sizeof(float)
```

### 4.11.5 Mask Not Set Before Predicated Store

`VSTORE` with `use_mask=1` uses mask register `m[mask]`. If that mask register
was never written (e.g., `VCMP` was not called), its value is zero (all zeros
at kernel initialization), and no lanes will be stored. The destination memory
will be unchanged. Always verify that the mask was initialized before the first
predicated operation.

---

## 4.12 Section 9 in the LLVM IR Lowering

For readers interested in how these instructions relate to the eventual LLVM
backend (documented in `llvm_lowering.md`):

| Section 9 instruction | LLVM IR construct                              |
|-----------------------|------------------------------------------------|
| `VSETVL`             | `@llvm.riscv.vsetvl` intrinsic (scalable vector model) |
| `VLOAD`              | `@llvm.npu.vload(ptr)` intrinsic → `V_VLOAD`  |
| `VSTORE`             | `@llvm.npu.vstore(ptr, <N x float>)` → `V_VSTORE` |
| `VGATHER`            | `@llvm.masked.gather` → `V_VGATHER`           |
| `VSCATTER`           | `@llvm.masked.scatter` → `V_VSCATTER`         |
| `VMOV`               | Register allocation / phi elimination          |
| `VBCAST`             | `shufflevector` with `zeroinitializer` splat → `V_VBCAST` |
| `SLW` / `SLF`        | `load i32` / `load float` from arg struct      |
| `SADD` / `SMUL` etc. | `add i64` / `mul i64` in scalar IR             |
| `SRECIP`             | `fdiv float 1.0, %x` → `V_SRECIP`             |

The instruction selector in the NPU LLVM backend would pattern-match these IR
constructs and emit the corresponding `vinst` struct. The `npu_encode_vinst`
function then converts the struct to the 32-byte binary encoding that the
simulator (and future hardware) loads.

---

## 4.13 Chapter Summary

Section 9 of the vector core ISA provides seven fundamental instructions that
every kernel uses:

1. **`VSETVL`** — sets the active lane count. Must be the first instruction.
   Enables length-agnostic, loop-tail-handling kernels.

2. **`VLOAD`** — contiguous burst load of VL floats from memory into a vector
   register. Extremely fast from SRAM (1 cycle), expensive from DRAM (100 cycles).

3. **`VSTORE`** — contiguous burst store of VL floats from a vector register to
   memory. Supports per-lane predication via the mask register.

4. **`VGATHER`** — per-lane indirect load; each lane uses its own index. Costs
   VL cycles; use sparingly, preferably with data in SRAM.

5. **`VSCATTER`** — per-lane indirect store; inverse of VGATHER. Same cost model.
   Beware duplicate-index write conflicts.

6. **`VMOV`** — register-to-register copy, with optional predicated (partial)
   update.

7. **`VBCAST`** — broadcast a scalar to all lanes. The bridge between scalar
   computation (reductions, argument values, scale factors) and vector registers.

Supporting these are the scalar helpers (`SLI`, `SFI`, `SADD`, `SADDI`, `SSUB`,
`SMUL`, `SMULI`, `SLW`, `SLF`, `SRECIP`, `KID`) that compute addresses and
single-value arithmetic in the kernel's prologue and loop control logic.

Together, these eleven instruction types (seven vector + scalar helpers) are
sufficient to write any ML data-movement and reduction pattern that the vector
core needs to support. The compute primitives (Section 10) and reductions
(Section 11) then operate on the data this section moves into and out of the
vector register file.

---

*Next: Chapter 5 — Compute Primitives (Section 10). Covers elementwise
arithmetic (VADD, VSUB, VMUL, VDIV, VMAX, VMIN, VFMA), comparisons (VCMP,
VSEL), transcendentals (VEXP, VLOG, VSQRT, VRSQRT, VRECIP), conversions
(VCVT), and sign operations (VABS, VNEG) — with cycle costs, hardware
considerations, and canonical ML usage patterns for each.*
