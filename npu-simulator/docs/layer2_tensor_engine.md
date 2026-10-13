# Layer 2 — The Tensor Engine

The tensor engine is the dense-math accelerator of the NPU. It handles matrix
multiplication, memory tiling, activations, and DMA — the heavy arithmetic of
a neural network's forward pass.

It needs **no KLAUNCH**. Its instructions live in the same `cinst[]` stream as
the control core. When the control core fetches an opcode `>= T_CFG_SHAPE`, it
dispatches to the tensor engine instead of the scalar ALU.

```
npu_run() dispatch:
  op < T_CFG_SHAPE   →  scalar ALU    (Layer 1)
  op >= T_CFG_SHAPE  →  tensor engine (Layer 2)
```

---

## 2.1 Tile Registers

8 tile registers `t0`–`t7`. Each holds up to `64 × 64 = 4096` floats (16 KB).
Actual working dimensions are set at runtime by `CFG.SHAPE`.

```
npu_t state:
  float tile[8][64 * 64];    8 tiles, 16 KB each
  int   cfg_M, cfg_K, cfg_N; set by CFG.SHAPE
```

Tile registers are **not general-purpose** — only tensor engine instructions
can read or write them. In LLVM IR they are `i64` constants 0–7 (the tile
number). There is no tile register allocator; the programmer assigns tiles
explicitly.

---

## 2.2 Instruction Set

### Config

```
CFG.SHAPE  M, K, N    Set tile working dimensions before any LOAD/MATMUL/MMA.
                       M = output rows, K = shared dim, N = output cols.
                       M/K/N are GPR values — can be runtime.
```

### Compute

```
ACC.CLR  dt            Zero out tile dt. Required before starting an MMA loop.

MATMUL   dt, A, B      dt  = A × B     overwrite — no ACC.CLR needed
MMA      dt, A, B      dt += A × B     accumulate — must ACC.CLR first
DOT      dt, A, B      dt += dot(A,B)  vector dot product, accumulate
```

### Data movement

```
LOAD     dt, base, count   Load count floats from address into tile dt.
                            Row-major. Use for activations (A matrix).

LOADW    dt, base, count   Load weight tile — B matrix stored TRANSPOSED in
                            memory. Columns of B become rows for cache-friendly
                            inner-product access.

STORE    base, dt, count   Write tile dt to memory, row-major.

DMA.IN   dst, src, bytes   Async copy DRAM → SRAM scratchpad. Non-blocking.
                            Issue FENCE before reading the data.

DMA.OUT  dst, src, bytes   Async copy SRAM → DRAM. Non-blocking.
                            Issue FENCE before caller reads DRAM.
```

### Activations

```
RELU     dt, a         Element-wise max(0, x). Always in-place: RELU dt, dt.

SCALE    dt, a, f      Multiply every element by float f.
                       f is a COMPILE-TIME constant — burned into instruction
                       word (fimm field, bytes 16–23).

BCAST    dt, xN        Fill every element of dt with float from GPR xN.
                       Control core does LI xN, <bits> first, then BCAST.
                       xN holds the IEEE 754 bit pattern of the float.
```

### Synchronization

```
FENCE      Flush all pending tensor-engine ops. Use after DMA.IN before LOAD,
           and after STORE before DMA.OUT.

WAIT       Stronger — wait until tensor engine is fully idle. Use before
           reading tile output with scalar LW instructions.

BARRIER    Global barrier across all vector cores. Used between KLAUNCH phases.
```

---

## 2.3 SCALE vs BCAST — Two Ways to Put a Float into a Tile

They look similar but encode the float completely differently.

| | SCALE | BCAST |
|---|---|---|
| What it does | Multiply every element by f | Fill every element with f |
| Float encoding | `fimm` field in instruction word | GPR register (via LI first) |
| Simulator reads from | `I->fimm` (bytes 16–23) | `getx(n, I->rs1)` |
| LLVM lowering | `TargetConstant:i64` (IEEE 754 double bits) | `ISD::Constant:i64` → LI |

```
; SCALE t0, t0, 0.5
SCALE t0, t0, 0.5         factor is in the instruction itself, no GPR used

; BCAST t1, 1.0
LI    x5, 1065353216      0x3F800000 = 1.0f bit pattern
BCAST t1, x5              simulator reads float from x5
```

Both require a **compile-time constant** float. A runtime variable aborts
compilation.

---

## 2.4 Complete Example — Single MATMUL

**Task:** multiply two 2×2 matrices.

```
A = | 1  2 |    B = | 5  6 |    C = A × B = ?
    | 3  4 |        | 7  8 |
```

**Memory before running:**

```
DRAM 0x1000 : [ 1.0, 2.0, 3.0, 4.0 ]      A, row-major
DRAM 0x2000 : [ 5.0, 7.0, 6.0, 8.0 ]      B, stored TRANSPOSED for LOADW
DRAM 0x3000 : [ ?,   ?,   ?,   ?   ]      C output (empty)
```

**Why is B transposed?** Computing C[i][j] = row_i(A) · col_j(B) requires
walking a column of B. Storing B transposed means that column becomes a row —
one contiguous read instead of a stride.

**Instruction stream:**

```
[0]  CFG.SHAPE  2, 2, 2
[1]  LOAD       t0, 0x1000, 4
[2]  LOADW      t1, 0x2000, 4
[3]  MATMUL     t2, t0, t1
[4]  STORE      t2, 0x3000, 4
[5]  HALT
```

**Step-by-step tile state:**

```
[0] CFG.SHAPE 2, 2, 2
    Engine knows: M=2, K=2, N=2. No tile changes yet.

[1] LOAD t0
    Reads [1, 2, 3, 4], reshapes to 2×2:
    t0 = | 1.0  2.0 |
         | 3.0  4.0 |

[2] LOADW t1
    Reads [5, 7, 6, 8] (B stored transposed):
    t1 = | 5.0  7.0 |   columns of B, stored as rows
         | 6.0  8.0 |

[3] MATMUL t2, t0, t1
    C[0][0] = [1,2]·[5,7] =  5+14 = 19
    C[0][1] = [1,2]·[6,8] =  6+16 = 22
    C[1][0] = [3,4]·[5,7] = 15+28 = 43
    C[1][1] = [3,4]·[6,8] = 18+32 = 50

    t2 = | 19.0  22.0 |
         | 43.0  50.0 |

[4] STORE t2
    DRAM 0x3000 ← [ 19.0, 22.0, 43.0, 50.0 ]
```

---

## 2.5 Complete Example — MMA K-block Loop

**Task:** 2×4 multiplied by 4×2, with tile_K=2. K=4 does not fit in one tile,
so it is split into 2 blocks.

```
A (2×4)              B (4×2)
| 1  2  3  4 |      | 1  2 |
| 5  6  7  8 |      | 3  4 |
                    | 5  6 |
                    | 7  8 |
Expected C = | 50   60 |
             | 114  140 |
```

**Memory:**

```
0x1000  A_block0 (A cols 0-1): [ 1, 2, 5, 6 ]
0x1010  A_block1 (A cols 2-3): [ 3, 4, 7, 8 ]
0x2000  B_block0 (B rows 0-1, transposed): [ 1, 3, 2, 4 ]
0x2010  B_block1 (B rows 2-3, transposed): [ 5, 7, 6, 8 ]
0x3000  C output
```

**Instruction stream:**

```
[0]  CFG.SHAPE  2, 2, 2
[1]  ACC.CLR    t2              zero accumulator BEFORE loop
[2]  LOAD       t0, 0x1000, 4
[3]  LOADW      t1, 0x2000, 4
[4]  MMA        t2, t0, t1     t2 += block 0 partial
[5]  LOAD       t0, 0x1010, 4
[6]  LOADW      t1, 0x2010, 4
[7]  MMA        t2, t0, t1     t2 += block 1 partial
[8]  STORE      t2, 0x3000, 4
[9]  HALT
```

**Tile state at each MMA:**

```
[1]  ACC.CLR t2    →   t2 = | 0   0 |
                             | 0   0 |

[2]  LOAD t0  →  t0 = | 1  2 |    [3] LOADW t1  →  t1 = | 1  3 |
                       | 5  6 |                           | 2  4 |

[4]  MMA t2, t0, t1   (A[:,0:2] × B[0:2,:])
     t2[0][0] += [1,2]·[1,3] = 1+6   =  7
     t2[0][1] += [1,2]·[2,4] = 2+8   = 10
     t2[1][0] += [5,6]·[1,3] = 5+18  = 23
     t2[1][1] += [5,6]·[2,4] = 10+24 = 34
     t2 = |  7   10 |
          | 23   34 |

[5]  LOAD t0  →  t0 = | 3  4 |    [6] LOADW t1  →  t1 = | 5  7 |
                       | 7  8 |                           | 6  8 |

[7]  MMA t2, t0, t1   (A[:,2:4] × B[2:4,:])
     t2[0][0] +=  7 + [3,4]·[5,7] =  7+43  =  50
     t2[0][1] += 10 + [3,4]·[6,8] = 10+50  =  60
     t2[1][0] += 23 + [7,8]·[5,7] = 23+91  = 114
     t2[1][1] += 34 + [7,8]·[6,8] = 34+106 = 140
     t2 = |  50   60 |
          | 114  140 |  ✓
```

---

## 2.6 MATMUL vs MMA — When to Use Each

| Situation | Use |
|---|---|
| K fits in one tile, single multiply | `MATMUL` |
| K too large, split into blocks | `ACC.CLR` + `MMA` loop |
| Adding multiple matrices into one result | `ACC.CLR` + `MMA` |
| Overwriting with a fresh result every time | `MATMUL` |

**Most common mistake:** forgetting `ACC.CLR` before an MMA loop. The first
MMA accumulates into garbage instead of zero.

---

## 2.7 Activation Pipeline

```
CFG.SHAPE  M, K, N
LOAD       t0, input_ptr, count
LOADW      t1, weight_ptr, count
MATMUL     t2, t0, t1

RELU       t2, t2             max(0, x), in-place
SCALE      t2, t2, 0.0039     multiply by constant (e.g. 1/256)

STORE      t2, output_ptr, count
```

---

## 2.8 DMA Staging Pattern

DRAM = 100-cycle latency. SRAM scratchpad (`SCRATCH_BASE = 0x80000000`) =
1-cycle latency. DMA moves data between them asynchronously.

```
DMA.IN  scratch_A, dram_A, bytes   async copy (returns immediately)
DMA.IN  scratch_B, dram_B, bytes   overlaps with above
FENCE                               wait for both to land in SRAM

CFG.SHAPE  M, K, N
LOAD       t0, scratch_A, count    fast SRAM access (1 cycle)
LOADW      t1, scratch_B, count
MATMUL     t2, t0, t1
STORE      scratch_C, t2, count

DMA.OUT    dram_C, scratch_C, bytes   async write-back
FENCE                                  ensure write-back completes
```

Without FENCE between `DMA.IN` and `LOAD`, the tensor engine may read stale
data the DMA has not yet written.

---

## 2.9 Known Gaps

| Gap | Detail |
|---|---|
| `T_CONV`, `T_POOL` | Opcodes exist but are stubs in simulator. No LLVM intrinsics. Build convolution from tiled MMA; pooling from vector-core reductions. |
| `T_CFG_DTYPE`, `T_QUANT`, `T_DEQUANT` | In enum, not implemented. |
| `T_CFG_STRIDE` | In enum, no LLVM intrinsic. |
| Tile count | Hard limit of 8. No spill or reload. |
| Runtime scale/bcast float | Must be compile-time — runtime value aborts `LowerIntrinsicCall`. |
| Runtime tile number | Must be `i64` literal 0–7 — variable aborts `getTileConst()`. |
