# Chapter 7: Reductions — Collapsing Vectors to Scalars

> **ISA Reference:** Section 11 of `npu-simulator/docs/fully_prog_npu.md`
> **Simulator source:** `npu-simulator/src/npu.c`, lines 260–265
> **Worked example:** `npu-simulator/examples/examples.c`, `demo_softmax()` and `demo_multicore()`

---

## 7.1 Why Reductions Are a First-Class Concern

Every neural network training or inference pass that involves probability distributions,
normalization, or loss computation eventually needs to answer one of these questions about
an entire vector of numbers:

- "What is the total?" (softmax denominator, mean, cross-entropy)
- "What is the largest?" (numerically stable softmax, output argmax)
- "What is the smallest?" (range clipping, quantization scale selection)
- "Are all these flags true?" (masking consistency checks)
- "Is any flag set?" (early-exit conditions)

On a scalar CPU these operations are O(N) sequential iterations — a simple loop. On a
vector processor, the lanes of a register all hold data simultaneously, but producing one
answer from all of them requires a _reduction_ across lanes. Without hardware support you
would write the loop yourself in vector assembly, which is both verbose and slow: each
iteration of such a loop would process a fraction of the register file in parallel, then
feed the partial result back — paying data hazard stalls and register pressure on every
step.

The NPU vector core elevates reductions to instructions. The reason this matters enormously
in practice is illustrated by a single statistic from the softmax kernel in
`examples/examples.c`: that kernel executes exactly two reductions (one `VREDMAX` and one
`VREDSUM`) out of nineteen instructions total, yet those two instructions are the
computational bottleneck in any implementation that lacks them — they each touch every lane
of `vl` elements. Hardwiring the reduction operation into the execution unit lets the
hardware exploit a binary-tree schedule internally (more on this in Section 7.5), bringing
the latency from O(vl) sequential operations down to O(log(vl)) stages.

### A Note on Scope: Reductions vs. Prefix Sums

The NPU draws a deliberate distinction:

- **True reductions** (`VREDSUM`, `VREDMAX`, `VREDMIN`, `VREDAND`, `VREDOR`): _many_ lanes
  in, _one_ scalar out. The result is written to a scalar register `s[rd]`, not a vector
  register. The vector operand is unchanged.
- **Prefix scan** (`VSCAN`): many lanes in, _many_ lanes out. The result is a vector where
  each output lane holds the running total up to and including that input lane. The result
  stays in the vector register file.

Both operations collapse information across lanes, but they serve different purposes.
Reductions are for computing a global statistic; prefix scans are for computing cumulative
distributions or positions.

---

## 7.2 The Instruction Set — Complete Reference

### 7.2.1 VREDSUM — Sum Reduction

**Syntax:** `VREDSUM rd, vs`

**Semantics:**
```
s[rd].f = vs[0] + vs[1] + vs[2] + ... + vs[vl-1]
```

**Simulator implementation** (`npu.c`, line 260):
```c
case V_VREDSUM: {
    float a = 0;
    for (int l = 0; l < vl; l++) a += n->v[I->vs][l];
    n->s[I->rd].f = a;
    break;
}
```

The simulator implements this as a sequential accumulation (straightforward for
correctness). Hardware implementations use a logarithmic reduction tree — see Section 7.5.

**Result type:** The result is stored in the `.f` (float) field of the scalar register union
(`sreg_t`). This is consistent with the convention that vector registers hold `float`
values throughout the NPU.

**Primary use cases:**
- Softmax denominator: `sum(exp(x))` after the exp pass
- Layer normalization mean: `sum(x) / vl`
- Cross-entropy loss accumulation
- Dot product accumulation after a lane-wise multiply

**Example — computing a mean:**
```c
/* Compute mean(v0) and store in s5 */
{ V_VREDSUM, .rd=5, .vs=0 },           /* s5 = sum of all lanes    */
{ V_SLI,     .rd=6, .imm=16 },         /* s6 = (int)vl             */
/* Convert count to float, divide — or precompute 1/vl as a scalar */
{ V_SFI,     .rd=7, .fimm=1.0/16.0 }, /* s7 = 1/vl (known at compile time) */
/* Now multiply: result = s5 * s7 (both scalar — no vector needed)  */
```

Note that the division by `vl` to compute the mean is a scalar operation on `s[5]` after
the reduction completes. The vector core's scalar sub-unit (`SRECIP`, `SMUL`, etc.) handles
this without returning to the host.

---

### 7.2.2 VREDMAX — Maximum Reduction

**Syntax:** `VREDMAX rd, vs`

**Semantics:**
```
s[rd].f = max(vs[0], vs[1], ..., vs[vl-1])
```

**Simulator implementation** (`npu.c`, line 261):
```c
case V_VREDMAX: {
    float a = -INFINITY;
    for (int l = 0; l < vl; l++)
        if (n->v[I->vs][l] > a) a = n->v[I->vs][l];
    n->s[I->rd].f = a;
    break;
}
```

The initial accumulator is `-INFINITY` — not `-FLT_MAX`. This is intentional: if all
lanes contain `-INFINITY` (which can happen after a log of a zero), the result is still
mathematically correct (`-INFINITY`), not an artifact of the implementation.

**Primary use cases:**
- Numerically stable softmax (subtract max before taking exp)
- Activations that clip to a maximum value
- Argmax: run VREDMAX, then scan to find which lane matched

**The stability argument:** The naive softmax formula is:

```
y[i] = exp(x[i]) / sum(exp(x[j]) for all j)
```

If any `x[i]` is large (e.g., 200), `exp(200)` overflows IEEE 754 float32
(`~1.6e38` is the limit; `exp(89)` is already near it). The standard remedy is:

```
m = max(x)
y[i] = exp(x[i] - m) / sum(exp(x[j] - m) for all j)
```

This is mathematically identical (the `exp(-m)` factors cancel), but now the maximum
argument to `exp` is exactly zero, and all other arguments are negative — no overflow
possible. `VREDMAX` is the hardware instruction that enables this pattern with a single
instruction rather than a scalar loop.

---

### 7.2.3 VREDMIN — Minimum Reduction

**Syntax:** `VREDMIN rd, vs`

**Semantics:**
```
s[rd].f = min(vs[0], vs[1], ..., vs[vl-1])
```

**Simulator implementation** (`npu.c`, line 262):
```c
case V_VREDMIN: {
    float a = INFINITY;
    for (int l = 0; l < vl; l++)
        if (n->v[I->vs][l] < a) a = n->v[I->vs][l];
    n->s[I->rd].f = a;
    break;
}
```

The initial accumulator is `+INFINITY` for the symmetric reason: any lane value is smaller
than `+INFINITY`, so a vector of all-`+INFINITY` gives the correct result.

**Primary use cases:**
- Quantization scale selection: `scale = (max - min) / 255.0`, requiring both `VREDMAX`
  and `VREDMIN` to determine the integer range
- Range normalization for display or debugging
- Numerically stable log-sum-exp from the minimum direction (less common than max)
- Lower-bound clamping for activation functions

**Quantization scale pattern:**
```c
{ V_VREDMAX, .rd=4, .vs=0 },   /* s4 = max(v0)               */
{ V_VREDMIN, .rd=5, .vs=0 },   /* s5 = min(v0)               */
{ V_SSUB,    .rd=6, .rs1=4, .rs2=5 },  /* s6 = max - min      */
/* s6 / 255 = scale (scalar division via SRECIP if 1/(max-min) is needed) */
```

---

### 7.2.4 VREDAND — Bitwise AND Reduction

**Syntax:** `VREDAND rd, vs`

**Semantics:**
```
s[rd].i = (int64_t)vs[0] & (int64_t)vs[1] & ... & (int64_t)vs[vl-1]
```

**Simulator implementation** (`npu.c`, line 263):
```c
case V_VREDAND: {
    int64_t a = ~0LL;   /* all bits set — identity element for AND */
    for (int l = 0; l < vl; l++) a &= (int64_t)n->v[I->vs][l];
    n->s[I->rd].i = a;
    break;
}
```

**Critical detail:** The vector register stores `float` values, but `VREDAND` reinterprets
those float bits as integers before ANDing. This is a bitwise operation, not an arithmetic
one. The result is stored in the `.i` (integer) field of the scalar register. The
simulator casts `float` to `int64_t` directly — this is a reinterpretation of bits, not
a numeric conversion (the floating-point value `1.0f` has bit pattern `0x3F800000`; that
is what gets ANDed, not the integer `1`).

This design choice — operating on float bit patterns — allows the vector register file to
be uniform (everything is `float`) while still supporting bitwise operations when needed.
A compiler targeting this ISA would need to be aware of this distinction.

**Primary use cases:**
- "All flags set" test: if you have computed `VCMP` results across multiple lanes
  (producing 0.0 or 1.0), a `VREDAND` on the raw float values would not give you the
  expected result because `1.0f` is not `0x00000001`. Use `VMASK_*` and check the mask
  register instead for predication logic. `VREDAND` is more useful when you have packed
  bit fields in float registers.
- Validating that all lanes carry the same flag pattern

**Warning for compiler authors:** Because `VREDAND` operates on float bits, an automatic
vectorizer must ensure it only emits this instruction for data that is semantically integer
or bitfield data, not for ordinary floating-point vectors where the bit pattern carries
no meaning independent of IEEE 754 encoding.

---

### 7.2.5 VREDOR — Bitwise OR Reduction

**Syntax:** `VREDOR rd, vs`

**Semantics:**
```
s[rd].i = (int64_t)vs[0] | (int64_t)vs[1] | ... | (int64_t)vs[vl-1]
```

**Simulator implementation** (`npu.c`, line 264):
```c
case V_VREDOR: {
    int64_t a = 0;   /* zero — identity element for OR */
    for (int l = 0; l < vl; l++) a |= (int64_t)n->v[I->vs][l];
    n->s[I->rd].i = a;
    break;
}
```

The identity element for OR is zero (a zero ORed with anything gives that thing). If all
lanes are zero, the result is zero. If any lane is nonzero, the result is nonzero.

**Primary use cases:**
- "Any flag set" test — the OR-reduction of a flag vector is nonzero if and only if any
  lane has a set bit. Combined with `BRZ`/`BRNZ`, this implements a vector early-exit.
- Accumulating error flags across lanes: if any lane sets an overflow bit, the OR-reduction
  will detect it.

**Pattern — early exit if any lane is nonzero:**
```c
{ V_VREDOR,  .rd=6, .vs=2 },       /* s6 = OR of all lanes in v2     */
{ V_BRNZ,    .rs=6, .imm=ERROR_PC }, /* jump to error handler if any set */
```

---

### 7.2.6 VSCAN — Prefix Sum (Inclusive Scan)

**Syntax:** `VSCAN vd, vs`

**Semantics:**
```
vd[0] = vs[0]
vd[1] = vs[0] + vs[1]
vd[2] = vs[0] + vs[1] + vs[2]
...
vd[l] = vs[0] + vs[1] + ... + vs[l]
```

More formally: `vd[l] = sum(vs[k] for k in 0..l)` — the _inclusive_ prefix sum.

**Simulator implementation** (`npu.c`, line 265):
```c
case V_VSCAN: {
    float a = 0;
    for (int l = 0; l < vl; l++) {
        a += n->v[I->vs][l];
        n->v[I->vd][l] = a;
    }
    break;
}
```

**Key distinction from reductions:** `VSCAN` writes a _vector_, not a scalar. The output
lives in a vector register `vd`. This is not a reduction in the sense that information
is not collapsed to one value — rather, a cumulative history is computed.

**ASCII diagram — 8-lane prefix sum:**

```
Input vs:   [ 1.0 ][ 2.0 ][ 3.0 ][ 4.0 ][ 5.0 ][ 6.0 ][ 7.0 ][ 8.0 ]
              lane0  lane1  lane2  lane3  lane4  lane5  lane6  lane7

              ↓      ↓      ↓      ↓      ↓      ↓      ↓      ↓

Output vd:  [ 1.0 ][ 3.0 ][ 6.0 ][10.0 ][15.0 ][21.0 ][28.0 ][36.0 ]
             =1    =1+2   =1+2+3 = ...                            =sum

Computation flow (conceptual binary tree for hardware):
Step 1 (shift 1):  partial[l] += partial[l-1]  (if l >= 1)
Step 2 (shift 2):  partial[l] += partial[l-2]  (if l >= 2)
Step 3 (shift 4):  partial[l] += partial[l-4]  (if l >= 4)
→ 3 passes for 8 lanes = log2(8) passes
```

Note that `vd[vl-1]` (the last lane) after `VSCAN` holds the total sum — the same value
that `VREDSUM` would produce. If you need both the prefix sums and the total, a single
`VSCAN` provides both.

**Primary use cases:**

1. **Causal attention mask (autoregressive models):** Compute cumulative sequence
   positions to determine which key-query pairs are in the causal window.

2. **Softmax with causal masking:** Zero out future positions, compute prefix sums
   to renormalize only the visible context.

3. **Cumulative distribution function (CDF):** Given a probability vector (output of a
   softmax), the CDF at each position is the prefix sum. Used in sampling and top-k
   filtering.

4. **Stream compaction:** `VSCAN` on a boolean predicate (0.0 or 1.0) produces the
   output indices for each matching element. Combined with `VSCATTER`, this implements
   a data-parallel filter — only elements where the predicate is true get written, and
   they are packed contiguously.

**Pattern — stream compaction:**
```c
/* v0 = float data, v1 = predicate (0.0 or 1.0 per lane)   */
{ V_VSCAN,    .vd=2, .vs=1 },         /* v2 = prefix_sum(predicate) = output indices */
{ V_VSCATTER, .vs=0, .vs1=2, .imm=OUTPUT_BASE }, /* scatter matching elements */
/* last lane of v2 = total count of matching elements */
{ V_VREDSUM,  .rd=7, .vs=1 },         /* s7 = sum of predicate = count */
```

---

## 7.3 Cycle Costs of Reductions

Understanding why reductions are expensive — and how hardware amortizes that cost — is
important for writing efficient kernels.

### 7.3.1 Cycle Model

From `npu.c`, the `vec_cost` function (lines 119–121):
```c
case V_VREDSUM: case V_VREDMAX: case V_VREDMIN:
case V_VREDAND: case V_VREDOR: case V_VSCAN:
    return (uint64_t)passes + 4;    /* tree reduction */
```

Where `passes = ceil(vl / HW_LANES)`. For `vl = 16` and `HW_LANES = 16`, passes = 1,
giving a cost of **5 cycles**. For `vl = 32`, passes = 2, cost = 6 cycles.

Compare this to a single elementwise operation (e.g., `VADD`), which costs `passes` cycles
(1 cycle for `vl = 16`). The extra `+4` represents the logarithmic reduction stages needed
after the initial parallel pass.

### 7.3.2 The Reduction Tree

A hardware reduction exploits associativity to compute the answer in O(log vl) time,
using a tournament-style binary tree:

```
8-lane reduction example:

Lane:   [0]   [1]   [2]   [3]   [4]   [5]   [6]   [7]

        [ 1]  [ 2]  [ 3]  [ 4]  [ 5]  [ 6]  [ 7]  [ 8]   (input)
          \   /       \   /       \   /       \   /
Stage 1:  [ 3]        [ 7]        [11]        [15]          (add pairs)
             \         /             \         /
Stage 2:     [10]                   [26]                    (add quads)
                  \                  /
Stage 3:          [        36      ]                        (total)

→ 3 stages = log2(8) for 8 lanes
→ Result broadcast back or written to scalar register
```

This tree structure is why the simulator's cycle formula adds a constant `+4` — it models
the pipeline latency of the logarithmic tree without requiring the simulator to explicitly
simulate each tree stage.

### 7.3.3 Implications for Kernel Design

1. **Batch reductions where possible.** If you need both `max(v)` and `sum(v)`, compute
   them on different vectors in separate instructions — do not serialize them in a
   dependency chain if the inputs are independent.

2. **VGATHER + VREDSUM = very expensive.** A gather from DRAM costs `NPU_DRAM_LATENCY ×
   vl = 100 × 16 = 1600` cycles, plus the reduction on top. Always DMA data to SRAM
   before reducing.

3. **VSCAN is as cheap as VREDSUM.** Both have the same cycle cost formula in the model
   because the hardware can pipeline the scan similarly to a tree reduction. Do not avoid
   VSCAN out of an assumption that it is slower.

---

## 7.4 Numerically Stable Softmax — Complete Walkthrough

Softmax is the canonical example of why hardware reductions matter. It appears in attention
mechanisms, output probability distributions, and gating functions. Let us build it
completely from ISA primitives, explaining every decision.

### 7.4.1 The Naive Formula and Its Failure Mode

The textbook softmax formula for a vector `x` of length N is:

```
softmax(x)[i] = exp(x[i]) / sum(exp(x[j]) for j = 0..N-1)
```

This fails in floating point when any element of `x` is large enough that `exp(x[i])`
overflows. In float32, `exp(89)` is approximately `4.5e38`, which is within range;
`exp(90)` is `Inf`. In practice, logit values in language models can easily exceed 90,
and in attention mechanisms the unnormalized scores `Q·K^T / sqrt(d_k)` can be larger
still.

### 7.4.2 The Safe Three-Pass Algorithm

The safe algorithm is mathematically equivalent but numerically stable:

```
m = max(x[i] for all i)
z[i] = exp(x[i] - m)        ; shifted: max argument is exactly 0.0
denom = sum(z[i] for all i) ; denom in range (1, N]; never 0, never Inf
softmax(x)[i] = z[i] / denom
```

The shift by `m` is safe because:
- `exp(x[i] - m)` is in `(0, 1]` for all i (since `x[i] - m <= 0`)
- The denominator is at least 1.0 (the lane that achieved the max contributes exactly 1.0)
- No lane can overflow; underflow to 0.0 for very negative inputs is correct behavior

### 7.4.3 Full Kernel — Annotated Assembly Struct Array

The following kernel matches the structure of `demo_softmax()` in `examples.c`, which
processes one row of a matrix per grid instance. All arguments are passed via the args
block pointed to by `s1`.

```c
static const vinst softmax_stable[] = {
    /*
     * Prologue: decode args block
     * s1 = args base address (set by kernel launch convention)
     * s0 = instance id (set by kernel launch convention)
     * Args layout at s1:
     *   [+0]  int32 x_base  -- base address of input matrix in DRAM/SRAM
     *   [+4]  int32 y_base  -- base address of output matrix in DRAM/SRAM
     *   [+8]  int32 n       -- number of columns (elements per row)
     */

    /* Step 1: Load args from the args block */
    { .op=V_SLW,   .rd=2, .use_base=1, .rs=1, .imm=0  },  /* s2 = x_base         */
    { .op=V_SLW,   .rd=3, .use_base=1, .rs=1, .imm=4  },  /* s3 = y_base         */
    { .op=V_SLW,   .rd=4, .use_base=1, .rs=1, .imm=8  },  /* s4 = n (col count)  */

    /* Step 2: Compute the byte offset for this instance's row.
     * Instance s0 processes row s0.
     * Row offset in bytes = s0 * n * sizeof(float) = s0 * n * 4 */
    { .op=V_SMUL,  .rd=5, .rs1=0, .rs2=4 },                /* s5 = id * n         */
    { .op=V_SMULI, .rd=5, .rs1=5, .imm=4 },                /* s5 = id * n * 4     */

    /* Step 3: Compute per-instance row pointers */
    { .op=V_SADD,  .rd=6, .rs1=2, .rs2=5 },                /* s6 = x_base + offset (input row)  */
    { .op=V_SADD,  .rd=7, .rs1=3, .rs2=5 },                /* s7 = y_base + offset (output row) */

    /* Step 4: Set vector length from n, then load input row into v0 */
    { .op=V_VSETVL, .rd=8, .rs1=4 },                       /* vl = n; s8 = actual vl set        */
    { .op=V_VLOAD,  .vd=0, .use_base=1, .rs=6, .imm=0 },   /* v0[0..n-1] = x row                */

    /* Step 5: Numerically stable softmax — pass 1: find max */
    { .op=V_VREDMAX, .rd=9,  .vs=0 },                      /* s9  = max(v0) -- THE STABILITY FIX */

    /* Step 6: Broadcast max to all lanes and subtract */
    { .op=V_VBCAST,  .vd=1, .rs=9 },                       /* v1[l] = s9 for all l              */
    { .op=V_VSUB,    .vd=0, .vs1=0, .vs2=1 },              /* v0 = x - max (shifted; max elem=0)*/

    /* Step 7: Exponentiate (all arguments <= 0; no overflow possible) */
    { .op=V_VEXP,    .vd=1, .vs=0 },                       /* v1 = exp(v0) = exp(x - max)       */

    /* Step 8: Sum the exponentiated values */
    { .op=V_VREDSUM, .rd=10, .vs=1 },                      /* s10 = sum(exp(x - max))           */

    /* Step 9: Compute 1/sum as a scalar, then broadcast and multiply */
    { .op=V_SRECIP,  .rd=11, .rs1=10 },                    /* s11 = 1 / sum                     */
    { .op=V_VBCAST,  .vd=2,  .rs=11 },                     /* v2[l] = 1/sum for all l           */
    { .op=V_VMUL,    .vd=1,  .vs1=1, .vs2=2 },             /* v1 = exp(x-max) * (1/sum) = p     */

    /* Step 10: Write output row and exit */
    { .op=V_VSTORE, .vs=1, .use_base=1, .rs=7, .imm=0 },   /* y row = v1                        */
    { .op=V_KEXIT },                                        /* signal completion to control core  */
};
```

**Instruction count breakdown:**
- 3 arg-loads + 2 address computations + 2 row-pointer computations = 7 prologue instructions
- 1 VSETVL + 1 VLOAD = 2 data setup instructions
- 1 VREDMAX + 1 VBCAST + 1 VSUB = 3 stability pass instructions
- 1 VEXP = 1 exp pass instruction
- 1 VREDSUM + 1 SRECIP + 1 VBCAST + 1 VMUL = 4 normalization instructions
- 1 VSTORE + 1 KEXIT = 2 epilogue instructions

**Total: 20 instructions** for a complete, numerically stable softmax on a variable-length
row. This is the direct cost of full programmability — you write the algorithm, not a
configuration register.

### 7.4.4 Register Usage

```
Scalar registers:
  s0   [read-only]  instance id (grid position = row index)
  s1   [read-only]  args base pointer
  s2   [scratch]    x_base address
  s3   [scratch]    y_base address
  s4   [scratch]    n (column count)
  s5   [scratch]    byte offset = id * n * 4
  s6   [scratch]    input row pointer
  s7   [scratch]    output row pointer
  s8   [scratch]    vl (returned by VSETVL)
  s9   [scratch]    max(row)  [consumed by VBCAST]
  s10  [scratch]    sum(exp)  [consumed by SRECIP]
  s11  [scratch]    1/sum     [consumed by VBCAST]

Vector registers:
  v0   input row, then x - max (overwritten)
  v1   exp(x - max), then final softmax output
  v2   broadcast of 1/sum
```

The kernel uses 12 scalar registers and 3 vector registers. With 32 of each available,
this kernel has ample headroom for loop unrolling or fusing with the next operation.

---

## 7.5 Hardware Implementation Perspective — Why a Tree?

This section explains the architecture behind the cycle model to build intuition for
when reductions are cheap and when they dominate.

### 7.5.1 Naive Sequential Reduction

The simulator implementation is sequential:
```c
float a = 0;
for (int l = 0; l < vl; l++) a += n->v[I->vs][l];
```

On hardware, if you serialized this, you would need `vl - 1` additions. For `vl = 16`,
that is 15 dependent operations — you cannot start the next addition until the previous
one completes. At 1 cycle per addition, this costs 15 cycles just for the arithmetic,
ignoring everything else.

### 7.5.2 Binary Tree Reduction

A tree reduction exploits the associativity of addition (and max, and min) to parallelize:

```
Level 0 (input): a0  a1  a2  a3  a4  a5  a6  a7  a8  a9  a10 a11 a12 a13 a14 a15
                   \  /    \  /    \  /    \  /    \  /     \  /    \  /    \  /
Level 1:           b0      b1      b2      b3      b4       b5      b6      b7
                    (a0+a1) (a2+a3) (a4+a5) (a6+a7) ...
                      \    /          \    /          \    /          \    /
Level 2:              c0              c1              c2              c3
                         \           /                   \           /
Level 3:                 d0                              d1
                             \                           /
Level 4:                                 result
```

For `vl = 16`: 4 levels = log2(16) = 4 stages. The hardware executes all additions within
each stage in parallel. This is what makes the `+4` constant in the cycle model accurate —
it represents the 4 pipeline stages of the reduction tree for 16-lane hardware.

### 7.5.3 Implications of Tree Non-Associativity in Float

Floating-point addition is not exactly associative: `(a + b) + c` may differ slightly from
`a + (b + c)` due to rounding. The hardware tree evaluates the reduction in a fixed binary
tree order that may differ from a left-to-right sequential accumulation. The simulator
evaluates left-to-right. This means results on hardware may differ from the simulator by
a ULP or two for VREDSUM.

This is not a bug — it is a fundamental property of floating-point arithmetic with
non-sequential evaluation. The key invariant that does hold is: the result is numerically
correct in the sense that all rounding errors are bounded by the standard floating-point
error analysis.

For `VREDMAX` and `VREDMIN`, the tree introduces no additional error because max/min is
exactly associative — the result of `max(a, max(b, c))` is identically equal to
`max(max(a, b), c)` for all IEEE 754 finite values.

---

## 7.6 Comparison with Other Architectures

Understanding how the NPU's reductions compare to other vector ISAs helps explain the
design choices.

### 7.6.1 RISC-V Vector (RVV)

The RISC-V V extension provides `vredsum.vs`, `vredmax.vs`, `vredmin.vs` etc. The NPU
reduction instructions are directly modeled on RVV semantics, with one simplification: RVV
has both masked and unmasked variants for reductions, while the NPU exposes only unmasked
reductions (the predicate mask `@m` applies to compute instructions but not to the current
reduction set). This simplification is deliberate — it avoids the complexity of "what is
the masked identity element" for each operation.

### 7.6.2 ARM SVE

ARM SVE's `faddv` (floating add reduction), `fmaxv`, `fminv` work similarly. SVE
additionally supports predicated reductions with a governing predicate register — a feature
the NPU could add in a future ISA revision.

### 7.6.3 x86 AVX-512

AVX-512 does not have a single-instruction horizontal reduction. Instead, you compose
`_mm512_reduce_add_ps` from a sequence of `vperm` + `add` operations — the very sequence
that the NPU encodes as a single `VREDSUM`. This is a common pain point for x86 kernel
writers. The NPU's dedicated reduction instructions eliminate this entirely.

---

## 7.7 Reduction Instruction Reference Table

| Instruction | Operands | Operation | Result Reg | Identity | Use Case |
|-------------|----------|-----------|------------|----------|----------|
| `VREDSUM` | `rd, vs` | `sum(vs[0..vl-1])` | `s[rd].f` | 0.0 | Softmax denom, mean, loss |
| `VREDMAX` | `rd, vs` | `max(vs[0..vl-1])` | `s[rd].f` | -Inf | Stable softmax, argmax |
| `VREDMIN` | `rd, vs` | `min(vs[0..vl-1])` | `s[rd].f` | +Inf | Quantization scale |
| `VREDAND` | `rd, vs` | `AND(bits(vs[0..vl-1]))` | `s[rd].i` | ~0 | All-flags check |
| `VREDOR`  | `rd, vs` | `OR(bits(vs[0..vl-1]))` | `s[rd].i` | 0 | Any-flag check |
| `VSCAN`   | `vd, vs` | `vd[l] = sum(vs[0..l])` | `v[vd]` | N/A | CDF, stream compact |

**Notes:**
- All arithmetic reductions (`VREDSUM`, `VREDMAX`, `VREDMIN`) operate on the float
  interpretation of lanes and return float in `s[rd].f`.
- Bitwise reductions (`VREDAND`, `VREDOR`) reinterpret float bits as `int64_t` and return
  integer in `s[rd].i`.
- `VSCAN` is the only reduction that returns a vector; it also never requires a second
  `VREDSUM` to get the total — `vd[vl-1]` after `VSCAN` equals `VREDSUM` of the same input.
- Predication (the `@m` prefix) is not currently supported on reduction instructions.
  If you need a masked reduction (e.g., sum only the positive elements), load into a
  scratch vector with a predicated `VMOV`, then reduce the scratch vector.

---

## 7.8 Extended Example — LayerNorm

Layer normalization is a more demanding kernel than softmax because it requires three
passes over the data:

```
pass 1: mean   = sum(x) / N
pass 2: var    = sum((x - mean)^2) / N
pass 3: y      = (x - mean) / sqrt(var + epsilon)  * gamma + beta
```

Each pass requires a reduction and a subsequent broadcast:

```c
static const vinst layernorm[] = {
    /* --- Load args --- */
    { .op=V_SLW,    .rd=2, .use_base=1, .rs=1, .imm=0  },  /* x ptr     */
    { .op=V_SLW,    .rd=3, .use_base=1, .rs=1, .imm=4  },  /* y ptr     */
    { .op=V_SLW,    .rd=4, .use_base=1, .rs=1, .imm=8  },  /* n         */
    { .op=V_SLW,    .rd=5, .use_base=1, .rs=1, .imm=12 },  /* gamma ptr */
    { .op=V_SLW,    .rd=6, .use_base=1, .rs=1, .imm=16 },  /* beta ptr  */
    /* --- Compute row offset for this instance (same as softmax) --- */
    { .op=V_SMUL,   .rd=10, .rs1=0, .rs2=4 },
    { .op=V_SMULI,  .rd=10, .rs1=10, .imm=4 },
    { .op=V_SADD,   .rd=11, .rs1=2, .rs2=10 },  /* x row ptr  */
    { .op=V_SADD,   .rd=12, .rs1=3, .rs2=10 },  /* y row ptr  */
    /* --- Load data --- */
    { .op=V_VSETVL, .rd=7, .rs1=4 },
    { .op=V_VLOAD,  .vd=0, .use_base=1, .rs=11 },  /* v0 = x row */
    /* --- Pass 1: compute mean = sum(x) / n --- */
    { .op=V_VREDSUM, .rd=20, .vs=0 },           /* s20 = sum(x)        */
    /* 1/n (precomputed or from SLF + SRECIP if n is runtime) */
    { .op=V_SLW,    .rd=21, .use_base=1, .rs=1, .imm=20 },  /* s21 = n (int)   */
    /* For simplicity assume 1/n was precomputed; s22 = mean: */
    { .op=V_SFI,    .rd=22, .fimm=1.0/16.0 },   /* s22 = 1/n (example: n=16) */
    /* mean = sum * (1/n) -- scalar multiply, not yet in ISA, approximate via: */
    /* Actually use SRECIP on the count, then multiply. Here we use SFI for clarity. */
    /* v1 = broadcast(mean) */
    { .op=V_VBCAST, .vd=1, .rs=20 },            /* v1 = sum (wrong; need /n first) */
    /* For a real implementation, multiply sum by (1/n) before broadcast.
     * This sketch highlights that you need a scalar multiply of s20 * (1/n).
     * The ISA lacks SFMUL but SMUL works on integer bits; use VBCAST of the
     * 1/n value and then VMUL with a one-element vector, or store to memory
     * and reload as float. This is a gap — a SMULF (float scalar multiply) would be
     * a useful addition to the ISA scalar helper set. */
    /* --- Pass 2: compute variance = sum((x - mean)^2) / n --- */
    { .op=V_VSUB,   .vd=2, .vs1=0, .vs2=1 },    /* v2 = x - mean (centered)    */
    { .op=V_VMUL,   .vd=3, .vs1=2, .vs2=2 },    /* v3 = (x-mean)^2             */
    { .op=V_VREDSUM, .rd=23, .vs=3 },            /* s23 = sum((x-mean)^2)       */
    /* var = s23 / n (same scalar issue as above) */
    /* --- Pass 3: normalize --- */
    /* rsqrt(var + eps): load eps, add to s23 (scalar), then VSQRT of broadcast */
    { .op=V_SFI,    .rd=24, .fimm=1e-5 },        /* s24 = epsilon               */
    /* s23 = var; s23 + eps requires scalar float add -- gap in ISA             */
    /* For now: broadcast var+eps and VRSQRT */
    { .op=V_VBCAST, .vd=4, .rs=23 },             /* v4 = var (approx)           */
    { .op=V_VRSQRT, .vd=4, .vs=4 },              /* v4 = 1/sqrt(var)            */
    { .op=V_VMUL,   .vd=2, .vs1=2, .vs2=4 },    /* v2 = (x-mean) * rsqrt(var)  */
    /* Load gamma and beta, apply affine transform */
    { .op=V_VLOAD,  .vd=5, .use_base=1, .rs=5 }, /* v5 = gamma                 */
    { .op=V_VLOAD,  .vd=6, .use_base=1, .rs=6 }, /* v6 = beta                  */
    { .op=V_VMUL,   .vd=2, .vs1=2, .vs2=5 },    /* v2 = normalized * gamma     */
    { .op=V_VADD,   .vd=2, .vs1=2, .vs2=6 },    /* v2 = v2 + beta              */
    { .op=V_VSTORE, .vs=2, .use_base=1, .rs=12 },
    { .op=V_KEXIT },
};
```

The layernorm sketch reveals a minor ISA gap: there is no scalar float multiply or float
add in the engine scalar helpers. `SMUL` operates on integers. A real implementation would
either precompute `1/n` on the host and store it in the args block, or use a one-element
`VMUL` workaround. This is typical of ISA evolution — reductions were designed primarily
for the softmax/argmax use cases, and the scalar arithmetic helpers needed for subsequent
computations reveal opportunities for future instructions.

---

## Summary

The six reduction instructions (`VREDSUM`, `VREDMAX`, `VREDMIN`, `VREDAND`, `VREDOR`,
`VSCAN`) are among the most impactful instructions in the vector core ISA because:

1. They each replace an O(N) scalar loop with a single instruction that the hardware
   executes in O(log N) time via a binary tree.
2. They are the linchpin of numerically stable algorithms: softmax stability requires
   `VREDMAX`; normalization requires `VREDSUM`.
3. `VSCAN` opens up a class of data-parallel algorithms (stream compaction, CDF
   computation, causal attention) that would be awkward or impossible to express
   efficiently without it.
4. The bitwise reductions (`VREDAND`, `VREDOR`) provide the infrastructure for
   predicate checking and early-exit control flow in data-parallel kernels.

Chapter 8 explains how to go further: instead of collapsing the vector to a scalar, how
do you selectively apply different operations to _different lanes_ within the same
instruction? That is the domain of predication.
