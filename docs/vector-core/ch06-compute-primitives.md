# Chapter 6 — Compute Primitives (Section 10)

> **Scope of this chapter.** This chapter covers the seventeen vector compute instructions
> that form the heart of the vector core: the arithmetic, transcendental, comparison, and
> type-conversion instructions of Section 10 of the ISA reference
> (`fully_prog_npu.md`, section 10). These are the instructions that operate on the vector
> register file `v0`–`v31`, applying the same operation to every active lane in a single
> instruction. Together they are the toolkit from which all arbitrary neural network
> operations are composed.

---

## 6.1  How Lane-Parallel Execution Works

The entire value proposition of the vector core rests on one idea: issue one instruction,
perform the operation on *all `vl` active lanes simultaneously*. The hardware replicates the
ALU `HW_LANES = 16` times (physically), so an instruction on 16 lanes costs 1 pass through
the ALU pipeline. For `vl > 16`, the hardware makes multiple passes.

### The VADD picture

```
  v0: [ 1.0 ][ 2.0 ][ 3.0 ][ 4.0 ][ 5.0 ][ 6.0 ][ 7.0 ][ 8.0 ]
        +       +       +       +       +       +       +       +
  v1: [ 0.5 ][ 0.5 ][ 0.5 ][ 0.5 ][ 0.5 ][ 0.5 ][ 0.5 ][ 0.5 ]
        │       │       │       │       │       │       │       │
        ▼       ▼       ▼       ▼       ▼       ▼       ▼       ▼
  v2: [ 1.5 ][ 2.5 ][ 3.5 ][ 4.5 ][ 5.5 ][ 6.5 ][ 7.5 ][ 8.5 ]

  One VADD instruction → eight additions happen in the same cycle (vl=8, HW_LANES=16 → 1 pass)
```

Contrast this with a scalar loop:
```c
/* Scalar equivalent: 8 iterations, 8 cycles minimum */
for (int l = 0; l < 8; l++)
    c[l] = a[l] + b[l];
```

The vector core eliminates the loop overhead (counter increment, branch, address update) and
performs all additions in one pass. At `vl = 16` the speedup over a scalar core is 16×; at
`vl = 256` it is 16× per element with `256/16 = 16` passes.

### The `ACTIVE(l)` predicate

Every compute instruction in Section 10 respects the per-instruction mask prefix `@mask`:

```c
const uint8_t *pm = I->use_mask ? n->m[I->mask] : NULL;
#define ACTIVE(L)  (pm == NULL || pm[(L)])
```

When `use_mask = 0` (no `@m` prefix), `ACTIVE(l)` is always true and every lane executes.
When `use_mask = 1`, only lanes where `m[mask][l] == 1` write to `vd`; the others leave the
destination unchanged. This is the predication mechanism described in Section 12 of the ISA
and used in `demo_predication` (Chapter 7).

### The cycle model

The `vec_cost` function in `npu.c` assigns cycle costs per instruction category:

```
  passes = ceil(vl / HW_LANES)        (minimum 1)

  Instruction class          Cost
  ─────────────────────────  ─────────────────────────────────
  Elementwise (VADD, etc.)   passes        (1 cycle per 16 lanes)
  VFMA                       passes        (same; FMA counts as one op)
  VDIV, VEXP, VLOG, etc.     passes × 4   (transcendental/reciprocal)
  VSQRT, VRSQRT, VRECIP      passes × 4
  VCMP, VSEL, VCVT           passes        (elementwise)
  VABS, VNEG                 passes        (elementwise)
  VGATHER, VSCATTER          vl            (1 lane per cycle; irregular)
  Reductions                 passes + 4    (tree + final scalar)
  Scalar helpers             1             (no vector datapath involved)
```

The `× 4` multiplier for transcendentals reflects the use of iterative hardware
approximations (CORDIC or polynomial evaluation) that require multiple pipeline stages beyond
a simple add/multiply. The actual multiplier on real hardware varies from 4× to 30×; the
simulator uses 4× as a conservative lower bound for design exploration.

---

## 6.2  Arithmetic Instructions

### 6.2.1  VADD  vd, vs1, vs2 — Elementwise Float Add

**Semantics:**
```
vd[l] = vs1[l] + vs2[l]    for each active lane l in [0, vl)
```

**Simulator implementation (`npu.c` line 221):**
```c
case V_VADD:
    for (int l = 0; l < vl; l++)
        if (ACTIVE(l)) n->v[I->vd][l] = n->v[I->vs1][l] + n->v[I->vs2][l];
    break;
```

**When to use it:** Any elementwise addition — residual connections in transformers, bias
addition, combining partial sums, the subtraction step in softmax (via VSUB), etc.

**Complete demo (from `demo_vadd`, `examples.c`):**
```c
static const vinst k[] = {
    { .op=V_SLW,    .rd=2, .use_base=1, .rs=1, .imm=0  },  /* s2 = ptr_a  */
    { .op=V_SLW,    .rd=3, .use_base=1, .rs=1, .imm=4  },  /* s3 = ptr_b  */
    { .op=V_SLW,    .rd=4, .use_base=1, .rs=1, .imm=8  },  /* s4 = ptr_c  */
    { .op=V_SLW,    .rd=5, .use_base=1, .rs=1, .imm=12 },  /* s5 = n      */
    { .op=V_VSETVL, .rd=6, .rs1=5 },                       /* VL = n      */
    { .op=V_VLOAD,  .vd=0, .use_base=1, .rs=2 },           /* v0 = a      */
    { .op=V_VLOAD,  .vd=1, .use_base=1, .rs=3 },           /* v1 = b      */
    { .op=V_VADD,   .vd=2, .vs1=0, .vs2=1 },               /* v2 = v0+v1  */
    { .op=V_VSTORE, .vs=2, .use_base=1, .rs=4 },           /* c = v2      */
    { .op=V_KEXIT },
};
```

**Assembly syntax:**
```asm
VADD v2, v0, v1
```

---

### 6.2.2  VSUB  vd, vs1, vs2 — Elementwise Float Subtract

**Semantics:**
```
vd[l] = vs1[l] - vs2[l]    for each active lane l
```

**Simulator implementation:**
```c
case V_VSUB:
    for (int l = 0; l < vl; l++)
        if (ACTIVE(l)) n->v[I->vd][l] = n->v[I->vs1][l] - n->v[I->vs2][l];
    break;
```

**Primary use — max-subtraction in numerically stable softmax.** The standard softmax formula
`exp(x) / sum(exp(x))` is numerically unstable for large `x` because `exp(x)` overflows.
The numerically stable version subtracts the row maximum before exponentiating:
```
y = exp(x - max(x)) / sum(exp(x - max(x)))
```
The subtraction step is a VSUB:

```c
{ .op=V_VREDMAX, .rd=9, .vs=0 },   /* s9 = max(x)        */
{ .op=V_VBCAST,  .vd=1, .rs=9 },  /* v1[l] = max(x)     */
{ .op=V_VSUB,    .vd=0, .vs1=0, .vs2=1 }, /* v0 = x - max */
```

This is lines 91–93 of `demo_softmax` in `examples.c`.

**Other uses:**
- Residual subtraction (error signals in loss functions)
- Centering data before normalization (subtract mean)
- Temporal differences (current - previous activation)

---

### 6.2.3  VMUL  vd, vs1, vs2 — Elementwise Float Multiply

**Semantics:**
```
vd[l] = vs1[l] * vs2[l]    for each active lane l
```

**Simulator implementation:**
```c
case V_VMUL:
    for (int l = 0; l < vl; l++)
        if (ACTIVE(l)) n->v[I->vd][l] = n->v[I->vs1][l] * n->v[I->vs2][l];
    break;
```

VMUL is one of the most-used instructions in the ISA. Its roles include:

| Use case | What is multiplied |
|----------|--------------------|
| Scaling/normalization | `exp(x)` × `(1/sum)` in softmax |
| Attention weights | query × key (element-wise, after softmax) |
| Gating | hidden state × sigmoid gate (in LSTM/GRU) |
| Hadamard product | two activation tensors |
| Polynomial evaluation (with VFMA) | coefficient × x^n |
| Quantization dequant | int8 weight × scale factor |

**From `demo_softmax`, the normalization step:**
```c
{ .op=V_VMUL, .vd=0, .vs1=0, .vs2=1 },   /* v0 = exp(x-max) * (1/sum) */
```

---

### 6.2.4  VDIV  vd, vs1, vs2 — Elementwise Float Divide

**Semantics:**
```
vd[l] = vs1[l] / vs2[l]    for each active lane l
```

**Simulator implementation:**
```c
case V_VDIV:
    for (int l = 0; l < vl; l++)
        if (ACTIVE(l)) n->v[I->vd][l] = n->v[I->vs1][l] / n->v[I->vs2][l];
    break;
```

**Cycle cost:** `passes × 4` — the most expensive elementwise instruction.

**When to prefer VRECIP + VMUL over VDIV:**

If you are dividing many vectors by the *same denominator* (a common situation in
normalization), compute the scalar reciprocal once with `SRECIP`, broadcast it with `VBCAST`,
and multiply with `VMUL`. This costs:

```
SRECIP:       1 cycle
VBCAST:       passes
VMUL:         passes
Total:        2*passes + 1
```

versus VDIV on `k` vectors:
```
k × VDIV:    k × passes × 4
```

For softmax with one shared denominator and vl = 16:
- VDIV: 4 cycles
- SRECIP + VBCAST + VMUL: 1 + 1 + 1 = 3 cycles

The gain is modest for one vector, but across many rows in a batched softmax the savings
accumulate.

**When VDIV is appropriate:** When the denominator is *per-lane* (a different value in each
lane), there is no scalar shortcut. VDIV is correct and necessary.

---

### 6.2.5  VMAX  vd, vs1, vs2 — Elementwise Float Maximum

**Semantics:**
```
vd[l] = max(vs1[l], vs2[l])    for each active lane l
```

**Simulator implementation:**
```c
case V_VMAX:
    for (int l = 0; l < vl; l++)
        if (ACTIVE(l)) n->v[I->vd][l] = fmaxf(n->v[I->vs1][l], n->v[I->vs2][l]);
    break;
```

**The ReLU recipe.** ReLU (Rectified Linear Unit) is `max(0, x)`. With VMAX:

```c
/* Step 1: build a vector of zeros */
{ .op=V_SFI,    .rd=5,  .fimm=0.0 }   /* s5 = 0.0f               */
{ .op=V_VBCAST, .vd=3,  .rs=5 }       /* v3[l] = 0.0 for all l   */

/* Step 2: ReLU */
{ .op=V_VMAX,   .vd=0, .vs1=0, .vs2=3 }  /* v0[l] = max(v0[l], 0) */
```

Total: 3 instructions (SFI + VBCAST amortize over many ReLU calls if v3 is kept live).

**Other uses:**
- Clipping to an upper bound: `VMIN(x, upper)` after VMAX gives clamp.
- Leaky ReLU: `VMAX(x, alpha*x)` after computing `alpha*x` with VMUL.
- Hard sigmoid approximation: clip `(x + 3) / 6` to [0, 1].
- Online maximum (iterative): VMAX accumulates the running max in a loop.

**VMAX vs VREDMAX:** VMAX compares two *vectors* lane-by-lane and produces a *vector*.
VREDMAX reduces one vector across all lanes and produces a *scalar*. They are different
operations serving different purposes.

---

### 6.2.6  VMIN  vd, vs1, vs2 — Elementwise Float Minimum

**Semantics:**
```
vd[l] = min(vs1[l], vs2[l])    for each active lane l
```

**Simulator implementation:**
```c
case V_VMIN:
    for (int l = 0; l < vl; l++)
        if (ACTIVE(l)) n->v[I->vd][l] = fminf(n->v[I->vs1][l], n->v[I->vs2][l]);
    break;
```

VMIN is symmetrical to VMAX. Together they implement clamp:

```
clamp(x, lo, hi) = min(max(x, lo), hi)
                 = VMIN(VMAX(x, v_lo), v_hi)
```

This is a common pattern in quantization-aware training (clipping activations to the
representable range of int8) and in attention masking (clipping logits to a large negative
value rather than −∞).

**Clamping example:**
```c
/* Clamp v0 to [0.0, 1.0] */
{ .op=V_SFI,    .rd=5,  .fimm=0.0 }          /* s5 = 0.0 */
{ .op=V_VBCAST, .vd=3,  .rs=5 }              /* v3 = 0.0 */
{ .op=V_SFI,    .rd=6,  .fimm=1.0 }          /* s6 = 1.0 */
{ .op=V_VBCAST, .vd=4,  .rs=6 }              /* v4 = 1.0 */
{ .op=V_VMAX,   .vd=0, .vs1=0, .vs2=3 }      /* v0 = max(v0, 0) */
{ .op=V_VMIN,   .vd=0, .vs1=0, .vs2=4 }      /* v0 = min(v0, 1) */
```

---

### 6.2.7  VFMA  vd, vs1, vs2 — Fused Multiply-Accumulate

**Semantics:**
```
vd[l] += vs1[l] * vs2[l]    for each active lane l
```

**Critical note: VFMA accumulates into `vd` rather than replacing it.** This is unlike VADD,
VSUB, or VMUL, which always write a fresh result to `vd`. The destination register appears on
*both sides* of the assignment.

**Simulator implementation:**
```c
case V_VFMA:
    for (int l = 0; l < vl; l++)
        if (ACTIVE(l)) n->v[I->vd][l] = n->v[I->vd][l] + n->v[I->vs1][l] * n->v[I->vs2][l];
    break;
```

### Why VFMA instead of VMUL + VADD?

Three reasons:

**1. Instruction count.** One VFMA replaces two instructions. In a tight inner loop this
matters for both code size and front-end pressure.

**2. Numerical accuracy.** A true hardware FMA computes `a + b*c` with only one rounding
step (the final one), rather than rounding twice as VMUL-then-VADD would. IEEE 754-2008
requires this behavior for FMA. The simulator does not currently implement the extra precision
(it uses C's `a + b*c` which may or may not elide the intermediate rounding depending on the
compiler), but a real hardware FMA is strictly more accurate.

**3. Accumulation patterns.** Many algorithms accumulate a running sum of products into one
register over many iterations. VFMA is the natural match:

```c
/* Pattern: vd = sum over K of vs1_k * vs2_k  (K-wise dot product, vectorized) */
/* vd must be zeroed before the loop */
for (int k = 0; k < K; k++) {
    /* load column k of A into v1, column k of B into v2 */
    VFMA vd, v1, v2    /* vd[l] += v1[l] * v2[l] */
}
```

**VMUL + VADD vs VFMA — side by side:**

```
VMUL + VADD (two-instruction version):
  VMUL v_tmp, vs1, vs2     /* v_tmp  = vs1 * vs2            */
  VADD vd,    vd,  v_tmp   /* vd     = vd  + v_tmp (= old vd + vs1*vs2) */
  Cost: 2 instructions, 2 × passes cycles, uses an extra temp register

VFMA (one-instruction version):
  VFMA vd, vs1, vs2        /* vd    += vs1 * vs2            */
  Cost: 1 instruction, 1 × passes cycles, no temp register needed
```

### Dot product using VFMA

The dot product `result = sum(a[l] * b[l])` for a length-`vl` vector:

```c
/* Zero out the accumulator */
{ .op=V_SFI,     .rd=5,  .fimm=0.0 }
{ .op=V_VBCAST,  .vd=2,  .rs=5 }        /* v2[l] = 0.0  for all l */

/* Accumulate */
{ .op=V_VFMA,    .vd=2,  .vs1=0, .vs2=1 }   /* v2 += v0 * v1       */

/* Reduce to scalar */
{ .op=V_VREDSUM, .rd=6,  .vs=2 }             /* s6 = sum(v2)        */
```

For a longer dot product (length > vl), wrap in a loop with SADDI to advance the pointers,
VLOAD to reload, and VFMA to accumulate — the same pattern as a scalar RISC dot-product loop
but 16× wider.

### Polynomial evaluation using VFMA (Horner's method)

Horner's method evaluates `p(x) = a0 + a1*x + a2*x^2 + a3*x^3` as
`((a3*x + a2)*x + a1)*x + a0`:

```c
/* Assume v0 = x (input), v_acc accumulates the result */
{ .op=V_VBCAST, .vd=1, .rs=reg_a3 }   /* v1 = a3 (constant)    */
{ .op=V_VMOV,   .vd=2, .vs=1 }        /* v2 = a3               */
{ .op=V_VBCAST, .vd=1, .rs=reg_a2 }   /* v1 = a2               */
{ .op=V_VMUL,   .vd=2, .vs1=2, .vs2=0 }  /* v2 = a3*x          */
{ .op=V_VFMA,   .vd=2, .vs1=1, .vs2=0 }  /* WRONG — VFMA is vd += vs1*vs2 */
```

Wait — VFMA accumulates `vd += vs1 * vs2`, so the Horner step `(acc*x + a_k)` is:
```
v_acc = v_acc * v_x + v_a_k
      → v_acc *= v_x  first  (VMUL)
      → v_acc += v_a_k       (VADD)
      OR:
      → swap roles: VFMA v_a_k, v_acc, v_x  gives v_a_k += v_acc * v_x
```

The operand order matters. VFMA is most directly useful when accumulating a sum of *pairwise*
products, not for the Horner recurrence where one operand is itself the accumulator.

---

### 6.2.8  VABS  vd, vs — Elementwise Absolute Value

**Semantics:**
```
vd[l] = |vs[l]|    for each active lane l
```

**Simulator implementation:**
```c
case V_VABS:
    for (int l = 0; l < vl; l++)
        if (ACTIVE(l)) n->v[I->vd][l] = fabsf(n->v[I->vs][l]);
    break;
```

Note the operand naming: VABS uses `vs` (singular), not `vs1`/`vs2`, because it is unary.

**Uses:**
- L1 norm: `sum(|x|)` — VABS then VREDSUM.
- Mean Absolute Error: VSUB then VABS then VREDSUM.
- Symmetric clipping: check `VABS(x) > threshold` via VCMP, then VSEL.
- Gradient clipping by L∞ norm: `VREDMAX(VABS(gradient))`.

**Example — L1 norm:**
```c
{ .op=V_VABS,    .vd=1, .vs=0 }       /* v1 = |v0|    */
{ .op=V_VREDSUM, .rd=5, .vs=1 }       /* s5 = sum(|v0|) = L1 norm */
```

---

### 6.2.9  VNEG  vd, vs — Elementwise Negation

**Semantics:**
```
vd[l] = -vs[l]    for each active lane l
```

**Simulator implementation:**
```c
case V_VNEG:
    for (int l = 0; l < vl; l++)
        if (ACTIVE(l)) n->v[I->vd][l] = -n->v[I->vs][l];
    break;
```

**Uses:**
- Negate logits before softmax in log-sum-exp tricks.
- Compute negative residuals (target − prediction).
- Flip attention mask values.
- Symmetric: `VNEG(VNEG(x)) == x`.

**Why not just use VSUB(zero_vector, x)?** VNEG is one instruction versus two (SFI +
VBCAST + VSUB = three). Code density matters in small kernel instruction budgets.

---

## 6.3  Transcendental Functions

This section covers the instructions that go beyond four-function arithmetic. They are the
most expensive instructions in the ISA — but they are what make the vector core *useful* for
real neural networks. An NPU that lacked transcendentals would need to synthesize them from
polynomial approximations using VFMA chains, paying more instructions and losing accuracy.

### The cost argument

At `vl = 16`, `HW_LANES = 16`, `passes = 1`:

| Instruction | Simulator cost | Real hardware typical |
|-------------|---------------|----------------------|
| VADD, VMUL, etc. | 1 cycle | 1–2 cycles |
| VRECIP, VDIV | 4 cycles | 4–12 cycles |
| VSQRT, VRSQRT | 4 cycles | 8–16 cycles |
| VEXP, VLOG | 4 cycles | 16–32 cycles |

The simulator uses `passes × 4` for all transcendentals as a simplified model. A real
implementation would distinguish VEXP (slowest) from VRECIP (fastest of the group).

---

### 6.3.1  VEXP  vd, vs — Elementwise Natural Exponential

**Semantics:**
```
vd[l] = expf(vs[l])    for each active lane l
```

**Simulator implementation:**
```c
case V_VEXP:
    for (int l = 0; l < vl; l++)
        if (ACTIVE(l)) n->v[I->vd][l] = expf(n->v[I->vs][l]);
    break;
```

**The single most important transcendental.** VEXP is required for:

| Algorithm | Where VEXP appears |
|-----------|--------------------|
| Softmax | `exp(x - max(x))` for each logit |
| Self-attention | `exp(QK^T / sqrt(d))` for attention scores |
| GELU activation | approximation involves `exp` |
| Sigmoid | `1 / (1 + exp(-x))` |
| Log-sum-exp | numerically stable log of softmax denominator |

**The numerical stability issue.** Raw `exp(x)` overflows for `x > 88.7` (IEEE 754 float).
The standard defense — always subtract the row maximum before exponentiating — requires the
VREDMAX → VSUB → VEXP pipeline seen in the softmax kernel:

```
v0 = [1.0, 2.0, 3.0, 4.0]         (logits)

VREDMAX s9, v0          s9 = 4.0   (max)
VBCAST  v1, s9          v1 = [4.0, 4.0, 4.0, 4.0]
VSUB    v0, v0, v1      v0 = [-3.0,-2.0,-1.0, 0.0]   (shifted)
VEXP    v0, v0          v0 = [0.050, 0.135, 0.368, 1.000]
```

After the shift, the largest argument to VEXP is 0, which gives `exp(0) = 1.0` — well within
the safe range.

**VEXP and the softmax kernel — full trace:**
```
Kernel entry (one instance = one row):
  s0 = 0   (first row)
  s1 = 0x1000 (args ptr)

  SLW s2, 0(s1)        s2 = X_base = 0x2000
  SLW s3, 4(s1)        s3 = Y_base = 0x4000
  SLW s4, 8(s1)        s4 = N = 4
  SMUL  s5, s0, s4     s5 = 0 * 4 = 0
  SMULI s5, s5, 4      s5 = 0 bytes offset
  SADD  s6, s2, s5     s6 = 0x2000 (row 0 of X)
  SADD  s7, s3, s5     s7 = 0x4000 (row 0 of Y)
  VSETVL s8, s4        VL = 4
  VLOAD  v0, 0(s6)     v0 = [1.0, 2.0, 3.0, 4.0]
  VREDMAX s9, v0       s9 = 4.0
  VBCAST  v1, s9       v1 = [4.0, 4.0, 4.0, 4.0]
  VSUB    v0, v0, v1   v0 = [-3.0,-2.0,-1.0, 0.0]
  VEXP    v0, v0       v0 = [0.0498, 0.1353, 0.3679, 1.0000]
  VREDSUM s10, v0      s10 = 1.5530
  SRECIP  s11, s10     s11 = 0.6439
  VBCAST  v1, s11      v1 = [0.6439, 0.6439, 0.6439, 0.6439]
  VMUL    v0, v0, v1   v0 = [0.0321, 0.0871, 0.2369, 0.6439]   ← softmax output
  VSTORE  v0, 0(s7)    Y[0..3] = v0
  KEXIT
```

Sum check: 0.0321 + 0.0871 + 0.2369 + 0.6439 = 1.0000 ✓

---

### 6.3.2  VLOG  vd, vs — Elementwise Natural Logarithm

**Semantics:**
```
vd[l] = logf(vs[l])    for each active lane l
```

**Simulator implementation:**
```c
case V_VLOG:
    for (int l = 0; l < vl; l++)
        if (ACTIVE(l)) n->v[I->vd][l] = logf(n->v[I->vs][l]);
    break;
```

**Domain:** Input must be positive (`vs[l] > 0`). `logf(0)` returns `-INFINITY`; `logf(x)`
for `x < 0` returns `NaN`. The ISA does not trap on these; the kernel author is responsible
for guarding with VMAX or VCMP.

**Uses:**

| Algorithm | Where VLOG appears |
|-----------|--------------------|
| Cross-entropy loss | `−sum(y * log(p))` where p = softmax output |
| Log-softmax | `log(softmax(x)) = x - max(x) - log(sum(exp(x-max)))` |
| KL divergence | `sum(p * log(p/q))` |
| Entropy | `−sum(p * log(p))` |
| Information-theoretic metrics | various |

**Log-softmax kernel sketch** (avoids the VEXP + VLOG pair by exploiting the identity):
```asm
; log_softmax(x) = x - max(x) - log(sum(exp(x - max(x))))
VLOAD    v0, x
VREDMAX  s9, v0
VBCAST   v1, s9
VSUB     v0, v0, v1        ; x - max(x)
VEXP     v2, v0            ; exp(x - max(x))
VREDSUM  s10, v2           ; sum(exp(x - max(x)))
; now log(sum) is needed as a scalar
VBCAST   v3, s10           ; broadcast sum (reuse a register)
VLOG     v3, v3            ; v3[l] = log(sum) — same value in all lanes
VSUB     v0, v0, v3        ; x - max(x) - log(sum)  = log_softmax
VSTORE   v0, y
KEXIT
```

Note: `VLOG` is applied to a vector where all lanes hold the same value (the broadcasted sum).
This is slightly wasteful — a scalar log would suffice — but the ISA has no `SLOG` instruction.
A future extension or a clever compiler could optimize this by keeping the sum in a scalar and
using a `VCVT` + `VBCAST` trick if log were implemented as a polynomial.

---

### 6.3.3  VSQRT  vd, vs — Elementwise Square Root

**Semantics:**
```
vd[l] = sqrtf(vs[l])    for each active lane l
```

**Simulator implementation:**
```c
case V_VSQRT:
    for (int l = 0; l < vl; l++)
        if (ACTIVE(l)) n->v[I->vd][l] = sqrtf(n->v[I->vs][l]);
    break;
```

**Domain:** Input must be non-negative. `sqrtf(x)` for `x < 0` returns `NaN`.

**Uses:**

| Use case | Expression |
|----------|-----------|
| L2 norm of a vector | `sqrt(sum(x^2))` — VFMA then VREDSUM then VSQRT |
| Attention score scaling | `score / sqrt(d_k)` — VSQRT on scalar, VMUL |
| Standard deviation | `sqrt(variance)` in layer normalization |
| RMS normalization | `x / sqrt(mean(x^2) + eps)` — often faster via VRSQRT |

**Layer normalization — where VSQRT appears:**
```
variance = mean((x - mean(x))^2)
std      = sqrt(variance + eps)
y        = (x - mean(x)) / std * gamma + beta
```

VSQRT computes `std` from `variance + eps`. Note that the division by `std` then requires
either VDIV or (preferably) VRECIP + VMUL.

---

### 6.3.4  VRSQRT  vd, vs — Elementwise Reciprocal Square Root

**Semantics:**
```
vd[l] = 1.0f / sqrtf(vs[l])    for each active lane l
```

**Simulator implementation:**
```c
case V_VRSQRT:
    for (int l = 0; l < vl; l++)
        if (ACTIVE(l)) n->v[I->vd][l] = 1.0f / sqrtf(n->v[I->vs][l]);
    break;
```

### Why VRSQRT when you already have VSQRT and VRECIP?

Because the combination `VSQRT then VRECIP` costs `2 × passes × 4 = 8 × passes` cycles,
while VRSQRT costs `passes × 4`. Normalization code almost always needs `1/sqrt(x)` rather
than `sqrt(x)` alone:

```
x / sqrt(var + eps)   =   x * (1 / sqrt(var + eps))
                       =   x * rsqrt(var + eps)
```

The `rsqrt` form avoids the subsequent VRECIP entirely:

```c
/* Without VRSQRT (expensive): */
{ .op=V_VSQRT,  .vd=4, .vs=3 }        /* v4 = sqrt(var + eps)      */
{ .op=V_VRECIP, .vd=4, .vs=4 }        /* v4 = 1/sqrt(var+eps)      */
{ .op=V_VMUL,   .vd=0, .vs1=0, .vs2=4 }  /* v0 = x / sqrt(var+eps) */
/* Cost: 4 + 4 + 1 = 9 passes for passes=1 */

/* With VRSQRT (fast): */
{ .op=V_VRSQRT, .vd=4, .vs=3 }        /* v4 = 1/sqrt(var+eps)      */
{ .op=V_VMUL,   .vd=0, .vs1=0, .vs2=4 }  /* v0 = x / sqrt(var+eps) */
/* Cost: 4 + 1 = 5 passes for passes=1 */
```

**RMS Norm (a simpler layernorm variant):**
```
rms_norm(x) = x / sqrt(mean(x^2) + eps) * gamma

Instructions:
  VFMA  v1, v0, v0     v1 += v0^2    (after zeroing v1; accumulate x^2)
  VREDSUM s9, v1        s9 = sum(x^2)
  /* compute mean: s9 /= N — but N is known at compile time, so SMULI by 1/N is better */
  SRECIP  s10, s_N      s10 = 1/N
  VBCAST  v2, s10
  VMUL    v2, v1, v2    v2 = x^2 / N  (one per lane — not needed if done at scalar level)
  /* or: just broadcast sum and multiply by 1/N at scalar level */
  → VRSQRT v3, v2       v3 = 1/sqrt(mean_x2 + eps)
  → VMUL   v0, v0, v3   v0 = x * rsqrt(mean_x2 + eps)
```

---

### 6.3.5  VRECIP  vd, vs — Elementwise Reciprocal

**Semantics:**
```
vd[l] = 1.0f / vs[l]    for each active lane l
```

**Simulator implementation:**
```c
case V_VRECIP:
    for (int l = 0; l < vl; l++)
        if (ACTIVE(l)) n->v[I->vd][l] = frecip(n->v[I->vs][l]);
    break;
```

VRECIP is the *per-lane* version of SRECIP. Use VRECIP when the denominator differs by lane;
use SRECIP when the denominator is a single scalar produced by a reduction.

**Comparison:**

| When to use | Which instruction |
|-------------|-------------------|
| Divide each lane by a *different* denominator (per-lane) | VRECIP |
| Divide all lanes by the *same* scalar denominator | SRECIP + VBCAST + VMUL |
| `1 / sqrt(x[l])` for each lane | VRSQRT |
| Divide one vector by another elementwise | VDIV or VRECIP + VMUL |

---

## 6.4  Compare and Select

Compare and select instructions are the basis of branchless conditional logic on the vector
core. Unlike a scalar CPU, the vector core cannot branch per-lane (that would require 16
parallel program counters). Instead, it computes a *mask* via VCMP and uses VSEL to pick
between two alternatives. This is the vector equivalent of the ternary operator `cond ? a : b`.

### The predication pipeline

```
  Input vectors
       │
       ▼
  ┌─────────┐        produces a mask register (1 bit per lane)
  │  VCMP   │ ─────────────────────────────────────────────────┐
  └─────────┘                                                  │
                                                               ▼
  ┌─────────┐   ┌─────────┐                           ┌──────────────┐
  │  vs1    │   │  vs2    │                           │   m[mask]    │
  └────┬────┘   └────┬────┘                           └──────┬───────┘
       │              │                                       │
       └──────────────┼───────────────────────────────────────┘
                      ▼
                 ┌─────────┐
                 │  VSEL   │  vd[l] = mask[l] ? vs1[l] : vs2[l]
                 └─────────┘
```

### 6.4.1  VCMP  md, vs1, vs2, cmp — Vector Compare

**Semantics:**
```
m[md][l] = (vs1[l]  cmp  vs2[l])    for each lane l in [0, vl)
```
where `cmp` ∈ {`CMP_LT`, `CMP_LE`, `CMP_EQ`, `CMP_NE`, `CMP_GT`, `CMP_GE`}.

The result is written to a *mask register* `m[md]` (not a vector register). Mask registers
have one bit per lane and are readable by VSEL, `@mask` prefixes, and the mask logic
instructions (VMASK_AND, VMASK_OR, VMASK_NOT).

**Simulator implementation:**
```c
case V_VCMP:
    for (int l = 0; l < vl; l++) {
        float a = n->v[I->vs1][l], b = n->v[I->vs2][l]; int r = 0;
        switch (I->cmp) {
        case CMP_LT: r = a < b;  break;
        case CMP_LE: r = a <= b; break;
        case CMP_EQ: r = a == b; break;
        case CMP_NE: r = a != b; break;
        case CMP_GT: r = a > b;  break;
        case CMP_GE: r = a >= b; break;
        }
        n->m[I->md][l] = (uint8_t)r;
    }
    break;
```

**cmpkind_t enum (from `npu.h`):**
```c
typedef enum { CMP_LT, CMP_LE, CMP_EQ, CMP_NE, CMP_GT, CMP_GE } cmpkind_t;
```

**Example — finding all positive lanes:**
```c
{ .op=V_SFI,    .rd=6,  .fimm=0.0 }
{ .op=V_VBCAST, .vd=3,  .rs=6 }
{ .op=V_VCMP,   .md=0, .vs1=0, .vs2=3, .cmp=CMP_GT }
/* m0[l] = 1 if v0[l] > 0.0, else 0 */
```

**Producing the zero vector for comparison:** The most common comparison operand is zero.
Since there is no instruction that produces an immediate zero in a vector register, the
standard idiom is `SFI s_zero, 0.0` + `VBCAST v_zero, s_zero`. If `v_zero` is pinned to a
register (say v31) and re-used throughout the kernel, this overhead is paid only once.

**Floating-point equality warning:** `CMP_EQ` on floats is almost never what you want in
production (two floats computed differently are rarely bit-identical). Use it only when
comparing to exact constants (e.g., 0.0 or 1.0 as sentinels) or when you know the values
are from the same computation path.

**Assembly syntax:**
```asm
VCMP m0, v0, v3, GT
```

---

### 6.4.2  VSEL  vd, mask, vs1, vs2 — Vector Select

**Semantics:**
```
vd[l] = m[mask][l] ? vs1[l] : vs2[l]    for each lane l
```

**Simulator implementation:**
```c
case V_VSEL:
    for (int l = 0; l < vl; l++)
        n->v[I->vd][l] = n->m[I->mask][l] ? n->v[I->vs1][l] : n->v[I->vs2][l];
    break;
```

Note: VSEL does *not* respect the `use_mask`/`ACTIVE` predicate — it always writes all `vl`
lanes. It uses its own `mask` field (a mask register index) rather than the instruction-level
predicate. This is intentional: VSEL *is* the selector; applying a second layer of masking on
top of it would be redundant.

**ReLU (the simplest useful example):**
```c
/* Build v_zero */
{ .op=V_SFI,    .rd=6,  .fimm=0.0 }
{ .op=V_VBCAST, .vd=3,  .rs=6 }

/* Compare: m0[l] = v0[l] > 0 */
{ .op=V_VCMP,   .md=0, .vs1=0, .vs2=3, .cmp=CMP_GT }

/* Select: v0[l] = (m0[l]) ? v0[l] : 0  */
{ .op=V_VSEL,   .vd=0, .mask=0, .vs1=0, .vs2=3 }
/* v0 is now ReLU(v0) */
```

From `demo_predication` (`examples.c` line 191):
```c
{ .op=V_VSEL, .vd=2, .vs1=0, .vs2=1, .mask=0 }  /* v2 = m0 ? v0 : v1 */
```

**Branchless if-else — the general pattern:**
```
if (condition[l]) {
    result[l] = then_value[l];
} else {
    result[l] = else_value[l];
}

→ VCMP  m_cond, v_condition, v_zero, GT
→ ... compute then_value into v_then ...
→ ... compute else_value into v_else ...
→ VSEL  v_result, m_cond, v_then, v_else
```

The key insight is that *both branches compute*, unlike a scalar if-else which skips one
path. This means VSEL is only beneficial when both branches are cheap. For expensive branches
(e.g., where one path involves VEXP), it may be cheaper to use the predication `@mask` prefix
on the expensive instruction so that masked-out lanes skip it.

**Predicated VADD after VSEL (from `demo_predication`):**
```c
/* m0 = (v0 > 0) */
{ .op=V_VSEL, .vd=2, .vs1=0, .vs2=1, .mask=0 },    /* v2 = ReLU(v0)         */
{ .op=V_SFI,  .rd=7, .fimm=100.0 },
{ .op=V_VBCAST, .vd=3, .rs=7 },                     /* v3 = 100.0            */
/* Add 100 ONLY to positive lanes (mask-gated VADD): */
{ .op=V_VADD, .vd=2, .vs1=2, .vs2=3, .use_mask=1, .mask=0 },
/* v2[l] = (v0[l] > 0) ? v0[l] + 100 : 0 */
```

This shows the two predication mechanisms working together: VSEL gives the branchless ReLU,
and `@m0` on VADD gates the bias addition to only the lanes where the original value was
positive.

---

## 6.5  Type Conversion — VCVT

### 6.5.1  VCVT  vd, vs, mode — Vector Type Convert

**Semantics:**
```
vd[l] = convert(vs[l], mode)    for each active lane l
```
where `mode` ∈ {`CVT_F32`, `CVT_I8`, `CVT_I32`}.

**Simulator implementation:**
```c
case V_VCVT:
    for (int l = 0; l < vl; l++) if (ACTIVE(l)) {
        float x = n->v[I->vs][l];
        switch (I->cvt) {
        case CVT_I8:
            { float r = roundf(x); if (r > 127) r = 127; if (r < -128) r = -128;
              n->v[I->vd][l] = r; }
            break;
        case CVT_I32:
            n->v[I->vd][l] = roundf(x);
            break;
        default:  /* CVT_F32 */
            n->v[I->vd][l] = x;
            break;
        }
    }
    break;
```

**cvtmode_t enum (from `npu.h`):**
```c
typedef enum { CVT_F32, CVT_I8, CVT_I32 } cvtmode_t;
```

**Important: all vector registers hold float internally.** The `v` array in the simulator is
declared as `float v[NVREG][VLEN_MAX]`. There is no int8 or int32 vector register file.
`VCVT` does not change the storage format — it applies rounding and clamping *within the
float representation*:

```
CVT_I8:  round to nearest integer, clamp to [-128, 127], store as float
CVT_I32: round to nearest integer, store as float
CVT_F32: identity (no-op semantically)
```

This means VCVT models the *semantic effect* of quantization (rounding + clamping) without
changing the physical storage. The binary encoder/decoder (`npu_asm.c`) would distinguish
the logical type in the instruction encoding; a future hardware implementation would use the
actual int8 lanes.

### The quantized inference pipeline

```
┌──────────────────────────────────────────────────────────────────┐
│                    Quantized Inference Flow                       │
│                                                                   │
│  int8 weights ──── VLOAD ──→ v_w (floats with integer values)   │
│                                │                                  │
│  scale factor ──── SLF  ──→ s_scale                              │
│                    VBCAST ──→ v_scale                             │
│                                │                                  │
│  v_w_f32 ────── VMUL(v_w, v_scale) ──→ v_w_dequant              │
│                                                                   │
│  float activations                                                │
│       │                                                           │
│       ▼                                                           │
│  VFMA(accumulator, v_w_dequant, v_act)                           │
│       │                                                           │
│       ▼                                                           │
│  VCVT(CVT_I8)  ← quantize output for next layer                 │
└──────────────────────────────────────────────────────────────────┘
```

**Why quantize at all?** int8 weights take one-quarter the memory bandwidth of float32. For
memory-bound operations (which most matrix multiplications are, at inference time), reducing
bandwidth by 4× roughly quadruples throughput. The accuracy loss from int8 quantization is
typically under 1% on well-trained models.

**Dequantization kernel snippet:**
```c
/* Assume v0 loaded from int8 weights stored as float (values in [-128, 127]) */
{ .op=V_SLF,    .rd=5,  .use_base=1, .rs=1, .imm=16 } /* s5 = scale from args  */
{ .op=V_VBCAST, .vd=3,  .rs=5 }                       /* v3[l] = scale         */
{ .op=V_VMUL,   .vd=0,  .vs1=0, .vs2=3 }              /* v0 = w * scale = w_f32*/
/* Now v0 holds dequantized float32 weights, ready for VFMA */
```

**Quantization before storing (output quantization):**
```c
/* v0 holds float32 accumulator output */
{ .op=V_SLF,    .rd=5, .use_base=1, .rs=1, .imm=20 }  /* s5 = inv_scale        */
{ .op=V_VBCAST, .vd=3, .rs=5 }
{ .op=V_VMUL,   .vd=0, .vs1=0, .vs2=3 }               /* v0 /= scale           */
{ .op=V_VCVT,   .vd=0, .vs=0,  .cvt=CVT_I8 }          /* round+clamp to int8   */
{ .op=V_VSTORE, .vs=0, .use_base=1, .rs=4 }           /* store (as float bits) */
```

---

## 6.6  Instruction Reference Summary Table

| Instruction | Semantics | Cycle cost (vl≤16) | Primary use case |
|-------------|-----------|-------------------|-----------------|
| `VADD vd, vs1, vs2` | `vd[l] = vs1[l] + vs2[l]` | 1 | residual, bias, partial sums |
| `VSUB vd, vs1, vs2` | `vd[l] = vs1[l] - vs2[l]` | 1 | softmax stability, error |
| `VMUL vd, vs1, vs2` | `vd[l] = vs1[l] * vs2[l]` | 1 | scale, gate, normalize |
| `VDIV vd, vs1, vs2` | `vd[l] = vs1[l] / vs2[l]` | 4 | per-lane division |
| `VMAX vd, vs1, vs2` | `vd[l] = max(vs1[l], vs2[l])` | 1 | ReLU, clip lower |
| `VMIN vd, vs1, vs2` | `vd[l] = min(vs1[l], vs2[l])` | 1 | clip upper, clamp |
| `VFMA vd, vs1, vs2` | `vd[l] += vs1[l] * vs2[l]` | 1 | dot product, conv, poly |
| `VABS vd, vs` | `vd[l] = \|vs[l]\|` | 1 | L1 norm, MAE, gradient clip |
| `VNEG vd, vs` | `vd[l] = -vs[l]` | 1 | negate, flip |
| `VEXP vd, vs` | `vd[l] = exp(vs[l])` | 4 | softmax, sigmoid, GELU |
| `VLOG vd, vs` | `vd[l] = log(vs[l])` | 4 | cross-entropy, KL div |
| `VSQRT vd, vs` | `vd[l] = sqrt(vs[l])` | 4 | L2 norm, attn scaling |
| `VRSQRT vd, vs` | `vd[l] = 1/sqrt(vs[l])` | 4 | fast layer/RMS norm |
| `VRECIP vd, vs` | `vd[l] = 1/vs[l]` | 4 | per-lane inversion |
| `VCMP md, vs1, vs2` | `m[md][l] = vs1[l] cmp vs2[l]` | 1 | branching, masking |
| `VSEL vd, mask, vs1, vs2` | `vd[l] = m[mask][l] ? vs1[l] : vs2[l]` | 1 | branchless if-else |
| `VCVT vd, vs, mode` | round/clamp to int8 or int32 | 2 | quantize, dequantize |

Cycle costs assume `vl ≤ HW_LANES = 16` (one pass). For `vl = N × HW_LANES`, multiply by N.

---

## 6.7  Composition Examples

The real test of the compute primitive set is whether it is expressive enough to build
*any* activation function or loss term without hardware support. This section shows three
worked compositions.

### 6.7.1  Sigmoid — built from VNEG + VEXP + VADD + VRECIP

```
sigmoid(x) = 1 / (1 + exp(-x))
```

```c
/* v0 = input x */

/* Step 1: -x */
{ .op=V_VNEG,   .vd=1, .vs=0 }          /* v1 = -x              */

/* Step 2: exp(-x) */
{ .op=V_VEXP,   .vd=1, .vs=1 }          /* v1 = exp(-x)         */

/* Step 3: 1 + exp(-x) */
{ .op=V_SFI,    .rd=5, .fimm=1.0 }
{ .op=V_VBCAST, .vd=2, .rs=5 }          /* v2 = 1.0             */
{ .op=V_VADD,   .vd=1, .vs1=1, .vs2=2 }  /* v1 = 1 + exp(-x)   */

/* Step 4: 1 / (1 + exp(-x)) */
{ .op=V_VRECIP, .vd=0, .vs=1 }          /* v0 = sigmoid(x)      */
```

Total: 6 instructions (SFI + VBCAST are shared across many sigmoid calls if v2 is pinned).
Cycle cost: 1 (VNEG) + 4 (VEXP) + 1 (VADD) + 4 (VRECIP) = 10 cycles for vl=16.

### 6.7.2  GELU (approximate) — built from VMUL + VFMA + VTANH-equivalent

The GELU (Gaussian Error Linear Unit) used in BERT and GPT-2:
```
GELU(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
```

Since `tanh(z) = (exp(2z) - 1) / (exp(2z) + 1)`, and the vector core has VEXP:

```c
/* Constants */
float c1 = sqrtf(2.0f / M_PI);     /* ≈ 0.7979 */
float c2 = 0.044715f;
float c3 = 0.5f;

/* v0 = x (input) */

/* Step 1: x^3 = x * x * x */
{ .op=V_VMUL, .vd=1, .vs1=0, .vs2=0 }   /* v1 = x^2 */
{ .op=V_VMUL, .vd=1, .vs1=1, .vs2=0 }   /* v1 = x^3 */

/* Step 2: 0.044715 * x^3 */
{ .op=V_SFI,    .rd=5, .fimm=0.044715 }
{ .op=V_VBCAST, .vd=2, .rs=5 }
{ .op=V_VMUL,   .vd=1, .vs1=1, .vs2=2 }  /* v1 = 0.044715 * x^3 */

/* Step 3: x + 0.044715 * x^3 */
{ .op=V_VADD, .vd=1, .vs1=0, .vs2=1 }    /* v1 = x + 0.044715*x^3 */

/* Step 4: sqrt(2/pi) * (...) */
{ .op=V_SFI,    .rd=6, .fimm=0.7979 }
{ .op=V_VBCAST, .vd=2, .rs=6 }
{ .op=V_VMUL,   .vd=1, .vs1=1, .vs2=2 }  /* v1 = c1 * (x + c2*x^3) */

/* Step 5: tanh(v1) via tanh(z) = 1 - 2/(exp(2z)+1) */
/* 2*v1 */
{ .op=V_VADD,   .vd=1, .vs1=1, .vs2=1 }  /* v1 = 2*z             */
{ .op=V_VEXP,   .vd=1, .vs=1 }            /* v1 = exp(2z)         */
/* exp(2z) + 1 */
{ .op=V_VADD,   .vd=2, .vs1=1, .vs2=v_one } /* v2 = exp(2z)+1    */
/* 2 / (exp(2z)+1) */
{ .op=V_VRECIP, .vd=2, .vs=2 }            /* v2 = 1/(exp(2z)+1)  */
{ .op=V_VADD,   .vd=2, .vs1=2, .vs2=2 }   /* v2 = 2/(exp(2z)+1)  */
{ .op=V_SFI,    .rd=7, .fimm=1.0 }
{ .op=V_VBCAST, .vd=3, .rs=7 }
{ .op=V_VSUB,   .vd=2, .vs1=v_one, .vs2=2 } /* v2 = 1 - 2/(exp(2z)+1) = tanh(z) */

/* Step 6: 1 + tanh(z) */
{ .op=V_VADD, .vd=2, .vs1=v_one, .vs2=2 } /* v2 = 1 + tanh(z) */

/* Step 7: 0.5 * x * (1 + tanh(z)) */
{ .op=V_VMUL, .vd=0, .vs1=0, .vs2=2 }     /* v0 = x * (1+tanh) */
{ .op=V_SFI,    .rd=8, .fimm=0.5 }
{ .op=V_VBCAST, .vd=3, .rs=8 }
{ .op=V_VMUL, .vd=0, .vs1=0, .vs2=3 }     /* v0 = 0.5 * x * (1+tanh) = GELU(x) */
```

This is 26 instructions and ~30 cycles for vl=16. A compiler would merge constant VBCAST
instructions and reuse pinned zero/one registers to bring this down to around 20 instructions.
The point is that GELU — an activation function that was not known when this ISA was designed —
is expressible from first principles without a `VGELU` opcode.

### 6.7.3  Layer Normalization — combining all compute primitives

LayerNorm is the composition that exercises the most compute primitives in a single kernel:

```
y = (x - mean(x)) / sqrt(var(x) + eps) * gamma + beta

where:
  mean = sum(x) / N
  var  = sum((x - mean)^2) / N
```

**Full kernel sketch:**
```c
/* Prologue: load args — ptr_x, ptr_y, ptr_gamma, ptr_beta, N, eps */

/* 1. Load input row into v0, set VL = N */
VLOAD   v0, 0(s_x)

/* 2. Compute mean = sum(x) / N */
VREDSUM s_sum, v0           /* s_sum = sum(x[l]) */
SMULI   s_N_inv, s_one, imm=K  /* or: compute 1/N as a constant and broadcast */
/* Multiply s_sum by 1/N (or: SLF from args if 1/N is precomputed on host) */
SRECIP  s_mean, s_sum       /* WRONG — SRECIP computes 1/s_sum, not s_sum/N */
/* Better: host precomputes 1/N and puts it in args; SLF loads it */
SLF     s_inv_N, offset(s1)    /* s_inv_N = 1/N from args */
VBCAST  v1, s_sum              /* v1 = sum(x) in all lanes */
VMUL    v1, v1, v_inv_N        /* v1 = sum(x)/N = mean for all l */

/* 3. Subtract mean: v0 = x - mean */
VSUB    v0, v0, v1

/* 4. Compute variance = sum((x-mean)^2) / N */
VFMA    v_sq, v0, v0       /* v_sq[l] = (x[l]-mean)^2 (after zeroing v_sq) */
VREDSUM s_var_sum, v_sq
/* Again multiply by 1/N — already loaded as s_inv_N */
VBCAST  v2, s_var_sum
VMUL    v2, v2, v_inv_N    /* v2 = variance in all lanes */

/* 5. Add epsilon: var + eps */
SLF     s_eps, offset2(s1)
VBCAST  v3, s_eps
VADD    v2, v2, v3         /* v2 = var + eps */

/* 6. Reciprocal square root: 1/sqrt(var+eps) */
VRSQRT  v2, v2             /* v2 = rsqrt(var+eps) */

/* 7. Normalize: (x - mean) * rsqrt(var+eps) */
VMUL    v0, v0, v2         /* v0 = x_norm */

/* 8. Scale and shift: gamma * x_norm + beta */
VLOAD   v_gamma, 0(s_gamma)
VLOAD   v_beta,  0(s_beta)
VFMA    v_beta, v0, v_gamma   /* v_beta += v0 * v_gamma = beta + gamma*x_norm */
/* Now v_beta holds the result (gamma*x_norm + beta) */
VSTORE  v_beta, 0(s_y)
KEXIT
```

This kernel uses: VLOAD, VSETVL, VREDSUM, VBCAST, VMUL, VSUB, VFMA, VADD, VRSQRT, VSTORE,
SLF, SRECIP — 12 of the 17 Section 10 instructions plus 5 memory/scalar helper instructions.
It cannot be expressed without a general compute primitive set.

---

## 6.8  Design Rationale: Why This Particular Set of 17 Instructions?

### Completeness for neural networks

The 17 instructions in Section 10 were chosen to be *sufficient* for all commonly deployed
neural network operations without requiring extensions. The test:

| Neural network primitive | Built from Section 10 instructions |
|--------------------------|-----------------------------------|
| ReLU | VMAX(x, 0) |
| LeakyReLU | VMUL + VMAX |
| Sigmoid | VNEG + VEXP + VADD + VRECIP |
| Tanh | VEXP + arithmetic |
| GELU | VMUL + VEXP + arithmetic |
| Softmax | VREDMAX + VSUB + VEXP + VREDSUM + SRECIP + VMUL |
| LayerNorm | VREDSUM + VSUB + VFMA + VRSQRT + VMUL + VADD |
| RMSNorm | VFMA + VRSQRT + VMUL |
| Attention score | VMUL + VRSQRT + VSUB + VEXP (softmax) |
| L1 norm | VABS + VREDSUM |
| L2 norm | VFMA + VREDSUM + VSQRT |
| Quantize | VMUL + VCVT(CVT_I8) |
| Dequantize | VMUL |

No operation in this table requires a hardware instruction that is not in Section 10 (plus
the reductions in Section 11 and the memory instructions in Section 9).

### Cost trade-offs

**VEXP is expensive (4+ passes × 4 = 16+ cycles for vl=16), but it is essential.** An NPU
without hardware VEXP would have to approximate it with a polynomial (using 6–8 VFMA
instructions), which takes 6–8 cycles — only marginally cheaper than a hardware approximation
at 16 cycles, and less accurate. Hardware VEXP pays for itself.

**VDIV is the one instruction to avoid.** It is 4× more expensive than VMUL, and in practice
it can almost always be replaced with VRECIP + VMUL (when the denominator is per-lane) or
SRECIP + VBCAST + VMUL (when the denominator is scalar). VDIV exists because some code
patterns genuinely need per-lane division and the programmer deserves a clear opcode, but the
compiler should prefer the reciprocal form.

**VFMA is free compared to VMUL + VADD.** On silicon, FMA is the same datapath as multiply
followed by add, with the adder's pipeline fused. It costs the same number of cycles as one
VMUL (in a pipelined implementation). The simulator charges `passes` for VFMA, the same as
VMUL. The instruction set would be strictly worse if VFMA did not exist — there is no
downside to including it.

---

## 6.9  Interaction with the Predication System

Every arithmetic instruction in Section 10 (except VSEL) respects the `@mask` predicate
prefix. When `use_mask = 1` and a mask register index is specified, the instruction skips
writing to lanes where `m[mask][l] == 0`.

```c
/* From npu.c — the ACTIVE macro used by every compute instruction */
const uint8_t *pm = I->use_mask ? n->m[I->mask] : NULL;
#define ACTIVE(L)  (pm == NULL || pm[(L)])

/* Applied in every compute case: */
case V_VADD:
    for (int l = 0; l < vl; l++)
        if (ACTIVE(l)) n->v[I->vd][l] = n->v[I->vs1][l] + n->v[I->vs2][l];
    break;
```

**The cost of predication.** Masking does *not* reduce the cycle cost in the simulator — all
`vl` lanes still iterate, the mask check is just a conditional write. On real hardware,
predication similarly does not reduce power if the ALU evaluates all lanes regardless;
whether it gates the write-back is microarchitecture-dependent. The key benefit of predication
is *correctness*: it prevents writes to lanes that should not be modified, enabling correct
handling of boundary conditions and data-dependent control flow without program counter
divergence.

**The predicated demo:**
```c
/* demo_predication: y = (x > 0) ? x + 100 : 0 */

/* Step 1: ReLU via VSEL (no mask needed — both paths are cheap) */
{ .op=V_VCMP,  .md=0, .vs1=0, .vs2=1, .cmp=CMP_GT }, /* m0 = x > 0 */
{ .op=V_VSEL,  .vd=2, .vs1=0, .vs2=1, .mask=0 },      /* v2 = x>0?x:0 */

/* Step 2: add 100 ONLY to positive lanes — predicated VADD */
{ .op=V_VADD, .vd=2, .vs1=2, .vs2=3, .use_mask=1, .mask=0 },
/* Lanes where m0=0 (the non-positive ones) skip the write; v2 unchanged there */
```

The result after these three instructions:
```
Input v0:   [-2, -1, 0, 1,  2,  3, -3, 4]
Mask m0:    [ 0,  0, 0, 1,  1,  1,  0, 1]   (x > 0)
After VSEL: [ 0,  0, 0, 1,  2,  3,  0, 4]   (ReLU)
After @VADD:[ 0,  0, 0,101,102,103,  0,104]  (add 100 to positive lanes only)
```

---

*This chapter has covered all 17 compute primitive instructions of Section 10. The next
chapter (Chapter 7) covers Section 11 — Reductions (VREDSUM, VREDMAX, VREDMIN, VREDAND,
VREDOR, VSCAN) — and Section 12 — Predication and Engine-Side Control Flow (VMASK_*, BR,
BRZ, BRNZ, LOOP, KEXIT).*
