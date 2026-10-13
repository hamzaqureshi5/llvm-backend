# Chapter 8: Predication and Control Flow

> **ISA Reference:** Section 12 of `npu-simulator/docs/fully_prog_npu.md`
> **Simulator source:** `npu-simulator/src/npu.c`, lines 267–278
> **Worked example:** `npu-simulator/examples/examples.c`, `demo_predication()`

---

## 8.1 The Problem: Data-Dependent Behavior in SIMD Machines

The fundamental promise of a vector processor is that the same instruction is applied to
all lanes simultaneously. This makes computation very efficient when all lanes are doing
the same thing — and seems to make it impossible when different lanes need to do different
things.

Consider ReLU, one of the most common activation functions in neural networks:

```
y[i] = (x[i] > 0) ? x[i] : 0
```

On a scalar CPU, this is trivially expressed as an `if` statement. The CPU evaluates the
condition, takes one branch or the other, and moves on. Different elements of `x` may take
different branches, and the CPU handles each element independently.

On a vector processor processing 16 elements at once, a naive approach would serialize:

```
// Naive approach — WRONG for a SIMD machine:
for each lane l:
    if v0[l] > 0:
        v1[l] = v0[l]   // "positive" branch
    else:
        v1[l] = 0        // "zero" branch
```

This is exactly what you get when you program a GPU with `if` statements and the warp
diverges — half the SIMD units sit idle while one branch executes, then the other half
execute the other branch. This "warp divergence" is one of the primary performance pitfalls
in GPU programming.

The NPU vector core avoids warp divergence entirely through a different mechanism:
**predication**. Instead of branching, the hardware executes instructions in a mode where
each lane independently decides whether to apply the instruction's result to the output,
based on a bit in a _mask register_. Masked-out lanes simply do not write their result —
they leave the destination unchanged.

This chapter covers the two complementary tools for data-dependent computation:

1. **Predication** — mask registers and masked instruction execution (per-lane
   fine-grained control)
2. **Control flow** — scalar branches and hardware loops (whole-kernel coarse-grained
   control)

---

## 8.2 The Mask Register File

The NPU vector core has eight mask registers: `m0` through `m7`.

**Physical layout:**
```
Each mask register holds VLEN_MAX bits (one per lane), stored as one uint8_t per bit:

m0:  [ bit for lane 0 ] [ bit for lane 1 ] [ bit for lane 2 ] ... [ bit for lane vl-1 ]
     m[0][0]              m[0][1]              m[0][2]               m[0][vl-1]
     uint8_t              uint8_t              uint8_t               uint8_t
```

From `npu.h` (line 157):
```c
uint8_t  m[NMREG][VLEN_MAX];   /* NMREG=8, VLEN_MAX=256 */
```

Each element is `uint8_t` rather than a true 1-bit value. In a hardware implementation
you would pack 8 bits per byte to save storage, but the simulator uses one byte per lane
for simplicity. The values stored are always 0 (lane inactive) or 1 (lane active).

**Initialization:** At kernel launch, all mask registers are initialized to all-ones (all
lanes active). From `run_vector_instance()` in `npu.c` (lines 143–145):

```c
memset(n->v, 0, sizeof n->v);
memset(n->m, 0, sizeof n->m);    /* masks start zero... */
memset(n->s, 0, sizeof n->s);
```

Wait — masks start at zero? Re-read line 144: `memset(n->m, 0, ...)` sets all bytes to
zero, which means all masks start fully inactive. This is the initialization from
`npu_reset()`. But the mask instructions (VMASK_SET, etc.) populate them on demand.

The key insight is: **the predicate mask on an instruction is opt-in**. An instruction
uses a mask only when `I->use_mask = 1`. If `use_mask = 0` (the default), all lanes are
active regardless of the mask register contents. This is represented by the `ACTIVE` macro
in `npu.c` (line 165):

```c
#define ACTIVE(L) (pm == NULL || pm[(L)])
```

Where `pm` is `NULL` when `use_mask = 0`, and `pm = n->m[I->mask]` when `use_mask = 1`.
When `pm` is NULL, `pm == NULL` is true and the entire expression short-circuits to true
for all lanes.

---

## 8.3 Building and Combining Masks

Before you can use a mask to gate instruction execution, you need to populate the mask
register with meaningful per-lane bits. There are four mask instructions.

### 8.3.1 VMASK_SET — Float Vector to Mask

**Syntax:** `VMASK_SET md, vs`

**Semantics:**
```
for each lane l in 0..vl-1:
    m[md][l] = (vs[l] != 0.0f) ? 1 : 0
```

**Simulator implementation** (`npu.c`, line 267):
```c
case V_VMASK_SET:
    for (int l = 0; l < vl; l++)
        n->m[I->md][l] = (uint8_t)(n->v[I->vs][l] != 0.0f);
    break;
```

This converts a float vector into a mask by treating zero as "false" and anything nonzero
as "true". The comparison `!= 0.0f` is an exact IEEE 754 comparison — negative zero
(-0.0f) would be treated as false because `-0.0f == 0.0f` in IEEE 754.

**Use case:** When you have computed a float vector that encodes boolean conditions (e.g.,
the output of a VCMP stored to a float register via some path), VMASK_SET converts it to
the mask register format required by predicated instructions.

**More common pattern:** Use `VCMP` directly, which writes to a mask register without
going through a float intermediate. `VMASK_SET` is for cases where you already have a
float vector of 0/1 values.

### 8.3.2 VCMP — The Primary Mask-Building Instruction

Although `VCMP` is listed in Section 10 (Compute Primitives), it is the canonical way to
build masks. Its semantics are:

**Syntax:** `VCMP md, vs1, vs2, cmp`

**Semantics:**
```
for each lane l in 0..vl-1:
    m[md][l] = compare(vs1[l], vs2[l], cmp_kind)
```

Where `cmp_kind` is one of:

| Enum | Meaning | Example |
|------|---------|---------|
| `CMP_LT` | `<` | strict less than |
| `CMP_LE` | `<=` | less than or equal |
| `CMP_EQ` | `==` | equality |
| `CMP_NE` | `!=` | inequality |
| `CMP_GT` | `>` | strict greater than |
| `CMP_GE` | `>=` | greater than or equal |

**Simulator implementation** (`npu.c`, lines 228–238):
```c
case V_VCMP:
    for (int l = 0; l < vl; l++) {
        float a = n->v[I->vs1][l], b = n->v[I->vs2][l]; int r = 0;
        switch (I->cmp) {
        case CMP_LT: r = a <  b; break;  case CMP_LE: r = a <= b; break;
        case CMP_EQ: r = a == b; break;  case CMP_NE: r = a != b; break;
        case CMP_GT: r = a >  b; break;  case CMP_GE: r = a >= b; break;
        }
        n->m[I->md][l] = (uint8_t)r;
    }
    break;
```

**Pattern — compare against zero:**
```c
{ .op=V_SFI,    .rd=5, .fimm=0.0 },
{ .op=V_VBCAST, .vd=3, .rs=5 },              /* v3 = 0 (zero reference) */
{ .op=V_VCMP,   .md=0, .vs1=0, .vs2=3, .cmp=CMP_GT },  /* m0[l] = (v0[l] > 0) */
```

This three-instruction sequence builds a mask for "positive elements". It is the standard
ReLU/LeakyReLU setup and appears verbatim in `demo_predication()`.

### 8.3.3 VMASK_AND — Intersection of Two Conditions

**Syntax:** `VMASK_AND md, m1, m2`

**Semantics:**
```
for each lane l in 0..vl-1:
    m[md][l] = m[m1][l] & m[m2][l]
```

**Simulator implementation** (`npu.c`, line 268):
```c
case V_VMASK_AND:
    for (int l = 0; l < vl; l++)
        n->m[I->md][l] = n->m[I->m1][l] & n->m[I->m2][l];
    break;
```

`VMASK_AND` implements logical AND on mask bits — a lane is active in the result only if
it was active in both input masks. This encodes the AND of two conditions.

**Use case — compound condition:**
```c
/* m2[l] = (v0[l] > 0) AND (v0[l] < 1.0) -- the unit interval (0,1) */
{ .op=V_VCMP,     .md=0, .vs1=0, .vs2=v_zero, .cmp=CMP_GT },    /* m0 = x > 0   */
{ .op=V_VCMP,     .md=1, .vs1=0, .vs2=v_one,  .cmp=CMP_LT },    /* m1 = x < 1   */
{ .op=V_VMASK_AND,.md=2, .m1=0, .m2=1 },                         /* m2 = m0 & m1 */
```

### 8.3.4 VMASK_OR — Union of Two Conditions

**Syntax:** `VMASK_OR md, m1, m2`

**Semantics:**
```
for each lane l in 0..vl-1:
    m[md][l] = m[m1][l] | m[m2][l]
```

**Simulator implementation** (`npu.c`, line 269):
```c
case V_VMASK_OR:
    for (int l = 0; l < vl; l++)
        n->m[I->md][l] = n->m[I->m1][l] | n->m[I->m2][l];
    break;
```

A lane is active in the result if it was active in either input mask.

**Use case — union of exceptional conditions:**
```c
/* m2[l] = (v0[l] < -5.0) OR (v0[l] > 5.0) -- outside range [-5, 5] */
{ .op=V_VCMP,    .md=0, .vs1=0, .vs2=v_neg5, .cmp=CMP_LT },
{ .op=V_VCMP,    .md=1, .vs1=0, .vs2=v_pos5, .cmp=CMP_GT },
{ .op=V_VMASK_OR,.md=2, .m1=0, .m2=1 },                    /* m2 = out of range */
```

### 8.3.5 VMASK_NOT — Complement of a Condition

**Syntax:** `VMASK_NOT md, m1`

**Semantics:**
```
for each lane l in 0..vl-1:
    m[md][l] = !m[m1][l]     (1 becomes 0, 0 becomes 1)
```

**Simulator implementation** (`npu.c`, line 270):
```c
case V_VMASK_NOT:
    for (int l = 0; l < vl; l++)
        n->m[I->md][l] = !n->m[I->m1][l];
    break;
```

This is the most frequently used mask combinator because it enables the else-branch of a
conditional. If `m0` is "positive lanes", then `VMASK_NOT m1, m0` produces `m1` =
"non-positive lanes". You can then execute both branches of a conditional using `m0` and
`m1` as predicates.

**The pattern for if-else without divergence:**
```c
/* Encode: y[l] = (x[l] > 0) ? x[l] * scale : 0 */

/* 1. Build the positive-lane mask */
{ .op=V_VCMP,     .md=0, .vs1=0, .vs2=v_zero, .cmp=CMP_GT },  /* m0 = positive */
/* 2. Compute the negative branch (no mask, compute on all lanes) */
{ .op=V_SFI,      .rd=7, .fimm=0.0 },
{ .op=V_VBCAST,   .vd=2, .rs=7 },                             /* v2 = 0 (else result) */
/* 3. Compute the positive branch (no mask, compute on all lanes) */
{ .op=V_SFI,      .rd=8, .fimm=2.5 },
{ .op=V_VBCAST,   .vd=3, .rs=8 },                             /* v3 = scale         */
{ .op=V_VMUL,     .vd=4, .vs1=0, .vs2=3 },                    /* v4 = x * scale      */
/* 4. Select: use VSEL for branchless merge */
{ .op=V_VSEL,     .vd=1, .vs1=4, .vs2=2, .mask=0 },           /* v1 = m0 ? v4 : v2   */
```

The key insight is that **both branches are computed on all lanes**, and `VSEL` picks the
correct result per lane based on the mask. This is efficient when both branches are cheap.
When one branch is expensive and rare, predicated writes become more attractive.

---

## 8.4 Using Masks on Arithmetic Instructions

Any arithmetic or memory instruction in the vector core can carry an optional predicate
mask. The mechanism is two fields in the `vinst` struct (from `npu.h`, lines 116–117):

```c
int use_mask, mask;   /* per-lane predicate (Section 12 "@m") */
```

When `use_mask = 1` and `mask = N`, only lanes where `m[N][l] = 1` have their results
written to the destination register. Inactive lanes leave the destination unchanged.

**The ACTIVE macro** from `npu.c` (line 165):
```c
#define ACTIVE(L) (pm == NULL || pm[(L)])
```

Applied to every per-lane operation in all compute instructions:
```c
case V_VADD:
    for (int l = 0; l < vl; l++)
        if (ACTIVE(l))
            n->v[I->vd][l] = n->v[I->vs1][l] + n->v[I->vs2][l];
    break;
```

When `pm == NULL` (no mask), `ACTIVE(l)` is always true and all lanes execute. When
`pm != NULL`, only lanes where `pm[l] != 0` execute.

**Example from `demo_predication()`** (`examples.c`, lines 191–194):
```c
/* v2 = x>0 ? x : 0 (ReLU) via VSEL */
{ .op=V_VSEL,   .vd=2, .vs1=0, .vs2=1, .mask=0 },
/* Then add 100 only to the positive lanes (predicated add) */
{ .op=V_VADD,   .vd=2, .vs1=2, .vs2=3, .use_mask=1, .mask=0 },
```

The `VADD` on the second line adds 100 only to lanes where `m0[l] = 1`. Negative-lane
slots in `v2` keep their value of 0 (set by the preceding `VSEL`).

### 8.4.1 Which Instructions Support Masks

Every instruction that writes a vector register can carry `use_mask`. Checking the
simulator's switch cases confirms this applies to:
- All elementwise binary ops: `VADD`, `VSUB`, `VMUL`, `VDIV`, `VMAX`, `VMIN`, `VFMA`
- All unary ops: `VEXP`, `VLOG`, `VSQRT`, `VRSQRT`, `VRECIP`, `VABS`, `VNEG`, `VCVT`
- Vector copies: `VMOV`
- Broadcast: `VBCAST`
- Stores: `VSTORE` (masked lanes are not written to memory)

Instructions that do NOT use `use_mask`:
- Reductions (`VREDSUM`, `VREDMAX`, etc.) — always operate on all active lanes
- Loads (`VLOAD`) — always loads all vl elements
- Scatter/gather — currently unmaskable (all lanes active)
- Scalar instructions

### 8.4.2 Predication vs. VSEL — When to Use Which

Two patterns implement "conditional write to a register":

**Pattern A — Predicated write:**
```c
{ .op=V_VMUL, .vd=1, .vs1=0, .vs2=2, .use_mask=1, .mask=0 }
/* Only m0-active lanes update v1; other lanes keep their previous value */
```

**Pattern B — VSEL:**
```c
{ .op=V_VMUL, .vd=3, .vs1=0, .vs2=2 },                   /* v3 = x*scale (all lanes) */
{ .op=V_VSEL, .vd=1, .vs1=3, .vs2=1, .mask=0 },           /* v1 = m0 ? v3 : v1       */
```

Pattern A is more efficient when you want to preserve the pre-existing values in inactive
lanes and you do not need a separate register for the "else" path. Pattern B requires an
extra register and instruction but makes the control flow explicit. Use Pattern B (VSEL)
when:
- The "false" result is a different computation, not the existing value of `vd`
- The code is being generated by a compiler where explicit data-flow is cleaner than
  read-modify-write register semantics

---

## 8.5 VSEL — Per-Lane Select

**Syntax:** `VSEL vd, vs1, vs2, mask`  (field `.mask` in the struct)

**Semantics:**
```
for each lane l in 0..vl-1:
    vd[l] = m[mask][l] ? vs1[l] : vs2[l]
```

**Simulator implementation** (`npu.c`, lines 239–241):
```c
case V_VSEL:
    for (int l = 0; l < vl; l++)
        n->v[I->vd][l] = n->m[I->mask][l] ? n->v[I->vs1][l] : n->v[I->vs2][l];
    break;
```

Note that `VSEL` does not use the `use_mask`/`mask` field pair (which is for predicated
execution). Instead, it has its own `.mask` field that selects the mask register to use as
the selector. `VSEL` is not itself predicated — it always writes all `vl` lanes.

`VSEL` is the branchless ternary operator `?:` for vectors. Unlike the predicated write
approach, `VSEL` always requires that both `vs1` (true) and `vs2` (false) are already
computed. This is the "compute both branches, select" pattern.

**GELU approximation using VSEL:**

GELU (Gaussian Error Linear Unit) is approximately `x * 0.5 * (1 + tanh(x * 0.7978 + x^3 * 0.0356))`.
A common fast approximation clips to ReLU for extreme values:

```c
/* if x < -3: gelu ≈ 0; if x > 3: gelu ≈ x; else: compute tanh approximation */
{ .op=V_SFI,      .rd=10, .fimm=-3.0 }, { .op=V_VBCAST, .vd=5, .rs=10 },  /* v5 = -3 */
{ .op=V_SFI,      .rd=11, .fimm=3.0  }, { .op=V_VBCAST, .vd=6, .rs=11 },  /* v6 = +3 */
{ .op=V_SFI,      .rd=12, .fimm=0.0  }, { .op=V_VBCAST, .vd=7, .rs=12 },  /* v7 = 0  */
{ .op=V_VCMP,     .md=0, .vs1=0, .vs2=5, .cmp=CMP_LT },  /* m0 = x < -3             */
{ .op=V_VCMP,     .md=1, .vs1=0, .vs2=6, .cmp=CMP_GT },  /* m1 = x > +3             */
/* ... compute tanh approx into v4 ... */
{ .op=V_VSEL,     .vd=4, .vs1=7, .vs2=4, .mask=0 },      /* if x<-3: 0 else tanh    */
{ .op=V_VSEL,     .vd=4, .vs1=0, .vs2=4, .mask=1 },      /* if x>+3: x else tanh    */
```

---

## 8.6 Engine-Side Control Flow Instructions

The reductions and predication in the previous sections handle data-dependent behavior at
the per-lane level. But the vector core also needs whole-kernel control flow: the ability
to loop, to branch based on a scalar condition, and to exit cleanly.

These instructions operate on the **scalar register file** (`s0..s31`) of the vector core
engine — not the control core's `x0..x31` registers. They are part of the vector kernel's
own program counter logic.

### 8.6.1 BR — Unconditional Branch

**Syntax:** `BR imm`

**Semantics:**
```
pc = imm   (absolute instruction index in the kernel's vinst array)
```

**Simulator implementation** (`npu.c`, line 271):
```c
case V_BR:
    pc = (int)I->imm;
    continue;   /* skip the normal pc++ at the bottom of the loop */
```

The `continue` is critical — it bypasses the `pc++` that normally advances the program
counter, so the branch is taken exactly to `imm` with no off-by-one.

Branch targets are **instruction indices**, not byte offsets. The 5th instruction in the
kernel has index 4 (zero-based). This simplifies the assembler — the text assembler in
`npu_asm.c` resolves labels to instruction indices.

**Use case:** Skip over an else-block when you have already handled the if-block with
predication, or implement an infinite loop with an exit condition inside.

### 8.6.2 BRZ — Branch if Zero

**Syntax:** `BRZ rs, imm`

**Semantics:**
```
if (s[rs].i == 0) pc = imm
```

**Simulator implementation** (`npu.c`, line 272):
```c
case V_BRZ:
    if (n->s[I->rs].i == 0) { pc = (int)I->imm; continue; }
    break;
```

The comparison is `== 0` on the integer interpretation (`.i`) of the scalar register. This
means the "zero" test is an exact integer zero test — a float `0.0f` would pass because
`0.0f` has bit pattern `0x00000000`, which equals integer zero. A float `-0.0f` has bit
pattern `0x80000000`, which does not equal integer zero — be careful.

**Use case:** Loop termination when a decrement-to-zero counter is used. After `SADDI rd,
rd, -1`, use `BRZ rd, END_PC` to exit when the counter hits zero.

**Iteration count = 0 guard:**
```c
{ .op=V_SLW,   .rd=10, .use_base=1, .rs=1, .imm=8 },  /* s10 = n (count) */
{ .op=V_BRZ,   .rs=10, .imm=EXIT_PC },                  /* exit if n==0    */
/* ... loop body ... */
```

### 8.6.3 BRNZ — Branch if Non-Zero

**Syntax:** `BRNZ rs, imm`

**Semantics:**
```
if (s[rs].i != 0) pc = imm
```

**Simulator implementation** (`npu.c`, line 273):
```c
case V_BRNZ:
    if (n->s[I->rs].i != 0) { pc = (int)I->imm; continue; }
    break;
```

This is the "branch back to loop top" instruction. After computing one iteration, decrement
the loop counter and branch back if it is still nonzero.

**Use case:** The back-edge of a counted loop. Paired with BRNZ, the decrement is explicit:

```c
{ .op=V_SADDI, .rd=10, .rs1=10, .imm=-1 },   /* s10-- */
{ .op=V_BRNZ,  .rs=10, .imm=LOOP_TOP_PC },   /* if s10 != 0: back */
```

### 8.6.4 LOOP — Combined Decrement-and-Branch

**Syntax:** `LOOP rs, imm`

**Semantics:**
```
s[rs] = s[rs] - 1
if (s[rs] != 0) pc = imm
```

**Simulator implementation** (`npu.c`, lines 274–277):
```c
case V_LOOP:
    n->s[I->rs].i -= 1;
    if (n->s[I->rs].i > 0) { pc = (int)I->imm; continue; }
    break;
```

Note the simulator uses `> 0` (strictly positive) rather than `!= 0`. This means that if
the counter starts at 1, the loop body executes once (decrement to 0, which is not > 0,
so fall through). If the counter starts at 0, the loop executes zero times? Actually the
loop body has already executed once before reaching LOOP, so this is the "execute-at-least-
once" semantic of a do-while loop.

More precisely: LOOP is placed at the **bottom** of the loop. The counter is initialized
**before** entering the loop body. The first time LOOP is reached, the counter has been
decremented once. The body was executed N times for counter initialized to N.

**Comparison with SADDI + BRNZ:**
```
LOOP rs, imm   (1 instruction, reads and writes rs)
   ≡
SADDI rs, rs, -1    (1 instruction)
BRNZ  rs, imm       (1 instruction)
```

`LOOP` saves one instruction and one program counter advance per iteration. For a kernel
that runs 10,000 iterations, this is a meaningful code density improvement. It also
communicates intent clearly to the assembler / disassembler.

### 8.6.5 KEXIT — Terminate Kernel Instance

**Syntax:** `KEXIT`

**Semantics:** Terminate the current grid instance. No return value.

**Simulator implementation** (`npu.c`, line 278):
```c
case V_KEXIT:
    return;
```

`KEXIT` exits the `run_vector_instance()` function. The control core, which is blocked in
`C_KWAIT`, counts completed instances. When all grid instances have called `KEXIT`, the
control core unblocks and proceeds past `KWAIT`.

**Every kernel must end with KEXIT.** If execution runs off the end of the `vinst` array
(reaches `pc >= k->len`), the while loop exits naturally — the simulator does not fault.
But in hardware, running off the end of the kernel text is undefined behavior. Always
terminate with `KEXIT`.

**What KEXIT does NOT do:**
- It does not flush memory (writes issued before KEXIT are already committed)
- It does not synchronize with other instances (there is no inter-instance synchronization)
- It does not clean up registers (the next instance launched gets a fresh register set)

---

## 8.7 Detailed Loop Example — Processing a Large Vector in Chunks

The vector length register `vl` is set by `VSETVL` and bounds how many elements the
vector instructions process at once. `VLEN_MAX = 256` is the hardware maximum. If your
data is larger than 256 elements, you need a loop.

This example processes an array of 1024 floats in chunks of 16 (vl = 16), applying
`y[i] = ReLU(x[i])`:

```c
static const vinst relu_loop[] = {
    /*
     * Args layout at s1:
     *   [+0]  x_ptr   -- input array base address
     *   [+4]  y_ptr   -- output array base address
     *   [+8]  n       -- total element count (1024 in this example)
     *   [+12] vl_val  -- chunk size (16 in this example)
     */
    { .op=V_SLW,    .rd=2, .use_base=1, .rs=1, .imm=0  },  /* s2 = x_ptr        */
    { .op=V_SLW,    .rd=3, .use_base=1, .rs=1, .imm=4  },  /* s3 = y_ptr        */
    { .op=V_SLW,    .rd=4, .use_base=1, .rs=1, .imm=8  },  /* s4 = n            */
    { .op=V_SLW,    .rd=5, .use_base=1, .rs=1, .imm=12 },  /* s5 = chunk size   */

    /* Set VL = chunk size */
    { .op=V_VSETVL, .rd=9, .rs1=5 },                        /* vl = s5           */

    /* Compute number of iterations = n / chunk_size */
    /* Assumes n is a multiple of chunk_size */
    /* Compute n/chunk via: store n, store chunk, scalar divide (no SDIV — use host) */
    /* Simplification: host stores iteration count in args[3] */
    { .op=V_SLW,    .rd=10, .use_base=1, .rs=1, .imm=16 }, /* s10 = iter_count  */

    /* s6 = current byte offset (starts at 0) */
    { .op=V_SLI,    .rd=6, .imm=0 },                        /* s6 = 0            */

    /* Compute chunk size in bytes = chunk_size * 4 */
    { .op=V_SMULI,  .rd=11, .rs1=5, .imm=4 },               /* s11 = chunk bytes */

    /* LOOP BODY — starts at instruction index 9 (adjust if you add/remove prologue) */
    /* Compute current read/write pointers: s12 = x_ptr + s6, s13 = y_ptr + s6 */
    { .op=V_SADD,   .rd=12, .rs1=2, .rs2=6 },               /* s12 = x_ptr+off   */
    { .op=V_SADD,   .rd=13, .rs1=3, .rs2=6 },               /* s13 = y_ptr+off   */

    /* Load, apply ReLU via predication, store */
    { .op=V_VLOAD,  .vd=0, .use_base=1, .rs=12, .imm=0 },   /* v0 = x chunk      */
    { .op=V_SFI,    .rd=7, .fimm=0.0 },
    { .op=V_VBCAST, .vd=1, .rs=7 },                          /* v1 = 0            */
    { .op=V_VCMP,   .md=0, .vs1=0, .vs2=1, .cmp=CMP_GT },   /* m0 = x > 0        */
    { .op=V_VSEL,   .vd=0, .vs1=0, .vs2=1, .mask=0 },       /* v0 = relu(x)      */
    { .op=V_VSTORE, .vs=0, .use_base=1, .rs=13, .imm=0 },   /* y chunk = v0      */

    /* Advance byte offset by chunk bytes */
    { .op=V_SADD,   .rd=6, .rs1=6, .rs2=11 },               /* s6 += chunk_bytes */

    /* Decrement and branch */
    { .op=V_LOOP,   .rs=10, .imm=9 },                        /* s10--; if>0: goto body */
    { .op=V_KEXIT },
};
```

**Instruction index comments are critical.** The `.imm=9` in the `LOOP` instruction must
match the actual instruction index of the loop body start. If you add or remove
instructions from the prologue, you must update the branch target. This fragility is one
reason compilers use symbolic labels and resolve them during assembly — see
`npu_asm_vector()` in `npu_asm.c`.

---

## 8.8 Predication vs. Branching — When to Use Each

The choice between predicated execution and scalar branches is a recurring decision in
vector kernel design. The NPU provides both; here is the guidance for choosing.

### 8.8.1 Use Predication When:

1. **Both outcomes are cheap.** If the "true" computation takes 3 instructions and the
   "false" computation takes 2 instructions, predicate both — you pay 5 instructions total
   but no branch overhead or serialization.

2. **The condition varies by lane.** If different lanes of the same vector want different
   results, you cannot branch at the kernel level (the whole kernel either branches or does
   not). Predication is the only option.

3. **The condition applies to a minority of lanes.** Even if only 1 lane out of 16 needs
   the true-branch result, predication correctly produces that result without affecting
   the other 15 lanes.

### 8.8.2 Use Scalar Branches (BRZ/BRNZ/LOOP) When:

1. **The entire kernel should skip a code block.** If a scalar condition determines
   whether to execute a whole section of the kernel (e.g., "if this row has all-zero
   input, skip everything"), a branch to `KEXIT` is appropriate.

2. **Looping over outer dimensions.** When a single instance needs to process multiple
   chunks sequentially (because data exceeds `vl`), the loop counter is a scalar quantity
   and `LOOP`/`BRNZ` is the right construct.

3. **The true-branch is very expensive.** If the true-branch requires 50 instructions
   (e.g., a complex fallback path), executing those 50 instructions on all lanes just
   to discard 15 of the 16 results is wasteful. A scalar branch that skips the block
   entirely (if the condition is all-false) is better.

### 8.8.3 The Warp Divergence Comparison

```
GPU (warp-based):                     NPU vector core (predicated):
─────────────────────────────────     ──────────────────────────────────────
if (x[lane] > 0):                     VCMP  m0, v0, v_zero, GT
   PASS 1: all 32 lanes "if" branch   VMUL  @m0  v1, v0, v_scale (masked)
   BUT: lanes where condition is      ← all 16 lanes "execute" in 1 cycle
        false are MASKED (idle)       ← inactive lanes just don't write
PASS 2: all 32 lanes "else" branch    VMASK_NOT m1, m0
   same: lanes where condition is     VBCAST @m1 v1, s_zero (masked)
        true are masked               ← again, 1 cycle for all 16 lanes
   → 2 passes, 2× slowdown
   → This IS the warp divergence      → No slowdown from data-dependent
     problem on GPUs                    conditions — just masked writes
```

The NPU design avoids the serialization penalty of GPU warp divergence. The trade-off is
that wasted compute on masked-out lanes is inherent — if only 1 of 16 lanes needs a
`VEXP`, the hardware still cycles through all 16 lanes, just ignoring the inactive ones.
For the NPU's primary use cases (uniform data transformations with some edge cases), this
is acceptable. For workloads with truly irregular access patterns (e.g., sparse attention
with highly variable density), this trade-off is less favorable.

---

## 8.9 Complete Example — LeakyReLU with Both Branches

LeakyReLU is defined as:
```
y[i] = x[i]         if x[i] >= 0
y[i] = alpha * x[i] if x[i] <  0
```

Where `alpha` is typically 0.01. This requires two distinct computations per lane, making
it an ideal example for predication.

```c
static const vinst leaky_relu[] = {
    /* Prologue: load args */
    { .op=V_SLW,    .rd=2, .use_base=1, .rs=1, .imm=0  },  /* s2 = x_ptr       */
    { .op=V_SLW,    .rd=3, .use_base=1, .rs=1, .imm=4  },  /* s3 = y_ptr       */
    { .op=V_SLW,    .rd=4, .use_base=1, .rs=1, .imm=8  },  /* s4 = n           */
    /* Compute per-instance byte offset */
    { .op=V_SMUL,   .rd=5, .rs1=0, .rs2=4 },               /* s5 = id * n      */
    { .op=V_SMULI,  .rd=5, .rs1=5, .imm=4 },               /* s5 = id * n * 4  */
    { .op=V_SADD,   .rd=6, .rs1=2, .rs2=5 },               /* s6 = x_row_ptr   */
    { .op=V_SADD,   .rd=7, .rs1=3, .rs2=5 },               /* s7 = y_row_ptr   */
    /* Set VL, load input */
    { .op=V_VSETVL, .rd=8, .rs1=4 },
    { .op=V_VLOAD,  .vd=0, .use_base=1, .rs=6, .imm=0 },  /* v0 = x           */
    /* Build masks for both conditions */
    { .op=V_SFI,    .rd=9,  .fimm=0.0 },
    { .op=V_VBCAST, .vd=3, .rs=9 },                        /* v3 = 0 (zero ref) */
    { .op=V_VCMP,   .md=0, .vs1=0, .vs2=3, .cmp=CMP_GE }, /* m0 = x >= 0      */
    { .op=V_VMASK_NOT, .md=1, .m1=0 },                     /* m1 = x < 0       */
    /* True branch (x >= 0): y = x — already in v0, just use predicated copy */
    { .op=V_VMOV,   .vd=1, .vs=0, .use_mask=1, .mask=0 }, /* v1[pos] = x[pos] */
    /* False branch (x < 0): y = alpha * x */
    { .op=V_SFI,    .rd=10, .fimm=0.01 },
    { .op=V_VBCAST, .vd=4, .rs=10 },                       /* v4 = 0.01 (alpha) */
    { .op=V_VMUL,   .vd=5, .vs1=0, .vs2=4 },               /* v5 = alpha*x (all lanes) */
    { .op=V_VMOV,   .vd=1, .vs=5, .use_mask=1, .mask=1 }, /* v1[neg] = v5[neg] */
    /* v1 now holds the complete leaky_relu result */
    { .op=V_VSTORE, .vs=1, .use_base=1, .rs=7, .imm=0 },
    { .op=V_KEXIT },
};
```

**Alternative using VSEL:**

The VSEL version is often cleaner because it avoids the read-modify-write ambiguity of
a predicated VMOV:

```c
/* Replace the two VMOV instructions with a single VSEL: */
{ .op=V_VMUL, .vd=5, .vs1=0, .vs2=4 },              /* v5 = alpha * x  (all lanes) */
{ .op=V_VSEL, .vd=1, .vs1=0, .vs2=5, .mask=0 },     /* v1 = m0 ? x : alpha*x      */
{ .op=V_VSTORE, .vs=1, .use_base=1, .rs=7, .imm=0 },
{ .op=V_KEXIT },
```

The VSEL version is 2 instructions fewer and clearer. The `v5 = alpha * x` computation
runs on all lanes (including the positive ones where the result will be discarded), but
that is only one extra multiply — the trade-off is worthwhile for code clarity.

---

## 8.10 Instruction Reference Table

### Mask Instructions

| Instruction | Syntax | Semantics | Writes | Notes |
|-------------|--------|-----------|--------|-------|
| `VMASK_SET` | `md, vs` | `md[l] = vs[l] != 0.0f` | `m[md]` | Float-to-mask |
| `VMASK_AND` | `md, m1, m2` | `md[l] = m1[l] & m2[l]` | `m[md]` | AND of two conditions |
| `VMASK_OR`  | `md, m1, m2` | `md[l] = m1[l] \| m2[l]` | `m[md]` | OR of two conditions |
| `VMASK_NOT` | `md, m1` | `md[l] = !m1[l]` | `m[md]` | Complement |
| `VCMP` | `md, vs1, vs2, cmp` | Per-lane compare → mask | `m[md]` | Primary mask builder |
| `VSEL` | `vd, vs1, vs2, mask` | `vd[l] = m[mask][l] ? vs1[l] : vs2[l]` | `v[vd]` | Branchless ternary |

### Control Flow Instructions

| Instruction | Syntax | Semantics | Notes |
|-------------|--------|-----------|-------|
| `BR` | `imm` | `pc = imm` | Unconditional absolute jump |
| `BRZ` | `rs, imm` | `if s[rs].i == 0: pc = imm` | Branch if scalar zero |
| `BRNZ` | `rs, imm` | `if s[rs].i != 0: pc = imm` | Branch if scalar nonzero |
| `LOOP` | `rs, imm` | `s[rs]--; if s[rs] > 0: pc = imm` | Decrement-and-branch |
| `KEXIT` | — | Terminate this grid instance | Required at kernel end |

### Predicated Instruction Execution

| Field | Value | Meaning |
|-------|-------|---------|
| `use_mask = 0` | (default) | All lanes active regardless of mask registers |
| `use_mask = 1, mask = N` | active | Only lanes where `m[N][l] = 1` write results |

---

## Summary

This chapter covered the NPU's approach to data-dependent computation without SIMD
divergence. The key design decisions and their rationale:

**Why predication over branching?**
Because different lanes of the same vector register may want different things simultaneously.
A single scalar branch cannot capture per-lane divergence. Predication applies the decision
at the granularity of individual lanes, within a single instruction's execution.

**Why 8 mask registers (m0–m7)?**
Enough to hold several simultaneously active conditions without spilling to memory, while
keeping the register file area cost reasonable. In practice, most kernels need 2–4 masks
simultaneously (one for the positive condition, its complement, possibly a compound
condition, and sometimes a result-valid mask).

**Why separate VMASK_AND/OR/NOT from the arithmetic ISA?**
Mask operations are fundamentally different from arithmetic — they operate on 1-bit boolean
values, not floating-point data. Keeping them separate preserves the conceptual clarity of
the ISA and allows the hardware to implement mask operations with a simple bitwise unit
rather than routing through the vector ALU.

**Why VSEL in addition to predicated writes?**
`VSEL` is cleaner for compiler code generation because it has explicit data flow (two input
registers, one output register, no read-modify-write). Predicated writes introduce register
liveness complexity for a compiler because the destination register carries its previous
value for inactive lanes. Both mechanisms are useful, and their coexistence reflects the
dual needs of hand-written kernels (predicated writes) and compiler-generated code (VSEL).

Chapter 9 steps back from individual instruction semantics and explains the complete
execution model: how kernels are launched, what state each instance gets, how the
multi-core model works, and what the cycle model actually measures.
