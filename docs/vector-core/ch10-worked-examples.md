# Chapter 10: Worked Examples — Complete Kernels from Scratch

This chapter presents five complete, self-contained vector-core kernels. Each example walks
from the mathematical definition of the operation, through algorithm design, to a fully
annotated `vinst[]` array and a step-by-step register trace. Every field of every instruction
is explained. Read them in order: each example builds on conventions established in the
previous one.

---

## 10.0 Conventions Used in This Chapter

### Register naming

The vector core has three register files. We use their assembler names throughout:

```
v0..v31   Vector registers — hold vl floats each (up to VLEN_MAX=256).
           All lanes are active when use_mask=0 (the default).

s0..s31   Engine scalar registers — each holds a float (s[n].f) or
           an int64 (s[n].i) in the same 8-byte union. Kernel entry
           wires two of them:
             s0 = grid instance id  (0..grid_size-1)
             s1 = args base pointer (address in NPU memory, uint32)
           s2..s31 are available for kernel use.

m0..m7    Mask registers — one bit per lane, set by VCMP/VMASK.SET,
           consumed by @mN predication prefix on any instruction.
```

### The `use_base` field on loads and stores

When `use_base=1`:
```
  effective_address = s[rs] + imm
```
When `use_base=0`:
```
  effective_address = imm    (absolute)
```
Nearly every kernel uses `use_base=1` to build a SIMT address from the instance id.

### Grid and slice conventions

A grid of G instances partitions an array of G*vl floats. Instance id `i` owns floats at
byte offset `i * vl * sizeof(float)`. With vl=16, each instance handles 16 floats = 64
bytes. Instance id comes from `s0` at kernel entry; we copy it with `V_KID` so we can
modify `s0` freely.

### Notation for traces

```
  vl = 16                     current vector length
  s3 = 0x80000080             engine scalar s3 holds the integer 0x80000080
  v0 = [1.0, 2.0, ...]        vector register v0; only first few lanes shown
  m0 = [1, 1, 0, 1, ...]      mask register m0; bit per lane
```

---

## 10.1 Element-wise ReLU

### 10.1.1 What is ReLU?

ReLU (Rectified Linear Unit) is the activation function `f(x) = max(0, x)`. It sets every
negative value to zero and passes positive values unchanged. ReLU is the most widely used
activation function in deep learning — it is cheap to compute, avoids the vanishing-gradient
problem of sigmoid/tanh, and produces sparse activations that benefit downstream layers.

ReLU appears after every linear layer in most convolutional, transformer, and MLP
architectures.

### 10.1.2 Mathematical Formulation

```
  y[i] = ReLU(x[i]) = max(0, x[i])    for i = 0, 1, ..., N-1
```

This is an element-wise operation: each output depends only on the corresponding input
element. There are no cross-element dependencies, so the operation is trivially parallel
across both vector lanes and grid instances.

### 10.1.3 Algorithm Design

The kernel runs one instance per 16-element chunk. With a grid of `ceil(N/16)` instances,
the entire array is processed in parallel.

```
Step 1:  VSETVL  — configure the vector engine to process 16 lanes.
Step 2:  KID     — read my instance id from hardware into a scratch scalar.
Step 3:  SMULI   — multiply instance id by 64 (= 16 floats × 4 bytes) to get byte offset.
Step 4:  SADD    — add byte offset to args_base (s1) to get my slice address.
Step 5:  VLOAD   — load 16 floats from my slice.
Step 6:  SFI     — load the float constant 0.0 into an engine scalar.
Step 7:  VBCAST  — broadcast 0.0 into every lane of a work vector.
Step 8:  VMAX    — element-wise max of input and zero vector = ReLU.
Step 9:  VSTORE  — write 16 results back to the same slice.
Step 10: KEXIT   — terminate this instance.
```

### 10.1.4 Complete Kernel

```c
/*
 * relu_kernel — applies ReLU to a contiguous float array.
 *
 * Launch as:
 *   npu_run(n, kernel_id, /*grid=*/ N/16, /*args_addr=*/ base_addr);
 *
 * Memory layout at args_addr:
 *   [base_addr + i*64 .. base_addr + i*64 + 60]  <-- instance i's 16 floats (in-place)
 *
 * Register plan:
 *   s0  = instance id (set by hardware, read via KID)
 *   s1  = args base address (set by hardware)
 *   s2  = byte offset for this instance (= id * 64)
 *   s3  = effective address for load/store (= s1 + s2)
 *   s4  = 0.0f (scalar zero for broadcast)
 *   v0  = input values (16 floats)
 *   v1  = zero vector (0.0 in every lane)
 *   v2  = output (= max(v0, v1) = ReLU(v0))
 */
static const vinst relu_kernel[] = {

    /* ------------------------------------------------------------------ */
    /* Step 1: Set vector length to 16.                                    */
    /*                                                                     */
    /* V_VSETVL: s[rd].i = min(imm, VLEN_MAX); vl = s[rd].i              */
    /* Here we use the immediate form: vl = min(16, 256) = 16.            */
    /* s9 receives the OLD vector length (before this instruction),        */
    /* which we don't use but must name a destination register.           */
    /* ------------------------------------------------------------------ */
    { V_VSETVL, .rd=9, .imm=16 },

    /* ------------------------------------------------------------------ */
    /* Step 2: Copy instance id into a working scalar.                     */
    /*                                                                     */
    /* V_KID: s[rd].i = current grid instance id (same value as s0).      */
    /* We use s2 as our scratch copy so we can compute with it freely.    */
    /* ------------------------------------------------------------------ */
    { V_KID, .rd=2 },

    /* ------------------------------------------------------------------ */
    /* Step 3: Compute byte offset = instance_id * 64.                    */
    /*                                                                     */
    /* V_SMULI: s[rd].i = s[rs1].i * imm                                  */
    /* imm = 64 because each instance owns 16 floats × 4 bytes = 64 bytes.*/
    /* ------------------------------------------------------------------ */
    { V_SMULI, .rd=2, .rs1=2, .imm=64 },

    /* ------------------------------------------------------------------ */
    /* Step 4: Compute effective address = args_base + byte_offset.        */
    /*                                                                     */
    /* V_SADD: s[rd].i = s[rs1].i + s[rs2].i                             */
    /* s1 = args_base (hardware-set), s2 = byte_offset (just computed).   */
    /* Result in s3 = address of this instance's 16-float slice.          */
    /* ------------------------------------------------------------------ */
    { V_SADD, .rd=3, .rs1=1, .rs2=2 },

    /* ------------------------------------------------------------------ */
    /* Step 5: Load 16 floats from this instance's slice.                  */
    /*                                                                     */
    /* V_VLOAD: v[vd][l] = float_at(s[rs] + imm + 4*l)  for l in 0..vl-1*/
    /* use_base=1 means effective_addr = s[rs] + imm = s3 + 0 = s3.      */
    /* Each lane l loads float at s3 + 4*l.                               */
    /* ------------------------------------------------------------------ */
    { V_VLOAD, .vd=0, .rs=3, .imm=0, .use_base=1 },

    /* ------------------------------------------------------------------ */
    /* Step 6: Load the float constant 0.0 into engine scalar s4.         */
    /*                                                                     */
    /* V_SFI: s[rd].f = (float)fimm                                       */
    /* fimm is encoded as an IEEE 754 double in the instruction word;     */
    /* the simulator reads it back as float. 0.0 is exactly representable.*/
    /* ------------------------------------------------------------------ */
    { V_SFI, .rd=4, .fimm=0.0 },

    /* ------------------------------------------------------------------ */
    /* Step 7: Broadcast 0.0 to all lanes of v1.                          */
    /*                                                                     */
    /* V_VBCAST: v[vd][l] = s[rs].f    for l in 0..vl-1                  */
    /* After this, v1 = [0.0, 0.0, 0.0, ..., 0.0] (16 lanes).            */
    /* ------------------------------------------------------------------ */
    { V_VBCAST, .vd=1, .rs=4 },

    /* ------------------------------------------------------------------ */
    /* Step 8: Apply ReLU = element-wise max of input and zero.            */
    /*                                                                     */
    /* V_VMAX: v[vd][l] = fmaxf(v[vs1][l], v[vs2][l])                   */
    /* For each lane: max(x[l], 0.0) = x[l] if x[l]>0, else 0.          */
    /* This is the exact IEEE 754 semantics of fmaxf (NaN propagation    */
    /* follows the hardware rule; in practice inputs are finite floats).  */
    /* ------------------------------------------------------------------ */
    { V_VMAX, .vd=2, .vs1=0, .vs2=1 },

    /* ------------------------------------------------------------------ */
    /* Step 9: Store 16 results back to this instance's slice.             */
    /*                                                                     */
    /* V_VSTORE: float_at(s[rs] + imm + 4*l) = v[vs][l]  for l in 0..vl-1*/
    /* Same address as the load: s3 + 0 = s3. In-place operation.        */
    /* ------------------------------------------------------------------ */
    { V_VSTORE, .vs=2, .rs=3, .imm=0, .use_base=1 },

    /* ------------------------------------------------------------------ */
    /* Step 10: Terminate this instance.                                   */
    /*                                                                     */
    /* V_KEXIT: causes run_vector_instance() to return. The grid          */
    /* scheduler moves to the next pending instance (or signals done).    */
    /* ------------------------------------------------------------------ */
    { V_KEXIT },
};
```

### 10.1.5 Register Trace (Instance id = 2, args_base = 0x80000080)

We trace through instance id=2 processing floats at byte offset 128 = 0x80 from args_base.
Because `args_base = 0x80000080`, the high bit is set, so this is SRAM (scratchpad) — load
latency will be 1 cycle, not 100. Input values are assumed to be `[-3.0, 1.5, -0.5, 2.0,
-1.0, 0.5, 3.0, -2.0, 0.0, 4.0, -0.1, 1.0, -4.0, 0.25, -0.75, 2.5]`.

```
INITIAL STATE:
  s0 = 2              (instance id, set by hardware)
  s1 = 0x80000080     (args base pointer, set by hardware)
  s2..s31 = 0
  v0..v31 = [0.0, 0.0, ..., 0.0]   (all lanes zeroed)
  m0..m7  = [0, 0, ..., 0]
  vl = 0

─────────────────────────────────────────────────────
Instr 0: V_VSETVL rd=9, imm=16
  vl = min(16, 256) = 16
  s9.i = 0   (old vl, which was 0)
─────────────────────────────────────────────────────
After:  vl=16,  s9=0

─────────────────────────────────────────────────────
Instr 1: V_KID rd=2
  s2.i = inst_id = 2   (hardware-supplied)
─────────────────────────────────────────────────────
After:  s2=2

─────────────────────────────────────────────────────
Instr 2: V_SMULI rd=2, rs1=2, imm=64
  s2.i = s2.i * 64 = 2 * 64 = 128 = 0x80
─────────────────────────────────────────────────────
After:  s2=128  (0x80)

─────────────────────────────────────────────────────
Instr 3: V_SADD rd=3, rs1=1, rs2=2
  s3.i = s1.i + s2.i = 0x80000080 + 0x80 = 0x80000100
─────────────────────────────────────────────────────
After:  s3=0x80000100

─────────────────────────────────────────────────────
Instr 4: V_VLOAD vd=0, rs=3, imm=0, use_base=1
  addr = s3 + 0 = 0x80000100  (high bit set => SRAM, 1-cycle latency)
  For lane l=0: v0[0] = float_at(0x80000100 + 4*0) = -3.0
  For lane l=1: v0[1] = float_at(0x80000104)        =  1.5
  For lane l=2: v0[2] = float_at(0x80000108)        = -0.5
  ...
  For lane l=15: v0[15] = float_at(0x8000013C)       =  2.5
─────────────────────────────────────────────────────
After:  v0 = [-3.0, 1.5, -0.5, 2.0, -1.0, 0.5, 3.0, -2.0,
               0.0, 4.0, -0.1, 1.0, -4.0, 0.25, -0.75, 2.5]

─────────────────────────────────────────────────────
Instr 5: V_SFI rd=4, fimm=0.0
  s4.f = 0.0
─────────────────────────────────────────────────────
After:  s4.f=0.0

─────────────────────────────────────────────────────
Instr 6: V_VBCAST vd=1, rs=4
  For all l in 0..15: v1[l] = s4.f = 0.0
─────────────────────────────────────────────────────
After:  v1 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
               0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

─────────────────────────────────────────────────────
Instr 7: V_VMAX vd=2, vs1=0, vs2=1
  For each lane l: v2[l] = fmaxf(v0[l], v1[l]) = fmaxf(input, 0.0)
  Lane  0: fmaxf(-3.0,  0.0) =  0.0
  Lane  1: fmaxf( 1.5,  0.0) =  1.5
  Lane  2: fmaxf(-0.5,  0.0) =  0.0
  Lane  3: fmaxf( 2.0,  0.0) =  2.0
  Lane  4: fmaxf(-1.0,  0.0) =  0.0
  Lane  5: fmaxf( 0.5,  0.0) =  0.5
  Lane  6: fmaxf( 3.0,  0.0) =  3.0
  Lane  7: fmaxf(-2.0,  0.0) =  0.0
  Lane  8: fmaxf( 0.0,  0.0) =  0.0   ← ties go to either (0.0 = 0.0)
  Lane  9: fmaxf( 4.0,  0.0) =  4.0
  Lane 10: fmaxf(-0.1,  0.0) =  0.0
  Lane 11: fmaxf( 1.0,  0.0) =  1.0
  Lane 12: fmaxf(-4.0,  0.0) =  0.0
  Lane 13: fmaxf( 0.25, 0.0) =  0.25
  Lane 14: fmaxf(-0.75, 0.0) =  0.0
  Lane 15: fmaxf( 2.5,  0.0) =  2.5
─────────────────────────────────────────────────────
After:  v2 = [0.0, 1.5, 0.0, 2.0, 0.0, 0.5, 3.0, 0.0,
               0.0, 4.0, 0.0, 1.0, 0.0, 0.25, 0.0, 2.5]

─────────────────────────────────────────────────────
Instr 8: V_VSTORE vs=2, rs=3, imm=0, use_base=1
  addr = s3 + 0 = 0x80000100
  For lane l in 0..15: float_at(0x80000100 + 4*l) = v2[l]
  Memory at 0x80000100..0x8000013F updated with ReLU results.
─────────────────────────────────────────────────────

─────────────────────────────────────────────────────
Instr 9: V_KEXIT
  run_vector_instance() returns.
  Grid scheduler checks for remaining instances.
─────────────────────────────────────────────────────

FINAL STATE (this instance):
  Memory[0x80000100..0x8000013F] = [0.0, 1.5, 0.0, 2.0, 0.0, 0.5, 3.0, 0.0,
                                     0.0, 4.0, 0.0, 1.0, 0.0, 0.25, 0.0, 2.5]
```

**Cycle cost** (SRAM path, vl=16, passes = ceil(16/16) = 1):
- `VSETVL, KID, SMULI, SADD, SFI`: 1 cycle each = 5
- `VLOAD`: passes + SRAM_LATENCY = 1 + 1 = 2
- `VBCAST, VMAX`: 1 each = 2
- `VSTORE`: passes + SRAM_LATENCY = 1 + 1 = 2
- `KEXIT`: 1
- **Total: ~13 cycles per instance**

---

## 10.2 Numerically Stable Softmax

### 10.2.1 What is Softmax?

Softmax converts an arbitrary vector of real numbers (called logits) into a probability
distribution — a vector of non-negative values that sum to exactly 1.0. It appears:
- At the end of classifier networks (output layer).
- Inside every attention head in transformers (the `softmax(QK^T/sqrt(d_k))` step).
- In sampling from language models (temperature scaling is applied before softmax).

### 10.2.2 Mathematical Formulation

**Naive form** (numerically unsafe):
```
  softmax(x)[i] = exp(x[i]) / sum_j(exp(x[j]))
```

**Problem**: if any `x[i]` is large (e.g., 100.0), `exp(100.0) = 2.69e43`, causing IEEE 754
float overflow to +Inf.

**Numerically stable form** (subtract max first):
```
  m    = max_j(x[j])
  y[i] = exp(x[i] - m) / sum_j(exp(x[j] - m))
```

Since `x[i] - m <= 0` for all i, every exponent argument is non-positive, so `exp` never
overflows. The subtraction does not change the result because:
```
  exp(x[i] - m) / sum_j(exp(x[j] - m))
= exp(x[i]) * exp(-m) / (sum_j(exp(x[j])) * exp(-m))
= exp(x[i]) / sum_j(exp(x[j]))
```
The `exp(-m)` cancels.

### 10.2.3 Algorithm Design

```
Step 1:  VSETVL + KID + SMULI + SADD  — as in ReLU, compute slice address.
Step 2:  VLOAD                          — load 16 input logits.
Step 3:  VREDMAX                        — find the maximum logit (scalar reduction).
Step 4:  VBCAST                         — broadcast max to all lanes.
Step 5:  VSUB                           — subtract max from every lane.
Step 6:  VEXP                           — compute exp(x[l] - max) for every lane.
Step 7:  VREDSUM                        — sum all exp values (scalar reduction).
Step 8:  SRECIP                         — compute 1.0 / sum (reciprocal).
Step 9:  VBCAST                         — broadcast 1/sum to all lanes.
Step 10: VMUL                           — multiply each exp by 1/sum.
Step 11: VSTORE                         — write result.
Step 12: KEXIT
```

### 10.2.4 Complete Kernel

```c
/*
 * softmax_kernel — applies numerically stable softmax in-place.
 *
 * Launch as:
 *   npu_run(n, kernel_id, /*grid=*/ N/16, /*args_addr=*/ base_addr);
 *
 * Each instance processes a contiguous 16-float vector independently.
 * (For cross-instance softmax, all instances would need to reduce together;
 * that requires a two-pass approach using shared SRAM — see Chapter 12.)
 *
 * Register plan:
 *   s0  = instance id  (hardware)
 *   s1  = args base    (hardware)
 *   s2  = byte offset  (= id * 64)
 *   s3  = slice addr   (= s1 + s2)
 *   s4  = max logit    (scalar, from VREDMAX)
 *   s5  = sum of exps  (scalar, from VREDSUM)
 *   s6  = 1/sum        (scalar, from SRECIP)
 *   v0  = input logits, then shifted logits (x - max)
 *   v1  = broadcast max, then exp values, then probabilities
 *   v2  = broadcast 1/sum (denominator vector)
 */
static const vinst softmax_kernel[] = {

    /* --- Setup: vector length and slice address --- */

    /* Set vl=16; s9 receives old vl (discarded). */
    { V_VSETVL, .rd=9, .imm=16 },

    /* s2 = instance id */
    { V_KID, .rd=2 },

    /* s2 = id * 64  (byte offset into contiguous array) */
    { V_SMULI, .rd=2, .rs1=2, .imm=64 },

    /* s3 = args_base + byte_offset */
    { V_SADD, .rd=3, .rs1=1, .rs2=2 },

    /* --- Load input --- */

    /* v0[l] = float_at(s3 + 4*l)  for l in 0..15 */
    { V_VLOAD, .vd=0, .rs=3, .imm=0, .use_base=1 },

    /* --- Numerically stable max subtraction --- */

    /*
     * VREDMAX: s[rd].f = max(v[vs][0], v[vs][1], ..., v[vs][vl-1])
     * This is a tree reduction over vl=16 lanes; costs passes+4 cycles.
     * s4 now holds the maximum logit in this 16-element vector.
     */
    { V_VREDMAX, .rd=4, .vs=0 },

    /*
     * Broadcast max to all lanes so we can subtract element-wise.
     * v1[l] = s4.f    for all l.
     */
    { V_VBCAST, .vd=1, .rs=4 },

    /*
     * Subtract max from every lane.
     * v0[l] = v0[l] - v1[l] = x[l] - max(x)
     * All results are <= 0, so exp will not overflow.
     * The original v0 is overwritten — we no longer need the raw logits.
     */
    { V_VSUB, .vd=0, .vs1=0, .vs2=1 },

    /* --- Exponentiation --- */

    /*
     * VEXP: v[vd][l] = expf(v[vs][l])
     * Since every input is <= 0, every output is in (0, 1].
     * Cost: passes * 4 = 4 cycles (exp is the most expensive primitive).
     */
    { V_VEXP, .vd=1, .vs=0 },

    /* --- Normalization --- */

    /*
     * VREDSUM: s[rd].f = sum(v[vs][0..vl-1])
     * s5 = sum of all exp values = denominator of softmax.
     * Cost: passes + 4 cycles (tree reduction).
     */
    { V_VREDSUM, .rd=5, .vs=1 },

    /*
     * SRECIP: s[rd].f = 1.0f / s[rs1].f
     * s6 = 1 / sum(exp values)
     * Scalar operation, 1 cycle.
     */
    { V_SRECIP, .rd=6, .rs1=5 },

    /*
     * Broadcast 1/sum to all lanes.
     * v2[l] = s6.f    for all l.
     */
    { V_VBCAST, .vd=2, .rs=6 },

    /*
     * Multiply: v1[l] = exp(x[l]-max) * (1/sum) = softmax(x)[l]
     * Result is in v1; each lane holds a valid probability.
     * All lanes sum to 1.0 (up to floating-point rounding).
     */
    { V_VMUL, .vd=1, .vs1=1, .vs2=2 },

    /* --- Store result --- */

    /* float_at(s3 + 4*l) = v1[l]   for l in 0..15 */
    { V_VSTORE, .vs=1, .rs=3, .imm=0, .use_base=1 },

    { V_KEXIT },
};
```

### 10.2.5 Numerical Example and Register Trace

**Input** (16 logits for instance id=0, args_base=0x1000):

```
  x = [1.0, 2.0, 3.0, 4.0, 0.5, 1.5, 2.5, 3.5,
        0.0, 1.0, 2.0, 3.0, 0.5, 1.0, 1.5, 2.0]
```

```
INITIAL STATE:
  s0=0, s1=0x1000, vl=0, all others zeroed

─────────────────────────────────────────────────────
Instr 0: V_VSETVL rd=9, imm=16
─────────────────────────────────────────────────────
  vl=16, s9=0

─────────────────────────────────────────────────────
Instr 1: V_KID rd=2
─────────────────────────────────────────────────────
  s2=0

─────────────────────────────────────────────────────
Instr 2: V_SMULI rd=2, rs1=2, imm=64
─────────────────────────────────────────────────────
  s2 = 0 * 64 = 0

─────────────────────────────────────────────────────
Instr 3: V_SADD rd=3, rs1=1, rs2=2
─────────────────────────────────────────────────────
  s3 = 0x1000 + 0 = 0x1000

─────────────────────────────────────────────────────
Instr 4: V_VLOAD vd=0, rs=3, imm=0, use_base=1
─────────────────────────────────────────────────────
  addr = 0x1000  (DRAM, latency=100 cycles)
  v0 = [1.0, 2.0, 3.0, 4.0, 0.5, 1.5, 2.5, 3.5,
          0.0, 1.0, 2.0, 3.0, 0.5, 1.0, 1.5, 2.0]

─────────────────────────────────────────────────────
Instr 5: V_VREDMAX rd=4, vs=0
─────────────────────────────────────────────────────
  Reduction tree over 16 values:
    Round 1 (8 compares): max pairs → [2.0, 4.0, 1.5, 3.5, 1.0, 3.0, 1.0, 2.0]
    Round 2 (4 compares): → [4.0, 3.5, 3.0, 2.0]
    Round 3 (2 compares): → [4.0, 3.0]
    Round 4 (1 compare):  → 4.0
  s4.f = 4.0   ← maximum logit

─────────────────────────────────────────────────────
Instr 6: V_VBCAST vd=1, rs=4
─────────────────────────────────────────────────────
  v1 = [4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0,
          4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0]

─────────────────────────────────────────────────────
Instr 7: V_VSUB vd=0, vs1=0, vs2=1
─────────────────────────────────────────────────────
  v0[l] = v0[l] - 4.0
  v0 = [-3.0, -2.0, -1.0,  0.0, -3.5, -2.5, -1.5, -0.5,
          -4.0, -3.0, -2.0, -1.0, -3.5, -3.0, -2.5, -2.0]
  All values <= 0. Max is 0.0 (lane 3). ✓

─────────────────────────────────────────────────────
Instr 8: V_VEXP vd=1, vs=0
─────────────────────────────────────────────────────
  v1[l] = expf(v0[l])
  Lane  0: exp(-3.0) = 0.04979
  Lane  1: exp(-2.0) = 0.13534
  Lane  2: exp(-1.0) = 0.36788
  Lane  3: exp( 0.0) = 1.00000   ← the max element becomes 1.0
  Lane  4: exp(-3.5) = 0.03020
  Lane  5: exp(-2.5) = 0.08208
  Lane  6: exp(-1.5) = 0.22313
  Lane  7: exp(-0.5) = 0.60653
  Lane  8: exp(-4.0) = 0.01832
  Lane  9: exp(-3.0) = 0.04979
  Lane 10: exp(-2.0) = 0.13534
  Lane 11: exp(-1.0) = 0.36788
  Lane 12: exp(-3.5) = 0.03020
  Lane 13: exp(-3.0) = 0.04979
  Lane 14: exp(-2.5) = 0.08208
  Lane 15: exp(-2.0) = 0.13534
  v1 = [0.04979, 0.13534, 0.36788, 1.00000, 0.03020, 0.08208,
          0.22313, 0.60653, 0.01832, 0.04979, 0.13534, 0.36788,
          0.03020, 0.04979, 0.08208, 0.13534]

─────────────────────────────────────────────────────
Instr 9: V_VREDSUM rd=5, vs=1
─────────────────────────────────────────────────────
  s5.f = sum(v1) = 0.04979 + 0.13534 + 0.36788 + 1.00000
                  + 0.03020 + 0.08208 + 0.22313 + 0.60653
                  + 0.01832 + 0.04979 + 0.13534 + 0.36788
                  + 0.03020 + 0.04979 + 0.08208 + 0.13534
       = 3.36369   (sum of all exp values)

─────────────────────────────────────────────────────
Instr 10: V_SRECIP rd=6, rs1=5
─────────────────────────────────────────────────────
  s6.f = 1.0f / 3.36369 = 0.29728

─────────────────────────────────────────────────────
Instr 11: V_VBCAST vd=2, rs=6
─────────────────────────────────────────────────────
  v2 = [0.29728, 0.29728, ..., 0.29728]   (16 lanes, all equal)

─────────────────────────────────────────────────────
Instr 12: V_VMUL vd=1, vs1=1, vs2=2
─────────────────────────────────────────────────────
  v1[l] = exp_l * (1/sum) = probability_l
  Lane  0:  0.04979 * 0.29728 = 0.01480
  Lane  1:  0.13534 * 0.29728 = 0.04023
  Lane  2:  0.36788 * 0.29728 = 0.10934
  Lane  3:  1.00000 * 0.29728 = 0.29728   ← highest prob (was max logit)
  Lane  4:  0.03020 * 0.29728 = 0.00898
  Lane  5:  0.08208 * 0.29728 = 0.02440
  Lane  6:  0.22313 * 0.29728 = 0.06632
  Lane  7:  0.60653 * 0.29728 = 0.18030
  ...
  Verification: sum of all 16 probabilities ≈ 1.00000 ✓

─────────────────────────────────────────────────────
Instr 13: V_VSTORE vs=1, rs=3, imm=0, use_base=1
─────────────────────────────────────────────────────
  Memory at 0x1000 updated with probability vector.

─────────────────────────────────────────────────────
Instr 14: V_KEXIT
─────────────────────────────────────────────────────
```

**Why use SRECIP instead of VDIV?** The denominator is the same scalar for all 16 lanes.
`SRECIP` computes `1/sum` once (1 cycle) and `VBCAST+VMUL` applies it (2 cycles). The
alternative — `VBCAST+VDIV` — costs `VBCAST` (1) + `VDIV` (passes*4 = 4) = 5 cycles. Using
the reciprocal saves 2 cycles and is the conventional approach when the divisor is uniform.

---

## 10.3 Layer Normalization

### 10.3.1 What is LayerNorm?

Layer Normalization (Ba et al., 2016) normalizes the activations across the feature dimension
of a single sample, then applies a learned affine transform (scale `gamma`, shift `beta`).
It stabilizes training in transformers and RNNs.

### 10.3.2 Mathematical Formulation

```
  mu    = (1/N) * sum_i(x[i])            // mean
  var   = (1/N) * sum_i((x[i] - mu)^2)   // variance
  x_hat = (x[i] - mu) / sqrt(var + eps)  // normalize
  y[i]  = gamma * x_hat + beta            // affine transform
```

`eps` is a small constant (e.g., 1e-5) added to the variance before the square root to
prevent division by zero when inputs are constant.

### 10.3.3 Algorithm Design

```
Pass 1 — Compute mean:
  Step 1: VLOAD              — load x.
  Step 2: VREDSUM            — sum all elements (s_sum).
  Step 3: scalar divide      — SFI (1/N), SMUL → s_mean = s_sum * (1/N).
                               (No scalar float division intrinsic; use SFI + SRECIP trick
                                or pre-compute 1/N at launch time.)

Pass 2 — Subtract mean and compute variance:
  Step 4: VBCAST s_mean      — broadcast mean to v1.
  Step 5: VSUB               — v0 = x - mean.
  Step 6: VMUL v0, v0        — square the deviations.
  Step 7: VREDSUM            — sum squared deviations.
  Step 8: scalar * 1/N       — variance.
  Step 9: scalar + eps       — var + eps.

Pass 3 — Normalize and scale:
  Step 10: VRSQRT            — 1/sqrt(var+eps) in a single instruction.
  Step 11: VBCAST            — broadcast to all lanes.
  Step 12: VMUL              — x_hat = (x - mean) * inv_std.
  Step 13: VBCAST gamma      — load and broadcast gamma.
  Step 14: VMUL              — x_hat * gamma.
  Step 15: VBCAST beta       — load and broadcast beta.
  Step 16: VADD              — + beta = y.
  Step 17: VSTORE.
```

### 10.3.4 Complete Kernel

```c
/*
 * layernorm_kernel — single-instance Layer Normalization over vl=16 floats.
 *
 * Args layout at s1:
 *   Offset  0: float x[16]      (input, overwritten with output)
 *   Offset 64: float gamma       (single scalar: per-channel scale)
 *   Offset 68: float beta        (single scalar: per-channel shift)
 *   Offset 72: float eps         (single scalar: variance floor, e.g. 1e-5)
 *   Offset 76: float inv_N       (= 1.0/16 = 0.0625, pre-computed by launcher)
 *
 * Register plan:
 *   s1  = args base     (hardware)
 *   s2  = sum/mean/scratch
 *   s3  = inv_N scalar (0.0625)
 *   s4  = mean
 *   s5  = variance
 *   s6  = eps scalar
 *   s7  = gamma scalar
 *   s8  = beta scalar
 *   s9  = 1/sqrt(var+eps)
 *   v0  = input x, then (x - mean), then normalized x_hat, then output y
 *   v1  = broadcast mean, then squared deviations, then reused
 *   v2  = broadcast of various scalars
 */
static const vinst layernorm_kernel[] = {

    /* ---- Setup ---- */

    /* vl = 16 */
    { V_VSETVL, .rd=10, .imm=16 },

    /* Load x[0..15] into v0 */
    { V_VLOAD, .vd=0, .rs=1, .imm=0, .use_base=1 },

    /* Load pre-computed inv_N = 1/16 = 0.0625 from args+76 */
    { V_SLF, .rd=3, .rs=1, .imm=76 },

    /* Load eps from args+72 */
    { V_SLF, .rd=6, .rs=1, .imm=72 },

    /* Load gamma from args+64 */
    { V_SLF, .rd=7, .rs=1, .imm=64 },

    /* Load beta from args+68 */
    { V_SLF, .rd=8, .rs=1, .imm=68 },

    /* ---- Pass 1: Compute mean ---- */

    /*
     * Sum all 16 elements. s2.f = x[0] + x[1] + ... + x[15].
     * Cost: passes + 4 = 5 cycles.
     */
    { V_VREDSUM, .rd=2, .vs=0 },

    /*
     * Broadcast inv_N into v2, then VMUL to divide sum by N.
     * Alternatively, we could multiply the scalar s2 by s3 directly —
     * but there is no scalar SMULF instruction. We use SRECIP on N
     * (which we know = 16) and multiply. Here we pre-loaded inv_N,
     * so we need to multiply: s4 = s2 * s3.
     *
     * There is no scalar FP multiply in the ISA. We broadcast s2 (sum)
     * into a vector, multiply by broadcast(inv_N), and reduce back.
     * This is a common pattern when computing mean.
     */
    { V_VBCAST, .vd=1, .rs=2 },       /* v1 = [sum, sum, ..., sum]       */
    { V_VBCAST, .vd=2, .rs=3 },       /* v2 = [inv_N, ..., inv_N]        */
    { V_VMUL,   .vd=1, .vs1=1, .vs2=2 }, /* v1[l] = sum * inv_N = mean   */

    /*
     * VREDMAX gives us the max, not a single lane. Since all lanes of v1
     * are identical (= mean), we use VREDMAX (or VREDSUM — same result).
     * This extracts the mean into a scalar register.
     */
    { V_VREDMAX, .rd=4, .vs=1 },      /* s4.f = mean                     */

    /* ---- Pass 2: Subtract mean, square, compute variance ---- */

    /*
     * Broadcast mean to all lanes of v2 for subtraction.
     */
    { V_VBCAST, .vd=2, .rs=4 },       /* v2 = [mean, mean, ..., mean]    */

    /*
     * v0 = x - mean  (centered inputs, zero-mean)
     * Overwrite v0 in place; we no longer need raw x.
     */
    { V_VSUB, .vd=0, .vs1=0, .vs2=2 },

    /*
     * Square each centered value: v1[l] = (x[l] - mean)^2
     * V_VMUL with vs1=vs2=v0 gives element-wise square.
     */
    { V_VMUL, .vd=1, .vs1=0, .vs2=0 },

    /*
     * Sum squared deviations. s5.f = sum((x[l]-mean)^2).
     */
    { V_VREDSUM, .rd=5, .vs=1 },

    /*
     * Compute variance = sum_of_sq_devs * inv_N.
     * Again, scalar FP multiply via vector round-trip:
     * broadcast s5, multiply by inv_N vector, reduce.
     */
    { V_VBCAST, .vd=1, .rs=5 },
    { V_VMUL,   .vd=1, .vs1=1, .vs2=2 }, /* v2 still holds inv_N broadcast */
    { V_VREDMAX, .rd=5, .vs=1 },          /* s5.f = variance               */

    /* ---- Pass 3: Add eps, compute rsqrt, normalize ---- */

    /*
     * Add eps to variance. Same scalar round-trip.
     * v1 = [var+eps, var+eps, ..., var+eps]
     */
    { V_VBCAST, .vd=1, .rs=5 },       /* v1 = [var, ...]                 */
    { V_VBCAST, .vd=2, .rs=6 },       /* v2 = [eps, ...]                 */
    { V_VADD,   .vd=1, .vs1=1, .vs2=2 }, /* v1 = [var+eps, ...]          */

    /*
     * VRSQRT: v[vd][l] = 1.0f / sqrtf(v[vs][l])
     * All lanes of v1 hold (var+eps), so all lanes of v2 = inv_std.
     * Cost: passes * 4 = 4 cycles.
     */
    { V_VRSQRT, .vd=2, .vs=1 },       /* v2 = [1/sqrt(var+eps), ...]     */

    /*
     * Normalize: x_hat = (x - mean) * inv_std
     * v0 = centered inputs, v2 = inv_std broadcast.
     */
    { V_VMUL, .vd=0, .vs1=0, .vs2=2 }, /* v0 = x_hat                    */

    /* ---- Affine transform: y = gamma * x_hat + beta ---- */

    /* Broadcast gamma into v2. */
    { V_VBCAST, .vd=2, .rs=7 },

    /* v0 = gamma * x_hat */
    { V_VMUL, .vd=0, .vs1=0, .vs2=2 },

    /* Broadcast beta into v2. */
    { V_VBCAST, .vd=2, .rs=8 },

    /* v0 = gamma * x_hat + beta = y */
    { V_VADD, .vd=0, .vs1=0, .vs2=2 },

    /* ---- Store ---- */
    { V_VSTORE, .vs=0, .rs=1, .imm=0, .use_base=1 },

    { V_KEXIT },
};
```

### 10.3.5 Key Observations

**Why no scalar FP multiply?** The engine scalar file (`s0..s31`) uses a union `{float f;
int64_t i;}`. The arithmetic instructions `V_SMUL`/`V_SMULI` operate on the `.i` field
(integer multiply). There is no `SMULF` (scalar float multiply). Float operations go through
vectors. This means computing a scalar float product requires: broadcast-to-vector → vmul →
reduce-back.

**Alternative: pre-compute inv_N on the host.** The example above loads `inv_N = 1/16` from
the args block. This is the recommended pattern: compute constants (like `1/N`, `1/sqrt(2*pi)`)
on the control core or host before kernel launch, and pass them as float args.

**Using VRSQRT vs VSQRT + VRECIP:** A single `VRSQRT` is one instruction (4 cycles). The
equivalent `VSQRT` (4 cycles) + `VRECIP` (4 cycles) costs 8 cycles. Always prefer `VRSQRT`
when you need `1/sqrt(x)`.

---

## 10.4 Vector Dot Product

### 10.4.1 What is a Dot Product?

The dot product (inner product) of two vectors `a` and `b` of length N is:
```
  result = sum_i(a[i] * b[i])    for i = 0..N-1
```

It is the fundamental primitive of matrix-vector multiplication, fully-connected layers,
and attention score computation.

### 10.4.2 Algorithm Design

The NPU vector core has `V_VFMA` (fused multiply-accumulate): `vd[l] += vs1[l] * vs2[l]`.
This is the ideal instruction for dot products. For a 16-element dot product (fits in one
vector): load `a` into `v0`, load `b` into `v1`, zero an accumulator `v2`, apply `VFMA` to
accumulate `a[l]*b[l]` into `v2[l]`, then `VREDSUM` to collapse lanes into a single scalar.

### 10.4.3 Complete Kernel

```c
/*
 * dot_product_kernel — computes dot product of two 16-element vectors.
 *
 * Args layout at s1:
 *   Offset  0: float a[16]       (first vector)
 *   Offset 64: float b[16]       (second vector)
 *   Offset 128: float result      (output: single float written by grid[0])
 *
 * Grid size: 1 (single instance; for larger vectors, see §10.5).
 *
 * Register plan:
 *   v0 = vector a
 *   v1 = vector b
 *   v2 = accumulator (a[l]*b[l] per lane, then reduced to scalar)
 *   s2 = 0.0f (for zeroing accumulator)
 *   s3 = address of result slot (= s1 + 128)
 *   s4 = dot product result scalar
 */
static const vinst dot_product_kernel[] = {

    /* Set vl=16. */
    { V_VSETVL, .rd=9, .imm=16 },

    /* Load a[0..15] from args+0. */
    { V_VLOAD, .vd=0, .rs=1, .imm=0, .use_base=1 },

    /* Load b[0..15] from args+64. */
    { V_VLOAD, .vd=1, .rs=1, .imm=64, .use_base=1 },

    /*
     * Zero the accumulator v2.
     * SFI: s2.f = 0.0
     * VBCAST: v2[l] = 0.0  for all l
     */
    { V_SFI,    .rd=2, .fimm=0.0 },
    { V_VBCAST, .vd=2, .rs=2 },

    /*
     * VFMA: v[vd][l] = v[vd][l] + v[vs1][l] * v[vs2][l]
     *
     * v2[l] = 0.0 + a[l] * b[l] = a[l] * b[l]
     *
     * This is a fused multiply-add: the multiplication and addition are
     * performed with a single rounding step, giving better accuracy than
     * separate VMUL + VADD.
     *
     * If the vectors were longer (say 256 elements), you would loop here,
     * loading successive chunks and FMAing into the same v2 accumulator.
     * The single-pass version here handles exactly 16 elements.
     */
    { V_VFMA, .vd=2, .vs1=0, .vs2=1 },

    /*
     * VREDSUM: collapse v2 across all lanes.
     * s4.f = v2[0] + v2[1] + ... + v2[15]
     *       = a[0]*b[0] + a[1]*b[1] + ... + a[15]*b[15]
     *       = dot(a, b)
     */
    { V_VREDSUM, .rd=4, .vs=2 },

    /*
     * Compute address of result slot: s1 + 128.
     * SADDI: s3.i = s1.i + 128
     */
    { V_SADDI, .rd=3, .rs1=1, .imm=128 },

    /*
     * Write the scalar result to memory.
     * We store via a 1-lane trick: set vl=1, broadcast s4, vstore.
     * Alternatively, use SFI + SLW; the sim does not have a scalar
     * float store instruction, so we repurpose vstore with vl=1.
     */
    { V_VSETVL,  .rd=9, .imm=1 },
    { V_VBCAST,  .vd=3, .rs=4 },
    { V_VSTORE,  .vs=3, .rs=3, .imm=0, .use_base=1 },

    { V_KEXIT },
};
```

### 10.4.4 Register Trace (a=[1,2,...,16], b=[1,1,...,1])

```
After VSETVL:  vl=16
After VLOAD v0:  v0 = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0,
                         9.0,10.0,11.0,12.0,13.0,14.0,15.0,16.0]
After VLOAD v1:  v1 = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                         1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
After SFI:       s2.f = 0.0
After VBCAST:    v2 = [0.0, 0.0, ..., 0.0]  (16 zeros)
After VFMA:      v2[l] = 0 + a[l]*b[l] = a[l]*1 = a[l]
  v2 = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0,
          9.0,10.0,11.0,12.0,13.0,14.0,15.0,16.0]
After VREDSUM:   s4.f = 1+2+3+4+5+6+7+8+9+10+11+12+13+14+15+16 = 136.0
After SADDI:     s3 = s1 + 128 = <result_address>
After VSETVL(1): vl=1
After VBCAST v3: v3[0] = 136.0
After VSTORE:    float_at(s3) = 136.0
KEXIT.
Expected: sum(1..16) = 16*17/2 = 136. ✓
```

**Performance note**: For very long dot products (N >> 16), keep vl=16 and loop. Each
`VFMA` costs 1 cycle (elementwise), so 256 elements = 16 iterations = 16 VFMA cycles plus
loop overhead.

---

## 10.5 Multi-Iteration Loop: Processing Arrays Larger than vl

### 10.5.1 The Problem

The vector register holds at most `VLEN_MAX=256` floats. Real workloads have tensors with
thousands or millions of elements. The loop construct (`V_LOOP`) solves this: each kernel
instance iterates over its portion of the array in chunks of `vl`.

### 10.5.2 The Loop Instruction

```
V_LOOP: rs, imm
  s[rs].i -= 1
  if s[rs].i > 0: pc = imm   (branch to instruction at index imm)
  else:           fall through to next instruction
```

This is a pre-decremented count: load the count *before* the loop body, and `LOOP` decrements
and branches at the *end* of the body. The body executes exactly `count` times (the count
must be >= 1 before the loop; a count of 0 means one iteration because LOOP decrements
first). To loop `count` times, initialize `s[rs] = count` before the loop.

### 10.5.3 Example: Elementwise exp() of 256 floats

Each kernel instance processes 256 floats (= 16 chunks of 16). With a grid of 1, all 256
floats are processed by instance 0.

```c
/*
 * exp256_kernel — applies expf to 256 contiguous floats.
 *
 * Args at s1: float x[256]  (in-place, 1024 bytes)
 * Grid: 1 instance
 *
 * Register plan:
 *   s2  = loop count (16 iterations)
 *   s3  = current byte offset (advances by 64 each iteration)
 *   s4  = current effective address (s1 + s3)
 */
static const vinst exp256_kernel[] = {
    /* 0 */ { V_VSETVL, .rd=9, .imm=16 },        /* vl = 16                     */

    /* 1 */ { V_SLI,    .rd=2, .imm=16 },         /* s2 = 16  (iteration count)  */

    /* 2 */ { V_SLI,    .rd=3, .imm=0  },         /* s3 = 0   (byte offset)      */

    /* LOOP BODY starts at instruction 3 */

    /* 3 */ { V_SADD,   .rd=4, .rs1=1, .rs2=3 }, /* s4 = s1 + s3 (cur address)  */

    /* 4 */ { V_VLOAD,  .vd=0, .rs=4, .imm=0,
               .use_base=1 },                     /* v0 = x[s3/4 .. s3/4+15]    */

    /* 5 */ { V_VEXP,   .vd=0, .vs=0 },           /* v0 = exp(v0) element-wise   */

    /* 6 */ { V_VSTORE, .vs=0, .rs=4, .imm=0,
               .use_base=1 },                     /* write back to same position */

    /* 7 */ { V_SADDI,  .rd=3, .rs1=3, .imm=64 },/* s3 += 64  (next chunk)      */

    /*
     * LOOP: s2 -= 1; if s2 > 0 goto instruction 3.
     * Iteration 1:  s2 was 16 → becomes 15, branch to 3.
     * Iteration 2:  s2 was 15 → becomes 14, branch to 3.
     * ...
     * Iteration 16: s2 was  1 → becomes  0, fall through to 9.
     */
    /* 8 */ { V_LOOP,   .rs=2, .imm=3 },

    /* 9 */ { V_KEXIT },
};
```

### 10.5.4 Instruction-by-Instruction Explanation

| # | Instruction | Purpose |
|---|-------------|---------|
| 0 | `VSETVL rd=9, imm=16` | Set vector length to 16. |
| 1 | `SLI rd=2, imm=16` | Load literal integer 16 into s2. This is the loop trip count. |
| 2 | `SLI rd=3, imm=0` | Initialize byte offset to 0. |
| 3 | `SADD rd=4, rs1=1, rs2=3` | Address = base + current offset. Recomputed each iteration. |
| 4 | `VLOAD vd=0, rs=4, imm=0` | Load chunk at current address. |
| 5 | `VEXP vd=0, vs=0` | Apply exp to all 16 lanes. |
| 6 | `VSTORE vs=0, rs=4, imm=0` | Write chunk back. |
| 7 | `SADDI rd=3, rs1=3, imm=64` | Advance offset by 16 floats × 4 bytes = 64. |
| 8 | `LOOP rs=2, imm=3` | Decrement s2; if > 0 jump to instr 3. |
| 9 | `KEXIT` | All chunks processed; terminate. |

### 10.5.5 Execution Trace Summary

```
Iteration 1 (i=0): s2=16→15, s3=0→64,   address=s1+0,   chunk  0
Iteration 2 (i=1): s2=15→14, s3=64→128,  address=s1+64,  chunk  1
Iteration 3 (i=2): s2=14→13, s3=128→192, address=s1+128, chunk  2
...
Iteration 16(i=15):s2= 1→ 0, s3=960→1024,address=s1+960, chunk 15
  → LOOP falls through
Instruction 9: KEXIT
```

### 10.5.6 Extending to Multiple Instances

For 4096 floats with grid=16:

```c
/* Each instance handles 256 floats. Adjust init to: */
/* Instance id = s0 = 0..15 */
/* Byte offset into array = id * 1024 (256 floats * 4 bytes) */

{ V_KID,    .rd=5              },
{ V_SMULI,  .rd=5, .rs1=5, .imm=1024 },
{ V_SADD,   .rd=6, .rs1=1, .rs2=5   },  /* s6 = args_base + id*1024 */
/* ... then proceed with the loop using s6 instead of s1 ... */
```

The loop body (instructions 3–8) stays identical; only the base address changes.

---

## 10.6 Predicated Computation: Masked ReLU with Threshold

### 10.6.1 Motivation

Sometimes not all elements of a vector should be updated. Predication (masking) lets you
apply an operation to only the lanes where a condition holds, while leaving other lanes
unchanged.

**Example use case**: Apply ReLU only to elements where `index % 2 == 0` (even elements).
Or apply a threshold: set to 1.0 only elements that exceed 0.5, leave others at 0.0.

### 10.6.2 Threshold Indicator Kernel

This kernel computes `y[i] = (x[i] > threshold) ? 1.0 : 0.0` — a step function.

```c
/*
 * threshold_kernel — binary threshold: y[i] = x[i] > thresh ? 1.0 : 0.0
 *
 * Args at s1:
 *   Offset  0: float x[16]       (input)
 *   Offset 64: float out[16]     (output, separate from input)
 *   Offset 128: float threshold   (single scalar threshold value)
 *
 * Register plan:
 *   v0 = input x
 *   v1 = threshold broadcast
 *   v2 = 0.0 vector
 *   v3 = 1.0 vector
 *   v4 = result y (starts as 0.0, lanes where x>thresh set to 1.0)
 *   m0 = mask: lane l is 1 if x[l] > threshold
 *   s2 = threshold scalar
 *   s3 = 0.0f
 *   s4 = 1.0f
 *   s5 = output address
 */
static const vinst threshold_kernel[] = {

    { V_VSETVL, .rd=9, .imm=16 },

    /* Load input */
    { V_VLOAD,  .vd=0, .rs=1, .imm=0, .use_base=1 },

    /* Load threshold scalar from args+128 */
    { V_SLF,    .rd=2, .rs=1, .imm=128 },

    /* Broadcast threshold to all lanes: v1 = [thresh, thresh, ...] */
    { V_VBCAST, .vd=1, .rs=2 },

    /*
     * VCMP.GT: m[md][l] = (v[vs1][l] > v[vs2][l])
     * Sets m0[l] = 1 if x[l] > threshold, else 0.
     * .cmp = CMP_GT = 4
     */
    { V_VCMP, .md=0, .vs1=0, .vs2=1, .cmp=4 },   /* CMP_GT = 4 */

    /* Build 0.0 and 1.0 vectors */
    { V_SFI,    .rd=3, .fimm=0.0 },
    { V_VBCAST, .vd=2, .rs=3 },      /* v2 = all zeros */

    { V_SFI,    .rd=4, .fimm=1.0 },
    { V_VBCAST, .vd=3, .rs=4 },      /* v3 = all ones  */

    /*
     * VSEL: v[vd][l] = mask[l] ? v[vs1][l] : v[vs2][l]
     * Where mask is m0.
     * v4[l] = m0[l] ? v3[l] : v2[l]
     *       = m0[l] ? 1.0   : 0.0
     *       = (x[l] > threshold) ? 1.0 : 0.0
     */
    { V_VSEL, .vd=4, .vs1=3, .vs2=2, .mask=0 },

    /* Output address = s1 + 64 */
    { V_SADDI,  .rd=5, .rs1=1, .imm=64 },

    { V_VSTORE, .vs=4, .rs=5, .imm=0, .use_base=1 },

    { V_KEXIT },
};
```

### 10.6.3 Using `use_mask` for Predicated Store

An alternative encoding predicate stores using `use_mask=1`:

```c
/*
 * @m0 VSTORE: only lanes where m0[l]=1 are written.
 * Lanes where m0[l]=0 leave memory unchanged.
 *
 * This is useful when the output array is pre-initialized and you
 * only want to overwrite certain positions.
 */
{ V_VSTORE, .vs=3, .rs=5, .imm=0, .use_base=1,
             .use_mask=1, .mask=0 },
```

With `use_mask=1`, only the lanes where `m0[l]=1` perform the store; all other lanes leave
their memory words untouched.

---

## 10.7 Summary: Pattern Catalog

The five kernels above cover the most common patterns in neural-network workloads. Here is
a quick reference:

```
Pattern                  Instructions Used           Example Kernel
─────────────────────────────────────────────────────────────────────
Element-wise unary       VLOAD, V_OP, VSTORE         ReLU (§10.1)
Element-wise with const  VBCAST, V_OP                ReLU (§10.1)
Reduction to scalar      VREDSUM/MAX/MIN              Softmax (§10.2)
Broadcast scalar result  VBCAST of reduction         Softmax (§10.2)
Scalar reciprocal        SRECIP                      Softmax (§10.2)
Float rsqrt              VRSQRT                      LayerNorm (§10.3)
Fused multiply-add       VFMA                        Dot product (§10.4)
Counted loop             SLI (count) + LOOP          exp256 (§10.5)
Predication              VCMP + VSEL / use_mask      Threshold (§10.6)
Conditional store        VSTORE + use_mask=1         Threshold (§10.6)
Scalar float load        SLF                         LayerNorm (§10.3)
Scalar float constant    SFI                         All kernels
```

For scatter/gather patterns (embedding lookups, sparse indexing), see Chapter 9. For
multi-instance data staging via SRAM, see Chapter 12.
