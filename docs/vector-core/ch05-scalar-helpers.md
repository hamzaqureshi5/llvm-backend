# Chapter 5 — Engine Scalar Helpers

> **Scope of this chapter.** The vector core exposes a small set of 32-bit scalar registers
> `s0`–`s31` and a matching set of scalar arithmetic instructions. These are *not* the control
> core's `x0`–`x31` registers — they live entirely inside the vector engine and are reset to
> zero at the start of every kernel instance. This chapter explains why they exist, what each
> instruction does, and how they combine into the SIMT addressing pattern that every real kernel
> uses.

---

## 5.1  Why Scalars in a Vector Core?

The central promise of a vector processor is that one instruction does the same operation on
*every active lane simultaneously*. That is efficient precisely because the same computation
repeats across all lanes without requiring per-lane program counter branches. But to issue even
a single useful vector instruction the engine must first answer three questions that have
nothing to do with lanes:

1. **Where in memory does my data live?** A pointer is one address — a scalar.
2. **How many elements am I processing?** The vector-length `VL` is a scalar.
3. **Which slice of the array belongs to me?** In a multi-instance launch each instance has a
   unique index — again, a scalar.

If the vector engine had no scalar registers of its own, the control core would have to compute
these values and somehow pass them to the engine before each instruction, which would serialize
the pipeline and defeat the purpose of a separate vector engine. The solution is to give the
engine its own small register file of scalars — `s0`–`s31` — so that address computation and
loop bookkeeping can run *inside* the engine without touching the control core.

### The analogy: GPU thread indices

A GPU SIMT core (e.g., a CUDA thread) executes the same program for every thread in a block,
but each thread reads its unique `threadIdx` and `blockIdx` to figure out which memory region
it owns. Those index values are scalars inside the thread — they are not vectors, even though
the goal is to process a vector of data. The NPU vector core uses the identical pattern:
`s0` at entry holds the *instance id*, playing exactly the role of `threadIdx`.

### Register file summary

```
 ┌─────────────────────────────────────────────────────────────────┐
 │              Engine Scalar Register File  s0 – s31             │
 │                                                                 │
 │  s0  ← instance id (set by kernel launch, readable via KID)   │
 │  s1  ← args base pointer (set by kernel launch)               │
 │  s2  ─┐                                                        │
 │  s3  ─┤  free for the kernel to use as temporaries            │
 │  ...  ─┤  (address computation, loop counters, scalars)        │
 │  s31 ─┘                                                        │
 │                                                                 │
 │  Each register is a union { float f; int64_t i; }             │
 │  — the same bits are reinterpreted as float (f) or int (i)    │
 │    depending on the instruction that writes/reads them.        │
 └─────────────────────────────────────────────────────────────────┘
```

The union type (`sreg_t` in `npu.h`) is critical: `SLI` writes `s[rd].i`; `SFI` writes
`s[rd].f`. Reading with the wrong interpretation gives garbage, so the convention is:
*address and loop-counter registers use the `.i` interpretation; floating-point constant
registers use the `.f` interpretation*.

### Entry convention

When the control core issues `C_KLAUNCH`, the simulator calls `run_vector_instance` with the
current grid index `g` (0 … grid−1) and the `args` pointer. The very first thing
`run_vector_instance` does — before executing instruction 0 — is:

```c
n->s[0].i = inst_id;   /* s0 = grid instance id  */
n->s[1].i = args_addr; /* s1 = args base pointer */
```

All other scalar registers start at zero. The kernel's job in its first few instructions is
almost always to *read parameters out of the args block via `SLW`/`SLF`* and *compute a
per-instance memory offset from `s0`*.

---

## 5.2  Instruction Reference — Engine Scalar Operations

Each instruction in this section operates solely on the `s` registers (or on one scalar and
one memory location). None of them touch the vector register file `v0`–`v31`. All are
single-cycle in the cycle model (see `vec_cost` in `npu.c`), because they mirror the trivial
operations of a RISC-V scalar core.

---

### 5.2.1  SLI  rd, imm — Scalar Load Integer Immediate

**Semantics:**
```
s[rd].i = imm
```

Load a small signed integer constant into a scalar register. `imm` is embedded directly in
the instruction struct's `imm` field (a 64-bit field in the simulator; a real encoder would
sign-extend a narrower field).

**Simulator implementation (`npu.c` line 209):**
```c
case V_SLI:   n->s[I->rd].i = I->imm; break;
```

**Typical uses:**

| Purpose | Example |
|---------|---------|
| Load a stride (bytes per element × vector-length) | `{ V_SLI, .rd=4, .imm=64 }` → s4 = 64 |
| Load a loop trip count | `{ V_SLI, .rd=5, .imm=32 }` → s5 = 32 |
| Load a fixed offset into an args block | `{ V_SLI, .rd=6, .imm=16 }` → s6 = 16 |
| Load the integer constant zero | `{ V_SLI, .rd=7, .imm=0 }` → s7 = 0 |

**Why not use SFI for zero?** SFI writes `s[rd].f`, so the bits are IEEE 754 zero (which
happens to be the same bit pattern as integer zero, but this is an implementation accident).
SLI with `imm=0` is the unambiguous way to load integer zero for use in address arithmetic.

**Example — loading a stride constant:**
```c
/* Process 16 floats per instance; stride = 16 * sizeof(float) = 64 bytes */
{ .op=V_SLI, .rd=4, .imm=64 }
/* Now s4 = 64 and can be used in SMUL or SADDI */
```

---

### 5.2.2  SFI  rd, fimm — Scalar Load Float Immediate

**Semantics:**
```
s[rd].f = (float)fimm
```

Load a floating-point constant into a scalar register. `fimm` is stored as `double` in the
instruction struct (`I->fimm`) and narrowed to `float` on write.

**Simulator implementation:**
```c
case V_SFI:   n->s[I->rd].f = (float)I->fimm; break;
```

**Typical uses:**

| Purpose | Example |
|---------|---------|
| Broadcast constant to all lanes via VBCAST | `{ V_SFI, .rd=6, .fimm=0.0 }` |
| Load a scale factor for normalization | `{ V_SFI, .rd=7, .fimm=0.00390625 }` (= 1/256) |
| Load a bias or threshold | `{ V_SFI, .rd=8, .fimm=1e-6 }` (epsilon for layernorm) |
| Load `100.0` for a bias test | seen in `demo_predication` |

**Pattern — broadcast a constant to every lane:**
```c
{ .op=V_SFI,    .rd=6,  .fimm=0.0    }   /* s6 = 0.0f               */
{ .op=V_VBCAST, .vd=1,  .rs=6        }   /* v1[l] = 0.0 for all l   */
/* now v1 is a vector of zeros, usable as a comparison operand        */
```
This two-instruction sequence — `SFI` then `VBCAST` — is the standard way to produce a
vector of a compile-time constant. It is used repeatedly in the predication demo and in the
softmax kernel.

**Assembly syntax** (as written in `demo_assembler`):
```asm
SFI  s6, 0.0
```

---

### 5.2.3  SADD  rd, rs1, rs2 — Scalar Add (Register + Register)

**Semantics:**
```
s[rd].i = s[rs1].i + s[rs2].i   (64-bit integer addition)
```

**Simulator implementation:**
```c
case V_SADD:  n->s[I->rd].i = n->s[I->rs1].i + n->s[I->rs2].i; break;
```

**Primary use — pointer arithmetic:** Once a per-instance offset has been computed (see
SMULI and SMUL below), it must be added to the args base to produce the actual memory
address. This is the job of SADD:

```c
/* s2 = args_base (from s1 on entry or loaded via SLW)    */
/* s5 = per-instance byte offset (computed via SMUL+SMULI) */
{ .op=V_SADD, .rd=6, .rs1=2, .rs2=5 }
/* s6 = args_base + per_instance_offset  → ready for VLOAD */
```

This exact pattern appears in every multi-instance kernel in `examples.c` (demos 2, 5, 6, 7).

**Assembly:**
```asm
SADD s6, s2, s5
```

---

### 5.2.4  SADDI  rd, rs1, imm — Scalar Add Immediate

**Semantics:**
```
s[rd].i = s[rs1].i + imm
```

**Simulator implementation:**
```c
case V_SADDI: n->s[I->rd].i = n->s[I->rs1].i + I->imm; break;
```

SADDI is to SADD what ADDI is to ADD in RISC-V: it encodes the constant directly in the
instruction, avoiding the need to first `SLI` the constant into a register. This makes it
the preferred instruction for advancing a pointer by a known offset.

**Common patterns:**

```c
/* Advance a pointer by one vector-length worth of bytes (stride = 64) */
{ .op=V_SADDI, .rd=3, .rs1=3, .imm=64 }   /* s3 += 64 */

/* Compute the address of the second field in an args block             */
/* (args is laid out as: [ptr_a, ptr_b, ptr_c, n] each 4 bytes apart) */
{ .op=V_SADDI, .rd=10, .rs1=1, .imm=4 }   /* s10 = s1 + 4 = addr of ptr_b */
```

**Why SADDI and not always SADD?** Saving a register and an instruction. Loading `64` into a
free scalar register before every loop-advance wastes register pressure. SADDI encodes the
constant inline, which is cheaper when the constant fits in the immediate field.

---

### 5.2.5  SSUB  rd, rs1, rs2 — Scalar Subtract

**Semantics:**
```
s[rd].i = s[rs1].i - s[rs2].i
```

**Simulator implementation:**
```c
case V_SSUB:  n->s[I->rd].i = n->s[I->rs1].i - n->s[I->rs2].i; break;
```

SSUB is most commonly used to compute a remaining element count or a backwards index:

```c
/* Compute elements remaining: total - already_processed */
{ .op=V_SSUB, .rd=8, .rs1=4, .rs2=9 }   /* s8 = n - done */

/* Decrement a loop counter */
{ .op=V_SSUB, .rd=5, .rs1=5, .rs2=temp_one }
```

*Note:* The `V_LOOP` instruction also decrements a counter implicitly (it decrements the
register it tests before branching). For simple counted loops, `LOOP` is cleaner. `SSUB` is
useful when you need the remaining count as a value — e.g., to pass it to `VSETVL` for
tail-loop handling.

---

### 5.2.6  SMUL  rd, rs1, rs2 — Scalar Multiply (Register × Register)

**Semantics:**
```
s[rd].i = s[rs1].i * s[rs2].i
```

**Simulator implementation:**
```c
case V_SMUL:  n->s[I->rd].i = n->s[I->rs1].i * n->s[I->rs2].i; break;
```

SMUL is the first half of the canonical SIMT offset pattern. Instance ID (`s0`) is multiplied
by the number of elements each instance processes to get an element index; a subsequent SMULI
then converts that element index to a byte offset.

**From `demo_softmax` in `examples.c` (line 87–88):**
```c
{ .op=V_SMUL,  .rd=5, .rs1=0, .rs2=4 },  /* s5 = instance_id * n   (element offset) */
{ .op=V_SMULI, .rd=5, .rs1=5, .imm=4  }, /* s5 *= 4                (byte offset)    */
```

Here `s0` = instance id, `s4` = N (loaded from args). This produces `s5 = id * N * 4`, the
byte offset to the start of this instance's row.

---

### 5.2.7  SMULI  rd, rs1, imm — Scalar Multiply by Immediate

**Semantics:**
```
s[rd].i = s[rs1].i * imm
```

**Simulator implementation:**
```c
case V_SMULI: n->s[I->rd].i = n->s[I->rs1].i * I->imm; break;
```

SMULI has two distinct roles in practice:

**Role 1 — byte-offset conversion.** After SMUL gives an element offset, SMULI multiplies
by `sizeof(float) = 4` to convert to bytes:
```c
{ .op=V_SMULI, .rd=5, .rs1=5, .imm=4 }   /* element_offset → byte_offset */
```

**Role 2 — direct per-instance byte offset** when the elements-per-instance count is a
compile-time constant. If every instance always handles exactly 64 floats (256 bytes), and
the instance id is already in `s0`, then:
```c
{ .op=V_SMULI, .rd=2, .rs1=0, .imm=256 }  /* s2 = instance_id * 256 bytes */
```
This collapses the SMUL+SMULI pair into a single instruction when the per-instance size is
known at compile time.

**Full addressing sequence (combined view):**

```
   s0 = instance id  (set by launcher)
   s1 = args_base    (set by launcher)
   s4 = N per row    (loaded via SLW from args)

   SMUL  s5, s0, s4     s5 = id * N          (element index)
   SMULI s5, s5, 4      s5 = id * N * 4      (byte offset)
   SADD  s6, s2, s5     s6 = x_base + offset (row pointer for this instance)
```

---

### 5.2.8  SLW  rd, s[rs]+imm — Scalar Load Word (Integer)

**Semantics:**
```
s[rd].i = *(int32_t*)(s[rs].i + imm)
```

**Simulator implementation:**
```c
case V_SLW:   n->s[I->rd].i = ld_w(n, addr); break;
/* where addr = use_base ? (s[rs] + imm) : imm */
```

SLW is almost always the *first instruction in a kernel*. The kernel launch convention passes
only two things to the vector core: the instance id (`s0`) and a pointer to an args block
(`s1`). All other parameters — tensor base addresses, shapes, strides, loop counts — live in
that args block in main memory (or scratchpad). SLW is how the kernel reads them out.

**Args block layout (a common convention):**
```
Offset  Content
   0    int32_t: pointer to tensor A
   4    int32_t: pointer to tensor B
   8    int32_t: pointer to output tensor C
  12    int32_t: element count N
  16    float:   scale factor (if needed)
  ...
```

**Prologue from `demo_vadd` (`examples.c` lines 36–39):**
```c
{ .op=V_SLW, .rd=2, .use_base=1, .rs=1, .imm=0  },   /* s2 = args[0] = ptr_a */
{ .op=V_SLW, .rd=3, .use_base=1, .rs=1, .imm=4  },   /* s3 = args[1] = ptr_b */
{ .op=V_SLW, .rd=4, .use_base=1, .rs=1, .imm=8  },   /* s4 = args[2] = ptr_c */
{ .op=V_SLW, .rd=5, .use_base=1, .rs=1, .imm=12 },   /* s5 = args[3] = n     */
```

This four-instruction prologue is the template for virtually every kernel: four SLW
instructions (one per parameter), all using `s1` as the base with increasing immediate
offsets.

**Assembly syntax** (from `demo_assembler`):
```asm
SLW s2, 0(s1)      ; s2 = *(s1 + 0)  → ptr_a
SLW s3, 4(s1)      ; s3 = *(s1 + 4)  → ptr_b
SLW s4, 8(s1)      ; s4 = *(s1 + 8)  → n
```

**What does SLW load from?** Any address in the NPU's unified 32-bit address space — either
DRAM (high bit clear) or the on-chip scratchpad (high bit set). Since the args block is
typically small and access to it happens once at kernel startup, it is usually placed in DRAM.
The memory latency cost is charged to `vector_cycles` just like VLOAD (see `vec_cost` in
`npu.c`). For frequently accessed parameters in tight loops, copying the args to scratchpad
before launching amortizes this cost.

---

### 5.2.9  SLF  rd, s[rs]+imm — Scalar Load Float

**Semantics:**
```
s[rd].f = *(float*)(s[rs].i + imm)
```

**Simulator implementation:**
```c
case V_SLF:   n->s[I->rd].f = ld_f(n, addr); break;
```

SLF is identical to SLW in addressing but interprets the 32 bits read from memory as an IEEE
754 single-precision float and writes them to the `.f` interpretation of the scalar register.
It is used when the args block contains floating-point parameters — scale factors, epsilon
values, bias terms.

**Example — loading a layernorm epsilon from the args block:**
```c
/* args block: [ptr_x, ptr_y, n, epsilon]
   offsets:       0     4    8     12    */
{ .op=V_SLW, .rd=2, .use_base=1, .rs=1, .imm=0  },  /* s2 = ptr_x         */
{ .op=V_SLW, .rd=3, .use_base=1, .rs=1, .imm=4  },  /* s3 = ptr_y         */
{ .op=V_SLW, .rd=4, .use_base=1, .rs=1, .imm=8  },  /* s4 = n             */
{ .op=V_SLF, .rd=5, .use_base=1, .rs=1, .imm=12 },  /* s5 = epsilon (f32) */
/* Later: broadcast s5 to all lanes and add to variance before VRSQRT */
{ .op=V_VBCAST, .vd=3, .rs=5 },                     /* v3[l] = epsilon    */
```

---

### 5.2.10  SRECIP  rd, rs1 — Scalar Reciprocal

**Semantics:**
```
s[rd].f = 1.0f / s[rs1].f
```

**Simulator implementation:**
```c
case V_SRECIP: n->s[I->rd].f = frecip(n->s[I->rs1].f); break;
/* frecip is simply: static float frecip(float x) { return 1.0f / x; } */
```

SRECIP computes a scalar reciprocal — the result is a single float in a scalar register, not
a vector. This is the workhorse of the normalization step in many kernels.

**Why a dedicated instruction?** The reduction-then-normalize pattern is so pervasive in
neural network operations that the instruction set gives it a dedicated opcode rather than
requiring a VRECIP (which operates on an entire vector, at vector cost) for what is logically
a single scalar divide:

```
VREDSUM  s_sum, v_exp        ← reduce N lanes to one scalar
SRECIP   s_inv, s_sum        ← compute 1/sum at scalar cost (not vector cost)
VBCAST   v_inv, s_inv        ← broadcast the scalar result back to all lanes
VMUL     v_out, v_exp, v_inv ← apply normalization
```

If SRECIP did not exist, the programmer would need to:
1. Use a VBCAST to put `s_sum` into every lane of some vector `v_tmp`.
2. Use VRECIP to compute `1/v_tmp[l]` for all `vl` lanes (at full vector cost).
3. Then use VMUL.

That approach computes `vl` redundant reciprocals (all identical) at vector throughput when
only one reciprocal is needed. SRECIP avoids this waste.

**Complete softmax denominator pattern** (from `demo_softmax`, `examples.c` lines 96–99):
```c
{ .op=V_VREDSUM, .rd=10, .vs=0 },    /* s10 = sum(exp(x-max))           */
{ .op=V_SRECIP,  .rd=11, .rs1=10 }, /* s11 = 1 / sum           (SRECIP)*/
{ .op=V_VBCAST,  .vd=1,  .rs=11 },  /* v1[l] = 1/sum for all l          */
{ .op=V_VMUL,    .vd=0, .vs1=0, .vs2=1 }, /* v0 = exp(x-max) / sum      */
```

**Assembly** (from `demo_assembler`):
```asm
VREDSUM  s10, v0
SRECIP   s11, s10
VBCAST   v1, s11
VMUL     v0, v0, v1
```

---

### 5.2.11  KID  rd — Get Kernel Instance ID

**Semantics:**
```
s[rd] = instance_id   (the same value that was placed in s0 at entry)
```

**Simulator implementation:**
```c
case V_KID:   n->s[I->rd].i = inst_id; break;
/* inst_id is the argument passed to run_vector_instance, not s[0] */
```

KID re-reads the kernel instance id into any scalar register at any point in the kernel. It
answers the question "which grid cell am I?" without requiring the programmer to preserve `s0`
across the entire kernel.

**Why not just use s0 directly?** Three reasons:

1. **s0 can be overwritten.** There is nothing preventing a poorly written kernel from
   accidentally writing s0. KID provides a read that goes back to the *hardware-provided*
   instance id rather than whatever s0 currently contains.

2. **Readability.** `KID s2` is more self-documenting than `VMOV v_trash, v_trash` followed
   by recovering s0 from memory.

3. **Multiple indexing points.** A kernel that needs the instance id at two different points
   (e.g., once for the input pointer, once for the output pointer with a different layout)
   can call KID twice rather than keeping s0 live across the whole body.

**Full addressing pattern using KID:**
```c
{ .op=V_KID,    .rd=2 },                        /* s2 = my instance id          */
{ .op=V_SMULI,  .rd=2, .rs1=2, .imm=VL*4 },    /* s2 = id * bytes_per_instance */
{ .op=V_SADD,   .rd=3, .rs1=ptr_a, .rs2=2 },   /* s3 = base_a + my_offset      */
{ .op=V_VLOAD,  .vd=0, .use_base=1, .rs=3 },   /* v0 = MY slice of tensor A    */
```

---

## 5.3  The SIMT Addressing Pattern

Understanding this pattern is the single most important thing you can know about programming
the vector core. Every real multi-instance kernel uses it. It is worth studying carefully.

### The problem statement

You have an array of `TOTAL` floats, and you launch `GRID` kernel instances. Each instance
must process exactly `CHUNK = TOTAL / GRID` elements. Instance 0 processes `[0, CHUNK)`,
instance 1 processes `[CHUNK, 2*CHUNK)`, and so on. All instances run the *same code*, but
each reads from a *different address*.

The key arithmetic:
```
my_byte_offset = instance_id * CHUNK * sizeof(float)
my_data_ptr    = base_address + my_byte_offset
```

### The instruction sequence

```c
/* --- Assumptions at entry ---
   s0 = instance_id
   s1 = args_base

   Args block at args_base:
     [0]: int32  ptr_x   (base address of the array)
     [4]: int32  N       (elements per instance = CHUNK)
*/

{ .op=V_SLW,   .rd=2, .use_base=1, .rs=1, .imm=0 },   /* s2 = ptr_x        */
{ .op=V_SLW,   .rd=4, .use_base=1, .rs=1, .imm=4 },   /* s4 = N per inst   */
{ .op=V_SMUL,  .rd=5, .rs1=0, .rs2=4 },               /* s5 = id * N       */
{ .op=V_SMULI, .rd=5, .rs1=5, .imm=4 },               /* s5 = id * N * 4   */
{ .op=V_SADD,  .rd=6, .rs1=2, .rs2=5 },               /* s6 = ptr_x + off  */
{ .op=V_VSETVL,.rd=7, .rs1=4 },                        /* VL = N            */
{ .op=V_VLOAD, .vd=0, .use_base=1, .rs=6 },            /* v0 = MY data      */
```

### Concrete example: 16 instances, 16 floats each

```
Total array: 256 floats = 1024 bytes starting at address 0x2000

                base = 0x2000
                │
                ▼
Instance  0  →  [0x2000 .. 0x203F]   floats[  0..  15]
Instance  1  →  [0x2040 .. 0x207F]   floats[ 16..  31]
Instance  2  →  [0x2080 .. 0x20BF]   floats[ 32..  47]
Instance  3  →  [0x20C0 .. 0x20FF]   floats[ 48..  63]
Instance  4  →  [0x2100 .. 0x213F]   floats[ 64..  79]
Instance  5  →  [0x2140 .. 0x217F]   floats[ 80..  95]
Instance  6  →  [0x2180 .. 0x21BF]   floats[ 96.. 111]
Instance  7  →  [0x21C0 .. 0x21FF]   floats[112.. 127]
Instance  8  →  [0x2200 .. 0x223F]   floats[128.. 143]
Instance  9  →  [0x2240 .. 0x227F]   floats[144.. 159]
Instance 10  →  [0x2280 .. 0x22BF]   floats[160.. 175]
Instance 11  →  [0x22C0 .. 0x22FF]   floats[176.. 191]
Instance 12  →  [0x2300 .. 0x233F]   floats[192.. 207]
Instance 13  →  [0x2340 .. 0x237F]   floats[208.. 223]
Instance 14  →  [0x2380 .. 0x23BF]   floats[224.. 239]
Instance 15  →  [0x23C0 .. 0x23FF]   floats[240.. 255]
```

For each instance, the SMULI of `id * 64` produces:

```
id=0  → 0x0000  offset → ptr = 0x2000
id=1  → 0x0040  offset → ptr = 0x2040
id=2  → 0x0080  offset → ptr = 0x2080
id=3  → 0x00C0  offset → ptr = 0x20C0
...
id=15 → 0x03C0  offset → ptr = 0x23C0
```

### The grid picture

```
 Control core launches: C_KLAUNCH  kid=0, grid=16, args=0x1000
                                      │
                     ┌────────────────┴───────────────────────────┐
                     │           Kernel Scheduler                  │
                     │   Spreads 16 instances across N_CORES       │
                     └─────────────┬──────────────────────────────┘
                                   │
        ┌──────────┬───────────────┼───────────────┬──────────────┐
        ▼          ▼               ▼               ▼              ▼
  Core 0       Core 1         Core 2   ...    Core 7        (more cores)
  inst 0       inst 1         inst 2          inst 7
  inst 8       inst 9         inst 10         inst 15
        │          │               │               │
        ▼          ▼               ▼               ▼
  floats[0..15] [16..31]       [32..47]      [112..127]
  floats[128..143][144..159] [160..175]     [240..255]
```

Each core runs its assigned instances sequentially (the simulator models cores as a cycle-max
model, not real threads). The important point is that the code is identical on every core; only
`s0` differs, and the addressing arithmetic magnifies that difference into non-overlapping
memory regions.

### Softmax: the SIMT pattern applied to rows

The softmax kernel in demo 2 uses the same pattern but the "chunk" is one row of a matrix:

```
Matrix X: ROWS rows, N columns each.
Launch: grid = ROWS (one instance per row).

Each instance:
  s0 = row index
  s4 = N  (columns per row, from args)
  s5 = row_index * N            (SMUL)
  s5 = row_index * N * 4        (SMULI by 4)
  s6 = x_base + row_byte_offset (SADD)
  s7 = y_base + row_byte_offset (SADD)
```

Instance 0 processes row 0. Instance 1 processes row 1. They run concurrently on different
cores. No synchronization is needed because they write to disjoint memory regions.

---

## 5.4  Scalar Register Allocation Conventions

There is no hardware-enforced calling convention for scalar registers beyond the entry state
(`s0` = id, `s1` = args). The following conventions are used throughout the examples and are
recommended for new kernels:

```
 Register  Conventional role
 ────────  ─────────────────────────────────────────────────────────────
 s0        Instance id (set by launcher; re-read via KID if clobbered)
 s1        Args base pointer (set by launcher; keep live across kernel)
 s2        Primary tensor A pointer (loaded via SLW from args)
 s3        Primary tensor B pointer (loaded via SLW from args)
 s4        N (element count; loaded via SLW from args)
 s5        Scratch / per-instance byte offset (computed from s0 * s4 * 4)
 s6        Row pointer for A (= s2 + s5)
 s7        Row pointer for B / output (= s3 + s5)
 s8        VL (returned by VSETVL; also holds requested length)
 s9        Reduction result A (VREDMAX / VREDMIN)
 s10       Reduction result B (VREDSUM)
 s11       Scalar reciprocal (SRECIP result)
 s12–s31   Free temporaries
 ────────  ─────────────────────────────────────────────────────────────
```

These are conventions only. The simulator enforces nothing. A compiler backend targeting the
vector core would establish its own register allocation.

---

## 5.5  Scalar Register Summary Table

| Instruction | Semantics | Interpretation | Common use |
|-------------|-----------|----------------|------------|
| `SLI rd, imm` | `s[rd].i = imm` | integer | load stride constant, loop count |
| `SFI rd, fimm` | `s[rd].f = fimm` | float | load scale/bias/threshold |
| `SADD rd, rs1, rs2` | `s[rd].i = s[rs1].i + s[rs2].i` | integer | pointer arithmetic |
| `SADDI rd, rs1, imm` | `s[rd].i = s[rs1].i + imm` | integer | advance pointer by stride |
| `SSUB rd, rs1, rs2` | `s[rd].i = s[rs1].i − s[rs2].i` | integer | remaining element count |
| `SMUL rd, rs1, rs2` | `s[rd].i = s[rs1].i × s[rs2].i` | integer | element-offset computation |
| `SMULI rd, rs1, imm` | `s[rd].i = s[rs1].i × imm` | integer | byte-offset (× sizeof(float)) |
| `SLW rd, rs+imm` | `s[rd].i = *(int32_t*)(s[rs]+imm)` | integer | read param from args block |
| `SLF rd, rs+imm` | `s[rd].f = *(float*)(s[rs]+imm)` | float | read float param from args |
| `SRECIP rd, rs1` | `s[rd].f = 1.0f / s[rs1].f` | float | softmax/layernorm denominator |
| `KID rd` | `s[rd].i = instance_id` | integer | recover instance id |

**Cycle cost:** All scalar helper instructions cost 1 cycle each in the NPU cycle model (see
`vec_cost` in `npu.c`, the `default` branch for `V_SLI`, `V_SFI`, `V_SADD`, etc.). This is
plausible for a real implementation because these operations touch no array datapath — they are
performed in a narrow scalar ALU that runs in parallel with (or slightly before) the vector
datapath.

---

## 5.6  Working Example: Complete Kernel Prologue

To make this concrete, here is the complete annotated prologue that appears in the actual
softmax kernel (`demo_softmax`, `examples.c`), with every instruction and its effect on the
scalar register file:

```
Args block layout (written by host before KLAUNCH):
  Address ARGS+0:  int32  X  = 0x2000  (base of input matrix)
  Address ARGS+4:  int32  Y  = 0x4000  (base of output matrix)
  Address ARGS+8:  int32  N  = 4       (columns per row)

Entry state (set by kernel launcher):
  s0 = instance_id   (0, 1, or 2 for a 3-row matrix)
  s1 = ARGS          (= 0x1000)
  All other s registers = 0

Instruction 0:
  SLW s2, 0(s1)          s2.i = *(int32*)(0x1000 + 0) = 0x2000  (X base)

Instruction 1:
  SLW s3, 4(s1)          s3.i = *(int32*)(0x1000 + 4) = 0x4000  (Y base)

Instruction 2:
  SLW s4, 8(s1)          s4.i = *(int32*)(0x1000 + 8) = 4       (N)

Instruction 3:
  SMUL s5, s0, s4        For instance 0: s5.i = 0 * 4 = 0
                         For instance 1: s5.i = 1 * 4 = 4
                         For instance 2: s5.i = 2 * 4 = 8

Instruction 4:
  SMULI s5, s5, 4        For instance 0: s5.i = 0 * 4 = 0
                         For instance 1: s5.i = 4 * 4 = 16
                         For instance 2: s5.i = 8 * 4 = 32

Instruction 5:
  SADD s6, s2, s5        For instance 0: s6.i = 0x2000 + 0  = 0x2000 (row 0 of X)
                         For instance 1: s6.i = 0x2000 + 16 = 0x2010 (row 1 of X)
                         For instance 2: s6.i = 0x2000 + 32 = 0x2020 (row 2 of X)

Instruction 6:
  SADD s7, s3, s5        For instance 0: s7.i = 0x4000 + 0  = 0x4000 (row 0 of Y)
                         For instance 1: s7.i = 0x4000 + 16 = 0x4010 (row 1 of Y)
                         For instance 2: s7.i = 0x4000 + 32 = 0x4020 (row 2 of Y)

Instruction 7:
  VSETVL s8, s4          VL = min(s4, VLEN_MAX) = 4
                         s8.i = 4

Instruction 8:
  VLOAD v0, 0(s6)        v0[0..3] = X[row 0]   (for inst 0)
                         v0[0..3] = X[row 1]   (for inst 1)
                         v0[0..3] = X[row 2]   (for inst 2)
```

After these 9 instructions (the prologue), each of the 3 instances has loaded its own
distinct row of the matrix into `v0`, even though every instance ran identical code. The
divergence came entirely from `s0` having a different value in each instance.

---

## 5.7  Design Rationale: Why These Eleven Instructions?

The eleven scalar helper instructions were chosen to cover exactly the operations needed for
SIMT addressing and loop control, nothing more:

- **Load immediates** (`SLI`, `SFI`): the minimum needed to materialize constants.
- **Arithmetic** (`SADD`, `SADDI`, `SSUB`, `SMUL`, `SMULI`): the minimum needed for offset
  computation. Notice there is no SDIV — integer division is not needed for power-of-two
  element sizes, and non-power-of-two layouts are handled by precomputing strides in the args
  block on the host.
- **Memory loads** (`SLW`, `SLF`): reading the args block is the only scalar memory access
  pattern. There is no scalar store (SSTORE) because kernels do not write scalar results to
  memory — they write vector results via VSTORE.
- **Reciprocal** (`SRECIP`): the one transcendental that needs a scalar form. No SEXP, SLOG,
  or SSQRT exists because those are always applied to vectors in practice.
- **Instance id** (`KID`): the one hardware-provided value that has no equivalent in the
  scalar arithmetic instructions.

This minimalism keeps the scalar engine small. In silicon, the scalar helpers would share the
instruction decode path with the vector instructions and use a fraction of the area of the
vector ALU.

---

*Next chapter: Chapter 6 covers the vector compute primitives — the Section 10 instructions
that operate across all active lanes simultaneously.*
