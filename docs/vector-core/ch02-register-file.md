# Chapter 2: The Register File

---

Every computation on the vector core happens through its register file. Before a single
instruction can produce a useful result, you need to understand what registers exist, what
they hold, how they are organized, and what state they carry when a kernel begins executing.
This chapter is a thorough treatment of all four register spaces the vector core exposes:
vector registers, scalar registers, mask registers, and the vector length register. It also
covers the state reset that happens at kernel launch and the implications of independent
register files for multi-instance execution.

The vector core's register file is not a single uniform namespace. It is four distinct files
with distinct purposes, distinct widths, and distinct interaction rules. Understanding why each
file exists — not just what it contains — is essential for writing efficient kernels.

---

## 2.1 Vector Registers: v0–v31

### 2.1.1 Overview

The vector registers are the primary working storage of the vector core. There are 32 of them,
named `v0` through `v31`. Each vector register holds up to `VLEN_MAX = 256` elements of type
`float32`. At any given moment, only the first `vl` elements (lanes) of each register are
considered *active* — the elements at indices `vl` through 255 are physically present in the
hardware but are not touched by any instruction while `vl` is at its current value.

```
VLEN_MAX = 256 float32 elements = 256 * 4 bytes = 1024 bytes per register
32 registers = 32 KB of vector register file total
```

In the simulator, each vector register is a C array:
```c
float v[NVREG][VLEN_MAX];  /* v[0..31][0..255] */
```

### 2.1.2 What Is a "Lane"?

The term *lane* refers to one element slot within a vector register. Lane `l` of register `v2`
is the floating-point value `v2[l]`. A lane is the unit of parallelism: every elementwise
instruction in the vector core operates on each lane independently and (conceptually)
simultaneously.

The mental model is that the vector core has `vl` copies of a scalar floating-point unit, all
executing the same operation at the same time on different data. Lane 0, lane 1, ..., lane
`vl-1` each process their own element. The results go back into the corresponding lane of the
destination register.

```
v0:  [ 1.0 | 2.0 | 3.0 | 4.0 | 5.0 | 6.0 | 7.0 | 8.0]   (vl = 8)
       L0    L1    L2    L3    L4    L5    L6    L7
v1:  [ 10.0 | 11.0 | 12.0 | 13.0 | 14.0 | 15.0 | 16.0 | 17.0 ]
       L0     L1     L2     L3     L4     L5     L6     L7

VADD v2, v0, v1  →  v2[l] = v0[l] + v1[l]  for each l in 0..vl-1

v2:  [ 11.0 | 13.0 | 15.0 | 17.0 | 19.0 | 21.0 | 23.0 | 25.0 ]
       L0     L1     L2     L3     L4     L5     L6     L7
```

This is the fundamental operation of a SIMD engine. The key insight is that the *same
instruction* is applied to *all active lanes* in a single hardware step (or in multiple
passes through the physical lanes, amortized). No explicit loop is written — the loop is
implicit in the vector instruction.

### 2.1.3 Physical Hardware Lanes vs. Logical Vector Length

The NPU has a distinction between *physical* lane count and *logical* vector length that is
important for understanding the cycle model.

The physical lane count is `HW_LANES = 16`. This means the hardware has 16 floating-point
units that operate in parallel. If `vl = 16`, a `VADD` completes in one pass. If `vl = 32`,
it requires two passes. If `vl = 256`, it requires 16 passes.

The logical vector length `vl` can be anywhere from 0 to 256 — it is not restricted to
multiples of `HW_LANES`. A kernel may set `vl = 7` for the tail iteration of a loop where the
array length was not a multiple of the tile size. The hardware handles this transparently: it
uses 7 lanes, and the remaining 9 physical lanes are gated off for that instruction.

```
                 Physical lane count: HW_LANES = 16
                 ┌────────────────────────────────────────────────────────────────┐
                 │  HW lane  │  0 │  1 │  2 │  3 │  4 │  5 │  6 │  7 │  8 ...  │
                 └────────────────────────────────────────────────────────────────┘

When vl = 16:    All 16 lanes active → 1 pass through the physical array
When vl = 32:    32 elements → 2 passes through the 16-wide physical array
When vl =  7:    7 elements → 1 pass, lanes 7-15 are inactive (gated)
When vl = 256:   256 elements → 16 passes through the 16-wide physical array
```

This distinction matters for the cycle model (discussed in Chapter 8) but is transparent to
the programmer. From a correctness standpoint, you think in terms of logical lanes 0 through
`vl-1`. From a performance standpoint, the cost of an instruction scales as
`ceil(vl / HW_LANES)` passes, with transcendentals costing 4 cycles per pass and most
elementwise operations costing 1 cycle per pass.

### 2.1.4 Diagram: A Vector Register at vl=16

```
  v0 register (vl = 16 active lanes, VLEN_MAX = 256 slots total):

  Active lanes (0..15):
  ┌──────┬──────┬──────┬──────┬──────┬──────┬──────┬──────┐
  │ 1.00 │ 2.00 │ 3.00 │ 4.00 │ 5.00 │ 6.00 │ 7.00 │ 8.00 │
  │  L0  │  L1  │  L2  │  L3  │  L4  │  L5  │  L6  │  L7  │
  ├──────┼──────┼──────┼──────┼──────┼──────┼──────┼──────┤
  │ 9.00 │10.00 │11.00 │12.00 │13.00 │14.00 │15.00 │16.00 │
  │  L8  │  L9  │ L10  │ L11  │ L12  │ L13  │ L14  │ L15  │
  └──────┴──────┴──────┴──────┴──────┴──────┴──────┴──────┘

  Inactive lanes (16..255) — present but not touched:
  ┌──────┬──────┬──────┬──────┬──────┬──────┬──────┬──────┐
  │ 0.00 │ 0.00 │ 0.00 │ 0.00 │ 0.00 │ ... │ 0.00 │ 0.00 │
  │ L16  │ L17  │ L18  │ L19  │ L20  │     │ L254 │ L255 │
  └──────┴──────┴──────┴──────┴──────┴──────┴──────┴──────┘
  (initialized to 0.0 at kernel launch; untouched by instructions while vl=16)
```

### 2.1.5 Float32 Is Universal

Every value in a vector register is stored as a single-precision IEEE 754 floating-point
number (float32, 32 bits: 1 sign, 8 exponent, 23 mantissa). This holds even when the logical
content of the value is an integer or an 8-bit quantized value.

Why? Because the vector core is designed to be simple: one data type, one format, one set of
arithmetic rules. The alternative — having integer registers and floating-point registers as
separate namespaces — would roughly double the instruction encoding space, complicate the
register file design, and add data-type tracking to every instruction.

The implication is that "integer" operations on the vector core are actually float operations
that happen to be integers. A `V_VGATHER` instruction uses a vector register as an index
vector: the values in that register are read as `int32_t` by casting the float bits to int.
The `V_VCVT` instruction handles the conversions between the logical representation (int8,
int32) and the float32 storage format: it rounds, clamps, and then stores the result as a
float whose value happens to be the quantized integer.

```
V_VCVT with CVT_I8:
  For each lane l:
    x = v_src[l]            (float, e.g. 1.7f)
    r = roundf(x)           (float, e.g. 2.0f)
    r = clamp(r, -128, 127) (float, still 2.0f)
    v_dst[l] = r            (stored as float 2.0f, represents int8 value 2)
```

This design is sometimes called "float-everywhere" or "functional integer in float": integer
arithmetic is performed by treating float results as integers when the values are whole numbers
within the representable range of float32 (approximately ±16 million without loss of precision).

The practical consequence for you as a programmer: when you use `V_VGATHER` with an index
vector, make sure the indices in that vector register are float representations of integers
(e.g., 0.0f, 1.0f, 2.0f, ...), not bit-cast integers. The gather implementation casts the
float to `int32_t` before using it as an index. If you have an index vector `idx[l] = l` (the
identity permutation), you get it by constructing a sequence `0.0, 1.0, 2.0, ...` in a vector
register, which you might build by loading from a precomputed array or by induction.

### 2.1.6 Reset at Kernel Launch

Every vector register is reset to all-zeros at the beginning of each kernel instance. The
simulator does this explicitly:

```c
memset(n->v, 0, sizeof n->v);
```

This guarantees that a kernel that reads a vector register it has not written will see 0.0f,
not garbage. This guarantee is essential for correctness when a kernel uses predication: with
some lanes masked off, the values in those lanes are not written by predicated instructions —
but they were zero from the start, so the output register contains meaningful values in active
lanes and 0.0f in inactive lanes. Whether that 0.0f is meaningful depends on the algorithm.

The zero initialization also means you do not need to initialize vector registers before using
them as accumulators if zero is the correct starting value (e.g., a partial sum starting at
0.0f, as in `V_VFMA`).

### 2.1.7 Why 32 Vector Registers?

The choice of 32 is common across RISC-V, x86, and ARM: it is large enough to keep a
moderately complex kernel's working set in registers without spilling to memory, while keeping
the instruction encoding tractable (5 bits to encode a register index fits in a 32-bit word).

For a kernel like softmax, 4–5 vector registers suffice (input, shifted, exponents, scale
broadcast, output). A more complex kernel like attention score computation might use 8–12.
Having 32 gives a comfortable margin for the most demanding kernels without the complexity of
register allocation becoming a constant bottleneck.

---

## 2.2 Engine Scalar Registers: s0–s31

### 2.2.1 Why Scalar Registers Exist Inside the Vector Core

At first glance, it might seem unnecessary for the vector core to have its own scalar registers.
After all, the control core has 32 scalar registers (`x0`–`x31`). Why not just use those?

The answer is isolation and latency. The vector core is a separate execution engine. Once a
kernel is launched, it runs independently of the control core. The control core may be busy
computing addresses for the next tile, waiting on a DMA, or simply idle. The vector kernel
cannot and should not block waiting to read a control-core register. The vector core needs its
own local scalars for:

**Address computation.** A vector load instruction (`V_VLOAD`) takes a base address. That base
address must be an integer (a byte offset into NPU memory). To compute it, the kernel needs
to do integer arithmetic: multiply the instance ID by the row stride, add the base pointer.
This arithmetic happens in scalar registers (`V_SMUL`, `V_SADD`).

**Loop counters.** If a kernel processes its input in multiple chunks (more elements than `vl`
can handle in one pass), it needs a counter in a scalar register that it decrements each
iteration, branching back to the top of the loop while the counter is nonzero (`V_LOOP`).

**Reduction results.** When a reduction instruction (`V_VREDSUM`, `V_VREDMAX`, etc.) executes,
it writes its result to a scalar register. The scalar result (e.g., the max value for softmax
stability, or the sum for normalization) is then used in subsequent vector instructions via
`V_VBCAST` (broadcast the scalar to all lanes).

**Scalar intermediate values.** The softmax algorithm needs the reciprocal of the sum: `1 /
sum`. This is a scalar operation (`V_SRECIP`) on the scalar result of `V_VREDSUM`. The result
is held in a scalar register before being broadcast to all vector lanes.

Without scalar registers, all of these would require round-trips through memory: store the
reduction result to memory, load it back into a vector lane, extract it. The scalar registers
make these operations fast and register-resident.

### 2.2.2 Count and Width

There are 32 scalar registers, named `s0` through `s31`. Each is 64 bits wide.

### 2.2.3 The Union: Float and Integer in One Register

The most important property of the scalar registers — and the one most likely to cause
confusion — is that each register is a *union* of two interpretations:

```c
typedef union { float f; int64_t i; } sreg_t;
```

The same 64 bits can be read as a `float` (the `.f` field) or as an `int64_t` (the `.i`
field). The register does not carry a type tag — the type is determined by which instruction
reads or writes it.

Consider what this means concretely:

- `V_SLI s2, 0x2000` writes `0x2000` (8192 as an integer) to `s2.i`. If you then use `s2`
  as a base address for `V_VLOAD v0, 0(s2)`, the integer value `8192` is used as a byte
  offset. Correct.
- `V_VREDSUM s10, v0` writes the floating-point sum of `v0`'s lanes to `s10.f`. If you then
  use `V_SRECIP s11, s10` (which reads `s10.f`), you get the reciprocal of that float. Correct.
- `V_SFI s6, 0.0` writes `0.0f` to `s6.f`. If you then use `V_VBCAST v1, s6` (which reads
  `s6.f`), you get a vector of zeros. Correct.
- If you write a float to a scalar register via `V_VREDSUM` and then try to use it as a
  memory address via `V_SLW`, you will read from the address equal to the *integer bit pattern*
  of that float — almost certainly not what you intended.

The programmer is responsible for tracking the semantic type of each scalar register's current
value and ensuring that it is read by an instruction that interprets it correctly. There is no
hardware enforcement.

This design mirrors the RISC-V integer register file, where all registers hold bits and the
interpretation depends on the instruction. It avoids the cost of a tagged register file while
still supporting both integer (address) and float (arithmetic) use cases.

### 2.2.4 Diagram: The Scalar Register as a Union

```
  s10 register (64 bits):

  As int64_t (.i):
  ┌──────────────────────────────────────────────────────────────────┐
  │  bit 63                                                  bit 0   │
  │  ██████████████████████████████████████████████████████████████  │
  │  Treated as a signed 64-bit integer (address, loop counter, etc) │
  └──────────────────────────────────────────────────────────────────┘

  As float (.f):
  ┌────────────┬───────────────────┬──────────────────────────────────┐
  │  sign (1b) │   exponent (8b)   │        mantissa (23b)            │
  │            │  (in low 32 bits) │                                  │
  └────────────┴───────────────────┴──────────────────────────────────┘
  Note: only the low 32 bits are meaningful for float; the high 32 bits
  are not used by float operations (float is 32-bit, register is 64-bit).

  The SAME 64 bits. No tag. The instruction decides which interpretation applies.
```

### 2.2.5 The Launch Convention: s0 and s1

Two scalar registers have a defined value at kernel entry, set by the launch convention:

- `s0` holds the **instance ID** — an integer from 0 to `grid - 1` that uniquely identifies
  this kernel instance within its launch. Every kernel uses `s0` to determine which portion
  of the input it should process.
- `s1` holds the **argument block address** — the integer byte address (in NPU memory) of the
  argument block that was passed to `C_KLAUNCH`. The kernel reads its input/output pointers
  and configuration values from this block using `V_SLW` or `V_SLF`.

All other scalar registers (`s2`–`s31`) are initialized to 0 at kernel entry. They are set
by the kernel itself.

A kernel that ignores `s0` will run the same computation on every instance — sometimes
intentional (a kernel that processes the whole tensor in a single instance), but usually a
bug.

A kernel that ignores `s1` has no way to find its input or output — it would have to use
hardwired literal addresses, which is inflexible and prevents kernel reuse.

### 2.2.6 Scalar Instructions in the Vector Core

The instructions that operate on scalar registers within the vector core form a small but
essential sublanguage. They are not the same as control-core instructions — they run on the
vector engine, inside the kernel's instruction stream:

| Instruction | Operation | Interpretation |
|---|---|---|
| `V_SLI rd, imm` | `s[rd].i = imm` | Load integer immediate into scalar reg |
| `V_SFI rd, fimm` | `s[rd].f = fimm` | Load float immediate into scalar reg |
| `V_SADD rd, rs1, rs2` | `s[rd].i = s[rs1].i + s[rs2].i` | Integer add (for address math) |
| `V_SADDI rd, rs1, imm` | `s[rd].i = s[rs1].i + imm` | Integer add immediate |
| `V_SSUB rd, rs1, rs2` | `s[rd].i = s[rs1].i - s[rs2].i` | Integer subtract |
| `V_SMUL rd, rs1, rs2` | `s[rd].i = s[rs1].i * s[rs2].i` | Integer multiply |
| `V_SMULI rd, rs1, imm` | `s[rd].i = s[rs1].i * imm` | Integer multiply immediate |
| `V_SLW rd, addr` | `s[rd].i = mem32[addr]` | Load 32-bit integer from memory |
| `V_SLF rd, addr` | `s[rd].f = mem_float[addr]` | Load 32-bit float from memory |
| `V_SRECIP rd, rs1` | `s[rd].f = 1.0f / s[rs1].f` | Scalar reciprocal (float) |
| `V_KID rd` | `s[rd].i = instance_id` | Read instance ID into scalar reg |

`V_KID` is a convenience that re-reads the instance ID. Since `s0` already holds the instance
ID at entry, `V_KID` is useful if the kernel has clobbered `s0` and needs to retrieve the ID
again.

The scalar instructions are single-cycle in the cost model. They are not vectorized — they
operate on one value at a time, like a scalar CPU instruction. This is intentional: scalar
operations like address computation and loop counting have no benefit from vectorization.

---

## 2.3 Mask Registers: m0–m7

### 2.3.1 The Problem Masks Solve

Neural network algorithms are full of data-dependent conditionals. Consider the ReLU
activation: `y[i] = max(0, x[i])`. This is trivially expressed with `V_VMAX v1, v0, vzero`,
but suppose instead we want: "add 100 to positive elements, set negative elements to zero."
This is:

```
y[i] = (x[i] > 0) ? (x[i] + 100.0) : 0.0
```

On a scalar CPU, you would write a loop with an `if` statement. On the vector core, you
cannot write a branch that executes differently for different lanes — the vector core
executes the same instruction for all lanes simultaneously. What you can do instead is
*predication*: use a mask register to record the condition, and then apply operations only
to the lanes where the mask is 1.

This is the essential function of mask registers: they make data-dependent, per-lane
conditional behavior possible without breaking the SIMD execution model. The mask register
stores the outcome of a comparison across all lanes, and subsequent instructions read the mask
to determine which lanes should participate.

### 2.3.2 Structure

There are 8 mask registers, named `m0` through `m7`. Each mask register holds `vl` bits (one
bit per active lane). In the simulator, each mask register is implemented as a uint8_t array:

```c
uint8_t m[NMREG][VLEN_MAX];  /* m[0..7][0..255] */
```

Each element of this array is either 0 (lane inactive/masked out) or 1 (lane active). The
value 1 means "this lane participates in predicated operations." The value 0 means "this
lane is skipped by predicated operations."

### 2.3.3 Diagram: A Mask Register Alongside a Vector

```
  Input vector v0 (vl = 8):
  ┌──────┬──────┬──────┬──────┬──────┬──────┬──────┬──────┐
  │  3.0 │ -1.0 │  5.0 │  0.0 │ -2.0 │  7.0 │ -4.0 │  1.0 │
  │  L0  │  L1  │  L2  │  L3  │  L4  │  L5  │  L6  │  L7  │
  └──────┴──────┴──────┴──────┴──────┴──────┴──────┴──────┘

  After VCMP m0, v0, vzero, CMP_GT  (m0[l] = 1 if v0[l] > 0):
  ┌──────┬──────┬──────┬──────┬──────┬──────┬──────┬──────┐
  │   1  │   0  │   1  │   0  │   0  │   1  │   0  │   1  │
  │  L0  │  L1  │  L2  │  L3  │  L4  │  L5  │  L6  │  L7  │
  └──────┴──────┴──────┴──────┴──────┴──────┴──────┴──────┘
  m0: "positive lanes"

  Predicated VADD v0, v0, v100  @m0  (adds 100.0 to positive lanes only):
  ┌──────┬──────┬──────┬──────┬──────┬──────┬──────┬──────┐
  │103.0 │ -1.0 │105.0 │  0.0 │ -2.0 │107.0 │ -4.0 │101.0 │
  │  L0  │  L1  │  L2  │  L3  │  L4  │  L5  │  L6  │  L7  │
  └──────┴──────┴──────┴──────┴──────┴──────┴──────┴──────┘
  Lanes 1, 3, 4, 6 (where m0=0) are unchanged.
```

### 2.3.4 Creating Masks: VMASK_SET and VCMP

Masks are created in two ways:

**`V_VMASK_SET md, vs`**: Sets each bit of mask `md` from the corresponding float lane of
vector `vs`. A lane is 1 (active) if `vs[l] != 0.0f`, and 0 (inactive) if `vs[l] == 0.0f`.
This is useful when you already have a vector of flags computed as 0.0 / 1.0 floats.

**`V_VCMP md, vs1, vs2, cmp_kind`**: Compares `vs1[l]` to `vs2[l]` for each lane using one
of six comparison predicates:

| `cmp_kind` | Condition | Example use |
|---|---|---|
| `CMP_LT` | `vs1[l] < vs2[l]` | Is input below threshold? |
| `CMP_LE` | `vs1[l] <= vs2[l]` | Is input at most threshold? |
| `CMP_EQ` | `vs1[l] == vs2[l]` | Exact equality (use with care for floats) |
| `CMP_NE` | `vs1[l] != vs2[l]` | Not equal |
| `CMP_GT` | `vs1[l] > vs2[l]` | Is input above threshold? |
| `CMP_GE` | `vs1[l] >= vs2[l]` | Is input at least threshold? |

The result is written to mask register `md`, with `md[l] = 1` where the condition is true.

### 2.3.5 Combining Masks: VMASK_AND, VMASK_OR, VMASK_NOT

Complex conditions are built by combining simpler masks:

```asm
; Compute: mask where (x > 0.0) AND (x < 10.0)
VCMP m0, v0, vzero,   CMP_GT    ; m0[l] = x[l] > 0
VCMP m1, v0, vten,    CMP_LT    ; m1[l] = x[l] < 10
VMASK_AND m2, m0, m1             ; m2[l] = m0[l] AND m1[l] = x in (0, 10)

; Compute: mask where NOT (x == 0.0)
VCMP m0, v0, vzero, CMP_EQ      ; m0[l] = x[l] == 0.0
VMASK_NOT m1, m0                 ; m1[l] = NOT m0[l] = x[l] != 0.0
```

This compositionality is what makes predication expressive enough for real algorithms, without
adding a complex condition evaluator to the hardware. The mask logic is simple bitwise
operations; the complexity comes from composing masks in the program.

### 2.3.6 Applying Masks: The @mask Prefix

Once a mask register is set, any arithmetic instruction can be "predicated" by specifying a
mask. In the instruction struct, this is encoded as `use_mask = 1` and `mask = <reg_index>`.
In assembly syntax, it is written as `@m<n>` after the mnemonic or as an extra operand.

When a predicated instruction executes:
- For each lane `l` where `m[mask][l] == 1`: the operation is performed normally, writing to
  `vd[l]`.
- For each lane `l` where `m[mask][l] == 0`: the destination lane is *not written*. It
  retains its previous value.

This is called *write-masking* or *merge predication*: masked-out lanes keep their old values,
they are not zeroed out.

The mask predicate applies to exactly the destination register. It does not affect reads from
source registers — a predicated `VADD v2, v0, v1 @m0` still reads all lanes of `v0` and `v1`
for the purpose of computing the addition. It just does not write the result back for
masked-out lanes of `v2`.

### 2.3.7 VSEL: Branchless Conditional Selection

`V_VSEL vd, vs1, vs2, mask` is a branchless select: for each lane, `vd[l] = mask[l] ?
vs1[l] : vs2[l]`. Unlike predicated arithmetic (which leaves non-selected lanes unchanged),
`VSEL` always writes every lane — either from `vs1` or from `vs2`.

```
; Implement ReLU: y = max(0, x) using VSEL
VCMP m0, v0, vzero, CMP_GT   ; m0[l] = (x[l] > 0)
VSEL v1, v0, vzero, m0       ; v1[l] = m0[l] ? x[l] : 0.0
; v1 now holds ReLU(v0)
```

`VSEL` is often more appropriate than predicated arithmetic when you need the false branch to
be a specific value (rather than "keep the old value of vd").

### 2.3.8 Reset at Kernel Launch

All mask registers are initialized to all-ones at kernel entry. This means every lane is
active by default. A kernel that never uses predication does not need to think about masks —
every instruction operates on all `vl` lanes.

The simulator implements this as:

```c
memset(n->m, 0, sizeof n->m);   /* first zero everything ... */
/* but after this, instructions set vl first, and masks start all-zeros */
```

Wait — the simulator zeroes the mask memory to 0, but conceptually the mask starts as
"all active." The resolution is that the *unset* state is all zeros, but the mask is not
consulted by default (`use_mask = 0` means no masking). The default "no masking" state means
all lanes participate. Only when a specific mask register is explicitly referenced does its
content matter — and at that point, the kernel has already written that mask via `VCMP` or
`VMASK_SET`. So the initial zero state of the mask registers does not affect programs that
do not use predication, and programs that use predication always set their masks before
using them.

For programs that *do* read a mask register before setting it (which would be a bug), the
zero initial state means all lanes are masked out — a fail-safe rather than a garbage value.

---

## 2.4 The Vector Length Register: vl

### 2.4.1 What vl Is

`vl` is a single integer, the *vector length register*. It holds the current number of active
lanes in every vector register. Every vector instruction — load, store, arithmetic, reduction,
masking — only touches lanes 0 through `vl - 1`. Lanes `vl` through `VLEN_MAX - 1` are
present but dormant.

In the simulator state:

```c
int vl;  /* current vector length, 0 <= vl <= VLEN_MAX = 256 */
```

`vl` starts at 0 at kernel entry. Every kernel must set `vl` before using any vector
instruction, using `V_VSETVL`.

### 2.4.2 Setting vl: VSETVL

`V_VSETVL rd, rs1` sets `vl` to the value in scalar register `rs1` (or to an immediate if
`rs1 == 0` and `use_base == 0`), clamped to the range `[0, VLEN_MAX]`. The resulting `vl`
is also written back to scalar register `rd` for use in subsequent calculations.

```asm
SLW   s4, 8(s1)      ; s4 = N (element count from argument block)
VSETVL s8, s4        ; vl = min(N, 256); s8 = vl (may differ from N if N > 256)
; Now vl = min(N, 256). All subsequent vector instructions process that many elements.
```

The reason `VSETVL` writes back the actual `vl` to a scalar register is that when `N >
VLEN_MAX`, the kernel may need to know the actual `vl` it got (which is `VLEN_MAX = 256`) in
order to compute the number of remaining elements for subsequent iterations. This pattern is
the standard way to handle arrays longer than `VLEN_MAX` using a loop.

### 2.4.3 Why Variable Vector Length Exists

The vector length mechanism solves a fundamental problem in SIMD computing: what do you do when
the problem size is not a multiple of the hardware vector width?

Suppose you have an array of 70 floats and `VLEN_MAX = 256`. You do not want to process 256
elements when only 70 are valid — the extra 186 lanes would read garbage data and write
garbage results. You do not want to pad the array with zeros either (it wastes memory and
complicates the caller). And you do not want to write 4 separate code paths for arrays of size
0–64, 65–128, 129–192, 193–256.

With `VSETVL`, you set `vl = 70` and run your kernel normally. The hardware processes exactly
70 lanes. For an array of 10 floats, you set `vl = 10`. For 256, `vl = 256`. The same kernel
code handles all cases.

This design is directly inspired by the RISC-V Vector (RVV) extension's `vsetvl` instruction
and length-agnostic vector model. The philosophy is: *write the kernel once, in terms of
"process vl elements," and let the hardware adapt to the problem size.* This is sometimes
called the *stripmining* or *vector length agnostic* (VLA) approach.

### 2.4.4 Handling Arrays Longer Than VLEN_MAX

For arrays longer than 256 elements, you need a loop that advances through the array in
chunks of `vl`. Each iteration of the loop processes `vl` elements and advances the address
pointer by `vl * 4` bytes (since each float32 is 4 bytes). The loop continues until all
elements are processed.

Here is the pattern for processing an array of `N` elements where `N` may exceed 256:

```asm
; s2 = array_base_addr, s4 = N (total element count)
; Initialize remaining = N
SADDI s5, s4, 0       ; s5 = remaining = N

loop_top:
; Set vl = min(remaining, VLEN_MAX)
VSETVL s6, s5         ; s6 = actual vl (may be clamped to 256)

; Load vl elements, process, store
VLOAD  v0, 0(s2)      ; load vl elements from s2
VEXP   v0, v0         ; apply per-element operation
VSTORE v0, 0(s2)      ; store vl results

; Advance pointer and remaining count
SMULI  s7, s6, 4      ; s7 = vl * 4 (bytes processed this iteration)
SADD   s2, s2, s7     ; advance pointer
SSUB   s5, s5, s6     ; remaining -= vl

; Loop while remaining > 0
BRNZ   s5, loop_top   ; branch to loop_top if remaining != 0
KEXIT
```

This pattern works correctly whether `N = 7`, `N = 256`, or `N = 100000`. When `N` is a
multiple of 256, every iteration processes exactly 256 elements. When `N` is not a multiple,
the final iteration processes fewer elements, and `VSETVL` automatically adjusts `vl` to the
remainder.

The cycle cost of this loop is proportional to `N / HW_LANES` total passes, plus the loop
overhead. For large `N`, the vector computation dominates and the loop overhead is negligible.

### 2.4.5 A Diagram: Stripmining a 320-Element Array

```
Array of 320 floats, VLEN_MAX = 256:

Iteration 1: vl = 256
  ┌──────────────────────────────────────────────────────────────────────────────┐
  │ Elements 0 ... 255                                                           │
  │ ████████████████████████████████████████████████████████████████████████████│
  └──────────────────────────────────────────────────────────────────────────────┘
  → Process 256 elements, advance pointer by 1024 bytes, remaining = 64

Iteration 2: vl = 64
  ┌──────────────────────┐
  │ Elements 256 ... 319 │
  │ ████████████████████ │
  └──────────────────────┘
  → Process 64 elements, remaining = 0, exit loop

Total: 2 iterations, correct results for all 320 elements.
No padding, no special-case code for the tail.
```

### 2.4.6 vl and Masks Interact

When masks are applied, the combination of `vl` and the mask determines which lanes
participate. A mask selects lanes within the active range `[0, vl)`. Lanes outside `[0, vl)`
are never touched, regardless of what the mask says.

This means: if `vl = 8` and mask `m0` has a 1 at position 200, that bit is irrelevant because
lane 200 is not in the active range. The active lanes are 0 through 7 only.

---

## 2.5 State Reset at Kernel Launch

### 2.5.1 The Complete Reset Sequence

When a vector kernel instance is dispatched by `C_KLAUNCH`, the simulator performs a complete
state reset before executing the first instruction:

```c
memset(n->v, 0, sizeof n->v);   /* all vector regs → 0.0f in every lane */
memset(n->m, 0, sizeof n->m);   /* all mask bits → 0 (but use_mask=0 by default) */
memset(n->s, 0, sizeof n->s);   /* all scalar regs → 0 (integer and float both 0) */
n->vl = 0;                       /* vector length → 0 */
/* launch convention: set after reset */
n->s[0].i = inst_id;             /* s0 = this instance's ID */
n->s[1].i = args_addr;           /* s1 = argument block address */
```

This reset happens at the beginning of every kernel instance, even if multiple instances are
of the same kernel. Each instance starts from the same clean state, differentiated only by its
`s0` value.

### 2.5.2 Why Fresh State Per Instance Matters

If the vector core did not reset state between instances, instance 1 would see the vector
register values left by instance 0. This would make kernel correctness depend on execution
order — a fragile and hard-to-debug property. With fresh state, every instance is an
independent computation: instance 5 of a softmax kernel processes row 5 and produces row 5's
output, regardless of what happened in instances 0–4.

This independence is what makes the SIMT (Single Instruction, Multiple Threads) programming
model work. Each instance is logically a separate thread with its own private register file.
The shared state is memory (DRAM and SRAM) — all instances share the same memory address
space and can read each other's outputs (though during execution, instances are designed to
write to non-overlapping memory regions, avoiding races).

### 2.5.3 Shared Memory: The Scratchpad

While registers are private to each instance, the SRAM scratchpad is shared. Multiple
instances launched by the same `C_KLAUNCH` have access to the same scratchpad addresses. This
enables communication patterns where one instance pre-computes shared data into the scratchpad
and other instances read from it — though this requires careful synchronization to avoid
races (the simulator does not model intra-launch synchronization primitives; instances run
sequentially in the simulator regardless of the core count).

### 2.5.4 Diagram: Two Instances with Independent Register Files

```
C_KLAUNCH kernel_id=0, grid=2, args=0x1000

Instance 0 (s0=0, s1=0x1000):             Instance 1 (s0=1, s1=0x1000):
  v0: [0, 0, 0, 0, 0, 0, 0, 0, ...]         v0: [0, 0, 0, 0, 0, 0, 0, 0, ...]
  v1: [0, 0, 0, 0, 0, 0, 0, 0, ...]         v1: [0, 0, 0, 0, 0, 0, 0, 0, ...]
  ... (all 32 vector regs zeroed)            ... (all 32 vector regs zeroed)
  s0=0  s1=0x1000  s2=0  ...                 s0=1  s1=0x1000  s2=0  ...
  m0=[0,0,...] ... m7=[0,0,...]              m0=[0,0,...] ... m7=[0,0,...]
  vl=0                                        vl=0

  After execution:                           After execution:
  v0: [1.0, 2.0, 3.0, 4.0, ...]             v0: [5.0, 6.0, 7.0, 8.0, ...]
  (row 0 loaded and processed)               (row 1 loaded and processed)

  Writes output to:  0x4000 + 0*stride       Writes output to: 0x4000 + 1*stride
         ▲                                          ▲
         │                                          │
         └──────────── Shared DRAM ─────────────────┘
                     0x4000 (output)
```

Both instances share the argument block at `0x1000` (read-only) and write to non-overlapping
regions of output memory (the instance ID `s0` determines which region each instance writes to).

---

## 2.6 Register File Summary

The following table summarizes all four register spaces of the vector core:

| Register File | Names | Count | Width per Register | Initialized At Launch | Set at Launch |
|---|---|---|---|---|---|
| Vector | `v0`–`v31` | 32 | `vl × float32` (up to 256 × 4 = 1024 B) | `0.0f` in all lanes | — |
| Scalar | `s0`–`s31` | 32 | 64-bit union (`float f` or `int64_t i`) | `0` (both `.f` and `.i`) | `s0 = instance_id`, `s1 = args_addr` |
| Mask | `m0`–`m7` | 8 | `vl` bits (up to 256 bits) | `0` (but not consulted unless `use_mask=1`) | — |
| Length | `vl` | 1 | Integer, range `[0, 256]` | `0` | — |

### 2.6.1 Register File Addressing

All registers are addressed by index in every instruction. The instruction encoding (the
`vinst` struct) holds separate integer fields for each register operand:

```c
typedef struct {
    vop_t op;
    int   vd, vs1, vs2, vs;   /* vector reg indices (0..31) */
    int   rd, rs, rs1, rs2;   /* scalar reg indices (0..31) */
    int   md, m1, m2;         /* mask reg indices (0..7) */
    int   use_mask, mask;     /* predicate mask for @m prefix */
    /* ... other fields ... */
} vinst;
```

There is no implicit accumulator, no special "result register," and no instruction that
implicitly reads a fixed register. Every source and destination is explicitly named in the
instruction. This makes the instruction set orthogonal and easy for compilers to handle.

### 2.6.2 Register Naming Conventions in Kernel Code

Although the register file has no hardware-enforced naming conventions beyond the launch
convention (s0=id, s1=args), programmers and compilers typically follow a soft convention
for clarity:

| Register Range | Typical Use |
|---|---|
| `s0` | Instance ID (set by launch convention, often not modified) |
| `s1` | Argument block base pointer (set by launch convention) |
| `s2`–`s7` | Addresses loaded from argument block (input, output, size) |
| `s8`–`s11` | Computed addresses (row offsets, strided pointers) |
| `s12`–`s15` | Loop counters, reduction results |
| `s16`–`s31` | Temporaries, additional scalars as needed |
| `v0`–`v3` | Primary computation registers (input, output, temporaries) |
| `v4`–`v7` | Broadcast scalars (zero, one, constants) |
| `v8`–`v15` | Additional working registers for complex kernels |
| `v16`–`v31` | Spill area (rarely used; prefer keeping values in registers) |
| `m0`–`m3` | Active computation masks |
| `m4`–`m7` | Temporary or combined masks |

This is a convention, not a rule. The ISA does not enforce it. A compiler would use a
register allocator to assign registers optimally, and a hand-written kernel may use whatever
layout makes the code clearest.

---

## 2.7 A Complete Worked Example: Elementwise Vector Add

To synthesize everything in this chapter, let us trace through the register file state at each
step of the simplest possible kernel: elementwise vector addition, `c[i] = a[i] + b[i]`.

**Setup:** Argument block at address `0x1000`:
```
offset 0:  0x2000  (address of array a, 8 floats)
offset 4:  0x3000  (address of array b, 8 floats)
offset 8:  0x4000  (address of output array c)
offset 12: 8       (N, element count as int32)
```
Array `a` contains `[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]` at `0x2000`.
Array `b` contains `[10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0]` at `0x3000`.

**Kernel entry state:**
```
s0 = 0 (instance 0, single-instance launch)
s1 = 0x1000 (args address)
s2..s31 = 0
v0..v31 = [0, 0, 0, 0, 0, 0, 0, 0, ... (256 zeros each)]
m0..m7 = [0, 0, 0, 0, ...] (but use_mask=0 by default)
vl = 0
```

**Instruction 1: `V_SLW s2, 0(s1)`** — load word from `s1 + 0 = 0x1000`
```
Memory[0x1000] = 0x00002000 = 8192 (decimal)
s2.i = 8192 = 0x2000
Register file: s2 = 0x2000 (= address of array a)
```

**Instruction 2: `V_SLW s3, 4(s1)`** — load word from `s1 + 4 = 0x1004`
```
Memory[0x1004] = 0x00003000 = 12288
s3.i = 0x3000 (= address of array b)
```

**Instruction 3: `V_SLW s4, 8(s1)`** — load word from `s1 + 8 = 0x1008`
```
Memory[0x1008] = 0x00004000 = 16384
s4.i = 0x4000 (= address of output array c)
```

**Instruction 4: `V_SLW s5, 12(s1)`** — load word from `s1 + 12 = 0x100C`
```
Memory[0x100C] = 8
s5.i = 8 (= N, element count)
```

**Instruction 5: `V_VSETVL s6, s5`** — set vl = min(s5.i, VLEN_MAX) = min(8, 256) = 8
```
vl = 8
s6.i = 8 (actual vl achieved)
```

**Instruction 6: `V_VLOAD v0, 0(s2)`** — load 8 floats from address s2 = 0x2000
```
Reads Memory[0x2000..0x201F] (8 floats × 4 bytes = 32 bytes)
v0 = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
     (lanes 8..255 remain 0.0, untouched)
```

**Instruction 7: `V_VLOAD v1, 0(s3)`** — load 8 floats from address s3 = 0x3000
```
v1 = [10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0]
```

**Instruction 8: `V_VADD v2, v0, v1`** — elementwise add
```
For each lane l in 0..7:
  v2[l] = v0[l] + v1[l]

v2 = [10.0, 12.0, 14.0, 16.0, 18.0, 20.0, 22.0, 24.0]
(no mask, all 8 lanes active)
```

**Instruction 9: `V_VSTORE v2, 0(s4)`** — store 8 floats to address s4 = 0x4000
```
Writes Memory[0x4000..0x401F]:
  [10.0, 12.0, 14.0, 16.0, 18.0, 20.0, 22.0, 24.0]
```

**Instruction 10: `V_KEXIT`** — signal completion
```
Instance terminates. Control returns to C_KWAIT in the control core.
```

The output array at `0x4000` now holds the correct elementwise sum. The entire computation
touched: 4 scalar registers (`s2`, `s3`, `s4`, `s5`), 1 scalar for the vl result (`s6`), 3
vector registers (`v0`, `v1`, `v2`), no mask registers, and `vl`. The other 27 vector
registers and 26 scalar registers were never touched.

This trace illustrates the essential rhythm of vector core programming:
1. Read pointers and configuration from the argument block into scalar registers (using `V_SLW`).
2. Set `vl` using `V_VSETVL`.
3. Load data into vector registers (using `V_VLOAD`).
4. Compute using vector arithmetic instructions.
5. Store results (using `V_VSTORE`).
6. Exit with `V_KEXIT`.

Everything else — predication, reductions, transcendentals, address computation for multi-row
kernels — is an elaboration of this basic pattern.

---

## 2.8 Chapter Summary

The vector core's register file comprises four distinct namespaces:

**Vector registers `v0`–`v31`**: The primary data registers, each holding up to 256 float32
lanes. All are reset to 0.0f at kernel launch. The number of active lanes is determined by
`vl`. Computation operates over all active lanes simultaneously.

**Scalar registers `s0`–`s31`**: 64-bit union registers that hold either an integer (`int64_t`)
or a float (`float`), depending on how they are written and read. Reset to 0 at kernel launch,
except `s0` (instance ID) and `s1` (argument block address) which are set by the launch
convention. Used for addresses, loop counters, and scalar intermediate values.

**Mask registers `m0`–`m7`**: Per-lane bit arrays (one bit per active lane). Used with the
`@mask` predicate to selectively disable lanes in arithmetic instructions. Created by `VCMP`
and `VMASK_SET`. Combined with `VMASK_AND`, `VMASK_OR`, `VMASK_NOT`. Reset to zero at launch
(but no-mask is the default; a register is not consulted unless explicitly referenced).

**Vector length `vl`**: A single integer controlling how many lanes are active (0 to 256).
Must be set by `VSETVL` before any vector instruction. The length-agnostic model allows the
same kernel to handle any problem size, from 1 element to millions (via stripmining loops),
without code changes.

Every grid instance begins with a completely fresh copy of all four register spaces. The only
pre-set values are `s0` (instance ID) and `s1` (argument block address). All other registers
start at zero. Memory (DRAM and SRAM scratchpad) is shared across all instances of a launch.

With the register file fully understood, Chapter 3 turns to the memory instructions that
bridge the register file and NPU memory: how kernels load data into registers and store
results back out.
