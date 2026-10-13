# Chapter 3: The NPU Memory Model

> **Series context.** This chapter is part of a multi-chapter reference for the
> fully programmable NPU vector core ISA. Chapter 1 introduced the three-layer
> architecture (control core, tensor engine, vector core). Chapter 2 covered the
> register file. This chapter explains **where data lives**, how the hardware
> routes each memory access, and why the programming discipline of staging data
> into on-chip SRAM before running a kernel is the single most important
> performance optimization a vector-core programmer can apply.

---

## 3.1 One 32-Bit Address Space, Two Physical Memories

The NPU presents a flat 32-bit address space to software, but behind that
unified view lies a fundamental architectural split: two physically distinct
memory arrays with radically different latencies, capacities, and cost models.
Understanding how to exploit that split is the central skill of NPU programming.

### 3.1.1 The High-Bit Encoding

The NPU uses a single bit — bit 31 of a 32-bit address — to route every memory
access to the correct physical array:

```
Address bit 31 = 0  →  DRAM   (off-chip, slow, large)
Address bit 31 = 1  →  SRAM   (on-chip, fast, small)
```

The relevant constants from `npu.h`:

```c
#define NPU_MEM_SIZE     (16u * 1024u * 1024u)  /* DRAM: 16 MB             */
#define NPU_SCRATCH_SIZE (256u * 1024u)          /* SRAM: 256 KB            */
#define SCRATCH_BASE     0x80000000u             /* base of SRAM region     */
#define SCRATCH_MASK     0x7fffffffu             /* strips high bit → offset */
```

To access SRAM at byte offset `off`, you write `SCRATCH_BASE | off`. This
encoding is cheap to decode in hardware: a single AND gate tests bit 31. The
low 31 bits are the offset within whichever memory.

### 3.1.2 Address Space Map

```
  32-bit virtual address space
  ═══════════════════════════════════════════════════════════════════════

  0x00000000 ┌─────────────────────────────────────────────────────────┐
             │                                                         │
             │                      D R A M                           │
             │               (off-chip, main memory)                  │
             │                                                         │
             │   • 16 MB total                                         │
             │   • Bit 31 = 0 (all addresses in this range)            │
             │   • Backed by external DDR / LPDDR                      │
             │   • Latency: 100 cycles per vector access               │
             │   • DMA bandwidth: 32 bytes / cycle                     │
             │                                                         │
  0x00FFFFFF └─────────────────────────────────────────────────────────┘
             │                                                         │
             │            (unmapped gap — abort if accessed)           │
             │                                                         │
  0x7FFFFFFF │                                                         │
  ═══════════╪════════════════════ bit 31 boundary ═══════════════════╪═
  0x80000000 ┌─────────────────────────────────────────────────────────┐
             │                                                         │
             │                    S R A M  Scratchpad                  │
             │               (on-chip, programmer-managed)             │
             │                                                         │
             │   • 256 KB total  (offsets 0x00000–0x3FFFF)             │
             │   • Bit 31 = 1, low bits = offset within scratchpad     │
             │   • Physical SRAM array inside the NPU die              │
             │   • Latency: 1 cycle per vector access                  │
             │   • No cache coherency issues — explicit DMA            │
             │                                                         │
  0x8003FFFF └─────────────────────────────────────────────────────────┘
             │                                                         │
             │   (addresses 0x80040000–0xFFFFFFFF: SRAM overrun →      │
             │    simulator aborts with bounds-check error)            │
             │                                                         │
  0xFFFFFFFF │                                                         │
```

### 3.1.3 The Hardware `resolve()` Path

In the simulator, every load and store goes through a routing function that
implements exactly this bit-test:

```c
static int is_scratch(uint32_t addr) {
    return (addr & SCRATCH_BASE) != 0;
}

static uint8_t *resolve(npu_t *n, uint32_t addr, size_t bytes) {
    if (is_scratch(addr)) {
        uint32_t off = addr & SCRATCH_MASK;   /* strip high bit */
        if ((size_t)off + bytes > n->scratch_size) {
            fprintf(stderr, "NPU: out-of-bounds SRAM access ...\n");
            abort();
        }
        return n->scratch + off;
    }
    /* else: DRAM */
    if ((size_t)addr + bytes > n->mem_size) {
        fprintf(stderr, "NPU: out-of-bounds DRAM access ...\n");
        abort();
    }
    return n->mem + addr;
}
```

This is the ground truth for how accesses are routed. Every `VLOAD`, `VSTORE`,
`SLW`, `SLF`, `VGATHER`, and `VSCATTER` instruction passes through `resolve()`.
The bounds check means an overrunning SRAM access will abort immediately rather
than silently clobbering adjacent data — a property that catches a class of
off-by-one errors at development time.

---

## 3.2 Latency Constants and the Cycle Model

The NPU provides rough (non-cycle-accurate) cost estimates through three
constants that govern the simulator's performance model:

```c
#define NPU_DRAM_LATENCY  100   /* cycles per off-chip vector access        */
#define NPU_SRAM_LATENCY    1   /* cycles per on-chip vector access         */
#define NPU_DMA_BW         32   /* DMA bandwidth, bytes per cycle           */
```

These are the architectural latency parameters; the simulator adds them to
`stats.vector_cycles` on every memory instruction. Real hardware latencies will
differ, but the ratio — DRAM is roughly two orders of magnitude more expensive
per access than SRAM — is representative of the real world.

### 3.2.1 The Latency Routing Function

```c
static uint64_t mem_latency(uint32_t addr) {
    return is_scratch(addr) ? NPU_SRAM_LATENCY : NPU_DRAM_LATENCY;
}
```

For a `VLOAD` or `VSTORE`, the simulator calls `mem_latency(addr)` and
accumulates the result into `vector_cycles`. This means the address space
choice shows up directly in your performance counter output from
`npu_print_stats()`.

### 3.2.2 Quantifying the Gap

Consider a vector-add kernel operating on N=256 floats (four 64-lane `VLOAD`s
and two `VSTORE`s):

| Memory location | VLOAD cost/access | Total for 4 loads + 2 stores |
|-----------------|-------------------|------------------------------|
| DRAM (0x002000) | 100 cycles        | 600 cycles                   |
| SRAM (0x802000) |   1 cycle         |   6 cycles                   |
| **Ratio**       | **100×**          | **100×**                     |

For a kernel that is memory-bound — and almost all ML inference kernels are, at
some level — this difference is the difference between an NPU that is and is not
meeting its performance target. The programming discipline of staging data into
SRAM before launching a vector kernel therefore has a direct, measurable, and
large payoff.

Demo 6 in `examples.c` shows this comparison experimentally: the same
vector-add kernel over 256 floats, once with operands in DRAM and once staged
into SRAM:

```
vector-core cycles:  DRAM-resident = 606,  SRAM-staged = 6
DMA cost (tensor engine) for staging = 16 cycles
```

The DMA cost is shared across the whole buffer; once you have staged a large
tensor that multiple kernel instances will read, the amortized cost per instance
is negligible.

---

## 3.3 The SRAM Scratchpad — Programmer-Managed, Not a Cache

The SRAM scratchpad is **not a hardware cache**. It is a directly addressable
SRAM bank whose contents are completely under software control. This has
important consequences:

**No cache miss surprises.** A traditional cache introduces unpredictable
latency spikes when a needed line is not resident. The scratchpad has exactly
one latency: 1 cycle. Performance becomes fully predictable at the instruction
level, which is essential for meeting hard real-time latency budgets (common in
edge inference).

**No eviction.** Data placed in the scratchpad stays there until you overwrite
it or DMA it out. Two successive kernel launches can share a weight tile that
was DMA'd in before either launch. In contrast, a cache might evict the tile
between launches if another buffer conflicts in the cache set.

**No coherency hardware.** Because data movement is explicit (DMA instructions
issued by the control core), there is no need for a cache-coherency protocol.
This saves substantial die area and power — a luxury a small NPU accelerator
can afford that a general-purpose CPU cannot.

**Limited capacity.** 256 KB sounds modest. The programmer must tile large
tensors so that the active working set fits. This is the dominant constraint
that shapes kernel design for ML workloads.

### 3.3.1 Scratchpad Layout Strategy

A typical ML kernel has three logical regions in the scratchpad:

```
  SRAM (256 KB total)
  ┌─────────────────────────────────────────────────────┐  0x80000000
  │                                                     │
  │   Input tile(s)   (A)                               │  e.g. 64 KB
  │                                                     │
  ├─────────────────────────────────────────────────────┤  0x80010000
  │                                                     │
  │   Weight tile(s)  (B)                               │  e.g. 64 KB
  │                                                     │
  ├─────────────────────────────────────────────────────┤  0x80020000
  │                                                     │
  │   Output tile     (C)                               │  e.g. 64 KB
  │                                                     │
  ├─────────────────────────────────────────────────────┤  0x80030000
  │                                                     │
  │   Kernel-private temporaries                        │  remaining ~64 KB
  │   (softmax intermediate, norm stats, etc.)          │
  │                                                     │
  └─────────────────────────────────────────────────────┘  0x8003FFFF
```

The addresses passed as arguments to a vector kernel (`s1` = args base) point
into this layout. The kernel reads from input/weight tiles and writes to the
output tile, all at 1-cycle latency.

---

## 3.4 Address Formation in the Vector Core

Vector memory instructions compute the effective address in one of two modes,
controlled by the `use_base` field of the `vinst` struct:

```c
typedef struct {
    vop_t   op;
    ...
    int     use_base;   /* 1 => addr = s[rs] + imm ; 0 => imm directly */
    int64_t imm;        /* immediate / offset */
    int     rs;         /* scalar register holding base address */
    ...
} vinst;
```

And in the simulator's dispatch loop:

```c
uint32_t addr = I->use_base ? (uint32_t)(n->s[I->rs].i + I->imm)
                            : (uint32_t)I->imm;
```

### 3.4.1 Mode 0: Absolute Immediate

When `use_base=0`, the address is the immediate field alone. This is useful when
the target address is compile-time constant — e.g., a fixed scratchpad region
known at kernel-compile time. It also works for accessing well-known control
structures at fixed DRAM addresses.

```c
/* Load from a fixed scratchpad address, no base register needed */
{ .op=V_VLOAD, .vd=0, .use_base=0, .imm=0x80000000 }
```

### 3.4.2 Mode 1: Base-Plus-Offset (the Common Case)

When `use_base=1`, the address is `s[rs] + imm`. This is the workhorse mode for
all real kernels. The scalar register holds a dynamically computed base address
(e.g., the start of the data for this grid instance), and the immediate provides
a field offset within that block.

```c
/* Load from s2 + 0 (first field), s2 + 64 (second field), etc. */
{ .op=V_VLOAD, .vd=0, .use_base=1, .rs=2, .imm=0   }
{ .op=V_VLOAD, .vd=1, .use_base=1, .rs=2, .imm=256 }
```

### 3.4.3 The SIMT Addressing Pattern

The vector core is a SIMT (Single Instruction, Multiple Threads) machine. Every
kernel instance receives the same instruction stream but is launched with a
different `instance id` in scalar register `s0`. The fundamental addressing
pattern uses `s0` to compute a unique base address for each instance:

```
                            Instance ID in s0
                                    │
                                    ▼
           ┌──────────────────────────────────────────┐
           │   SMULI s2, s0, STRIDE                   │  s2 = id * stride
           │   SADD  s3, s1, s2                       │  s3 = args_base + offset
           └──────────────────────────────────────────┘
                                    │
                                    ▼
                         Per-instance data address
                                    │
           ┌──────────────────────────────────────────┐
           │   VLOAD v0, s3+0      ; first 64 floats  │
           │   VLOAD v1, s3+256    ; next 64 floats   │
           │   ...                                    │
           └──────────────────────────────────────────┘
```

The launch convention guarantees:

- `s0` = grid instance id (0, 1, 2, ..., grid−1)
- `s1` = args base pointer (address of the argument block in NPU memory)

From these two registers, every instance independently computes its working
addresses. No kernel-level communication between instances is needed; the
control core has already set up the shared SRAM layout before issuing
`C_KLAUNCH`.

### 3.4.4 Extended Example: Four Instances, Independent 64-Float Chunks

Suppose a tensor has 256 floats split across four vector-core instances, each
responsible for one 64-float chunk. The kernel body:

```
; s0 = instance id (0, 1, 2, or 3)
; s1 = args_base (points to arg block in SRAM)
;
; Arg block layout:
;   args[0] = base address of input tensor (in SRAM)
;   args[4] = base address of output tensor (in SRAM)

SLW   s2, 0(s1)       ; s2 = input base addr  (from args[0])
SLW   s3, 4(s1)       ; s3 = output base addr (from args[4])
SLI   s4, 64          ; s4 = elements per instance
SMULI s5, s0, 256     ; s5 = id * 64 * sizeof(float) = id * 256 bytes
SADD  s6, s2, s5      ; s6 = my input base
SADD  s7, s3, s5      ; s7 = my output base
VSETVL s8, s4         ; VL = 64
VLOAD v0, 0(s6)       ; load my 64 floats
... compute ...
VSTORE v0, 0(s7)      ; store my 64 results
KEXIT
```

The four instances execute the same sequence but with `s0` equal to 0, 1, 2, 3
respectively, producing four independent base addresses `s6` that span
non-overlapping chunks of the tensor:

```
Instance 0:  s6 = base + 0
Instance 1:  s6 = base + 256
Instance 2:  s6 = base + 512
Instance 3:  s6 = base + 768
```

This is the fundamental SIMT data-parallel pattern. The programmer expresses it
once; the simulator (and future hardware) runs it across the grid.

---

## 3.5 Access Granularity and Alignment

### 3.5.1 Vector Access Width

A `VLOAD` or `VSTORE` accesses exactly `vl × 4` bytes, where `vl` is the
current vector length set by `VSETVL`. Each lane holds one `float` (4 bytes,
IEEE 754 single precision). There is no sub-float granularity for vector
accesses: you load `vl` consecutive floats, or you load nothing.

```
VLOAD v0, addr    (vl = 8)

  memory: addr+0  addr+4  addr+8  addr+12 addr+16 addr+20 addr+24 addr+28
            [f0]    [f1]    [f2]    [f3]    [f4]    [f5]    [f6]    [f7]
             |       |       |       |       |       |       |       |
  v0:      [ f0 ][ f1 ][ f2 ][ f3 ][ f4 ][ f5 ][ f6 ][ f7 ]

  Memory span: vl * sizeof(float) = 8 * 4 = 32 bytes
```

### 3.5.2 Alignment Expectations

All vector loads and stores should be naturally aligned: an address holding `vl`
floats should be 4-byte aligned (float alignment). The simulator does not
enforce an alignment trap, but unaligned accesses would be split into multiple
transactions on real hardware, incurring extra cycles and potentially
correctness issues if spanning a cache-line boundary in the DRAM controller.

For SRAM accesses, alignment is less critical from a latency standpoint (SRAM
is single-cycle regardless), but it is still good practice and avoids surprises
if the design ever adds an alignment check.

**Rule of thumb:** When allocating scratchpad regions, align each region to at
least 64 bytes (16 floats, one cacheline-equivalent). When allocating large
buffers, align to 256 bytes (64 floats, one full max-vector-length worth).

### 3.5.3 Contiguous vs. Non-Contiguous Access

`VLOAD` and `VSTORE` always access `vl` consecutive floats. For
**non-contiguous** access patterns — e.g., loading every Nth element, loading
from a hash table keyed by lane, or doing an embedding lookup — use `VGATHER`
and `VSCATTER`, which allow each lane to have an independent address. See
Chapter 4 for the full treatment of scatter/gather.

---

## 3.6 The DMA Engine and the Staging Discipline

The tensor engine provides two DMA instructions that are the foundation of the
staging pattern:

| Instruction | Direction         | Issued by      |
|-------------|-------------------|----------------|
| `T_DMA_IN`  | DRAM → SRAM       | Control core   |
| `T_DMA_OUT` | SRAM → DRAM       | Control core   |

The DMA engine operates at 32 bytes/cycle (`NPU_DMA_BW`). Transferring N bytes
costs approximately `N/32 + 1` cycles, charged to `tensor_cycles` in the
simulator.

### 3.6.1 The Five-Step Pattern

Every ML kernel should follow this pattern:

```
Step 1  [Control core]   DMA.IN activations:   DRAM → SRAM
Step 2  [Control core]   DMA.IN weights:       DRAM → SRAM
Step 3  [Control core]   KLAUNCH vector kernel (args point into SRAM)
Step 4  [Vector core]    VLOAD from SRAM (1 cyc)  →  compute  →  VSTORE to SRAM (1 cyc)
Step 5  [Control core]   DMA.OUT results:      SRAM → DRAM
```

As a data-flow diagram:

```
      ┌──────────────────────────────────────────────────────────────┐
      │  Host / Control Core                                         │
      │                                                              │
      │    npu_write_f32(n, 0x002000, input, N)   ← host writes DRAM│
      │    npu_write_f32(n, 0x003000, weights, M)                    │
      │                                                              │
      │    DMA.IN  sA ← DRAM[0x002000], N floats ──────────────────┐│
      │    DMA.IN  sB ← DRAM[0x003000], M floats ──────────────────┤│
      │                                                             ││
      └─────────────────────────────────────────────────────────────┤│
                                                                    ││
      DRAM ───[DMA.IN]──────────────────────────────────────────────▼▼
                                                         ┌──────────────────┐
                                                         │  SRAM Scratchpad │
                                                         │                  │
                                                         │  sA: input tile  │◄─ VLOAD (1 cyc)
                                                         │  sB: weight tile │◄─ VLOAD (1 cyc)
                                                         │  sC: output tile │──► VSTORE (1 cyc)
                                                         │                  │
                                                         └──────────────────┘
                                                                    │
      DRAM ◄──[DMA.OUT]─────────────────────────────────────────────┘
                                                         Vector core computes
                                                         inside SRAM at 1 cyc/access
```

### 3.6.2 Code Example: SRAM Staging for Vector-Add

From Demo 6 in `examples.c`, the DMA-staging variant of a 256-float vector add:

```c
/* Scratchpad addresses for this kernel's working set */
const uint32_t sA = SCRATCH_BASE + 0x0000;   /* input A in SRAM  */
const uint32_t sB = SCRATCH_BASE + 0x1000;   /* input B in SRAM  */
const uint32_t sC = SCRATCH_BASE + 0x2000;   /* result  in SRAM  */
const uint32_t bytes = 256 * 4;              /* 256 floats = 1 KB */

const cinst prog[] = {
    /* Step 1+2: stage inputs into SRAM */
    { .op=C_LI, .rd=5, .imm=sA }, { .op=C_LI, .rd=6, .imm=A_DRAM },
    { .op=T_DMA_IN, .rd=5, .rs1=6, .count=bytes },   /* sA ← DRAM A */

    { .op=C_LI, .rd=7, .imm=sB }, { .op=C_LI, .rd=8, .imm=B_DRAM },
    { .op=T_DMA_IN, .rd=7, .rs1=8, .count=bytes },   /* sB ← DRAM B */

    /* Step 3: launch kernel; arg block points into SRAM */
    { .op=C_LI, .rd=10, .imm=1 },          /* grid = 1 instance */
    { .op=C_LI, .rd=11, .imm=ARGS_SRAM },  /* args in SRAM      */
    { .op=C_KLAUNCH, .imm=kid, .rs1=10, .rs2=11 },

    /* Step 4 executes inside the vector core: VLOAD from SRAM, VADD, VSTORE */
    { .op=C_KWAIT },

    /* Step 5: write result back to DRAM */
    { .op=C_LI, .rd=9,  .imm=C_DRAM }, { .op=C_LI, .rd=12, .imm=sC },
    { .op=T_DMA_OUT, .rd=9, .rs1=12, .count=bytes },  /* DRAM C ← sC */

    { .op=C_HALT },
};
```

The vector kernel itself is unchanged from the DRAM version — it just receives
SRAM addresses in its arg block instead of DRAM addresses. The 100× speedup
comes entirely from the DMA staging, not from any change to the kernel code.

---

## 3.7 Working Set Tiling for ML Workloads

Real ML tensors do not fit in 256 KB. A ResNet layer might process 7×7×512
activations (about 100 KB for float32) plus 512×512×9 weights (about 9 MB).
The weights alone exceed the entire scratchpad by 35×. The solution is **tiling**:
the control core outer-loops over tiles, staging each tile into SRAM, running
the kernel, and writing the result back before moving to the next tile.

### 3.7.1 The Tiled Loop Structure

```
for each output tile (ib, jb):
  DMA.IN  scratchpad[A_tile] ← DRAM A[ib*T : (ib+1)*T, :]
  DMA.IN  scratchpad[B_tile] ← DRAM B[:, jb*T : (jb+1)*T]

  for each K tile (kb):
    ; either use tensor engine fast path...
    T_MMA  tile2, tile0, tile1

    ; ...or launch vector core for a non-standard op
    KLAUNCH  kernel_id, grid, args_ptr

  KWAIT
  DMA.OUT DRAM C[ib*T, jb*T] ← scratchpad[C_tile]
```

The tile size `T` is chosen to satisfy:

```
  sizeof(A_tile) + sizeof(B_tile) + sizeof(C_tile) + sizeof(temporaries) ≤ 256 KB
```

For float32 (4 bytes/element) and a square tile of side T:

```
  3 * T^2 * 4 ≤ 262144
  T^2 ≤ 21845
  T   ≤ 147
```

So a 128×128 tile fits comfortably (3 × 128^2 × 4 = 196 608 bytes, leaving
~65 KB for temporaries). This matches the `MAX_TILE = 64` constant in `npu.h`
for the tensor engine's fast path, which uses 64×64 tiles.

### 3.7.2 DMA Overlap (Pipeline Opportunity)

On hardware (and in future simulator versions), DMA can be pipelined with
compute: while the vector core is processing tile `(ib, jb)`, the DMA engine
can be prefetching tile `(ib, jb+1)` into a second scratchpad bank. This
double-buffering scheme is the standard technique for hiding DMA latency:

```
Time:  |  DMA tile 0  |  compute tile 0  |  compute tile 1  |  ...
                       |  DMA tile 1      |  DMA tile 2      |  ...
```

The simulator does not model double-buffering (it executes instructions
sequentially), but the ISA supports it: the control core issues a `T_DMA_IN`
to the second bank, then immediately issues `C_KLAUNCH` for the first bank.
The hardware runs them concurrently.

---

## 3.8 Scalar Memory Instructions in the Vector Core

The vector core also has scalar memory instructions for loading individual
values (such as arguments from the arg block in SRAM):

| Mnemonic | Operation               | Use case                        |
|----------|-------------------------|---------------------------------|
| `SLW`    | `s[rd] = mem[s[rs]+imm]` (i32) | Load a 32-bit arg or counter |
| `SLF`    | `s[rd] = mem[s[rs]+imm]` (f32) | Load a float scale factor    |

These are the instructions a kernel uses at startup to read its argument block:

```c
/* Typical kernel prologue — reading arguments from SRAM via s1 */
{ .op=V_SLW, .rd=2, .use_base=1, .rs=1, .imm=0  },  /* s2 = A ptr  */
{ .op=V_SLW, .rd=3, .use_base=1, .rs=1, .imm=4  },  /* s3 = B ptr  */
{ .op=V_SLW, .rd=4, .use_base=1, .rs=1, .imm=8  },  /* s4 = C ptr  */
{ .op=V_SLW, .rd=5, .use_base=1, .rs=1, .imm=12 },  /* s5 = N      */
```

The scalar load writes to a scalar register (`s0`–`s31`) rather than to a
vector register. Scalar loads are single-element accesses — they do not pay a
per-lane latency, but they still pay the base `mem_latency` for the address.
Keeping the arg block in SRAM (high-bit set) therefore also reduces arg-loading
overhead.

---

## 3.9 Memory Ordering and Fences

The vector core is a single-threaded instruction stream within one kernel
instance: loads and stores execute in program order within that instance. There
is no out-of-order memory unit in the vector core.

Between the control core (DMA issuance) and the vector core (kernel execution),
ordering is enforced by the `C_KLAUNCH` / `C_KWAIT` protocol: `C_KLAUNCH` does
not start the kernel until preceding DMA operations have completed (in the
simulator, all operations are synchronous; on hardware, `C_KLAUNCH` would check
a completion semaphore before starting instances). `C_KWAIT` ensures the control
core does not proceed to `T_DMA_OUT` until all kernel instances have executed
`V_KEXIT`.

The tensor engine provides its own synchronization primitives (`T_FENCE`,
`T_WAIT`, `T_BARRIER`) for ordering within a stream of tensor operations, but
these are not visible to the vector core.

---

## 3.10 Error Conditions

The simulator enforces strict bounds checking on every memory access:

| Condition                           | What happens                |
|-------------------------------------|-----------------------------|
| DRAM access beyond 16 MB            | `abort()` with address/size |
| SRAM access beyond 256 KB           | `abort()` with offset/size  |
| Vector load/store of `vl=0` lanes   | No-op (zero bytes accessed) |
| `VSETVL` with `imm > VLEN_MAX(256)` | Clamped to 256              |
| `VSETVL` with `imm < 0`             | Clamped to 0                |

The clamp behavior for `VSETVL` is intentional: it allows length-agnostic
kernels to request more lanes than physically available and receive the hardware
maximum, exactly as in the RISC-V Vector extension model.

---

## 3.11 Summary

| Property            | DRAM                      | SRAM Scratchpad           |
|---------------------|---------------------------|---------------------------|
| Address range       | 0x00000000 – 0x00FFFFFF   | 0x80000000 – 0x8003FFFF   |
| Capacity            | 16 MB                     | 256 KB                    |
| Latency (per access)| 100 cycles                | 1 cycle                   |
| Management          | Host fills via `npu_write_f32` | Software/DMA explicit |
| DMA bandwidth       | 32 bytes/cycle (in and out) | N/A (direct access)     |
| Access bit          | Bit 31 = 0                | Bit 31 = 1                |
| Use for             | Input tensors, model weights, output results | Active working sets during kernel execution |

The central lesson: the NPU's memory hierarchy is simple — two tiers, one bit to
distinguish them — but it demands discipline. Kernels that load directly from
DRAM pay 100× the latency of kernels that operate from SRAM. The entire
programming model of staging (DMA in → compute → DMA out) exists to exploit
this gap. Every performance-critical kernel must be written around this
discipline; it is the primary knob the programmer has for closing the gap
between peak-FLOP-rate and sustained-FLOP-rate on real workloads.

---

*Next: Chapter 4 — Data Movement Instructions. We examine each of the seven data
movement instructions in the vector core ISA (VSETVL, VLOAD, VSTORE, VGATHER,
VSCATTER, VMOV, VBCAST) in full detail: instruction encoding, simulator
semantics, cycle cost, use cases, and worked examples.*
