# Chapter 9: The Execution Model

> **ISA Reference:** Section 13 of `npu-simulator/docs/fully_prog_npu.md`
> **Simulator source:** `npu-simulator/src/npu.c`, `npu_create()`, `npu_register_kernel()`,
>   `npu_run()` (`C_KLAUNCH` case), `run_vector_instance()`
> **Header definitions:** `npu-simulator/include/npu.h`
> **Worked examples:** `npu-simulator/examples/examples.c`, all seven demos

---

## 9.1 Overview — A Three-Engine Machine

Before diving into any individual engine's mechanics, it is worth establishing the complete
picture of the NPU as a system. The machine has three engines that operate as a pipeline:

```
                    ┌──────────────────────────────────────────────────────┐
                    │                    NPU CHIP                          │
                    │                                                      │
  Host ──────────── │ ──► Control Core ──► Tensor Engine ──► [sync]        │
  (CPU)   write     │     (scalar ISA)     (fixed matmul)    KWAIT         │
          memory    │         │                                   ▲        │
          launch    │         │  KLAUNCH(kid, grid, args)         │        │
                    │         ▼                                   │        │
                    │    Vector Core Array                         │        │
                    │    ┌────┬────┬────┬────┐                    │        │
                    │    │VC0 │VC1 │VC2 │... │ (num_cores)        │        │
                    │    │inst│inst│inst│    │ KEXIT ─────────────┘        │
                    │    │  0 │  1 │  2 │    │                             │
                    │    └────┴────┴────┴────┘                             │
                    │                                                      │
                    │     ┌──────────────┬──────────────┐                 │
                    │     │  DRAM (16MB) │ SRAM (256KB)  │                 │
                    │     │  off-chip    │ on-chip        │                 │
                    │     │  100cy/access│ 1cy/access     │                 │
                    │     └──────────────┴──────────────┘                 │
                    └──────────────────────────────────────────────────────┘
```

The three engines are not independent agents — they are stages in a coordinated pipeline:

1. The **Control Core** (Layer 1) is the master sequencer. It sets up data in memory,
   configures the tensor engine, launches vector kernels, and waits for completion. It is
   modeled on the RISC-V base ISA.

2. The **Tensor Engine** (Layer 2) executes high-throughput dense matrix operations
   (`T_MATMUL`, `T_MMA`, etc.) and DMA transfers (`T_DMA_IN`, `T_DMA_OUT`). Its
   instructions appear in the same `cinst` stream as the control core — the instruction
   decode logic routes them to the appropriate execution unit based on opcode.

3. The **Vector Core** (Layer 3) executes programmer-specified kernels (`vinst` streams)
   on a grid of instances in parallel. It is the "fully programmable" part of the NPU.

This chapter is about the execution model of the vector core — specifically:
- How kernels are registered and launched
- What "grid instance" means and what state each instance gets
- How the multi-core model works
- What the cycle model measures and what it does not
- The complete end-to-end timeline of a compute operation

---

## 9.2 Kernel Registration

Before the control core can launch a vector kernel with `C_KLAUNCH`, the kernel must be
registered. This is done with `npu_register_kernel()` from `npu.h`:

```c
int npu_register_kernel(npu_t *n, const vinst *code, int len);
```

**Parameters:**
- `n`: the NPU instance
- `code`: pointer to a `vinst` array (the kernel's instruction stream)
- `len`: number of instructions in the array

**Returns:** a kernel ID in the range `[0, MAX_KERNELS-1]`. `MAX_KERNELS = 32`, so an NPU
instance can hold up to 32 distinct kernels simultaneously.

**Simulator implementation** (`npu.c`, lines 99–105):
```c
int npu_register_kernel(npu_t *n, const vinst *code, int len) {
    assert(n->nkernels < MAX_KERNELS);
    int id = n->nkernels++;
    n->kernels[id].code = code;
    n->kernels[id].len  = len;
    return id;
}
```

**The kernel table** is a simple array of `npu_kernel` structs (from `npu.h`, line 137):
```c
typedef struct {
    const vinst *code;
    int          len;
} npu_kernel;
```

The NPU does not copy the instruction array — it stores the pointer. The caller is
responsible for keeping the `vinst` array alive as long as the kernel is registered. In
practice, kernels are declared as `static const vinst` arrays, so they live in the
program's read-only data segment for the duration of execution.

**Registration ordering matters:** The first kernel registered gets ID 0, the second gets
ID 1, and so on. If you register the softmax kernel first and the vadd kernel second, and
then issue `{ .op=C_KLAUNCH, .imm=0, ... }`, you get the softmax kernel. This is simple
but means that kernel IDs are assigned sequentially and cannot be reused within an NPU
instance's lifetime.

**How text-assembled kernels are registered:** The assembler in `npu_asm.c` produces a
`vinst` buffer. You register the buffer, not the text:

```c
vinst kbuf[64];
int kn = npu_asm_vector(ksrc, kbuf, 64, err, sizeof err);
int kid = npu_register_kernel(n, kbuf, kn);
```

---

## 9.3 Kernel Launch — C_KLAUNCH

`C_KLAUNCH` is the control core instruction that launches a vector kernel:

**Syntax:** `KLAUNCH kaddr, grid, args`

In the `cinst` struct (from `npu.h`):
```c
{ .op=C_KLAUNCH,
  .imm = kid,      /* kernel id returned by npu_register_kernel() */
  .rs1 = rx,       /* x[rx] = grid size (number of instances)      */
  .rs2 = ry,       /* x[ry] = args pointer (address in NPU memory) */
}
```

**Simulator implementation** (`npu.c`, lines 392–424, slightly condensed):
```c
case C_KLAUNCH: {
    int kid     = (int)I->imm;
    int64_t grid = getx(n, I->rs1);   /* number of grid instances   */
    int64_t args = getx(n, I->rs2);   /* args block address         */
    assert(kid >= 0 && kid < n->nkernels);
    n->stats.kernels_launched++;
    n->stats.ctrl_cycles += 10;        /* model: 10-cycle launch overhead */

    int cores = n->num_cores;
    uint64_t core_busy[NPU_MAX_CORES];
    memset(core_busy, 0, (size_t)cores * sizeof core_busy[0]);
    uint64_t launch_sum = 0;

    for (int64_t g = 0; g < grid; g++) {
        uint64_t snap = n->stats.vector_cycles;
        run_vector_instance(n, &n->kernels[kid], g, args);
        uint64_t inst_cost = n->stats.vector_cycles - snap;
        core_busy[g % cores] += inst_cost;       /* round-robin placement */
        launch_sum += inst_cost;
        n->stats.instances_run++;
    }

    /* Replace serial accumulation with parallel (max-over-cores) model */
    uint64_t busiest = 0;
    for (int c = 0; c < cores; c++)
        if (core_busy[c] > busiest) busiest = core_busy[c];
    n->stats.vector_cycles -= launch_sum;
    n->stats.vector_cycles += busiest;
    n->stats.vector_work   += launch_sum;
    break;
}
```

Three key observations about the implementation:

1. **The simulator runs instances sequentially.** Despite modeling parallel execution, the
   C code runs one instance at a time with `run_vector_instance()` in a for loop. Parallelism
   is modeled only at the cycle-accounting level — the actual C instructions execute one
   after another. This is why the simulator is an **instruction-set simulator (ISS)**, not
   a cycle-accurate simulator.

2. **Round-robin placement.** Instance `g` is assigned to core `g % cores`. For 8 instances
   on 4 cores: instance 0 → core 0, instance 1 → core 1, ..., instance 4 → core 0 again.
   This is the simplest scheduling policy. A real hardware scheduler might use a
   work-stealing policy, but round-robin is both simple and good enough for balanced
   workloads (where all instances take roughly the same time).

3. **Serial work vs. parallel cycles.** `launch_sum` accumulates the total serial cost of
   all instances. `busiest` is the maximum load on any one core. The cycle counter is
   updated to reflect `busiest` — the time from KLAUNCH to the point when the last instance
   finishes, assuming all cores run simultaneously. `launch_sum` is preserved separately in
   `stats.vector_work` for efficiency analysis.

---

## 9.4 The Grid Instance Model

A "grid" is the set of all vector core instances spawned by a single `C_KLAUNCH`. The grid
has one dimension: `grid_size` instances, numbered `0` to `grid_size - 1`.

### 9.4.1 Instance Identity

Each instance receives two values at startup (from `run_vector_instance()` in `npu.c`,
lines 148–149):

```c
n->s[0].i = inst_id;     /* s0 = this instance's position in the grid */
n->s[1].i = args_addr;   /* s1 = args block base address               */
```

`s0` is the instance ID — an integer in `[0, grid_size - 1]`. The kernel uses this to
determine which slice of the input data to process. `s1` is the args pointer — shared
across all instances. All instances read the same args block to discover tensor base
addresses, dimensions, and other parameters.

**The canonical addressing pattern:**
```
instance id: s0
dimension N from args: s4
byte offset for this instance's slice: s0 * N * sizeof(float)
                                     = s0 * N * 4
```

In code:
```c
{ .op=V_SMUL,  .rd=5, .rs1=0, .rs2=4 },  /* s5 = s0 * N             */
{ .op=V_SMULI, .rd=5, .rs1=5, .imm=4 },  /* s5 = s0 * N * 4 (bytes) */
{ .op=V_SADD,  .rd=6, .rs1=2, .rs2=5 },  /* s6 = base + offset      */
```

This pattern appears in every multi-instance kernel: softmax, matmul row processing,
multi-core batch norm, etc.

### 9.4.2 Grid Sizing and Work Partitioning

How many instances should a KLAUNCH spawn? The answer depends on the structure of the
computation:

**One instance per row (softmax, layernorm):**
```
matrix: ROWS × COLS
instances: ROWS
instance i: processes row i (COLS elements)
```

```c
{ .op=C_LI, .rd=10, .imm=ROWS },
{ .op=C_KLAUNCH, .imm=kid, .rs1=10, .rs2=11 },  /* 1 instance per row */
```

**One instance per tile (large matrix multiply):**
```
output matrix: M × N
tile size: TILE_M × TILE_N
instances: (M / TILE_M) × (N / TILE_N)    -- flattened to 1D
instance i: processes output tile at row=(i / N_tiles), col=(i % N_tiles)
```

**One instance per element (trivial parallelism):**
```
array: N elements
instances: N
instance i: processes element i (vl=1, trivially)
```

This is rarely efficient — the kernel launch overhead (10 cycles) plus per-instance
initialization amortizes poorly over a 1-element computation.

**The fundamental rule:** Each instance should process enough work to make the launch
overhead and register initialization negligible. For vl=16 operations, processing at least
16 elements per instance (one full vector load-compute-store) is a minimum. Processing 64
or more elements per instance is typically better.

### 9.4.3 Args Block Convention

The args block is a small array of 32-bit words stored in NPU memory. All instances read
the same args block — it is the mechanism for passing "compile-time" parameters that are
fixed across all instances. The per-instance parameter (which slice to process) is derived
from `s0` (the instance ID) inside the kernel.

**Typical args block layout:**
```
Offset  Field          Size    Notes
+0      x_base         4 bytes Base address of input tensor
+4      y_base         4 bytes Base address of output tensor
+8      n              4 bytes Number of elements per instance (e.g., row width)
+12     [optional]     4 bytes Additional per-launch parameters
...
```

The args block is typically written to SRAM before the KLAUNCH:
```c
int32_t args[3] = { X_ADDR, Y_ADDR, N };
npu_write_i32(n, ARGS_SRAM_ADDR, args, 3);
/* Then KLAUNCH with .rs2 pointing to ARGS_SRAM_ADDR */
```

Using SRAM for the args block is important for performance: every instance reads the args
at startup (3 SLW instructions in the typical prologue). If the args block is in DRAM, each
read pays `NPU_DRAM_LATENCY = 100` cycles. With SRAM, each read pays `NPU_SRAM_LATENCY = 1`
cycle. For 64 instances launching a kernel with a 3-word args block, the difference is:

```
DRAM args:  64 instances × 3 reads × 100 cycles = 19,200 cycles of args-read overhead
SRAM args:  64 instances × 3 reads × 1 cycle    =    192 cycles of args-read overhead
```

The 100x difference is dramatic. Always put the args block in SRAM.

---

## 9.5 Instance State Isolation

One of the most important properties of the grid model is that instances are completely
isolated from each other. There is no shared state among instances except through memory.

### 9.5.1 What Each Instance Gets Fresh

From `run_vector_instance()` (lines 143–149):
```c
/* per-instance state: fresh registers, shared memory/scratchpad */
memset(n->v, 0, sizeof n->v);    /* v0..v31 all zero          */
memset(n->m, 0, sizeof n->m);    /* m0..m7  all zero          */
memset(n->s, 0, sizeof n->s);    /* s0..s31 all zero          */
n->vl = 0;                        /* vector length = 0         */
n->s[0].i = inst_id;              /* s0 = this instance's id   */
n->s[1].i = args_addr;            /* s1 = args pointer         */
```

Each instance gets:
- **All vector registers zeroed:** `v0..v31` are all-zeros. You must explicitly load data
  before using a vector register.
- **All mask registers zeroed:** `m0..m7` have all bits zero. A predicated instruction
  using an uninitialized mask would skip all lanes. Always set up masks with `VCMP` or
  `VMASK_SET` before using them as predicates.
- **All scalar registers zeroed, except s0 and s1:** `s0 = instance_id`, `s1 = args_ptr`,
  all others zero.
- **vl = 0:** You must call `VSETVL` before any vector instruction. An operation on `vl=0`
  would process zero lanes — a kernel without `VSETVL` would be a no-op.

### 9.5.2 What is Shared Between Instances

Instances do **not** share:
- Vector registers
- Scalar registers
- Mask registers
- Vector length (vl)
- Program counter

Instances **do** share:
- **DRAM** (the `n->mem` array): any instance can read or write any address in DRAM. This
  is how the host writes input data (before launch) and how instances write their results.
- **SRAM scratchpad** (the `n->scratch` array): similarly shared. Instances can
  communicate through SRAM if they write to non-overlapping regions.

**Critical implication:** If two instances write to the same DRAM address, the result is
undefined (a data race). The programmer must ensure that output regions are partitioned
such that each instance writes to a unique range. The standard pattern — "instance i writes
to rows starting at `i * row_bytes`" — guarantees disjoint writes.

### 9.5.3 No Barriers, No Synchronization Within a KLAUNCH

There is no instruction for one instance to wait for another within the same KLAUNCH. If
you need inter-instance synchronization (e.g., a global max over all instances' local
maxima), you must:

1. Have each instance write its local result to a unique SRAM address (no conflict).
2. End the first KLAUNCH (`KEXIT` in all instances, `C_KWAIT` in the control core).
3. Launch a second KLAUNCH (or use the control core) to combine the per-instance results.

This is a fundamental architectural constraint. It simplifies the hardware (no barrier
logic, no cache coherence protocol between vector cores) at the cost of requiring
multi-phase algorithms.

**Example — two-phase softmax for large matrices:**
```
Phase 1 KLAUNCH: Each instance computes max of its row.
                 Writes max to SRAM[instance * 4].
C_KWAIT
Phase 2 KLAUNCH: Each instance reads all maxes, computes global max,
                 then runs the full numerically-stable softmax.
C_KWAIT
```

In practice, if each instance's computation is independent (which is true for row-wise
softmax), phase 2 is unnecessary — each instance already has its own local max. The
two-phase approach is needed only when instances depend on aggregate information.

---

## 9.6 The Multi-Core Model

The NPU supports a configurable number of vector cores via `npu_set_cores()`:

```c
void npu_set_cores(npu_t *n, int num_cores) {
    if (num_cores < 1) num_cores = 1;
    if (num_cores > NPU_MAX_CORES) num_cores = NPU_MAX_CORES;
    n->num_cores = num_cores;
}
```

`NPU_MAX_CORES = 1024` (from `npu.h`). The default is 1 core.

### 9.6.1 What "Core" Means in the Model

A "vector core" in the NPU model is a physical execution unit capable of running one grid
instance at a time. With N cores, N instances run simultaneously (in the parallel sense).

### 9.6.2 Round-Robin Instance Distribution

Grid instances are distributed across cores in a simple round-robin pattern. For `grid_size
= 8` instances and `num_cores = 4`:

```
Core 0: processes instances 0, 4
Core 1: processes instances 1, 5
Core 2: processes instances 2, 6
Core 3: processes instances 3, 7

Core 0 busy time: cycles(instance 0) + cycles(instance 4)
Core 1 busy time: cycles(instance 1) + cycles(instance 5)
Core 2 busy time: cycles(instance 2) + cycles(instance 6)
Core 3 busy time: cycles(instance 3) + cycles(instance 7)

vector_cycles for this launch = max(Core 0, Core 1, Core 2, Core 3)
```

The cycle model charges the wall-clock time of the KLAUNCH as the maximum over all core
busy times — this models the case where all cores run in parallel and the KWAIT stalls
until the last core finishes.

### 9.6.3 Cycle Model Mathematics

Let `t(g)` be the vector cycle cost of instance `g`. Then:

```
serial_work  = sum(t(g) for g = 0..grid_size-1)
core_busy[c] = sum(t(g) for g where g % cores == c)
parallel_time = max(core_busy[c] for c = 0..num_cores-1)
```

For uniformly-expensive instances (all `t(g)` equal, which is the common case):
```
t(g) = T for all g
serial_work  = grid_size * T
core_busy[c] = ceil(grid_size / cores) * T      (or floor for the remainder cores)
parallel_time = ceil(grid_size / cores) * T
speedup vs 1 core = serial_work / parallel_time
               = grid_size / ceil(grid_size / cores)
               ≈ min(cores, grid_size)   for large grid_size
```

For `grid_size = 64, cores = 8, T = 100 cycles`:
```
serial_work   = 6400 cycles
parallel_time = ceil(64/8) * 100 = 800 cycles
speedup       = 8x
```

`npu_print_stats()` reports this directly (from `npu.c`, lines 495–498):
```c
if (s->vector_work && n->num_cores > 1) {
    double sp = s->vector_cycles ?
                (double)s->vector_work / (double)s->vector_cycles : 1.0;
    printf("  (serial work ~%llu cycles -> %.2fx speedup vs 1 core)\n",
           (unsigned long long)s->vector_work, sp);
}
```

### 9.6.4 When the Round-Robin Model is Pessimistic

Round-robin is pessimistic when instances have highly variable cost. Consider 8 instances
on 4 cores where instance costs are:

```
Instance 0: 100 cycles    Instance 4: 100 cycles
Instance 1: 1000 cycles   Instance 5: 100 cycles
Instance 2: 100 cycles    Instance 6: 100 cycles
Instance 3: 100 cycles    Instance 7: 100 cycles
```

Round-robin assignment:
```
Core 0: instance 0 (100) + instance 4 (100) = 200 cycles
Core 1: instance 1 (1000) + instance 5 (100) = 1100 cycles  ← bottleneck
Core 2: instance 2 (100) + instance 6 (100) = 200 cycles
Core 3: instance 3 (100) + instance 7 (100) = 200 cycles

parallel_time = max(200, 1100, 200, 200) = 1100 cycles
```

An optimal scheduler would assign instance 1 to a core alone, achieving:
```
Core 0: instance 0 + instance 2 + instance 3 + instance 5 + instance 6 + instance 7 = 600 cycles
Core 1: instance 1 = 1000 cycles
parallel_time = 1000 cycles   (9% better)
```

For the NPU's primary use cases (uniform data — same operation on all rows of a matrix),
all instances have equal cost and round-robin is optimal.

---

## 9.7 The Cycle Model — What It Measures

The cycle model in the simulator is described in `npu.h` as "rough cycle estimates". This
section explains what that means precisely and what the counts do and do not capture.

### 9.7.1 Per-Instruction Vector Costs

From `vec_cost()` in `npu.c` (lines 110–132):

```c
static uint64_t vec_cost(int op, int vl) {
    int passes = (vl + HW_LANES - 1) / HW_LANES;
    if (passes < 1) passes = 1;
    switch (op) {
    case V_VEXP: case V_VLOG: case V_VSQRT: case V_VRSQRT:
    case V_VRECIP: case V_VDIV:
        return (uint64_t)passes * 4;            /* transcendental/divide  */
    case V_VGATHER: case V_VSCATTER:
        return (uint64_t)(vl > 0 ? vl : 1);    /* ~1 lane per cycle      */
    case V_VREDSUM: case V_VREDMAX: case V_VREDMIN:
    case V_VREDAND: case V_VREDOR: case V_VSCAN:
        return (uint64_t)passes + 4;            /* tree reduction         */
    case V_VLOAD: case V_VSTORE:
        return (uint64_t)passes;
    case V_SLI: case V_SFI: case V_SADD: ...:
        return 1;                               /* scalar + control ops   */
    default:
        return (uint64_t)passes;                /* elementwise            */
    }
}
```

The `passes` variable is `ceil(vl / HW_LANES)`. With `HW_LANES = 16` and `vl = 16`,
`passes = 1`. With `vl = 32`, `passes = 2`. This models a hardware with 16 physical ALU
lanes that needs multiple passes to process a logical vector wider than 16 elements.

**The physical lane model:**
```
Logical vector (vl = 32 elements):

  v0:  [e0][e1][e2]...[e15][e16][e17]...[e31]
                            ^
                            pass boundary

Physical hardware (HW_LANES = 16):
  Pass 1: processes e0..e15  in cycle N
  Pass 2: processes e16..e31 in cycle N+1
  Total:  2 passes = 2 cycles for the compute part
```

**Memory latency is additional** (added on top of `passes` for VLOAD/VSTORE):
```c
case V_VLOAD:
    n->stats.vector_cycles += mem_latency(addr);  /* +100 or +1 */
    for (int l = 0; l < vl; l++) n->v[I->vd][l] = ld_f(n, addr + 4u * l);
    break;
```

For `vl = 16` from DRAM: total cost = 1 (pass) + 100 (DRAM latency) = 101 cycles.
For `vl = 16` from SRAM: total cost = 1 (pass) + 1 (SRAM latency) = 2 cycles.

This 50x difference for a single VLOAD is the primary quantitative argument for staging
data through SRAM.

### 9.7.2 What the Cycle Model Does NOT Capture

The cycle model is deliberately simplified. Here is what it omits:

| Omitted Effect | Real Hardware Impact | Consequence |
|----------------|----------------------|-------------|
| Pipeline stalls | Read-after-write hazards add 1-4 cycles between dependent instructions | Model underestimates latency for dependent instruction chains |
| Instruction fetch | Kernel instructions must be loaded from memory | Negligible for short kernels in I-cache; could matter for kernels >1KB |
| Launch latency | KLAUNCH charges only 10 cycles | Real hardware may take 50-500 cycles to configure the vector core grid |
| Register bank conflicts | Two reads from the same bank port conflict | Model ignores register bank structure |
| Scratchpad bank conflicts | Multiple simultaneous accesses to the same SRAM bank stall | Model charges `SRAM_LATENCY=1` regardless of bank conflicts |
| Cache effects | DRAM access pattern (sequential vs. random) changes effective bandwidth | Model charges flat 100 cycles per access regardless of locality |
| Prefetching | DMA can overlap with compute | Model does not overlap DMA and compute cycles |
| Power throttling | Sustained high compute throughput triggers thermal throttling | Model ignores thermal/power state |

The model's accuracy goal is order-of-magnitude correctness — it will correctly tell you
that a VGATHER from DRAM is 100x more expensive than from SRAM, but it will not correctly
predict whether a specific kernel runs in 847 cycles or 923 cycles.

### 9.7.3 The VGATHER Cost — A Cautionary Tale

```c
case V_VGATHER: case V_VSCATTER:
    return (uint64_t)(vl > 0 ? vl : 1);    /* ~1 lane per cycle */
```

For `vl = 16` from DRAM: `16` cycles for the `vec_cost` part, PLUS `16 × NPU_DRAM_LATENCY`
since each lane makes an independent random DRAM access:

Looking at the VGATHER implementation in `npu.c` (lines 189–193):
```c
case V_VGATHER:
    for (int l = 0; l < vl; l++) {
        int32_t idx = (int32_t)n->v[I->vs1][l];
        n->v[I->vd][l] = ld_f(n, addr + 4u * (uint32_t)idx);
    }
    break;
```

Notice: the loop calls `ld_f()` which calls `resolve()` which does NOT add memory latency
to the stats. The memory latency for VGATHER/VSCATTER is only the `vec_cost` return value
(vl cycles). This is actually an underestimate of the real cost on hardware — each random
lane access to DRAM would trigger a separate cache miss, costing 100 cycles each. The
correct model would be `vl * NPU_DRAM_LATENCY` for DRAM gathers.

**For kernel writers:** Treat VGATHER from DRAM as costing approximately `100 × vl` cycles
(1,600 cycles for `vl=16`). Always gather from SRAM, where the cost is approximately
`1 × vl` cycles (16 cycles for `vl=16`) — a 100x improvement.

---

## 9.8 Complete End-to-End Timeline

This section traces the complete lifecycle of a computation from host setup to result
readback, using the multi-row softmax as the example.

### 9.8.1 The Computation: Softmax on a 64×16 Matrix

Input: a 64×16 matrix of float32 values in DRAM.
Output: the row-wise softmax of that matrix, written back to DRAM.
Configuration: 8 vector cores, 1 KLAUNCH with grid_size=64 (one instance per row).

### 9.8.2 Full Timeline

```
Wall-clock time →
(not to scale; DRAM phases dominate)

═══════════════════════════════════════════════════════════════════════════════════
Phase 1: Host Setup (outside NPU simulation, before npu_run)
─────────────────────────────────────────────────────────────────────────────────
Host CPU:
  npu_create()              ← allocate 16MB DRAM + 256KB SRAM
  npu_register_kernel()     ← install softmax vinst[] at kernels[0]
  npu_write_f32(X, ...)     ← write 64×16 floats to DRAM at address X
  npu_write_i32(ARGS, ...)  ← write {X, Y, 16} to SRAM args block

═══════════════════════════════════════════════════════════════════════════════════
Phase 2: DMA Staging (optional; skip if data fits in SRAM)
─────────────────────────────────────────────────────────────────────────────────
Control core executes:                     Cycle count:
  C_LI x5, SRAM_X; C_LI x6, DRAM_X        2 cycles
  T_DMA_IN dst=x5, src=x6, count=4096B    4096/32 + 1 = 129 cycles
  C_LI x7, SRAM_Y; C_LI x8, DRAM_Y        2 cycles

  Note: DMA cost = ceil(bytes / NPU_DMA_BW) = ceil(4096/32) = 128 + 1 = 129 cycles

═══════════════════════════════════════════════════════════════════════════════════
Phase 3: Kernel Launch and Execution
─────────────────────────────────────────────────────────────────────────────────
Control core executes:
  C_LI x10, 64           ← grid size = 64 rows
  C_LI x11, ARGS_SRAM    ← args in SRAM
  C_KLAUNCH kid, x10, x11   ← launch overhead: 10 cycles

Then the simulator runs instances 0..63 sequentially (for functional correctness),
but models them as running on 8 cores:

  Core 0 (instances 0, 8, 16, 24, 32, 40, 48, 56):
  │  instance 0: ~30 cycles (19 instructions + mem latency from SRAM)
  │  instance 8: ~30 cycles
  │  ...
  └─ Core 0 total: 8 × 30 = 240 cycles

  Core 1 (instances 1, 9, 17, 25, 33, 41, 49, 57):
  └─ Core 1 total: ~240 cycles

  ... (all cores roughly equal for uniform workload) ...

  vector_cycles for this launch = max(240, 240, ..., 240) = 240 cycles
  vector_work = 64 × 30 = 1920 cycles (serial total)
  speedup = 1920 / 240 = 8x

After all instances complete:
  C_KWAIT               ← 2 cycles in simulator

═══════════════════════════════════════════════════════════════════════════════════
Phase 4: DMA Out (if staging was used)
─────────────────────────────────────────────────────────────────────────────────
  T_DMA_OUT dst=DRAM_Y, src=SRAM_Y, count=4096B  ← 129 cycles

═══════════════════════════════════════════════════════════════════════════════════
Phase 5: Host Readback
─────────────────────────────────────────────────────────────────────────────────
Host CPU:
  npu_read_f32(Y, ...)   ← read 64×16 floats from DRAM/SRAM
  check results

═══════════════════════════════════════════════════════════════════════════════════
```

### 9.8.3 Cycle Budget Summary

| Phase | Cycles | Dominant Factor |
|-------|--------|-----------------|
| DMA in (4KB) | ~129 | DMA bandwidth (32 B/cycle) |
| Launch overhead | 10 | Fixed per KLAUNCH |
| Vector execution (8 cores) | ~240 | 8 instances/core × 30 cy/instance |
| KWAIT | 2 | Fixed |
| DMA out (4KB) | ~129 | DMA bandwidth |
| **Total** | **~510** | DMA dominates for small matrices |

For larger matrices (1024×16), the DMA cost is still ~2000 cycles but the vector
execution is 1024 instances / 8 cores × 30 cycles = 3840 cycles — now vector execution
dominates. The "right" size for SRAM staging to pay off requires the compute benefit to
exceed the DMA staging cost.

---

## 9.9 NPU State — The Complete Machine

The complete `npu_t` struct (from `npu.h`, lines 141–168) gives the full picture of what
state the machine tracks:

```c
typedef struct {
    /* memory */
    uint8_t *mem;           /* off-chip DRAM: 16 MB                        */
    size_t   mem_size;
    uint8_t *scratch;       /* on-chip SRAM: 256 KB                        */
    size_t   scratch_size;

    /* control core */
    int64_t  x[NCREG];     /* x0..x31 (x0 hardwired to 0)                 */

    /* tensor engine */
    float    tile[NTREG][MAX_TILE * MAX_TILE]; /* tile registers t0..t7   */
    int      cfg_M, cfg_K, cfg_N;              /* current tile dimensions  */
    int      cfg_in_dtype, cfg_acc_dtype;      /* precision config         */
    int      cfg_mode, cfg_rnd;                /* engine mode, rounding    */
    int      cfg_stride_r, cfg_stride_c;       /* tile addressing strides  */
    int64_t  tensor_status;                    /* engine status flags      */

    /* vector engine (reset per grid instance) */
    float    v[NVREG][VLEN_MAX];   /* v0..v31, each up to 256 lanes        */
    uint8_t  m[NMREG][VLEN_MAX];   /* m0..m7, 1 bit/lane (stored as bytes) */
    sreg_t   s[NSREG];             /* s0..s31, float/int union             */
    int      vl;                   /* current vector length                */

    /* kernel table */
    npu_kernel kernels[MAX_KERNELS]; /* up to 32 registered kernels        */
    int        nkernels;             /* count of registered kernels        */

    /* configuration */
    int        num_cores;   /* number of simulated vector cores            */
    int        trace;       /* 1 = print each executed instruction         */

    /* statistics */
    npu_stats  stats;
} npu_t;
```

**Size of the vector register file:**
```
v[32][256] × sizeof(float) = 32 × 256 × 4 = 32,768 bytes = 32 KB
```

For comparison, the SRAM scratchpad is 256 KB — 8× larger than the vector register file.
This ratio matters: you can hold 8 full register-file-equivalents in SRAM, which enables
staging and double-buffering schemes.

**Size of the mask register file:**
```
m[8][256] × sizeof(uint8_t) = 8 × 256 × 1 = 2,048 bytes = 2 KB
```

In hardware with 1-bit-per-lane encoding: 8 × 256 bits = 256 bytes.

---

## 9.10 Tracing and Debugging

The NPU simulator includes a trace mode for kernel debugging. Set `n->trace = 1` before
running:

```c
npu_t *n = npu_create();
n->trace = 1;   /* enable instruction trace */
npu_run(n, prog, len);
```

The trace prints each executed instruction in the control core:
```
  [  0] LI x10, 64                imm=64
       → x10 = 64
  [  1] LI x11, 4096              imm=4096
       → x11 = 4096
  [  2] KLAUNCH 0, x10, x11
  [vec pc=0 op=16 vl=0]    ← V_SLW
  [vec pc=1 op=16 vl=0]    ← V_SLW
  ...
```

The vector core trace (from `run_vector_instance()` at line 161):
```c
if (n->trace)
    fprintf(stderr, "  [vec pc=%d op=%d vl=%d]\n", pc, I->op, vl);
```

Currently, the vector trace only prints the opcode number and vl — not the instruction
mnemonic. For more readable output, use the disassembler:

```c
/* Disassemble a vinst to text */
char dis[128];
npu_disasm_vinst(&k[pc], dis, sizeof dis);
fprintf(stderr, "  [vec pc=%d] %s\n", pc, dis);
```

The disassembler (`npu_asm.c`) produces human-readable output like `"VREDMAX s9, v0"` for
the reduction instruction.

---

## 9.11 From Simulator to Hardware — What Changes

The simulator is an instructional tool, not a hardware model. This section describes what
would change in a real hardware implementation and why.

### 9.11.1 True Parallel Execution

The simulator runs instances sequentially in a for loop. Real hardware runs them in
parallel on separate physical cores. Each core has its own copy of:
- The vector register file (v0..v31)
- The mask register file (m0..m7)
- The scalar register file (s0..s31)
- The program counter

The kernel instruction stream (`vinst[]`) is shared — it sits in instruction memory and
is read by all cores simultaneously (different PCs may read different instructions on each
cycle, but the instruction memory itself is shared read-only).

### 9.11.2 Instruction Memory

In the simulator, kernel code is a pointer to a `const vinst[]` in the host C program's
memory. In hardware, kernel instructions must be loaded into the NPU's instruction RAM
before execution. This is why the binary encoder/decoder (`npu_encode_vinst` /
`npu_decode_vinst` in `npu_asm.c`) exists — the `vinst` struct (72 bytes in the current
C implementation) would be encoded into a compact fixed-width machine word (the simulator
uses `NPU_WORD_BYTES = 32` bytes per instruction) for storage in instruction memory.

Demo 7 in `examples.c` exercises this encode/decode path and verifies that the decoded
instructions produce bit-identical results:
```c
for (int i = 0; i < kn; i++) {
    unsigned char word[NPU_WORD_BYTES];
    npu_encode_vinst(&kbuf[i], word);
    npu_decode_vinst(word, &kbuf2[i]);
}
```

### 9.11.3 Memory Subsystem

The simulator models memory as two flat `malloc()`-ed arrays. A real implementation has:
- A DRAM controller with burst transfers, refresh cycles, and row open/close latency
- An on-chip SRAM with banking, arbitration, and ECC
- A DMA engine with scatter-gather descriptors and completion interrupts
- L1/L2 caches (optional, depending on the NPU architecture)

The 100-cycle DRAM latency and 1-cycle SRAM latency in the simulator are rough averages
that capture the 100x difference in access cost without modeling any of the subsystem
details.

### 9.11.4 The Compiler Gap

The simulator receives `vinst[]` arrays written in C designated-initializer syntax. This
is the ISS-programmer interface. A real NPU deployment would have:
- A compiler (e.g., an LLVM backend for the NPU vector core ISA)
- A kernel language (e.g., a dialect of MLIR, or a domain-specific language)
- A runtime that loads compiled ELF files into the NPU

The design document `npu-simulator/docs/llvm_lowering.md` sketches this compiler path.
The file `kernels/vadd.ll` shows an example of what LLVM IR targeting the NPU vector core
might look like. Connecting the compiler output to the simulator's `vinst` format is the
"last mile" that makes the system end-to-end.

---

## 9.12 Statistics Reference

`npu_print_stats()` reports the following counters after execution:

```
  --- NPU stats ---------------------------------
  control core : [instret] instret, ~[cycles] cycles
  tensor engine: [instret] instret, ~[cycles] cycles
  vector core  : [instret] instret, ~[cycles] cycles on [N] core(s)
               : (serial work ~[work] cycles -> [speedup]x speedup vs 1 core)
  kernels launched: [count], grid instances run: [count]
  -----------------------------------------------
```

| Counter | Field in `npu_stats` | Meaning |
|---------|----------------------|---------|
| `ctrl_instret` | Instructions retired by control core | Count of cinst executed (excludes tensor) |
| `ctrl_cycles` | Modeled control core cycles | 1 per scalar instruction + memory latency |
| `tensor_instret` | Tensor engine instructions retired | Count of T_* instructions executed |
| `tensor_cycles` | Modeled tensor engine cycles | Depends on op: MACs / (HW_LANES^2) + latency |
| `vector_instret` | Total vector instructions retired across all instances | Sum over all instances |
| `vector_cycles` | Modeled parallel vector cycles | max_over_cores(instance costs) |
| `vector_work` | Total serial vector work | sum_over_instances(instance costs) |
| `kernels_launched` | Times C_KLAUNCH was executed | — |
| `instances_run` | Total grid instances completed | sum of grid sizes |

**Interpreting `vector_work` vs `vector_cycles`:**
- `vector_work / vector_cycles = num_cores` for perfectly balanced workloads
- A ratio significantly less than `num_cores` indicates load imbalance (some cores idle)
- A ratio of 1.0 means single-core execution or perfectly serialized instances

---

## 9.13 Worked Example — Complete Program Trace

The following is the complete control program for the multicore softmax demo, with cycle
annotations:

```c
const cinst prog[] = {
    /* Instruction 0 */ { .op=C_LI, .rd=10, .imm=64 },    /* x10 = 64 (rows)      */
    /* Instruction 1 */ { .op=C_LI, .rd=11, .imm=ARGS },  /* x11 = args address   */
    /* Instruction 2 */ { .op=C_KLAUNCH, .imm=0, .rs1=10, .rs2=11 },
    /* Instruction 3 */ { .op=C_KWAIT },
    /* Instruction 4 */ { .op=C_HALT },
};
```

Cycle accounting for each instruction:
```
Inst 0: C_LI          → ctrl_cycles += 1       (total: 1)
Inst 1: C_LI          → ctrl_cycles += 1       (total: 2)
Inst 2: C_KLAUNCH     → ctrl_cycles += 10      (total: 12)
        runs 64 instances, models 8 cores
        vector_cycles += max_core_busy
        (for 64 uniform instances of ~30 cycles on 8 cores:
         each core runs 8 instances × 30 = 240 cycles)
        vector_cycles += 240               (vector_cycles: 240)
        vector_work += 1920               (vector_work: 1920)
Inst 3: C_KWAIT       → ctrl_cycles += 2       (total: 14)
Inst 4: C_HALT        → return from npu_run()
```

Final stats:
```
  control core : 5 instret, ~14 cycles
  vector core  : 64×19 = 1216 instret, ~240 cycles on 8 core(s)
               : (serial work ~1920 cycles -> 8.00x speedup vs 1 core)
  kernels launched: 1, grid instances run: 64
```

---

## Summary

The NPU execution model has three key properties that define how to write efficient kernels:

**1. State isolation enables embarrassing parallelism.**
Instances share memory but have private registers. This means:
- No race conditions in register operations
- Race conditions in memory are possible if output regions overlap — avoid this
- No inter-instance communication except through memory (with a KWAIT in between)

**2. The SRAM scratchpad is the performance lever.**
A 100x difference in memory latency (1 vs. 100 cycles) is the dominant performance
variable. The programming pattern: DMA inputs to SRAM, compute in SRAM, DMA outputs out.
Demo 6 in `examples.c` measures this concretely.

**3. The cycle model quantifies trade-offs, not exact timings.**
Use the cycle model to compare architectural choices (DRAM vs. SRAM, 1 core vs. 8 cores,
VGATHER vs. sequential VLOAD with VSCAN) rather than to predict absolute latencies. The
model is accurate for relative comparisons within its stated assumptions.

These three properties, combined with the predication model from Chapter 8 and the
reduction instructions from Chapter 7, define the complete programming model for the NPU
vector core. A compiler targeting this ISA would need to map high-level tensor operations
to the three-layer pipeline (control core orchestration, optional tensor engine fast path,
vector core general computation), choose SRAM staging strategies, and pack work into
grid instances sized to amortize launch overhead.

---

## Appendix 9A — Putting It All Together: Single Attention Head

This section shows how ALL THREE layers of the NPU collaborate to compute one attention
head. It is the most complete example in this book.

### The Operation

```
Attention(Q, K, V) = softmax( Q × Kᵀ / √d_k ) × V
```

Inputs (all in DRAM, float32):
- `Q` : [seq_len × d_k]   query matrix
- `K` : [seq_len × d_k]   key matrix
- `V` : [seq_len × d_v]   value matrix

Output (DRAM):
- `O` : [seq_len × d_v]   attention output

We use: `seq_len = 64`, `d_k = d_v = 16` (small but illustrative).

### SRAM Layout

```
SRAM (256 KB scratchpad):
┌─────────────────────────────────────────────────────────────┐
│ 0x80000000  Q_tile   [64 × 16 floats = 4 KB]               │
│ 0x80001000  K_tile   [64 × 16 floats = 4 KB]               │
│ 0x80002000  V_tile   [64 × 16 floats = 4 KB]               │
│ 0x80003000  scores   [64 × 64 floats = 16 KB]  Q × Kᵀ      │
│ 0x80007000  attn     [64 × 64 floats = 16 KB]  softmax(·)  │
│ 0x8000B000  O_tile   [64 × 16 floats = 4 KB]   final out   │
│ 0x8000C000  (free)                                          │
└─────────────────────────────────────────────────────────────┘
```

### Stage 1 — Control Core: DMA inputs in

```c
cinst stage1[] = {
    /* DMA Q from DRAM → SRAM */
    { .op=T_DMA_IN,  .a=0, .count=64*16, .imm=(int64_t)Q_dram_addr,
      .use_base=0 },   /* destination: 0x80000000 (tile reg a=0 = SRAM Q_tile) */

    /* DMA K from DRAM → SRAM */
    { .op=T_DMA_IN,  .a=1, .count=64*16, .imm=(int64_t)K_dram_addr },

    /* DMA V from DRAM → SRAM */
    { .op=T_DMA_IN,  .a=2, .count=64*16, .imm=(int64_t)V_dram_addr },
};
```

### Stage 2 — Tensor Engine: Q × Kᵀ → scores (fast path)

The matmul Q × Kᵀ is a perfect tensor engine workload — dense, large, regular.

```c
cinst stage2[] = {
    /* Configure tile shape: M=64, K=16, N=64  (64×16 × 16×64 = 64×64) */
    { .op=T_CFG_SHAPE, .M=64, .K=16, .N=64 },

    /* Zero accumulator tile 3 */
    { .op=T_ACC_CLR, .dt=3 },

    /* MMA: tile3 += Q_tile(tile0) × K_tile(tile1)ᵀ */
    { .op=T_MMA, .dt=3, .a=0, .b=1 },

    /* Store scores [64×64] to SRAM at 0x80003000 */
    { .op=T_STORE, .a=3, .count=64*64, .imm=(int64_t)0x80003000 },

    { .op=T_FENCE },
};
```

### Stage 3 — Vector Core: scale + softmax over each row of scores

Each of the 64 rows of the scores matrix needs:
1. Scale by `1/√d_k = 1/4 = 0.25`
2. Numerically stable softmax

We launch **64 instances** (one per row), each processing one row of 64 scores.

```c
/* The vector kernel */
static const vinst attn_softmax[] = {
    /* --- Setup: vl=64, compute my row base address --- */
    { .op=V_VSETVL, .rd=9, .imm=64 },         /* vl = 64 (one full row)     */
    { .op=V_KID,    .rd=2 },                   /* s2 = row index (0..63)     */
    { .op=V_SMULI,  .rd=2, .rs1=2, .imm=256 },/* s2 = row * 64*4 bytes      */
    /* s1 = args_base = 0x80003000 (scores) */
    { .op=V_SADD,   .rd=3, .rs1=1, .rs2=2 },  /* s3 = scores + row_offset   */

    /* --- Load this row's 64 raw scores --- */
    { .op=V_VLOAD,  .vd=0, .rs=3, .imm=0, .use_base=1 },

    /* --- Scale by 1/sqrt(d_k) = 0.25 --- */
    { .op=V_SFI,    .rd=5, .fimm=0.25 },       /* s5 = 0.25f                 */
    { .op=V_VBCAST, .vd=1, .rs=5 },            /* v1[l] = 0.25               */
    { .op=V_VMUL,   .vd=0, .vs1=0, .vs2=1 },   /* v0 = scores * 0.25         */

    /* --- Numerically stable softmax --- */
    { .op=V_VREDMAX, .rd=4, .vs=0 },           /* s4 = max(scaled scores)    */
    { .op=V_VBCAST,  .vd=1, .rs=4 },           /* v1[l] = max                */
    { .op=V_VSUB,    .vd=0, .vs1=0, .vs2=1 },  /* v0 = score - max           */
    { .op=V_VEXP,    .vd=1, .vs=0 },           /* v1 = exp(score - max)      */
    { .op=V_VREDSUM, .rd=4, .vs=1 },           /* s4 = sum(exp values)       */
    { .op=V_SRECIP,  .rd=4, .rs1=4 },          /* s4 = 1/sum                 */
    { .op=V_VBCAST,  .vd=2, .rs=4 },           /* v2[l] = 1/sum              */
    { .op=V_VMUL,    .vd=1, .vs1=1, .vs2=2 },  /* v1 = attn weights          */

    /* --- Write attn row to SRAM --- */
    /* s3 already points to scores row; output is at scores + 64*64*4 bytes  */
    { .op=V_SADDI,   .rd=6, .rs1=3, .imm=16384 }, /* s6 = attn_base + row  */
    { .op=V_VSTORE,  .vs=1, .rs=6, .imm=0, .use_base=1 },

    { .op=V_KEXIT },
};
```

Launch:
```c
cinst stage3[] = {
    { .op=C_LI,      .rd=1,  .imm=0x80003000 },  /* x1 = scores SRAM base */
    { .op=C_LI,      .rd=2,  .imm=64 },          /* x2 = grid size        */
    { .op=C_KLAUNCH, .imm=softmax_kid, .rd=2, .rs1=1 },
    { .op=C_KWAIT },
};
```

### Stage 4 — Tensor Engine: attn × V → output (fast path)

```c
cinst stage4[] = {
    /* attn[64×64] × V[64×16] → O[64×16] */
    { .op=T_CFG_SHAPE, .M=64, .K=64, .N=16 },
    { .op=T_ACC_CLR,   .dt=4 },
    /* Load attn matrix into tile 5 */
    { .op=T_LOAD,  .a=5, .count=64*64, .imm=(int64_t)0x80007000 },
    /* V is already in tile 2 from stage 1 */
    { .op=T_MMA,   .dt=4, .a=5, .b=2 },
    { .op=T_STORE, .a=4, .count=64*16, .imm=(int64_t)0x8000B000 },
    { .op=T_FENCE },
};
```

### Stage 5 — Control Core: DMA output back to DRAM

```c
cinst stage5[] = {
    { .op=T_DMA_OUT, .a=4, .count=64*16, .imm=(int64_t)O_dram_addr },
    { .op=C_HALT },
};
```

### Full Timeline

```
Time →
                                                          
Ctrl:  [DMA Q,K,V in]──[KLAUNCH matmul]──[KWAIT]──[KLAUNCH softmax]──[KWAIT]──[matmul attn×V]──[DMA out]

TE:                       [Q×Kᵀ matmul]                               [attn×V matmul]

VC:                                                [softmax ×64 rows]

             Stage 1       Stage 2          Stage 3                     Stage 4        Stage 5
```

### What each engine contributed

| Engine | Task | Why this engine? |
|--------|------|-----------------|
| Tensor Engine | Q×Kᵀ matmul (64×16 × 16×64) | Dense regular matmul → fast path |
| Vector Core | row-wise scale + softmax | Cross-lane reduction + transcendental → must be VC |
| Tensor Engine | attn×V matmul (64×64 × 64×16) | Dense regular matmul → fast path |
| Control Core | DMA, sequencing, launch | Orchestration |

This is the canonical NPU programming pattern for transformer inference: use the tensor
engine for the heavy matmuls, hand off to the vector core for everything the tensor engine
cannot express (softmax, layernorm, activations), and use the control core to orchestrate
the handoffs.
