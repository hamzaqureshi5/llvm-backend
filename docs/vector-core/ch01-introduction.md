# Chapter 1: Introduction — The NPU and the Vector Core

---

## 1.1 The Problem: Neural Network Inference at the Edge

The performance requirements of modern deep learning models are staggering. A transformer model
performing self-attention computes scaled dot-product attention across a sequence. A language model
generating a single token runs a matrix multiplication for every layer on every forward pass.
Convolutional neural networks processing video frames must sustain dozens of such passes per second
to be usable in real-time applications. And all of this must happen — increasingly — not in a
hyperscale data center with racks of GPUs and kilowatts of cooling, but inside a smartphone, a
camera module, a hearing aid, an autonomous vehicle controller, or an IoT gateway.

The challenge is not just raw compute throughput. It is *throughput per watt*, *throughput per
dollar of silicon area*, and *latency*. An inference engine that can process one image in ten
milliseconds using twenty watts is useless in a battery-powered device that has a power budget of
two watts. An engine that can sustain a hundred-millisecond latency is useless for real-time speech
recognition. The design space is brutal: you must be fast, you must be cheap, and you must be
miserly with power.

### 1.1.1 The Operations That Matter

Every major neural network architecture, despite the dizzying variety of layer types and topologies,
reduces at inference time to a small set of core mathematical operations:

**Matmul and linear projections.** Every fully-connected layer, every transformer's Q, K, V
projection, every embedding lookup aggregation is ultimately `Y = X * W + b` — a matrix
multiplication followed by a bias addition. For a transformer with model dimension d=4096,
each attention projection computes a `[seq_len, 4096] x [4096, 4096]` matmul. These dominate
the floating-point operation count of virtually every model.

**Softmax.** The attention mechanism requires softmax over the attention scores. Softmax is
the operation `y_i = exp(x_i - max(x)) / sum(exp(x_j - max(x)))`. It involves a maximum
reduction, an elementwise exponential, a sum reduction, a reciprocal, and an elementwise
multiply. It is not a single fused hardware operation — it is an *algorithm* composed of
several distinct primitives.

**LayerNorm and RMSNorm.** Transformer blocks sandwich their sublayers with normalization.
LayerNorm computes the mean and variance of an activation vector, then rescales: `y = (x -
mean) / sqrt(var + eps) * gamma + beta`. This involves two reductions (mean, variance), a
square root, and two elementwise operations. RMSNorm is simpler but structurally similar.

**Elementwise activations.** GELU, SiLU, and the venerable ReLU are applied after linear
projections. GELU involves a tanh or error function approximation. SiLU is `x * sigmoid(x)`,
which involves an exponential. ReLU is just `max(0, x)`, but even it requires branchless
per-element conditionals.

**Convolution.** CNNs for image recognition and object detection depend on 2D convolution with
learnable weight tensors. At its core, convolution is a sliding window of dot products — a
structured form of matrix multiplication.

**Gather and scatter.** Sparse models, mixture-of-experts routing, and embedding lookups all
involve indirect memory accesses: "load from address `base + index[l]`" where `index[l]` is
itself a data-dependent value. No fixed-function unit can handle the address stream; you need
programmable per-element control.

The thesis of this book, and the design thesis of this NPU, is that *the first class of operations
(matmul) is so dominant that dedicated hardwired silicon is justified*, while *the rest of the
operations are too varied and too algorithm-dependent to hardwire* — they require a
general-purpose programmable compute engine. The NPU architecture provides both.

---

## 1.2 Why Not a CPU? Why Not a GPU?

Before introducing the NPU's architecture, it is worth understanding why the two most common
general-purpose compute platforms — the CPU and the GPU — are suboptimal for embedded neural
network inference.

### 1.2.1 The CPU's Limitation: Scalar Execution

A modern out-of-order CPU is an extraordinary engineering achievement. Its branch predictor
anticipates control flow with uncanny accuracy. Its out-of-order execution engine extracts
instruction-level parallelism from a sequential program. Its prefetcher hides DRAM latency by
speculatively loading cache lines before they are needed.

But all of this sophistication is directed at a single thread of scalar execution. A CPU's SIMD
units (SSE, AVX, NEON) add vector width, but they are wide datapaths bolted onto a scalar
architecture. The memory system is designed around the assumption of *temporal and spatial
locality* — the cache hierarchy works when a program accesses the same memory addresses
repeatedly, in predictable patterns. A matrix multiplication over `[4096, 4096]` matrices moves
gigabytes of data, far exceeding any cache. A CPU's area and power budget goes heavily to control
logic — the branch predictor, the reorder buffer, the out-of-order execution window — rather
than to raw multiply-accumulate hardware.

For neural network inference, the CPU's properties are mismatched to the workload:

- The workload has trivial control flow (no branch mispredictions to exploit speculation for)
  but enormous arithmetic density.
- The workload's memory access patterns are regular and predictable (strided reads across matrix
  rows and columns), so the cache hierarchy helps some, but the working sets are large.
- The CPU's wide out-of-order machinery is mostly idle, since the bottleneck is the
  multiply-accumulate units and the memory bandwidth, not instruction scheduling.

The CPU achieves low efficiency (useful operations per joule) on this workload because most of
its power is spent on control infrastructure that does not contribute to the computation.

### 1.2.2 The GPU's Limitation: Power Consumption at Embedded Scale

A GPU is, in many respects, the right architecture for neural network inference. It has thousands
of floating-point multiply-accumulate units, it is designed from the ground up for high
memory bandwidth, and its SIMT (single instruction, multiple threads) execution model maps
naturally onto the data-parallel structure of matrix operations and convolutions.

At datacenter scale, GPUs are indeed the dominant inference platform. But at embedded scale,
the GPU's power consumption is its fatal flaw. A mid-range data-center GPU draws 300 to 700
watts under inference load. Even a mobile GPU (integrated in an SoC) draws several watts for
sustained inference — too much for always-on camera processing in a wearable or for a
battery-powered IoT device.

The GPU's high power comes partly from the overhead of its general-purpose programmability: the
full virtual memory system, the complex thread scheduler, the large register file designed to
hide memory latency by context-switching thousands of threads. These are reasonable costs when
the GPU is running arbitrary compute workloads with complex control flow and irregular memory
access patterns. They are wasteful when the workload is a known set of tensor operations with
highly regular structure.

### 1.2.3 The NPU's Sweet Spot

A Neural Processing Unit (NPU), as the term is used in this book, occupies the design space
between the CPU and the GPU. It is:

- **More specialized than a GPU.** An NPU is purpose-built for tensor computation. It does not
  need a full virtual memory system, a complex thread scheduler, or general-purpose
  programmability across arbitrary workloads.
- **More programmable than a fixed-function accelerator.** Unlike a dedicated convolution
  engine or a hardwired transformer block, a programmable NPU can execute new operations as
  they emerge from research, without silicon respins.
- **More power-efficient than either.** By stripping away the general-purpose overhead and
  sizing the hardware to the actual workload characteristics (e.g., the specific vector widths
  used in typical network layers, the typical tile sizes for matmul), an NPU achieves far
  higher operations-per-watt than a CPU or GPU on the inference workload.

The design space for NPUs is large. Different products make different tradeoffs. The NPU
described in this book makes one particular set of choices, which will be explained in detail
throughout. The goal is not to describe a specific commercial product but to develop a concrete,
fully-specified ISA and simulator that illustrates the key principles.

---

## 1.3 The Three-Layer Architecture

The central organizing principle of this NPU is that its compute hardware is divided into three
distinct engines, each with its own instruction set, its own register file, and its own
execution semantics. These three layers are not redundant alternatives — they are
*complementary*, each optimized for a different class of operation that appears in neural
network workloads.

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                              NPU CHIP                                        │
│                                                                              │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────────────┐  │
│  │                 │    │                 │    │                         │  │
│  │  Control Core   │    │  Tensor Engine  │    │     Vector Core         │  │
│  │   (Layer 1)     │    │   (Layer 2)     │    │     (Layer 3)           │  │
│  │                 │    │                 │    │                         │  │
│  │  x0..x31        │    │  t0..t7         │    │  v0..v31                │  │
│  │  (scalar regs)  │    │  (tile regs)    │    │  s0..s31  m0..m7  vl   │  │
│  │                 │    │                 │    │                         │  │
│  │  C_ADD          │    │  T_MATMUL       │    │  V_VADD                 │  │
│  │  C_LW / C_SW    │    │  T_MMA          │    │  V_VEXP / V_VLOG        │  │
│  │  C_BEQ / C_BNE  │    │  T_LOAD/STORE   │    │  V_VREDSUM / V_VREDMAX  │  │
│  │  C_KLAUNCH ─────┼────┼─────────────────┼───▶│  V_VMASK_SET            │  │
│  │  C_KWAIT        │    │  T_DMA_IN/OUT   │    │  V_KEXIT                │  │
│  │  C_HALT         │    │  T_FENCE        │    │                         │  │
│  │                 │    │                 │    │                         │  │
│  └─────────────────┘    └─────────────────┘    └─────────────────────────┘  │
│          │                                               ▲                   │
│          │              C_KLAUNCH(kid, grid, args)       │                   │
│          └───────────────────────────────────────────────┘                   │
│                                                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                         MEMORY SYSTEM                                │   │
│  │                                                                      │   │
│  │   Off-chip DRAM (16 MB)               On-chip SRAM Scratchpad       │   │
│  │   addr: 0x0000_0000..0x00FF_FFFF      addr: 0x8000_0000+offset      │   │
│  │   Latency: ~100 cycles                Latency: ~1 cycle             │   │
│  │                                                                      │   │
│  │              ← T_DMA_IN / T_DMA_OUT bridge both →                   │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────────────────────┘
```

### 1.3.1 Layer 1: The Control Core

The control core is the orchestration engine. It is a scalar processor — one operation at a
time, one value at a time — with 32 general-purpose integer registers (`x0` through `x31`,
with `x0` hardwired to zero in the RISC-V tradition) and an instruction set deliberately
modeled after the RISC-V base ISA.

The control core is responsible for:

- **Host-side computation.** Setting up loop bounds, computing tile indices and memory
  offsets, computing addresses to pass to the tensor engine or vector core.
- **Sequencing.** Deciding which operation to run next: "run this matmul, then launch this
  softmax kernel, then store the result."
- **Launching vector kernels.** The `C_KLAUNCH` instruction directs the control core to
  dispatch a vector kernel onto the vector core with a specified grid size (number of
  parallel instances) and an argument pointer.
- **Synchronization.** The `C_KWAIT` instruction stalls the control core until all
  previously launched vector kernels have completed, providing the ordering guarantee that
  makes the separation of control and compute safe.

Control core instructions follow standard RISC-V-like encoding conventions:

| Category | Opcodes | Purpose |
|---|---|---|
| Arithmetic | `C_ADD`, `C_SUB`, `C_ADDI`, `C_MUL` | Integer arithmetic for address and loop computation |
| Logic | `C_AND`, `C_OR`, `C_XOR`, `C_SLL`, `C_SRL`, `C_SLT` | Bitwise operations and shift |
| Immediate load | `C_LI` | Load a constant into a register |
| Memory | `C_LW`, `C_LH`, `C_LB`, `C_SW`, `C_SH`, `C_SB` | Scalar load/store (word, halfword, byte) |
| Branches | `C_BEQ`, `C_BNE`, `C_BLT`, `C_BGE` | Conditional branches for loops |
| Jump | `C_JAL`, `C_JALR` | Unconditional jump and link (subroutine call/return) |
| Launch | `C_KLAUNCH` | Launch a vector kernel onto the vector core |
| Sync | `C_KWAIT` | Wait for in-flight vector kernels to finish |
| Halt | `C_HALT` | Stop the control core program |

The control core is *not* where tensor computation happens. It does not hold any matrix data.
It holds integers and addresses. Its job is to be the director: "now load this tile, now
multiply these tiles, now launch a kernel on this memory region with this many threads."

### 1.3.2 Layer 2: The Tensor Engine

The tensor engine is the hardwired fast path. Its reason for existing is simple: matrix
multiplication between large tiles is the single most common operation in neural network
inference, and implementing it from elementwise vector primitives is wasteful when dedicated
matrix multiply-accumulate arrays can do it far more efficiently.

The tensor engine has eight *tile registers* (`t0` through `t7`), each capable of holding a
`MAX_TILE x MAX_TILE` (64×64) array of floats. This is hardware that holds a 2D matrix in
registers rather than in memory. An instruction like `T_MATMUL t2, t0, t1` multiplies the
entire matrix in `t0` by the entire matrix in `t1` and writes the result to `t2` — one
instruction doing millions of multiply-accumulate operations.

The tensor engine instruction set covers:

| Category | Opcodes | Purpose |
|---|---|---|
| Configuration | `T_CFG_SHAPE`, `T_CFG_DTYPE`, `T_CFG_STRIDE`, `T_CFG_MODE`, `T_CFG_RND` | Set tile dimensions, data types, memory layout |
| Compute | `T_MATMUL`, `T_MMA`, `T_DOT`, `T_ACC_CLR` | Dense matrix multiply (overwrite and accumulate forms) |
| Data movement | `T_LOAD`, `T_STORE`, `T_LOADW`, `T_DMA_IN`, `T_DMA_OUT`, `T_BCAST` | Load/store tiles, DMA between DRAM and SRAM |
| Activations | `T_RELU`, `T_QUANT`, `T_DEQUANT`, `T_SCALE` | Hardwired common-case activations |
| Synchronization | `T_FENCE`, `T_WAIT`, `T_BARRIER`, `T_STATUS` | Ordering between DMA, compute, and stores |

The crucial distinction between `T_MATMUL` and `T_MMA` is accumulation. `T_MATMUL` overwrites
the destination tile: `d = a * b`. `T_MMA` accumulates: `d = d + a * b`. This is essential
for tiled matmul over large matrices, where the output tile is the sum of partial products
over K-blocks: you clear the accumulator once, then call `T_MMA` for each K-block, then
store.

The tensor engine exists *alongside* the vector core, not instead of it. Dense matmul goes
through the tensor engine because it is worth the silicon cost of a dedicated array. Everything
else — softmax, layernorm, activations, sparse ops — runs on the vector core.

### 1.3.3 Layer 3: The Vector Core

The vector core is the subject of this book. It is the engine that makes this NPU *fully
programmable* — meaning it can execute any algorithm expressed as a program, not just
operations the hardware designer anticipated.

The vector core is a SIMD engine: it processes N elements simultaneously, where N is the
current vector length (`vl`). Each instruction operates across a full vector register, applying
the same operation to every lane in parallel. But unlike a fixed-function SIMD unit, the
vector core has:

- **A full register file:** 32 vector registers (`v0`–`v31`), 32 scalar registers (`s0`–`s31`),
  and 8 mask registers (`m0`–`m7`).
- **Engine-side control flow:** branches (`V_BRZ`, `V_BRNZ`, `V_BR`), hardware loops
  (`V_LOOP`), and a kernel exit signal (`V_KEXIT`).
- **Per-lane predication:** any arithmetic instruction can be qualified with `@mask`, so that
  the operation only applies to lanes where the mask bit is 1 and leaves other lanes unchanged.
- **Transcendental operations:** `V_VEXP`, `V_VLOG`, `V_VSQRT`, `V_VRSQRT`, `V_VRECIP` — the
  building blocks of softmax, layernorm, GELU, and sigmoid.
- **Reductions:** `V_VREDSUM`, `V_VREDMAX`, `V_VREDMIN` collapse an entire vector to a scalar.
- **Indirect memory access:** `V_VGATHER` and `V_VSCATTER` support data-dependent addressing.
- **A kernel program model:** kernels are programs, loaded into a kernel table and launched
  with a grid size. Each grid instance gets its own register state.

The vector core is described in detail throughout this book. Chapters 2 through 7 build up
the complete picture of its register file, memory model, instruction groups, and execution
model. The remainder of this chapter provides the conceptual foundation: why the vector core
exists, how it fits into the system, and what you will learn.

---

## 1.4 Why the Vector Core Exists: The Gap the Tensor Engine Cannot Fill

The tensor engine is efficient and specialized. Precisely because of its specialization, it
can only do what it was designed to do. Let us be concrete about what falls outside its
repertoire.

### 1.4.1 Softmax Is Not a Single Instruction

Softmax is the canonical example of an operation that is algorithmically necessary, ubiquitous
in transformer models, and yet inherently composite. There is no hardwired "softmax instruction"
on this NPU (or on most real NPUs, for that matter), because softmax is not a simple
elementwise function — it is a *reduction followed by elementwise operations followed by
another reduction followed by another elementwise operation*.

The numerically stable form of softmax for a vector `x` of length N:

```
max_x   = max(x[0], x[1], ..., x[N-1])         -- cross-lane reduction
shifted = x[i] - max_x                           -- elementwise subtract
exps    = exp(shifted[i])                        -- elementwise exponential
sum_exp = sum(exps[0], exps[1], ..., exps[N-1]) -- cross-lane reduction
output  = exps[i] / sum_exp                      -- elementwise divide
```

The cross-lane reductions (`VREDMAX`, `VREDSUM`) produce a scalar result. The elementwise
operations (`VSUB`, `VEXP`, `VMUL`) work on the full vector. These operations interleave and
depend on each other's results. A tensor engine tile register cannot hold the intermediate
scalar results, and a matmul array cannot compute an exponential.

On the vector core, softmax is expressed naturally:

```asm
VSETVL  s8, s4          ; set VL = N (the row length)
VLOAD   v0, 0(s6)       ; load input row into v0
VREDMAX s9, v0          ; s9 = max(v0)  -- reduction to scalar
VBCAST  v1, s9          ; v1 = broadcast max to all lanes
VSUB    v0, v0, v1      ; v0 = v0 - max (numerically stable shift)
VEXP    v0, v0          ; v0 = exp(v0)  -- per-lane exponential
VREDSUM s10, v0         ; s10 = sum(v0) -- reduction to scalar
SRECIP  s11, s10        ; s11 = 1/sum   -- scalar reciprocal
VBCAST  v1, s11         ; v1 = broadcast 1/sum to all lanes
VMUL    v0, v0, v1      ; v0 = v0 * (1/sum) -- final normalization
VSTORE  v0, 0(s7)       ; write output row
KEXIT
```

This is not a workaround or a hack — this *is* softmax. The vector core gives you exactly the
right primitives to express it efficiently.

### 1.4.2 LayerNorm Requires Two Reductions

LayerNorm computes the mean and variance of a vector, then normalizes it. The mean requires
summing all elements and dividing by the count. The variance requires summing the squared
deviations from the mean. Then the normalization uses a reciprocal square root. This involves
three passes over the data (one for mean, one for variance, one for the final scale), with
inter-pass data dependencies on the scalar results of the reductions.

A tensor engine `T_MMA` cannot compute a mean. A fixed-function unit that can only do matrix
multiply has no representation of "subtract the mean from every element." The vector core's
reductions, broadcast, and elementwise arithmetic make layernorm a straightforward multi-step
kernel.

### 1.4.3 Activations: GELU and SiLU

GELU (Gaussian Error Linear Unit) is the activation function of choice in modern transformers.
Its approximate form is:

```
GELU(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
```

This involves a cube (`x^3 = x * x * x`), a multiply by a constant, an addition, a
multiplication by `sqrt(2/pi)`, a tanh (which is itself expressed as a rational or exponential
approximation), and several more multiplies and additions. All per-element.

SiLU (Sigmoid Linear Unit) is `x * sigmoid(x) = x / (1 + exp(-x))`, requiring a per-element
exponential.

Neither of these fits in a tensor engine. The tensor engine's `T_RELU` instruction is a
convenience — it is hardwired because ReLU is so simple (a single elementwise max with zero)
and so common that it is worth encoding. But GELU and SiLU are not. On a fixed-function
accelerator, adding a new activation function may require a new chip revision. On this NPU,
you write a new vector kernel.

### 1.4.4 Sparse and Mixture-of-Experts Operations

Mixture-of-experts (MoE) models route each input token to a subset of "expert" feed-forward
networks. The routing decision is data-dependent: the softmax of the router's output determines
which experts are active. Once the routing is decided, you need to *gather* the tokens destined
for each expert (indirect indexing by the routing decision), run the expert's linear layer on
that subset, and *scatter* the results back.

This is `V_VGATHER` and `V_VSCATTER`. The address of each load is not a fixed stride — it
depends on the routing table produced by a previous computation. A tensor engine works with
tile registers at known addresses; it has no mechanism for data-dependent addressing. The
vector core's gather and scatter instructions handle this naturally.

### 1.4.5 The General Principle

The pattern is consistent: any operation that combines a cross-lane reduction with
per-lane elementwise math, or that involves data-dependent control flow or addressing, falls
outside what a hardwired tensor engine can express. The vector core fills this gap by providing:

1. Cross-lane reductions to scalars (`V_VREDSUM`, `V_VREDMAX`, `V_VREDMIN`)
2. Broadcast from scalar to all lanes (`V_VBCAST`)
3. Transcendental elementwise operations (`V_VEXP`, `V_VLOG`, `V_VSQRT`, `V_VRSQRT`, `V_VRECIP`)
4. Elementwise arithmetic across the full register width (`V_VADD`, `V_VSUB`, `V_VMUL`,
   `V_VDIV`, `V_VFMA`, `V_VMAX`, `V_VMIN`, `V_VABS`, `V_VNEG`)
5. Per-lane predication and conditional selection (`V_VCMP`, `V_VSEL`, `@mask` prefix)
6. Data-dependent memory access (`V_VGATHER`, `V_VSCATTER`)
7. Engine-side scalar computation for addresses and loop control (`V_SLI`, `V_SADD`,
   `V_SMUL`, `V_SLW`, etc.)
8. A kernel program model with independent state per grid instance

Together, these primitives allow any algorithm expressible as data-parallel computation over
one-dimensional vectors to be efficiently implemented on the vector core.

---

## 1.5 The Execution Flow: From Control Core to Vector Core and Back

Understanding how a computation actually flows through the three layers of the NPU is essential
for writing vector core kernels. The execution model is straightforward but has specific rules
about state, ordering, and synchronization.

### 1.5.1 Host-Side Setup

Before any computation runs on the NPU, the host CPU (or host-side code) sets up the NPU's
memory. It writes input data to NPU DRAM via `npu_write_f32` or `npu_write_i32`. It sets up
an *argument block* — a small struct in NPU memory containing pointers to input and output
tensors, shapes, and other parameters. The address of this argument block will be passed to
the vector kernel at launch time.

Then the host registers a vector kernel by calling `npu_register_kernel`, which returns a
kernel ID (an integer). This ID is what the control core program uses in `C_KLAUNCH`.

### 1.5.2 Control Core Program Execution

The control core runs its program sequentially, instruction by instruction. A typical control
core program for launching a vector kernel looks like this:

```asm
; Control core program
LI   x10, 64         ; x10 = grid size (number of vector instances to launch)
LI   x11, 0x1000     ; x11 = address of the argument block in NPU memory
KLAUNCH 0, x10, x11  ; launch kernel 0, 64 instances, args at 0x1000
KWAIT                ; wait for all 64 instances to complete
HALT                 ; done
```

The `C_KLAUNCH` instruction has three parameters: the kernel ID, a register holding the grid
size, and a register holding the argument block address. The grid size determines how many
independent instances of the vector kernel will run. Each instance gets a unique *instance ID*
from 0 to grid-1.

### 1.5.3 Vector Kernel Execution

When `C_KLAUNCH` fires, the NPU dispatches `grid` instances of the named kernel. In the
simulator, these run sequentially on the same host thread (multi-core is modeled as a
max-over-cores cycle count, not real parallelism — see Chapter 8 for the cycle model). In
hardware, multiple vector cores would run instances in parallel.

```
C_KLAUNCH fires with grid=64:
  Instance 0 runs with s0=0, s1=args_addr
  Instance 1 runs with s0=1, s1=args_addr
  Instance 2 runs with s0=2, s1=args_addr
  ...
  Instance 63 runs with s0=63, s1=args_addr
```

Each instance starts with a *fresh register file*: all vector registers `v0`–`v31` are zero,
all scalar registers `s0`–`s31` are zero (with the critical exception that `s0` is set to the
instance ID and `s1` is set to the argument block address), all mask registers `m0`–`m7` are
all-ones (all lanes active), and `vl` is 0 (to be set by the kernel's first `V_VSETVL`).

The kernel uses `s0` to determine *which part of the problem* this instance is responsible for.
For example, in the softmax demo, the kernel handles one row of the input matrix. `s0` is the
row index. The kernel reads the row start address from the argument block (`s1` base + offset),
multiplies the row index by the row byte stride, adds them to get the actual DRAM address of
its row, sets `vl` to the row length, loads the row, and runs the algorithm.

### 1.5.4 The Launch Convention in Detail

The launch convention is the contract between the control core and the vector kernel. It must
be respected by every kernel:

- On entry, `s0` holds the instance ID (an integer from 0 to grid-1).
- On entry, `s1` holds the address of the argument block in NPU memory.
- The kernel must read all input pointers and parameters from the argument block via `V_SLW`
  (scalar load word) or `V_SLF` (scalar load float) from addresses derived from `s1`.
- The kernel signals completion with `V_KEXIT`.
- All other state (`v0`–`v31`, `s2`–`s31`, `m0`–`m7`, `vl`) is uninitialized at entry (but
  reset to zero for v and s, all-ones for m — the hardware guarantees this).

This is shown in a typical kernel prologue:

```asm
; Kernel entry: s0 = instance_id, s1 = args_ptr
; Args layout: [0] = input_addr, [4] = output_addr, [8] = N (element count)
SLW s2, 0(s1)       ; s2 = input base address  (load word from args[0])
SLW s3, 4(s1)       ; s3 = output base address (load word from args[1])
SLW s4, 8(s1)       ; s4 = N, the element count

; Compute this instance's slice of the input
SMUL  s5, s0, s4    ; s5 = instance_id * N (element offset)
SMULI s5, s5, 4     ; s5 = s5 * 4 (byte offset, since each float is 4 bytes)
SADD  s6, s2, s5    ; s6 = input_addr + byte_offset (this instance's input row)
SADD  s7, s3, s5    ; s7 = output_addr + byte_offset (this instance's output row)

VSETVL s8, s4       ; set vl = N
; ... body of computation ...
KEXIT
```

### 1.5.5 Synchronization: C_KWAIT

The control core's `C_KWAIT` instruction is a synchronization barrier. It blocks the control
core from advancing until all instances from the most recent `C_KLAUNCH` have executed
`V_KEXIT`. In the simulator, because instances are run synchronously, `C_KWAIT` is essentially
a no-op (all instances have already completed by the time `C_KLAUNCH` returns). In hardware,
`C_KWAIT` would stall the control pipeline until the vector engine signals completion.

This synchronization model is important for correctness: a control core that runs a
`C_KLAUNCH` and immediately reads the output buffer (without `C_KWAIT`) might see partial or
stale results if the hardware executes instances in parallel.

The full execution flow, from the host's perspective:

```
┌─────────────────────────────────────────────────────────────────────┐
│  HOST CPU                                                           │
│  1. Write input data to NPU DRAM                                    │
│  2. Write argument block to NPU DRAM                               │
│  3. Register kernel (get kid)                                       │
│  4. Run control-core program                                        │
└──────────────────────────┬──────────────────────────────────────────┘
                           │ npu_run()
                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│  CONTROL CORE (sequential)                                          │
│  C_LI x10, grid_size                                                │
│  C_LI x11, args_addr                                               │
│  C_KLAUNCH kid, x10, x11 ──────────────────────────────────────┐  │
│  C_KWAIT ◀─────────────────────────────────────────────────┐   │  │
│  C_HALT                                                    │   │  │
└────────────────────────────────────────────────────────────│───│──┘
                                                             │   │
                           ┌─────────────────────────────────│───┘
                           │  VECTOR CORE (parallel)          │
                           ▼                                  │
        ┌──────────────────────────────────────────────────┐  │
        │  Instance 0    Instance 1    ...    Instance N-1  │  │
        │  s0=0, s1=a    s0=1, s1=a         s0=N-1, s1=a  │  │
        │  run kernel    run kernel         run kernel      │  │
        │  V_KEXIT       V_KEXIT            V_KEXIT ────────┼──┘
        └──────────────────────────────────────────────────┘

        Then:
┌─────────────────────────────────────────────────────────────────────┐
│  HOST CPU                                                           │
│  5. Read output data from NPU DRAM via npu_read_f32                │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 1.6 Memory Architecture

The NPU has two distinct memory regions, unified in a single 32-bit address space through an
address-tagging scheme.

### 1.6.1 Off-Chip DRAM

The NPU's main memory is 16 MB of off-chip DRAM, addressed from `0x0000_0000` to
`0x00FF_FFFF`. DRAM is large (16 MB can hold many matrices), but access latency is
approximately 100 cycles — every vector load or store that hits DRAM pays this cost.

For a vector load of 256 float32 elements (1 KB), reading from DRAM costs 100 cycles of
latency plus the vectorized load time. At high vector lengths, DRAM latency dominates.

### 1.6.2 On-Chip SRAM Scratchpad

The on-chip SRAM scratchpad is 256 KB of fast memory, located physically on the NPU chip
alongside the compute engines. Its access latency is approximately 1 cycle. Addresses in the
scratchpad have bit 31 set: the base address is `0x8000_0000`, and the scratchpad occupies
offsets `0x0000` through `0x3FFFF`.

The scratchpad is the NPU's critical performance resource. The standard usage pattern is:

1. DMA input data from DRAM to SRAM before computation begins (using `T_DMA_IN`).
2. Run the vector kernel with all loads and stores hitting SRAM.
3. DMA results from SRAM back to DRAM after computation (using `T_DMA_OUT`).

This staging pattern transforms 100-cycle DRAM loads into 1-cycle SRAM loads, dramatically
improving compute throughput. The cycle model in the simulator captures this: running a VADD
kernel with operands in DRAM versus SRAM shows a factor-of-100 difference in vector-core
cycles (the DMA cost is paid in tensor-engine cycles, which can overlap with the vector
pipeline in hardware).

### 1.6.3 Address Resolution in the Vector Core

The vector core resolves memory addresses by checking bit 31. If bit 31 is set, the access
is routed to the SRAM scratchpad (offset = address `& 0x7FFF_FFFF`). If bit 31 is clear, the
access goes to DRAM. This happens transparently in every `V_VLOAD`, `V_VSTORE`, `V_SLW`,
`V_SLF`, `V_VGATHER`, and `V_VSCATTER` instruction.

The programmer is responsible for not overrunning the 256 KB scratchpad. Accessing
`SCRATCH_BASE + offset` where `offset >= 0x40000` (256 KB) triggers a fault in the simulator.

---

## 1.7 What Makes This NPU Fully Programmable

The term "fully programmable" has a precise meaning in this context, defined by the ISA
reference (`fully_prog_npu.md`). A fully programmable engine must have all of the following
properties together — any one alone is not sufficient:

**1. General compute primitives.** The instruction set must include arbitrary elementwise math,
transcendentals, comparisons, and type conversions. Operations not listed in the hardware's
menu are built by composing these primitives, not by waiting for the next chip revision.

**2. Engine-side control flow and predication.** The vector engine must have its own program
counter, branches, and per-lane masking. Without these, the engine can only run a single
fixed sequence of operations per kernel launch — it cannot respond to data-dependent
conditions discovered mid-computation.

**3. A free register and memory model.** General vector registers (not just fixed tile slots)
plus addressable scratchpad memory with gather/scatter. Fixed-function engines have fixed
data flows between fixed buffers; a programmable engine lets software choose its own data
routing.

**4. A kernel program model.** The engine fetches and executes its own instruction stream.
Software loads kernels as programs. Hardware runs them. This is distinct from an engine where
the control core issues every operation one by one — which would serialize the computation
and eliminate the throughput benefit of the vector engine.

**5. A compiler or toolchain** (practically essential). An LLVM backend, or a kernel
language (like CUDA for GPUs, or a future NPU-specific language) that can translate arbitrary
algorithms into the engine's instruction stream. Without a compiler, the hardware is
programmable only in theory — nobody will write production kernels by hand in a binary ISA.
The LLVM integration path for this NPU is documented in `llvm_lowering.md`.

This NPU satisfies all five properties. The vector core ISA, described in detail in the
remainder of this book, is the concrete realization of these requirements.

---

## 1.8 What You Will Learn in This Book

This book is a complete technical reference and programming guide for the NPU vector core ISA.
It is organized to build understanding progressively, from the register file through every
instruction group to complete algorithm examples.

**Chapter 2: The Register File.** The four types of state a kernel can access — vector
registers `v0`–`v31`, scalar registers `s0`–`s31`, mask registers `m0`–`m7`, and the vector
length register `vl`. How they are organized, what they hold, how they interact, and how they
are initialized at kernel entry.

**Chapter 3: Memory Instructions.** How a vector kernel loads and stores data. The vector
length model and how it governs how many elements are moved. VLOAD and VSTORE for sequential
access. VGATHER and VSCATTER for indirect access. The address calculation model (base register
plus immediate offset). VMOV and VBCAST for register-to-register data movement.

**Chapter 4: Scalar Engine Instructions.** The scalar instructions that live inside the vector
core: `V_SLI`, `V_SFI`, `V_SADD`, `V_SSUB`, `V_SMUL`, `V_SLW`, `V_SLF`, `V_SRECIP`, and
`V_KID`. Why scalar instructions are needed inside a vector engine, and how they interact with
the vector pipeline.

**Chapter 5: Compute Instructions.** Every elementwise arithmetic and math instruction:
`V_VADD`, `V_VSUB`, `V_VMUL`, `V_VDIV`, `V_VMAX`, `V_VMIN`, `V_VFMA`, `V_VEXP`, `V_VLOG`,
`V_VSQRT`, `V_VRSQRT`, `V_VRECIP`, `V_VABS`, `V_VNEG`, `V_VCMP`, `V_VSEL`, `V_VCVT`.
The cost model for transcendentals versus elementwise operations.

**Chapter 6: Reduction Instructions.** The instructions that collapse a vector to a scalar:
`V_VREDSUM`, `V_VREDMAX`, `V_VREDMIN`, `V_VREDAND`, `V_VREDOR`, and `V_VSCAN` (prefix sum).
The tree-reduction structure and its cycle cost. Why reductions are the key to softmax,
layernorm, and attention.

**Chapter 7: Predication and Control Flow.** Mask registers and how they are created
(`V_VMASK_SET`, `V_VMASK_AND`, `V_VMASK_OR`, `V_VMASK_NOT`) and applied (`@mask` prefix on
any arithmetic instruction). Engine-side branches (`V_BR`, `V_BRZ`, `V_BRNZ`) and the hardware
loop (`V_LOOP`). VSETVL and the variable-length vector model.

**Chapter 8: The Execution Model and Cycle Estimates.** How the simulator models instruction
latency, how multi-core parallelism is modeled (max-over-cores), and how to reason about
performance from cycle counts.

**Chapter 9: Complete Kernel Examples.** Worked, self-checking implementations of vadd,
softmax, layernorm, predicated ReLU+bias, and attention, with full instruction listings and
discussion of the design choices at every step.

---

## 1.8 Your First Vector Kernel — End-to-End Walkthrough

Before diving into individual instruction groups, here is a complete, working example that
ties the entire architecture together. We will write a kernel that adds two arrays of 16
floats and stores the result. This is the "Hello World" of NPU programming.

### The Goal

```
C[i] = A[i] + B[i]   for i = 0..15
```

### Step 1 — Memory Layout

We place all three arrays in the SRAM scratchpad (bit-31 set → 1-cycle access):

```
SRAM layout (starts at 0x80000000):
┌─────────────────────────────────────────┐
│ 0x80000000  A[0..15]   (64 bytes, input) │
│ 0x80000040  B[0..15]   (64 bytes, input) │
│ 0x80000080  C[0..15]   (64 bytes, output)│
└─────────────────────────────────────────┘
```

The host (control core) copies A and B from DRAM to these SRAM addresses using DMA before
launching the kernel. The args pointer (`s1` on entry) points to `0x80000000`.

### Step 2 — The Vector Kernel

```c
#include "npu.h"

static const vinst vadd_kernel[] = {
    /*
     * Instruction 0: set vector length to 16
     * vl = 16 means every subsequent vector op processes lanes 0..15
     * s9 gets the old vl (we don't use it, rd=9 is a scratch reg)
     */
    { .op = V_VSETVL, .rd = 9, .imm = 16 },

    /*
     * Instruction 1: load A[0..15] from SRAM into v0
     * s1 = args base = 0x80000000
     * imm = 0 → address = s1 + 0 = 0x80000000
     * Loads 16 × 4 = 64 bytes into v0[0..15]
     */
    { .op = V_VLOAD, .vd = 0, .rs = 1, .imm = 0, .use_base = 1 },

    /*
     * Instruction 2: load B[0..15] from SRAM into v1
     * imm = 64 → address = s1 + 64 = 0x80000040
     */
    { .op = V_VLOAD, .vd = 1, .rs = 1, .imm = 64, .use_base = 1 },

    /*
     * Instruction 3: add element-wise → v2[l] = v0[l] + v1[l]
     * All 16 additions happen simultaneously.
     */
    { .op = V_VADD, .vd = 2, .vs1 = 0, .vs2 = 1 },

    /*
     * Instruction 4: store result v2 → C[0..15] in SRAM
     * imm = 128 → address = s1 + 128 = 0x80000080
     */
    { .op = V_VSTORE, .vs = 2, .rs = 1, .imm = 128, .use_base = 1 },

    /*
     * Instruction 5: terminate this grid instance
     * The control core's C_KWAIT unblocks once all instances hit KEXIT.
     */
    { .op = V_KEXIT },
};
```

Only **6 instructions** — and all the heavy lifting happens in a single `V_VADD`.

### Step 3 — Register Trace (single instance, id=0)

| After instruction | v0 | v1 | v2 | s1 | vl |
|---|---|---|---|---|---|
| (entry) | `[0,0,0,…]` | `[0,0,0,…]` | `[0,0,0,…]` | `0x80000000` | 0 |
| `V_VSETVL` | unchanged | unchanged | unchanged | unchanged | **16** |
| `V_VLOAD v0` | **`[A0,A1,…,A15]`** | unchanged | unchanged | unchanged | 16 |
| `V_VLOAD v1` | unchanged | **`[B0,B1,…,B15]`** | unchanged | unchanged | 16 |
| `V_VADD v2` | unchanged | unchanged | **`[A0+B0,…,A15+B15]`** | unchanged | 16 |
| `V_VSTORE v2` | unchanged | unchanged | unchanged | unchanged | 16 |
| `V_KEXIT` | — | — | — | — | — |

### Step 4 — Launching from the Control Core

```c
npu_t *n = npu_create();

/* Stage data into SRAM */
float *sram = (float *)(n->scratch);          /* SCRATCH_BASE = 0x80000000 */
memcpy(sram + 0,  A, 16 * sizeof(float));     /* A → 0x80000000 */
memcpy(sram + 16, B, 16 * sizeof(float));     /* B → 0x80000040 */

/* Register the kernel, get a kernel id */
int kid = npu_register_kernel(n, vadd_kernel,
                               sizeof(vadd_kernel) / sizeof(vinst));

/* Build control program */
cinst ctrl[] = {
    /*
     * KLAUNCH: run vadd_kernel with 1 grid instance
     *   imm = kernel id
     *   rd  = grid size (1 instance)
     *   rs1 = args pointer register (x1 holds SCRATCH_BASE)
     */
    { .op = C_LI, .rd = 1, .imm = (int64_t)SCRATCH_BASE }, /* x1 = SRAM base */
    { .op = C_KLAUNCH, .imm = kid, .rd = 1 /* grid=1 */, .rs1 = 1 },
    { .op = C_KWAIT  },
    { .op = C_HALT   },
};

npu_run(n, ctrl, sizeof(ctrl) / sizeof(cinst));

/* Read result back */
float *C_out = sram + 32;                     /* C at 0x80000080 */
```

### Step 5 — What Actually Runs

```
Control core:  C_LI x1, 0x80000000
               C_KLAUNCH kid, grid=1, args=x1
               C_KWAIT  ←── blocks here

Vector core:   [instance 0]
               V_VSETVL 16
               V_VLOAD  v0 ← mem[0x80000000..0x8000003F]  (A)
               V_VLOAD  v1 ← mem[0x80000040..0x8000007F]  (B)
               V_VADD   v2 = v0 + v1
               V_VSTORE v2 → mem[0x80000080..0x800000BF]  (C)
               V_KEXIT

Control core:  ← unblocks
               C_HALT
```

Total: **4 control instructions + 6 vector instructions = 10 instructions** to do a 16-wide
vector add. The vector add itself (the `V_VADD`) is one instruction regardless of whether
you're adding 1 float or 256 floats.

This pattern — VSETVL, VLOAD inputs, compute, VSTORE output, KEXIT — repeats in every
vector kernel you will ever write. The chapters that follow fill in every part of it.

---

## 1.9 Chapter Summary

This chapter has established the context for everything that follows:

- Neural network inference involves a specific set of operations — matmul, softmax, layernorm,
  activations, and sparse ops — that have very different computational structures.
- CPUs are scalar and power-inefficient for this workload. GPUs are powerful but too
  power-hungry for embedded deployment.
- The NPU's three-layer architecture dedicates specialized hardware to matmul (the tensor
  engine) while providing general-purpose programmability for everything else (the vector core).
- The control core orchestrates the two compute engines, handles addressing and loop control,
  and launches vector kernels via `C_KLAUNCH`.
- The vector core is a programmable SIMD engine with its own program counter, register file,
  and instruction set. It runs kernels as programs, with one instance per grid point.
- The memory system provides 16 MB of off-chip DRAM (slow) and 256 KB of on-chip SRAM
  (fast), unified in a single address space via bit-31 tagging.
- The NPU is fully programmable because it satisfies all five conditions: general primitives,
  engine-side control flow, free register model, kernel program model, and compiler support.

With this foundation established, Chapter 2 dives into the heart of the vector core: the
register file.
