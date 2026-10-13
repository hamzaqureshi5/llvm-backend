# Chapter 12 — The NPU Kernel Driver (`npu_kdrv`)

> **Prerequisites:** Chapter 3 (memory model), Chapter 9 (execution model).
> **Source files:** `npu-simulator/src/npu_kdrv.c`, `npu-simulator/include/npu_kdrv.h`

---

## 12.1 What a Kernel Driver Does

In a production Linux system, an NPU kernel driver lives in the kernel (`drivers/npu/`).
User-space programs reach it through:

- `ioctl()` — device control: allocate DRAM, submit commands, query stats
- `mmap()` — map a DMA-coherent command ring into user space so submissions don't
  require a syscall per command

In the simulator, `npu_kdrv` is a plain C library that **models the same interface in user
space**.  The contract is identical to what a real driver would expose; everything above the
driver (the runtime in Chapter 13) never calls `npu_t` directly.

```
┌─────────────────────────────────────────────┐
│  Application                                │
├─────────────────────────────────────────────┤
│  npu_rt  (CUDA-style async runtime)         │  npu_rt.h
├─────────────────────────────────────────────┤
│  npu_kdrv  (kernel driver model)            │  npu_kdrv.h  ← THIS CHAPTER
├─────────────────────────────────────────────┤
│  npu_t  (ISA simulator)                     │  npu.h
├─────────────────────────────────────────────┤
│  "Physical" NPU chip (future hardware)      │
└─────────────────────────────────────────────┘
```

The driver owns three resources:

| Resource | Purpose |
|----------|---------|
| `npu_t *sim` | The simulator — the "hardware" |
| `heap[]` | DRAM allocator metadata |
| `cmdq[]` | Command ring buffer |

---

## 12.2 Driver Handle and Key Constants

```c
/* npu_kdrv.h */
typedef struct npu_kdrv_s *npu_kdrv_t;

kdrv_err_t npu_kdrv_probe (npu_kdrv_t *out);
kdrv_err_t npu_kdrv_remove(npu_kdrv_t  drv);
kdrv_err_t npu_kdrv_reset (npu_kdrv_t  drv);

kdrv_err_t npu_kdrv_alloc(npu_kdrv_t drv, uint32_t *out, size_t bytes);
kdrv_err_t npu_kdrv_free (npu_kdrv_t drv, uint32_t  addr);

kdrv_err_t npu_kdrv_submit(npu_kdrv_t drv, const npu_cmd_t *cmd);
kdrv_err_t npu_kdrv_flush (npu_kdrv_t drv);

kdrv_err_t npu_kdrv_fence_create(npu_kdrv_t drv, int *id_out);
kdrv_err_t npu_kdrv_fence_wait  (npu_kdrv_t drv, int  id);
```

Key constants defined in `npu_kdrv.c`:

```c
#define CMDQ_CAP     4096   /* max pending commands per flush */
#define MAX_FENCES    256   /* fence table entries */
#define MAX_HEAP_BLK 2048   /* heap metadata slots */
#define HEAP_ALIGN     64   /* DRAM allocation granularity (bytes) */
#define HEAP_GUARD     64   /* reserved bytes at DRAM address 0 */
```

Why `HEAP_ALIGN = 64`?
- 64 bytes = one typical cache line
- Aligned DMA transfers are more efficient on real hardware
- Keeps heap metadata simple (sizes always multiples of 64)

Why `HEAP_GUARD = 64`?
- DRAM address 0 is the "null pointer" for device memory
- If a bug passes address 0 to a DMA operation, we hit the guard region
  (never allocated), producing garbage data rather than silently corrupting
  another allocation

---

## 12.3 Internal Driver Structure

```c
typedef struct {
    uint32_t base;   /* DRAM byte address of this block */
    uint32_t size;   /* size in bytes — always a multiple of HEAP_ALIGN */
    int      used;   /* 1 = allocated, 0 = free */
} heap_blk_t;

struct npu_kdrv_s {
    npu_t      *sim;                  /* simulator handle ("hardware") */
    int         clk_enabled;

    /* DRAM heap */
    heap_blk_t  heap[MAX_HEAP_BLK];  /* metadata; actual data lives in sim->mem */
    int         nblocks;              /* number of live entries in heap[] */

    /* Command ring */
    npu_cmd_t   cmdq[CMDQ_CAP];      /* ring buffer */
    int         head;                 /* firmware reads from here */
    int         tail;                 /* runtime writes here */

    /* Fence table */
    int         fences[MAX_FENCES];  /* 0 = pending, 1 = signaled */
    int         nfences;

    /* Statistics */
    uint64_t    cmds_processed;
    uint64_t    dma_h2d_bytes;
    uint64_t    dma_d2h_bytes;
    uint64_t    kernels_launched;
    uint64_t    gemm_ops;
};
```

Note carefully: `heap[]` holds **metadata**, not actual bytes.  The real data lives
in `sim->mem[]` — a flat byte array representing the 16 MB DRAM.  When the heap
allocates a block at address `0x40`, that means bytes `sim->mem[0x40 .. 0x40+size-1]`
are reserved.

---

## 12.4 Device Lifecycle

### 12.4.1 Probe (initialise)

```c
kdrv_err_t npu_kdrv_probe(npu_kdrv_t *out)
{
    npu_kdrv_t drv = calloc(1, sizeof *drv);
    drv->sim = npu_create();          /* create simulator instance */
    heap_init(drv);                   /* one big free block */
    drv->clk_enabled = 1;
    *out = drv;
    return KDRV_OK;
}

static void heap_init(npu_kdrv_t drv)
{
    drv->heap[0].base = HEAP_GUARD;
    drv->heap[0].size = NPU_MEM_SIZE - HEAP_GUARD;
    drv->heap[0].used = 0;
    drv->nblocks      = 1;
}
```

After `npu_kdrv_probe`, the heap looks like this:

```
DRAM address space (16 MB = 0x00000000 … 0x00FFFFFF):

0x00000000  ╔══════════════════════╗
            ║  NULL GUARD          ║  64 bytes — never allocated
0x00000040  ╠══════════════════════╣
            ║                      ║
            ║   ONE BIG FREE BLOCK ║  16 MB − 64 B = 16 776 128 bytes
            ║                      ║
0x00FFFFFF  ╚══════════════════════╝

heap[] metadata:
  [0] = { base=0x000040, size=0xFFFFBF, used=0 }
  nblocks = 1
```

### 12.4.2 Remove and Reset

```c
kdrv_err_t npu_kdrv_remove(npu_kdrv_t drv) {
    npu_destroy(drv->sim);
    free(drv);
    return KDRV_OK;
}

kdrv_err_t npu_kdrv_reset(npu_kdrv_t drv) {
    npu_reset(drv->sim);
    memset(drv->cmdq,   0, sizeof drv->cmdq);
    memset(drv->fences, 0, sizeof drv->fences);
    drv->head = drv->tail = drv->nfences = 0;
    heap_init(drv);
    return KDRV_OK;
}
```

`reset` is used between test runs; it wipes the simulator state and resets all
bookkeeping without tearing down the OS handle.

---

## 12.5 The DRAM Heap Allocator

This section explains every decision in `heap_alloc` and `heap_free`.

### 12.5.1 Algorithm Choice: First-Fit

The allocator uses **first-fit**: scan `heap[]` from index 0, take the first free block
large enough.

Why not best-fit?
- Best-fit minimises wasted space per allocation but requires scanning the
  entire array every time, and the "best" fragment is often too small for future
  allocations anyway.
- For NPU workloads (allocate tensors once, run, free all), first-fit is faster
  and fragmentation is not a concern.

### 12.5.2 `heap_alloc` — Annotated Source

```c
static kdrv_err_t heap_alloc(npu_kdrv_t drv, size_t bytes, uint32_t *out)
{
    /* Round up to alignment boundary */
    uint32_t need = drv_align_up((uint32_t)bytes, HEAP_ALIGN);
    if (need == 0) need = HEAP_ALIGN;   /* minimum allocation = one align unit */

    for (int i = 0; i < drv->nblocks; i++) {
        heap_blk_t *b = &drv->heap[i];
        if (b->used || b->size < need) continue;

        /* Found a free block big enough.
         * Split it if the remainder is worth keeping.
         * "Worth keeping" = at least 2 × HEAP_ALIGN bytes (128 bytes).
         * Below that, the fragment is too small to serve any realistic tensor
         * and would just pollute the metadata array. */
        if (b->size - need >= HEAP_ALIGN * 2u
            && drv->nblocks < MAX_HEAP_BLK)
        {
            /* Make room for the new block by shifting everything after i */
            int tail = drv->nblocks;
            memmove(&drv->heap[i + 2], &drv->heap[i + 1],
                    (size_t)(tail - i - 1) * sizeof(heap_blk_t));
            drv->nblocks++;

            /* The remainder block starts right after the allocated portion */
            drv->heap[i + 1].base = b->base + need;
            drv->heap[i + 1].size = b->size - need;
            drv->heap[i + 1].used = 0;

            /* Shrink the current block to exactly what was requested */
            b->size = need;
        }

        b->used = 1;
        *out    = b->base;
        return KDRV_OK;
    }
    return KDRV_NO_MEM;
}
```

### 12.5.3 Step-by-Step Allocation Trace

Starting state: `nblocks=1`, single free block at `0x40`.

**Allocation 1:** `npu_kdrv_alloc(drv, &d_A, 256)`
```
need = align_up(256, 64) = 256 = 0x100

heap[0]: { base=0x0040, size=0xFFFFBF, used=0 }
  → free and big enough
  → remainder = 0xFFFFBF − 0x100 = 0xFFFEBF ≥ 128 → split

After split:
  heap[0]: { base=0x0040, size=0x0100, used=1 }   ← d_A
  heap[1]: { base=0x0140, size=0xFFFEBF, used=0 }  ← new free block
  nblocks = 2

d_A = 0x0040
```

**Allocation 2:** `npu_kdrv_alloc(drv, &d_B, 64)`
```
need = 64 = 0x40

Scans: heap[0] is used → skip
        heap[1]: { base=0x0140, size=0xFFFEBF, used=0 } → free, big enough → split

After split:
  heap[0]: { base=0x0040, size=0x0100, used=1 }   ← d_A
  heap[1]: { base=0x0140, size=0x0040, used=1 }   ← d_B
  heap[2]: { base=0x0180, size=0xFFFE7F, used=0 }  ← remaining free
  nblocks = 3

d_B = 0x0140
```

**Allocation 3:** `npu_kdrv_alloc(drv, &d_C, 1024)`
```
need = align_up(1024, 64) = 1024 = 0x400

Scans: heap[0] used, heap[1] used, heap[2]: size=0xFFFE7F ≥ 0x400 → split

After split:
  heap[0]: { 0x0040, 0x0100, used=1 }  ← d_A (256 B)
  heap[1]: { 0x0140, 0x0040, used=1 }  ← d_B (64 B)
  heap[2]: { 0x0180, 0x0400, used=1 }  ← d_C (1024 B)
  heap[3]: { 0x0580, 0xFFFA7F, used=0 } ← free
  nblocks = 4
```

### 12.5.4 `heap_free` — Annotated Source

```c
static kdrv_err_t heap_free(npu_kdrv_t drv, uint32_t addr)
{
    /* Find the block */
    int found = -1;
    for (int i = 0; i < drv->nblocks; i++) {
        if (drv->heap[i].base == addr && drv->heap[i].used) {
            found = i; break;
        }
    }
    if (found < 0) return KDRV_INVALID;

    drv->heap[found].used = 0;

    /* Coalesce RIGHT: merge with the block immediately after if it is also free */
    if (found + 1 < drv->nblocks && !drv->heap[found + 1].used) {
        drv->heap[found].size += drv->heap[found + 1].size;
        /* Remove the merged block by shifting the tail left */
        memmove(&drv->heap[found + 1], &drv->heap[found + 2],
                (size_t)(drv->nblocks - found - 2) * sizeof(heap_blk_t));
        drv->nblocks--;
    }

    /* Coalesce LEFT: merge into the block immediately before if it is also free */
    if (found > 0 && !drv->heap[found - 1].used) {
        drv->heap[found - 1].size += drv->heap[found].size;
        memmove(&drv->heap[found], &drv->heap[found + 1],
                (size_t)(drv->nblocks - found - 1) * sizeof(heap_blk_t));
        drv->nblocks--;
    }

    return KDRV_OK;
}
```

### 12.5.5 Coalescing Trace

Continuing from the 3-allocation state above. Free in order: d_B, d_A, d_C.

**Free d_B (0x0140):**
```
heap[1].used = 0

Coalesce right? heap[2] = { 0x0180, used=1 } → NO (used)
Coalesce left?  heap[0] = { 0x0040, used=1 } → NO (used)

State:
  heap[0]: { 0x0040, 0x0100, used=1 }   ← d_A
  heap[1]: { 0x0140, 0x0040, used=0 }   ← free hole
  heap[2]: { 0x0180, 0x0400, used=1 }   ← d_C
  heap[3]: { 0x0580, 0xFFFA7F, used=0 } ← big free
```

**Free d_A (0x0040):**
```
heap[0].used = 0

Coalesce right? heap[1] = { 0x0140, used=0 } → YES
  heap[0].size += heap[1].size  →  0x100 + 0x040 = 0x140
  memmove: remove heap[1], shift left
  nblocks = 3

State:
  heap[0]: { 0x0040, 0x0140, used=0 }   ← merged A+hole
  heap[1]: { 0x0180, 0x0400, used=1 }   ← d_C
  heap[2]: { 0x0580, 0xFFFA7F, used=0 } ← big free

Coalesce left? found=0, no block before → NO
```

**Free d_C (0x0180):**
```
heap[1].used = 0

Coalesce right? heap[2] = { 0x0580, used=0 } → YES
  heap[1].size += heap[2].size  →  0x400 + 0xFFFA7F = 0xFFFE7F
  nblocks = 2

State:
  heap[0]: { 0x0040, 0x0140, used=0 }
  heap[1]: { 0x0180, 0xFFFE7F, used=0 }

Coalesce left? heap[0] = { 0x0040, used=0 } → YES
  heap[0].size += heap[1].size  →  0x0140 + 0xFFFE7F = 0xFFFFBF
  nblocks = 1

Final state:
  heap[0]: { base=0x0040, size=0xFFFFBF, used=0 }
  nblocks = 1
```

**Fully back to the initial state.** Three allocations, three frees, zero fragmentation.

Why coalescing matters — without it, the heap after the same sequence would be:
```
  heap[0]: { 0x0040, 0x0100, free }  ← only 256 B
  heap[1]: { 0x0140, 0x0040, free }  ← only 64 B
  heap[2]: { 0x0180, 0x0400, free }  ← only 1024 B
  heap[3]: { 0x0580, big, free }
```
A new allocation of 512 B would have to use `heap[3]` even though heaps 0–2 sum to
1344 B of free space.  Coalescing prevents this **external fragmentation**.

---

## 12.6 The Command Ring Buffer

### 12.6.1 Command Structure

```c
typedef struct {
    uint32_t opcode;  /* which operation */
    uint32_t extra;   /* opcode-specific field (e.g. nargs, GEMM dims) */
    uint64_t arg0;    /* first argument */
    uint64_t arg1;    /* second argument */
    uint64_t arg2;    /* third argument */
} npu_cmd_t;
```

All commands are the same fixed size.  This is deliberate — it lets the ring be a
plain array with O(1) index arithmetic and no parsing overhead.

### 12.6.2 Submit (Producer Side)

```c
kdrv_err_t npu_kdrv_submit(npu_kdrv_t drv, const npu_cmd_t *cmd)
{
    if (drv->tail >= CMDQ_CAP) return KDRV_QUEUE_FULL;
    drv->cmdq[drv->tail++] = *cmd;   /* struct copy */
    return KDRV_OK;
}
```

This does **one thing**: copy the command struct into the next ring slot and advance
`tail`.  No locking needed in the simulator because there is one producer (the
runtime) and one consumer (firmware, during flush).

Ring state after three submits:
```
cmdq[]:
  [0] CMD_DMA_H2D  { dst=0x40, src=<host_ptr>, bytes=256 }
  [1] CMD_DMA_H2D  { dst=0x140, src=<host_ptr>, bytes=64 }
  [2] CMD_LAUNCH_KERN { fn=<fn_ptr>, args=<pool_ptr>, nargs=3 }
  [3..4095] ← empty

head = 0   (firmware hasn't read anything yet)
tail = 3   (next write goes here)
```

### 12.6.3 Flush (Consumer Side — the Firmware)

```c
kdrv_err_t npu_kdrv_flush(npu_kdrv_t drv)
{
    /* Append END sentinel so firmware knows where to stop */
    if (drv->tail < CMDQ_CAP) {
        npu_cmd_t end = {0};
        end.opcode = NPU_CMD_END;
        drv->cmdq[drv->tail++] = end;
    }

    firmware_run(drv);   /* runs synchronously in the simulator */

    /* Reset the ring — all commands have been consumed */
    drv->head = 0;
    drv->tail = 0;
    return KDRV_OK;
}
```

After flush, both `head` and `tail` reset to 0.  The ring is empty and ready for the
next batch.  In real hardware, `flush` would kick the DMA engine or ring a doorbell
register; the firmware would drain the ring asynchronously on the device CPU.

---

## 12.7 The Firmware Loop

`firmware_run` is the heart of the driver.  It loops over `cmdq[head..tail-1]` and
dispatches each command to the appropriate handler.

```c
static void firmware_run(npu_kdrv_t drv)
{
    while (drv->head < drv->tail) {
        npu_cmd_t *cmd = &drv->cmdq[drv->head++];
        switch (cmd->opcode) {
            case NPU_CMD_NOP:          /* do nothing */                    break;
            case NPU_CMD_DMA_H2D:      drv_do_dma_h2d(drv, cmd);          break;
            case NPU_CMD_DMA_D2H:      drv_do_dma_d2h(drv, cmd);          break;
            case NPU_CMD_DMA_D2D:      drv_do_dma_d2d(drv, cmd);          break;
            case NPU_CMD_MEMSET_I32:   drv_do_memset(drv, cmd);           break;
            case NPU_CMD_LAUNCH_KERN:  drv_do_launch(drv, cmd);           break;
            case NPU_CMD_TENSOR_GEMM:  drv_do_gemm(drv, cmd);             break;
            case NPU_CMD_TENSOR_RELU:  drv_do_relu(drv, cmd);             break;
            case NPU_CMD_FENCE_SIGNAL: drv->fences[cmd->arg0] = 1;        break;
            case NPU_CMD_BARRIER:      /* all prior cmds done — NOP here */break;
            case NPU_CMD_END:          return;  /* normal exit */
            default:                   break;
        }
        drv->cmds_processed++;
    }
}
```

Each handler is described in detail below.

### 12.7.1 `NPU_CMD_DMA_H2D` — Host-to-Device Transfer

```c
static void drv_do_dma_h2d(npu_kdrv_t drv, const npu_cmd_t *cmd)
{
    uint32_t    dst   = (uint32_t)cmd->arg0;           /* DRAM destination */
    const void *src   = (const void *)(uintptr_t)cmd->arg1; /* host pointer */
    size_t      bytes = (size_t)cmd->arg2;

    /* npu_write_i32 operates in 32-bit words; round up and zero-pad */
    int      nwords = ((int)bytes + 3) / 4;
    int32_t *tmp    = calloc((size_t)nwords, 4);
    memcpy(tmp, src, bytes);
    npu_write_i32(drv->sim, dst, tmp, nwords);
    free(tmp);

    drv->dma_h2d_bytes += bytes;
}
```

Why the temporary buffer?  `npu_write_i32` writes whole 32-bit words.  If `bytes`
is not a multiple of 4, writing directly would read past the end of `src`.  The
temp buffer (zero-initialized by `calloc`) safely pads the last word.

```
Host RAM:   [f0][f1][f2][f3][f4][f5][f6][f7]   (8 floats = 32 bytes)
               ↓    ↓    ↓    ↓    ↓    ↓    ↓    ↓
npu_write:  memcpy → sim->mem[0x40..0x5F]
```

### 12.7.2 `NPU_CMD_DMA_D2H` — Device-to-Host Transfer

The reverse: `npu_read_i32(sim, src_addr, tmp, nwords)` followed by
`memcpy(dst_host, tmp, bytes)`.  Same word-rounding concern applies.

### 12.7.3 `NPU_CMD_DMA_D2D` — Device-to-Device Copy

Both addresses are in DRAM.  The driver reads into a temp buffer and writes back
— there is no direct DRAM-to-DRAM path in the simulator API.

### 12.7.4 `NPU_CMD_MEMSET_I32` — Fill a DRAM Region

```c
static void drv_do_memset(npu_kdrv_t drv, const npu_cmd_t *cmd)
{
    uint32_t addr  = (uint32_t)cmd->arg0;
    int32_t  value = (int32_t) cmd->arg1;
    int      count = (int)     cmd->arg2;   /* in 32-bit words */

    int32_t *tmp = malloc((size_t)count * 4);
    for (int i = 0; i < count; i++) tmp[i] = value;
    npu_write_i32(drv->sim, addr, tmp, count);
    free(tmp);
}
```

Commonly used to zero-initialise a DRAM accumulator buffer before a kernel
accumulates into it.

### 12.7.5 `NPU_CMD_LAUNCH_KERN` — Execute a Compiled Kernel

This is the most important command.  It sets up the register file per the NPU
calling convention and then calls `npu_run` to execute the control-core program.

```c
static void drv_do_launch(npu_kdrv_t drv, const npu_cmd_t *cmd)
{
    npu_fn_t   fn   = (npu_fn_t)  (uintptr_t)cmd->arg0;
    npu_arg_t *args = (npu_arg_t *)(uintptr_t)cmd->arg1;
    int        nargs = (int)(uint32_t)cmd->extra;

    /* Zero the integer register file */
    memset(drv->sim->x, 0, sizeof drv->sim->x);

    /* x1 (link register) = index of last instruction (HALT) */
    drv->sim->x[1] = (int64_t)(fn->len - 1);

    /* x10–x17 = arguments (up to 8) — per calling convention */
    for (int i = 0; i < nargs && i < 8; i++) {
        int64_t v = (args[i].kind == NPU_ARG_PTR)
                  ? (int64_t)(uint64_t)(uintptr_t)args[i].ptr
                  : args[i].i64;
        drv->sim->x[10 + i] = v;
    }

    npu_run(drv->sim, fn->prog, fn->len);
    drv->kernels_launched++;
}
```

**Calling convention:**

| Register | Role |
|----------|------|
| x0 | Always 0 (hard-wired zero, zeroed by memset) |
| x1 | Link register → set to index of the HALT instruction |
| x10 | arg0 / return value |
| x11 | arg1 |
| ... | ... |
| x17 | arg7 (max 8 args) |

Why `x1 = fn->len - 1`?  The HALT instruction is always the **last** instruction
in a compiled function.  When the control core executes a `JALR x1` (indirect jump
to link register), it jumps to that index, landing on HALT, which stops
`npu_run`.  This is how functions "return".

### 12.7.6 `NPU_CMD_TENSOR_GEMM` — Tensor Engine Matrix Multiply

The GEMM command builds a 6-instruction micro-program **on the stack** and runs it
through `npu_run`.  This is a key design pattern: the control core is the scheduler
for the tensor engine, feeding it instructions on demand.

```c
static void drv_do_gemm(npu_kdrv_t drv, const npu_cmd_t *cmd)
{
    /* Unpack dimensions from extra field: M | (K<<8) | (N<<16) */
    int M = (int)( cmd->extra        & 0xFF);
    int K = (int)((cmd->extra >>  8) & 0xFF);
    int N = (int)((cmd->extra >> 16) & 0xFF);

    uint32_t A_addr = (uint32_t)cmd->arg0;
    uint32_t B_addr = (uint32_t)cmd->arg1;
    uint32_t C_addr = (uint32_t)cmd->arg2;

    /* Build a minimal tensor-engine program */
    cinst prog[6];
    memset(prog, 0, sizeof prog);

    prog[0] = (cinst){ .op=T_CFG_SHAPE, .m=M, .k=K, .n=N };
    prog[1] = (cinst){ .op=T_ACC_CLR };               /* clear accumulator */
    prog[2] = (cinst){ .op=T_LOAD,  .tile=0 };        /* load A into tile 0 */
    prog[3] = (cinst){ .op=T_LOAD,  .tile=1 };        /* load B into tile 1 */
    prog[4] = (cinst){ .op=T_MMA,   .rd=2, .rs1=0, .rs2=1 }; /* C=A×B in tile 2 */
    prog[5] = (cinst){ .op=T_STORE, .tile=2 };        /* store tile 2 → C */

    /* Provide addresses via registers */
    drv->sim->x[1]  = 5;        /* link → HALT at index 5 (T_STORE doubles as HALT) */
    drv->sim->x[11] = A_addr;
    drv->sim->x[12] = B_addr;
    drv->sim->x[13] = C_addr;

    npu_run(drv->sim, prog, 6);
    drv->gemm_ops++;
}
```

The dimensions are packed into 24 bits: `extra = M | (K<<8) | (N<<16)`.  Each
field is 8 bits, so tiles up to 255×255 are supported.

### 12.7.7 `NPU_CMD_FENCE_SIGNAL` — Signal a Fence

```c
case NPU_CMD_FENCE_SIGNAL:
    drv->fences[(int)cmd->arg0] = 1;   /* mark fence as signaled */
    break;
```

Fences are covered in Section 12.8.

### 12.7.8 `NPU_CMD_END` — End of Batch

```c
case NPU_CMD_END:
    return;   /* exits firmware_run() */
```

The END command is appended by `npu_kdrv_flush` as a sentinel.  It makes the
firmware loop terminate cleanly even if `head < tail` would otherwise allow
overrun.

---

## 12.8 Fence Management

### 12.8.1 What Fences Are

A fence is an ordered completion token.  Inserting a `CMD_FENCE_SIGNAL` into the
command stream guarantees that all commands submitted before the fence are complete
by the time the fence is signaled.

```
Submit: CMD_DMA_H2D     (copy weights)
Submit: CMD_LAUNCH_KERN  (inference kernel)
Submit: CMD_FENCE_SIGNAL (fence id=0)
Submit: CMD_DMA_D2H     (copy results)
Flush → firmware runs all 4 in order → fences[0] = 1

npu_kdrv_fence_wait(drv, 0)
  → fences[0] == 1 → return KDRV_OK immediately
```

### 12.8.2 Why Fences Exist if Flush is Synchronous

In the simulator, `firmware_run` is synchronous — everything is done by the time
`npu_kdrv_flush` returns.  Fences appear to be redundant.

They exist because the **interface mirrors real hardware**, where:
- The driver submits commands asynchronously to a hardware ring buffer
- The firmware runs on the NPU's control core concurrently
- The CPU needs a way to know when a specific point in the stream is complete
  without waiting for the entire queue to drain

Preserving the fence API means code written against the simulator ports unchanged
to real hardware drivers.

### 12.8.3 Creating and Waiting on Fences

```c
kdrv_err_t npu_kdrv_fence_create(npu_kdrv_t drv, int *id_out) {
    if (drv->nfences >= MAX_FENCES) return KDRV_NO_MEM;
    int id = drv->nfences++;
    drv->fences[id] = 0;   /* start unsignaled */
    *id_out = id;
    return KDRV_OK;
}

kdrv_err_t npu_kdrv_fence_wait(npu_kdrv_t drv, int id) {
    if (id < 0 || id >= drv->nfences) return KDRV_INVALID;
    /* In the sim: if the firmware has run, fences[id] is already 1 */
    if (!drv->fences[id]) return KDRV_PENDING;
    return KDRV_OK;
}
```

In real hardware, `fence_wait` would block (or poll with a timeout) until the
firmware signals the fence from the device side.

---

## 12.9 Complete Data Flow: `npuMalloc` → `npuMemcpy_h2d` → DRAM

This diagram shows the **complete path** from a user `npuMalloc` call to data
sitting in simulated DRAM, with every layer's participation:

```
Application:
    NpuDevPtr d_A;
    npuMalloc(ctx, &d_A, 256);

npu_rt.c:
    → npu_kdrv_alloc(ctx->drv, &d_A, 256)

npu_kdrv.c: heap_alloc(drv, 256, &d_A)
    → scan heap[], find heap[0] (free, big enough)
    → split: heap[0]={0x40,256,used=1}, heap[1]={0x140,big,free}
    → *out = 0x40
    d_A = 0x00000040  ← device pointer returned to application

Application:
    npuMemcpy_h2d(ctx, d_A, host_A, 256);  // ASYNC!

npu_rt.c:
    npu_cmd_t cmd = { .opcode=NPU_CMD_DMA_H2D,
                      .arg0=0x40, .arg1=(uintptr_t)host_A, .arg2=256 };
    → npu_kdrv_submit(ctx->drv, &cmd)

npu_kdrv.c: submit
    → cmdq[tail++] = cmd;    // copy struct into ring
    // NOTHING RUNS YET

Application:
    npuSynchronize(ctx);

npu_rt.c:
    → npu_kdrv_flush(ctx->drv)

npu_kdrv.c: flush
    → append NPU_CMD_END to cmdq
    → firmware_run(drv):
        head=0: CMD_DMA_H2D
          → drv_do_dma_h2d(drv, &cmdq[0])
             → nwords = (256+3)/4 = 64
             → tmp = calloc(64, 4)
             → memcpy(tmp, host_A, 256)
             → npu_write_i32(sim, 0x40, tmp, 64)
                  → memcpy(&sim->mem[0x40], tmp, 256)
             → free(tmp)
          ← Data now in sim->mem[0x40 .. 0x13F]
        head=1: NPU_CMD_END → return
    → drv->head=0; drv->tail=0;  (reset ring)
```

---

## 12.10 Error Codes

| Code | Value | Meaning |
|------|-------|---------|
| `KDRV_OK` | 0 | Success |
| `KDRV_INVALID` | 1 | Null pointer, bad address, or invalid id |
| `KDRV_NO_MEM` | 2 | Heap full or metadata table full |
| `KDRV_QUEUE_FULL` | 3 | Command ring at capacity (> 4096 pending) |
| `KDRV_PENDING` | 4 | Fence not yet signaled |
| `KDRV_IO` | 5 | Simulator I/O error |

---

## 12.11 Summary

| Component | Key Decisions |
|-----------|--------------|
| Heap allocator | First-fit; 64-byte alignment; HEAP_GUARD null region; split on alloc, coalesce on free |
| Command ring | Fixed-size structs; single producer / single consumer; END sentinel |
| Firmware loop | Sequential (in simulator); mirrors async hardware firmware |
| GEMM | Builds micro-program on stack; dispatched via npu_run |
| Fences | API mirrors real HW; simulator satisfies them synchronously |

The next chapter covers the runtime layer (`npu_rt`) that sits on top of the driver
and provides the CUDA-style async API that application code uses.
