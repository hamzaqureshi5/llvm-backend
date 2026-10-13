# Chapter 13 — The Runtime API (`npu_rt`)

> **Prerequisites:** Chapter 12 (kernel driver), Chapter 9 (execution model).
> **Source files:** `npu-simulator/src/npu_rt.c`, `npu-simulator/include/npu_rt.h`
> **Live example:** `npu-simulator/examples/demo_matmul_elf.c`

---

## 13.1 Design Philosophy — CUDA as the Model

The `npu_rt` API is deliberately modelled after CUDA.  Anyone familiar with CUDA
can read an NPU runtime program without a manual.  The key idea borrowed from CUDA:

> **Push now, execute later.**  Memory copies and kernel launches push command
> structs into a queue.  No actual work happens until `npuSynchronize` drains the
> queue.

This batching reduces context-switch overhead in real hardware (each push is a
lightweight user-space write; only `Synchronize` costs a kernel syscall via flush)
and gives the firmware latitude to reorder or pipeline commands.

### CUDA ↔ NPU API Mapping

| CUDA concept | NPU equivalent | Notes |
|---|---|---|
| `CUcontext` | `NpuCtx` | Owns the device, heap, module table |
| `CUdeviceptr` | `NpuDevPtr` (`uint32_t`) | 32-bit DRAM address |
| `CUmodule` | `NpuModule` | Loaded ELF or assembled program |
| `CUfunction` | `NpuFunction` | One kernel entry point in a module |
| `cudaMalloc` | `npuMalloc` | Synchronous — allocates immediately |
| `cudaMemcpy` | `npuMemcpy_h2d/d2h` | Async — pushes to queue |
| `cudaLaunchKernel` | `npuLaunchKernel` | Async — pushes to queue |
| `cudaDeviceSynchronize` | `npuSynchronize` | Drains queue, waits for completion |
| `cuModuleLoad` | `npuModuleLoad` | Loads ELF `.o` file |
| `cuModuleGetFunction` | `npuGetFunction` | Looks up kernel by name |

---

## 13.2 The Two Runtimes in This Codebase

Before diving in, it is worth noting that the simulator ships **two** runtime
implementations:

| Library | Header | Style | Driver layer |
|---------|--------|-------|--------------|
| `npu_runtime` | `npu_runtime.h` | Synchronous — launch executes immediately | Direct `npu_t` access |
| `npu_stack` (this chapter) | `npu_rt.h` | Async CUDA-style — push/synchronize | Via `npu_kdrv` |

`npu_runtime` is the simpler option, used in `demo_vadd`.  `npu_rt` is the
production-style stack, used in `demo_rt`, `demo_rt_te_vc`, `demo_elf_load`, and
`demo_matmul_elf`.  This chapter covers `npu_rt` exclusively.

---

## 13.3 Context Structure

The `NpuCtx` handle carries everything the runtime owns:

```c
#define MAX_RT_MODULES  64
#define MAX_ARG_SLOTS  1024
#define MAX_ARGS_PER_KERN 8

typedef struct {
    uint64_t  i64;   /* integer / float value */
    void     *ptr;   /* host pointer */
    int       kind;  /* NPU_ARG_I64 or NPU_ARG_PTR */
} npu_arg_t;

typedef struct {
    npu_arg_t args[MAX_ARGS_PER_KERN];
    int       nargs;
    int       used;
} arg_slot_t;

struct npu_ctx_s {
    npu_kdrv_t  drv;                     /* kernel driver handle */
    NpuModule   mods[MAX_RT_MODULES];    /* up to 64 loaded modules */
    int         nmods;

    /* Argument copy pool — see Section 13.6 */
    arg_slot_t  arg_pool[MAX_ARG_SLOTS]; /* 1024 slots × 8 args each */
    int         arg_next;                /* hint for next free slot search */
};
```

Key observation: `drv` is the only handle to the actual "hardware".  All memory
and compute operations eventually go through it.

---

## 13.4 Context Lifecycle

```c
NpuError npuInit(NpuCtx *out)
{
    npu_ctx_t *ctx = calloc(1, sizeof *ctx);
    npu_kdrv_probe(&ctx->drv);   /* create simulator + heap */
    *out = ctx;
    return NPU_OK;
}

NpuError npuFinalize(NpuCtx ctx)
{
    for (int i = 0; i < ctx->nmods; i++)
        module_destroy(ctx->mods[i]);    /* free decoded programs */
    npu_kdrv_remove(ctx->drv);          /* destroy simulator */
    free(ctx);
    return NPU_OK;
}
```

Always pair `npuInit` with `npuFinalize`.  Modules and device allocations owned by
the context are freed by `npuFinalize`; any `NpuDevPtr` values you hold become
invalid afterward.

---

## 13.5 Memory — Synchronous vs Async

This is the most important design distinction in the runtime.

### 13.5.1 `npuMalloc` — Synchronous

```c
NpuError npuMalloc(NpuCtx ctx, NpuDevPtr *ptr, size_t bytes)
{
    return from_kdrv(npu_kdrv_alloc(ctx->drv, ptr, bytes));
}
```

`npuMalloc` calls the driver **directly** — no command is pushed to the queue.
The heap metadata is updated immediately and `*ptr` holds a valid DRAM address
before `npuMalloc` returns.

**Why synchronous?**  You need the device address *before* you can enqueue a DMA
command that uses it.  The address must exist in the call that builds `CMD_DMA_H2D`.

```
npuMalloc(ctx, &d_A, 256);              // d_A = 0x40  ← KNOWN NOW
npuMemcpy_h2d(ctx, d_A, h_A, 256);     // arg0=0x40  ← used here
```

If `npuMalloc` were async, `d_A` would be undefined when `npuMemcpy_h2d` ran.

### 13.5.2 `npuMemcpy_h2d` — Async

```c
NpuError npuMemcpy_h2d(NpuCtx ctx, NpuDevPtr dst,
                        const void *src, size_t bytes)
{
    npu_cmd_t cmd = {0};
    cmd.opcode = NPU_CMD_DMA_H2D;
    cmd.arg0   = (uint64_t)dst;                      /* DRAM destination */
    cmd.arg1   = (uint64_t)(uintptr_t)src;           /* host source pointer */
    cmd.arg2   = (uint64_t)bytes;
    return from_kdrv(npu_kdrv_submit(ctx->drv, &cmd));
}
```

The `src` host pointer is stored in the command struct as a raw integer.  It is
safe because:

1. In the simulator, `npuSynchronize` runs the firmware in the **same process**
   on the same thread.  The host pointer is valid as long as the caller doesn't
   free the host buffer before calling `npuSynchronize`.

2. In real hardware, the driver would **pin** (page-lock) the host buffer when the
   DMA command is submitted, preventing the OS from swapping it out before the
   DMA engine reads it.

**D2H and D2D variants** follow the same pattern: push a `CMD_DMA_D2H` or
`CMD_DMA_D2D` struct, execute on synchronize.

### 13.5.3 Timeline of a Complete Program

```
npuInit()                 ─── create context, probe driver
  │
npuMalloc(d_A, 256)       ─── SYNC: heap_alloc → d_A=0x40
npuMalloc(d_B, 256)       ─── SYNC: heap_alloc → d_B=0x140
npuMalloc(d_C, 256)       ─── SYNC: heap_alloc → d_C=0x180
  │
npuModuleLoad(mod, "f.o") ─── SYNC: mmap ELF, decode instructions
npuGetFunction(fn, "f")   ─── SYNC: name lookup in module
  │
npuMemcpy_h2d(d_A, h_A)  ─── ASYNC: push CMD_DMA_H2D #1
npuMemcpy_h2d(d_B, h_B)  ─── ASYNC: push CMD_DMA_H2D #2
npuLaunchKernel(fn, ...)  ─── ASYNC: push CMD_LAUNCH_KERN
npuMemcpy_d2h(h_C, d_C)  ─── ASYNC: push CMD_DMA_D2H
  │                            ← 4 commands in queue, nothing executed yet
npuSynchronize()          ─── flush → firmware runs all 4 commands
  │
npuRead_f32(d_C, h_C)     ─── SYNC: read from sim DRAM (firmware done)
  │
npuFree(d_A)              ─── SYNC: heap_free
npuFree(d_B)              ─── SYNC: heap_free
npuFree(d_C)              ─── SYNC: heap_free
npuModuleUnload(mod)       ─── SYNC: free decoded programs
npuFinalize()              ─── SYNC: destroy driver, free context
```

---

## 13.6 The Argument-Copy Pool — The Subtlest Part

### 13.6.1 The Problem

`npuLaunchKernel` is async — it only pushes a command.  The firmware executes that
command later, during `npuSynchronize`.

Consider this caller:

```c
void do_relu(NpuCtx ctx, NpuFunction fn, NpuDevPtr d_in, NpuDevPtr d_out)
{
    npu_arg_t args[] = {
        NPU_PTR(d_in),    /* args[0] — on the STACK */
        NPU_PTR(d_out),   /* args[1] — on the STACK */
    };
    npuLaunchKernel(ctx, fn, args, 2);
    /* ← function returns here; args[] is gone */
}

do_relu(ctx, fn, d_A, d_B);   /* stack frame destroyed here */
npuSynchronize(ctx);           /* firmware tries to read args… from deallocated stack */
```

If `CMD_LAUNCH_KERN` stored a pointer to `args[]` on the caller's stack, that
memory would be invalid by the time the firmware ran it.  This is a classic
**dangling pointer** bug.

### 13.6.2 The Fix: Pool Allocation

The runtime maintains a pool of 1024 slots, each large enough for 8 args.  When
`npuLaunchKernel` is called, it copies the args into a free pool slot and stores a
pointer to the **pool copy** in the command:

```c
NpuError npuLaunchKernel(NpuCtx ctx, NpuFunction fn,
                          const npu_arg_t *args, int nargs)
{
    /* Find a free slot, starting from the hint */
    int slot = -1;
    for (int i = 0; i < MAX_ARG_SLOTS; i++) {
        int idx = (ctx->arg_next + i) % MAX_ARG_SLOTS;
        if (!ctx->arg_pool[idx].used) { slot = idx; break; }
    }
    if (slot < 0) return NPU_ERR_NO_MEM;

    /* Copy args into the pool — safe to keep after caller returns */
    arg_slot_t *s = &ctx->arg_pool[slot];
    s->used  = 1;
    s->nargs = nargs;
    memcpy(s->args, args, (size_t)nargs * sizeof(npu_arg_t));
    ctx->arg_next = (slot + 1) % MAX_ARG_SLOTS;

    /* Build the command with a pointer to the POOL COPY */
    npu_cmd_t cmd = {0};
    cmd.opcode = NPU_CMD_LAUNCH_KERN;
    cmd.extra  = (uint32_t)nargs;
    cmd.arg0   = (uint64_t)(uintptr_t)fn;
    cmd.arg1   = (uint64_t)(uintptr_t)s->args;   /* ← pool copy, NOT caller's stack */
    return from_kdrv(npu_kdrv_submit(ctx->drv, &cmd));
}
```

### 13.6.3 Pool Lifecycle Diagram

```
Before npuSynchronize — two kernels queued:

arg_pool[]:
  [0]: used=1, nargs=3, args=[NPU_PTR(d_C), NPU_PTR(d_A), NPU_PTR(d_B)]
       ↑ referenced by CMD_LAUNCH_KERN #1
  [1]: used=1, nargs=2, args=[NPU_PTR(d_in), NPU_PTR(d_out)]
       ↑ referenced by CMD_LAUNCH_KERN #2
  [2..1023]: used=0

cmdq[]:
  [0] CMD_DMA_H2D
  [1] CMD_LAUNCH_KERN { arg1 = &pool[0].args }
  [2] CMD_LAUNCH_KERN { arg1 = &pool[1].args }
  [3] CMD_DMA_D2H

npuSynchronize():
  → npu_kdrv_flush → firmware_run:
      CMD_DMA_H2D      → transfers data
      CMD_LAUNCH_KERN  → drv reads pool[0].args ← still valid!
      CMD_LAUNCH_KERN  → drv reads pool[1].args ← still valid!
      CMD_DMA_D2H      → transfers data back
  → bulk recycle:
      for i in 0..1023: arg_pool[i].used = 0
      arg_next = 0

After Synchronize — pool is empty again:
  [0..1023]: used=0
```

The bulk recycle after firmware_run is safe because the firmware has consumed
every arg copy.  The pool becomes available for the next batch.

---

## 13.7 Module Loading — The ELF Loader

`npuModuleLoad` reads a compiled ELF `.o` file, locates all global function
symbols, decodes the binary instruction stream, and registers each function for
later lookup.

### 13.7.1 The ELF File Format (Minimal Refresher)

An ELF object file has:

```
ELF header (64 bytes)
  e_shoff  → offset to section header table
  e_shnum  → number of sections
  e_shstrndx → section holding section names

Section header table (one 64-byte Shdr per section)
  Each section: name, type, flags, file offset, size

Key sections for us:
  .text    — raw binary instructions (32 bytes per NPU instruction)
  .symtab  — symbol table (Elf64_Sym entries)
  .strtab  — null-terminated strings (symbol names)
```

### 13.7.2 Loader Walkthrough

```c
NpuError npuModuleLoad(NpuCtx ctx, NpuModule *out, const char *path)
{
    /* Step 1: mmap the file — read-only, private */
    int fd = open(path, O_RDONLY);
    struct stat st;
    fstat(fd, &st);
    uint8_t *data = mmap(NULL, (size_t)st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);

    /* Step 2: Validate ELF magic and class */
    if (memcmp(data, ELFMAG, SELFMAG) != 0)      /* "\x7fELF" */
        return NPU_ERR_BAD_MODULE;
    if (data[EI_CLASS] != ELFCLASS64)             /* must be 64-bit ELF */
        return NPU_ERR_BAD_MODULE;

    /* Step 3: Find .text, .symtab, .strtab */
    Elf64_Ehdr *eh    = (Elf64_Ehdr *)data;
    Elf64_Shdr *shdrs = (Elf64_Shdr *)(data + eh->e_shoff);
    const char *shstrtab = (const char *)(data + shdrs[eh->e_shstrndx].sh_offset);

    Elf64_Shdr *sh_text = NULL, *sh_sym = NULL, *sh_str = NULL;
    for (int i = 0; i < eh->e_shnum; i++) {
        const char *name = shstrtab + shdrs[i].sh_name;
        if (strcmp(name, ".text")   == 0) sh_text = &shdrs[i];
        if (shdrs[i].sh_type == SHT_SYMTAB)    sh_sym  = &shdrs[i];
        if (shdrs[i].sh_type == SHT_STRTAB
            && strcmp(name, ".strtab") == 0)    sh_str  = &shdrs[i];
    }

    /* Step 4: Walk symbol table — find GLOBAL FUNC symbols */
    Elf64_Sym  *syms    = (Elf64_Sym *)(data + sh_sym->sh_offset);
    const char *strtab  = (const char *)(data + sh_str->sh_offset);
    int         nsyms   = (int)(sh_sym->sh_size / sizeof(Elf64_Sym));
    uint8_t    *text    = data + sh_text->sh_offset;

    npu_module_t *mod = module_create();
    for (int i = 0; i < nsyms; i++) {
        Elf64_Sym *sym = &syms[i];
        if (ELF64_ST_BIND(sym->st_info) != STB_GLOBAL) continue;
        if (ELF64_ST_TYPE(sym->st_info) != STT_FUNC)   continue;
        if (sym->st_size == 0) continue;

        /* Step 5: Decode the instruction stream for this function */
        int       fn_words = (int)(sym->st_size / NPU_WORD_BYTES); /* 32 bytes/instr */
        uint8_t  *fn_bytes = text + sym->st_value;
        cinst    *prog     = malloc((size_t)fn_words * sizeof(cinst));

        for (int k = 0; k < fn_words; k++)
            npu_decode_cinst(fn_bytes + k * NPU_WORD_BYTES, &prog[k]);

        module_add_fn(mod, strtab + sym->st_name, prog, fn_words);
    }

    munmap(data, (size_t)st.st_size);
    *out = mod;
    return NPU_OK;
}
```

### 13.7.3 ELF → Decoded Program Diagram

For a `matmul_4x4` function compiled to 9 instructions (288 bytes):

```
matmul.o layout:

  ELF header ─────────────────────────────────
  .text section (288 bytes = 9 × 32):
    offset 0x00: [32 raw bytes] ← instruction 0 (T_CFG_SHAPE)
    offset 0x20: [32 raw bytes] ← instruction 1 (T_ACC_CLR)
    offset 0x40: [32 raw bytes] ← instruction 2 (T_LOAD tile=0)
    offset 0x60: [32 raw bytes] ← instruction 3 (T_LOAD tile=1)
    offset 0x80: [32 raw bytes] ← instruction 4 (T_MMA)
    offset 0xA0: [32 raw bytes] ← instruction 5 (T_STORE)
    offset 0xC0: [32 raw bytes] ← instruction 6 (T_FENCE)
    offset 0xE0: [32 raw bytes] ← instruction 7 (...)
    offset 0x100: [32 bytes]    ← instruction 8 (C_HALT)

  .symtab: Elf64_Sym {
    st_name  = 4  (index into .strtab → "matmul_4x4")
    st_value = 0  (offset in .text)
    st_size  = 288
    st_info  = STB_GLOBAL | STT_FUNC
  }
  .strtab: "\0_start\0matmul_4x4\0"

npuModuleLoad processes:
  fn_words = 288 / 32 = 9
  prog = malloc(9 × sizeof(cinst))
  for k in 0..8:
    npu_decode_cinst(text + k*32, &prog[k])

module now contains:
  fn "matmul_4x4" → { prog=<9 cinst structs>, len=9 }
```

### 13.7.4 ASM Loading Alternative

`npuModuleLoadAsm` accepts a text program directly (as a string of assembly
instructions):

```c
const char *asm_src =
    "T.CFG.SHAPE 4 4 4\n"
    "T.ACC.CLR\n"
    "T.LOAD 0\n"
    "T.LOAD 1\n"
    "T.MMA 2 0 1\n"
    "T.STORE 2\n"
    "C.HALT\n";

NpuModule mod;
npuModuleLoadAsm(ctx, &mod, asm_src);
```

Under the hood, this calls `npu_asm_control(asm_src, prog, &len)` from `npu_asm.h`
to produce the `cinst[]` array, then wraps it in a module.  Useful for quick
experiments without invoking the LLVM backend.

---

## 13.8 `npuGetFunction` — Name Lookup

```c
NpuError npuGetFunction(NpuModule mod, NpuFunction *out, const char *name)
{
    for (int i = 0; i < mod->nfns; i++) {
        if (strcmp(mod->fns[i].name, name) == 0) {
            *out = &mod->fns[i];
            return NPU_OK;
        }
    }
    return NPU_ERR_NOT_FOUND;
}
```

Linear search over functions in the module.  For a typical module with ≤ 64
functions, this is negligible overhead.  The returned `NpuFunction` pointer is
valid for the lifetime of the module.

---

## 13.9 `npuSynchronize` — The Execution Barrier

```c
NpuError npuSynchronize(NpuCtx ctx)
{
    /* Drain the command queue — this is where all work happens */
    NpuError rc = from_kdrv(npu_kdrv_flush(ctx->drv));

    /* Bulk-recycle the arg pool — firmware has consumed all args */
    for (int i = 0; i < MAX_ARG_SLOTS; i++)
        ctx->arg_pool[i].used = 0;
    ctx->arg_next = 0;

    return rc;
}
```

Three things happen in order:

1. **`npu_kdrv_flush`** appends `CMD_END` and runs `firmware_run`, which executes
   every pending command in submission order.
2. **Arg pool recycle** — all 1024 slots are marked free.  This is safe because
   the firmware has consumed every `CMD_LAUNCH_KERN` that referenced the pool.
3. **Return** — the caller knows all submitted work is complete.

After `npuSynchronize` returns:
- DMA_H2D data is in DRAM
- All kernels have finished execution
- DMA_D2H data is in host memory
- The command queue is empty (head=0, tail=0)
- The arg pool is empty

---

## 13.10 High-Level Operations

### 13.10.1 `npuGemm` — Tensor Engine Matrix Multiply

```c
NpuError npuGemm(NpuCtx ctx,
                  NpuDevPtr A, NpuDevPtr B, NpuDevPtr C,
                  int M, int K, int N)
{
    npu_cmd_t cmd = {0};
    cmd.opcode = NPU_CMD_TENSOR_GEMM;
    cmd.extra  = (uint32_t)(M | (K << 8) | (N << 16));
    cmd.arg0   = (uint64_t)A;
    cmd.arg1   = (uint64_t)B;
    cmd.arg2   = (uint64_t)C;
    return from_kdrv(npu_kdrv_submit(ctx->drv, &cmd));
}
```

One call → one queued command → firmware builds 6-instruction micro-program →
`npu_run`.  Compare to `npuLaunchKernel` which requires a compiled ELF or ASM
program; `npuGemm` is the pre-built fast path for tensor-engine GEMM.

### 13.10.2 `npuRelu` — Tensor Engine ReLU

Similar pattern: queues `CMD_TENSOR_RELU` with address and count.  The firmware
handler allocates a temp buffer, reads from DRAM, applies `fmaxf(v, 0)` per
element, writes back.

### 13.10.3 `npuMemset` — Fill Device Memory

```c
NpuError npuMemset(NpuCtx ctx, NpuDevPtr ptr, int32_t value, int count)
{
    npu_cmd_t cmd = {0};
    cmd.opcode = NPU_CMD_MEMSET_I32;
    cmd.arg0   = (uint64_t)ptr;
    cmd.arg1   = (uint64_t)(uint32_t)value;
    cmd.arg2   = (uint64_t)count;
    return from_kdrv(npu_kdrv_submit(ctx->drv, &cmd));
}
```

Async.  Commonly used to zero an accumulator buffer before accumulating into it.

### 13.10.4 `npuRead_f32` / `npuRead_i32` — Synchronous Readback

```c
NpuError npuRead_f32(NpuCtx ctx, NpuDevPtr src, float *dst, int count)
{
    /* Direct sim DRAM read — no command queue involvement */
    int32_t *tmp = malloc((size_t)count * 4);
    npu_read_i32(ctx->drv->sim, src, tmp, count);
    memcpy(dst, tmp, (size_t)count * 4);
    free(tmp);
    return NPU_OK;
}
```

`npuRead_f32` bypasses the command queue entirely — it reads directly from the
simulator's DRAM array.  Call it **after** `npuSynchronize`, when you know the
DMA_D2H has already run and the results are in DRAM.

If you call `npuRead_f32` before `npuSynchronize`, you will read stale data (the
kernel hasn't run yet).

---

## 13.11 Fences — Ordering Guarantees

```c
NpuError npuFenceCreate(NpuCtx ctx, NpuFence *out) {
    int id;
    NpuError rc = from_kdrv(npu_kdrv_fence_create(ctx->drv, &id));
    *out = (NpuFence)(uintptr_t)id;
    return rc;
}

NpuError npuFenceWait(NpuCtx ctx, NpuFence fence) {
    /* Insert a FENCE_SIGNAL command to mark "all prior work is done" */
    npu_cmd_t cmd = {0};
    cmd.opcode = NPU_CMD_FENCE_SIGNAL;
    cmd.arg0   = (uint64_t)(uintptr_t)fence;
    NpuError rc = from_kdrv(npu_kdrv_submit(ctx->drv, &cmd));
    if (rc != NPU_OK) return rc;

    /* Flush and wait */
    rc = from_kdrv(npu_kdrv_flush(ctx->drv));
    if (rc != NPU_OK) return rc;
    return from_kdrv(npu_kdrv_fence_wait(ctx->drv, (int)(uintptr_t)fence));
}
```

`npuFenceWait` combines two operations: inserting a signal point in the stream, and
waiting for it.  In the simulator, the fence is always signaled by the time flush
returns, so the wait is a no-op.  On real hardware, the wait would block until the
device firmware signals the fence register.

---

## 13.12 Error Handling

### 13.12.1 Error Codes

| Code | Value | Typical cause |
|------|-------|---------------|
| `NPU_OK` | 0 | Success |
| `NPU_ERR_INVALID` | 1 | NULL handle, bad module, wrong arg type |
| `NPU_ERR_NO_MEM` | 2 | DRAM heap full or arg pool exhausted |
| `NPU_ERR_QUEUE_FULL` | 3 | > 4096 commands pending without Synchronize |
| `NPU_ERR_IO` | 4 | ELF file not found or unreadable |
| `NPU_ERR_NOT_FOUND` | 5 | Function name not in module |
| `NPU_ERR_BAD_MODULE` | 6 | ELF magic wrong or not ELFCLASS64 |

### 13.12.2 `NPU_RT_CHECK` Macro

```c
#define NPU_RT_CHECK(call)                                              \
    do {                                                                \
        NpuError _e = (call);                                           \
        if (_e != NPU_OK) {                                             \
            fprintf(stderr, "NPU error %s:%d  %s → %s\n",              \
                    __FILE__, __LINE__, #call, npuGetErrorString(_e));  \
            abort();                                                    \
        }                                                               \
    } while (0)
```

The macro stringifies the call expression via `#call`, so error messages look like:

```
NPU error demo_rt.c:47  npuModuleLoad(ctx, &mod, "matmul.o") → NPU_ERR_IO
```

Use `NPU_RT_CHECK` in demo and test code.  Production code should propagate errors
rather than abort.

---

## 13.13 Complete End-to-End Example — ELF Matmul

This is the code in `demo_matmul_elf.c`, annotated to show what happens at every
layer.

```c
#include "npu_rt.h"
#include <math.h>

#define M 4
#define K 4
#define N 4
#define NELEMS (M * N)

int main(int argc, char **argv)
{
    const char *obj_path = argv[1];   /* path to matmul.o */

    /* ── Host matrices ──────────────────────────────────────────────── */
    float h_A[16], h_B[16], h_C[16], h_ref[16];

    /* Fill A: diagonal = (i+1), off-diagonal = 0.5 */
    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 4; j++)
            h_A[i*4+j] = (i==j) ? (float)(i+1) : 0.5f;

    /* Fill B: B[k][j] = k*4+j+1 (1..16) */
    for (int k = 0; k < 4; k++)
        for (int j = 0; j < 4; j++)
            h_B[k*4+j] = (float)(k*4+j+1);

    /* CPU reference */
    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 4; j++) {
            float acc = 0;
            for (int k = 0; k < 4; k++) acc += h_A[i*4+k] * h_B[k*4+j];
            h_ref[i*4+j] = acc;
        }

    /* ── Step 1: Init ───────────────────────────────────────────────── */
    NpuCtx ctx;
    NPU_RT_CHECK(npuInit(&ctx));
    /*
     * Inside: calloc(npu_ctx_s), npu_kdrv_probe → npu_create + heap_init
     * Result: 16 MB simulator DRAM, one free heap block at 0x40
     */

    /* ── Step 2: Allocate device buffers (synchronous) ──────────────── */
    NpuDevPtr d_A, d_B, d_C;
    NPU_RT_CHECK(npuMalloc(ctx, &d_A, NELEMS * sizeof(float)));
    /* Inside: npu_kdrv_alloc → heap_alloc(drv, 64, &d_A) → d_A=0x40 */

    NPU_RT_CHECK(npuMalloc(ctx, &d_B, NELEMS * sizeof(float)));
    /* d_B = 0x80 (next 64-byte slot) */

    NPU_RT_CHECK(npuMalloc(ctx, &d_C, NELEMS * sizeof(float)));
    /* d_C = 0xC0 */

    /* ── Step 3: Load ELF module (synchronous) ──────────────────────── */
    NpuModule   mod;
    NpuFunction fn;
    NPU_RT_CHECK(npuModuleLoad(ctx, &mod, obj_path));
    /*
     * Inside:
     *   mmap matmul.o
     *   validate ELF magic + ELFCLASS64
     *   find .text, .symtab, .strtab
     *   find symbol "matmul_4x4": offset=0, size=9*32=288 bytes
     *   for k in 0..8: npu_decode_cinst(text+k*32, &prog[k])
     *   module_add_fn(mod, "matmul_4x4", prog, 9)
     *   munmap
     */

    NPU_RT_CHECK(npuGetFunction(mod, &fn, "matmul_4x4"));
    /* Inside: linear scan of module's fn table → finds "matmul_4x4" */

    /* ── Step 4: Upload inputs (async) ──────────────────────────────── */
    NPU_RT_CHECK(npuMemcpy_h2d(ctx, d_A, h_A, NELEMS * sizeof(float)));
    /* Pushes CMD_DMA_H2D { dst=0x40, src=h_A, bytes=64 } — nothing runs */

    NPU_RT_CHECK(npuMemcpy_h2d(ctx, d_B, h_B, NELEMS * sizeof(float)));
    /* Pushes CMD_DMA_H2D { dst=0x80, src=h_B, bytes=64 } — nothing runs */

    /* ── Step 5: Launch kernel (async) ──────────────────────────────── */
    npu_arg_t args[] = {
        NPU_PTR(d_C),   /* x10: output C at 0xC0 */
        NPU_PTR(d_A),   /* x11: input  A at 0x40 */
        NPU_PTR(d_B),   /* x12: input  B at 0x80 */
    };
    NPU_RT_CHECK(npuLaunchKernel(ctx, fn, args, 3));
    /*
     * Inside:
     *   Find free arg_pool slot → slot 0
     *   Copy args to pool[0]: { NPU_PTR(0xC0), NPU_PTR(0x40), NPU_PTR(0x80) }
     *   Pushes CMD_LAUNCH_KERN {
     *     fn  = &fn (decoded 9-instruction program),
     *     arg = &pool[0].args   ← pool copy, NOT args[] on this stack frame
     *     nargs = 3
     *   }
     *
     * cmdq now holds 3 commands (2 DMA, 1 LAUNCH). Nothing has run.
     */

    /* ── Step 6: Synchronize — this is where ALL work happens ──────── */
    NPU_RT_CHECK(npuSynchronize(ctx));
    /*
     * Inside: npu_kdrv_flush → append CMD_END → firmware_run:
     *
     *   head=0: CMD_DMA_H2D { dst=0x40, src=h_A, bytes=64 }
     *     → drv_dma_h2d: calloc tmp[16], memcpy(tmp,h_A,64),
     *                    npu_write_i32(sim,0x40,tmp,16), free(tmp)
     *     → h_A now in sim->mem[0x40..0x7F]
     *
     *   head=1: CMD_DMA_H2D { dst=0x80, src=h_B, bytes=64 }
     *     → h_B now in sim->mem[0x80..0xBF]
     *
     *   head=2: CMD_LAUNCH_KERN { fn=matmul_4x4, args=pool[0], nargs=3 }
     *     → drv_do_launch:
     *         memset(sim->x, 0, 256)           zero register file
     *         sim->x[1]  = 8                   link = HALT at index 8
     *         sim->x[10] = 0xC0                x10 = d_C
     *         sim->x[11] = 0x40                x11 = d_A
     *         sim->x[12] = 0x80                x12 = d_B
     *         npu_run(sim, fn->prog, 9)
     *           instr 0: T_CFG_SHAPE 4 4 4
     *           instr 1: T_ACC_CLR
     *           instr 2: T_LOAD tile=0 ← reads sim->mem[0x40..0x7F] = A
     *           instr 3: T_LOAD tile=1 ← reads sim->mem[0x80..0xBF] = B
     *           instr 4: T_MMA  tile2=tile0×tile1  (4×4 matmul)
     *           instr 5: T_STORE tile=2 → sim->mem[0xC0..0xFF] = C = A×B
     *           instr 6..7: bookkeeping
     *           instr 8: C_HALT → npu_run returns
     *
     *   head=3: CMD_END → firmware_run returns
     *
     *   Reset: head=0, tail=0
     *   Recycle arg pool: pool[0].used = 0
     */

    /* ── Step 7: Read back results ──────────────────────────────────── */
    NPU_RT_CHECK(npuRead_f32(ctx, d_C, h_C, NELEMS));
    /*
     * Direct DRAM read: npu_read_i32(sim, 0xC0, tmp, 16)
     * h_C now holds C = A × B from the simulator
     */

    /* ── Step 8: Verify ──────────────────────────────────────────────── */
    int pass = 1;
    for (int i = 0; i < NELEMS; i++) {
        if (fabsf(h_C[i] - h_ref[i]) > 1e-3f) {
            fprintf(stderr, "MISMATCH [%d]: NPU=%.4f ref=%.4f\n",
                    i, h_C[i], h_ref[i]);
            pass = 0;
        }
    }
    printf("%s matmul_4x4\n", pass ? "PASS" : "FAIL");

    npuPrintStats(ctx);   /* print cmds_processed, dma bytes, kernel count */

    /* ── Step 9: Cleanup ────────────────────────────────────────────── */
    npuModuleUnload(mod);     /* free decoded prog[] arrays */
    npuFree(ctx, d_A);        /* heap_free(0x40) → coalesce */
    npuFree(ctx, d_B);
    npuFree(ctx, d_C);
    npuFinalize(ctx);         /* npu_kdrv_remove → npu_destroy → free(ctx) */

    return pass ? 0 : 1;
}
```

---

## 13.14 `npuPrintStats` — Observability

```c
void npuPrintStats(NpuCtx ctx)
{
    const npu_kdrv_s *drv = ctx->drv;
    printf("NPU stats:\n");
    printf("  Commands processed : %" PRIu64 "\n", drv->cmds_processed);
    printf("  DMA H2D bytes      : %" PRIu64 "\n", drv->dma_h2d_bytes);
    printf("  DMA D2H bytes      : %" PRIu64 "\n", drv->dma_d2h_bytes);
    printf("  Kernels launched   : %" PRIu64 "\n", drv->kernels_launched);
    printf("  GEMM ops           : %" PRIu64 "\n", drv->gemm_ops);
}
```

Sample output from `demo_matmul_elf`:

```
NPU stats:
  Commands processed : 4
  DMA H2D bytes      : 128
  DMA D2H bytes      : 0
  Kernels launched   : 1
  GEMM ops           : 0
```

(D2H bytes are 0 because `npuRead_f32` bypasses the command queue.)

---

## 13.15 `npu_runtime.h` — The Simpler Alternative

For quick demos, `npu_runtime.h` provides a synchronous API that skips the driver
layer entirely:

```c
/* npu_runtime.h — synchronous, no driver layer */
npu_ctx_t *npu_ctx_create(void);
void        npu_ctx_destroy(npu_ctx_t *ctx);
uint32_t    npu_malloc(npu_ctx_t *ctx, size_t bytes);
void        npu_free(npu_ctx_t *ctx, uint32_t addr);
void        npu_memcpy_h2d(npu_ctx_t *ctx, uint32_t dst, const void *src, size_t n);
void        npu_memcpy_d2h(npu_ctx_t *ctx, void *dst, uint32_t src, size_t n);
int         npu_module_load(npu_ctx_t *ctx, const char *path, npu_module_t **out);
npu_fn_t   *npu_module_get_fn(npu_module_t *mod, const char *name);
void        npu_launch(npu_ctx_t *ctx, npu_fn_t *fn, const npu_arg_t *args, int n);
```

`npu_launch` executes **immediately** — no queue, no synchronize.  This is simpler
to use for learning but doesn't expose the async batching model.

| Feature | `npu_runtime` | `npu_rt` |
|---------|:---:|:---:|
| Async command queue | — | ✓ |
| Kernel driver layer | — | ✓ |
| Arg copy pool | — | ✓ |
| Fence API | — | ✓ |
| `npuGemm` / `npuRelu` | — | ✓ |
| Ease of use | simpler | full-featured |

Use `npu_runtime` for the demos in `demo_vadd.c`.  Use `npu_rt` for anything that
resembles production ML workloads.

---

## 13.16 Summary

| API call | Sync/Async | Driver action |
|----------|-----------|---------------|
| `npuInit` | sync | `npu_kdrv_probe` → `npu_create` + `heap_init` |
| `npuMalloc` | sync | `npu_kdrv_alloc` → `heap_alloc` |
| `npuFree` | sync | `npu_kdrv_free` → `heap_free` + coalesce |
| `npuMemcpy_h2d` | async | push `CMD_DMA_H2D` |
| `npuMemcpy_d2h` | async | push `CMD_DMA_D2H` |
| `npuGemm` | async | push `CMD_TENSOR_GEMM` |
| `npuRelu` | async | push `CMD_TENSOR_RELU` |
| `npuLaunchKernel` | async | copy args to pool, push `CMD_LAUNCH_KERN` |
| `npuSynchronize` | sync | `npu_kdrv_flush` → `firmware_run`; recycle pool |
| `npuRead_f32` | sync | direct `npu_read_i32` (call after sync) |
| `npuModuleLoad` | sync | mmap, ELF parse, `npu_decode_cinst` per function |
| `npuGetFunction` | sync | linear name scan in module |
| `npuFenceWait` | sync | push signal, flush, `npu_kdrv_fence_wait` |
| `npuFinalize` | sync | unload modules, `npu_kdrv_remove`, `free(ctx)` |

The golden rule: **anything that affects DRAM content or kernel execution is async
— push now, run at `npuSynchronize`**.  Everything that returns a value you need
before the next push (addresses, function handles) is synchronous.
