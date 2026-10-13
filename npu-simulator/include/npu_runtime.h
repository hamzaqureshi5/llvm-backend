/*
 * npu_runtime.h — CUDA-style runtime API for the NPU simulator
 *
 * Three-layer model (mirrors CUDA):
 *
 *   User program
 *       │  npu_malloc / npu_memcpy / npu_launch
 *       ▼
 *   npu_runtime  (this library)
 *       │  npu_create / npu_run / npu_write_i32
 *       ▼
 *   npu simulator (npu.h / npu_asm.h)
 *
 * Concepts:
 *
 *   npu_ctx_t     Execution context.  Owns the simulator instance, the
 *                 device DRAM heap, and the loaded module table.  Like a
 *                 CUDA context or a Vulkan device.
 *
 *   npu_devptr_t  32-bit byte address in the NPU's DRAM flat address space.
 *                 Analogous to CUdeviceptr.  Returned by npu_malloc, passed
 *                 directly as kernel arguments.
 *
 *   npu_module_t  A loaded object file (.o from llc) or assembled source (.s).
 *                 Contains one or more compiled functions.  Like a CUDA module
 *                 (cubin / PTX).
 *
 *   npu_fn_t      A single kernel function inside a module.  Like a
 *                 CUfunction.  Passed to npu_launch together with its args.
 *
 *   npu_arg_t     One launch argument: either a raw 64-bit integer or a device
 *                 pointer.  Both become integer register values (x10, x11 …)
 *                 inside the kernel.  Use the NPU_I64() / NPU_PTR() macros.
 *
 * Calling convention (CC_NPU, matches the LLVM backend):
 *   x10 = arg0 (also return value)
 *   x11 = arg1
 *   x12 = arg2
 *   x13 = arg3
 *   …
 *   x1  = link register (set to HALT instruction index by the runtime)
 *   x0  = always zero
 *
 * Memory layout:
 *   [0x00000000, NPU_MEM_SIZE)   DRAM (managed by the heap allocator)
 *   [SCRATCH_BASE, …)            On-chip SRAM (accessed by kernel code only)
 *
 * Error handling:
 *   Every API function returns npu_error_t.  On failure the context records
 *   the last error, retrievable with npu_ctx_last_error().  Pass NPU_CHECK()
 *   calls through npu_check() for automatic abort-on-error.
 */

#pragma once
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ── error codes ─────────────────────────────────────────────────────────────── */

typedef enum {
    NPU_SUCCESS            =  0,
    NPU_ERROR_INVALID_CTX  =  1,  /* null or corrupt context                  */
    NPU_ERROR_INVALID_ARG  =  2,  /* null pointer or bad count passed to API  */
    NPU_ERROR_OUT_OF_MEM   =  3,  /* device DRAM heap exhausted               */
    NPU_ERROR_INVALID_ADDR =  4,  /* devptr not allocated by this context     */
    NPU_ERROR_BAD_MODULE   =  5,  /* ELF parse error or bad .s source         */
    NPU_ERROR_NOT_FOUND    =  6,  /* function name not in module              */
    NPU_ERROR_LAUNCH       =  7,  /* simulator aborted (runaway, fault)       */
    NPU_ERROR_IO           =  8,  /* file open / read failure                 */
    NPU_ERROR_TOO_MANY     =  9,  /* module/alloc table full                  */
} npu_error_t;

/* ── opaque handles ──────────────────────────────────────────────────────────── */

typedef struct npu_ctx_s    *npu_ctx_t;
typedef struct npu_module_s *npu_module_t;
typedef struct npu_fn_s     *npu_fn_t;
typedef uint32_t             npu_devptr_t;   /* byte address in DRAM           */

#define NPU_NULL ((npu_devptr_t)0)

/* ── kernel arguments ────────────────────────────────────────────────────────── */

typedef enum { NPU_ARG_I64 = 0, NPU_ARG_PTR = 1 } npu_arg_kind_t;

typedef struct {
    npu_arg_kind_t kind;
    union {
        int64_t     i64;
        npu_devptr_t ptr;
    };
} npu_arg_t;

/* Convenience constructors. */
#define NPU_I64(v)   ((npu_arg_t){ NPU_ARG_I64, { .i64 = (int64_t)(v)     }})
#define NPU_PTR(p)   ((npu_arg_t){ NPU_ARG_PTR, { .ptr = (npu_devptr_t)(p)}})

/* ── device properties ───────────────────────────────────────────────────────── */

typedef struct {
    char   name[64];          /* "NPU Simulator"                               */
    size_t dram_bytes;        /* total DRAM (default 16 MiB)                   */
    size_t scratch_bytes;     /* on-chip SRAM (default 256 KiB)                */
    int    max_cores;         /* maximum vector cores                           */
    int    max_kernels;       /* max registered vector kernels                  */
} npu_props_t;

/* ── heap snapshot (for debugging) ──────────────────────────────────────────── */

typedef struct {
    npu_devptr_t addr;
    size_t       size;
    int          free;
} npu_heap_block_t;

/* ════════════════════════════════════════════════════════════════════════════════
 * API
 * ════════════════════════════════════════════════════════════════════════════════ */

/* ── device ──────────────────────────────────────────────────────────────────── */

/* Query simulator device properties (does not require a context). */
void npu_get_props(npu_props_t *props);

/* ── context ─────────────────────────────────────────────────────────────────── */

/* Create a context backed by a fresh simulator instance.
 * The DRAM heap starts at byte 0 and grows upward. */
npu_error_t npu_ctx_create(npu_ctx_t *ctx);

/* Destroy context: unloads all modules, frees all device memory, destroys sim. */
npu_error_t npu_ctx_destroy(npu_ctx_t ctx);

/* Set the number of simulated vector cores (default: 1). */
npu_error_t npu_ctx_set_cores(npu_ctx_t ctx, int cores);

/* Enable/disable per-instruction trace output from the simulator. */
npu_error_t npu_ctx_set_trace(npu_ctx_t ctx, int enable);

/* Zero the cycle/instruction counters. */
npu_error_t npu_ctx_reset_stats(npu_ctx_t ctx);

/* Print cycle model statistics to stdout. */
npu_error_t npu_ctx_print_stats(npu_ctx_t ctx);

/* Return the last error recorded by this context (does not clear it). */
npu_error_t npu_ctx_last_error(npu_ctx_t ctx);

/* ── device memory ───────────────────────────────────────────────────────────── */

/* Allocate `bytes` from the DRAM heap.  *ptr is the device address.
 * Alignment: 64 bytes.  Returns NPU_ERROR_OUT_OF_MEM if the heap is full. */
npu_error_t npu_malloc(npu_ctx_t ctx, npu_devptr_t *ptr, size_t bytes);

/* Free a device allocation.  Coalesces adjacent free blocks. */
npu_error_t npu_free(npu_ctx_t ctx, npu_devptr_t ptr);

/* Copy bytes from host memory to DRAM.  bytes must be > 0. */
npu_error_t npu_memcpy_h2d(npu_ctx_t ctx, npu_devptr_t dst,
                            const void *src, size_t bytes);

/* Copy bytes from DRAM to host memory. */
npu_error_t npu_memcpy_d2h(npu_ctx_t ctx, void *dst,
                            npu_devptr_t src, size_t bytes);

/* Copy bytes between two DRAM regions (may overlap). */
npu_error_t npu_memcpy_d2d(npu_ctx_t ctx, npu_devptr_t dst,
                            npu_devptr_t src, size_t bytes);

/* Fill `count` i32 words starting at `ptr` with `value`. */
npu_error_t npu_memset_i32(npu_ctx_t ctx, npu_devptr_t ptr,
                            int32_t value, size_t count);

/* ── modules ─────────────────────────────────────────────────────────────────── */

/* Load an ELF64 relocatable object (.o) produced by:
 *   llc -mtriple=npu -filetype=obj -o out.o input.ll
 * Decodes all global functions and stores them in the module. */
npu_error_t npu_module_load(npu_ctx_t ctx, npu_module_t *mod,
                             const char *elf_path);

/* Assemble text source and load as a module.
 * src uses the same mnemonic syntax as llc output. */
npu_error_t npu_module_load_asm(npu_ctx_t ctx, npu_module_t *mod,
                                 const char *src);

/* Unload module and free all decoded function programs. */
npu_error_t npu_module_unload(npu_module_t mod);

/* Look up a function by name within a module. */
npu_error_t npu_module_get_fn(npu_module_t mod, npu_fn_t *fn,
                               const char *name);

/* ── launch ──────────────────────────────────────────────────────────────────── */

/* Launch a kernel function synchronously.
 *
 * args[0] → x10, args[1] → x11, …  (up to 8 arguments, x10–x17)
 * NPU_ARG_I64: value loaded directly into register.
 * NPU_ARG_PTR: device address (uint32) zero-extended to i64.
 *
 * Registers x0 and x1 are set by the runtime (zero / link).
 * The return value is left in x10 after the run; read with npu_ctx_retval(). */
npu_error_t npu_launch(npu_ctx_t ctx, npu_fn_t fn,
                        const npu_arg_t *args, int nargs);

/* Return the value of x10 after the most recent npu_launch(). */
int64_t npu_ctx_retval(npu_ctx_t ctx);

/* ── debug ───────────────────────────────────────────────────────────────────── */

/* Dump `count` i32 words from device memory at `ptr` to `out`. */
npu_error_t npu_mem_dump(npu_ctx_t ctx, npu_devptr_t ptr,
                          size_t count, FILE *out);

/* Print all 32 GPR values from the last run to `out`. */
npu_error_t npu_reg_dump(npu_ctx_t ctx, FILE *out);

/* Print the heap allocation table to `out`. */
npu_error_t npu_heap_dump(npu_ctx_t ctx, FILE *out);

/* ── error ───────────────────────────────────────────────────────────────────── */

/* Human-readable error string. */
const char *npu_strerror(npu_error_t err);

/* Abort with a message if err != NPU_SUCCESS.  Use in example programs. */
#define NPU_CHECK(call)                                                  \
    do {                                                                 \
        npu_error_t _e = (call);                                        \
        if (_e != NPU_SUCCESS) {                                        \
            fprintf(stderr, "NPU error at %s:%d — %s: %s\n",           \
                    __FILE__, __LINE__, #call, npu_strerror(_e));        \
            abort();                                                     \
        }                                                                \
    } while (0)

#ifdef __cplusplus
}
#endif
