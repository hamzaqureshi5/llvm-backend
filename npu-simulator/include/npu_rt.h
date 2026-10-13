/* npu_rt.h — CUDA-style runtime API layered on top of the NPU kernel driver.
 *
 * Stack:
 *
 *   Application
 *       │  npuInit / npuMalloc / npuMemcpy / npuLaunchKernel / npuSynchronize
 *       ▼
 *   npu_rt  (this layer)
 *       │  npu_kdrv_submit / npu_kdrv_flush / npu_kdrv_alloc
 *       ▼
 *   npu_kdrv  (kernel driver / firmware)
 *       │  npu_run / npu_write_i32 / npu_read_f32
 *       ▼
 *   npu simulator (npu.h)
 *
 * Async model:
 *   npuMemcpy / npuLaunchKernel / npuGemm / npuRelu push commands to the
 *   driver's command queue.  npuSynchronize flushes the queue and waits for
 *   all commands to complete.  npuMalloc / npuFree are always synchronous.
 *
 * Error handling:
 *   Every function returns NpuError.  Use NPU_RT_CHECK() to abort on error.
 */
#ifndef NPU_RT_H
#define NPU_RT_H

#include <stdint.h>
#include <stddef.h>
#include <stdio.h>
#include "npu_cmd.h"   /* npu_fn_s_t, npu_fn_t, npu_arg_t */

#ifdef __cplusplus
extern "C" {
#endif

/* ── opaque handles ──────────────────────────────────────────────────────── */

typedef struct npu_ctx_s    *NpuCtx;
typedef uint32_t             NpuDevPtr;
typedef struct npu_module_s *NpuModule;
typedef npu_fn_s_t          *NpuFunction;  /* same layout as npu_fn_t      */
typedef uint32_t             NpuFence;

#define NPU_NULL_PTR ((NpuDevPtr)0)

/* ── error codes ─────────────────────────────────────────────────────────── */

typedef enum {
    NPU_OK             = 0,
    NPU_ERR_INVALID    = 1,  /* null handle or bad argument                 */
    NPU_ERR_NO_MEM     = 2,  /* device DRAM heap or internal table full     */
    NPU_ERR_QUEUE_FULL = 3,  /* command queue full                          */
    NPU_ERR_IO         = 4,  /* file I/O error                              */
    NPU_ERR_NOT_FOUND  = 5,  /* function name not in module                 */
    NPU_ERR_BAD_MODULE = 6,  /* ELF parse error or ASM assembly error       */
} NpuError;

/* NpuMemcpyKind kept for internal use; user code should call the typed
 * functions below (npuMemcpy_h2d / d2h / d2d) instead. */
typedef enum {
    NPU_MEMCPY_H2D,
    NPU_MEMCPY_D2H,
    NPU_MEMCPY_D2D,
} NpuMemcpyKind;

/* ── device properties ───────────────────────────────────────────────────── */

typedef struct {
    char   name[64];       /* "NPU Simulator"                               */
    size_t dram_bytes;     /* total DRAM                                    */
    size_t scratch_bytes;  /* on-chip SRAM                                  */
    int    max_cores;      /* maximum simulated vector cores                */
} NpuDeviceProps;

/* ════════════════════════════════════════════════════════════════════════════
 * API
 * ════════════════════════════════════════════════════════════════════════════ */

/* ── device ──────────────────────────────────────────────────────────────── */

void     npuGetDeviceProps(NpuDeviceProps *props);

/* ── context ─────────────────────────────────────────────────────────────── */

NpuError npuInit    (NpuCtx *ctx);
NpuError npuFinalize(NpuCtx  ctx);
NpuError npuSetCores(NpuCtx  ctx, int cores);
NpuError npuSetTrace(NpuCtx  ctx, int enable);
NpuError npuResetStats(NpuCtx ctx);

/* ── memory (npuMalloc / npuFree are synchronous) ────────────────────────── */

NpuError npuMalloc(NpuCtx ctx, NpuDevPtr *ptr, size_t bytes);
NpuError npuFree  (NpuCtx ctx, NpuDevPtr  ptr);

/* ── async memory transfers (push command to queue) ─────────────────────── */

/* Host → Device: src is a host pointer, dst is a device address. */
NpuError npuMemcpy_h2d(NpuCtx ctx, NpuDevPtr dst, const void *src, size_t bytes);

/* Device → Host: dst is a host pointer, src is a device address. */
NpuError npuMemcpy_d2h(NpuCtx ctx, void *dst, NpuDevPtr src, size_t bytes);

/* Device → Device: both addresses are device addresses. */
NpuError npuMemcpy_d2d(NpuCtx ctx, NpuDevPtr dst, NpuDevPtr src, size_t bytes);

NpuError npuMemset(NpuCtx ctx, NpuDevPtr ptr, int32_t val, size_t count);

/* ── module loading (synchronous) ───────────────────────────────────────── */

NpuError npuModuleLoad   (NpuCtx ctx, NpuModule *mod, const char *elf_path);
NpuError npuModuleLoadAsm(NpuCtx ctx, NpuModule *mod, const char *src);
NpuError npuModuleUnload (NpuModule mod);
NpuError npuGetFunction  (NpuModule mod, NpuFunction *fn, const char *name);

/* ── tensor engine (async — push CMD_TENSOR_*) ───────────────────────────── */

NpuError npuGemm(NpuCtx ctx,
                 NpuDevPtr A, NpuDevPtr B, NpuDevPtr C,
                 int M, int K, int N);
NpuError npuRelu(NpuCtx ctx, NpuDevPtr src, NpuDevPtr dst, int M, int N);

/* ── kernel launch (async — push CMD_LAUNCH_KERN) ───────────────────────── *
 * The runtime copies the args array internally so the caller's stack copy
 * is safe to discard after this call returns.                              */

NpuError npuLaunchKernel(NpuCtx ctx, NpuFunction fn,
                          const npu_arg_t *args, int nargs);

/* ── synchronize — flush queue, execute all pending commands ─────────────── */

NpuError npuSynchronize(NpuCtx ctx);

/* ── fences ──────────────────────────────────────────────────────────────── */

NpuError npuFenceCreate(NpuCtx ctx, NpuFence *fence);
NpuError npuFenceWait  (NpuCtx ctx, NpuFence  fence);

/* ── result read-back (synchronous, use after npuSynchronize) ────────────── */

NpuError npuRead_f32(NpuCtx ctx, NpuDevPtr src, float   *dst, int count);
NpuError npuRead_i32(NpuCtx ctx, NpuDevPtr src, int32_t *dst, int count);

/* ── debug ───────────────────────────────────────────────────────────────── */

NpuError    npuMemDump  (NpuCtx ctx, NpuDevPtr ptr, size_t count, FILE *out);
NpuError    npuPrintStats(NpuCtx ctx);
void        npuHeapDump (NpuCtx ctx, FILE *out);
const char *npuGetErrorString(NpuError err);

/* ── convenience macro ───────────────────────────────────────────────────── */

#define NPU_RT_CHECK(call) \
    do { \
        NpuError _e = (call); \
        if (_e != NPU_OK) { \
            fprintf(stderr, "NPU error %s:%d  %s: %s\n", \
                    __FILE__, __LINE__, #call, npuGetErrorString(_e)); \
            abort(); \
        } \
    } while (0)

#ifdef __cplusplus
}
#endif

#endif /* NPU_RT_H */
