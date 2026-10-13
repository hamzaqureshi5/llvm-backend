/* npu_kdrv.h - Simulated kernel driver interface.
 *
 * Models the Linux kernel-driver side of an NPU software stack.  In real
 * hardware this layer would run in kernel space and expose a ioctl/mmap
 * interface to user space.  Here it is a plain C library that owns:
 *
 *   - one npu_t simulator instance (the "hardware")
 *   - a DRAM heap allocator  (first-fit, 64-byte aligned)
 *   - a command ring buffer  (host-side, models DMA-coherent memory)
 *   - a lightweight fence table for ordering
 *
 * The runtime layer (npu_rt.h) is the only intended consumer of this API.
 * Application code should use npu_rt.h instead.
 *
 * Error model: every call returns kdrv_err_t; KDRV_OK == 0.
 */
#ifndef NPU_KDRV_H
#define NPU_KDRV_H

#include <stddef.h>
#include <stdint.h>
#include "npu_cmd.h"   /* npu_cmd_t, npu_fn_t */

#ifdef __cplusplus
extern "C" {
#endif

/* ── opaque driver handle ────────────────────────────────────────────────── */

typedef struct npu_kdrv_s *npu_kdrv_t;

/* ── error codes ─────────────────────────────────────────────────────────── */

typedef enum {
    KDRV_OK         = 0,  /* success                                        */
    KDRV_NO_MEM     = 1,  /* DRAM heap or internal table exhausted          */
    KDRV_QUEUE_FULL = 2,  /* command ring buffer full                       */
    KDRV_INVALID    = 3,  /* null handle, bad address, etc.                 */
    KDRV_IO_ERR     = 4,  /* unexpected internal error                      */
} kdrv_err_t;

/* ── device lifecycle ────────────────────────────────────────────────────── *
 * npu_kdrv_probe  — allocate driver struct, create simulator, init heap.
 * npu_kdrv_remove — destroy driver and simulator; frees *drv.
 * npu_kdrv_reset  — soft-reset simulator and flush any pending commands.   */

kdrv_err_t npu_kdrv_probe (npu_kdrv_t *out);
void       npu_kdrv_remove(npu_kdrv_t  drv);
void       npu_kdrv_reset (npu_kdrv_t  drv);

/* ── DRAM allocation ─────────────────────────────────────────────────────── *
 * npu_kdrv_alloc — first-fit allocator; *addr receives the DRAM byte address.
 * npu_kdrv_free  — release a previous allocation and coalesce neighbours.  */

kdrv_err_t npu_kdrv_alloc(npu_kdrv_t drv, uint32_t *addr, size_t bytes);
kdrv_err_t npu_kdrv_free (npu_kdrv_t drv, uint32_t  addr);

/* ── command queue ───────────────────────────────────────────────────────── *
 * npu_kdrv_submit — append one command packet to the ring.  Returns
 *                   KDRV_QUEUE_FULL if the ring is at capacity.
 * npu_kdrv_flush  — "ring the doorbell": run the firmware loop over every
 *                   queued command then reset head/tail to zero.           */

kdrv_err_t npu_kdrv_submit(npu_kdrv_t drv, const npu_cmd_t *cmd);
kdrv_err_t npu_kdrv_flush (npu_kdrv_t drv);

/* ── fence management ────────────────────────────────────────────────────── *
 * Fences are lightweight 0/1 flags allocated from a fixed table.
 * npu_kdrv_fence_create — allocate a new unsignaled fence; *fence_id is set.
 * npu_kdrv_fence_wait   — busy-waits (or checks) until the fence is signaled.
 *                         Since the firmware is synchronous this returns
 *                         immediately if called after npu_kdrv_flush.      */

kdrv_err_t npu_kdrv_fence_create(npu_kdrv_t drv, uint32_t *fence_id);
kdrv_err_t npu_kdrv_fence_wait  (npu_kdrv_t drv, uint32_t  fence_id);

/* ── power management (simulated) ───────────────────────────────────────── */

void npu_kdrv_clock_enable (npu_kdrv_t drv);
void npu_kdrv_clock_disable(npu_kdrv_t drv);

/* ── simulator access ────────────────────────────────────────────────────── *
 * The runtime layer uses this to read result data back from device memory
 * via npu_read_f32 / npu_read_i32 after a synchronize.                    */

npu_t *npu_kdrv_get_sim(npu_kdrv_t drv);

/* ── statistics ──────────────────────────────────────────────────────────── */

void npu_kdrv_print_stats(npu_kdrv_t drv);

#ifdef __cplusplus
}
#endif

#endif /* NPU_KDRV_H */
