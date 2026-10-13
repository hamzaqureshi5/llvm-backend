/* npu_kdrv.c - Simulated kernel driver + firmware implementation.
 *
 * Implements npu_kdrv.h.  Link with:
 *   gcc ... npu_kdrv.c npu.c npu_asm.c -lm
 *
 * Design notes:
 *   - The firmware (firmware_run) is a simple sequential command interpreter;
 *     in real hardware it would be an interrupt-driven control-core program.
 *   - The DRAM heap is a host-side metadata table; actual data lives in the
 *     simulator's npu_t::mem array and is accessed via npu_read/write_i32.
 *   - DMA helpers convert byte-level data to/from the i32-word API exposed
 *     by the simulator.
 */

#include "npu_kdrv.h"
#include "npu.h"
#include "npu_asm.h"

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <stdint.h>
#include <assert.h>

/* ════════════════════════════════════════════════════════════════════════════
 * Internal constants
 * ════════════════════════════════════════════════════════════════════════════ */

#define CMDQ_CAP     4096   /* max pending commands in the ring              */
#define MAX_FENCES   256    /* fence table size                              */
#define MAX_HEAP_BLK 2048   /* max heap metadata blocks                     */
#define HEAP_ALIGN   64     /* DRAM allocation granularity (bytes)          */
#define HEAP_GUARD   64     /* first 64 bytes reserved (null guard)         */

/* ════════════════════════════════════════════════════════════════════════════
 * Internal structs
 * ════════════════════════════════════════════════════════════════════════════ */

typedef struct {
    uint32_t base;   /* DRAM byte address                                   */
    uint32_t size;   /* bytes, always a multiple of HEAP_ALIGN              */
    int      used;   /* 1 = allocated, 0 = free                             */
} heap_blk_t;

struct npu_kdrv_s {
    npu_t      *sim;
    int         clk_enabled;

    /* DRAM heap metadata */
    heap_blk_t  heap[MAX_HEAP_BLK];
    int         nblocks;

    /* Command ring buffer (host-side, models DMA-coherent memory) */
    npu_cmd_t   cmdq[CMDQ_CAP];
    int         head;   /* consumer index (firmware reads from here)        */
    int         tail;   /* producer index (submit writes here)              */

    /* Fence table: 0 = pending, 1 = signaled */
    int         fences[MAX_FENCES];
    int         nfences;

    /* Statistics */
    uint64_t    cmds_processed;
    uint64_t    dma_h2d_bytes;
    uint64_t    dma_d2h_bytes;
    uint64_t    kernels_launched;
    uint64_t    gemm_ops;
};

/* ════════════════════════════════════════════════════════════════════════════
 * Heap helpers
 * ════════════════════════════════════════════════════════════════════════════ */

static uint32_t drv_align_up(uint32_t v, uint32_t a)
{
    return (v + a - 1u) & ~(a - 1u);
}

static void heap_init(npu_kdrv_t drv)
{
    drv->heap[0].base = HEAP_GUARD;
    drv->heap[0].size = NPU_MEM_SIZE - HEAP_GUARD;
    drv->heap[0].used = 0;
    drv->nblocks      = 1;
}

static kdrv_err_t heap_alloc(npu_kdrv_t drv, size_t bytes, uint32_t *out)
{
    uint32_t need = drv_align_up((uint32_t)bytes, HEAP_ALIGN);
    if (need == 0) need = HEAP_ALIGN;

    for (int i = 0; i < drv->nblocks; i++) {
        heap_blk_t *b = &drv->heap[i];
        if (b->used || b->size < need)
            continue;

        /* Split if remainder is worth keeping */
        if (b->size - need >= HEAP_ALIGN * 2u &&
            drv->nblocks < MAX_HEAP_BLK) {
            int tail = drv->nblocks;
            memmove(&drv->heap[i + 2], &drv->heap[i + 1],
                    (size_t)(tail - i - 1) * sizeof(heap_blk_t));
            drv->nblocks++;
            drv->heap[i + 1].base = b->base + need;
            drv->heap[i + 1].size = b->size - need;
            drv->heap[i + 1].used = 0;
            b->size = need;
        }
        b->used = 1;
        *out    = b->base;
        return KDRV_OK;
    }
    return KDRV_NO_MEM;
}

static kdrv_err_t heap_free(npu_kdrv_t drv, uint32_t addr)
{
    int found = -1;
    for (int i = 0; i < drv->nblocks; i++) {
        if (drv->heap[i].base == addr && drv->heap[i].used) {
            found = i;
            break;
        }
    }
    if (found < 0) return KDRV_INVALID;

    drv->heap[found].used = 0;

    /* Coalesce with next free block */
    if (found + 1 < drv->nblocks && !drv->heap[found + 1].used) {
        drv->heap[found].size += drv->heap[found + 1].size;
        memmove(&drv->heap[found + 1], &drv->heap[found + 2],
                (size_t)(drv->nblocks - found - 2) * sizeof(heap_blk_t));
        drv->nblocks--;
    }
    /* Coalesce with previous free block */
    if (found > 0 && !drv->heap[found - 1].used) {
        drv->heap[found - 1].size += drv->heap[found].size;
        memmove(&drv->heap[found], &drv->heap[found + 1],
                (size_t)(drv->nblocks - found - 1) * sizeof(heap_blk_t));
        drv->nblocks--;
    }
    return KDRV_OK;
}

/* ════════════════════════════════════════════════════════════════════════════
 * DMA helpers (used by the firmware dispatch loop)
 * ════════════════════════════════════════════════════════════════════════════ */

/* Copy `bytes` bytes from host pointer `src` into simulator DRAM at `dst`. */
static void drv_dma_h2d(npu_kdrv_t drv, uint32_t dst,
                         const void *src, size_t bytes)
{
    if (!bytes) return;
    size_t   nwords = (bytes + 3) / 4;
    int32_t *tmp    = (int32_t *)malloc(nwords * 4);
    if (!tmp) { fprintf(stderr, "npu_kdrv: h2d malloc failed\n"); return; }
    memset(tmp, 0, nwords * 4);
    memcpy(tmp, src, bytes);
    npu_write_i32(drv->sim, dst, tmp, (int)nwords);
    free(tmp);
    drv->dma_h2d_bytes += bytes;
}

/* Copy `bytes` bytes from simulator DRAM at `src` into host pointer `dst`. */
static void drv_dma_d2h(npu_kdrv_t drv, void *dst,
                         uint32_t src, size_t bytes)
{
    if (!bytes) return;
    size_t   nwords = (bytes + 3) / 4;
    int32_t *tmp    = (int32_t *)malloc(nwords * 4);
    if (!tmp) { fprintf(stderr, "npu_kdrv: d2h malloc failed\n"); return; }
    npu_read_i32(drv->sim, src, tmp, (int)nwords);
    memcpy(dst, tmp, bytes);
    free(tmp);
    drv->dma_d2h_bytes += bytes;
}

/* ════════════════════════════════════════════════════════════════════════════
 * Firmware — sequential command interpreter
 *
 * In real hardware this would be a program running on the control core.
 * Here it is a plain C function called synchronously from npu_kdrv_flush.
 * ════════════════════════════════════════════════════════════════════════════ */

static void firmware_run(npu_kdrv_t drv)
{
    while (drv->head < drv->tail) {
        const npu_cmd_t *cmd = &drv->cmdq[drv->head++];
        drv->cmds_processed++;

        switch (cmd->opcode) {

        /* ── NOP ─────────────────────────────────────────────────────────── */
        case NPU_CMD_NOP:
            break;

        /* ── DMA: host → device ──────────────────────────────────────────── */
        case NPU_CMD_DMA_H2D: {
            uint32_t    dst  = (uint32_t)cmd->arg0;
            const void *src  = (const void *)(uintptr_t)cmd->arg1;
            size_t      sz   = (size_t)cmd->arg2;
            drv_dma_h2d(drv, dst, src, sz);
            break;
        }

        /* ── DMA: device → host ──────────────────────────────────────────── */
        case NPU_CMD_DMA_D2H: {
            void    *dst = (void *)(uintptr_t)cmd->arg0;
            uint32_t src = (uint32_t)cmd->arg1;
            size_t   sz  = (size_t)cmd->arg2;
            drv_dma_d2h(drv, dst, src, sz);
            break;
        }

        /* ── DMA: device → device ────────────────────────────────────────── */
        case NPU_CMD_DMA_D2D: {
            uint32_t dst = (uint32_t)cmd->arg0;
            uint32_t src = (uint32_t)cmd->arg1;
            size_t   sz  = (size_t)cmd->arg2;
            if (!sz) break;
            size_t   nwords = (sz + 3) / 4;
            int32_t *tmp    = (int32_t *)malloc(nwords * 4);
            if (!tmp) { fprintf(stderr, "npu_kdrv: d2d malloc failed\n"); break; }
            npu_read_i32 (drv->sim, src, tmp, (int)nwords);
            npu_write_i32(drv->sim, dst, tmp, (int)nwords);
            free(tmp);
            break;
        }

        /* ── MEMSET i32 ──────────────────────────────────────────────────── */
        case NPU_CMD_MEMSET_I32: {
            uint32_t addr  = (uint32_t)cmd->arg0;
            int32_t  val   = (int32_t)cmd->arg1;
            size_t   count = (size_t)cmd->arg2;
            if (!count) break;
            int32_t *tmp = (int32_t *)malloc(count * 4);
            if (!tmp) { fprintf(stderr, "npu_kdrv: memset malloc failed\n"); break; }
            for (size_t i = 0; i < count; i++) tmp[i] = val;
            npu_write_i32(drv->sim, addr, tmp, (int)count);
            free(tmp);
            break;
        }

        /* ── LAUNCH_KERN: run a control-core function ────────────────────── */
        case NPU_CMD_LAUNCH_KERN: {
            npu_fn_t    fn   = (npu_fn_t)(uintptr_t)cmd->arg0;
            npu_arg_t  *args = (npu_arg_t *)(uintptr_t)cmd->arg1;
            int         nargs = (int)(uint32_t)cmd->extra;

            if (!fn || !fn->prog || fn->len <= 0) {
                fprintf(stderr, "npu_kdrv: LAUNCH_KERN: invalid fn\n");
                break;
            }
            if (nargs < 0 || nargs > 8) {
                fprintf(stderr, "npu_kdrv: LAUNCH_KERN: nargs out of range (%d)\n", nargs);
                break;
            }

            /* Set up register file per calling convention. */
            memset(drv->sim->x, 0, sizeof(drv->sim->x));

            /* x1 = link register = index of HALT instruction */
            drv->sim->x[1] = fn->len - 1;

            fprintf(stderr,
                "[npu_kdrv] LAUNCH \"%s\"  insns=%d  x1(link)=%lld\n",
                fn->name, fn->len, (long long)drv->sim->x[1]);

            /* x10..x17 = args */
            for (int i = 0; i < nargs; i++) {
                int64_t val = 0;
                if (args[i].kind == NPU_ARG_PTR)
                    val = (int64_t)(uint64_t)args[i].ptr;
                else
                    val = args[i].i64;
                drv->sim->x[10 + i] = val;
                fprintf(stderr, "  x%d = 0x%llx (%lld)  [%s]\n",
                        10 + i, (unsigned long long)val, (long long)val,
                        args[i].kind == NPU_ARG_PTR ? "ptr" : "i64");
            }
            fprintf(stderr, "[npu_kdrv] --- execution trace ---\n");

            npu_run(drv->sim, fn->prog, fn->len);

            fprintf(stderr, "[npu_kdrv] --- execution complete ---\n");
            drv->kernels_launched++;
            break;
        }

        /* ── TENSOR_GEMM: C = A × B using the tensor engine ─────────────── *
         * extra = M | (K << 8) | (N << 16)                                 *
         * Builds a minimal cinst sequence:                                  *
         *   T_CFG_SHAPE M K N                                               *
         *   T_LOAD  dt=0  rs1=11  count=M*K   (A tile)                     *
         *   T_LOAD  dt=1  rs1=12  count=K*N   (B tile)                     *
         *   T_MATMUL dt=2 a=0 b=1                                           *
         *   T_STORE dt=2  rs1=13  count=M*N   (C tile)                     *
         *   C_HALT                                                           */
        case NPU_CMD_TENSOR_GEMM: {
            uint32_t A_dev = (uint32_t)cmd->arg0;
            uint32_t B_dev = (uint32_t)cmd->arg1;
            uint32_t C_dev = (uint32_t)cmd->arg2;
            int M = (int)( cmd->extra        & 0xFFu);
            int K = (int)((cmd->extra >>  8) & 0xFFu);
            int N = (int)((cmd->extra >> 16) & 0xFFu);

            if (M <= 0 || K <= 0 || N <= 0) {
                fprintf(stderr, "npu_kdrv: TENSOR_GEMM: bad dims M=%d K=%d N=%d\n", M, K, N);
                break;
            }

            /* Build the cinst micro-program on the stack. */
            cinst prog[6];
            memset(prog, 0, sizeof(prog));

            /* 0: T_CFG_SHAPE */
            prog[0].op = T_CFG_SHAPE;
            prog[0].M  = M;
            prog[0].K  = K;
            prog[0].N  = N;

            /* 1: T_LOAD A → tile 0 from address in x[11] */
            prog[1].op       = T_LOAD;
            prog[1].dt       = 0;
            prog[1].rs1      = 11;
            prog[1].count    = (uint32_t)(M * K);
            prog[1].use_base = 1;

            /* 2: T_LOAD B → tile 1 from address in x[12] */
            prog[2].op       = T_LOAD;
            prog[2].dt       = 1;
            prog[2].rs1      = 12;
            prog[2].count    = (uint32_t)(K * N);
            prog[2].use_base = 1;

            /* 3: T_MATMUL result → tile 2, inputs from tile 0 and tile 1 */
            prog[3].op = T_MATMUL;
            prog[3].dt = 2;
            prog[3].a  = 0;
            prog[3].b  = 1;

            /* 4: T_STORE tile 2 → address in x[13] */
            prog[4].op       = T_STORE;
            prog[4].dt       = 2;
            prog[4].rs1      = 13;
            prog[4].count    = (uint32_t)(M * N);
            prog[4].use_base = 1;

            /* 5: C_HALT */
            prog[5].op = C_HALT;

            /* Set up registers: zero all, then set addresses and link reg. */
            memset(drv->sim->x, 0, sizeof(drv->sim->x));
            drv->sim->x[1]  = 5; /* HALT is at index 5 */
            drv->sim->x[11] = (int64_t)(uint64_t)A_dev;
            drv->sim->x[12] = (int64_t)(uint64_t)B_dev;
            drv->sim->x[13] = (int64_t)(uint64_t)C_dev;

            npu_run(drv->sim, prog, 6);
            drv->gemm_ops++;
            break;
        }

        /* ── TENSOR_RELU: dst[i] = max(0, src[i]) ───────────────────────── *
         * extra = M | (N << 16)                                             *
         *   T_CFG_SHAPE M 0 N                                               *
         *   T_LOAD  dt=0  rs1=11  count=M*N                                 *
         *   T_RELU  dt=2  a=0                                               *
         *   T_STORE dt=2  rs1=12  count=M*N                                 *
         *   C_HALT                                                           */
        case NPU_CMD_TENSOR_RELU: {
            uint32_t src_dev = (uint32_t)cmd->arg0;
            uint32_t dst_dev = (uint32_t)cmd->arg1;
            int M = (int)( cmd->extra        & 0xFFFFu);
            int N = (int)((cmd->extra >> 16) & 0xFFFFu);

            if (M <= 0 || N <= 0) {
                fprintf(stderr, "npu_kdrv: TENSOR_RELU: bad dims M=%d N=%d\n", M, N);
                break;
            }

            cinst prog[5];
            memset(prog, 0, sizeof(prog));

            /* 0: T_CFG_SHAPE — K unused for elementwise, set to 0 */
            prog[0].op = T_CFG_SHAPE;
            prog[0].M  = M;
            prog[0].K  = 1;   /* K=1 to avoid divide-by-zero in simulator */
            prog[0].N  = N;

            /* 1: T_LOAD src → tile 0 */
            prog[1].op       = T_LOAD;
            prog[1].dt       = 0;
            prog[1].rs1      = 11;
            prog[1].count    = (uint32_t)(M * N);
            prog[1].use_base = 1;

            /* 2: T_RELU tile 0 → tile 2 */
            prog[2].op = T_RELU;
            prog[2].dt = 2;
            prog[2].a  = 0;

            /* 3: T_STORE tile 2 → dst */
            prog[3].op       = T_STORE;
            prog[3].dt       = 2;
            prog[3].rs1      = 12;
            prog[3].count    = (uint32_t)(M * N);
            prog[3].use_base = 1;

            /* 4: C_HALT */
            prog[4].op = C_HALT;

            memset(drv->sim->x, 0, sizeof(drv->sim->x));
            drv->sim->x[1]  = 4;
            drv->sim->x[11] = (int64_t)(uint64_t)src_dev;
            drv->sim->x[12] = (int64_t)(uint64_t)dst_dev;

            npu_run(drv->sim, prog, 5);
            break;
        }

        /* ── FENCE_SIGNAL ────────────────────────────────────────────────── */
        case NPU_CMD_FENCE_SIGNAL: {
            uint32_t fid = (uint32_t)cmd->arg0;
            if ((int)fid < drv->nfences)
                drv->fences[fid] = 1;
            break;
        }

        /* ── BARRIER — no-op (sequential execution) ──────────────────────── */
        case NPU_CMD_BARRIER:
            break;

        /* ── END — stop firmware processing this batch ───────────────────── */
        case NPU_CMD_END:
            return;

        default:
            fprintf(stderr, "npu_kdrv: firmware: unknown opcode 0x%02x\n",
                    cmd->opcode);
            break;
        }
    }
}

/* ════════════════════════════════════════════════════════════════════════════
 * Device lifecycle
 * ════════════════════════════════════════════════════════════════════════════ */

kdrv_err_t npu_kdrv_probe(npu_kdrv_t *out)
{
    if (!out) return KDRV_INVALID;

    npu_kdrv_t drv = (npu_kdrv_t)calloc(1, sizeof(struct npu_kdrv_s));
    if (!drv) return KDRV_NO_MEM;

    drv->sim = npu_create();
    if (!drv->sim) { free(drv); return KDRV_NO_MEM; }

    heap_init(drv);
    drv->clk_enabled = 1;

    *out = drv;
    return KDRV_OK;
}

void npu_kdrv_remove(npu_kdrv_t drv)
{
    if (!drv) return;
    npu_destroy(drv->sim);
    free(drv);
}

void npu_kdrv_reset(npu_kdrv_t drv)
{
    if (!drv) return;
    npu_reset(drv->sim);
    drv->head   = 0;
    drv->tail   = 0;
    /* Reset fence table */
    memset(drv->fences, 0, sizeof(drv->fences));
    drv->nfences = 0;
    /* Re-initialise heap */
    heap_init(drv);
    /* Clear stats */
    drv->cmds_processed  = 0;
    drv->dma_h2d_bytes   = 0;
    drv->dma_d2h_bytes   = 0;
    drv->kernels_launched = 0;
    drv->gemm_ops         = 0;
}

/* ════════════════════════════════════════════════════════════════════════════
 * DRAM allocation
 * ════════════════════════════════════════════════════════════════════════════ */

kdrv_err_t npu_kdrv_alloc(npu_kdrv_t drv, uint32_t *addr, size_t bytes)
{
    if (!drv || !addr || !bytes) return KDRV_INVALID;
    return heap_alloc(drv, bytes, addr);
}

kdrv_err_t npu_kdrv_free(npu_kdrv_t drv, uint32_t addr)
{
    if (!drv || !addr) return KDRV_INVALID;
    return heap_free(drv, addr);
}

/* ════════════════════════════════════════════════════════════════════════════
 * Command queue
 * ════════════════════════════════════════════════════════════════════════════ */

kdrv_err_t npu_kdrv_submit(npu_kdrv_t drv, const npu_cmd_t *cmd)
{
    if (!drv || !cmd) return KDRV_INVALID;
    if (drv->tail >= CMDQ_CAP) return KDRV_QUEUE_FULL;
    drv->cmdq[drv->tail++] = *cmd;
    return KDRV_OK;
}

kdrv_err_t npu_kdrv_flush(npu_kdrv_t drv)
{
    if (!drv) return KDRV_INVALID;

    /* Append an END sentinel so firmware_run always terminates cleanly. */
    if (drv->tail < CMDQ_CAP) {
        npu_cmd_t end;
        memset(&end, 0, sizeof(end));
        end.opcode = NPU_CMD_END;
        drv->cmdq[drv->tail++] = end;
    }

    firmware_run(drv);

    /* Reset ring so the next submit batch starts fresh. */
    drv->head = 0;
    drv->tail = 0;
    return KDRV_OK;
}

/* ════════════════════════════════════════════════════════════════════════════
 * Fence management
 * ════════════════════════════════════════════════════════════════════════════ */

kdrv_err_t npu_kdrv_fence_create(npu_kdrv_t drv, uint32_t *fence_id)
{
    if (!drv || !fence_id)          return KDRV_INVALID;
    if (drv->nfences >= MAX_FENCES) return KDRV_NO_MEM;

    int id = drv->nfences++;
    drv->fences[id] = 0;   /* unsignaled */
    *fence_id = (uint32_t)id;
    return KDRV_OK;
}

kdrv_err_t npu_kdrv_fence_wait(npu_kdrv_t drv, uint32_t fence_id)
{
    if (!drv) return KDRV_INVALID;
    if ((int)fence_id >= drv->nfences) return KDRV_INVALID;
    /* The firmware is synchronous: if the caller got here after flush the
     * fence has already been signaled.  A simple check suffices. */
    if (!drv->fences[fence_id]) {
        /* Not yet signaled — caller should have flushed first. */
        fprintf(stderr, "npu_kdrv: fence_wait: fence %u not yet signaled "
                        "(did you forget to call npu_kdrv_flush?)\n", fence_id);
        return KDRV_INVALID;
    }
    return KDRV_OK;
}

/* ════════════════════════════════════════════════════════════════════════════
 * Power management (simulated)
 * ════════════════════════════════════════════════════════════════════════════ */

void npu_kdrv_clock_enable(npu_kdrv_t drv)
{
    if (drv) drv->clk_enabled = 1;
}

void npu_kdrv_clock_disable(npu_kdrv_t drv)
{
    if (drv) drv->clk_enabled = 0;
}

/* ════════════════════════════════════════════════════════════════════════════
 * Simulator access
 * ════════════════════════════════════════════════════════════════════════════ */

npu_t *npu_kdrv_get_sim(npu_kdrv_t drv)
{
    return drv ? drv->sim : NULL;
}

/* ════════════════════════════════════════════════════════════════════════════
 * Statistics
 * ════════════════════════════════════════════════════════════════════════════ */

void npu_kdrv_print_stats(npu_kdrv_t drv)
{
    if (!drv) return;
    printf("=== NPU kernel-driver statistics ===\n");
    printf("  Commands processed : %llu\n", (unsigned long long)drv->cmds_processed);
    printf("  DMA H2D bytes      : %llu\n", (unsigned long long)drv->dma_h2d_bytes);
    printf("  DMA D2H bytes      : %llu\n", (unsigned long long)drv->dma_d2h_bytes);
    printf("  Kernels launched   : %llu\n", (unsigned long long)drv->kernels_launched);
    printf("  GEMM operations    : %llu\n", (unsigned long long)drv->gemm_ops);
    printf("--- Simulator stats ---\n");
    npu_print_stats(drv->sim);
}
