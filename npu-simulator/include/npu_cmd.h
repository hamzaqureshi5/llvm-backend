/* npu_cmd.h - Command packet definitions shared between kernel driver and
 * runtime.  Models the 32-byte command packets that fill the hardware command
 * queue, analogous to GPU command buffers.
 *
 * Every packet is exactly 32 bytes (verified with _Static_assert below).
 * Fields not relevant to a particular opcode should be left zero.
 */
#ifndef NPU_CMD_H
#define NPU_CMD_H

#include <stdint.h>
#include <stddef.h>
#include <assert.h>

/* ── command opcodes ─────────────────────────────────────────────────────── */

#define NPU_CMD_NOP          0x00  /* no-op, skip                            */
#define NPU_CMD_DMA_H2D      0x01  /* arg0=dst_dev, arg1=(uint64_t)src_host, arg2=bytes */
#define NPU_CMD_DMA_D2H      0x02  /* arg0=(uint64_t)dst_host, arg1=src_dev, arg2=bytes */
#define NPU_CMD_DMA_D2D      0x03  /* arg0=dst_dev, arg1=src_dev, arg2=bytes */
#define NPU_CMD_MEMSET_I32   0x04  /* arg0=dev_addr, arg1=i32_value, arg2=count_words */
#define NPU_CMD_LAUNCH_KERN  0x10  /* arg0=(uint64_t)fn_ptr, arg1=(uint64_t)args_ptr, extra=nargs */
#define NPU_CMD_TENSOR_GEMM  0x20  /* arg0=A_dev(float), arg1=B_dev, arg2=C_dev; extra=M|(K<<8)|(N<<16) */
#define NPU_CMD_TENSOR_RELU  0x21  /* arg0=src_dev(float), arg1=dst_dev; extra=M|(N<<16) */
#define NPU_CMD_FENCE_SIGNAL 0x40  /* arg0=fence_id                         */
#define NPU_CMD_BARRIER      0xFE  /* wait for all prior commands (no-op in sequential fw) */
#define NPU_CMD_END          0xFF  /* terminate firmware dispatch loop       */

/* ── command packet ──────────────────────────────────────────────────────── */

typedef struct {
    uint8_t  opcode;   /* NPU_CMD_* above                                   */
    uint8_t  flags;    /* reserved, set to 0                                */
    uint16_t pad;      /* reserved, set to 0                                */
    uint32_t extra;    /* opcode-specific small payload (dims, nargs, …)    */
    uint64_t arg0;     /* first argument (meaning depends on opcode)        */
    uint64_t arg1;     /* second argument                                   */
    uint64_t arg2;     /* third argument                                    */
} npu_cmd_t;

/* Guarantee the ABI is exactly 32 bytes so the command ring is simple to
 * index with a fixed stride. */
_Static_assert(sizeof(npu_cmd_t) == 32,
               "npu_cmd_t must be exactly 32 bytes");

/* ── kernel function descriptor ─────────────────────────────────────────── *
 * CMD_LAUNCH_KERN stores an (npu_fn_t) pointer — a pointer to this struct.
 * The firmware casts arg0 back to npu_fn_t and reads prog/len.            */

#include "npu.h"   /* for cinst typedef */

typedef struct npu_fn_s {
    char   name[128]; /* human-readable name (for debugging)               */
    cinst *prog;      /* heap-allocated cinst array assembled by the runtime */
    int    len;       /* number of instructions (including the HALT)        */
} npu_fn_s_t;

typedef npu_fn_s_t *npu_fn_t;

/* ── kernel argument descriptor ─────────────────────────────────────────── *
 * Matches the definition in npu_runtime.h but is reproduced here so that
 * npu_cmd.h is self-contained when included by npu_kdrv.c without pulling
 * in the old runtime header.  The layouts are binary-compatible.          */

typedef enum { NPU_ARG_I64 = 0, NPU_ARG_PTR = 1 } npu_arg_kind_t;

typedef struct {
    npu_arg_kind_t kind;
    union {
        int64_t  i64;
        uint32_t ptr;
    };
} npu_arg_t;

/* Convenience constructors — produce an npu_arg_t literal. */
#define NPU_I64(v)  ((npu_arg_t){NPU_ARG_I64, {.i64 = (int64_t)(v)}})
#define NPU_PTR(p)  ((npu_arg_t){NPU_ARG_PTR, {.ptr = (uint32_t)(p)}})

#endif /* NPU_CMD_H */
