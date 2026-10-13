/*
 * demo_vadd.c — Vector-add kernel using the NPU CUDA-style runtime.
 *
 * Demonstrates the full workflow:
 *
 *   1. Create context
 *   2. Allocate device DRAM buffers    (npu_malloc)
 *   3. Upload host data                (npu_memcpy_h2d)
 *   4. Load a kernel from inline asm   (npu_module_load_asm)
 *   5. Dispatch the kernel             (npu_launch)
 *   6. Download result                 (npu_memcpy_d2h)
 *   7. Verify + print
 *   8. Clean up
 *
 * The same program also shows: heap dump, register dump, stat dump, and a
 * scalar reduction kernel running in the same context.
 *
 * Build:
 *   gcc -O2 -o demo_vadd demo_vadd.c npu_runtime.c npu.c npu_asm.c -lm
 * Run:
 *   ./demo_vadd
 */

#include "npu_runtime.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

/* ─── kernel source ─────────────────────────────────────────────────────────
 *
 * void vadd(int32_t *c, int32_t *a, int32_t *b, int64_t n)
 *
 * Calling convention: x10=c, x11=a, x12=b, x13=n
 *
 * Algorithm: iterate n times, incrementing all three base pointers by 4 each
 * step (one i32 element per iteration).
 */
static const char *VADD_SRC =
    "vadd:\n"
    "    LI   x14, 0\n"          /* i = 0                                 */
    ".Lvadd_loop:\n"
    "    LW   x15, 0(x11)\n"     /* x15 = a[i]                            */
    "    LW   x16, 0(x12)\n"     /* x16 = b[i]                            */
    "    ADD  x15, x15, x16\n"   /* x15 = a[i] + b[i]                     */
    "    SW   x15, 0(x10)\n"     /* c[i] = x15                            */
    "    ADDI x10, x10, 4\n"     /* c++                                   */
    "    ADDI x11, x11, 4\n"     /* a++                                   */
    "    ADDI x12, x12, 4\n"     /* b++                                   */
    "    ADDI x14, x14, 1\n"     /* i++                                   */
    "    BNE  x14, x13, .Lvadd_loop\n"
    "    JALR x0, x1, 0\n"       /* return                                */
    "    HALT\n";

/* ─── scalar dot-product kernel ─────────────────────────────────────────────
 *
 * int64_t dot(int32_t *a, int32_t *b, int64_t n)
 * Returns x10 = sum(a[i] * b[i])
 */
static const char *DOT_SRC =
    "dot:\n"
    "    LI   x14, 0\n"          /* acc = 0                               */
    "    LI   x15, 0\n"          /* i   = 0                               */
    ".Ldot_loop:\n"
    "    LW   x16, 0(x10)\n"     /* x16 = a[i]                            */
    "    LW   x17, 0(x11)\n"     /* x17 = b[i]                            */
    "    MUL  x16, x16, x17\n"   /* x16 = a[i] * b[i]                     */
    "    ADD  x14, x14, x16\n"   /* acc += x16                            */
    "    ADDI x10, x10, 4\n"     /* a++                                   */
    "    ADDI x11, x11, 4\n"     /* b++                                   */
    "    ADDI x15, x15, 1\n"     /* i++                                   */
    "    BNE  x15, x12, .Ldot_loop\n"
    "    OR   x10, x14, x0\n"    /* return value = acc                    */
    "    JALR x0, x1, 0\n"
    "    HALT\n";

/* ─── saxpy kernel ───────────────────────────────────────────────────────────
 *
 * void saxpy(int32_t *y, int32_t alpha, int32_t *x, int64_t n)
 * y[i] = y[i] + alpha * x[i]
 * x10=y, x11=alpha, x12=x, x13=n
 */
static const char *SAXPY_SRC =
    "saxpy:\n"
    "    LI   x14, 0\n"
    ".Lsaxpy_loop:\n"
    "    LW   x15, 0(x10)\n"     /* y[i]                                  */
    "    LW   x16, 0(x12)\n"     /* x[i]                                  */
    "    MUL  x16, x16, x11\n"   /* alpha * x[i]                          */
    "    ADD  x15, x15, x16\n"   /* y[i] + alpha*x[i]                     */
    "    SW   x15, 0(x10)\n"
    "    ADDI x10, x10, 4\n"
    "    ADDI x12, x12, 4\n"
    "    ADDI x14, x14, 1\n"
    "    BNE  x14, x13, .Lsaxpy_loop\n"
    "    JALR x0, x1, 0\n"
    "    HALT\n";

/* ─── helpers ────────────────────────────────────────────────────────────── */

static void separator(const char *title)
{
    printf("\n══ %s ", title);
    for (int i = (int)strlen(title); i < 50; i++) putchar('=');
    putchar('\n');
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Demo 1 — vadd: c[i] = a[i] + b[i]
 * ═══════════════════════════════════════════════════════════════════════════ */
static void demo_vadd(npu_ctx_t ctx)
{
    separator("Demo 1: vadd (c = a + b)");

    const int N = 256;

    /* 1. Allocate device buffers. */
    npu_devptr_t d_a, d_b, d_c;
    NPU_CHECK(npu_malloc(ctx, &d_a, N * sizeof(int32_t)));
    NPU_CHECK(npu_malloc(ctx, &d_b, N * sizeof(int32_t)));
    NPU_CHECK(npu_malloc(ctx, &d_c, N * sizeof(int32_t)));

    printf("  Buffers:  d_a=0x%08x  d_b=0x%08x  d_c=0x%08x\n",
           d_a, d_b, d_c);

    /* 2. Initialise host data. */
    int32_t h_a[256], h_b[256], h_c[256];
    for (int i = 0; i < N; i++) { h_a[i] = i; h_b[i] = N - i; }

    /* 3. Upload. */
    NPU_CHECK(npu_memcpy_h2d(ctx, d_a, h_a, N * sizeof(int32_t)));
    NPU_CHECK(npu_memcpy_h2d(ctx, d_b, h_b, N * sizeof(int32_t)));
    NPU_CHECK(npu_memset_i32(ctx, d_c, 0, N));

    /* 4. Load kernel. */
    npu_module_t mod;
    NPU_CHECK(npu_module_load_asm(ctx, &mod, VADD_SRC));

    npu_fn_t fn_vadd;
    NPU_CHECK(npu_module_get_fn(mod, &fn_vadd, "vadd"));

    /* 5. Launch: vadd(c, a, b, N) */
    npu_arg_t args[] = { NPU_PTR(d_c), NPU_PTR(d_a), NPU_PTR(d_b), NPU_I64(N) };
    NPU_CHECK(npu_launch(ctx, fn_vadd, args, 4));

    /* 6. Download. */
    NPU_CHECK(npu_memcpy_d2h(ctx, h_c, d_c, N * sizeof(int32_t)));

    /* 7. Verify. */
    int ok = 1;
    for (int i = 0; i < N; i++) {
        if (h_c[i] != h_a[i] + h_b[i]) { ok = 0; break; }
    }
    printf("  Result:   vadd[0..%d] = %s\n", N - 1, ok ? "PASS" : "FAIL");

    /* Spot-check a few values. */
    printf("  Sample:   c[0]=%d  c[1]=%d  c[127]=%d  c[255]=%d\n",
           h_c[0], h_c[1], h_c[127], h_c[255]);

    /* Stats for this launch. */
    npu_ctx_print_stats(ctx);

    /* 8. Free. */
    npu_free(ctx, d_a);
    npu_free(ctx, d_b);
    npu_free(ctx, d_c);
    npu_module_unload(mod);
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Demo 2 — dot product
 * ═══════════════════════════════════════════════════════════════════════════ */
static void demo_dot(npu_ctx_t ctx)
{
    separator("Demo 2: dot product");

    const int N = 8;
    int32_t h_a[] = {1, 2, 3, 4, 5, 6, 7, 8};   /* dot(a,a) = 204 */
    int32_t h_b[] = {1, 2, 3, 4, 5, 6, 7, 8};

    npu_devptr_t d_a, d_b;
    NPU_CHECK(npu_malloc(ctx, &d_a, N * sizeof(int32_t)));
    NPU_CHECK(npu_malloc(ctx, &d_b, N * sizeof(int32_t)));
    NPU_CHECK(npu_memcpy_h2d(ctx, d_a, h_a, N * sizeof(int32_t)));
    NPU_CHECK(npu_memcpy_h2d(ctx, d_b, h_b, N * sizeof(int32_t)));

    npu_module_t mod;
    NPU_CHECK(npu_module_load_asm(ctx, &mod, DOT_SRC));
    npu_fn_t fn_dot;
    NPU_CHECK(npu_module_get_fn(mod, &fn_dot, "dot"));

    npu_arg_t args[] = { NPU_PTR(d_a), NPU_PTR(d_b), NPU_I64(N) };
    NPU_CHECK(npu_launch(ctx, fn_dot, args, 3));

    int64_t result = npu_ctx_retval(ctx);
    printf("  dot([1..8],[1..8]) = %lld  (expected 204)  %s\n",
           (long long)result, result == 204 ? "PASS" : "FAIL");

    npu_free(ctx, d_a);
    npu_free(ctx, d_b);
    npu_module_unload(mod);
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Demo 3 — saxpy: y[i] = y[i] + alpha * x[i]
 * ═══════════════════════════════════════════════════════════════════════════ */
static void demo_saxpy(npu_ctx_t ctx)
{
    separator("Demo 3: saxpy (y += alpha * x)");

    const int N     = 4;
    const int alpha = 3;
    int32_t h_y[] = {10, 20, 30, 40};
    int32_t h_x[] = { 1,  2,  3,  4};
    /* expected: {13, 26, 39, 52} */

    npu_devptr_t d_y, d_x;
    NPU_CHECK(npu_malloc(ctx, &d_y, N * sizeof(int32_t)));
    NPU_CHECK(npu_malloc(ctx, &d_x, N * sizeof(int32_t)));
    NPU_CHECK(npu_memcpy_h2d(ctx, d_y, h_y, N * sizeof(int32_t)));
    NPU_CHECK(npu_memcpy_h2d(ctx, d_x, h_x, N * sizeof(int32_t)));

    npu_module_t mod;
    NPU_CHECK(npu_module_load_asm(ctx, &mod, SAXPY_SRC));
    npu_fn_t fn_saxpy;
    NPU_CHECK(npu_module_get_fn(mod, &fn_saxpy, "saxpy"));

    /* saxpy(y, alpha, x, N) */
    npu_arg_t args[] = { NPU_PTR(d_y), NPU_I64(alpha), NPU_PTR(d_x), NPU_I64(N) };
    NPU_CHECK(npu_launch(ctx, fn_saxpy, args, 4));

    int32_t h_out[4];
    NPU_CHECK(npu_memcpy_d2h(ctx, h_out, d_y, N * sizeof(int32_t)));

    int32_t exp[] = {13, 26, 39, 52};
    int ok = memcmp(h_out, exp, N * sizeof(int32_t)) == 0;
    printf("  y = [%d, %d, %d, %d]  (expected [13,26,39,52])  %s\n",
           h_out[0], h_out[1], h_out[2], h_out[3], ok ? "PASS" : "FAIL");

    npu_free(ctx, d_y);
    npu_free(ctx, d_x);
    npu_module_unload(mod);
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Demo 4 — memcpy_d2d + reg dump + heap dump
 * ═══════════════════════════════════════════════════════════════════════════ */
static void demo_debug(npu_ctx_t ctx)
{
    separator("Demo 4: d2d memcpy + heap dump + reg dump");

    /* d2d: copy 4 words from one device region to another. */
    int32_t src_data[] = {100, 200, 300, 400};
    npu_devptr_t d_src, d_dst;
    NPU_CHECK(npu_malloc(ctx, &d_src, sizeof(src_data)));
    NPU_CHECK(npu_malloc(ctx, &d_dst, sizeof(src_data)));

    NPU_CHECK(npu_memcpy_h2d(ctx, d_src, src_data, sizeof(src_data)));
    NPU_CHECK(npu_memcpy_d2d(ctx, d_dst, d_src, sizeof(src_data)));

    /* Verify via d2h. */
    int32_t dst_data[4] = {0};
    NPU_CHECK(npu_memcpy_d2h(ctx, dst_data, d_dst, sizeof(dst_data)));
    printf("  d2d copy: [%d,%d,%d,%d]  %s\n",
           dst_data[0], dst_data[1], dst_data[2], dst_data[3],
           memcmp(src_data, dst_data, sizeof(src_data)) == 0 ? "PASS" : "FAIL");

    /* Dump a small memory region. */
    printf("\n");
    npu_mem_dump(ctx, d_src, 4, stdout);

    /* Heap state. */
    printf("\n");
    npu_heap_dump(ctx, stdout);

    /* Register file from the last launch (saxpy). */
    printf("\n");
    npu_reg_dump(ctx, stdout);

    npu_free(ctx, d_src);
    npu_free(ctx, d_dst);
}

/* ─── main ──────────────────────────────────────────────────────────────── */

int main(void)
{
    printf("NPU Runtime Demo\n");
    printf("================\n");

    /* Print device properties. */
    npu_props_t props;
    npu_get_props(&props);
    printf("Device: %s\n", props.name);
    printf("  DRAM: %.1f MiB  SRAM: %zu KiB  max_cores: %d\n",
           props.dram_bytes / (1024.0 * 1024.0),
           props.scratch_bytes / 1024,
           props.max_cores);

    /* Create a context. */
    npu_ctx_t ctx;
    NPU_CHECK(npu_ctx_create(&ctx));

    demo_vadd(ctx);
    demo_dot(ctx);
    demo_saxpy(ctx);
    demo_debug(ctx);

    /* Cumulative stats across all four demos. */
    separator("Total stats");
    npu_ctx_print_stats(ctx);

    npu_ctx_destroy(ctx);
    printf("\nAll demos done.\n");
    return 0;
}
