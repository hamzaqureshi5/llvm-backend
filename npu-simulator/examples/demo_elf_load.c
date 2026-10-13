/* demo_elf_load.c — load an LLVM-compiled NPU kernel (.o) and run it.
 *
 * Workflow:
 *   1. llc -mtriple=npu -filetype=obj -o build/vadd.o kernels/vadd.ll
 *   2. ./build/demo_elf_load build/vadd.o          (normal)
 *   2. ./build/demo_elf_load build/vadd.o --trace  (instruction-level trace)
 *
 * The kernel: void vadd(i32 *c, i32 *a, i32 *b, i64 n)  — c[i] = a[i] + b[i]
 * Args land in x10–x13 per the NPU calling convention.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "npu_rt.h"

/* Keep N small when tracing — 256 elements × 10 instructions = 2560 lines. */
#define N_FULL  256
#define N_TRACE   4   /* show first 4 elements when --trace is on */

static void print_array(const char *label, const int32_t *a, int n) {
    printf("  %-4s [", label);
    int show = n > 8 ? 8 : n;
    for (int i = 0; i < show; i++) printf("%s%d", i ? ", " : "", a[i]);
    if (n > show) printf(", …(%d more)", n - show);
    printf("]\n");
}

int main(int argc, char **argv) {
    const char *obj   = NULL;
    int         trace = 0;

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--trace")) trace = 1;
        else                             obj   = argv[i];
    }
    if (!obj) {
        fprintf(stderr,
            "Usage: %s <vadd.o> [--trace]\n"
            "  Compile the kernel first:\n"
            "    llc -mtriple=npu -filetype=obj -o build/vadd.o kernels/vadd.ll\n",
            argv[0]);
        return 1;
    }

    int N = trace ? N_TRACE : N_FULL;

    /* ── 1. init ──────────────────────────────────────────────────────────── */
    NpuCtx ctx;
    NPU_RT_CHECK(npuInit(&ctx));
    if (trace) NPU_RT_CHECK(npuSetTrace(ctx, 1)); /* enables per-instruction log */

    /* ── 2. allocate device buffers ─────────────────────────────────────── */
    NpuDevPtr d_a, d_b, d_c;
    NPU_RT_CHECK(npuMalloc(ctx, &d_a, (size_t)N * sizeof(int32_t)));
    NPU_RT_CHECK(npuMalloc(ctx, &d_b, (size_t)N * sizeof(int32_t)));
    NPU_RT_CHECK(npuMalloc(ctx, &d_c, (size_t)N * sizeof(int32_t)));

    printf("[host] device buffers  d_a=0x%08x  d_b=0x%08x  d_c=0x%08x\n",
           d_a, d_b, d_c);

    /* ── 3. fill host arrays and upload ────────────────────────────────── */
    int32_t *h_a = calloc((size_t)N, sizeof(int32_t));
    int32_t *h_b = calloc((size_t)N, sizeof(int32_t));
    for (int i = 0; i < N; i++) {
        h_a[i] = i;
        h_b[i] = N - i;
    }

    printf("[host] input arrays (n=%d):\n", N);
    print_array("a", h_a, N);
    print_array("b", h_b, N);
    printf("       expected c[i] = a[i]+b[i] = %d for all i\n\n", N);

    NPU_RT_CHECK(npuMemcpy_h2d(ctx, d_a, h_a, (size_t)N * sizeof(int32_t)));
    NPU_RT_CHECK(npuMemcpy_h2d(ctx, d_b, h_b, (size_t)N * sizeof(int32_t)));

    /* ── 4. load ELF module ─────────────────────────────────────────────── */
    printf("[host] loading ELF: %s\n", obj);
    NpuModule   mod;
    NpuFunction fn_vadd;
    NPU_RT_CHECK(npuModuleLoad  (ctx, &mod,     obj));
    NPU_RT_CHECK(npuGetFunction (mod, &fn_vadd, "vadd"));
    printf("[host] got function \"vadd\"\n\n");

    /* ── 5. launch  —  args map to x10..x13 ────────────────────────────── */
    printf("[host] launching vadd(c=0x%x, a=0x%x, b=0x%x, n=%d)\n\n",
           d_c, d_a, d_b, N);

    npu_arg_t args[] = {
        NPU_PTR(d_c),   /* x10 = c (output) */
        NPU_PTR(d_a),   /* x11 = a          */
        NPU_PTR(d_b),   /* x12 = b          */
        NPU_I64(N),     /* x13 = n          */
    };
    NPU_RT_CHECK(npuLaunchKernel(ctx, fn_vadd, args, 4));

    /* ── 6. synchronize (executes h2d DMAs + kernel) ───────────────────── */
    NPU_RT_CHECK(npuSynchronize(ctx));

    /* ── 7. read back — npuRead_i32 is synchronous (direct DRAM read) ──── */
    printf("\n[host] reading results:\n");
    int32_t *h_c = calloc((size_t)N, sizeof(int32_t));
    NPU_RT_CHECK(npuRead_i32(ctx, d_c, h_c, N));
    print_array("c", h_c, N);

    /* ── 8. verify ──────────────────────────────────────────────────────── */
    int pass = 1;
    for (int i = 0; i < N; i++) {
        if (h_c[i] != N) {
            fprintf(stderr, "  FAIL c[%d] = %d  expected %d\n", i, h_c[i], N);
            pass = 0;
        }
    }
    printf("\n%s  vadd [%d elements]\n", pass ? "PASS" : "FAIL", N);

    npuPrintStats(ctx);

    /* ── 9. cleanup ─────────────────────────────────────────────────────── */
    npuModuleUnload(mod);
    npuFree(ctx, d_a); npuFree(ctx, d_b); npuFree(ctx, d_c);
    free(h_a); free(h_b); free(h_c);
    npuFinalize(ctx);

    return pass ? 0 : 1;
}
