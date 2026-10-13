/* demo_matmul_elf.c — Load an LLVM-compiled NPU matmul kernel and verify.
 *
 * End-to-end flow:
 *   1. Compile the kernel:
 *        llc -mtriple=npu-unknown-unknown -filetype=obj \
 *            examples/matmul.ll -o build/matmul.o
 *   2. Run this demo:
 *        ./build/demo_matmul_elf build/matmul.o [--trace]
 *
 * The kernel implements C = A × B for 4×4 float matrices via the tensor
 * engine (CFG.SHAPE → ACC.CLR → LOAD → LOAD → MMA → STORE → FENCE).
 *
 * Args arrive in x10–x12 per the NPU calling convention:
 *   x10 = ptr to C (output)
 *   x11 = ptr to A (input)
 *   x12 = ptr to B (input)
 *
 * Results are verified against a scalar CPU reference.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "npu_rt.h"

#define M 4
#define K 4
#define N 4
#define NELEMS (M * N)   /* 16 floats per matrix */

/* ── CPU reference ────────────────────────────────────────────────────────── */

static void cpu_matmul(const float *A, const float *B, float *C,
                       int rows, int inner, int cols) {
    for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols; j++) {
            float acc = 0.0f;
            for (int k = 0; k < inner; k++)
                acc += A[i * inner + k] * B[k * cols + j];
            C[i * cols + j] = acc;
        }
}

static void print_matrix(const char *label, const float *m, int rows, int cols) {
    printf("  %s:\n", label);
    for (int i = 0; i < rows; i++) {
        printf("    [");
        for (int j = 0; j < cols; j++)
            printf("%7.2f%s", m[i * cols + j], j < cols - 1 ? ", " : "");
        printf("]\n");
    }
}

int main(int argc, char **argv) {
    const char *obj_path = NULL;
    int         trace    = 0;

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--trace")) trace = 1;
        else                             obj_path = argv[i];
    }
    if (!obj_path) {
        fprintf(stderr,
            "Usage: %s <matmul.o> [--trace]\n"
            "  Compile the kernel first:\n"
            "    llc -mtriple=npu-unknown-unknown -filetype=obj \\\n"
            "        examples/matmul.ll -o build/matmul.o\n",
            argv[0]);
        return 1;
    }

    /* ── host matrices ────────────────────────────────────────────────────── */
    float h_A[NELEMS], h_B[NELEMS], h_C[NELEMS], h_ref[NELEMS];

    /* Identity-ish A: A[i][j] = (i == j) ? (i+1.0f) : 0.5f */
    for (int i = 0; i < M; i++)
        for (int j = 0; j < K; j++)
            h_A[i * K + j] = (i == j) ? (float)(i + 1) : 0.5f;

    /* Ascending B: B[k][j] = k * N + j + 1 */
    for (int k = 0; k < K; k++)
        for (int j = 0; j < N; j++)
            h_B[k * N + j] = (float)(k * N + j + 1);

    memset(h_C,   0, sizeof h_C);
    memset(h_ref, 0, sizeof h_ref);

    cpu_matmul(h_A, h_B, h_ref, M, K, N);

    printf("=== NPU matmul_4x4 ELF demo ===\n\n");
    print_matrix("A (4×4)", h_A, M, K);
    print_matrix("B (4×4)", h_B, K, N);
    print_matrix("CPU reference C = A×B", h_ref, M, N);
    printf("\n");

    /* ── NPU setup ────────────────────────────────────────────────────────── */
    NpuCtx ctx;
    NPU_RT_CHECK(npuInit(&ctx));
    if (trace) NPU_RT_CHECK(npuSetTrace(ctx, 1));

    /* Allocate device buffers */
    NpuDevPtr d_A, d_B, d_C;
    NPU_RT_CHECK(npuMalloc(ctx, &d_A, NELEMS * sizeof(float)));
    NPU_RT_CHECK(npuMalloc(ctx, &d_B, NELEMS * sizeof(float)));
    NPU_RT_CHECK(npuMalloc(ctx, &d_C, NELEMS * sizeof(float)));

    printf("[host] device buffers  d_A=0x%08x  d_B=0x%08x  d_C=0x%08x\n",
           d_A, d_B, d_C);

    /* Upload A and B */
    NPU_RT_CHECK(npuMemcpy_h2d(ctx, d_A, h_A, NELEMS * sizeof(float)));
    NPU_RT_CHECK(npuMemcpy_h2d(ctx, d_B, h_B, NELEMS * sizeof(float)));

    /* ── Load ELF module ──────────────────────────────────────────────────── */
    printf("[host] loading ELF: %s\n", obj_path);
    NpuModule   mod;
    NpuFunction fn_matmul;
    NPU_RT_CHECK(npuModuleLoad  (ctx, &mod,        obj_path));
    NPU_RT_CHECK(npuGetFunction (mod, &fn_matmul,  "matmul_4x4"));
    printf("[host] got function \"matmul_4x4\"\n\n");

    /* ── Launch ───────────────────────────────────────────────────────────── */
    printf("[host] launching matmul_4x4(C=0x%x, A=0x%x, B=0x%x)\n\n",
           d_C, d_A, d_B);

    npu_arg_t args[] = {
        NPU_PTR(d_C),   /* x10 = C (output) */
        NPU_PTR(d_A),   /* x11 = A          */
        NPU_PTR(d_B),   /* x12 = B          */
    };
    NPU_RT_CHECK(npuLaunchKernel(ctx, fn_matmul, args, 3));

    /* ── Synchronize ──────────────────────────────────────────────────────── */
    NPU_RT_CHECK(npuSynchronize(ctx));

    /* ── Read back & verify ───────────────────────────────────────────────── */
    NPU_RT_CHECK(npuRead_f32(ctx, d_C, h_C, NELEMS));

    print_matrix("NPU result C", h_C, M, N);
    printf("\n");

    int pass = 1;
    for (int i = 0; i < NELEMS; i++) {
        if (fabsf(h_C[i] - h_ref[i]) > 1e-3f) {
            fprintf(stderr, "  MISMATCH C[%d]: NPU=%.4f  ref=%.4f\n",
                    i, h_C[i], h_ref[i]);
            pass = 0;
        }
    }

    if (pass)
        printf("PASS  matmul_4x4: NPU result matches CPU reference.\n");
    else
        printf("FAIL  matmul_4x4: results differ (see above).\n");

    npuPrintStats(ctx);

    /* ── Cleanup ──────────────────────────────────────────────────────────── */
    npuModuleUnload(mod);
    npuFree(ctx, d_A);
    npuFree(ctx, d_B);
    npuFree(ctx, d_C);
    npuFinalize(ctx);

    return pass ? 0 : 1;
}
