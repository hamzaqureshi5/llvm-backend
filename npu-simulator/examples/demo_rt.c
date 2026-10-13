/*
 * demo_rt.c — Full-stack NPU runtime demo following NPU_DRIVER.md architecture.
 *
 * Stack exercised (bottom to top):
 *
 *   npu.c / npu_asm.c        — simulator
 *   npu_kdrv.c (firmware)    — kernel driver + firmware dispatch loop
 *   npu_rt.c                 — CUDA-style runtime
 *   demo_rt.c                — application (this file)
 *
 * Build:
 *   gcc -O2 -o demo_rt demo_rt.c npu_rt.c npu_kdrv.c npu.c npu_asm.c -lm
 * Run:
 *   ./demo_rt
 */

#include "npu_rt.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* ── helpers ─────────────────────────────────────────────────────────────── */

static void banner(const char *title)
{
    printf("\n");
    for (int i = 0; i < 60; i++) putchar('=');
    printf("\n %s\n", title);
    for (int i = 0; i < 60; i++) putchar('=');
    putchar('\n');
}

static int float_eq(float a, float b, float tol)
{
    return fabsf(a - b) <= tol;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Demo 1 — Tensor engine GEMM
 *
 * C (4×4) = A (4×4) × B (4×4)   using npuGemm() → CMD_TENSOR_GEMM
 * ═══════════════════════════════════════════════════════════════════════════ */
static void demo_gemm(NpuCtx ctx)
{
    banner("Demo 1: Tensor GEMM  C(4x4) = A(4x4) x B(4x4)");

    const int M = 4, K = 4, N = 4;
    const int sz = M * N;

    /* Identity matrix A = I, B = arbitrary → C should equal B. */
    float h_A[16] = {
        1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 0,
        0, 0, 0, 1,
    };
    float h_B[16] = {
         1,  2,  3,  4,
         5,  6,  7,  8,
         9, 10, 11, 12,
        13, 14, 15, 16,
    };
    float h_C[16] = {0};

    /* Allocate device buffers. */
    NpuDevPtr d_A, d_B, d_C;
    NPU_RT_CHECK(npuMalloc(ctx, &d_A, sz * sizeof(float)));
    NPU_RT_CHECK(npuMalloc(ctx, &d_B, sz * sizeof(float)));
    NPU_RT_CHECK(npuMalloc(ctx, &d_C, sz * sizeof(float)));

    printf("  d_A=0x%08x  d_B=0x%08x  d_C=0x%08x\n", d_A, d_B, d_C);

    /* Upload A and B. */
    NPU_RT_CHECK(npuMemcpy_h2d(ctx, d_A, h_A, sz*sizeof(float)));
    NPU_RT_CHECK(npuMemcpy_h2d(ctx, d_B, h_B, sz*sizeof(float)));

    /* Dispatch GEMM. */
    NPU_RT_CHECK(npuGemm(ctx, d_A, d_B, d_C, M, K, N));

    /* Synchronize: firmware executes H2D copies then GEMM. */
    NPU_RT_CHECK(npuSynchronize(ctx));

    /* Read back result. */
    NPU_RT_CHECK(npuRead_f32(ctx, d_C, h_C, sz));

    /* Verify: I × B = B */
    int ok = 1;
    for (int i = 0; i < sz; i++) {
        if (!float_eq(h_C[i], h_B[i], 1e-3f)) { ok = 0; break; }
    }
    printf("  GEMM (I*B == B): %s\n", ok ? "PASS" : "FAIL");
    printf("  C row0: [%.0f %.0f %.0f %.0f]\n", h_C[0],h_C[1],h_C[2],h_C[3]);
    printf("  C row3: [%.0f %.0f %.0f %.0f]\n", h_C[12],h_C[13],h_C[14],h_C[15]);

    npuFree(ctx, d_A);
    npuFree(ctx, d_B);
    npuFree(ctx, d_C);
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Demo 2 — Tensor engine ReLU
 *
 * out[i] = max(0, in[i])   via npuRelu() → CMD_TENSOR_RELU
 * ═══════════════════════════════════════════════════════════════════════════ */
static void demo_relu(NpuCtx ctx)
{
    banner("Demo 2: Tensor ReLU  out = max(0, in)");

    const int M = 2, N = 4;
    const int sz = M * N;

    float h_in[8]  = { -3.0f, -1.0f, 0.0f, 1.0f, 2.0f, -5.0f, 7.0f, -0.5f };
    float h_out[8] = {0};
    float h_exp[8] = {  0.0f,  0.0f, 0.0f, 1.0f, 2.0f,  0.0f, 7.0f,  0.0f };

    NpuDevPtr d_in, d_out;
    NPU_RT_CHECK(npuMalloc(ctx, &d_in,  sz * sizeof(float)));
    NPU_RT_CHECK(npuMalloc(ctx, &d_out, sz * sizeof(float)));

    NPU_RT_CHECK(npuMemcpy_h2d(ctx, d_in, h_in, sz*sizeof(float)));
    NPU_RT_CHECK(npuRelu(ctx, d_in, d_out, M, N));
    NPU_RT_CHECK(npuSynchronize(ctx));
    NPU_RT_CHECK(npuRead_f32(ctx, d_out, h_out, sz));

    int ok = 1;
    for (int i = 0; i < sz; i++)
        if (!float_eq(h_out[i], h_exp[i], 1e-6f)) { ok = 0; break; }
    printf("  relu([-3,-1,0,1,2,-5,7,-0.5]): %s\n", ok ? "PASS" : "FAIL");
    printf("  out: [");
    for (int i = 0; i < sz; i++) printf("%.1f%s", h_out[i], i<sz-1?", ":"");
    printf("]\n");

    npuFree(ctx, d_in);
    npuFree(ctx, d_out);
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Demo 3 — Custom kernel via command queue
 *
 * vadd: c[i] = a[i] + b[i]   (int32, N=128)
 * Full flow: H2D → LAUNCH_KERN → D2H in one synchronized submission
 * ═══════════════════════════════════════════════════════════════════════════ */
static const char *VADD_ASM =
    "vadd:\n"
    "    LI   x14, 0\n"
    ".Lvadd_loop:\n"
    "    LW   x15, 0(x11)\n"
    "    LW   x16, 0(x12)\n"
    "    ADD  x15, x15, x16\n"
    "    SW   x15, 0(x10)\n"
    "    ADDI x10, x10, 4\n"
    "    ADDI x11, x11, 4\n"
    "    ADDI x12, x12, 4\n"
    "    ADDI x14, x14, 1\n"
    "    BNE  x14, x13, .Lvadd_loop\n"
    "    JALR x0, x1, 0\n"
    "    HALT\n";

static void demo_vadd(NpuCtx ctx)
{
    banner("Demo 3: Custom kernel (vadd i32, N=128) via CMD_LAUNCH_KERN");

    const int N = 128;
    int32_t h_a[128], h_b[128], h_c[128];
    for (int i = 0; i < N; i++) { h_a[i] = i * 2; h_b[i] = i + 1; }

    NpuDevPtr d_a, d_b, d_c;
    NPU_RT_CHECK(npuMalloc(ctx, &d_a, N * sizeof(int32_t)));
    NPU_RT_CHECK(npuMalloc(ctx, &d_b, N * sizeof(int32_t)));
    NPU_RT_CHECK(npuMalloc(ctx, &d_c, N * sizeof(int32_t)));

    /* Load kernel module. */
    NpuModule   mod;
    NpuFunction fn_vadd;
    NPU_RT_CHECK(npuModuleLoadAsm(ctx, &mod, VADD_ASM));
    NPU_RT_CHECK(npuGetFunction(mod, &fn_vadd, "vadd"));

    /* Queue: H2D → H2D → kernel → synchronize → D2H. */
    NPU_RT_CHECK(npuMemcpy_h2d(ctx, d_a, h_a, N*sizeof(int32_t)));
    NPU_RT_CHECK(npuMemcpy_h2d(ctx, d_b, h_b, N*sizeof(int32_t)));

    npu_arg_t args[] = { NPU_PTR(d_c), NPU_PTR(d_a), NPU_PTR(d_b), NPU_I64(N) };
    NPU_RT_CHECK(npuLaunchKernel(ctx, fn_vadd, args, 4));

    NPU_RT_CHECK(npuSynchronize(ctx));

    NPU_RT_CHECK(npuRead_i32(ctx, d_c, h_c, N));

    int ok = 1;
    for (int i = 0; i < N; i++)
        if (h_c[i] != h_a[i] + h_b[i]) { ok = 0; break; }
    printf("  vadd[0..%d]: %s\n", N - 1, ok ? "PASS" : "FAIL");
    printf("  c[0]=%d c[1]=%d c[63]=%d c[127]=%d\n",
           h_c[0], h_c[1], h_c[63], h_c[127]);

    npuFree(ctx, d_a);
    npuFree(ctx, d_b);
    npuFree(ctx, d_c);
    npuModuleUnload(mod);
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Demo 4 — D2D memcpy + memset + fence
 * ═══════════════════════════════════════════════════════════════════════════ */
static void demo_d2d_fence(NpuCtx ctx)
{
    banner("Demo 4: D2D memcpy + memset + fence");

    int32_t src[8] = {10, 20, 30, 40, 50, 60, 70, 80};
    NpuDevPtr d_src, d_dst, d_zero;
    NPU_RT_CHECK(npuMalloc(ctx, &d_src,  sizeof(src)));
    NPU_RT_CHECK(npuMalloc(ctx, &d_dst,  sizeof(src)));
    NPU_RT_CHECK(npuMalloc(ctx, &d_zero, sizeof(src)));

    /* Upload src, memset zero region, D2D copy src→dst, then fence. */
    NPU_RT_CHECK(npuMemcpy_h2d(ctx, d_src, src, sizeof(src)));
    NPU_RT_CHECK(npuMemset(ctx, d_zero, 0, 8));
    NPU_RT_CHECK(npuMemcpy_d2d(ctx, d_dst, d_src, sizeof(src)));

    NpuFence fence;
    NPU_RT_CHECK(npuFenceCreate(ctx, &fence));

    NPU_RT_CHECK(npuSynchronize(ctx));
    NPU_RT_CHECK(npuFenceWait(ctx, fence));

    int32_t got[8] = {0}, zeros[8] = {0};
    NPU_RT_CHECK(npuRead_i32(ctx, d_dst,  got,   8));
    NPU_RT_CHECK(npuRead_i32(ctx, d_zero, zeros, 8));

    int ok1 = memcmp(got,   src,        sizeof(src)) == 0;
    int ok2 = memcmp(zeros, (int32_t[]){0,0,0,0,0,0,0,0}, sizeof(src)) == 0;
    printf("  D2D copy:  %s\n", ok1 ? "PASS" : "FAIL");
    printf("  Memset(0): %s\n", ok2 ? "PASS" : "FAIL");
    printf("  Fence signaled after synchronize: PASS\n");

    npuFree(ctx, d_src);
    npuFree(ctx, d_dst);
    npuFree(ctx, d_zero);
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Demo 5 — Full ML pipeline: GEMM → ReLU
 *
 * Simulates one layer of a neural network:
 *   hidden = ReLU(W × x)
 *
 * W (4×4), x is 4 vectors packed as 4×4, hidden is 4×4
 * ═══════════════════════════════════════════════════════════════════════════ */
static void demo_ml_pipeline(NpuCtx ctx)
{
    banner("Demo 5: ML Pipeline — hidden = ReLU(W * x)");

    const int M = 4, K = 4, N = 4, sz = 16;

    /* Weight matrix W. */
    float h_W[16] = {
        1, -1,  0,  2,
       -1,  1, -1,  0,
        0, -1,  1, -1,
        2,  0, -1,  1,
    };
    /* Input batch X (4 vectors of dim 4). */
    float h_X[16] = {
        1,  2,  3,  4,
        5,  6,  7,  8,
        9, 10, 11, 12,
       13, 14, 15, 16,
    };

    NpuDevPtr d_W, d_X, d_Z, d_H;
    NPU_RT_CHECK(npuMalloc(ctx, &d_W, sz*sizeof(float)));
    NPU_RT_CHECK(npuMalloc(ctx, &d_X, sz*sizeof(float)));
    NPU_RT_CHECK(npuMalloc(ctx, &d_Z, sz*sizeof(float)));  /* GEMM output  */
    NPU_RT_CHECK(npuMalloc(ctx, &d_H, sz*sizeof(float)));  /* ReLU output  */

    /* Queue all work: H2D → GEMM → ReLU */
    NPU_RT_CHECK(npuMemcpy_h2d(ctx, d_W, h_W, sz*sizeof(float)));
    NPU_RT_CHECK(npuMemcpy_h2d(ctx, d_X, h_X, sz*sizeof(float)));
    NPU_RT_CHECK(npuGemm(ctx, d_W, d_X, d_Z, M, K, N));
    NPU_RT_CHECK(npuRelu(ctx, d_Z, d_H, M, N));
    NPU_RT_CHECK(npuSynchronize(ctx));

    float h_Z[16], h_H[16];
    NPU_RT_CHECK(npuRead_f32(ctx, d_Z, h_Z, sz));
    NPU_RT_CHECK(npuRead_f32(ctx, d_H, h_H, sz));

    /* Verify ReLU: no negative values. */
    int ok = 1;
    for (int i = 0; i < sz; i++) {
        if (h_H[i] < 0.0f)             { ok = 0; break; }
        if (h_Z[i] > 0 && h_H[i] != h_Z[i]) { ok = 0; break; }
        if (h_Z[i] < 0 && h_H[i] != 0)      { ok = 0; break; }
    }
    printf("  GEMM output row0: [%.1f %.1f %.1f %.1f]\n",
           h_Z[0], h_Z[1], h_Z[2], h_Z[3]);
    printf("  ReLU output row0: [%.1f %.1f %.1f %.1f]\n",
           h_H[0], h_H[1], h_H[2], h_H[3]);
    printf("  Pipeline (GEMM→ReLU): %s\n", ok ? "PASS" : "FAIL");

    npuFree(ctx, d_W);
    npuFree(ctx, d_X);
    npuFree(ctx, d_Z);
    npuFree(ctx, d_H);
}

/* ─── main ──────────────────────────────────────────────────────────────── */

int main(void)
{
    printf("NPU Runtime (full stack: Driver → Firmware → Simulator)\n");
    printf("=========================================================\n");

    /* Device properties. */
    NpuDeviceProps props;
    npuGetDeviceProps(&props);
    printf("Device: %s | DRAM: %.0f MiB | SRAM: %zu KiB | max_cores: %d\n",
           props.name,
           props.dram_bytes / (1024.0*1024.0),
           props.scratch_bytes / 1024,
           props.max_cores);

    /* Create context. */
    NpuCtx ctx;
    NPU_RT_CHECK(npuInit(&ctx));

    demo_gemm(ctx);
    demo_relu(ctx);
    demo_vadd(ctx);
    demo_d2d_fence(ctx);
    demo_ml_pipeline(ctx);

    banner("Cumulative Statistics");
    NPU_RT_CHECK(npuPrintStats(ctx));

    banner("Heap State");
    npuHeapDump(ctx, stdout);

    NPU_RT_CHECK(npuFinalize(ctx));
    printf("\nAll demos complete.\n");
    return 0;
}
