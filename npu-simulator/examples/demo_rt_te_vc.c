/* demo_rt_te_vc.c — Tensor Engine (Layer 2) and Vector Core (Layer 3) demos.
 *
 * Stack exercised:
 *
 *   Demos 1–2  Tensor Engine (T_* opcodes)
 *   ─────────────────────────────────────────────────────────────────────────
 *   npuModuleLoadAsm / npuLaunchKernel        npu_rt.h  (this layer)
 *   firmware LAUNCH_KERN → npu_run(T_* prog)  npu_kdrv.c
 *   npu.c tensor-engine interpreter
 *
 *   Demos 3–4  Vector Core (V_* opcodes)
 *   ─────────────────────────────────────────────────────────────────────────
 *   npu_kdrv_probe + npu_run directly         npu_kdrv.h / npu.h
 *
 *   NOTE: npu_rt.h has no npuRegisterVectorKernel() yet. Vector kernels must
 *   be registered via npu_kdrv_get_sim() + npu_register_kernel() before the
 *   control program that launches them can be run. That is the gap bridged
 *   here by using the kdrv API directly.
 *
 * Build:
 *   cmake --build build --target demo_rt_te_vc
 * Run:
 *   ./build/demo_rt_te_vc
 */

#include "npu_rt.h"    /* NpuCtx, npuInit, npuMalloc, npuModuleLoadAsm, … */
#include "npu_kdrv.h"  /* npu_kdrv_probe, npu_kdrv_get_sim, npu_kdrv_alloc */
#include "npu_asm.h"   /* npu_asm_control, npu_asm_vector               */
#include "npu.h"       /* npu_t, npu_register_kernel, npu_write_f32, …  */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define N_TILE  16   /* 4×4 tile = 16 elements */
#define N_VEC   16   /* one hardware lane-width of floats */

/* ── helpers ─────────────────────────────────────────────────────────────── */

static void banner(const char *title)
{
    printf("\n");
    for (int i = 0; i < 64; i++) putchar('=');
    printf("\n %s\n", title);
    for (int i = 0; i < 64; i++) putchar('=');
    putchar('\n');
}

static int float_near(float a, float b, float tol)
{
    return fabsf(a - b) <= tol;
}

static void identity_4x4(float *m)
{
    memset(m, 0, 16 * sizeof(float));
    m[0] = m[5] = m[10] = m[15] = 1.0f;
}

/* Abort on assembly error. */
static void asm_check(int rc, const char *err, const char *ctx)
{
    if (rc < 0) {
        fprintf(stderr, "Assembly error (%s): %s\n", ctx, err);
        abort();
    }
}

/* ═══════════════════════════════════════════════════════════════════════════
 * TE Demo 1 — MMA accumulation chain
 *
 * C(4×4) = A(4×4)×B1(4×4) + A(4×4)×B2(4×4)
 *
 * Two MMA instructions accumulate into tile t0 (ACC.CLR first):
 *   ACC.CLR t0          ; t0 = 0
 *   MMA t0, t1, t2      ; t0 += t1 * t2  (A × B1)
 *   MMA t0, t1, t3      ; t0 += t1 * t3  (A × B2)
 *
 * With A = I, B1 = ones, B2 = twos:
 *   C = I×ones + I×twos = ones + twos = threes → all 3.0
 * ═══════════════════════════════════════════════════════════════════════════ */
static const char *TE_MMA_CHAIN_ASM =
    "te_mma_chain:\n"
    "    CFG.SHAPE 4, 4, 4\n"
    "    ACC.CLR t0\n"
    "    LOAD t1, 0(x11), 16\n"   /* A  tile from arg1 */
    "    LOAD t2, 0(x12), 16\n"   /* B1 tile from arg2 */
    "    LOAD t3, 0(x13), 16\n"   /* B2 tile from arg3 */
    "    MMA t0, t1, t2\n"        /* t0 += A * B1 */
    "    MMA t0, t1, t3\n"        /* t0 += A * B2 */
    "    STORE t0, 0(x10), 16\n"  /* C  tile to   arg0 */
    "    FENCE\n"
    "    JALR x0, x1, 0\n"
    "    HALT\n";

static void demo_te_mma_chain(NpuCtx ctx)
{
    banner("TE Demo 1: MMA Accumulation Chain  C = A×B1 + A×B2");

    float h_A[N_TILE], h_B1[N_TILE], h_B2[N_TILE], h_C[N_TILE];
    identity_4x4(h_A);
    for (int i = 0; i < N_TILE; i++) h_B1[i] = 1.0f;
    for (int i = 0; i < N_TILE; i++) h_B2[i] = 2.0f;

    NpuDevPtr d_A, d_B1, d_B2, d_C;
    NPU_RT_CHECK(npuMalloc(ctx, &d_A,  N_TILE * sizeof(float)));
    NPU_RT_CHECK(npuMalloc(ctx, &d_B1, N_TILE * sizeof(float)));
    NPU_RT_CHECK(npuMalloc(ctx, &d_B2, N_TILE * sizeof(float)));
    NPU_RT_CHECK(npuMalloc(ctx, &d_C,  N_TILE * sizeof(float)));

    NPU_RT_CHECK(npuMemcpy_h2d(ctx, d_A,  h_A,  N_TILE * sizeof(float)));
    NPU_RT_CHECK(npuMemcpy_h2d(ctx, d_B1, h_B1, N_TILE * sizeof(float)));
    NPU_RT_CHECK(npuMemcpy_h2d(ctx, d_B2, h_B2, N_TILE * sizeof(float)));

    NpuModule   mod;
    NpuFunction fn;
    NPU_RT_CHECK(npuModuleLoadAsm(ctx, &mod, TE_MMA_CHAIN_ASM));
    NPU_RT_CHECK(npuGetFunction(mod, &fn, "te_mma_chain"));

    /* arg0=C, arg1=A, arg2=B1, arg3=B2 → x10..x13 */
    npu_arg_t args[] = { NPU_PTR(d_C), NPU_PTR(d_A), NPU_PTR(d_B1), NPU_PTR(d_B2) };
    NPU_RT_CHECK(npuLaunchKernel(ctx, fn, args, 4));
    NPU_RT_CHECK(npuSynchronize(ctx));
    NPU_RT_CHECK(npuRead_f32(ctx, d_C, h_C, N_TILE));

    int ok = 1;
    for (int i = 0; i < N_TILE; i++)
        if (!float_near(h_C[i], 3.0f, 1e-4f)) { ok = 0; break; }

    printf("  A = I4, B1 = ones, B2 = twos\n");
    printf("  C[0..3]: [%.1f %.1f %.1f %.1f]\n", h_C[0],h_C[1],h_C[2],h_C[3]);
    printf("  MMA chain (C == 3.0 everywhere): %s\n", ok ? "PASS" : "FAIL");

    npuFree(ctx, d_A); npuFree(ctx, d_B1); npuFree(ctx, d_B2); npuFree(ctx, d_C);
    npuModuleUnload(mod);
}

/* ═══════════════════════════════════════════════════════════════════════════
 * TE Demo 2 — MATMUL + SCALE
 *
 * C(4×4) = (A(4×4) × B(4×4)) × 0.5
 *
 * With A = I, B = fours:
 *   MATMUL → t2 = I × fours = fours
 *   SCALE  → t2 = fours × 0.5 = twos → all 2.0
 * ═══════════════════════════════════════════════════════════════════════════ */
static const char *TE_SCALE_ASM =
    "te_scale:\n"
    "    CFG.SHAPE 4, 4, 4\n"
    "    LOAD t0, 0(x11), 16\n"   /* A tile from arg1 */
    "    LOAD t1, 0(x12), 16\n"   /* B tile from arg2 */
    "    MATMUL t2, t0, t1\n"     /* t2 = A × B       */
    "    SCALE t2, t2, 0.5\n"     /* t2 *= 0.5        */
    "    STORE t2, 0(x10), 16\n"  /* C tile to   arg0 */
    "    FENCE\n"
    "    JALR x0, x1, 0\n"
    "    HALT\n";

static void demo_te_scale(NpuCtx ctx)
{
    banner("TE Demo 2: MATMUL + SCALE  C = (A×B) × 0.5");

    float h_A[N_TILE], h_B[N_TILE], h_C[N_TILE];
    identity_4x4(h_A);
    for (int i = 0; i < N_TILE; i++) h_B[i] = 4.0f;

    NpuDevPtr d_A, d_B, d_C;
    NPU_RT_CHECK(npuMalloc(ctx, &d_A, N_TILE * sizeof(float)));
    NPU_RT_CHECK(npuMalloc(ctx, &d_B, N_TILE * sizeof(float)));
    NPU_RT_CHECK(npuMalloc(ctx, &d_C, N_TILE * sizeof(float)));

    NPU_RT_CHECK(npuMemcpy_h2d(ctx, d_A, h_A, N_TILE * sizeof(float)));
    NPU_RT_CHECK(npuMemcpy_h2d(ctx, d_B, h_B, N_TILE * sizeof(float)));

    NpuModule   mod;
    NpuFunction fn;
    NPU_RT_CHECK(npuModuleLoadAsm(ctx, &mod, TE_SCALE_ASM));
    NPU_RT_CHECK(npuGetFunction(mod, &fn, "te_scale"));

    npu_arg_t args[] = { NPU_PTR(d_C), NPU_PTR(d_A), NPU_PTR(d_B) };
    NPU_RT_CHECK(npuLaunchKernel(ctx, fn, args, 3));
    NPU_RT_CHECK(npuSynchronize(ctx));
    NPU_RT_CHECK(npuRead_f32(ctx, d_C, h_C, N_TILE));

    int ok = 1;
    for (int i = 0; i < N_TILE; i++)
        if (!float_near(h_C[i], 2.0f, 1e-4f)) { ok = 0; break; }

    printf("  A = I4, B = fours\n");
    printf("  C[0..3]: [%.1f %.1f %.1f %.1f]\n", h_C[0],h_C[1],h_C[2],h_C[3]);
    printf("  MATMUL+SCALE (C == 2.0 everywhere): %s\n", ok ? "PASS" : "FAIL");

    npuFree(ctx, d_A); npuFree(ctx, d_B); npuFree(ctx, d_C);
    npuModuleUnload(mod);
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Vector Core helper — lower-level kdrv path
 *
 * npu_rt.h gap: there is no npuRegisterVectorKernel().
 * Vector kernels must be registered via:
 *   npu_t *sim = npu_kdrv_get_sim(drv);
 *   int kid = npu_register_kernel(sim, vinst_code, len);
 * and then launched from a control program (cinst) via KLAUNCH kid, …
 *
 * The control program is assembled at runtime with snprintf so the runtime
 * kid and args DRAM address can be embedded as integer literals.
 * ═══════════════════════════════════════════════════════════════════════════ */

/* Run a single-instance (grid=1) vector kernel.
 * vkernel_src : vector assembly (vinst[] text)
 * args_dev[]  : DRAM device addresses written into DRAM as an int32 array;
 *               the kernel receives s1 = base address of this array on entry.
 * nargs       : number of entries in args_dev
 * drv         : kdrv handle (owns DRAM allocator + sim)
 */
static void run_vector_kernel(const char *vkernel_src,
                              const uint32_t *args_dev, int nargs,
                              npu_kdrv_t drv)
{
    npu_t *sim = npu_kdrv_get_sim(drv);
    char err[256];

    /* Assemble and register vector kernel. */
    vinst vk[64];
    int vlen = npu_asm_vector(vkernel_src, vk, 64, err, sizeof(err));
    asm_check(vlen, err, "vector kernel");
    int kid = npu_register_kernel(sim, vk, vlen);

    /* Write args array into DRAM (each entry is a 4-byte device address). */
    uint32_t args_dram;
    kdrv_err_t ke = npu_kdrv_alloc(drv, &args_dram, (size_t)nargs * sizeof(int32_t));
    if (ke != KDRV_OK) { fprintf(stderr, "kdrv_alloc failed\n"); abort(); }
    npu_write_i32(sim, args_dram, (const int32_t *)args_dev, nargs);

    /* Control program: set up x14=grid_size, x15=args_dram, then KLAUNCH. */
    char cprog_src[256];
    snprintf(cprog_src, sizeof(cprog_src),
             "LI x14, 1\n"
             "LI x15, %u\n"
             "KLAUNCH %d, x14, x15\n"
             "KWAIT\n"
             "HALT\n",
             (unsigned)args_dram, kid);

    cinst cp[8];
    int clen = npu_asm_control(cprog_src, cp, 8, err, sizeof(err));
    asm_check(clen, err, "control program");

    npu_run(sim, cp, clen);

    npu_kdrv_free(drv, args_dram);
}

/* ═══════════════════════════════════════════════════════════════════════════
 * VC Demo 3 — Vector float add
 *
 * c[i] = a[i] + b[i]   for i in 0..N_VEC-1
 *
 * Kernel: VSETVL → VLOAD a → VLOAD b → VADD → VSTORE c
 *
 * a[i] = (float)i, b[i] = 1.0  →  c[i] = (float)(i+1)
 *   c[0] = 1.0, c[15] = 16.0
 * ═══════════════════════════════════════════════════════════════════════════ */
static const char *VC_VADD_KERNEL =
    "SLI s2, 16\n"         /* s2 = requested VL = 16   */
    "VSETVL s2, s2\n"      /* vl = min(16, VLEN_MAX)   */
    "SLW s3, 0(s1)\n"      /* s3 = args[0] = d_a       */
    "SLW s4, 4(s1)\n"      /* s4 = args[1] = d_b       */
    "SLW s5, 8(s1)\n"      /* s5 = args[2] = d_c       */
    "VLOAD v0, 0(s3)\n"    /* v0 = a                   */
    "VLOAD v1, 0(s4)\n"    /* v1 = b                   */
    "VADD v2, v0, v1\n"    /* v2 = a + b               */
    "VSTORE v2, 0(s5)\n"   /* store result             */
    "KEXIT\n";

static void demo_vc_vadd(npu_kdrv_t drv)
{
    banner("VC Demo 3: Vector Float Add  c[i] = a[i] + b[i]  (N=16)");

    npu_t *sim = npu_kdrv_get_sim(drv);

    float h_a[N_VEC], h_b[N_VEC], h_c[N_VEC];
    for (int i = 0; i < N_VEC; i++) { h_a[i] = (float)i; h_b[i] = 1.0f; }

    uint32_t d_a, d_b, d_c;
    npu_kdrv_alloc(drv, &d_a, N_VEC * sizeof(float));
    npu_kdrv_alloc(drv, &d_b, N_VEC * sizeof(float));
    npu_kdrv_alloc(drv, &d_c, N_VEC * sizeof(float));

    npu_write_f32(sim, d_a, h_a, N_VEC);
    npu_write_f32(sim, d_b, h_b, N_VEC);

    uint32_t args[] = { d_a, d_b, d_c };
    run_vector_kernel(VC_VADD_KERNEL, args, 3, drv);

    npu_read_f32(sim, d_c, h_c, N_VEC);

    int ok = 1;
    for (int i = 0; i < N_VEC; i++)
        if (!float_near(h_c[i], (float)(i + 1), 1e-5f)) { ok = 0; break; }

    printf("  a[i] = i, b[i] = 1.0\n");
    printf("  c[0..3]: [%.1f %.1f %.1f %.1f]\n", h_c[0],h_c[1],h_c[2],h_c[3]);
    printf("  c[12..15]: [%.1f %.1f %.1f %.1f]\n", h_c[12],h_c[13],h_c[14],h_c[15]);
    printf("  VADD (c[i] == i+1): %s\n", ok ? "PASS" : "FAIL");

    npu_kdrv_free(drv, d_a);
    npu_kdrv_free(drv, d_b);
    npu_kdrv_free(drv, d_c);
}

/* ═══════════════════════════════════════════════════════════════════════════
 * VC Demo 4 — Softmax numerics (VEXP + VREDSUM)
 *
 * sum = Σ exp(x[i])  for i in 0..N_VEC-1
 *
 * Kernel:
 *   VSETVL → VLOAD x → VEXP → VREDSUM s6 → VBCAST v1 → VSTORE sum_vec
 *
 * VBCAST + VSTORE is used to move the scalar VREDSUM result to DRAM.
 *
 * x[i] = 0.0  →  exp(0) = 1.0  →  sum = 16.0
 * ═══════════════════════════════════════════════════════════════════════════ */
static const char *VC_SOFTMAX_KERNEL =
    "SLI s2, 16\n"         /* s2 = VL = 16                         */
    "VSETVL s2, s2\n"
    "SLW s3, 0(s1)\n"      /* s3 = args[0] = d_x (input)           */
    "SLW s4, 4(s1)\n"      /* s4 = args[1] = d_out (output)        */
    "VLOAD v0, 0(s3)\n"    /* v0 = x                               */
    "VEXP v0, v0\n"        /* v0 = exp(x)                          */
    "VREDSUM s6, v0\n"     /* s6 = sum(exp(x))  [scalar float]     */
    "VBCAST v1, s6\n"      /* v1 = [sum, sum, …]  broadcast scalar */
    "VSTORE v1, 0(s4)\n"   /* out[0..VL-1] = sum                   */
    "KEXIT\n";

static void demo_vc_softmax(npu_kdrv_t drv)
{
    banner("VC Demo 4: Softmax Numerics  sum = Σ exp(x[i])  (N=16, x=0)");

    npu_t *sim = npu_kdrv_get_sim(drv);

    float h_x[N_VEC], h_out[N_VEC];
    for (int i = 0; i < N_VEC; i++) h_x[i] = 0.0f;   /* exp(0) = 1 */

    uint32_t d_x, d_out;
    npu_kdrv_alloc(drv, &d_x,   N_VEC * sizeof(float));
    npu_kdrv_alloc(drv, &d_out, N_VEC * sizeof(float));

    npu_write_f32(sim, d_x, h_x, N_VEC);

    uint32_t args[] = { d_x, d_out };
    run_vector_kernel(VC_SOFTMAX_KERNEL, args, 2, drv);

    npu_read_f32(sim, d_out, h_out, N_VEC);

    float sum = h_out[0];                      /* all lanes carry the same sum */
    float expected = (float)N_VEC * 1.0f;      /* 16 × exp(0) = 16.0           */

    printf("  x[i] = 0.0  →  exp(0) = 1.0\n");
    printf("  VREDSUM result: %.4f  (expected %.1f)\n", sum, expected);
    printf("  Softmax sum (≈ 16.0): %s\n",
           float_near(sum, expected, 1e-4f) ? "PASS" : "FAIL");

    /* Bonus: run again with x[i] = 1.0 to demonstrate exp(1) = e. */
    for (int i = 0; i < N_VEC; i++) h_x[i] = 1.0f;
    npu_write_f32(sim, d_x, h_x, N_VEC);
    run_vector_kernel(VC_SOFTMAX_KERNEL, args, 2, drv);
    npu_read_f32(sim, d_out, h_out, N_VEC);

    float sum_e = h_out[0];
    float expected_e = (float)N_VEC * expf(1.0f);
    printf("  x[i] = 1.0  →  VREDSUM = %.4f  (expected %.4f)\n",
           sum_e, expected_e);
    printf("  Softmax sum (≈ 16e): %s\n",
           float_near(sum_e, expected_e, 1e-3f) ? "PASS" : "FAIL");

    npu_kdrv_free(drv, d_x);
    npu_kdrv_free(drv, d_out);
}

/* ─── main ──────────────────────────────────────────────────────────────── */

int main(void)
{
    printf("NPU Tensor Engine + Vector Core Demos\n");
    printf("======================================\n");

    /* ── TE demos: use the full RT stack ──────────────────────────────────── */
    NpuCtx ctx;
    NPU_RT_CHECK(npuInit(&ctx));

    demo_te_mma_chain(ctx);
    demo_te_scale(ctx);

    printf("\n[TE stats]\n");
    NPU_RT_CHECK(npuPrintStats(ctx));
    NPU_RT_CHECK(npuFinalize(ctx));

    /* ── VC demos: use kdrv directly (npu_rt.h gap) ───────────────────────── */
    printf("\n[Vector Core — kdrv direct path]\n");
    printf("  (npu_rt.h has no npuRegisterVectorKernel(); kdrv API used directly)\n");

    npu_kdrv_t kdrv;
    kdrv_err_t ke = npu_kdrv_probe(&kdrv);
    if (ke != KDRV_OK) { fprintf(stderr, "kdrv_probe failed\n"); return 1; }

    demo_vc_vadd(kdrv);
    demo_vc_softmax(kdrv);

    printf("\n[VC stats]\n");
    npu_kdrv_print_stats(kdrv);
    npu_kdrv_remove(kdrv);

    banner("All demos complete");
    return 0;
}
