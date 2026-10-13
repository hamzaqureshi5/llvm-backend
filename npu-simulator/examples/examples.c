/* examples.c - worked examples that exercise all three NPU layers and
 * self-check their results. Build with `make` then run ./npu_demo.
 *
 * Launch convention for vector kernels (see run_vector_instance):
 *   s0 = grid instance id,  s1 = args base pointer (in NPU memory).
 */
#include "npu.h"
#include "npu_asm.h"
#include <stdio.h>
#include <math.h>
#include <string.h>

static int g_fail = 0;

static void check_f32(const char *name, const float *got, const float *exp, int n) {
    int ok = 1;
    for (int i = 0; i < n; i++)
        if (fabsf(got[i] - exp[i]) > 1e-4f * (1.0f + fabsf(exp[i]))) ok = 0;
    printf("  [%s] %s\n", ok ? "PASS" : "FAIL", name);
    if (!ok) {
        g_fail++;
        printf("    got:");      for (int i=0;i<n;i++) printf(" %.4f", got[i]); printf("\n");
        printf("    expected:"); for (int i=0;i<n;i++) printf(" %.4f", exp[i]); printf("\n");
    }
}

/* ======================================================================= */
/* Demo 1: elementwise vector add (vector core)                            */
/* ======================================================================= */
static void demo_vadd(void) {
    printf("\n== Demo 1: VADD (vector core) ==\n");
    enum { ARGS=0x1000, A=0x2000, B=0x3000, C=0x4000, N=8 };

    /* kernel: c[i] = a[i] + b[i] */
    static const vinst k[] = {
        { .op=V_SLW, .rd=2, .use_base=1, .rs=1, .imm=0  },   /* s2 = a ptr  */
        { .op=V_SLW, .rd=3, .use_base=1, .rs=1, .imm=4  },   /* s3 = b ptr  */
        { .op=V_SLW, .rd=4, .use_base=1, .rs=1, .imm=8  },   /* s4 = c ptr  */
        { .op=V_SLW, .rd=5, .use_base=1, .rs=1, .imm=12 },   /* s5 = n      */
        { .op=V_VSETVL, .rd=6, .rs1=5 },                     /* VL = n      */
        { .op=V_VLOAD,  .vd=0, .use_base=1, .rs=2 },         /* v0 = a      */
        { .op=V_VLOAD,  .vd=1, .use_base=1, .rs=3 },         /* v1 = b      */
        { .op=V_VADD,   .vd=2, .vs1=0, .vs2=1 },             /* v2 = v0+v1  */
        { .op=V_VSTORE, .vs=2, .use_base=1, .rs=4 },         /* c = v2      */
        { .op=V_KEXIT },
    };

    npu_t *n = npu_create();
    int kid = npu_register_kernel(n, k, sizeof k / sizeof k[0]);

    float a[N], b[N], c[N], expc[N];
    for (int i = 0; i < N; i++) { a[i]=(float)i; b[i]=10.0f+i; expc[i]=a[i]+b[i]; }
    npu_write_f32(n, A, a, N);
    npu_write_f32(n, B, b, N);
    int32_t args[4] = { A, B, C, N };
    npu_write_i32(n, ARGS, args, 4);

    /* host (control core) sets up grid=1, args ptr, launches */
    const cinst prog[] = {
        { .op=C_LI, .rd=10, .imm=1    },        /* x10 = grid (1 row)      */
        { .op=C_LI, .rd=11, .imm=ARGS },        /* x11 = args ptr          */
        { .op=C_KLAUNCH, .imm=0 /*kid*/, .rs1=10, .rs2=11 },
        { .op=C_KWAIT },
        { .op=C_HALT },
    };
    (void)kid; /* kid==0 here */
    npu_run(n, prog, sizeof prog / sizeof prog[0]);

    npu_read_f32(n, C, c, N);
    check_f32("vadd", c, expc, N);
    npu_print_stats(n);
    npu_destroy(n);
}

/* ======================================================================= */
/* Demo 2: softmax per row (Section 14 kernel, written from primitives)    */
/* ======================================================================= */
static void demo_softmax(void) {
    printf("\n== Demo 2: SOFTMAX (vector core, one instance per row) ==\n");
    enum { ARGS=0x1000, X=0x2000, Y=0x4000, N=4, ROWS=3 };

    static const vinst k[] = {
        { .op=V_SLW, .rd=2, .use_base=1, .rs=1, .imm=0 },    /* s2 = x base */
        { .op=V_SLW, .rd=3, .use_base=1, .rs=1, .imm=4 },    /* s3 = y base */
        { .op=V_SLW, .rd=4, .use_base=1, .rs=1, .imm=8 },    /* s4 = n      */
        { .op=V_SMUL,  .rd=5, .rs1=0, .rs2=4 },              /* s5 = id*n   */
        { .op=V_SMULI, .rd=5, .rs1=5, .imm=4 },              /* s5 *= 4 (bytes) */
        { .op=V_SADD,  .rd=6, .rs1=2, .rs2=5 },              /* s6 = x row  */
        { .op=V_SADD,  .rd=7, .rs1=3, .rs2=5 },              /* s7 = y row  */
        { .op=V_VSETVL, .rd=8, .rs1=4 },                     /* VL = n      */
        { .op=V_VLOAD,  .vd=0, .use_base=1, .rs=6 },         /* v0 = row    */
        { .op=V_VREDMAX, .rd=9, .vs=0 },                     /* s9 = max(x) */
        { .op=V_VBCAST, .vd=1, .rs=9 },                      /* v1 = max    */
        { .op=V_VSUB,   .vd=0, .vs1=0, .vs2=1 },             /* x - max     */
        { .op=V_VEXP,   .vd=0, .vs=0 },                      /* exp(.)      */
        { .op=V_VREDSUM, .rd=10, .vs=0 },                    /* s10 = sum   */
        { .op=V_SRECIP, .rd=11, .rs1=10 },                   /* s11 = 1/sum */
        { .op=V_VBCAST, .vd=1, .rs=11 },                     /* v1 = 1/sum  */
        { .op=V_VMUL,   .vd=0, .vs1=0, .vs2=1 },             /* normalize   */
        { .op=V_VSTORE, .vs=0, .use_base=1, .rs=7 },         /* y = v0      */
        { .op=V_KEXIT },
    };

    npu_t *n = npu_create();
    npu_register_kernel(n, k, sizeof k / sizeof k[0]);

    float x[ROWS*N] = {
        1.0f, 2.0f, 3.0f, 4.0f,
        0.5f, 0.5f, 0.5f, 0.5f,
       -1.0f, 0.0f, 1.0f, 2.0f,
    };
    npu_write_f32(n, X, x, ROWS*N);
    int32_t args[3] = { X, Y, N };
    npu_write_i32(n, ARGS, args, 3);

    const cinst prog[] = {
        { .op=C_LI, .rd=10, .imm=ROWS },
        { .op=C_LI, .rd=11, .imm=ARGS },
        { .op=C_KLAUNCH, .imm=0, .rs1=10, .rs2=11 },
        { .op=C_KWAIT },
        { .op=C_HALT },
    };
    npu_run(n, prog, sizeof prog / sizeof prog[0]);

    /* reference softmax */
    float exp_[ROWS*N];
    for (int r = 0; r < ROWS; r++) {
        float mx=-INFINITY, sum=0;
        for (int i=0;i<N;i++) if (x[r*N+i]>mx) mx=x[r*N+i];
        for (int i=0;i<N;i++) { exp_[r*N+i]=expf(x[r*N+i]-mx); sum+=exp_[r*N+i]; }
        for (int i=0;i<N;i++) exp_[r*N+i]/=sum;
    }
    float y[ROWS*N];
    npu_read_f32(n, Y, y, ROWS*N);
    check_f32("softmax", y, exp_, ROWS*N);
    npu_print_stats(n);
    npu_destroy(n);
}

/* ======================================================================= */
/* Demo 3: matmul on the tensor-engine fast path                           */
/* ======================================================================= */
static void demo_matmul(void) {
    printf("\n== Demo 3: MATMUL (tensor engine, 2x3 * 3x2) ==\n");
    enum { A=0x2000, B=0x3000, D=0x4000, M=2, K=3, NN=2 };

    npu_t *n = npu_create();
    float a[M*K] = { 1,2,3, 4,5,6 };
    float b[K*NN] = { 7,8, 9,10, 11,12 };
    npu_write_f32(n, A, a, M*K);
    npu_write_f32(n, B, b, K*NN);

    const cinst prog[] = {
        { .op=T_CFG_SHAPE, .M=M, .K=K, .N=NN },
        { .op=C_LI, .rd=5, .imm=A },
        { .op=C_LI, .rd=6, .imm=B },
        { .op=C_LI, .rd=7, .imm=D },
        { .op=T_LOAD,  .dt=0, .use_base=1, .rs1=5, .count=M*K },   /* tile0 = A */
        { .op=T_LOADW, .dt=1, .use_base=1, .rs1=6, .count=K*NN },  /* tile1 = B */
        { .op=T_ACC_CLR, .dt=2 },
        { .op=T_MMA, .dt=2, .a=0, .b=1 },                          /* D = A*B   */
        { .op=T_STORE, .dt=2, .use_base=1, .rs1=7, .count=M*NN },
        { .op=C_HALT },
    };
    npu_run(n, prog, sizeof prog / sizeof prog[0]);

    float d[M*NN], expd[M*NN] = { 58, 64, 139, 154 };
    npu_read_f32(n, D, d, M*NN);
    check_f32("matmul", d, expd, M*NN);
    npu_print_stats(n);
    npu_destroy(n);
}

/* ======================================================================= */
/* Demo 4: predication & branchless conditionals (Section 12)              */
/*   y = (x > 0) ? x + 100 : 0   built from VCMP + VSEL + predicated VADD   */
/* ======================================================================= */
static void demo_predication(void) {
    printf("\n== Demo 4: PREDICATION (VCMP / VSEL / @mask, vector core) ==\n");
    enum { ARGS=0x1000, X=0x2000, Y=0x4000, N=8 };

    static const vinst k[] = {
        { .op=V_SLW, .rd=2, .use_base=1, .rs=1, .imm=0 },    /* s2 = x ptr  */
        { .op=V_SLW, .rd=3, .use_base=1, .rs=1, .imm=4 },    /* s3 = y ptr  */
        { .op=V_SLW, .rd=4, .use_base=1, .rs=1, .imm=8 },    /* s4 = n      */
        { .op=V_VSETVL, .rd=5, .rs1=4 },
        { .op=V_VLOAD,  .vd=0, .use_base=1, .rs=2 },         /* v0 = x      */
        { .op=V_SFI,    .rd=6, .fimm=0.0 },
        { .op=V_VBCAST, .vd=1, .rs=6 },                      /* v1 = 0      */
        { .op=V_VCMP,   .md=0, .vs1=0, .vs2=1, .cmp=CMP_GT },/* m0 = x>0    */
        { .op=V_VSEL,   .vd=2, .vs1=0, .vs2=1, .mask=0 },    /* v2 = x>0?x:0 (ReLU) */
        { .op=V_SFI,    .rd=7, .fimm=100.0 },
        { .op=V_VBCAST, .vd=3, .rs=7 },                      /* v3 = 100    */
        { .op=V_VADD,   .vd=2, .vs1=2, .vs2=3, .use_mask=1, .mask=0 }, /* +100 on positive lanes */
        { .op=V_VSTORE, .vs=2, .use_base=1, .rs=3 },
        { .op=V_KEXIT },
    };

    npu_t *n = npu_create();
    npu_register_kernel(n, k, sizeof k / sizeof k[0]);

    float x[N] = { -2,-1,0,1,2,3,-3,4 }, y[N], expy[N];
    for (int i=0;i<N;i++) expy[i] = x[i] > 0 ? x[i] + 100.0f : 0.0f;
    npu_write_f32(n, X, x, N);
    int32_t args[3] = { X, Y, N };
    npu_write_i32(n, ARGS, args, 3);

    const cinst prog[] = {
        { .op=C_LI, .rd=10, .imm=1 },
        { .op=C_LI, .rd=11, .imm=ARGS },
        { .op=C_KLAUNCH, .imm=0, .rs1=10, .rs2=11 },
        { .op=C_KWAIT },
        { .op=C_HALT },
    };
    npu_run(n, prog, sizeof prog / sizeof prog[0]);

    npu_read_f32(n, Y, y, N);
    check_f32("predicated relu+bias", y, expy, N);
    npu_print_stats(n);
    npu_destroy(n);
}

/* ======================================================================= */
/* Demo 5: multi-core cycle model (max-over-cores)                         */
/*   Same softmax kernel over a large grid; instances spread across cores. */
/* ======================================================================= */
static void demo_multicore(void) {
    printf("\n== Demo 5: MULTI-CORE (softmax over 64 rows, 8 vector cores) ==\n");
    enum { ARGS=0x1000, X=0x2000, Y=0x40000, N=16, ROWS=64, CORES=8 };

    static const vinst k[] = {
        { .op=V_SLW, .rd=2, .use_base=1, .rs=1, .imm=0 },
        { .op=V_SLW, .rd=3, .use_base=1, .rs=1, .imm=4 },
        { .op=V_SLW, .rd=4, .use_base=1, .rs=1, .imm=8 },
        { .op=V_SMUL,  .rd=5, .rs1=0, .rs2=4 },
        { .op=V_SMULI, .rd=5, .rs1=5, .imm=4 },
        { .op=V_SADD,  .rd=6, .rs1=2, .rs2=5 },
        { .op=V_SADD,  .rd=7, .rs1=3, .rs2=5 },
        { .op=V_VSETVL, .rd=8, .rs1=4 },
        { .op=V_VLOAD,  .vd=0, .use_base=1, .rs=6 },
        { .op=V_VREDMAX, .rd=9, .vs=0 },
        { .op=V_VBCAST, .vd=1, .rs=9 },
        { .op=V_VSUB,   .vd=0, .vs1=0, .vs2=1 },
        { .op=V_VEXP,   .vd=0, .vs=0 },
        { .op=V_VREDSUM, .rd=10, .vs=0 },
        { .op=V_SRECIP, .rd=11, .rs1=10 },
        { .op=V_VBCAST, .vd=1, .rs=11 },
        { .op=V_VMUL,   .vd=0, .vs1=0, .vs2=1 },
        { .op=V_VSTORE, .vs=0, .use_base=1, .rs=7 },
        { .op=V_KEXIT },
    };

    npu_t *n = npu_create();
    npu_set_cores(n, CORES);
    npu_register_kernel(n, k, sizeof k / sizeof k[0]);

    float x[ROWS*N];
    for (int i = 0; i < ROWS*N; i++) x[i] = (float)((i * 7) % 13) - 6.0f;
    npu_write_f32(n, X, x, ROWS*N);
    int32_t args[3] = { X, Y, N };
    npu_write_i32(n, ARGS, args, 3);

    const cinst prog[] = {
        { .op=C_LI, .rd=10, .imm=ROWS },
        { .op=C_LI, .rd=11, .imm=ARGS },
        { .op=C_KLAUNCH, .imm=0, .rs1=10, .rs2=11 },
        { .op=C_KWAIT },
        { .op=C_HALT },
    };
    npu_run(n, prog, sizeof prog / sizeof prog[0]);

    /* reference softmax to confirm correctness is unaffected by core count */
    float exp_[ROWS*N];
    for (int r = 0; r < ROWS; r++) {
        float mx=-INFINITY, sum=0;
        for (int i=0;i<N;i++) if (x[r*N+i]>mx) mx=x[r*N+i];
        for (int i=0;i<N;i++) { exp_[r*N+i]=expf(x[r*N+i]-mx); sum+=exp_[r*N+i]; }
        for (int i=0;i<N;i++) exp_[r*N+i]/=sum;
    }
    float y[ROWS*N];
    npu_read_f32(n, Y, y, ROWS*N);
    check_f32("softmax x64 (8 cores)", y, exp_, ROWS*N);
    npu_print_stats(n);
    npu_destroy(n);
}

/* ======================================================================= */
/* Demo 6: on-chip SRAM scratchpad + DMA staging                           */
/*   The SAME vadd kernel, run two ways:                                    */
/*     (A) operands live in DRAM  -> every vector load/store pays DRAM cost */
/*     (B) DMA operands into SRAM -> vector loads/stores are ~free          */
/*   Shows why staging to on-chip memory is the core NPU programming move.  */
/* ======================================================================= */
static const vinst vadd_kernel[] = {
    { .op=V_SLW, .rd=2, .use_base=1, .rs=1, .imm=0  },   /* s2 = a ptr */
    { .op=V_SLW, .rd=3, .use_base=1, .rs=1, .imm=4  },   /* s3 = b ptr */
    { .op=V_SLW, .rd=4, .use_base=1, .rs=1, .imm=8  },   /* s4 = c ptr */
    { .op=V_SLW, .rd=5, .use_base=1, .rs=1, .imm=12 },   /* s5 = n     */
    { .op=V_VSETVL, .rd=6, .rs1=5 },
    { .op=V_VLOAD,  .vd=0, .use_base=1, .rs=2 },
    { .op=V_VLOAD,  .vd=1, .use_base=1, .rs=3 },
    { .op=V_VADD,   .vd=2, .vs1=0, .vs2=1 },
    { .op=V_VSTORE, .vs=2, .use_base=1, .rs=4 },
    { .op=V_KEXIT },
};

static void demo_sram(void) {
    printf("\n== Demo 6: ON-CHIP SRAM + DMA STAGING (vadd, N=256) ==\n");
    const int N = 256;
    const uint32_t ARGSD=0x1000, ARGSS=0x1100;
    const uint32_t A=0x2000, B=0x3000, C=0x4000, COUT=0x5000;
    const uint32_t sA=SCRATCH_BASE+0x0000, sB=SCRATCH_BASE+0x1000, sC=SCRATCH_BASE+0x2000;
    const uint32_t bytes = (uint32_t)N * 4;

    float a[256], b[256], expc[256];
    for (int i=0;i<N;i++){ a[i]=(float)i; b[i]=2.0f*i; expc[i]=a[i]+b[i]; }

    /* ---- (A) operands in DRAM ---- */
    npu_t *n1 = npu_create();
    npu_register_kernel(n1, vadd_kernel, sizeof vadd_kernel / sizeof vadd_kernel[0]);
    npu_write_f32(n1, A, a, N);
    npu_write_f32(n1, B, b, N);
    int32_t argsd[4] = { (int32_t)A, (int32_t)B, (int32_t)C, N };
    npu_write_i32(n1, ARGSD, argsd, 4);
    const cinst progA[] = {
        { .op=C_LI, .rd=10, .imm=1 },
        { .op=C_LI, .rd=11, .imm=ARGSD },
        { .op=C_KLAUNCH, .imm=0, .rs1=10, .rs2=11 },
        { .op=C_KWAIT },
        { .op=C_HALT },
    };
    npu_run(n1, progA, sizeof progA / sizeof progA[0]);
    float c1[256]; npu_read_f32(n1, C, c1, N);
    check_f32("vadd from DRAM", c1, expc, N);

    /* ---- (B) DMA operands into SRAM, compute on-chip, DMA result out ---- */
    npu_t *n2 = npu_create();
    npu_register_kernel(n2, vadd_kernel, sizeof vadd_kernel / sizeof vadd_kernel[0]);
    npu_write_f32(n2, A, a, N);
    npu_write_f32(n2, B, b, N);
    int32_t argss[4] = { (int32_t)sA, (int32_t)sB, (int32_t)sC, N };
    npu_write_i32(n2, ARGSS, argss, 4);
    const cinst progB[] = {
        { .op=C_LI, .rd=5,  .imm=sA }, { .op=C_LI, .rd=6,  .imm=A },
        { .op=T_DMA_IN,  .rd=5, .rs1=6, .count=bytes },        /* sA <- A */
        { .op=C_LI, .rd=7,  .imm=sB }, { .op=C_LI, .rd=8,  .imm=B },
        { .op=T_DMA_IN,  .rd=7, .rs1=8, .count=bytes },        /* sB <- B */
        { .op=C_LI, .rd=10, .imm=1 },
        { .op=C_LI, .rd=11, .imm=ARGSS },
        { .op=C_KLAUNCH, .imm=0, .rs1=10, .rs2=11 },           /* compute in SRAM */
        { .op=C_KWAIT },
        { .op=C_LI, .rd=9,  .imm=COUT }, { .op=C_LI, .rd=12, .imm=sC },
        { .op=T_DMA_OUT, .rd=9, .rs1=12, .count=bytes },       /* COUT <- sC */
        { .op=C_HALT },
    };
    npu_run(n2, progB, sizeof progB / sizeof progB[0]);
    float c2[256]; npu_read_f32(n2, COUT, c2, N);
    check_f32("vadd via SRAM staging", c2, expc, N);

    printf("  vector-core cycles:  DRAM-resident = %llu,  SRAM-staged = %llu\n",
           (unsigned long long)n1->stats.vector_cycles,
           (unsigned long long)n2->stats.vector_cycles);
    printf("  DMA cost (tensor engine) for staging = %llu cycles\n",
           (unsigned long long)n2->stats.tensor_cycles);
    printf("  -> on-chip loads/stores cost %d cyc each vs %d for DRAM (see npu.h)\n",
           NPU_SRAM_LATENCY, NPU_DRAM_LATENCY);
    npu_destroy(n1);
    npu_destroy(n2);
}

/* ======================================================================= */
/* Demo 7: text assembler + binary encoder/decoder + disassembler          */
/*   Write softmax in ASSEMBLY, assemble -> run. Then round-trip the kernel */
/*   through the binary encoder/decoder and confirm it still runs.          */
/* ======================================================================= */
static void demo_assembler(void) {
    printf("\n== Demo 7: ASSEMBLER + ENCODER/DECODER (softmax from text) ==\n");
    enum { ARGS=0x1000, X=0x2000, Y=0x4000, N=4, ROWS=3 };

    /* The softmax kernel, written as assembly text instead of C structs.    */
    const char *ksrc =
        "softmax:\n"
        "  SLW   s2, 0(s1)      ; s2 = x base\n"
        "  SLW   s3, 4(s1)      ; s3 = y base\n"
        "  SLW   s4, 8(s1)      ; s4 = n\n"
        "  SMUL  s5, s0, s4     ; id*n\n"
        "  SMULI s5, s5, 4      ; bytes\n"
        "  SADD  s6, s2, s5     ; x row ptr\n"
        "  SADD  s7, s3, s5     ; y row ptr\n"
        "  VSETVL s8, s4        ; VL = n\n"
        "  VLOAD v0, 0(s6)\n"
        "  VREDMAX s9, v0\n"
        "  VBCAST v1, s9\n"
        "  VSUB  v0, v0, v1\n"
        "  VEXP  v0, v0\n"
        "  VREDSUM s10, v0\n"
        "  SRECIP s11, s10\n"
        "  VBCAST v1, s11\n"
        "  VMUL  v0, v0, v1\n"
        "  VSTORE v0, 0(s7)\n"
        "  KEXIT\n";

    const char *csrc =
        "  LI x10, 3            ; grid = ROWS\n"
        "  LI x11, 0x1000       ; args ptr\n"
        "  KLAUNCH 0, x10, x11\n"
        "  KWAIT\n"
        "  HALT\n";

    char err[128];
    vinst kbuf[64]; cinst cbuf[16];
    int kn = npu_asm_vector(ksrc, kbuf, 64, err, sizeof err);
    if (kn < 0) { printf("  [FAIL] assemble kernel: %s\n", err); g_fail++; return; }
    int cn = npu_asm_control(csrc, cbuf, 16, err, sizeof err);
    if (cn < 0) { printf("  [FAIL] assemble control: %s\n", err); g_fail++; return; }
    printf("  assembled %d kernel + %d control instructions\n", kn, cn);

    /* show the assembler round-trips through the disassembler */
    char dis[80];
    npu_disasm_vinst(&kbuf[9], dis, sizeof dis);   /* VREDMAX */
    printf("  disasm kbuf[9]: %s\n", dis);

    float x[ROWS*N] = { 1,2,3,4,  0.5f,0.5f,0.5f,0.5f,  -1,0,1,2 };
    float exp_[ROWS*N];
    for (int r=0;r<ROWS;r++){ float mx=-INFINITY,sum=0;
        for(int i=0;i<N;i++) if(x[r*N+i]>mx)mx=x[r*N+i];
        for(int i=0;i<N;i++){ exp_[r*N+i]=expf(x[r*N+i]-mx); sum+=exp_[r*N+i]; }
        for(int i=0;i<N;i++) exp_[r*N+i]/=sum; }
    int32_t args[3] = { X, Y, N };

    /* (A) run the assembled kernel directly */
    npu_t *n = npu_create();
    npu_register_kernel(n, kbuf, kn);
    npu_write_f32(n, X, x, ROWS*N);
    npu_write_i32(n, ARGS, args, 3);
    npu_run(n, cbuf, cn);
    float y[ROWS*N]; npu_read_f32(n, Y, y, ROWS*N);
    check_f32("assembled-from-text softmax", y, exp_, ROWS*N);
    npu_destroy(n);

    /* (B) encode each instruction to bytes, decode back, run the decoded copy */
    vinst kbuf2[64]; int identical = 1;
    for (int i = 0; i < kn; i++) {
        unsigned char word[NPU_WORD_BYTES];
        npu_encode_vinst(&kbuf[i], word);
        npu_decode_vinst(word, &kbuf2[i]);
        if (memcmp(&kbuf[i], &kbuf2[i], sizeof(vinst)) != 0) identical = 0;
    }
    printf("  encode->decode round-trip %s (%d bytes/instr)\n",
           identical ? "bit-identical" : "MISMATCH", NPU_WORD_BYTES);
    if (!identical) g_fail++;

    npu_t *n2 = npu_create();
    npu_register_kernel(n2, kbuf2, kn);
    npu_write_f32(n2, X, x, ROWS*N);
    npu_write_i32(n2, ARGS, args, 3);
    npu_run(n2, cbuf, cn);
    float y2[ROWS*N]; npu_read_f32(n2, Y, y2, ROWS*N);
    check_f32("decoded-from-bytes softmax", y2, exp_, ROWS*N);
    npu_destroy(n2);
}

int main(void) {
    printf("Fully-programmable NPU functional simulator - demos\n");
    demo_vadd();
    demo_softmax();
    demo_matmul();
    demo_predication();
    demo_multicore();
    demo_sram();
    demo_assembler();
    printf("\n%s\n", g_fail == 0 ? "ALL DEMOS PASSED" : "SOME DEMOS FAILED");
    return g_fail == 0 ? 0 : 1;
}
