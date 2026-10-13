/* npu.c - implementation of the fully-programmable NPU functional simulator.
 * See npu.h and ../fully_prog_npu.md. */
#include "npu.h"
#include "npu_asm.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>

/* ======================================================================= */
/* Memory helpers                                                          */
/*                                                                         */
/* Two address spaces share one 32-bit space:                             */
/*   - DRAM (main memory): addresses with the high bit clear.             */
/*   - on-chip SRAM scratchpad: addresses with SCRATCH_BASE bit set;      */
/*     the low bits are the offset within the scratchpad.                 */
/* resolve() routes an access to the right backing array and bounds-checks */
/* it. is_scratch()/mem_latency() drive the cycle model.                  */
/* ======================================================================= */
static int is_scratch(uint32_t addr) { return (addr & SCRATCH_BASE) != 0; }

/* access latency in cycles for one memory transaction at this address */
static uint64_t mem_latency(uint32_t addr) {
    return is_scratch(addr) ? NPU_SRAM_LATENCY : NPU_DRAM_LATENCY;
}

static uint8_t *resolve(npu_t *n, uint32_t addr, size_t bytes) {
    if (is_scratch(addr)) {
        uint32_t off = addr & SCRATCH_MASK;
        if ((size_t)off + bytes > n->scratch_size) {
            fprintf(stderr, "NPU: out-of-bounds SRAM access at scratch+0x%x (+%zu, size %zu)\n",
                    off, bytes, n->scratch_size);
            abort();
        }
        return n->scratch + off;
    }
    if ((size_t)addr + bytes > n->mem_size) {
        fprintf(stderr, "NPU: out-of-bounds DRAM access at 0x%x (+%zu)\n", addr, bytes);
        abort();
    }
    return n->mem + addr;
}

static int32_t ld_w(npu_t *n, uint32_t a) { int32_t v; memcpy(&v, resolve(n,a,4), 4); return v; }
static void    st_w(npu_t *n, uint32_t a, int32_t v) { memcpy(resolve(n,a,4), &v, 4); }
static float   ld_f(npu_t *n, uint32_t a) { float v; memcpy(&v, resolve(n,a,4), 4); return v; }
static void    st_f(npu_t *n, uint32_t a, float v) { memcpy(resolve(n,a,4), &v, 4); }

void npu_write_f32(npu_t *n, uint32_t addr, const float *src, int c) {
    memcpy(resolve(n, addr, (size_t)c * 4), src, (size_t)c * 4);
}
void npu_read_f32(npu_t *n, uint32_t addr, float *dst, int c) {
    memcpy(dst, resolve(n, addr, (size_t)c * 4), (size_t)c * 4);
}
void npu_write_i32(npu_t *n, uint32_t addr, const int32_t *src, int c) {
    memcpy(resolve(n, addr, (size_t)c * 4), src, (size_t)c * 4);
}
void npu_read_i32(npu_t *n, uint32_t addr, int32_t *dst, int c) {
    memcpy(dst, resolve(n, addr, (size_t)c * 4), (size_t)c * 4);
}

/* ======================================================================= */
/* Control-core register helpers (x0 hardwired to zero)                    */
/* ======================================================================= */
static int64_t getx(const npu_t *n, int r) { return r == 0 ? 0 : n->x[r]; }
static void    setx(npu_t *n, int r, int64_t v) { if (r != 0) n->x[r] = v; }

/* ======================================================================= */
/* Lifecycle                                                               */
/* ======================================================================= */
npu_t *npu_create(void) {
    npu_t *n = calloc(1, sizeof(npu_t));
    n->mem = calloc(1, NPU_MEM_SIZE);
    n->mem_size = NPU_MEM_SIZE;
    n->scratch = calloc(1, NPU_SCRATCH_SIZE);
    n->scratch_size = NPU_SCRATCH_SIZE;
    n->num_cores = 1;
    return n;
}

void npu_set_cores(npu_t *n, int num_cores) {
    if (num_cores < 1) num_cores = 1;
    if (num_cores > NPU_MAX_CORES) num_cores = NPU_MAX_CORES;
    n->num_cores = num_cores;
}
void npu_destroy(npu_t *n) { if (n) { free(n->mem); free(n->scratch); free(n); } }

void npu_reset(npu_t *n) {
    memset(n->x, 0, sizeof n->x);
    memset(n->tile, 0, sizeof n->tile);
    memset(n->v, 0, sizeof n->v);
    memset(n->m, 0, sizeof n->m);
    memset(n->s, 0, sizeof n->s);
    n->vl = 0;
    memset(&n->stats, 0, sizeof n->stats);
}

int npu_register_kernel(npu_t *n, const vinst *code, int len) {
    assert(n->nkernels < MAX_KERNELS);
    int id = n->nkernels++;
    n->kernels[id].code = code;
    n->kernels[id].len  = len;
    return id;
}

/* ======================================================================= */
/* Cycle model (rough; not microarchitecturally accurate)                  */
/* ======================================================================= */
static uint64_t vec_cost(int op, int vl) {
    int passes = (vl + HW_LANES - 1) / HW_LANES;
    if (passes < 1) passes = 1;
    switch (op) {
    case V_VEXP: case V_VLOG: case V_VSQRT: case V_VRSQRT:
    case V_VRECIP: case V_VDIV:
        return (uint64_t)passes * 4;             /* transcendental/divide */
    case V_VGATHER: case V_VSCATTER:
        return (uint64_t)(vl > 0 ? vl : 1);      /* ~1 lane per cycle     */
    case V_VREDSUM: case V_VREDMAX: case V_VREDMIN:
    case V_VREDAND: case V_VREDOR: case V_VSCAN:
        return (uint64_t)passes + 4;             /* tree reduction        */
    case V_VLOAD: case V_VSTORE:
        return (uint64_t)passes;
    /* scalar-engine + control ops are single-cycle */
    case V_SLI: case V_SFI: case V_SADD: case V_SADDI: case V_SSUB:
    case V_SMUL: case V_SMULI: case V_SLW: case V_SLF: case V_SRECIP:
    case V_KID: case V_BR: case V_BRZ: case V_BRNZ: case V_LOOP:
    case V_KEXIT: case V_VSETVL:
        return 1;
    default:
        return (uint64_t)passes;                 /* elementwise           */
    }
}

/* ======================================================================= */
/* Vector core: execute one kernel instance                               */
/* ======================================================================= */
static float frecip(float x) { return 1.0f / x; }

static void run_vector_instance(npu_t *n, const npu_kernel *k,
                                int64_t inst_id, int64_t args_addr) {
    /* per-instance state: fresh registers, shared memory/scratchpad */
    memset(n->v, 0, sizeof n->v);
    memset(n->m, 0, sizeof n->m);
    memset(n->s, 0, sizeof n->s);
    n->vl = 0;
    /* launch convention: s0 = instance id, s1 = args base pointer */
    n->s[0].i = inst_id;
    n->s[1].i = args_addr;

    int pc = 0;
    int guard = 0;
    while (pc < k->len) {
        if (++guard > 100000000) { fprintf(stderr, "NPU: kernel runaway\n"); abort(); }
        const vinst *I = &k->code[pc];
        int vl = n->vl;
        n->stats.vector_cycles += vec_cost(I->op, vl);
        n->stats.vector_instret++;

        if (n->trace)
            fprintf(stderr, "  [vec pc=%d op=%d vl=%d]\n", pc, I->op, vl);

        /* per-lane predicate mask (Section 12 "@m" prefix) */
        const uint8_t *pm = I->use_mask ? n->m[I->mask] : NULL;
        #define ACTIVE(L) (pm == NULL || pm[(L)])

        uint32_t addr = I->use_base ? (uint32_t)(n->s[I->rs].i + I->imm)
                                    : (uint32_t)I->imm;

        switch (I->op) {
        /* ---- registers & memory ---- */
        case V_VSETVL: {
            int64_t req = I->use_base ? n->s[I->rs1].i : (I->rs1 ? n->s[I->rs1].i : I->imm);
            if (req > VLEN_MAX) req = VLEN_MAX;
            if (req < 0) req = 0;
            n->vl = (int)req;
            n->s[I->rd].i = n->vl;
            break;
        }
        case V_VLOAD:
            n->stats.vector_cycles += mem_latency(addr);   /* DRAM slow, SRAM fast */
            for (int l = 0; l < vl; l++) n->v[I->vd][l] = ld_f(n, addr + 4u * l);
            break;
        case V_VSTORE:
            n->stats.vector_cycles += mem_latency(addr);
            for (int l = 0; l < vl; l++) if (ACTIVE(l)) st_f(n, addr + 4u * l, n->v[I->vs][l]);
            break;
        case V_VGATHER:
            for (int l = 0; l < vl; l++) {
                int32_t idx = (int32_t)n->v[I->vs1][l];      /* index vector */
                n->v[I->vd][l] = ld_f(n, addr + 4u * (uint32_t)idx);
            }
            break;
        case V_VSCATTER:
            for (int l = 0; l < vl; l++) {
                int32_t idx = (int32_t)n->v[I->vs1][l];
                st_f(n, addr + 4u * (uint32_t)idx, n->v[I->vs][l]);
            }
            break;
        case V_VMOV:
            for (int l = 0; l < vl; l++) if (ACTIVE(l)) n->v[I->vd][l] = n->v[I->vs][l];
            break;
        case V_VBCAST: {
            float s = n->s[I->rs].f;
            for (int l = 0; l < vl; l++) if (ACTIVE(l)) n->v[I->vd][l] = s;
            break;
        }
        /* ---- engine scalar helpers ---- */
        case V_SLI:   n->s[I->rd].i = I->imm; break;
        case V_SFI:   n->s[I->rd].f = (float)I->fimm; break;
        case V_SADD:  n->s[I->rd].i = n->s[I->rs1].i + n->s[I->rs2].i; break;
        case V_SADDI: n->s[I->rd].i = n->s[I->rs1].i + I->imm; break;
        case V_SSUB:  n->s[I->rd].i = n->s[I->rs1].i - n->s[I->rs2].i; break;
        case V_SMUL:  n->s[I->rd].i = n->s[I->rs1].i * n->s[I->rs2].i; break;
        case V_SMULI: n->s[I->rd].i = n->s[I->rs1].i * I->imm; break;
        case V_SLW:   n->s[I->rd].i = ld_w(n, addr); break;
        case V_SLF:   n->s[I->rd].f = ld_f(n, addr); break;
        case V_SRECIP:n->s[I->rd].f = frecip(n->s[I->rs1].f); break;
        case V_KID:   n->s[I->rd].i = inst_id; break;
        /* ---- compute primitives ---- */
        case V_VADD: for (int l=0;l<vl;l++) if(ACTIVE(l)) n->v[I->vd][l]=n->v[I->vs1][l]+n->v[I->vs2][l]; break;
        case V_VSUB: for (int l=0;l<vl;l++) if(ACTIVE(l)) n->v[I->vd][l]=n->v[I->vs1][l]-n->v[I->vs2][l]; break;
        case V_VMUL: for (int l=0;l<vl;l++) if(ACTIVE(l)) n->v[I->vd][l]=n->v[I->vs1][l]*n->v[I->vs2][l]; break;
        case V_VDIV: for (int l=0;l<vl;l++) if(ACTIVE(l)) n->v[I->vd][l]=n->v[I->vs1][l]/n->v[I->vs2][l]; break;
        case V_VMAX: for (int l=0;l<vl;l++) if(ACTIVE(l)) n->v[I->vd][l]=fmaxf(n->v[I->vs1][l],n->v[I->vs2][l]); break;
        case V_VMIN: for (int l=0;l<vl;l++) if(ACTIVE(l)) n->v[I->vd][l]=fminf(n->v[I->vs1][l],n->v[I->vs2][l]); break;
        case V_VFMA: for (int l=0;l<vl;l++) if(ACTIVE(l)) n->v[I->vd][l]=n->v[I->vd][l]+n->v[I->vs1][l]*n->v[I->vs2][l]; break;
        case V_VCMP:
            for (int l=0;l<vl;l++) {
                float a=n->v[I->vs1][l], b=n->v[I->vs2][l]; int r=0;
                switch (I->cmp) {
                case CMP_LT: r=a<b; break;  case CMP_LE: r=a<=b; break;
                case CMP_EQ: r=a==b; break; case CMP_NE: r=a!=b; break;
                case CMP_GT: r=a>b; break;  case CMP_GE: r=a>=b; break;
                }
                n->m[I->md][l] = (uint8_t)r;
            }
            break;
        case V_VSEL: /* vd = mask ? vs1 : vs2 */
            for (int l=0;l<vl;l++) n->v[I->vd][l] = n->m[I->mask][l] ? n->v[I->vs1][l] : n->v[I->vs2][l];
            break;
        case V_VEXP:  for (int l=0;l<vl;l++) if(ACTIVE(l)) n->v[I->vd][l]=expf(n->v[I->vs][l]); break;
        case V_VLOG:  for (int l=0;l<vl;l++) if(ACTIVE(l)) n->v[I->vd][l]=logf(n->v[I->vs][l]); break;
        case V_VSQRT: for (int l=0;l<vl;l++) if(ACTIVE(l)) n->v[I->vd][l]=sqrtf(n->v[I->vs][l]); break;
        case V_VRSQRT:for (int l=0;l<vl;l++) if(ACTIVE(l)) n->v[I->vd][l]=1.0f/sqrtf(n->v[I->vs][l]); break;
        case V_VRECIP:for (int l=0;l<vl;l++) if(ACTIVE(l)) n->v[I->vd][l]=frecip(n->v[I->vs][l]); break;
        case V_VABS:  for (int l=0;l<vl;l++) if(ACTIVE(l)) n->v[I->vd][l]=fabsf(n->v[I->vs][l]); break;
        case V_VNEG:  for (int l=0;l<vl;l++) if(ACTIVE(l)) n->v[I->vd][l]=-n->v[I->vs][l]; break;
        case V_VCVT:
            for (int l=0;l<vl;l++) if(ACTIVE(l)) {
                float x = n->v[I->vs][l];
                switch (I->cvt) {
                case CVT_I8:  { float r=roundf(x); if(r>127)r=127; if(r<-128)r=-128; n->v[I->vd][l]=r; } break;
                case CVT_I32: n->v[I->vd][l]=roundf(x); break;
                default:      n->v[I->vd][l]=x; break;   /* CVT_F32 */
                }
            }
            break;
        /* ---- reductions ---- */
        case V_VREDSUM: { float a=0; for(int l=0;l<vl;l++) a+=n->v[I->vs][l]; n->s[I->rd].f=a; break; }
        case V_VREDMAX: { float a=-INFINITY; for(int l=0;l<vl;l++) if(n->v[I->vs][l]>a)a=n->v[I->vs][l]; n->s[I->rd].f=a; break; }
        case V_VREDMIN: { float a= INFINITY; for(int l=0;l<vl;l++) if(n->v[I->vs][l]<a)a=n->v[I->vs][l]; n->s[I->rd].f=a; break; }
        case V_VREDAND: { int64_t a=~0; for(int l=0;l<vl;l++) a&=(int64_t)n->v[I->vs][l]; n->s[I->rd].i=a; break; }
        case V_VREDOR:  { int64_t a=0;  for(int l=0;l<vl;l++) a|=(int64_t)n->v[I->vs][l]; n->s[I->rd].i=a; break; }
        case V_VSCAN:   { float a=0; for(int l=0;l<vl;l++){ a+=n->v[I->vs][l]; n->v[I->vd][l]=a; } break; }
        /* ---- predication & control flow ---- */
        case V_VMASK_SET: for(int l=0;l<vl;l++) n->m[I->md][l]=(uint8_t)(n->v[I->vs][l]!=0.0f); break;
        case V_VMASK_AND: for(int l=0;l<vl;l++) n->m[I->md][l]=n->m[I->m1][l]&n->m[I->m2][l]; break;
        case V_VMASK_OR:  for(int l=0;l<vl;l++) n->m[I->md][l]=n->m[I->m1][l]|n->m[I->m2][l]; break;
        case V_VMASK_NOT: for(int l=0;l<vl;l++) n->m[I->md][l]=!n->m[I->m1][l]; break;
        case V_BR:   pc = (int)I->imm; continue;
        case V_BRZ:  if (n->s[I->rs].i == 0) { pc = (int)I->imm; continue; } break;
        case V_BRNZ: if (n->s[I->rs].i != 0) { pc = (int)I->imm; continue; } break;
        case V_LOOP: /* decrement counter rs; branch to imm while non-zero */
            n->s[I->rs].i -= 1;
            if (n->s[I->rs].i > 0) { pc = (int)I->imm; continue; }
            break;
        case V_KEXIT:
            return;
        default:
            fprintf(stderr, "NPU: unknown vector op %d\n", I->op); abort();
        }
        pc++;
        #undef ACTIVE
    }
}

/* ======================================================================= */
/* Tensor engine: dense matmul fast path                                   */
/* ======================================================================= */
static void tensor_matmul(npu_t *n, int dt, int a, int b, int accumulate) {
    int M = n->cfg_M, K = n->cfg_K, N = n->cfg_N;
    assert(M <= MAX_TILE && K <= MAX_TILE && N <= MAX_TILE);
    float *A = n->tile[a], *B = n->tile[b], *D = n->tile[dt];
    for (int i = 0; i < M; i++)
        for (int j = 0; j < N; j++) {
            float acc = accumulate ? D[i * N + j] : 0.0f;
            for (int p = 0; p < K; p++)
                acc += A[i * K + p] * B[p * N + j];   /* A: MxK, B: KxN */
            D[i * N + j] = acc;
        }
    /* cycle model: M*N*K MACs at HW_LANES*HW_LANES MACs/cycle */
    uint64_t macs = (uint64_t)M * N * K;
    n->stats.tensor_cycles += (macs + (HW_LANES * HW_LANES) - 1) / (HW_LANES * HW_LANES) + 1;
}

/* ======================================================================= */
/* Control core: top-level program execution                              */
/* ======================================================================= */
void npu_run(npu_t *n, const cinst *prog, int len) {
    int pc = 0;
    int guard = 0;
    while (pc < len) {
        if (++guard > 100000000) { fprintf(stderr, "NPU: control runaway\n"); abort(); }
        const cinst *I = &prog[pc];
        int is_tensor = (I->op >= T_CFG_SHAPE);
        if (is_tensor) n->stats.tensor_instret++;
        else { n->stats.ctrl_instret++; n->stats.ctrl_cycles += 1; }

        if (n->trace) {
            char dis[128];
            npu_disasm_cinst(I, dis, sizeof dis);
            /* Pre-execution: print pc + disassembly.
             * For loads/stores also show the address; for branches show targets.
             * Register values are printed BEFORE the instruction executes so
             * the reader can follow the dataflow.                              */
            fprintf(stderr, "  [%3d] %-36s", pc, dis);
            if (I->op == C_LW || I->op == C_LH || I->op == C_LB ||
                I->op == C_SW || I->op == C_SH || I->op == C_SB) {
                uint32_t a = I->use_base
                    ? (uint32_t)(getx(n, I->rs1) + I->imm)
                    : (uint32_t)I->imm;
                if (I->op == C_SW || I->op == C_SH || I->op == C_SB)
                    fprintf(stderr, "addr=0x%08x  val=%lld",
                            (unsigned)a, (long long)getx(n, I->rs2));
                else
                    fprintf(stderr, "addr=0x%08x", (unsigned)a);
            } else if (I->op == C_ADD || I->op == C_SUB || I->op == C_MUL ||
                       I->op == C_AND || I->op == C_OR  || I->op == C_XOR ||
                       I->op == C_SLL || I->op == C_SRL || I->op == C_SLT) {
                fprintf(stderr, "x%d=%lld  x%d=%lld",
                        I->rs1, (long long)getx(n,I->rs1),
                        I->rs2, (long long)getx(n,I->rs2));
            } else if (I->op == C_ADDI) {
                fprintf(stderr, "x%d=%lld  imm=%lld",
                        I->rs1, (long long)getx(n,I->rs1), (long long)I->imm);
            } else if (I->op == C_LI) {
                fprintf(stderr, "imm=%lld", (long long)I->imm);
            } else if (I->op == C_BEQ || I->op == C_BNE ||
                       I->op == C_BLT || I->op == C_BGE) {
                fprintf(stderr, "x%d=%lld  x%d=%lld  target=%lld",
                        I->rs1, (long long)getx(n,I->rs1),
                        I->rs2, (long long)getx(n,I->rs2), (long long)I->imm);
            } else if (I->op == C_JAL) {
                fprintf(stderr, "target=%lld", (long long)I->imm);
            } else if (I->op == C_JALR) {
                fprintf(stderr, "x%d=%lld  imm=%lld",
                        I->rs1, (long long)getx(n,I->rs1), (long long)I->imm);
            }
            fprintf(stderr, "\n");
        }

        uint32_t addr = I->use_base ? (uint32_t)(getx(n, I->rs1) + I->imm)
                                    : (uint32_t)I->imm;
        switch (I->op) {
        /* ---- scalar arithmetic ---- */
        case C_ADD:  setx(n, I->rd, getx(n,I->rs1) + getx(n,I->rs2)); break;
        case C_SUB:  setx(n, I->rd, getx(n,I->rs1) - getx(n,I->rs2)); break;
        case C_ADDI: setx(n, I->rd, getx(n,I->rs1) + I->imm); break;
        case C_MUL:  setx(n, I->rd, getx(n,I->rs1) * getx(n,I->rs2)); break;
        case C_AND:  setx(n, I->rd, getx(n,I->rs1) & getx(n,I->rs2)); break;
        case C_OR:   setx(n, I->rd, getx(n,I->rs1) | getx(n,I->rs2)); break;
        case C_XOR:  setx(n, I->rd, getx(n,I->rs1) ^ getx(n,I->rs2)); break;
        case C_SLL:  setx(n, I->rd, getx(n,I->rs1) << getx(n,I->rs2)); break;
        case C_SRL:  setx(n, I->rd, (int64_t)((uint64_t)getx(n,I->rs1) >> getx(n,I->rs2))); break;
        case C_SLT:  setx(n, I->rd, getx(n,I->rs1) < getx(n,I->rs2)); break;
        case C_LI:   setx(n, I->rd, I->imm); break;
        /* ---- scalar memory ---- */
        case C_LW: setx(n, I->rd, ld_w(n, addr)); n->stats.ctrl_cycles += mem_latency(addr); break;
        case C_LH: { int16_t v; memcpy(&v,resolve(n,addr,2),2); setx(n,I->rd,v); n->stats.ctrl_cycles += mem_latency(addr); } break;
        case C_LB: { int8_t v = (int8_t)*resolve(n,addr,1); setx(n,I->rd,v); n->stats.ctrl_cycles += mem_latency(addr); } break;
        case C_SW: st_w(n, addr, (int32_t)getx(n,I->rs2)); n->stats.ctrl_cycles += mem_latency(addr); break;
        case C_SH: { int16_t v=(int16_t)getx(n,I->rs2); memcpy(resolve(n,addr,2),&v,2); n->stats.ctrl_cycles += mem_latency(addr); } break;
        case C_SB: { *resolve(n,addr,1)=(uint8_t)getx(n,I->rs2); n->stats.ctrl_cycles += mem_latency(addr); } break;
        /* ---- control flow ---- */
        case C_BEQ: if (getx(n,I->rs1)==getx(n,I->rs2)) { pc=(int)I->imm; continue; } break;
        case C_BNE: if (getx(n,I->rs1)!=getx(n,I->rs2)) { pc=(int)I->imm; continue; } break;
        case C_BLT: if (getx(n,I->rs1)< getx(n,I->rs2)) { pc=(int)I->imm; continue; } break;
        case C_BGE: if (getx(n,I->rs1)>=getx(n,I->rs2)) { pc=(int)I->imm; continue; } break;
        case C_JAL:  setx(n, I->rd, pc + 1); pc=(int)I->imm; continue;
        case C_JALR: { int64_t t=getx(n,I->rs1)+I->imm; setx(n,I->rd,pc+1); pc=(int)t; continue; }
        case C_KLAUNCH: {
            int kid = (int)I->imm;
            int64_t grid = getx(n, I->rs1);
            int64_t args = getx(n, I->rs2);
            assert(kid >= 0 && kid < n->nkernels);
            n->stats.kernels_launched++;
            n->stats.ctrl_cycles += 10;   /* launch overhead */

            int cores = n->num_cores;
            /* per-core busy time for this launch (instances spread round-robin) */
            uint64_t core_busy[NPU_MAX_CORES];
            memset(core_busy, 0, (size_t)cores * sizeof core_busy[0]);

            uint64_t launch_sum = 0;
            for (int64_t g = 0; g < grid; g++) {
                uint64_t snap = n->stats.vector_cycles;        /* before instance  */
                run_vector_instance(n, &n->kernels[kid], g, args);
                uint64_t inst = n->stats.vector_cycles - snap; /* this instance    */
                core_busy[g % cores] += inst;
                launch_sum += inst;
                n->stats.instances_run++;
            }
            /* run_vector_instance accumulated the *serial sum* into vector_cycles.
             * Replace this launch's contribution with the max-over-cores time, and
             * record the serial total separately as "work". */
            uint64_t busiest = 0;
            for (int c = 0; c < cores; c++)
                if (core_busy[c] > busiest) busiest = core_busy[c];
            n->stats.vector_cycles -= launch_sum;   /* undo serial accumulation */
            n->stats.vector_cycles += busiest;      /* charge parallel time     */
            n->stats.vector_work   += launch_sum;   /* keep serial total        */
            break;
        }
        case C_KWAIT:  n->stats.ctrl_cycles += 2; break;  /* synchronous sim */
        case C_HALT:   return;
        /* ---- tensor: config ---- */
        case T_CFG_SHAPE:  n->cfg_M=I->M; n->cfg_K=I->K; n->cfg_N=I->N; n->stats.tensor_cycles++; break;
        case T_CFG_DTYPE:  n->cfg_in_dtype=(int)I->rs1; n->cfg_acc_dtype=(int)I->rs2; n->stats.tensor_cycles++; break;
        case T_CFG_STRIDE: n->cfg_stride_r=(int)getx(n,I->rs1); n->cfg_stride_c=(int)getx(n,I->rs2); n->stats.tensor_cycles++; break;
        case T_CFG_MODE:   n->cfg_mode=(int)I->imm; n->stats.tensor_cycles++; break;
        case T_CFG_RND:    n->cfg_rnd=(int)I->imm; n->stats.tensor_cycles++; break;
        /* ---- tensor: compute ---- */
        case T_MATMUL: tensor_matmul(n, I->dt, I->a, I->b, 0); break;
        case T_MMA:    tensor_matmul(n, I->dt, I->a, I->b, 1); break;
        case T_DOT: {
            float acc=0; for (int p=0;p<n->cfg_K;p++) acc+=n->tile[I->a][p]*n->tile[I->b][p];
            n->tile[I->dt][0]=acc; n->stats.tensor_cycles += (uint64_t)n->cfg_K/HW_LANES + 1; break;
        }
        case T_CONV:   fprintf(stderr, "NPU: T_CONV not implemented (use vector core)\n"); break;
        case T_ACC_CLR: { int e=n->cfg_M*n->cfg_N; for(int i=0;i<e;i++) n->tile[I->dt][i]=0.0f; n->stats.tensor_cycles++; break; }
        /* ---- tensor: data movement ---- */
        case T_LOAD: case T_LOADW:
            for (uint32_t e=0;e<I->count;e++) n->tile[I->dt][e]=ld_f(n, addr+4u*e);
            n->stats.tensor_cycles += I->count/HW_LANES + 1 + mem_latency(addr); break;
        case T_STORE:
            for (uint32_t e=0;e<I->count;e++) st_f(n, addr+4u*e, n->tile[I->dt][e]);
            n->stats.tensor_cycles += I->count/HW_LANES + 1 + mem_latency(addr); break;
        case T_DMA_IN: case T_DMA_OUT: {
            uint32_t dst=(uint32_t)getx(n,I->rd), src=(uint32_t)getx(n,I->rs1);
            memmove(resolve(n,dst,I->count), resolve(n,src,I->count), I->count);
            n->stats.tensor_cycles += I->count/NPU_DMA_BW + 1; break;
        }
        case T_BCAST: { float s=(float)getx(n,I->rs1); int e=n->cfg_M*n->cfg_N; for(int i=0;i<e;i++) n->tile[I->dt][i]=s; n->stats.tensor_cycles++; break; }
        /* ---- tensor: activation ---- */
        case T_RELU:  { int e=n->cfg_M*n->cfg_N; for(int i=0;i<e;i++){ float x=n->tile[I->a][i]; n->tile[I->dt][i]=x>0?x:0; } n->stats.tensor_cycles += (uint64_t)e/HW_LANES+1; break; }
        case T_SCALE: { int e=n->cfg_M*n->cfg_N; for(int i=0;i<e;i++) n->tile[I->dt][i]=n->tile[I->a][i]*(float)I->fimm; n->stats.tensor_cycles += (uint64_t)e/HW_LANES+1; break; }
        case T_QUANT: { int e=n->cfg_M*n->cfg_N; for(int i=0;i<e;i++){ float q=roundf(n->tile[I->a][i]/(float)I->fimm); if(q>127)q=127; if(q<-128)q=-128; n->tile[I->dt][i]=q; } n->stats.tensor_cycles += (uint64_t)e/HW_LANES+1; break; }
        case T_DEQUANT:{ int e=n->cfg_M*n->cfg_N; for(int i=0;i<e;i++) n->tile[I->dt][i]=n->tile[I->a][i]*(float)I->fimm; n->stats.tensor_cycles += (uint64_t)e/HW_LANES+1; break; }
        case T_POOL:   fprintf(stderr, "NPU: T_POOL not implemented (use vector core)\n"); break;
        /* ---- tensor: sync ---- */
        case T_FENCE: case T_WAIT: case T_BARRIER: n->stats.tensor_cycles++; break;
        case T_STATUS: setx(n, I->rd, n->tensor_status); break;
        default:
            fprintf(stderr, "NPU: unknown control op %d\n", I->op); abort();
        }
        /* Post-execution: print what was written to a destination register. */
        if (n->trace) {
            if (I->op == C_ADD  || I->op == C_SUB  || I->op == C_MUL  ||
                I->op == C_AND  || I->op == C_OR   || I->op == C_XOR  ||
                I->op == C_SLL  || I->op == C_SRL  || I->op == C_SLT  ||
                I->op == C_ADDI || I->op == C_LI   ||
                I->op == C_LW   || I->op == C_LH   || I->op == C_LB) {
                fprintf(stderr, "         → x%d = %lld\n",
                        I->rd, (long long)getx(n, I->rd));
            }
        }
        pc++;
    }
}

/* ======================================================================= */
/* Stats                                                                   */
/* ======================================================================= */
void npu_print_stats(const npu_t *n) {
    const npu_stats *s = &n->stats;
    printf("  --- NPU stats ---------------------------------\n");
    printf("  control core : %8llu instret, ~%8llu cycles\n",
           (unsigned long long)s->ctrl_instret, (unsigned long long)s->ctrl_cycles);
    printf("  tensor engine: %8llu instret, ~%8llu cycles\n",
           (unsigned long long)s->tensor_instret, (unsigned long long)s->tensor_cycles);
    printf("  vector core  : %8llu instret, ~%8llu cycles on %d core(s)\n",
           (unsigned long long)s->vector_instret, (unsigned long long)s->vector_cycles,
           n->num_cores);
    if (s->vector_work && n->num_cores > 1) {
        double sp = s->vector_cycles ? (double)s->vector_work / (double)s->vector_cycles : 1.0;
        printf("                 (serial work ~%llu cycles -> %.2fx speedup vs 1 core)\n",
               (unsigned long long)s->vector_work, sp);
    }
    printf("  kernels launched: %llu, grid instances run: %llu\n",
           (unsigned long long)s->kernels_launched, (unsigned long long)s->instances_run);
    printf("  -----------------------------------------------\n");
}
