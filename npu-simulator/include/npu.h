/* npu.h - Functional simulator for the fully-programmable NPU described in
 * ../fully_prog_npu.md
 *
 * Three layers, all interpreted from C struct instruction arrays:
 *   Layer 1 - Control core  (scalar ISA + kernel launch)        -> cinst stream
 *   Layer 2 - Tensor engine (dense matmul fast path)            -> cinst stream
 * (NPU.* ops) Layer 3 - Vector core   (programmable per-grid-point kernel) ->
 * vinst stream
 *
 * Fidelity: functional (correct results) + rough cycle estimates per engine.
 * This is an instruction-set simulator (ISS), not cycle-accurate hardware.
 */
#ifndef NPU_H
#define NPU_H

#include <stddef.h>
#include <stdint.h>

/* ----------------------------------------------------------------------- */
/* Machine sizing                                                          */
/* ----------------------------------------------------------------------- */
#define NPU_MEM_SIZE (16u * 1024u * 1024u) /* off-chip DRAM (main memory)  */
#define NPU_SCRATCH_SIZE (256u * 1024u)    /* on-chip SRAM scratchpad      */
/* Address tagging: high bit set => on-chip scratchpad, low bits = offset.
 * High bit clear => DRAM. (DRAM is 16 MB, so the high bit is always free.) */
#define SCRATCH_BASE 0x80000000u
#define SCRATCH_MASK 0x7fffffffu
/* Rough per-access latencies (cycles) used only by the cycle model. */
#define NPU_DRAM_LATENCY 100 /* off-chip access is expensive                */
#define NPU_SRAM_LATENCY 1   /* on-chip scratchpad is cheap                 */
#define NPU_DMA_BW 32        /* DMA engine bandwidth, bytes per cycle       */
#define VLEN_MAX 256         /* max lanes in a vector register              */
#define HW_LANES 16          /* physical lanes (used only for cycle model)  */
#define NVREG 32             /* v0..v31  */
#define NMREG 8              /* m0..m7   */
#define NSREG 32             /* engine scalar regs s0..s31 */
#define NCREG 32             /* control core regs x0..x31  */
#define NTREG 8              /* tensor tile registers      */
#define MAX_TILE 64          /* max tile edge (MAX_TILE x MAX_TILE floats)  */
#define MAX_KERNELS 32       /* kernels the host can register               */
#define NPU_MAX_CORES 1024   /* upper bound on simulated vector cores       */

/* ----------------------------------------------------------------------- */
/* Layer 1+2: control-core / tensor-engine opcodes (one stream)            */
/* ----------------------------------------------------------------------- */
typedef enum {
  /* --- scalar arithmetic (Section 1) --- */
  C_ADD,
  C_SUB,
  C_ADDI,
  C_MUL,
  C_AND,
  C_OR,
  C_XOR,
  C_SLL,
  C_SRL,
  C_SLT,
  C_LI, /* load immediate (convenience)            */
  /* --- scalar memory (Section 2) --- */
  C_LW,
  C_LH,
  C_LB,
  C_SW,
  C_SH,
  C_SB,
  /* --- control flow + launch (Section 3) --- */
  C_BEQ,
  C_BNE,
  C_BLT,
  C_BGE,
  C_JAL,
  C_JALR,
  C_KLAUNCH,
  C_KWAIT,
  C_HALT,

  /* --- tensor engine config (Section 4) --- */
  T_CFG_SHAPE,
  T_CFG_DTYPE,
  T_CFG_STRIDE,
  T_CFG_MODE,
  T_CFG_RND,
  /* --- tensor engine compute (Section 5) --- */
  T_MATMUL,
  T_MMA,
  T_DOT,
  T_CONV,
  T_ACC_CLR,
  /* --- tensor engine data movement (Section 6) --- */
  T_LOAD,
  T_STORE,
  T_LOADW,
  T_DMA_IN,
  T_DMA_OUT,
  T_BCAST,
  /* --- tensor engine activation (Section 7) --- */
  T_RELU,
  T_QUANT,
  T_DEQUANT,
  T_SCALE,
  T_POOL,
  /* --- tensor engine sync (Section 8) --- */
  T_FENCE,
  T_WAIT,
  T_BARRIER,
  T_STATUS
} cop_t;

/* ----------------------------------------------------------------------- */
/* Layer 3: vector-core opcodes                                            */
/* ----------------------------------------------------------------------- */
typedef enum {
  /* --- registers & memory (Section 9) --- */
  V_VSETVL,
  V_VLOAD,
  V_VSTORE,
  V_VGATHER,
  V_VSCATTER,
  V_VMOV,
  V_VBCAST,
  /* --- engine-side scalar helpers (needed for SIMT addressing) --- */
  V_SLI,
  V_SFI,
  V_SADD,
  V_SADDI,
  V_SSUB,
  V_SMUL,
  V_SMULI,
  V_SLW,
  V_SLF,
  V_SRECIP,
  V_KID,
  /* --- compute primitives (Section 10) --- */
  V_VADD,
  V_VSUB,
  V_VMUL,
  V_VDIV,
  V_VMAX,
  V_VMIN,
  V_VFMA,
  V_VCMP,
  V_VSEL,
  V_VEXP,
  V_VLOG,
  V_VRSQRT,
  V_VSQRT,
  V_VRECIP,
  V_VCVT,
  V_VABS,
  V_VNEG,
  /* --- reductions (Section 11) --- */
  V_VREDSUM,
  V_VREDMAX,
  V_VREDMIN,
  V_VREDAND,
  V_VREDOR,
  V_VSCAN,
  /* --- predication & control flow (Section 12) --- */
  V_VMASK_SET,
  V_VMASK_AND,
  V_VMASK_OR,
  V_VMASK_NOT,
  V_BR,
  V_BRZ,
  V_BRNZ,
  V_LOOP,
  V_KEXIT
} vop_t;

/* VCMP comparison kinds */
typedef enum { CMP_LT, CMP_LE, CMP_EQ, CMP_NE, CMP_GT, CMP_GE } cmpkind_t;

/* VCVT conversion modes */
typedef enum { CVT_F32, CVT_I8, CVT_I32 } cvtmode_t;

/* ----------------------------------------------------------------------- */
/* Instruction structs (built with designated initializers in C)          */
/* ----------------------------------------------------------------------- */

/* Control / tensor instruction. Unused fields are simply never read for a
 * given opcode, so zero-initialization is safe. */
typedef struct {
  cop_t op;
  int rd, rs1, rs2; /* control integer regs x0..x31              */
  int64_t imm;      /* immediate / branch target / kernel id     */
  /* tensor operands */
  int dt, a, b;   /* tile registers                            */
  int M, K, N;    /* CFG.SHAPE dims                            */
  uint32_t count; /* element/byte count for load/store/dma     */
  int use_base;   /* 1 => address = x[rs1] + imm ; 0 => imm    */
  double fimm;    /* float immediate (e.g. T_SCALE factor)     */
} cinst;

/* Vector-core instruction. */
typedef struct {
  vop_t op;
  int vd, vs1, vs2, vs; /* vector regs */
  int rd, rs, rs1, rs2; /* engine scalar regs s0..s31 */
  int md, m1, m2;       /* mask regs */
  int use_mask, mask;   /* per-lane predicate (Section 12 "@m") */
  int use_base;         /* 1 => addr = s[rs] + imm ; 0 => imm */
  int64_t imm;          /* immediate / branch target index */
  double fimm;          /* float immediate */
  int cmp;              /* cmpkind_t for VCMP */
  int cvt;              /* cvtmode_t for VCVT */
} vinst;

/* ----------------------------------------------------------------------- */
/* State                                                                   */
/* ----------------------------------------------------------------------- */
typedef union {
  float f;
  int64_t i;
} sreg_t; /* engine scalar register */

typedef struct {
  uint64_t ctrl_cycles, tensor_cycles, vector_cycles;
  uint64_t vector_work; /* serial sum of per-instance vector cycles      */
  uint64_t ctrl_instret, tensor_instret, vector_instret;
  uint64_t kernels_launched, instances_run;
} npu_stats;

typedef struct {
  const vinst *code;
  int len;
} npu_kernel;

typedef struct {
  /* memory */
  uint8_t *mem; /* off-chip DRAM (main memory)        */
  size_t mem_size;
  uint8_t *scratch; /* on-chip SRAM scratchpad            */
  size_t scratch_size;
  /* control core */
  int64_t x[NCREG];
  /* tensor engine */
  float tile[NTREG][MAX_TILE * MAX_TILE];
  int cfg_M, cfg_K, cfg_N;
  int cfg_in_dtype, cfg_acc_dtype;
  int cfg_mode, cfg_rnd;
  int cfg_stride_r, cfg_stride_c;
  int64_t tensor_status;
  /* vector engine (reset per grid instance) */
  float v[NVREG][VLEN_MAX];
  uint8_t m[NMREG][VLEN_MAX];
  sreg_t s[NSREG];
  int vl;
  /* kernel table */
  npu_kernel kernels[MAX_KERNELS];
  int nkernels;
  /* misc */
  int num_cores; /* vector cores; grid instances spread across them */
  int trace;     /* 1 => print each executed instruction */
  npu_stats stats;
} npu_t;

/* ----------------------------------------------------------------------- */
/* Public API                                                              */
/* ----------------------------------------------------------------------- */
npu_t *npu_create(void);
void npu_destroy(npu_t *n);
void npu_reset(npu_t *n);

/* Set the number of simulated vector cores (clamped to [1, NPU_MAX_CORES]).
 * Grid instances of a KLAUNCH are spread round-robin across the cores; the
 * modeled vector cycles for a launch become the max over cores (the busiest
 * core), while stats.vector_work keeps the serial sum. */
void npu_set_cores(npu_t *n, int num_cores);

/* Register a vector-core kernel; returns a kernel id for use with C_KLAUNCH. */
int npu_register_kernel(npu_t *n, const vinst *code, int len);

/* Run a control-core program until C_HALT. */
void npu_run(npu_t *n, const cinst *prog, int len);

/* Host helpers for poking/peeking memory (load inputs, check outputs). */
void npu_write_f32(npu_t *n, uint32_t addr, const float *src, int count);
void npu_read_f32(npu_t *n, uint32_t addr, float *dst, int count);
void npu_write_i32(npu_t *n, uint32_t addr, const int32_t *src, int count);
void npu_read_i32(npu_t *n, uint32_t addr, int32_t *dst, int count);

void npu_print_stats(const npu_t *n);

#endif /* NPU_H */
