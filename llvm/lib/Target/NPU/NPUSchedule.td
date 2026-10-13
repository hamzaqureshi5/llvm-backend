//===- NPUSchedule.td - NPU scheduling model ---------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Scheduling model for the NPU control core, tensor engine, and vector core.
//
// Fidelity note: this model is derived from the simulator's rough cycle
// estimates (npu.c cycle model), not from real hardware timing.  It captures
// the major relative costs (DRAM >> SRAM >> ALU; MATMUL >> elementwise).
// The model uses in-order execution (MicroOpBufferSize=0) which is appropriate
// for the control core's straight-line, single-issue ISA.
//
// Latency constants (cycles, from npu.h / npu.c):
//   NPU_DRAM_LATENCY = 100 (off-chip access)
//   NPU_SRAM_LATENCY = 1   (on-chip scratchpad)
//   NPU_DMA_BW       = 32  (bytes/cycle)
//   HW_LANES         = 16  (physical vector lanes)
//
//===----------------------------------------------------------------------===//

// ── Machine model ────────────────────────────────────────────────────────────

// Single-issue, in-order control core.  MicroOpBufferSize=0 forces
// the scheduler to treat this as a fully in-order machine — instructions are
// issued in program order with no out-of-order window.
def NPUSchedModel : SchedMachineModel {
  let MicroOpBufferSize    = 0;   // in-order issue
  let IssueWidth           = 1;   // one instruction per cycle
  let LoadLatency          = 4;   // baseline memory latency (SRAM path)
  let MispredictPenalty    = 4;   // branch mispredict (flush + refetch)
  let CompleteModel        = 0;   // allow instructions without explicit sched info
}

// ── SchedWrite / SchedRead type declarations ──────────────────────────────────
// SchedWrite and SchedRead are pure type markers — they have no SchedModel
// field, so they must be defined at the top level, NOT inside a
// "let SchedModel = ... in { }" block.

// Control-core ALU: 1 cycle (ADD, SUB, MUL, AND, OR, XOR, SLL, SRL, SLT).
def WriteALU : SchedWrite;
def WriteALUI : SchedWrite;    // register-immediate form (ADDI)

// Loads from on-chip SRAM: 1 cycle; from DRAM: up to 100 cycles.
// We model a conservative 4-cycle average matching most SRAM cache hits.
def WriteLoad  : SchedWrite;
def WriteStore : SchedWrite;

// Control-flow instructions consume the ALU for 1 cycle.
def WriteBranch : SchedWrite;
def WriteCall   : SchedWrite;

// Tensor engine operations.  Latencies reflect the simulator's cycle model:
//   MATMUL/MMA: O(M*K*N / HW_LANES^2) — use 64 cycles for the default 64×64×64 tile
//   DMA: O(bytes / NPU_DMA_BW) — 32 cycles for a 1KB tile
def WriteTensorMatmul : SchedWrite;  // T_MATMUL, T_MMA, T_DOT
def WriteTensorLoad   : SchedWrite;  // T_LOAD, T_LOADW
def WriteTensorStore  : SchedWrite;  // T_STORE
def WriteTensorDMA    : SchedWrite;  // T_DMA_IN, T_DMA_OUT
def WriteTensorAct    : SchedWrite;  // T_RELU, T_SCALE
def WriteTensorCfg    : SchedWrite;  // T_CFG_SHAPE, T_FENCE, T_WAIT, T_BARRIER

// Vector core operations.  The vector engine runs HW_LANES=16 lanes in
// parallel; latency = SIMD_DEPTH / HW_LANES ≈ 1 cycle for elementwise,
// 16 cycles for transcendentals, 32 cycles for reductions.
def WriteVectorALU     : SchedWrite;  // VADD, VSUB, VMUL, VDIV, VMAX, VMIN
def WriteVectorFMA     : SchedWrite;  // VFMA
def WriteVectorTrans   : SchedWrite;  // VEXP, VLOG, VRSQRT, VSQRT, VRECIP
def WriteVectorReduce  : SchedWrite;  // VREDSUM, VREDMAX, VREDMIN, VREDAND, VREDOR
def WriteVectorMem     : SchedWrite;  // VLOAD, VSTORE, VGATHER, VSCATTER
def WriteVectorMisc    : SchedWrite;  // VBCAST, VMOV, VSEL, VCMP, VCVT, VSETVL
def WriteVectorMask    : SchedWrite;  // VMASK.SET, VMASK.AND, VMASK.OR, VMASK.NOT

// Corresponding read latencies (all use the default: 0 cycles after write).
def ReadALU    : SchedRead;
def ReadLoad   : SchedRead;
def ReadStore  : SchedRead;
def ReadBranch : SchedRead;
def ReadVector : SchedRead;
def ReadTensor : SchedRead;

// ── Resource definitions and WriteRes mappings ────────────────────────────────
// ProcResource and WriteRes DO have a SchedModel field, so they belong inside
// the "let SchedModel = NPUSchedModel in { }" block.

let SchedModel = NPUSchedModel in {

// The control core has one integer ALU pipeline.
def NPU_ALU : ProcResource<1>;

// The tensor engine and vector engine are separate functional units.
// They share the same instruction issue slot but model their own latencies.
def NPU_TensorEngine : ProcResource<1>;
def NPU_VectorEngine : ProcResource<1>;

// ── Instruction mappings ──────────────────────────────────────────────────────

// Control-core ALU (1 cycle latency, uses the ALU resource).
def : WriteRes<WriteALU,  [NPU_ALU]> { let Latency = 1; }
def : WriteRes<WriteALUI, [NPU_ALU]> { let Latency = 1; }

// Loads / stores (4 cycles for SRAM hit; DRAM is 100× more but we don't know
// at compile time — software must stage data through the scratchpad explicitly).
def : WriteRes<WriteLoad,  [NPU_ALU]> { let Latency = 4; }
def : WriteRes<WriteStore, [NPU_ALU]> { let Latency = 1; }

// Branches, calls, returns.
def : WriteRes<WriteBranch, [NPU_ALU]> { let Latency = 1; }
def : WriteRes<WriteCall,   [NPU_ALU]> { let Latency = 1; }

// Tensor engine.
def : WriteRes<WriteTensorMatmul, [NPU_TensorEngine]> { let Latency = 64; }
def : WriteRes<WriteTensorLoad,   [NPU_TensorEngine]> { let Latency = 8;  }
def : WriteRes<WriteTensorStore,  [NPU_TensorEngine]> { let Latency = 8;  }
def : WriteRes<WriteTensorDMA,    [NPU_TensorEngine]> { let Latency = 32; }
def : WriteRes<WriteTensorAct,    [NPU_TensorEngine]> { let Latency = 4;  }
def : WriteRes<WriteTensorCfg,    [NPU_TensorEngine]> { let Latency = 1;  }

// Vector engine.
def : WriteRes<WriteVectorALU,    [NPU_VectorEngine]> { let Latency = 1;  }
def : WriteRes<WriteVectorFMA,    [NPU_VectorEngine]> { let Latency = 2;  }
def : WriteRes<WriteVectorTrans,  [NPU_VectorEngine]> { let Latency = 16; }
def : WriteRes<WriteVectorReduce, [NPU_VectorEngine]> { let Latency = 32; }
def : WriteRes<WriteVectorMem,    [NPU_VectorEngine]> { let Latency = 8;  }
def : WriteRes<WriteVectorMisc,   [NPU_VectorEngine]> { let Latency = 1;  }
def : WriteRes<WriteVectorMask,   [NPU_VectorEngine]> { let Latency = 1;  }

} // SchedModel = NPUSchedModel
