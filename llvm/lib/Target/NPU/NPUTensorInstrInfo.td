//===- NPUTensorInstrInfo.td - NPU tensor-engine instructions ---*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Instruction definitions for Layer 2: the NPU Tensor Engine (Sections 4–8 of
// fully_prog_npu.md / npu.h cop_t enum entries starting at T_CFG_SHAPE).
//
// The tensor engine operates on "tile registers" (t0..t7, represented in LLVM
// IR as i64 SSA values holding the register number).  It shares the control
// stream (cinst[]) with the scalar control core — these are cinst instructions
// dispatched to the tensor engine's execution unit rather than the scalar ALU.
//
// Assembly syntax mirrors the simulator's CTAB table (npu_asm.c), so emitted
// text assembly can be verified against or parsed by npu_asm_control().
//
// Scheduling: all tensor-engine instructions use NPUSchedModel with latencies
// defined in NPUSchedule.td.
//
//===----------------------------------------------------------------------===//

// ── Tile register operand ─────────────────────────────────────────────────────
// Tile registers t0..t7 are encoded as a 3-bit field in the instruction.
// In LLVM IR they are i64 values; the printer formats them as "t<n>".

def TileImm : Operand<i64> {
  let PrintMethod = "printTileOperand";
}

// ── Instruction format helpers ────────────────────────────────────────────────

// No explicit operands (FENCE, WAIT, BARRIER, etc.).
class TensorNoOp<string mn>
    : NPUInst<(outs), (ins), mn, []>;

// Single tile destination (e.g. ACC.CLR t0).
class TensorTile1<string mn>
    : NPUInst<(outs), (ins TileImm:$dt), mn # " $dt", []>;

// Three tile operands: dest, a, b (MATMUL dt, a, b).
class TensorTile3<string mn>
    : NPUInst<(outs), (ins TileImm:$dt, TileImm:$a, TileImm:$b),
              mn # " $dt, $a, $b", []>;

// Two tile operands: dest, src (RELU, SCALE).
class TensorTile2<string mn>
    : NPUInst<(outs), (ins TileImm:$dt, TileImm:$a),
              mn # " $dt, $a", []>;

// LOAD/STORE: tile register + memory address + count.
// Format: "LOAD t0, count(xN)" — base register + count encoded in cinst fields.
class TensorLoad<string mn>
    : NPUInst<(outs), (ins TileImm:$dt, GPR:$base, i64imm:$count),
              mn # " $dt, ${count}(${base})", []>;

class TensorStore<string mn>
    : NPUInst<(outs), (ins GPR:$base, TileImm:$dt, i64imm:$count),
              mn # " $dt, ${count}(${base})", []>;

// DMA: "DMA.IN dst(xN), src(xM), bytes"
class TensorDMA<string mn>
    : NPUInst<(outs), (ins GPR:$dst, GPR:$src, i64imm:$count),
              mn # " $dst, $src, $count", []>;

// ── Section 4 : Config instructions ──────────────────────────────────────────

// Configure tile shape before any LOAD / STORE / MATMUL sequence.
// "CFG.SHAPE M, K, N" — all three dimensions as immediates.
def T_CFG_SHAPE : NPUInst<(outs), (ins i64imm:$M, i64imm:$K, i64imm:$N),
    "CFG.SHAPE $M, $K, $N", []> {
  let hasSideEffects = 1;
  let SchedRW = [WriteTensorCfg];
}

// ── Section 5 : Compute instructions ─────────────────────────────────────────

// Clear accumulator tile (dt = 0).  Used before starting a K-block loop.
def T_ACC_CLR : TensorTile1<"ACC.CLR"> {
  let hasSideEffects = 1;
  let SchedRW = [WriteTensorCfg];
}

// Matrix multiply-accumulate: dt += a * b.
// K-block loops chain MMA instructions with an SSA phi on the accumulator.
def T_MMA : TensorTile3<"MMA"> {
  let hasSideEffects = 1;
  let SchedRW = [WriteTensorMatmul];
}

// Matrix multiply (overwrite): dt = a * b.  No accumulation.
def T_MATMUL : TensorTile3<"MATMUL"> {
  let hasSideEffects = 1;
  let SchedRW = [WriteTensorMatmul];
}

// Dot product accumulate.
def T_DOT : TensorTile3<"DOT"> {
  let hasSideEffects = 1;
  let SchedRW = [WriteTensorMatmul];
}

// ── Section 6 : Data movement ─────────────────────────────────────────────────

// Load activations tile from DRAM/SRAM.
def T_LOAD : TensorLoad<"LOAD"> {
  let mayLoad = 1;
  let SchedRW = [WriteTensorLoad];
}

// Load weight tile (stored transposed for efficient inner-product access).
def T_LOADW : TensorLoad<"LOADW"> {
  let mayLoad = 1;
  let SchedRW = [WriteTensorLoad];
}

// Store tile to DRAM/SRAM.
def T_STORE : TensorStore<"STORE"> {
  let mayStore = 1;
  let SchedRW = [WriteTensorStore];
}

// DMA: async copy from DRAM to on-chip SRAM scratchpad.
// This is non-blocking; the control core can issue more cinsts while the DMA
// runs.  Use T_FENCE or T_WAIT to ensure completion before reading the data.
def T_DMA_IN : TensorDMA<"DMA.IN"> {
  let hasSideEffects = 1;
  let SchedRW = [WriteTensorDMA];
}

// DMA: async copy from SRAM scratchpad back to DRAM.
def T_DMA_OUT : TensorDMA<"DMA.OUT"> {
  let hasSideEffects = 1;
  let SchedRW = [WriteTensorDMA];
}

// ── Section 7 : Activation functions ─────────────────────────────────────────

// ReLU element-wise on a tile (max(0, x)).
def T_RELU : TensorTile2<"RELU"> {
  let hasSideEffects = 1;
  let SchedRW = [WriteTensorAct];
}

// Scale all elements of a tile by an immediate float factor.
def T_SCALE : NPUInst<(outs), (ins TileImm:$dt, TileImm:$a, f64imm:$factor),
    "SCALE $dt, $a, $factor", []> {
  let hasSideEffects = 1;
  let SchedRW = [WriteTensorAct];
}

// Broadcast a scalar from a GPR to every element of a tile.
// "BCAST dt, xS" — fills tile[dt] with float bits in integer register xS.
// Unlike T_SCALE, the value comes from a GPR (I->rs1 in npu.c), NOT an fimm.
def T_BCAST : NPUInst<(outs), (ins TileImm:$dt, GPR:$rs),
    "BCAST $dt, $rs", []> {
  let hasSideEffects = 1;
  let SchedRW = [WriteTensorAct];
}

// ── Section 8 : Synchronization ───────────────────────────────────────────────

// Memory fence: flush all pending tensor-engine operations.
def T_FENCE : TensorNoOp<"FENCE"> {
  let hasSideEffects = 1;
  let SchedRW = [WriteTensorCfg];
}

// Wait until the tensor engine is fully idle.
def T_WAIT : TensorNoOp<"WAIT"> {
  let hasSideEffects = 1;
  let SchedRW = [WriteTensorCfg];
}

// Global barrier: stall until all cores reach this point.
// Used to synchronize between KLAUNCH phases.
def T_BARRIER : TensorNoOp<"BARRIER"> {
  let hasSideEffects = 1;
  let SchedRW = [WriteTensorCfg];
}

// ── Intrinsic → instruction patterns ─────────────────────────────────────────
// These match @llvm.npu.* calls in the SelectionDAG and select the
// corresponding cinst.  The tile register number is encoded in the i64 operand.
//
// Most tile-returning intrinsics (acc_clr, mma, matmul, dot, load, loadw,
// relu, scale, bcast) are handled in LowerIntrinsicCall + NPUISelDAGToDAG
// because they need to emit a side-effecting instruction AND return a value.
// Only void intrinsics that need no return value have simple Pat<> defs here.

// Synchronization intrinsics: no operands, no return value — perfect for Pat<>.
def : Pat<(int_npu_fence),   (T_FENCE)>;
def : Pat<(int_npu_wait),    (T_WAIT)>;
def : Pat<(int_npu_barrier), (T_BARRIER)>;

// CFG.SHAPE: LowerIntrinsicCall emits NPUISD::CFG_SHAPE which NPUISelDAGToDAG
// selects to T_CFG_SHAPE.  No separate Pat<> is needed here; the NPUISD node
// is used so that runtime (non-constant) M/K/N could be supported later.
