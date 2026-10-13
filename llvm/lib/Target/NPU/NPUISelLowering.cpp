//===-- NPUISelLowering.cpp - NPU DAG Lowering Implementation -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "NPUISelLowering.h"
#include "NPUSelectionDAGInfo.h"
#include "NPUSubtarget.h"
#include "llvm/CodeGen/CallingConvLower.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/SelectionDAG.h"
#include "llvm/IR/IntrinsicsNPU.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/Support/ErrorHandling.h"

using namespace llvm;

#define DEBUG_TYPE "npu-lower"

#include "NPUGenCallingConv.inc"

NPUTargetLowering::NPUTargetLowering(const TargetMachine &TM,
                                     const NPUSubtarget &STI)
    : TargetLowering(TM, STI) {
  // ── Control core ─────────────────────────────────────────────────────────
  addRegisterClass(MVT::i64, &NPU::GPRRegClass);

  // ── Vector core register classes ──────────────────────────────────────────
  // Must be registered before computeRegisterProperties() so the legalizer
  // knows which register class holds each MVT.
  addRegisterClass(MVT::v16f32, &NPU::VRegRegClass); // v0..v31, 16-wide floats
  addRegisterClass(MVT::f32,    &NPU::SRegRegClass);  // s0..s31, scalar floats

  computeRegisterProperties(STI.getRegisterInfo());

  setStackPointerRegisterToSaveRestore(NPU::X2);

  // ── Control-flow ──────────────────────────────────────────────────────────
  // Keep BR_CC for direct selection in NPUDAGToDAGISel; expand everything else.
  setOperationAction(ISD::BR_CC,   MVT::i64,  Legal);
  setOperationAction(ISD::BRCOND,  MVT::Other, Expand);
  setOperationAction(ISD::BR_JT,   MVT::Other, Expand);
  setOperationAction(ISD::SELECT,  MVT::i64,  Expand);
  setOperationAction(ISD::SELECT_CC, MVT::i64, Expand);

  // ── Integer arithmetic ────────────────────────────────────────────────────
  // The control core has no divide/remainder or wide-multiply instructions.
  setOperationAction(ISD::SDIV,     MVT::i64, Expand);
  setOperationAction(ISD::UDIV,     MVT::i64, Expand);
  setOperationAction(ISD::SREM,     MVT::i64, Expand);
  setOperationAction(ISD::UREM,     MVT::i64, Expand);
  setOperationAction(ISD::SDIVREM,  MVT::i64, Expand);
  setOperationAction(ISD::UDIVREM,  MVT::i64, Expand);
  setOperationAction(ISD::MULHU,    MVT::i64, Expand);
  setOperationAction(ISD::MULHS,    MVT::i64, Expand);
  setOperationAction(ISD::UMUL_LOHI,MVT::i64, Expand);
  setOperationAction(ISD::SMUL_LOHI,MVT::i64, Expand);
  setOperationAction(ISD::ROTR,     MVT::i64, Expand);
  setOperationAction(ISD::ROTL,     MVT::i64, Expand);
  setOperationAction(ISD::BSWAP,    MVT::i64, Expand);

  // ── Vector arithmetic (<16 x float>) ─────────────────────────────────────
  // All elementwise float ops are Legal — TableGen VALURR patterns in
  // NPUVectorInstrInfo.td select them directly to V_VADD / V_VMUL / etc.
  setOperationAction(ISD::FADD,    MVT::v16f32, Legal);
  setOperationAction(ISD::FSUB,    MVT::v16f32, Legal);
  setOperationAction(ISD::FMUL,    MVT::v16f32, Legal);
  setOperationAction(ISD::FDIV,    MVT::v16f32, Legal);
  setOperationAction(ISD::FMAXNUM, MVT::v16f32, Legal);
  setOperationAction(ISD::FMINNUM, MVT::v16f32, Legal);
  setOperationAction(ISD::FMA,     MVT::v16f32, Legal);
  setOperationAction(ISD::FABS,    MVT::v16f32, Legal);
  setOperationAction(ISD::FNEG,    MVT::v16f32, Legal);

  // ── Scalar float (f32) ────────────────────────────────────────────────────
  // f32 values are produced only by reduction intrinsics (VREDSUM etc.).
  // The control core has no f32 ALU; expand all f32 arithmetic so the
  // legalizer does not try to soften them.
  setOperationAction(ISD::FADD,   MVT::f32, Expand);
  setOperationAction(ISD::FSUB,   MVT::f32, Expand);
  setOperationAction(ISD::FMUL,   MVT::f32, Expand);
  setOperationAction(ISD::FDIV,   MVT::f32, Expand);
  setOperationAction(ISD::FNEG,   MVT::f32, Expand);
  setOperationAction(ISD::FABS,   MVT::f32, Expand);
  setOperationAction(ISD::FSQRT,  MVT::f32, Expand);
  // f32 load/store are needed if a reduction result is spilled to the stack.
  setOperationAction(ISD::LOAD,   MVT::f32, Expand);
  setOperationAction(ISD::STORE,  MVT::f32, Expand);

  // ── Intrinsics ────────────────────────────────────────────────────────────
  // Route @llvm.npu.* intrinsic calls to LowerIntrinsicCall.
  // IMPORTANT: The legalizer looks up INTRINSIC_W_CHAIN's action by the FIRST
  // value result type (not the chain type MVT::Other).  We must register
  // Custom for every return type our intrinsics produce:
  //   i64     — tile tokens, vsetvl result, acc_clr/mma/matmul/dot/load/loadw
  //   v16f32  — vload, vgather
  //   f32     — vredsum/vredmax/vredmin/vredand/vredor (handled by TableGen Pat<>
  //             but routing through Custom won't hurt)
  // INTRINSIC_VOID uses MVT::Other (the chain) as its key — that is correct.
  setOperationAction(ISD::INTRINSIC_VOID,    MVT::Other,  Custom);
  setOperationAction(ISD::INTRINSIC_W_CHAIN, MVT::Other,  Custom); // fallback
  setOperationAction(ISD::INTRINSIC_W_CHAIN, MVT::i64,    Custom); // tile tokens, vsetvl
  setOperationAction(ISD::INTRINSIC_W_CHAIN, MVT::v16f32, Custom); // vload, vgather
  setOperationAction(ISD::INTRINSIC_W_CHAIN, MVT::f32,    Custom); // future use

  // Booleans always contain 0 or 1.
  setBooleanContents(ZeroOrOneBooleanContent);

  setMinFunctionAlignment(Align(4));
}

// ── LowerOperation ────────────────────────────────────────────────────────────

SDValue NPUTargetLowering::LowerOperation(SDValue Op, SelectionDAG &DAG) const {
  switch (Op.getOpcode()) {
  // Route all @llvm.npu.* intrinsic calls to the dedicated handler.
  case ISD::INTRINSIC_VOID:
  case ISD::INTRINSIC_W_CHAIN:
    return LowerIntrinsicCall(Op, DAG);
  default:
    llvm_unreachable("unimplemented custom operation lowering");
  }
}

// ── LowerIntrinsicCall ────────────────────────────────────────────────────────
//
// Converts @llvm.npu.* intrinsic DAG nodes into NPUISD nodes that
// NPUISelDAGToDAG::Select() then maps to real machine instructions.
//
// Partitioned by engine layer:
//   Layer 1 (control core): klaunch, kwait
//   Layer 2 (tensor engine): cfg_shape, acc_clr, mma, matmul, dot,
//                             load, loadw, store, dma_in, dma_out,
//                             bcast, relu, scale
//   Layer 3 (vector core):  vsetvl, vload, vstore, vgather, vscatter
//
// Note: fence / wait / barrier are matched directly by TableGen patterns in
// NPUTensorInstrInfo.td and never reach here.

// Helper: extract a tile register number (0–7) from an intrinsic operand and
// return it as a TargetConstant (i64) suitable for use in machine nodes.
//
// Tile numbers arrive in several forms depending on where they come from:
//  1. ConstantSDNode  — an IR literal like `i64 0`
//  2. ISD::TargetConstant — already converted by a prior getTileConst call
//  3. ISD::MERGE_VALUES ref — when the tile token is the i64 result of a
//     previous tile intrinsic (e.g. %t0 = @llvm.npu.acc.clr(i64 0)):
//       getMergeValues({DT:i64, chain}) creates a MERGE_VALUES node;
//       %t0 maps to SDValue(mergeNode, 0); we look through to operand[0]=DT.
static SDValue getTileConst(SDValue Op, unsigned Idx, SDLoc &DL,
                            SelectionDAG &DAG) {
  SDValue V = Op.getOperand(Idx);

  // Peel through MERGE_VALUES — tile tokens returned by prior tile intrinsics
  // are wrapped in a MergeValues(tile_token, chain) node after legalization.
  while (V.getOpcode() == ISD::MERGE_VALUES)
    V = V.getOperand(V.getResNo());

  // Handle un-legalized INTRINSIC_W_CHAIN tile ops.
  //
  // All tile-returning intrinsics are defined so that their return value IS the
  // same tile register number as their destination/accumulator argument:
  //   acc_clr(dt)      → dt      (operand 2 = dt)
  //   mma(acc, A, B)   → acc     (operand 2 = acc)
  //   matmul(dt, A, B) → dt      (operand 2 = dt)
  //   dot(dt, acc, A, B) → dt    (operand 2 = dt)
  //   relu(tile)       → tile    (operand 2 = tile)
  //   scale(tile, s)   → tile    (operand 2 = tile)
  //   bcast(dt, s)     → dt      (operand 2 = dt)
  //   load/loadw(dt,…) → dt      (operand 2 = dt)
  //
  // When the legalizer visits the INTRINSIC_W_CHAIN for mma before its acc
  // operand (acc_clr) has been replaced in the DAG, we recurse into the
  // producing intrinsic's own destination argument to find the constant.
  if (V.getOpcode() == ISD::INTRINSIC_W_CHAIN)
    return getTileConst(V, 2, DL, DAG);

  // Regular IR constant literal — return as ISD::Constant (not TargetConstant).
  // ISD::Constant can be materialized into a GPR when the tile token is used
  // as an SSA value (e.g., as a function return value or argument to the next
  // intrinsic).  TargetConstant is immediate-only and cannot be placed in a
  // virtual register.  The conversion to TargetConstant happens later in
  // NPUISelDAGToDAG when the NPUISD machine node is being emitted.
  if (const ConstantSDNode *CN = dyn_cast<ConstantSDNode>(V))
    return DAG.getConstant(CN->getZExtValue(), DL, MVT::i64);

  // Already converted from TargetConstant — normalise back to regular Constant.
  if (V.getOpcode() == ISD::TargetConstant)
    return DAG.getConstant(
        cast<ConstantSDNode>(V)->getZExtValue(), DL, MVT::i64);

  // Dump the problematic node to stderr for debugging, then abort.
  V.dump();
  report_fatal_error(Twine("NPU tensor-engine: tile operand #") + Twine(Idx) +
      " is not constant — opcode=" + Twine(V.getOpcode()) +
      " resno=" + Twine(V.getResNo()));
}

SDValue NPUTargetLowering::LowerIntrinsicCall(SDValue Op,
                                              SelectionDAG &DAG) const {
  SDLoc DL(Op);
  // Both INTRINSIC_VOID and INTRINSIC_W_CHAIN carry the intrinsic ID in Op 1.
  unsigned IntNo = cast<ConstantSDNode>(Op.getOperand(1))->getZExtValue();
  SDValue Chain = Op.getOperand(0);

  switch (IntNo) {

  // ── Layer 1 : Control core ───────────────────────────────────────────────

  // @llvm.npu.klaunch(kernel_id:i32, grid_size:i32)
  // Emits: C_KLAUNCH xN, grid_size_imm
  // kernel_id may be runtime-computed; grid_size must be a constant.
  case Intrinsic::npu_klaunch: {
    SDValue KernelID = Op.getOperand(2); // i32 in a GPR after promotion
    SDValue GridSize = Op.getOperand(3);
    SDValue Ops[] = {Chain, KernelID, GridSize};
    return DAG.getNode(NPUISD::KLAUNCH, DL, MVT::Other, Ops);
  }

  // @llvm.npu.kwait()
  // Emits: C_KWAIT
  case Intrinsic::npu_kwait:
    return DAG.getNode(NPUISD::KWAIT, DL, MVT::Other, Chain);

  // ── Layer 2 : Tensor Engine ───────────────────────────────────────────────

  // @llvm.npu.cfg_shape(M:i64, K:i64, N:i64)
  // Emits: T_CFG_SHAPE M, K, N  (all immediates)
  case Intrinsic::npu_cfg_shape: {
    SDValue M = Op.getOperand(2);
    SDValue K = Op.getOperand(3);
    SDValue N = Op.getOperand(4);
    SDValue Ops[] = {Chain, M, K, N};
    return DAG.getNode(NPUISD::CFG_SHAPE, DL, MVT::Other, Ops);
  }

  // @llvm.npu.acc_clr(dt:i64) → i64
  // Emits: T_ACC_CLR dt.  Returns dt as the tile token for data-flow tracking.
  case Intrinsic::npu_acc_clr: {
    SDValue DT = getTileConst(Op, 2, DL, DAG);
    SDValue NewChain = DAG.getNode(NPUISD::ACC_CLR, DL, MVT::Other, Chain, DT);
    // The SSA result is just the tile number — the actual data lives in HW.
    return DAG.getMergeValues({DT, NewChain}, DL);
  }

  // @llvm.npu.mma(acc:i64, A:i64, B:i64) → i64
  // Emits: T_MMA dt=acc, A, B.  Returns acc (same tile; accumulation in-place).
  case Intrinsic::npu_mma: {
    SDValue Acc = getTileConst(Op, 2, DL, DAG);
    SDValue A   = getTileConst(Op, 3, DL, DAG);
    SDValue B   = getTileConst(Op, 4, DL, DAG);
    SDValue NewChain = DAG.getNode(NPUISD::MMA, DL, MVT::Other,
                                   Chain, Acc, A, B);
    return DAG.getMergeValues({Acc, NewChain}, DL);
  }

  // @llvm.npu.matmul(dt:i64, A:i64, B:i64) → i64
  // Emits: T_MATMUL dt, A, B.  Returns dt.
  case Intrinsic::npu_matmul: {
    SDValue DT = getTileConst(Op, 2, DL, DAG);
    SDValue A  = getTileConst(Op, 3, DL, DAG);
    SDValue B  = getTileConst(Op, 4, DL, DAG);
    SDValue NewChain = DAG.getNode(NPUISD::MATMUL, DL, MVT::Other,
                                   Chain, DT, A, B);
    return DAG.getMergeValues({DT, NewChain}, DL);
  }

  // @llvm.npu.dot(dt:i64, A:i64, B:i64) → i64
  // CF_T3 format (same as MMA/MATMUL): dt is the destination AND accumulator.
  // Emits: T_DOT dt, A, B.  Returns dt.
  case Intrinsic::npu_dot: {
    SDValue DT = getTileConst(Op, 2, DL, DAG);
    SDValue A  = getTileConst(Op, 3, DL, DAG);
    SDValue B  = getTileConst(Op, 4, DL, DAG);
    SDValue NewChain = DAG.getNode(NPUISD::DOT, DL, MVT::Other,
                                   Chain, DT, A, B);
    return DAG.getMergeValues({DT, NewChain}, DL);
  }

  // @llvm.npu.load(dt:i64, ptr:i64, count:i64) → i64
  // Emits: T_LOAD dt, count(xBase).  Returns dt.
  case Intrinsic::npu_load: {
    SDValue DT    = getTileConst(Op, 2, DL, DAG);
    SDValue Ptr   = Op.getOperand(3); // i64 address
    SDValue Count = Op.getOperand(4); // i64 element count
    SDValue NewChain = DAG.getNode(NPUISD::TELOAD, DL, MVT::Other,
                                   Chain, DT, Ptr, Count);
    return DAG.getMergeValues({DT, NewChain}, DL);
  }

  // @llvm.npu.loadw(dt:i64, ptr:i64, count:i64) → i64  (transposed weights)
  case Intrinsic::npu_loadw: {
    SDValue DT    = getTileConst(Op, 2, DL, DAG);
    SDValue Ptr   = Op.getOperand(3);
    SDValue Count = Op.getOperand(4);
    SDValue NewChain = DAG.getNode(NPUISD::TELOADW, DL, MVT::Other,
                                   Chain, DT, Ptr, Count);
    return DAG.getMergeValues({DT, NewChain}, DL);
  }

  // @llvm.npu.store(ptr:i64, tile:i64, count:i64)
  // Emits: T_STORE tile, count(xBase).
  case Intrinsic::npu_store: {
    SDValue Ptr   = Op.getOperand(2);
    SDValue Tile  = getTileConst(Op, 3, DL, DAG);
    SDValue Count = Op.getOperand(4);
    SDValue Ops[] = {Chain, Ptr, Tile, Count};
    return DAG.getNode(NPUISD::TESTORE, DL, MVT::Other, Ops);
  }

  // @llvm.npu.dma_in(dst:i64, src:i64, bytes:i64)
  // Emits: T_DMA_IN dst, src, bytes  — async DRAM→SRAM.
  case Intrinsic::npu_dma_in: {
    SDValue Dst   = Op.getOperand(2);
    SDValue Src   = Op.getOperand(3);
    SDValue Bytes = Op.getOperand(4);
    SDValue Ops[] = {Chain, Dst, Src, Bytes};
    return DAG.getNode(NPUISD::DMA_IN, DL, MVT::Other, Ops);
  }

  // @llvm.npu.dma_out(dst:i64, src:i64, bytes:i64)
  // Emits: T_DMA_OUT dst, src, bytes  — async SRAM→DRAM.
  case Intrinsic::npu_dma_out: {
    SDValue Dst   = Op.getOperand(2);
    SDValue Src   = Op.getOperand(3);
    SDValue Bytes = Op.getOperand(4);
    SDValue Ops[] = {Chain, Dst, Src, Bytes};
    return DAG.getNode(NPUISD::DMA_OUT, DL, MVT::Other, Ops);
  }

  // @llvm.npu.relu(tile:i64) → i64
  // Emits: T_RELU dt=tile, a=tile.  Returns tile (in-place).
  case Intrinsic::npu_relu: {
    SDValue Tile = getTileConst(Op, 2, DL, DAG);
    SDValue NewChain = DAG.getNode(NPUISD::RELU, DL, MVT::Other, Chain, Tile);
    return DAG.getMergeValues({Tile, NewChain}, DL);
  }

  // @llvm.npu.scale(tile:i64, factor:float) → i64
  // Emits: T_SCALE dt=tile, a=tile, factor.  Returns tile (in-place).
  //
  // The scale factor is encoded in the instruction binary (bytes 16–23 of the
  // 32-byte cinst word, same as the imm/fimm field).  We must convert the f32
  // operand to a TargetConstantFP(f64) immediately here, bypassing the f32
  // type legalizer.  If we let the ConstantFP:f32 node live as a regular
  // operand, the legalizer tries to materialize it into an SReg and then load
  // it, which fails because SRegs hold reduction results, not general FP values.
  case Intrinsic::npu_scale: {
    SDValue Tile = getTileConst(Op, 2, DL, DAG);
    const ConstantFPSDNode *FP = dyn_cast<ConstantFPSDNode>(Op.getOperand(3));
    if (!FP)
      report_fatal_error("NPU scale: factor must be a compile-time constant float");
    // Encode the float factor as a TargetConstant:i64 containing the
    // IEEE 754 double-precision bit pattern.  We use i64 (not TargetConstantFP)
    // because f64 is not a legal register type on the NPU control core, and
    // TargetConstantFP with an illegal type triggers the legalizer assertion.
    // NPUISelDAGToDAG::Select converts the bits back to TargetConstantFP:f64
    // (which is fine there — machine-node operands skip type legalization).
    APFloat fval = FP->getValueAPF();
    bool lossy;
    fval.convert(APFloat::IEEEdouble(), APFloat::rmNearestTiesToEven, &lossy);
    SDValue FactorBits = DAG.getTargetConstant(
        fval.bitcastToAPInt().getZExtValue(), DL, MVT::i64);
    SDValue NewChain = DAG.getNode(NPUISD::SCALE, DL, MVT::Other,
                                   Chain, Tile, FactorBits);
    return DAG.getMergeValues({Tile, NewChain}, DL);
  }

  // @llvm.npu.bcast(dt:i64, val:float) → i64
  // Emits: T_BCAST dt, xS — broadcasts float bits from GPR xS to all tile elements.
  // Unlike T_SCALE, the simulator reads the value from I->rs1 (an integer register),
  // NOT an fimm.  Encode the f32 bit pattern as an ISD::Constant:i64 so it gets
  // materialized into a GPR via LI before the T_BCAST instruction fires.
  case Intrinsic::npu_bcast: {
    SDValue DT = getTileConst(Op, 2, DL, DAG);
    const ConstantFPSDNode *FP = dyn_cast<ConstantFPSDNode>(Op.getOperand(3));
    if (!FP)
      report_fatal_error("NPU bcast: broadcast value must be a compile-time constant float");
    // Store the f32 bit pattern in the low 32 bits of an i64 GPR (zero-extended).
    uint32_t Bits32 = FP->getValueAPF().bitcastToAPInt().getZExtValue();
    SDValue ValI64 = DAG.getConstant(Bits32, DL, MVT::i64);
    SDValue NewChain = DAG.getNode(NPUISD::BCAST, DL, MVT::Other,
                                   Chain, DT, ValI64);
    return DAG.getMergeValues({DT, NewChain}, DL);
  }

  // ── Layer 3 : Vector Core ─────────────────────────────────────────────────

  // @llvm.npu.vsetvl(length:i64) → i64
  // Emits: V_VSETVL xD, xS — reads requested VL from GPR xS, writes actual VL to xD.
  // Returns the actual VL as i64 (clamped to VLEN_MAX by the hardware).
  case Intrinsic::npu_vsetvl: {
    SDValue Len = Op.getOperand(2); // requested VL (i64, lives in a GPR)
    SDVTList VTs = DAG.getVTList(MVT::i64, MVT::Other);
    return DAG.getNode(NPUISD::VSETVL, DL, VTs, Chain, Len);
  }

  // @llvm.npu.vload(ptr:i64, stride:i64) → <16 x float>
  // Emits: V_VLOAD vD, stride(xBase).
  case Intrinsic::npu_vload: {
    SDValue Ptr    = Op.getOperand(2);
    SDValue Stride = Op.getOperand(3);
    SDVTList VTs = DAG.getVTList(MVT::v16f32, MVT::Other);
    SDValue Ops[] = {Chain, Ptr, Stride};
    return DAG.getNode(NPUISD::VLOAD, DL, VTs, Ops);
  }

  // @llvm.npu.vstore(ptr:i64, vec:<16 x float>, stride:i64)
  // Emits: V_VSTORE vS, stride(xBase).
  case Intrinsic::npu_vstore: {
    SDValue Ptr    = Op.getOperand(2);
    SDValue Vec    = Op.getOperand(3);
    SDValue Stride = Op.getOperand(4);
    SDValue Ops[] = {Chain, Ptr, Vec, Stride};
    return DAG.getNode(NPUISD::VSTORE, DL, MVT::Other, Ops);
  }

  // @llvm.npu.vgather(ptr:i64, idx:<16 x float>) → <16 x float>
  // Emits: V_VGATHER vD, 0(xBase), vIdx.
  case Intrinsic::npu_vgather: {
    SDValue Ptr = Op.getOperand(2);
    SDValue Idx = Op.getOperand(3);
    SDVTList VTs = DAG.getVTList(MVT::v16f32, MVT::Other);
    SDValue Ops[] = {Chain, Ptr, Idx};
    return DAG.getNode(NPUISD::VGATHER, DL, VTs, Ops);
  }

  // @llvm.npu.vscatter(ptr:i64, idx:<16 x float>, data:<16 x float>)
  // Emits: V_VSCATTER vS, 0(xBase), vIdx.
  case Intrinsic::npu_vscatter: {
    SDValue Ptr  = Op.getOperand(2);
    SDValue Idx  = Op.getOperand(3);
    SDValue Data = Op.getOperand(4);
    SDValue Ops[] = {Chain, Ptr, Idx, Data};
    return DAG.getNode(NPUISD::VSCATTER, DL, MVT::Other, Ops);
  }

  // ── Synchronization (TableGen Pat<> overridden by Custom legalization) ────
  // Setting ISD::INTRINSIC_VOID → Custom causes ALL void intrinsics to go
  // through LowerOperation before the instruction selector can apply
  // TableGen Pat<> matches.  Handle fence/wait/barrier here by emitting
  // machine nodes directly (NPUISelDAGToDAG::Select skips isMachineOpcode nodes).
  case Intrinsic::npu_fence:
    return SDValue(DAG.getMachineNode(NPU::T_FENCE, DL, MVT::Other, Chain), 0);

  case Intrinsic::npu_wait:
    return SDValue(DAG.getMachineNode(NPU::T_WAIT, DL, MVT::Other, Chain), 0);

  case Intrinsic::npu_barrier:
    return SDValue(DAG.getMachineNode(NPU::T_BARRIER, DL, MVT::Other, Chain), 0);

  default:
    report_fatal_error("NPU: unhandled @llvm.npu.* intrinsic — check "
                       "LowerIntrinsicCall or add a case for the new intrinsic");
  }
}

const char *NPUTargetLowering::getTargetNodeName(unsigned Opcode) const {
  switch ((NPUISD::GenNodeType)Opcode) {
  // ── Layer 1 ──────────────────────────────────────────────────────────────
  case NPUISD::CALL:      return "NPUISD::CALL";
  case NPUISD::RET_GLUE:  return "NPUISD::RET_GLUE";
  case NPUISD::KLAUNCH:   return "NPUISD::KLAUNCH";
  case NPUISD::KWAIT:     return "NPUISD::KWAIT";
  // ── Layer 2 : Tensor Engine ───────────────────────────────────────────────
  case NPUISD::CFG_SHAPE: return "NPUISD::CFG_SHAPE";
  case NPUISD::ACC_CLR:   return "NPUISD::ACC_CLR";
  case NPUISD::MMA:       return "NPUISD::MMA";
  case NPUISD::MATMUL:    return "NPUISD::MATMUL";
  case NPUISD::DOT:       return "NPUISD::DOT";
  case NPUISD::TELOAD:    return "NPUISD::TELOAD";
  case NPUISD::TELOADW:   return "NPUISD::TELOADW";
  case NPUISD::TESTORE:   return "NPUISD::TESTORE";
  case NPUISD::DMA_IN:    return "NPUISD::DMA_IN";
  case NPUISD::DMA_OUT:   return "NPUISD::DMA_OUT";
  case NPUISD::RELU:      return "NPUISD::RELU";
  case NPUISD::SCALE:     return "NPUISD::SCALE";
  case NPUISD::BCAST:     return "NPUISD::BCAST";
  // ── Layer 3 : Vector Core ─────────────────────────────────────────────────
  case NPUISD::VSETVL:    return "NPUISD::VSETVL";
  case NPUISD::VLOAD:     return "NPUISD::VLOAD";
  case NPUISD::VSTORE:    return "NPUISD::VSTORE";
  case NPUISD::VGATHER:   return "NPUISD::VGATHER";
  case NPUISD::VSCATTER:  return "NPUISD::VSCATTER";
  default:                return nullptr;
  }
}

//===----------------------------------------------------------------------===//
// Calling convention implementation
//===----------------------------------------------------------------------===//

SDValue NPUTargetLowering::LowerFormalArguments(
    SDValue Chain, CallingConv::ID CallConv, bool IsVarArg,
    const SmallVectorImpl<ISD::InputArg> &Ins, const SDLoc &DL,
    SelectionDAG &DAG, SmallVectorImpl<SDValue> &InVals) const {
  if (IsVarArg)
    report_fatal_error("NPU: varargs are not supported");

  MachineFunction &MF = DAG.getMachineFunction();
  MachineFrameInfo &MFI = MF.getFrameInfo();
  MachineRegisterInfo &RegInfo = MF.getRegInfo();

  SmallVector<CCValAssign, 16> ArgLocs;
  CCState CCInfo(CallConv, IsVarArg, MF, ArgLocs, *DAG.getContext());
  CCInfo.AnalyzeFormalArguments(Ins, CC_NPU);

  for (const CCValAssign &VA : ArgLocs) {
    if (VA.isRegLoc()) {
      // Pick the register class that matches the argument's machine value type.
      // getRegClassFor() maps: i64 → GPRRegClass, v16f32 → VRegRegClass,
      // f32 → SRegRegClass (registered in the constructor above).
      const TargetRegisterClass *RC = getRegClassFor(VA.getLocVT());
      Register VReg = RegInfo.createVirtualRegister(RC);
      RegInfo.addLiveIn(VA.getLocReg(), VReg);
      SDValue ArgValue = DAG.getCopyFromReg(Chain, DL, VReg, VA.getLocVT());

      if (VA.getLocInfo() == CCValAssign::SExt)
        ArgValue = DAG.getNode(ISD::AssertSext, DL, VA.getLocVT(), ArgValue,
                               DAG.getValueType(VA.getValVT()));
      else if (VA.getLocInfo() == CCValAssign::ZExt)
        ArgValue = DAG.getNode(ISD::AssertZext, DL, VA.getLocVT(), ArgValue,
                               DAG.getValueType(VA.getValVT()));

      if (VA.getLocInfo() != CCValAssign::Full)
        ArgValue = DAG.getNode(ISD::TRUNCATE, DL, VA.getValVT(), ArgValue);

      InVals.push_back(ArgValue);
    } else {
      assert(VA.isMemLoc());
      unsigned ObjSize = VA.getLocVT().getStoreSize();
      int FI = MFI.CreateFixedObject(ObjSize, VA.getLocMemOffset(), true);
      SDValue FIN = DAG.getFrameIndex(FI, MVT::i64);
      InVals.push_back(DAG.getLoad(
          VA.getLocVT(), DL, Chain, FIN,
          MachinePointerInfo::getFixedStack(MF, FI)));
    }
  }

  return Chain;
}

SDValue
NPUTargetLowering::LowerReturn(SDValue Chain, CallingConv::ID CallConv,
                               bool IsVarArg,
                               const SmallVectorImpl<ISD::OutputArg> &Outs,
                               const SmallVectorImpl<SDValue> &OutVals,
                               const SDLoc &DL, SelectionDAG &DAG) const {
  SmallVector<CCValAssign, 16> RVLocs;
  CCState CCInfo(CallConv, IsVarArg, DAG.getMachineFunction(), RVLocs,
                 *DAG.getContext());
  CCInfo.AnalyzeReturn(Outs, RetCC_NPU);

  SDValue Glue;
  SmallVector<SDValue, 4> RetOps(1, Chain);

  for (unsigned i = 0; i != RVLocs.size(); ++i) {
    CCValAssign &VA = RVLocs[i];
    assert(VA.isRegLoc() && "Can only return in registers!");
    Chain = DAG.getCopyToReg(Chain, DL, VA.getLocReg(), OutVals[i], Glue);
    Glue = Chain.getValue(1);
    RetOps.push_back(DAG.getRegister(VA.getLocReg(), VA.getLocVT()));
  }

  RetOps[0] = Chain;
  if (Glue.getNode())
    RetOps.push_back(Glue);

  return DAG.getNode(NPUISD::RET_GLUE, DL, MVT::Other, RetOps);
}

SDValue NPUTargetLowering::LowerCallResult(
    SDValue Chain, SDValue InGlue, CallingConv::ID CallConv, bool IsVarArg,
    const SmallVectorImpl<ISD::InputArg> &Ins, const SDLoc &DL,
    SelectionDAG &DAG, SmallVectorImpl<SDValue> &InVals) const {
  SmallVector<CCValAssign, 16> RVLocs;
  CCState CCInfo(CallConv, IsVarArg, DAG.getMachineFunction(), RVLocs,
                 *DAG.getContext());
  CCInfo.AnalyzeCallResult(Ins, RetCC_NPU);

  for (unsigned I = 0; I != RVLocs.size(); ++I) {
    Chain = DAG.getCopyFromReg(Chain, DL, RVLocs[I].getLocReg(),
                               RVLocs[I].getValVT(), InGlue)
                .getValue(1);
    InGlue = Chain.getValue(2);
    InVals.push_back(Chain.getValue(0));
  }

  return Chain;
}

SDValue NPUTargetLowering::LowerCall(TargetLowering::CallLoweringInfo &CLI,
                                     SmallVectorImpl<SDValue> &InVals) const {
  SelectionDAG &DAG = CLI.DAG;
  SDLoc &DL = CLI.DL;
  SmallVectorImpl<ISD::OutputArg> &Outs = CLI.Outs;
  SmallVectorImpl<SDValue> &OutVals = CLI.OutVals;
  SmallVectorImpl<ISD::InputArg> &Ins = CLI.Ins;
  SDValue Chain = CLI.Chain;
  SDValue Callee = CLI.Callee;
  CallingConv::ID CallConv = CLI.CallConv;
  bool IsVarArg = CLI.IsVarArg;

  CLI.IsTailCall = false;
  if (IsVarArg)
    report_fatal_error("NPU: varargs calls are not supported");

  MachineFunction &MF = DAG.getMachineFunction();
  SmallVector<CCValAssign, 16> ArgLocs;
  CCState CCInfo(CallConv, IsVarArg, MF, ArgLocs, *DAG.getContext());
  CCInfo.AnalyzeCallOperands(Outs, CC_NPU);

  unsigned NumBytes = CCInfo.getStackSize();
  Chain = DAG.getCALLSEQ_START(Chain, NumBytes, 0, DL);

  SmallVector<std::pair<unsigned, SDValue>, 8> RegsToPass;
  SmallVector<SDValue, 12> MemOpChains;
  SDValue StackPtr;

  for (unsigned I = 0, E = ArgLocs.size(); I != E; ++I) {
    CCValAssign &VA = ArgLocs[I];
    SDValue Arg = OutVals[I];

    switch (VA.getLocInfo()) {
    case CCValAssign::Full:
      break;
    case CCValAssign::SExt:
      Arg = DAG.getNode(ISD::SIGN_EXTEND, DL, VA.getLocVT(), Arg);
      break;
    case CCValAssign::ZExt:
      Arg = DAG.getNode(ISD::ZERO_EXTEND, DL, VA.getLocVT(), Arg);
      break;
    case CCValAssign::AExt:
      Arg = DAG.getNode(ISD::ANY_EXTEND, DL, VA.getLocVT(), Arg);
      break;
    default:
      llvm_unreachable("Unknown loc info!");
    }

    if (VA.isRegLoc()) {
      RegsToPass.push_back(std::make_pair(VA.getLocReg(), Arg));
    } else {
      assert(VA.isMemLoc());
      if (!StackPtr.getNode())
        StackPtr = DAG.getCopyFromReg(Chain, DL, NPU::X2, MVT::i64);
      SDValue PtrOff =
          DAG.getNode(ISD::ADD, DL, MVT::i64, StackPtr,
                      DAG.getIntPtrConstant(VA.getLocMemOffset(), DL));
      MemOpChains.push_back(
          DAG.getStore(Chain, DL, Arg, PtrOff, MachinePointerInfo()));
    }
  }

  if (!MemOpChains.empty())
    Chain = DAG.getNode(ISD::TokenFactor, DL, MVT::Other, MemOpChains);

  SDValue InGlue;
  for (const auto &[Reg, N] : RegsToPass) {
    Chain = DAG.getCopyToReg(Chain, DL, Reg, N, InGlue);
    InGlue = Chain.getValue(1);
  }

  if (GlobalAddressSDNode *G = dyn_cast<GlobalAddressSDNode>(Callee))
    Callee = DAG.getTargetGlobalAddress(G->getGlobal(), DL, MVT::i64, 0, 0);
  else if (ExternalSymbolSDNode *E = dyn_cast<ExternalSymbolSDNode>(Callee))
    Callee = DAG.getTargetExternalSymbol(E->getSymbol(), MVT::i64, 0);

  SDVTList NodeTys = DAG.getVTList(MVT::Other, MVT::Glue);
  SmallVector<SDValue, 8> Ops;
  Ops.push_back(Chain);
  Ops.push_back(Callee);

  const NPURegisterInfo *TRI =
      MF.getSubtarget<NPUSubtarget>().getRegisterInfo();
  const uint32_t *Mask = TRI->getCallPreservedMask(MF, CallConv);
  assert(Mask && "Missing call preserved mask for calling convention");
  Ops.push_back(DAG.getRegisterMask(Mask));

  for (const auto &[Reg, N] : RegsToPass)
    Ops.push_back(DAG.getRegister(Reg, N.getValueType()));

  if (InGlue.getNode())
    Ops.push_back(InGlue);

  Chain = DAG.getNode(NPUISD::CALL, DL, NodeTys, Ops);
  InGlue = Chain.getValue(1);

  Chain = DAG.getCALLSEQ_END(Chain, NumBytes, 0, InGlue, DL);
  InGlue = Chain.getValue(1);

  return LowerCallResult(Chain, InGlue, CallConv, IsVarArg, Ins, DL, DAG,
                         InVals);
}
