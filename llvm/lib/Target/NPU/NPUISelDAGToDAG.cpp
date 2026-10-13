//===-- NPUISelDAGToDAG.cpp - A dag to dag inst selector for NPU ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Custom DAG-to-DAG instruction selector for the NPU backend.  Most
// instruction patterns are matched automatically by TableGen; the Select()
// override handles the cases that need special treatment:
//
//   FrameIndex   — materialise as ADDI xN, x2, offset.
//   ISD::BR_CC   — lower conditional branches:
//                    • Signed conditions (SETEQ/NE/LT/GE/GT/LE):
//                        select BEQ / BNE / BLT / BGE directly.
//                    • Unsigned conditions (SETULT/UGT/ULE/UGE):
//                        the control core has no BLTU/BGEU, so we synthesise
//                        the equivalent signed comparison by flipping the MSB
//                        of both operands via XOR (converting unsigned to
//                        biased signed), then using BLT / BGE.
//   NPUISD::KLAUNCH — lower to the KLAUNCH machine instruction.
//   NPUISD::KWAIT   — lower to the KWAIT machine instruction.
//
//===----------------------------------------------------------------------===//

#include "NPU.h"
#include "NPUISelLowering.h"
#include "NPUTargetMachine.h"
#include "llvm/CodeGen/SelectionDAGISel.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"

using namespace llvm;

#define DEBUG_TYPE "npu-isel"
#define PASS_NAME "NPU DAG->DAG Pattern Instruction Selection"

namespace {

class NPUDAGToDAGISel : public SelectionDAGISel {
public:
  NPUDAGToDAGISel() = delete;
  explicit NPUDAGToDAGISel(NPUTargetMachine &TM) : SelectionDAGISel(TM) {}

private:
#include "NPUGenDAGISel.inc"

  void Select(SDNode *N) override;

  bool selectAddrRI(SDValue Addr, SDValue &Base, SDValue &Offset);
};

class NPUDAGToDAGISelLegacy : public SelectionDAGISelLegacy {
public:
  static char ID;
  explicit NPUDAGToDAGISelLegacy(NPUTargetMachine &TM)
      : SelectionDAGISelLegacy(ID, std::make_unique<NPUDAGToDAGISel>(TM)) {}
};

} // namespace

char NPUDAGToDAGISelLegacy::ID = 0;

INITIALIZE_PASS(NPUDAGToDAGISelLegacy, DEBUG_TYPE, PASS_NAME, false, false)

// Match "base + imm" and bare frame indices for load/store addressing.
bool NPUDAGToDAGISel::selectAddrRI(SDValue Addr, SDValue &Base,
                                   SDValue &Offset) {
  SDLoc DL(Addr);
  EVT VT = Addr.getValueType();

  if (FrameIndexSDNode *FIN = dyn_cast<FrameIndexSDNode>(Addr)) {
    Base = CurDAG->getTargetFrameIndex(FIN->getIndex(), VT);
    Offset = CurDAG->getTargetConstant(0, DL, MVT::i64);
    return true;
  }

  if (Addr.getOpcode() == ISD::ADD) {
    if (ConstantSDNode *CN = dyn_cast<ConstantSDNode>(Addr.getOperand(1))) {
      SDValue N0 = Addr.getOperand(0);
      if (FrameIndexSDNode *FIN = dyn_cast<FrameIndexSDNode>(N0))
        Base = CurDAG->getTargetFrameIndex(FIN->getIndex(), VT);
      else
        Base = N0;
      Offset = CurDAG->getTargetConstant(CN->getSExtValue(), DL, MVT::i64);
      return true;
    }
  }

  Base = Addr;
  Offset = CurDAG->getTargetConstant(0, DL, MVT::i64);
  return true;
}

void NPUDAGToDAGISel::Select(SDNode *Node) {
  if (Node->isMachineOpcode()) {
    Node->setNodeId(-1);
    return;
  }

  SDLoc DL(Node);

  switch (Node->getOpcode()) {
  // ── FrameIndex materialisation ───────────────────────────────────────────
  // A bare FrameIndex node (from alloca / spill) becomes ADDI xN, x2, offset.
  case ISD::FrameIndex: {
    int FI = cast<FrameIndexSDNode>(Node)->getIndex();
    EVT VT = Node->getValueType(0);
    SDValue TFI = CurDAG->getTargetFrameIndex(FI, VT);
    SDValue Zero = CurDAG->getTargetConstant(0, DL, MVT::i64);
    CurDAG->SelectNodeTo(Node, NPU::ADDI, VT, TFI, Zero);
    return;
  }

  // ── Conditional branches ─────────────────────────────────────────────────
  // BR_CC operands: (Chain, CondCode, LHS, RHS, Dest).
  //
  // Signed conditions map directly onto the control core's BEQ/BNE/BLT/BGE.
  // GT and LE are handled by swapping LHS and RHS.
  //
  // Unsigned conditions (ULT, UGT, ULE, UGE) are synthesised:
  //   a <u b  ≡  (a^MSB) <s (b^MSB)
  // We emit:
  //   XOR  tmp_a, LHS, MSB_const    ; flip sign bit of LHS
  //   XOR  tmp_b, RHS, MSB_const    ; flip sign bit of RHS
  //   BLT/BGE tmp_a, tmp_b, Dest    ; signed comparison on biased values
  // This is correct for all unsigned comparisons and requires only the
  // instructions already defined in the control-core ISA.
  case ISD::BR_CC: {
    SDValue Chain = Node->getOperand(0);
    ISD::CondCode CC = cast<CondCodeSDNode>(Node->getOperand(1))->get();
    SDValue LHS = Node->getOperand(2);
    SDValue RHS = Node->getOperand(3);
    SDValue Dest = Node->getOperand(4);

    // Helper: emit "XOR vd, vs, MSB" to convert a value from unsigned to
    // biased-signed (flip bit 63) so that unsigned order ≡ signed order.
    auto flipMSB = [&](SDValue V) -> SDValue {
      SDValue MSB = CurDAG->getTargetConstant(
          (int64_t)1 << 63, DL, MVT::i64);
      // Build an XOR SDNode and immediately select it into an NPU::XOR MI.
      // We use getNode() which may CSE; the final SelectCode pass will select.
      return CurDAG->getNode(ISD::XOR, DL, MVT::i64, V, MSB);
    };

    unsigned Opc;
    bool IsUnsigned = false;

    switch (CC) {
    // ── Signed conditions ──────────────────────────────────────────
    case ISD::SETEQ:
      Opc = NPU::BEQ;
      break;
    case ISD::SETNE:
      Opc = NPU::BNE;
      break;
    case ISD::SETLT:
      Opc = NPU::BLT;
      break;
    case ISD::SETGE:
      Opc = NPU::BGE;
      break;
    case ISD::SETGT: // a >s b  ≡  b <s a
      Opc = NPU::BLT;
      std::swap(LHS, RHS);
      break;
    case ISD::SETLE: // a <=s b  ≡  b >=s a
      Opc = NPU::BGE;
      std::swap(LHS, RHS);
      break;

    // ── Unsigned conditions — synthesise via XOR sign-flip ────────
    case ISD::SETULT: // a <u b  ≡  (a^MSB) <s (b^MSB)
      Opc = NPU::BLT;
      IsUnsigned = true;
      break;
    case ISD::SETUGE: // a >=u b  ≡  (a^MSB) >=s (b^MSB)
      Opc = NPU::BGE;
      IsUnsigned = true;
      break;
    case ISD::SETUGT: // a >u b  ≡  b <u a  ≡  (b^MSB) <s (a^MSB)
      Opc = NPU::BLT;
      std::swap(LHS, RHS);
      IsUnsigned = true;
      break;
    case ISD::SETULE: // a <=u b  ≡  b >=u a  ≡  (b^MSB) >=s (a^MSB)
      Opc = NPU::BGE;
      std::swap(LHS, RHS);
      IsUnsigned = true;
      break;

    default:
      report_fatal_error("NPU: unsupported branch condition code");
    }

    if (IsUnsigned) {
      // Insert the sign-bit flip before branching.  The XOR nodes will be
      // selected in a recursive pass through SelectCode().
      LHS = flipMSB(LHS);
      RHS = flipMSB(RHS);
    }

    SDValue Ops[] = {LHS, RHS, Dest, Chain};
    CurDAG->SelectNodeTo(Node, Opc, MVT::Other, Ops);
    return;
  }

  // ── NPUISD::KLAUNCH ──────────────────────────────────────────────────────
  // (chain, kernel_id_val, grid_size_val) → NPU::KLAUNCH xN, grid_size_imm
  // kernel_id goes in a GPR; grid_size must be a compile-time constant.
  case NPUISD::KLAUNCH: {
    SDValue Chain    = Node->getOperand(0);
    SDValue KernelID = Node->getOperand(1);
    SDValue GridSize = Node->getOperand(2);

    if (!isa<ConstantSDNode>(GridSize))
      report_fatal_error("NPU KLAUNCH: grid_size must be a compile-time constant");

    int64_t Grid = cast<ConstantSDNode>(GridSize)->getSExtValue();
    SDValue GridImm = CurDAG->getTargetConstant(Grid, DL, MVT::i64);

    SDValue Ops[] = {KernelID, GridImm, Chain};
    CurDAG->SelectNodeTo(Node, NPU::KLAUNCH, MVT::Other, Ops);
    return;
  }

  // ── NPUISD::KWAIT ────────────────────────────────────────────────────────
  case NPUISD::KWAIT: {
    SDValue Chain = Node->getOperand(0);
    CurDAG->SelectNodeTo(Node, NPU::KWAIT, MVT::Other, Chain);
    return;
  }

  // ── Tensor Engine ─────────────────────────────────────────────────────────
  //
  // Helper: convert an ISD::Constant (i64) to a TargetConstant suitable for
  // use as a TileImm or i64imm machine instruction operand.  LowerIntrinsicCall
  // produces ISD::Constant so the tile SSA value can be materialized in a GPR
  // when needed (e.g., returned from a function); here we re-canonicalize to
  // TargetConstant for emission as an inline immediate.
  // NOTE: this lambda is defined locally so it captures DL/CurDAG.

  // NPUISD::CFG_SHAPE(chain, M:i64, K:i64, N:i64) → T_CFG_SHAPE M, K, N
  // All three dimensions must be compile-time constants.
  case NPUISD::CFG_SHAPE: {
    SDValue Chain = Node->getOperand(0);
    SDValue M = Node->getOperand(1);
    SDValue K = Node->getOperand(2);
    SDValue N = Node->getOperand(3);

    auto toImm = [&](SDValue V, const char *which) -> SDValue {
      if (!isa<ConstantSDNode>(V))
        report_fatal_error(Twine("NPU CFG_SHAPE: ") + which +
                           " must be a compile-time constant");
      return CurDAG->getTargetConstant(
          cast<ConstantSDNode>(V)->getZExtValue(), DL, MVT::i64);
    };

    SDValue Ops[] = {toImm(M, "M"), toImm(K, "K"), toImm(N, "N"), Chain};
    CurDAG->SelectNodeTo(Node, NPU::T_CFG_SHAPE, MVT::Other, Ops);
    return;
  }

  // NPUISD::ACC_CLR(chain, dt:i64) — emits T_ACC_CLR dt
  // The NPUISD node has chain-only output; the tile token SSA value is
  // threaded back by LowerIntrinsicCall via getMergeValues({dt_const, chain}).
  case NPUISD::ACC_CLR: {
    SDValue Chain = Node->getOperand(0);
    SDValue DT    = CurDAG->getTargetConstant(
        cast<ConstantSDNode>(Node->getOperand(1))->getZExtValue(), DL, MVT::i64);
    SDValue Ops[] = {DT, Chain};
    CurDAG->SelectNodeTo(Node, NPU::T_ACC_CLR, MVT::Other, Ops);
    return;
  }

  // NPUISD::MMA(chain, acc:i64, A:i64, B:i64) — emits T_MMA acc, A, B
  case NPUISD::MMA: {
    SDValue Chain = Node->getOperand(0);
    auto TC = [&](unsigned idx) {
      return CurDAG->getTargetConstant(
          cast<ConstantSDNode>(Node->getOperand(idx))->getZExtValue(),
          DL, MVT::i64);
    };
    SDValue Ops[] = {TC(1), TC(2), TC(3), Chain};
    CurDAG->SelectNodeTo(Node, NPU::T_MMA, MVT::Other, Ops);
    return;
  }

  // NPUISD::MATMUL(chain, dt:i64, A:i64, B:i64) — emits T_MATMUL dt, A, B
  case NPUISD::MATMUL: {
    SDValue Chain = Node->getOperand(0);
    auto TC = [&](unsigned idx) {
      return CurDAG->getTargetConstant(
          cast<ConstantSDNode>(Node->getOperand(idx))->getZExtValue(),
          DL, MVT::i64);
    };
    SDValue Ops[] = {TC(1), TC(2), TC(3), Chain};
    CurDAG->SelectNodeTo(Node, NPU::T_MATMUL, MVT::Other, Ops);
    return;
  }

  // NPUISD::DOT(chain, dt:i64, A:i64, B:i64) — emits T_DOT dt, A, B
  // CF_T3 format: same 3-register layout as MMA and MATMUL.
  case NPUISD::DOT: {
    SDValue Chain = Node->getOperand(0);
    auto TC = [&](unsigned idx) {
      return CurDAG->getTargetConstant(
          cast<ConstantSDNode>(Node->getOperand(idx))->getZExtValue(),
          DL, MVT::i64);
    };
    SDValue Ops[] = {TC(1), TC(2), TC(3), Chain};
    CurDAG->SelectNodeTo(Node, NPU::T_DOT, MVT::Other, Ops);
    return;
  }

  // NPUISD::TELOAD(chain, dt:i64, ptr:i64, count:i64) — emits T_LOAD
  case NPUISD::TELOAD: {
    SDValue Chain = Node->getOperand(0);
    SDValue DT    = CurDAG->getTargetConstant(
        cast<ConstantSDNode>(Node->getOperand(1))->getZExtValue(), DL, MVT::i64);
    SDValue Ptr   = Node->getOperand(2);
    SDValue Count = CurDAG->getTargetConstant(
        cast<ConstantSDNode>(Node->getOperand(3))->getZExtValue(), DL, MVT::i64);
    SDValue Ops[] = {DT, Ptr, Count, Chain};
    CurDAG->SelectNodeTo(Node, NPU::T_LOAD, MVT::Other, Ops);
    return;
  }

  // NPUISD::TELOADW(chain, dt:i64, ptr:i64, count:i64) — emits T_LOADW
  case NPUISD::TELOADW: {
    SDValue Chain = Node->getOperand(0);
    SDValue DT    = CurDAG->getTargetConstant(
        cast<ConstantSDNode>(Node->getOperand(1))->getZExtValue(), DL, MVT::i64);
    SDValue Ptr   = Node->getOperand(2);
    SDValue Count = CurDAG->getTargetConstant(
        cast<ConstantSDNode>(Node->getOperand(3))->getZExtValue(), DL, MVT::i64);
    SDValue Ops[] = {DT, Ptr, Count, Chain};
    CurDAG->SelectNodeTo(Node, NPU::T_LOADW, MVT::Other, Ops);
    return;
  }

  // NPUISD::TESTORE(chain, ptr:i64, tile:i64, count:i64) — emits T_STORE
  case NPUISD::TESTORE: {
    SDValue Chain = Node->getOperand(0);
    SDValue Ptr   = Node->getOperand(1);
    SDValue Tile  = CurDAG->getTargetConstant(
        cast<ConstantSDNode>(Node->getOperand(2))->getZExtValue(), DL, MVT::i64);
    SDValue Count = CurDAG->getTargetConstant(
        cast<ConstantSDNode>(Node->getOperand(3))->getZExtValue(), DL, MVT::i64);
    SDValue Ops[] = {Ptr, Tile, Count, Chain};
    CurDAG->SelectNodeTo(Node, NPU::T_STORE, MVT::Other, Ops);
    return;
  }

  // NPUISD::DMA_IN(chain, dst:i64, src:i64, bytes:i64) — emits T_DMA_IN
  case NPUISD::DMA_IN: {
    SDValue Chain = Node->getOperand(0);
    SDValue Dst   = Node->getOperand(1);
    SDValue Src   = Node->getOperand(2);
    SDValue Bytes = CurDAG->getTargetConstant(
        cast<ConstantSDNode>(Node->getOperand(3))->getZExtValue(), DL, MVT::i64);
    SDValue Ops[] = {Dst, Src, Bytes, Chain};
    CurDAG->SelectNodeTo(Node, NPU::T_DMA_IN, MVT::Other, Ops);
    return;
  }

  // NPUISD::DMA_OUT(chain, dst:i64, src:i64, bytes:i64) — emits T_DMA_OUT
  case NPUISD::DMA_OUT: {
    SDValue Chain = Node->getOperand(0);
    SDValue Dst   = Node->getOperand(1);
    SDValue Src   = Node->getOperand(2);
    SDValue Bytes = CurDAG->getTargetConstant(
        cast<ConstantSDNode>(Node->getOperand(3))->getZExtValue(), DL, MVT::i64);
    SDValue Ops[] = {Dst, Src, Bytes, Chain};
    CurDAG->SelectNodeTo(Node, NPU::T_DMA_OUT, MVT::Other, Ops);
    return;
  }

  // NPUISD::RELU(chain, tile:i64) — emits T_RELU dt=tile, a=tile
  case NPUISD::RELU: {
    SDValue Chain = Node->getOperand(0);
    SDValue Tile  = CurDAG->getTargetConstant(
        cast<ConstantSDNode>(Node->getOperand(1))->getZExtValue(), DL, MVT::i64);
    // T_RELU format: TensorTile2 (dt, a) — in-place, so dt=a=tile
    SDValue Ops[] = {Tile, Tile, Chain};
    CurDAG->SelectNodeTo(Node, NPU::T_RELU, MVT::Other, Ops);
    return;
  }

  // NPUISD::SCALE(chain, tile:i64, factor_bits:i64) — emits T_SCALE dt=tile, a=tile, factor
  // factor_bits is a TargetConstant:i64 holding the IEEE 754 double bit pattern
  // of the scale factor.  Convert it back to TargetConstantFP:f64 here; machine
  // node operand types are not type-legalized, so f64 is fine at this point.
  case NPUISD::SCALE: {
    SDValue Chain = Node->getOperand(0);
    SDValue Tile  = CurDAG->getTargetConstant(
        cast<ConstantSDNode>(Node->getOperand(1))->getZExtValue(), DL, MVT::i64);
    uint64_t Bits = cast<ConstantSDNode>(Node->getOperand(2))->getZExtValue();
    APFloat FVal(APFloat::IEEEdouble(), APInt(64, Bits));
    SDValue Factor = CurDAG->getTargetConstantFP(FVal, DL, MVT::f64);
    // T_SCALE format: (dt, a, factor) where dt=a=tile (in-place scaling)
    SDValue Ops[] = {Tile, Tile, Factor, Chain};
    CurDAG->SelectNodeTo(Node, NPU::T_SCALE, MVT::Other, Ops);
    return;
  }

  // NPUISD::BCAST(chain, dt:i64, val_bits:i64) — emits T_BCAST dt, xS
  // The value is the f32 bits zero-extended to i64, living in a GPR (not fimm).
  // The simulator reads float s = (float)getx(n, I->rs1) — integer reg, not fimm.
  case NPUISD::BCAST: {
    SDValue Chain = Node->getOperand(0);
    SDValue DT = CurDAG->getTargetConstant(
        cast<ConstantSDNode>(Node->getOperand(1))->getZExtValue(), DL, MVT::i64);
    // operand(2) is ISD::Constant:i64 (f32 bits); let the register allocator
    // materialize it into a GPR via LI — pass it directly as an SDValue.
    SDValue Val = Node->getOperand(2);
    SDValue Ops[] = {DT, Val, Chain};
    CurDAG->SelectNodeTo(Node, NPU::T_BCAST, MVT::Other, Ops);
    return;
  }

  // ── Vector Core ───────────────────────────────────────────────────────────

  // NPUISD::VSETVL(chain, len:i64) → (actual_vl:i64, chain)
  // Emits: V_VSETVL xD, xS — xD = min(xS, VLEN_MAX)
  case NPUISD::VSETVL: {
    SDValue Chain = Node->getOperand(0);
    SDValue Len   = Node->getOperand(1); // requested VL in a GPR (i64)
    SDVTList VTs = CurDAG->getVTList(MVT::i64, MVT::Other);
    CurDAG->SelectNodeTo(Node, NPU::V_VSETVL, VTs, {Len, Chain});
    return;
  }

  // NPUISD::VLOAD(chain, ptr:i64, stride:i64) → (v16f32, chain)
  // Emits: V_VLOAD vD, stride(xBase)
  case NPUISD::VLOAD: {
    SDValue Chain  = Node->getOperand(0);
    SDValue Ptr    = Node->getOperand(1);
    SDValue Stride = Node->getOperand(2);

    // If stride is a constant, use it as the imm field; otherwise use 0
    // and add a comment that runtime strides require a different approach.
    SDValue Imm = isa<ConstantSDNode>(Stride)
        ? CurDAG->getTargetConstant(
              cast<ConstantSDNode>(Stride)->getZExtValue(), DL, MVT::i64)
        : CurDAG->getTargetConstant(0, DL, MVT::i64);

    SDValue Ops[] = {Ptr, Imm, Chain};
    CurDAG->SelectNodeTo(Node, NPU::V_VLOAD, MVT::v16f32, MVT::Other, Ops);
    return;
  }

  // NPUISD::VSTORE(chain, ptr:i64, vec:v16f32, stride:i64) → chain
  // Emits: V_VSTORE vS, stride(xBase)
  case NPUISD::VSTORE: {
    SDValue Chain  = Node->getOperand(0);
    SDValue Ptr    = Node->getOperand(1);
    SDValue Vec    = Node->getOperand(2);
    SDValue Stride = Node->getOperand(3);

    SDValue Imm = isa<ConstantSDNode>(Stride)
        ? CurDAG->getTargetConstant(
              cast<ConstantSDNode>(Stride)->getZExtValue(), DL, MVT::i64)
        : CurDAG->getTargetConstant(0, DL, MVT::i64);

    SDValue Ops[] = {Vec, Ptr, Imm, Chain};
    CurDAG->SelectNodeTo(Node, NPU::V_VSTORE, MVT::Other, Ops);
    return;
  }

  // NPUISD::VGATHER(chain, ptr:i64, idx:v16f32) → (v16f32, chain)
  // Emits: V_VGATHER vD, 0(xBase), vIdx
  case NPUISD::VGATHER: {
    SDValue Chain = Node->getOperand(0);
    SDValue Ptr   = Node->getOperand(1);
    SDValue Idx   = Node->getOperand(2);
    SDValue Zero  = CurDAG->getTargetConstant(0, DL, MVT::i64);
    SDValue Ops[] = {Ptr, Zero, Idx, Chain};
    CurDAG->SelectNodeTo(Node, NPU::V_VGATHER, MVT::v16f32, MVT::Other, Ops);
    return;
  }

  // NPUISD::VSCATTER(chain, ptr:i64, idx:v16f32, data:v16f32) → chain
  // Emits: V_VSCATTER vS, 0(xBase), vIdx
  case NPUISD::VSCATTER: {
    SDValue Chain = Node->getOperand(0);
    SDValue Ptr   = Node->getOperand(1);
    SDValue Idx   = Node->getOperand(2);
    SDValue Data  = Node->getOperand(3);
    SDValue Zero  = CurDAG->getTargetConstant(0, DL, MVT::i64);
    SDValue Ops[] = {Data, Ptr, Zero, Idx, Chain};
    CurDAG->SelectNodeTo(Node, NPU::V_VSCATTER, MVT::Other, Ops);
    return;
  }

  default:
    break;
  }

  SelectCode(Node);
}

FunctionPass *llvm::createNPUISelDag(NPUTargetMachine &TM) {
  return new NPUDAGToDAGISelLegacy(TM);
}
