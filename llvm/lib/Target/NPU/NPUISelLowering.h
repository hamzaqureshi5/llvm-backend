//===-- NPUISelLowering.h - NPU DAG Lowering Interface ----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Defines the SelectionDAG lowering for the NPU target.  The class handles
// three categories of lowering:
//
//   1. Calling convention (LowerFormalArguments / LowerCall / LowerReturn)
//      — maps function arguments and return values to the RISC-V-style ABI
//        implemented by CC_NPU / RetCC_NPU in NPUCallingConv.td.
//
//   2. Intrinsic calls (LowerIntrinsicCall)
//      — converts @llvm.npu.* calls in the DAG to target-specific nodes
//        (NPUISD::KLAUNCH etc.) that the instruction selector can then match
//        to real machine instructions.
//
//   3. Unsigned comparisons (used by LowerOperation for BR_CC with unsigned CC)
//      — the control core has no BLTU/BGEU; unsigned conditions are synthesized
//        by XOR-flipping the sign bit then using a signed branch.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_NPU_NPUISELLOWERING_H
#define LLVM_LIB_TARGET_NPU_NPUISELLOWERING_H

#include "NPU.h"
#include "llvm/CodeGen/TargetLowering.h"

// Target-specific SelectionDAG node opcodes (NPUISD namespace).
// Generated from the SDNode defs in NPUInstrInfo.td (npucall, npuretglue,
// npuklaunch, npukwait).  Must be included at file scope, NOT inside any
// "namespace llvm { }" block, because the generated file uses the C++17
// "namespace llvm::NPUISD { }" syntax which would otherwise nest as
// "namespace llvm::llvm::NPUISD { }".
#define GET_SDNODE_ENUM
#include "NPUGenSDNodeInfo.inc"

namespace llvm {

class NPUSubtarget;

class NPUTargetLowering : public TargetLowering {
public:
  NPUTargetLowering(const TargetMachine &TM, const NPUSubtarget &STI);

  // Called by SelectionDAGISel for nodes marked Custom in the legalization
  // table.  Currently handles:
  //   ISD::INTRINSIC_VOID — void @llvm.npu.* calls (klaunch, kwait, dma, etc.)
  //   ISD::INTRINSIC_W_CHAIN — chained @llvm.npu.* calls (vsetvl)
  SDValue LowerOperation(SDValue Op, SelectionDAG &DAG) const override;

  // Called for every intrinsic that wasn't already legalized away.
  SDValue LowerIntrinsicCall(SDValue Op, SelectionDAG &DAG) const;

  const char *getTargetNodeName(unsigned Opcode) const override;

private:
  SDValue LowerCallResult(SDValue Chain, SDValue InGlue,
                          CallingConv::ID CallConv, bool IsVarArg,
                          const SmallVectorImpl<ISD::InputArg> &Ins,
                          const SDLoc &DL, SelectionDAG &DAG,
                          SmallVectorImpl<SDValue> &InVals) const;

  SDValue LowerCall(TargetLowering::CallLoweringInfo &CLI,
                    SmallVectorImpl<SDValue> &InVals) const override;

  SDValue LowerFormalArguments(SDValue Chain, CallingConv::ID CallConv,
                               bool IsVarArg,
                               const SmallVectorImpl<ISD::InputArg> &Ins,
                               const SDLoc &DL, SelectionDAG &DAG,
                               SmallVectorImpl<SDValue> &InVals) const override;

  SDValue LowerReturn(SDValue Chain, CallingConv::ID CallConv, bool IsVarArg,
                      const SmallVectorImpl<ISD::OutputArg> &Outs,
                      const SmallVectorImpl<SDValue> &OutVals, const SDLoc &DL,
                      SelectionDAG &DAG) const override;
};

} // namespace llvm

#endif // LLVM_LIB_TARGET_NPU_NPUISELLOWERING_H
