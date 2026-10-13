//===- NPUMachineFunctionInfo.h - NPU machine func info ---------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_NPU_NPUMACHINEFUNCTIONINFO_H
#define LLVM_LIB_TARGET_NPU_NPUMACHINEFUNCTIONINFO_H

#include "llvm/CodeGen/MachineFunction.h"

namespace llvm {

class NPUMachineFunctionInfo : public MachineFunctionInfo {
  virtual void anchor();

public:
  NPUMachineFunctionInfo(const Function &F, const TargetSubtargetInfo *STI) {}

  MachineFunctionInfo *
  clone(BumpPtrAllocator &Allocator, MachineFunction &DestMF,
        const DenseMap<MachineBasicBlock *, MachineBasicBlock *> &Src2DstMBB)
      const override;
};

} // namespace llvm

#endif // LLVM_LIB_TARGET_NPU_NPUMACHINEFUNCTIONINFO_H
