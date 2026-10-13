//===-- NPUSelectionDAGInfo.h - NPU SelectionDAG Info -----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_NPU_NPUSELECTIONDAGINFO_H
#define LLVM_LIB_TARGET_NPU_NPUSELECTIONDAGINFO_H

#include "llvm/CodeGen/SelectionDAGTargetInfo.h"

// The NPUISD opcode enum (GET_SDNODE_ENUM) is defined once in NPUISelLowering.h.
// Do NOT re-include NPUGenSDNodeInfo.inc here with GET_SDNODE_ENUM — any TU
// that includes both NPUISelLowering.h and NPUSelectionDAGInfo.h would get
// the "redefinition of GenNodeType" error because the second #define would
// re-open the already-processed section.

namespace llvm {

class NPUSelectionDAGInfo : public SelectionDAGGenTargetInfo {
public:
  NPUSelectionDAGInfo();
};

} // namespace llvm

#endif // LLVM_LIB_TARGET_NPU_NPUSELECTIONDAGINFO_H
