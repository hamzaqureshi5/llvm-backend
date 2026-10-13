//===-- NPU.h - Top-level interface for NPU representation ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the entry points for global functions defined in the LLVM
// NPU back-end.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_NPU_NPU_H
#define LLVM_LIB_TARGET_NPU_NPU_H

#include "llvm/Pass.h"

namespace llvm {
class FunctionPass;
class NPUTargetMachine;
class PassRegistry;

// createNPUISelDag - This pass converts a legalized DAG into an NPU-specific
// DAG, ready for instruction scheduling.
FunctionPass *createNPUISelDag(NPUTargetMachine &TM);

void initializeNPUDAGToDAGISelLegacyPass(PassRegistry &);

} // namespace llvm

#endif // LLVM_LIB_TARGET_NPU_NPU_H
