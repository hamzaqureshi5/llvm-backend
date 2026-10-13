//===-- NPUMCTargetDesc.h - NPU Target Descriptions -------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file provides NPU specific target descriptions.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_NPU_MCTARGETDESC_NPUMCTARGETDESC_H
#define LLVM_LIB_TARGET_NPU_MCTARGETDESC_NPUMCTARGETDESC_H

#include "llvm/Support/DataTypes.h"

namespace llvm {
class MCAsmBackend;
class MCRegisterInfo;
class MCSubtargetInfo;
class MCTargetOptions;
class Target;
MCAsmBackend *createNPUMCAsmBackend(const Target &T,
                                    const MCSubtargetInfo &STI,
                                    const MCRegisterInfo &MRI,
                                    const MCTargetOptions &Options);
} // namespace llvm

// Symbolic names for NPU registers (NPU::X0, NPU::V0, …).
// Guarded: NPUGenRegisterInfoEnums.inc has no include guard of its own, so
// including it twice via GET_REGINFO_ENUM causes enumerator redefinitions.
// Any header that also includes this file (e.g. NPUInstrInfo.h) uses the same
// guard so only the first inclusion wins.
#ifndef NPU_REGINFO_ENUM_INCLUDED
#define NPU_REGINFO_ENUM_INCLUDED
#define GET_REGINFO_ENUM
#include "NPUGenRegisterInfo.inc"
#endif

// Symbolic names for NPU instructions (NPU::ADD, NPU::HALT, …).
// Same guard pattern for the same reason.
#ifndef NPU_INSTRINFO_ENUM_INCLUDED
#define NPU_INSTRINFO_ENUM_INCLUDED
#define GET_INSTRINFO_ENUM
#include "NPUGenInstrInfo.inc"
#endif

// MC helper decls (verifyInstructionPredicates etc.).  These are function
// declarations, which are safe to repeat; no guard needed.
#define GET_INSTRINFO_MC_HELPER_DECLS
#include "NPUGenInstrInfo.inc"

#define GET_SUBTARGETINFO_ENUM
#include "NPUGenSubtargetInfo.inc"

#endif // LLVM_LIB_TARGET_NPU_MCTARGETDESC_NPUMCTARGETDESC_H
