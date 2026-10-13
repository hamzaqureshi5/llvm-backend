//===-- NPUInstrInfo.h - NPU Instruction Information ------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_NPU_NPUINSTRINFO_H
#define LLVM_LIB_TARGET_NPU_NPUINSTRINFO_H

#include "NPURegisterInfo.h"
#include "llvm/CodeGen/TargetInstrInfo.h"

// Register enum (NPU::X0 … NPU::X31, NPU::V0 …) and instruction enum
// (NPU::ADD, NPU::HALT …) for CodeGen files.
// Both are guarded with the same macro used in NPUMCTargetDesc.h so that a TU
// which includes both headers (e.g. NPUAsmPrinter.cpp through its MCTargetDesc
// include and its NPUSubtarget.h include) processes each enum exactly once.
#ifndef NPU_REGINFO_ENUM_INCLUDED
#define NPU_REGINFO_ENUM_INCLUDED
#define GET_REGINFO_ENUM
#include "NPUGenRegisterInfo.inc"
#endif

#ifndef NPU_INSTRINFO_ENUM_INCLUDED
#define NPU_INSTRINFO_ENUM_INCLUDED
#define GET_INSTRINFO_ENUM
#include "NPUGenInstrInfo.inc"
#endif

// Class-definition header (NPUGenInstrInfo class body, always safe to include).
#define GET_INSTRINFO_HEADER
#include "NPUGenInstrInfo.inc"

namespace llvm {

class NPUSubtarget;

class NPUInstrInfo : public NPUGenInstrInfo {
  const NPURegisterInfo RegisterInfo;

public:
  explicit NPUInstrInfo(const NPUSubtarget &STI);

  const NPURegisterInfo &getRegisterInfo() const { return RegisterInfo; }

  void copyPhysReg(MachineBasicBlock &MBB, MachineBasicBlock::iterator Position,
                   const DebugLoc &DL, Register DestinationRegister,
                   Register SourceRegister, bool KillSource,
                   bool RenamableDest = false,
                   bool RenamableSrc = false) const override;

  void storeRegToStackSlot(
      MachineBasicBlock &MBB, MachineBasicBlock::iterator Position,
      Register SourceRegister, bool IsKill, int FrameIndex,
      const TargetRegisterClass *RegisterClass, Register VReg,
      MachineInstr::MIFlag Flags = MachineInstr::NoFlags) const override;

  void loadRegFromStackSlot(
      MachineBasicBlock &MBB, MachineBasicBlock::iterator Position,
      Register DestinationRegister, int FrameIndex,
      const TargetRegisterClass *RegisterClass, Register VReg,
      unsigned SubReg = 0,
      MachineInstr::MIFlag Flags = MachineInstr::NoFlags) const override;
};

} // namespace llvm

#endif // LLVM_LIB_TARGET_NPU_NPUINSTRINFO_H
