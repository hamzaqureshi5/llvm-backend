//===-- NPUFrameLowering.cpp - NPU Frame Information ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Minimal frame lowering for the NPU control core. The simulator has no real
// stack; this adjusts the (invented) stack pointer x2 by the frame size using
// ADDI when a frame is needed, and is empty otherwise.
//
//===----------------------------------------------------------------------===//

#include "NPUFrameLowering.h"
#include "NPUInstrInfo.h"
#include "NPUSubtarget.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"

using namespace llvm;

void NPUFrameLowering::emitPrologue(MachineFunction &MF,
                                    MachineBasicBlock &MBB) const {
  MachineFrameInfo &MFI = MF.getFrameInfo();
  uint64_t StackSize = MFI.getStackSize();
  if (StackSize == 0 && !MFI.adjustsStack())
    return;

  const NPUInstrInfo &TII =
      *static_cast<const NPUInstrInfo *>(STI.getInstrInfo());
  MachineBasicBlock::iterator MBBI = MBB.begin();
  DebugLoc DL;
  // x2 = x2 - StackSize (encoded as ADDI x2, x2, -StackSize).
  BuildMI(MBB, MBBI, DL, TII.get(NPU::ADDI), NPU::X2)
      .addReg(NPU::X2)
      .addImm(-(int64_t)StackSize);
}

void NPUFrameLowering::emitEpilogue(MachineFunction &MF,
                                    MachineBasicBlock &MBB) const {
  MachineFrameInfo &MFI = MF.getFrameInfo();
  uint64_t StackSize = MFI.getStackSize();
  if (StackSize == 0 && !MFI.adjustsStack())
    return;

  const NPUInstrInfo &TII =
      *static_cast<const NPUInstrInfo *>(STI.getInstrInfo());
  MachineBasicBlock::iterator MBBI = MBB.getLastNonDebugInstr();
  DebugLoc DL = MBBI != MBB.end() ? MBBI->getDebugLoc() : DebugLoc();
  BuildMI(MBB, MBBI, DL, TII.get(NPU::ADDI), NPU::X2)
      .addReg(NPU::X2)
      .addImm((int64_t)StackSize);
}

MachineBasicBlock::iterator NPUFrameLowering::eliminateCallFramePseudoInstr(
    MachineFunction &MF, MachineBasicBlock &MBB,
    MachineBasicBlock::iterator I) const {
  // We do not reserve call frame space dynamically; just drop the pseudos.
  return MBB.erase(I);
}
