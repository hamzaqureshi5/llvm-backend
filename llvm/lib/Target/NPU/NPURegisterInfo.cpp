//===-- NPURegisterInfo.cpp - NPU Register Information --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "NPURegisterInfo.h"
#include "NPUInstrInfo.h"
#include "NPUSubtarget.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/TargetFrameLowering.h"
#include "llvm/CodeGen/TargetInstrInfo.h"

#define GET_REGINFO_TARGET_DESC
#include "NPUGenRegisterInfo.inc"

using namespace llvm;

NPURegisterInfo::NPURegisterInfo() : NPUGenRegisterInfo(/*RA=*/NPU::X1) {}

const MCPhysReg *
NPURegisterInfo::getCalleeSavedRegs(const MachineFunction * /*MF*/) const {
  return CSR_SaveList;
}

const uint32_t *
NPURegisterInfo::getCallPreservedMask(const MachineFunction & /*MF*/,
                                      CallingConv::ID /*CC*/) const {
  return CSR_RegMask;
}

BitVector NPURegisterInfo::getReservedRegs(const MachineFunction & /*MF*/) const {
  BitVector Reserved(getNumRegs());
  Reserved.set(NPU::X0); // hardwired zero
  Reserved.set(NPU::X1); // return address
  Reserved.set(NPU::X2); // stack pointer
  return Reserved;
}

bool NPURegisterInfo::eliminateFrameIndex(MachineBasicBlock::iterator II,
                                          int SPAdj, unsigned FIOperandNum,
                                          RegScavenger * /*RS*/) const {
  assert(SPAdj == 0 && "Unexpected non-zero SPAdj");
  MachineInstr &MI = *II;
  MachineFunction &MF = *MI.getParent()->getParent();

  int FrameIndex = MI.getOperand(FIOperandNum).getIndex();
  int64_t Offset = MF.getFrameInfo().getObjectOffset(FrameIndex) +
                   MF.getFrameInfo().getStackSize() +
                   MI.getOperand(FIOperandNum + 1).getImm();

  // Replace the frame index with the stack pointer and a concrete offset.
  MI.getOperand(FIOperandNum).ChangeToRegister(getFrameRegister(MF),
                                               /*isDef=*/false);
  MI.getOperand(FIOperandNum + 1).ChangeToImmediate(Offset);
  return false;
}

Register
NPURegisterInfo::getFrameRegister(const MachineFunction & /*MF*/) const {
  return NPU::X2;
}
