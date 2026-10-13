//===-- NPUInstrInfo.cpp - NPU Instruction Information --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "NPUInstrInfo.h"
#include "NPUSubtarget.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/Support/ErrorHandling.h"

#define GET_INSTRINFO_CTOR_DTOR
#include "NPUGenInstrInfo.inc"

using namespace llvm;

NPUInstrInfo::NPUInstrInfo(const NPUSubtarget &STI)
    : NPUGenInstrInfo(STI, RegisterInfo, NPU::ADJCALLSTACKDOWN,
                      NPU::ADJCALLSTACKUP),
      RegisterInfo() {}

void NPUInstrInfo::copyPhysReg(MachineBasicBlock &MBB,
                               MachineBasicBlock::iterator Position,
                               const DebugLoc &DL, Register DestinationRegister,
                               Register SourceRegister, bool KillSource,
                               bool /*RenamableDest*/,
                               bool /*RenamableSrc*/) const {
  // ── Control core GPR copy: OR xd, xs, x0 ────────────────────────────────
  if (NPU::GPRRegClass.contains(DestinationRegister, SourceRegister)) {
    BuildMI(MBB, Position, DL, get(NPU::OR), DestinationRegister)
        .addReg(SourceRegister, getKillRegState(KillSource))
        .addReg(NPU::X0);
    return;
  }

  // ── Vector core VReg copy: VMOV vd, vs ───────────────────────────────────
  // V_VMOV is defined in NPUVectorInstrInfo.td; it emits a vinst VMOV opcode.
  if (NPU::VRegRegClass.contains(DestinationRegister, SourceRegister)) {
    BuildMI(MBB, Position, DL, get(NPU::V_VMOV), DestinationRegister)
        .addReg(SourceRegister, getKillRegState(KillSource));
    return;
  }

  // ── Vector core SReg copy: SADD sd, ss, s0 (s0 is the zero SReg) ────────
  // The SReg file has no dedicated "copy" instruction; use SADD ss, ss, s0.
  // Note: unlike the control core which has a hardwired x0=0, the SReg file
  // does not have a guaranteed-zero register.  For the teaching backend this
  // is an acceptable simplification — s0 holds the kernel instance id on
  // entry but we reuse it as zero source for copies outside kernel context.
  if (NPU::SRegRegClass.contains(DestinationRegister, SourceRegister)) {
    BuildMI(MBB, Position, DL, get(NPU::V_SADD), DestinationRegister)
        .addReg(SourceRegister, getKillRegState(KillSource))
        .addReg(NPU::S0);
    return;
  }

  llvm_unreachable("Impossible reg-to-reg copy between incompatible classes");
}

void NPUInstrInfo::storeRegToStackSlot(
    MachineBasicBlock &MBB, MachineBasicBlock::iterator Position,
    Register SourceRegister, bool IsKill, int FrameIndex,
    const TargetRegisterClass * /*RegisterClass*/, Register /*VReg*/,
    MachineInstr::MIFlag /*Flags*/) const {
  DebugLoc DL;
  if (Position != MBB.end())
    DL = Position->getDebugLoc();

  BuildMI(MBB, Position, DL, get(NPU::SW))
      .addReg(SourceRegister, getKillRegState(IsKill))
      .addFrameIndex(FrameIndex)
      .addImm(0);
}

void NPUInstrInfo::loadRegFromStackSlot(
    MachineBasicBlock &MBB, MachineBasicBlock::iterator Position,
    Register DestinationRegister, int FrameIndex,
    const TargetRegisterClass * /*RegisterClass*/, Register /*VReg*/,
    unsigned /*SubReg*/, MachineInstr::MIFlag /*Flags*/) const {
  DebugLoc DL;
  if (Position != MBB.end())
    DL = Position->getDebugLoc();

  BuildMI(MBB, Position, DL, get(NPU::LW), DestinationRegister)
      .addFrameIndex(FrameIndex)
      .addImm(0);
}
