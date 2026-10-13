//===-- NPUMCTargetDesc.cpp - NPU Target Descriptions ---------------------===//
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

#include "NPUMCTargetDesc.h"
#include "NPUInstPrinter.h"
#include "NPUMCAsmInfo.h"
#include "NPUMCCodeEmitter.h"
#include "TargetInfo/NPUTargetInfo.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/Compiler.h"
#include "llvm/TargetParser/Triple.h"
#include <string>

#define GET_INSTRINFO_MC_DESC
#define ENABLE_INSTR_PREDICATE_VERIFIER
#include "NPUGenInstrInfo.inc"

#define GET_SUBTARGETINFO_MC_DESC
#include "NPUGenSubtargetInfo.inc"

#define GET_REGINFO_MC_DESC
#include "NPUGenRegisterInfo.inc"

using namespace llvm;

static MCInstrInfo *createNPUMCInstrInfo() {
  MCInstrInfo *X = new MCInstrInfo();
  InitNPUMCInstrInfo(X);
  return X;
}

static MCRegisterInfo *createNPUMCRegisterInfo(const Triple & /*TT*/) {
  MCRegisterInfo *X = new MCRegisterInfo();
  // RAReg = X1 (return address), PCReg = X0 (no real PC register).
  InitNPUMCRegisterInfo(X, NPU::X1, 0, 0, NPU::X0);
  return X;
}

static MCSubtargetInfo *
createNPUMCSubtargetInfo(const Triple &TT, StringRef CPU, StringRef FS) {
  std::string CPUName = std::string(CPU);
  if (CPUName.empty())
    CPUName = "generic";
  return createNPUMCSubtargetInfoImpl(TT, CPUName, /*TuneCPU*/ CPUName, FS);
}

static MCInstPrinter *createNPUMCInstPrinter(const Triple & /*T*/,
                                             unsigned SyntaxVariant,
                                             const MCAsmInfo &MAI,
                                             const MCInstrInfo &MII,
                                             const MCRegisterInfo &MRI) {
  if (SyntaxVariant == 0)
    return new NPUInstPrinter(MAI, MII, MRI);
  return nullptr;
}

extern "C" LLVM_ABI LLVM_EXTERNAL_VISIBILITY void LLVMInitializeNPUTargetMC() {
  Target &T = getTheNPUTarget();
  RegisterMCAsmInfo<NPUMCAsmInfo> X(T);
  TargetRegistry::RegisterMCInstrInfo(T, createNPUMCInstrInfo);
  TargetRegistry::RegisterMCRegInfo(T, createNPUMCRegisterInfo);
  TargetRegistry::RegisterMCSubtargetInfo(T, createNPUMCSubtargetInfo);
  TargetRegistry::RegisterMCInstPrinter(T, createNPUMCInstPrinter);
  // Register the binary code emitter so that object-file output works.
  // This is what bridges llc -filetype=obj → 32-byte binary records
  // that the simulator's npu_decode_cinst / npu_decode_vinst can consume.
  TargetRegistry::RegisterMCCodeEmitter(T, createNPUMCCodeEmitter);
  TargetRegistry::RegisterMCAsmBackend(T, createNPUMCAsmBackend);
}
