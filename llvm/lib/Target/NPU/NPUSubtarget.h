//===-- NPUSubtarget.h - Define Subtarget for the NPU -----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_NPU_NPUSUBTARGET_H
#define LLVM_LIB_TARGET_NPU_NPUSUBTARGET_H

#include "NPUFrameLowering.h"
#include "NPUISelLowering.h"
#include "NPUInstrInfo.h"
#include "NPUSelectionDAGInfo.h"
#include "llvm/CodeGen/TargetSubtargetInfo.h"
#include "llvm/Target/TargetMachine.h"

#define GET_SUBTARGETINFO_HEADER
#include "NPUGenSubtargetInfo.inc"

namespace llvm {

class NPUSubtarget : public NPUGenSubtargetInfo {
public:
  NPUSubtarget(const Triple &TargetTriple, StringRef Cpu, StringRef FeatureString,
               const TargetMachine &TM);

  void ParseSubtargetFeatures(StringRef CPU, StringRef TuneCPU, StringRef FS);

  NPUSubtarget &initializeSubtargetDependencies(StringRef CPU, StringRef FS);

  const NPUInstrInfo *getInstrInfo() const override { return &InstrInfo; }

  const TargetFrameLowering *getFrameLowering() const override {
    return &FrameLowering;
  }

  const NPURegisterInfo *getRegisterInfo() const override {
    return &InstrInfo.getRegisterInfo();
  }

  const NPUTargetLowering *getTargetLowering() const override {
    return &TLInfo;
  }

  const NPUSelectionDAGInfo *getSelectionDAGInfo() const override {
    return &TSInfo;
  }

private:
  NPUInstrInfo InstrInfo;
  NPUFrameLowering FrameLowering;
  NPUTargetLowering TLInfo;
  NPUSelectionDAGInfo TSInfo;
};

} // namespace llvm

#endif // LLVM_LIB_TARGET_NPU_NPUSUBTARGET_H
