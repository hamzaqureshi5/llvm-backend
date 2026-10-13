//===-- NPUSubtarget.cpp - NPU Subtarget Information ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "NPUSubtarget.h"

#define DEBUG_TYPE "npu-subtarget"

#define GET_SUBTARGETINFO_TARGET_DESC
#define GET_SUBTARGETINFO_CTOR
#include "NPUGenSubtargetInfo.inc"

using namespace llvm;

NPUSubtarget &NPUSubtarget::initializeSubtargetDependencies(StringRef CPU,
                                                            StringRef FS) {
  std::string CPUName = std::string(CPU);
  if (CPUName.empty())
    CPUName = "generic";
  ParseSubtargetFeatures(CPUName, /*TuneCPU*/ CPUName, FS);
  return *this;
}

NPUSubtarget::NPUSubtarget(const Triple &TargetTriple, StringRef Cpu,
                           StringRef FeatureString, const TargetMachine &TM)
    : NPUGenSubtargetInfo(TargetTriple, Cpu, /*TuneCPU*/ Cpu, FeatureString),
      InstrInfo(initializeSubtargetDependencies(Cpu, FeatureString)),
      FrameLowering(*this), TLInfo(TM, *this) {}
