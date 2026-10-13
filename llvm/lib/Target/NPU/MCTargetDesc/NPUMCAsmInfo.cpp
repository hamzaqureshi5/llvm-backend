//===-- NPUMCAsmInfo.cpp - NPU asm properties -----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "NPUMCAsmInfo.h"
#include "llvm/TargetParser/Triple.h"

using namespace llvm;

void NPUMCAsmInfo::anchor() {}

NPUMCAsmInfo::NPUMCAsmInfo(const Triple & /*TheTriple*/,
                           const MCTargetOptions &Options)
    : MCAsmInfoELF(Options) {
  IsLittleEndian = true;
  InternalSymbolPrefix = ".L";
  // The simulator's assembler treats '#' (and ';') as comment markers.
  CommentString = "#";
  SupportsDebugInformation = true;
  MinInstAlignment = 4;
}
