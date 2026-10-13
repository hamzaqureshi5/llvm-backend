//===-- NPUMCAsmInfo.h - NPU asm properties ---------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_NPU_MCTARGETDESC_NPUMCASMINFO_H
#define LLVM_LIB_TARGET_NPU_MCTARGETDESC_NPUMCASMINFO_H

#include "llvm/MC/MCAsmInfoELF.h"

namespace llvm {
class Triple;

class NPUMCAsmInfo : public MCAsmInfoELF {
  void anchor() override;

public:
  explicit NPUMCAsmInfo(const Triple &TheTriple,
                        const MCTargetOptions &Options);
};

} // namespace llvm

#endif // LLVM_LIB_TARGET_NPU_MCTARGETDESC_NPUMCASMINFO_H
