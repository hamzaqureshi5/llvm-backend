//===-- NPUMCInstLower.h - Lower MachineInstr to MCInst ---------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_NPU_NPUMCINSTLOWER_H
#define LLVM_LIB_TARGET_NPU_NPUMCINSTLOWER_H

#include "llvm/Support/Compiler.h"

namespace llvm {
class AsmPrinter;
class MCContext;
class MCInst;
class MCOperand;
class MCSymbol;
class MachineInstr;
class MachineOperand;

class LLVM_LIBRARY_VISIBILITY NPUMCInstLower {
  MCContext &Ctx;
  AsmPrinter &Printer;

public:
  NPUMCInstLower(MCContext &Ctx, AsmPrinter &AP) : Ctx(Ctx), Printer(AP) {}

  void Lower(const MachineInstr *MI, MCInst &OutMI) const;
  MCOperand LowerSymbolOperand(const MachineOperand &MO, MCSymbol *Sym) const;
};

} // namespace llvm

#endif // LLVM_LIB_TARGET_NPU_NPUMCINSTLOWER_H
