//===-- NPUAsmPrinter.cpp - NPU LLVM assembly writer ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Converts NPU MachineInstrs to assembly text. The control core executes one
// flat program terminated by HALT, so a single HALT is emitted at end of file.
//
//===----------------------------------------------------------------------===//

#include "MCTargetDesc/NPUInstPrinter.h"
#include "MCTargetDesc/NPUMCTargetDesc.h"
#include "NPU.h"
#include "NPUMCInstLower.h"
#include "NPUTargetMachine.h"
#include "TargetInfo/NPUTargetInfo.h"
#include "llvm/CodeGen/AsmPrinter.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/Compiler.h"

using namespace llvm;

#define DEBUG_TYPE "asm-printer"

namespace {
class NPUAsmPrinter : public AsmPrinter {
public:
  explicit NPUAsmPrinter(TargetMachine &TM,
                         std::unique_ptr<MCStreamer> Streamer)
      : AsmPrinter(TM, std::move(Streamer), ID) {}

  StringRef getPassName() const override { return "NPU Assembly Printer"; }

  void emitInstruction(const MachineInstr *MI) override;
  void emitFunctionBodyEnd() override;

  static char ID;
};
} // namespace

void NPUAsmPrinter::emitInstruction(const MachineInstr *MI) {
  NPUMCInstLower MCInstLowering(OutContext, *this);
  MCInst TmpInst;
  MCInstLowering.Lower(MI, TmpInst);
  EmitToStreamer(*OutStreamer, TmpInst);
}

void NPUAsmPrinter::emitFunctionBodyEnd() {
  // The simulator runs a single flat control program terminated by HALT.
  // Emit one after the function body (where the MachineFunction is still
  // valid, so the subtarget info is available).
  MCInst Halt;
  Halt.setOpcode(NPU::HALT);
  EmitToStreamer(*OutStreamer, Halt);
}

char NPUAsmPrinter::ID = 0;

extern "C" LLVM_ABI LLVM_EXTERNAL_VISIBILITY void LLVMInitializeNPUAsmPrinter() {
  RegisterAsmPrinter<NPUAsmPrinter> X(getTheNPUTarget());
}
