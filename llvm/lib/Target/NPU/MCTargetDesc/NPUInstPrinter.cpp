//===-- NPUInstPrinter.cpp - Convert NPU MCInst to asm syntax -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "NPUInstPrinter.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/APInt.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCInst.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

#include "NPUGenAsmWriter.inc"

void NPUInstPrinter::printInst(const MCInst *MI, uint64_t Address,
                               StringRef Annot, const MCSubtargetInfo & /*STI*/,
                               raw_ostream &O) {
  printInstruction(MI, Address, O);
  printAnnotation(O, Annot);
}

void NPUInstPrinter::printOperand(const MCInst *MI, unsigned OpNo,
                                  raw_ostream &O) {
  const MCOperand &Op = MI->getOperand(OpNo);
  if (Op.isReg()) {
    O << getRegisterName(Op.getReg());
    return;
  }
  if (Op.isImm()) {
    O << Op.getImm();
    return;
  }
  if (Op.isDFPImm()) {
    // Floating-point immediate (T_SCALE / T_BCAST factor).
    // The value is stored as a double bit pattern; reconstruct and print it.
    APFloat FVal(APFloat::IEEEdouble(), APInt(64, Op.getDFPImm()));
    SmallString<16> Str;
    FVal.toString(Str, /*FormatPrecision=*/6, /*FormatMaxPadding=*/0,
                  /*TruncateZero=*/false);
    O << Str;
    return;
  }
  assert(Op.isExpr() && "unknown operand kind in printOperand");
  MAI.printExpr(O, *Op.getExpr());
}

// Print "offset(xN)" memory operands.
// The base register is OpNo+0, the offset immediate is OpNo+1, matching
// the InOperandList layout in NPUInstrInfo.td (memri operand).
void NPUInstPrinter::printMemRIOperand(const MCInst *MI, unsigned OpNo,
                                       raw_ostream &O) {
  const MCOperand &Base   = MI->getOperand(OpNo);
  const MCOperand &Offset = MI->getOperand(OpNo + 1);
  assert(Offset.isImm() && "expected immediate offset");
  O << Offset.getImm() << '(' << getRegisterName(Base.getReg()) << ')';
}

// Print tile register numbers as "t<n>" to match the simulator's assembler
// syntax (npu_asm.c CTAB entries for tensor-engine instructions).
void NPUInstPrinter::printTileOperand(const MCInst *MI, unsigned OpNo,
                                      raw_ostream &O) {
  const MCOperand &Op = MI->getOperand(OpNo);
  assert(Op.isImm() && "tile operand must be an immediate");
  O << 't' << Op.getImm();
}
