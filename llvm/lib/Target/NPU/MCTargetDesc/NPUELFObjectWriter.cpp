//===-- NPUELFObjectWriter.cpp - NPU ELF Writer ---------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Maps NPU fixup kinds to ELF RELA relocation types (defined in NPU.def).
//
// The ELF machine type is EM_NPU (315), a custom non-standard value assigned
// for this experimental target.
//
// Relocation applied by the NPU linker:
//   R_NPU_BRANCH: *(i64 at r_offset) = (S + A) / 32
//   R_NPU_32:     *(u32 at r_offset) =  S + A
//   R_NPU_64:     *(u64 at r_offset) =  S + A
//
// r_offset for R_NPU_BRANCH points to byte 16 of the branch instruction (the
// imm field), because the fixup is created with offset 16 in the emitter.
//
//===----------------------------------------------------------------------===//

#include "NPUFixups.h"
#include "llvm/BinaryFormat/ELF.h"
#include "llvm/MC/MCELFObjectWriter.h"
#include "llvm/MC/MCFixup.h"
#include "llvm/MC/MCValue.h"
#include "llvm/Support/ErrorHandling.h"

using namespace llvm;

namespace {

class NPUELFObjectWriter : public MCELFObjectTargetWriter {
public:
  NPUELFObjectWriter()
      : MCELFObjectTargetWriter(/*Is64Bit=*/true,
                                /*OSABI=*/0,
                                /*EMachine=*/ELF::EM_NPU,
                                /*HasRelocationAddend=*/true) {}

  unsigned getRelocType(const MCFixup &Fixup, const MCValue &,
                        bool /*IsPCRel*/) const override {
    MCFixupKind Kind = Fixup.getKind();

    // NPU-specific fixup: branch target stored as absolute instruction index.
    if ((unsigned)Kind == NPU::fixup_NPU_branch)
      return ELF::R_NPU_BRANCH;

    // Generic data fixups produced by inline assembly or data directives.
    switch (Kind) {
    case FK_Data_4: return ELF::R_NPU_32;
    case FK_Data_8: return ELF::R_NPU_64;
    default:
      llvm_unreachable("NPU: unsupported relocation type");
    }
  }
};

} // namespace

std::unique_ptr<MCObjectTargetWriter> createNPUELFObjectWriter() {
  return std::make_unique<NPUELFObjectWriter>();
}
