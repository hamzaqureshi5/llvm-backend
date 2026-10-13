//===-- NPUAsmBackend.cpp - NPU Assembler Backend -------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Fixup kinds and applyFixup for NPU's fixed-width (32-byte) instruction set.
//
// All branch instructions (BEQ, BNE, BLT, BGE, JAL/BR) store their target
// as an ABSOLUTE instruction index in bytes 16–23 of the 32-byte word.
//
// Fixup layout (fixup_NPU_branch):
//   The MCCodeEmitter emits MCFixup::create(16, expr, fixup_NPU_branch), so
//   the fixup is anchored at offset 16 within the instruction word, pointing
//   directly at the imm field.  Data therefore already points at the imm field
//   when applyFixup is called.
//
// For same-section (resolved) fixups:
//   applyFixup receives Value = absolute byte address of the target.
//   It writes Value / 32 (the instruction index) as a little-endian i64.
//
// For external / cross-section (unresolved) fixups:
//   MCELFObjectTargetWriter::recordRelocation() calls getRelocType() and emits
//   an ELF RELA record (R_NPU_BRANCH).  The future NPU linker applies:
//     *(i64 at r_offset) = (S + A) / 32
//
//===----------------------------------------------------------------------===//

#include "NPUFixups.h"
#include "NPUMCTargetDesc.h"
#include "llvm/MC/MCAsmBackend.h"
#include "llvm/MC/MCAssembler.h"
#include "llvm/MC/MCFixup.h"
#include "llvm/MC/MCObjectWriter.h"
#include "llvm/MC/MCSection.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/Support/raw_ostream.h"
#include <cstring>

using namespace llvm;

std::unique_ptr<MCObjectTargetWriter> createNPUELFObjectWriter();

namespace {

class NPUMCAsmBackend : public MCAsmBackend {
public:
  NPUMCAsmBackend() : MCAsmBackend(llvm::endianness::little) {}

  // ── Fixup metadata ────────────────────────────────────────────────────────

  MCFixupKindInfo getFixupKindInfo(MCFixupKind Kind) const override {
    // Name, bit-offset within field, bit-size of field, flags.
    // TargetOffset/TargetSize are relative to the fixup anchor (byte 16 of
    // the instruction, since the fixup is created at offset 16).
    static const MCFixupKindInfo Infos[] = {
      // Name               TargetOffset  TargetSize  Flags
      { "fixup_NPU_branch",      0,           64,       0 },
    };
    static_assert(std::size(Infos) == NPU::NumTargetFixupKinds,
                  "NPU fixup table size mismatch");

    if (Kind < FirstTargetFixupKind)
      return MCAsmBackend::getFixupKindInfo(Kind);

    unsigned Idx = (unsigned)Kind - FirstTargetFixupKind;
    assert(Idx < NPU::NumTargetFixupKinds && "invalid NPU fixup kind");
    return Infos[Idx];
  }

  // ── Fixup application ─────────────────────────────────────────────────────

  // Called by the assembler for every fixup — resolved or not.
  // For unresolved (external) symbols, maybeAddReloc() triggers
  // recordRelocation() on the ELF writer so that an RELA entry is emitted.
  //
  // Data already points to the imm field (offset 16 within the 32-byte word)
  // because the fixup was created with MCFixup::create(16, ...).
  //
  // Value = absolute byte address of the branch target in the section.
  // Instruction index = Value / 32  (instructions are 32 bytes each).
  void applyFixup(const MCFragment &F, const MCFixup &Fixup,
                  const MCValue &Target, uint8_t *Data, uint64_t Value,
                  bool IsResolved) override {
    // Emit an ELF RELA record for every unresolved fixup.
    maybeAddReloc(F, Fixup, Target, Value, IsResolved);

    MCFixupKind Kind = Fixup.getKind();
    if ((unsigned)Kind == NPU::fixup_NPU_branch) {
      // For resolved fixups, write the instruction index (Value / 32).
      // For unresolved ones the linker fills in the imm field; write 0.
      int64_t AbsInstrIdx = IsResolved
                                ? static_cast<int64_t>(Value) / 32
                                : 0;
      uint64_t LE = static_cast<uint64_t>(AbsInstrIdx);
      memcpy(Data, &LE, 8);
      return;
    }
    // Generic data fixups (FK_Data_4, FK_Data_8) — patch the raw bytes.
    if (Kind == FK_Data_4) {
      uint32_t V = static_cast<uint32_t>(Value);
      memcpy(Data, &V, 4);
    } else if (Kind == FK_Data_8) {
      memcpy(Data, &Value, 8);
    }
  }

  // No instruction relaxation needed for the fixed-width NPU ISA.
  bool mayNeedRelaxation(unsigned /*Opcode*/, ArrayRef<MCOperand> /*Ops*/,
                         const MCSubtargetInfo &) const override {
    return false;
  }

  void relaxInstruction(MCInst &, const MCSubtargetInfo &) const override {}

  // Emit zero-fill bytes for alignment gaps.
  bool writeNopData(raw_ostream &OS, uint64_t Count,
                    const MCSubtargetInfo *) const override {
    for (uint64_t I = 0; I < Count; ++I)
      OS.write('\0');
    return true;
  }

  std::unique_ptr<MCObjectTargetWriter>
  createObjectTargetWriter() const override {
    return createNPUELFObjectWriter();
  }
};

} // namespace

namespace llvm {
MCAsmBackend *createNPUMCAsmBackend(const Target &,
                                    const MCSubtargetInfo &,
                                    const MCRegisterInfo &,
                                    const MCTargetOptions &) {
  return new NPUMCAsmBackend();
}
} // namespace llvm
