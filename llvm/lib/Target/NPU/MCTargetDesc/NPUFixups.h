//===-- NPUFixups.h - NPU-specific fixup kinds ----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// NPU fixup kinds for branch / call targets.
//
// NPU instructions are 32 bytes each. Branch targets are stored as ABSOLUTE
// instruction indices in the 64-bit imm field (bytes 16–23 of the word).
//
// Fixup layout for fixup_NPU_branch:
//   - The fixup is created at offset 16 within the 32-byte instruction word,
//     pointing directly at the 8-byte imm field.
//   - applyFixup writes:  *(i64 at Data) = target_byte_address / 32
//   - ELF RELA record:    R_NPU_BRANCH, r_offset = imm-field byte in section
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_NPU_MCTARGETDESC_NPUFIXUPS_H
#define LLVM_LIB_TARGET_NPU_MCTARGETDESC_NPUFIXUPS_H

#include "llvm/MC/MCFixup.h"

namespace llvm {
namespace NPU {

enum Fixups {
  // Branch / JAL target: the imm field is written with (S + A) / 32.
  // The fixup is at offset +16 within the 32-byte instruction word so that
  // the ELF r_offset points directly at the field being patched.
  fixup_NPU_branch = FirstTargetFixupKind,

  NumTargetFixupKinds = fixup_NPU_branch - FirstTargetFixupKind + 1,
};

} // namespace NPU
} // namespace llvm

#endif // LLVM_LIB_TARGET_NPU_MCTARGETDESC_NPUFIXUPS_H
