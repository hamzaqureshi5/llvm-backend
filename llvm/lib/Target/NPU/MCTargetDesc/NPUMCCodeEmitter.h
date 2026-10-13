//===-- NPUMCCodeEmitter.h - NPU Binary Instruction Emitter -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Converts an MCInst (one NPU instruction) into the 32-byte binary encoding
// defined by the NPU simulator (npu_asm.h / npu_asm.c :: npu_encode_cinst /
// npu_encode_vinst).
//
// Binary record layout (NPU_WORD_BYTES = 32 bytes, little-endian):
//
//   Byte 0   : stream type   (0 = cinst / control+tensor, 1 = vinst / vector)
//   Byte 1   : opcode        (cop_t or vop_t enum value from npu.h)
//   Bytes 2–4: rd, rs1, rs2  (control-core GPR numbers) OR
//              vd, vs1, vs2  (vector register numbers)
//   …        : remaining operand fields (see encoding tables in .cpp)
//   Bytes 16–23: imm (i64 little-endian) or fimm (double IEEE 754)
//   Bytes 24–31: cmp/cvt kind + padding
//
// All multi-byte integers are stored little-endian to match the simulator's
// put_u32() / put_i64() helpers.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_NPU_MCTARGETDESC_NPUMCCODEEMITTER_H
#define LLVM_LIB_TARGET_NPU_MCTARGETDESC_NPUMCCODEEMITTER_H

#include "llvm/MC/MCCodeEmitter.h"
#include <cstdint>

namespace llvm {

class MCContext;
class MCInst;
class MCInstrInfo;
class MCRegisterInfo;
class MCSubtargetInfo;
class raw_ostream;

/// NPUMCCodeEmitter – emits the NPU 32-byte binary instruction record.
///
/// Each instruction is written as one contiguous NPU_WORD_BYTES (32) blob.
/// The stream-type byte (byte 0) distinguishes control/tensor instructions
/// from vector-core instructions, matching the format that npu_decode_cinst()
/// and npu_decode_vinst() expect.
class NPUMCCodeEmitter : public MCCodeEmitter {
  const MCInstrInfo &MCII;
  MCContext &Ctx;

public:
  NPUMCCodeEmitter(const MCInstrInfo &MCII, MCContext &Ctx)
      : MCII(MCII), Ctx(Ctx) {}

  ~NPUMCCodeEmitter() override = default;

  // Emit the 32-byte binary encoding of MI into OS.
  void encodeInstruction(const MCInst &MI, SmallVectorImpl<char> &CB,
                         SmallVectorImpl<MCFixup> &Fixups,
                         const MCSubtargetInfo &STI) const override;

private:
  // Write a 32-bit unsigned integer in little-endian byte order.
  static void putU32LE(uint8_t *out, uint32_t v) {
    out[0] = v & 0xFF;
    out[1] = (v >> 8) & 0xFF;
    out[2] = (v >> 16) & 0xFF;
    out[3] = (v >> 24) & 0xFF;
  }

  // Write a 64-bit signed integer in little-endian byte order.
  static void putI64LE(uint8_t *out, int64_t v) {
    for (int i = 0; i < 8; ++i)
      out[i] = (v >> (8 * i)) & 0xFF;
  }
};

/// Factory called from NPUMCTargetDesc.cpp :: LLVMInitializeNPUTargetMC().
MCCodeEmitter *createNPUMCCodeEmitter(const MCInstrInfo &MCII,
                                      MCContext &Ctx);

} // namespace llvm

#endif // LLVM_LIB_TARGET_NPU_MCTARGETDESC_NPUMCCODEEMITTER_H
