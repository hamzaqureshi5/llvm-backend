//===-- NPUMCCodeEmitter.cpp - NPU Binary Instruction Emitter -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Emits NPU instructions as 32-byte (NPU_WORD_BYTES) binary records, matching
// the encoding used by npu_encode_cinst() and npu_encode_vinst() in the
// simulator (npu-simulator/npu_asm.c).
//
// Encoding map (cinst, stream type byte = 0):
//   [0]   stream type (0)
//   [1]   cop_t opcode
//   [2]   rd  (destination GPR)
//   [3]   rs1 (source GPR 1)
//   [4]   rs2 (source GPR 2)
//   [5]   dt  (destination tile register, tensor engine)
//   [6]   a   (source tile register A, tensor engine)
//   [7]   b   (source tile register B, tensor engine)
//   [8]   M   (tile dimension M)
//   [9]   K   (tile dimension K)
//   [10]  N   (tile dimension N)
//   [11]  use_base (1 if address = x[rs1] + imm)
//   [12–15] count (u32 LE, element/byte count for loads/stores/DMA)
//   [16–23] imm (i64 LE) or fimm (double IEEE 754, for SCALE)
//   [24–31] padding
//
// Encoding map (vinst, stream type byte = 1):
//   [0]   stream type (1)
//   [1]   vop_t opcode
//   [2]   vd  (destination vector register)
//   [3]   vs1 (source vector register 1)
//   [4]   vs2 (source vector register 2)
//   [5]   vs  (source vector register, unary ops)
//   [6]   rd  (destination engine scalar)
//   [7]   rs  (source engine scalar, base for load/store)
//   [8]   rs1 (engine scalar operand 1)
//   [9]   rs2 (engine scalar operand 2)
//   [10]  md  (destination mask register)
//   [11]  m1  (source mask register 1)
//   [12]  m2  (source mask register 2)
//   [13]  use_mask (1 = @m predicated)
//   [14]  mask (predicate mask register index)
//   [15]  use_base (1 = address uses base register)
//   [16–23] imm (i64 LE) or fimm (double)
//   [24]  cmp  (cmpkind_t: LT=0, LE=1, EQ=2, NE=3, GT=4, GE=5)
//   [25]  cvt  (cvtmode_t: F32=0, I8=1, I32=2)
//   [26–31] padding
//
//===----------------------------------------------------------------------===//

#include "NPUMCCodeEmitter.h"
#include "NPUFixups.h"
#include "MCTargetDesc/NPUMCTargetDesc.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCFixup.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/raw_ostream.h"
#include <cstring>

// NPU::ADD, NPU::VADD, NPU::X0, NPU::V0, etc. are already defined by the
// GET_INSTRINFO_ENUM / GET_REGINFO_ENUM includes in NPUMCTargetDesc.h which is
// pulled in transitively via NPUMCCodeEmitter.h → MCTargetDesc/NPUMCTargetDesc.h.
// Do NOT re-include the generated .inc files here — the .inc files use an enum
// definition that cannot be opened twice in the same translation unit.

using namespace llvm;

#define DEBUG_TYPE "npu-mccodeemitter"

// ── NPU simulator opcode numbers ─────────────────────────────────────────────
// These match cop_t / vop_t in npu.h exactly (order matters).
// Keeping them here avoids pulling in the simulator header.

namespace cop {
enum : uint8_t {
  ADD=0, SUB=1, ADDI=2, MUL=3, AND=4, OR=5, XOR=6, SLL=7, SRL=8, SLT=9,
  LI=10, LW=11, LH=12, LB=13, SW=14, SH=15, SB=16,
  BEQ=17, BNE=18, BLT=19, BGE=20, JAL=21, JALR=22,
  KLAUNCH=23, KWAIT=24, HALT=25,
  // Tensor engine:
  T_CFG_SHAPE=26, T_CFG_DTYPE=27, T_CFG_STRIDE=28, T_CFG_MODE=29, T_CFG_RND=30,
  T_MATMUL=31, T_MMA=32, T_DOT=33, T_CONV=34, T_ACC_CLR=35,
  T_LOAD=36, T_STORE=37, T_LOADW=38, T_DMA_IN=39, T_DMA_OUT=40, T_BCAST=41,
  T_RELU=42, T_QUANT=43, T_DEQUANT=44, T_SCALE=45, T_POOL=46,
  T_FENCE=47, T_WAIT=48, T_BARRIER=49, T_STATUS=50
};
} // namespace cop

namespace vop {
enum : uint8_t {
  VSETVL=0, VLOAD=1, VSTORE=2, VGATHER=3, VSCATTER=4, VMOV=5, VBCAST=6,
  SLI=7, SFI=8, SADD=9, SADDI=10, SSUB=11, SMUL=12, SMULI=13,
  SLW=14, SLF=15, SRECIP=16, KID=17,
  VADD=18, VSUB=19, VMUL=20, VDIV=21, VMAX=22, VMIN=23, VFMA=24,
  VCMP=25, VSEL=26, VEXP=27, VLOG=28, VRSQRT=29, VSQRT=30, VRECIP=31,
  VCVT=32, VABS=33, VNEG=34,
  VREDSUM=35, VREDMAX=36, VREDMIN=37, VREDAND=38, VREDOR=39, VSCAN=40,
  VMASK_SET=41, VMASK_AND=42, VMASK_OR=43, VMASK_NOT=44,
  BR=45, BRZ=46, BRNZ=47, LOOP=48, KEXIT=49
};
} // namespace vop

// ── Helper: extract LLVM physical register number (0-based) ──────────────────
// NPU::X0 = first GPR, NPU::V0 = first vector register, etc.
// The LLVM register enum has a gap at 0 (NoReg); physical regs start at 1.

static uint8_t gprNum(unsigned Reg) {
  // GPR: X0..X31 are numbered consecutively starting at NPU::X0.
  return (uint8_t)(Reg - NPU::X0);
}

static uint8_t vrNum(unsigned Reg) {
  return (uint8_t)(Reg - NPU::V0);
}

static uint8_t mrNum(unsigned Reg) {
  return (uint8_t)(Reg - NPU::M0);
}

static uint8_t srNum(unsigned Reg) {
  return (uint8_t)(Reg - NPU::S0);
}

// ── Encoding tables: LLVM opcode → (stream_type, sim_opcode) ─────────────────

struct OpcodeEntry {
  uint8_t StreamType;  // 0 = cinst, 1 = vinst
  uint8_t SimOpcode;   // cop_t or vop_t value
};

// Returns {stream, sim_opcode} for a given LLVM machine opcode.
// Returns {0xFF, 0xFF} for unrecognized opcodes (will crash later — intentional).
static OpcodeEntry lookupOpcode(unsigned LLVMOpc) {
  using namespace NPU;
  switch (LLVMOpc) {
  // ── Control core ─────────────────────────────────────────────────────
  case ADD:          return {0, cop::ADD};
  case SUB:          return {0, cop::SUB};
  case ADDI:         return {0, cop::ADDI};
  case MUL:          return {0, cop::MUL};
  case AND:          return {0, cop::AND};
  case OR:           return {0, cop::OR};
  case XOR:          return {0, cop::XOR};
  case SLL:          return {0, cop::SLL};
  case SRL:          return {0, cop::SRL};
  case SLT:          return {0, cop::SLT};
  case LI:           return {0, cop::LI};
  case LW:           return {0, cop::LW};
  case LH:           return {0, cop::LH};
  case LB:           return {0, cop::LB};
  case SW:           return {0, cop::SW};
  case SH:           return {0, cop::SH};
  case SB:           return {0, cop::SB};
  case BEQ:          return {0, cop::BEQ};
  case BNE:          return {0, cop::BNE};
  case BLT:          return {0, cop::BLT};
  case BGE:          return {0, cop::BGE};
  // JAL x0 (unconditional branch) and JAL x1 (call) both use C_JAL.
  case BR:
  case JAL:          return {0, cop::JAL};
  // JALR x0 (return) and JALR x1 (indirect call) both use C_JALR.
  case RET:
  case JALRcall:     return {0, cop::JALR};
  case KLAUNCH:      return {0, cop::KLAUNCH};
  case KWAIT:        return {0, cop::KWAIT};
  case HALT:         return {0, cop::HALT};
  // ── Tensor engine ─────────────────────────────────────────────────────
  case T_CFG_SHAPE:  return {0, cop::T_CFG_SHAPE};
  case T_ACC_CLR:    return {0, cop::T_ACC_CLR};
  case T_MMA:        return {0, cop::T_MMA};
  case T_MATMUL:     return {0, cop::T_MATMUL};
  case T_DOT:        return {0, cop::T_DOT};
  case T_LOAD:       return {0, cop::T_LOAD};
  case T_LOADW:      return {0, cop::T_LOADW};
  case T_STORE:      return {0, cop::T_STORE};
  case T_DMA_IN:     return {0, cop::T_DMA_IN};
  case T_DMA_OUT:    return {0, cop::T_DMA_OUT};
  case T_RELU:       return {0, cop::T_RELU};
  case T_SCALE:      return {0, cop::T_SCALE};
  case T_FENCE:      return {0, cop::T_FENCE};
  case T_WAIT:       return {0, cop::T_WAIT};
  case T_BARRIER:    return {0, cop::T_BARRIER};
  // ── Vector core ───────────────────────────────────────────────────────
  case V_VSETVL:     return {1, vop::VSETVL};
  case V_VLOAD:      return {1, vop::VLOAD};
  case V_VSTORE:     return {1, vop::VSTORE};
  case V_VGATHER:    return {1, vop::VGATHER};
  case V_VSCATTER:   return {1, vop::VSCATTER};
  case V_VMOV:       return {1, vop::VMOV};
  case V_VBCAST:     return {1, vop::VBCAST};
  case V_SLI:        return {1, vop::SLI};
  case V_SFI:        return {1, vop::SFI};
  case V_SADD:       return {1, vop::SADD};
  case V_SADDI:      return {1, vop::SADDI};
  case V_SSUB:       return {1, vop::SSUB};
  case V_SMUL:       return {1, vop::SMUL};
  case V_SMULI:      return {1, vop::SMULI};
  case V_SLW:        return {1, vop::SLW};
  case V_SLF:        return {1, vop::SLF};
  case V_SRECIP:     return {1, vop::SRECIP};
  case V_KID:        return {1, vop::KID};
  case V_VADD:       return {1, vop::VADD};
  case V_VSUB:       return {1, vop::VSUB};
  case V_VMUL:       return {1, vop::VMUL};
  case V_VDIV:       return {1, vop::VDIV};
  case V_VMAX:       return {1, vop::VMAX};
  case V_VMIN:       return {1, vop::VMIN};
  case V_VFMA:       return {1, vop::VFMA};
  case V_VCMP:       return {1, vop::VCMP};
  case V_VSEL:       return {1, vop::VSEL};
  case V_VEXP:       return {1, vop::VEXP};
  case V_VLOG:       return {1, vop::VLOG};
  case V_VRSQRT:     return {1, vop::VRSQRT};
  case V_VSQRT:      return {1, vop::VSQRT};
  case V_VRECIP:     return {1, vop::VRECIP};
  case V_VCVT:       return {1, vop::VCVT};
  case V_VABS:       return {1, vop::VABS};
  case V_VNEG:       return {1, vop::VNEG};
  case V_VREDSUM:    return {1, vop::VREDSUM};
  case V_VREDMAX:    return {1, vop::VREDMAX};
  case V_VREDMIN:    return {1, vop::VREDMIN};
  case V_VREDAND:    return {1, vop::VREDAND};
  case V_VREDOR:     return {1, vop::VREDOR};
  case V_VSCAN:      return {1, vop::VSCAN};
  case V_VMASK_SET:  return {1, vop::VMASK_SET};
  case V_VMASK_AND:  return {1, vop::VMASK_AND};
  case V_VMASK_OR:   return {1, vop::VMASK_OR};
  case V_VMASK_NOT:  return {1, vop::VMASK_NOT};
  case V_BR:         return {1, vop::BR};
  case V_BRZ:        return {1, vop::BRZ};
  case V_BRNZ:       return {1, vop::BRNZ};
  case V_LOOP:       return {1, vop::LOOP};
  case V_KEXIT:      return {1, vop::KEXIT};
  default:
    return {0xFF, 0xFF};
  }
}

// ── Main encoding entry point ─────────────────────────────────────────────────

void NPUMCCodeEmitter::encodeInstruction(const MCInst &MI,
                                         SmallVectorImpl<char> &CB,
                                         SmallVectorImpl<MCFixup> &Fixups,
                                         const MCSubtargetInfo &STI) const {
  // Allocate the 32-byte record (NPU_WORD_BYTES), zero-initialized.
  uint8_t Word[32] = {};

  unsigned LLVMOpc = MI.getOpcode();
  OpcodeEntry OE = lookupOpcode(LLVMOpc);

  // Pseudo instructions (ADJCALLSTACKDOWN/UP, SLTU, etc.) are eliminated
  // before emission; reaching here with an unknown opcode is a compiler bug.
  if (OE.StreamType == 0xFF) {
    llvm_unreachable("NPUMCCodeEmitter: unhandled opcode in encodeInstruction");
  }

  Word[0] = OE.StreamType;
  Word[1] = OE.SimOpcode;

  using namespace NPU;

  if (OE.StreamType == 0) {
    // ── Control / tensor instruction encoding ───────────────────────────
    // Extract operands by opcode category.  The operand layout in each
    // MCInst matches the InOperandList defined in the TableGen .td files.

    switch (LLVMOpc) {
    // RRR: rd=op[0], rs1=op[1], rs2=op[2]
    case ADD: case SUB: case MUL: case AND: case OR: case XOR:
    case SLL: case SRL: case SLT:
      Word[2] = gprNum(MI.getOperand(0).getReg());
      Word[3] = gprNum(MI.getOperand(1).getReg());
      Word[4] = gprNum(MI.getOperand(2).getReg());
      break;

    // RRI: rd=op[0], rs1=op[1], imm=op[2]
    case ADDI:
      Word[2] = gprNum(MI.getOperand(0).getReg());
      Word[3] = gprNum(MI.getOperand(1).getReg());
      putI64LE(Word + 16, MI.getOperand(2).getImm());
      break;

    // RI: rd=op[0], imm=op[1]
    case LI:
      Word[2] = gprNum(MI.getOperand(0).getReg());
      putI64LE(Word + 16, MI.getOperand(1).getImm());
      break;

    // LOAD: rd=op[0], rs1=op[1] (base), imm=op[2] (offset).
    // use_base=1 to tell the simulator to compute addr = x[rs1] + imm.
    case LW: case LH: case LB:
      Word[2]  = gprNum(MI.getOperand(0).getReg());
      Word[3]  = gprNum(MI.getOperand(1).getReg());  // base
      Word[11] = 1;                                   // use_base
      putI64LE(Word + 16, MI.getOperand(2).getImm()); // offset
      break;

    // STORE: rs2=op[0] (value), rs1=op[1] (base), imm=op[2] (offset).
    case SW: case SH: case SB:
      Word[4]  = gprNum(MI.getOperand(0).getReg());  // rs2 = value reg
      Word[3]  = gprNum(MI.getOperand(1).getReg());  // rs1 = base
      Word[11] = 1;
      putI64LE(Word + 16, MI.getOperand(2).getImm());
      break;

    // BRANCH (conditional): rs1=op[0], rs2=op[1], target=op[2].
    case BEQ: case BNE: case BLT: case BGE: {
      Word[3] = gprNum(MI.getOperand(0).getReg());  // rs1
      Word[4] = gprNum(MI.getOperand(1).getReg());  // rs2
      const MCOperand &Target = MI.getOperand(2);
      if (Target.isImm())
        putI64LE(Word + 16, Target.getImm());
      else if (Target.isExpr())
        // Fixup at offset 16 (the imm field) so r_offset in RELA points
        // directly at the bytes being patched. Non-PCRel: value = absolute
        // target byte address; applyFixup divides by 32 to get instr index.
        Fixups.push_back(MCFixup::create(
            16, Target.getExpr(),
            MCFixupKind(NPU::fixup_NPU_branch)));
      break;
    }

    // JAL (unconditional branch) / BR pseudo: target=op[0].
    // JAL for calls: target=op[0] (calltarget operand, no explicit rd in MCInst).
    case BR: case JAL: {
      if (MI.getNumOperands() == 0) break;
      const MCOperand &Target = MI.getOperand(0);
      if (Target.isImm())
        putI64LE(Word + 16, Target.getImm());
      else if (Target.isExpr())
        Fixups.push_back(MCFixup::create(
            16, Target.getExpr(),
            MCFixupKind(NPU::fixup_NPU_branch)));
      break;
    }

    // JALR (indirect call / return): rs1=op[0] (return target or callee ptr).
    case RET:
      Word[3] = gprNum(NPU::X1);  // return address in x1
      break;
    case JALRcall:
      Word[3] = gprNum(MI.getOperand(0).getReg());
      break;

    // KLAUNCH: rs1=op[0] (kernel id register), imm=op[1] (grid size).
    case KLAUNCH:
      Word[3] = gprNum(MI.getOperand(0).getReg());
      putI64LE(Word + 16, MI.getOperand(1).getImm());
      break;

    // KWAIT, HALT, FENCE, WAIT, BARRIER: no operands.
    case KWAIT: case HALT:
    case T_FENCE: case T_WAIT: case T_BARRIER:
      break;

    // CFG.SHAPE: M=op[0], K=op[1], N=op[2].
    case T_CFG_SHAPE:
      Word[8] = (uint8_t)MI.getOperand(0).getImm();  // M
      Word[9] = (uint8_t)MI.getOperand(1).getImm();  // K
      Word[10]= (uint8_t)MI.getOperand(2).getImm();  // N
      break;

    // ACC.CLR: dt=op[0] (tile register number).
    case T_ACC_CLR:
      Word[5] = (uint8_t)MI.getOperand(0).getImm();  // dt
      break;

    // Tile three-operand: dt, a, b.
    case T_MMA: case T_MATMUL: case T_DOT:
      Word[5] = (uint8_t)MI.getOperand(0).getImm();  // dt
      Word[6] = (uint8_t)MI.getOperand(1).getImm();  // a
      Word[7] = (uint8_t)MI.getOperand(2).getImm();  // b
      break;

    // Tile two-operand: dt, a.
    case T_RELU:
      Word[5] = (uint8_t)MI.getOperand(0).getImm();  // dt
      Word[6] = (uint8_t)MI.getOperand(1).getImm();  // a
      break;

    // Tile SCALE: dt, a, factor (float immediate).
    case T_SCALE: {
      Word[5] = (uint8_t)MI.getOperand(0).getImm();
      Word[6] = (uint8_t)MI.getOperand(1).getImm();
      // Double immediate stored as raw IEEE 754 bits in bytes 16–23.
      // getDFPImm() returns the bit pattern as uint64 (LLVM renamed getFPImm).
      uint64_t bits = MI.getOperand(2).getDFPImm();
      memcpy(Word + 16, &bits, 8);
      break;
    }

    // Tensor LOAD / LOADW: dt=op[0], base=op[1], count=op[2].
    case T_LOAD: case T_LOADW:
      Word[5]  = (uint8_t)MI.getOperand(0).getImm();     // dt
      Word[3]  = gprNum(MI.getOperand(1).getReg());       // rs1 = base
      Word[11] = 1;
      putU32LE(Word + 12, (uint32_t)MI.getOperand(2).getImm()); // count
      break;

    // Tensor STORE: base=op[0], dt=op[1], count=op[2].
    case T_STORE:
      Word[3]  = gprNum(MI.getOperand(0).getReg());
      Word[5]  = (uint8_t)MI.getOperand(1).getImm();
      Word[11] = 1;
      putU32LE(Word + 12, (uint32_t)MI.getOperand(2).getImm());
      break;

    // DMA IN/OUT: dst=op[0], src=op[1], bytes=op[2].
    case T_DMA_IN: case T_DMA_OUT:
      Word[2] = gprNum(MI.getOperand(0).getReg());        // rd = dst
      Word[3] = gprNum(MI.getOperand(1).getReg());        // rs1 = src
      putU32LE(Word + 12, (uint32_t)MI.getOperand(2).getImm());
      break;

    default:
      break;
    }

  } else {
    // ── Vector instruction encoding ──────────────────────────────────────
    switch (LLVMOpc) {
    // VVV: vd, vs1, vs2.
    case V_VADD: case V_VSUB: case V_VMUL: case V_VDIV:
    case V_VMAX: case V_VMIN:
      Word[2] = vrNum(MI.getOperand(0).getReg());
      Word[3] = vrNum(MI.getOperand(1).getReg());
      Word[4] = vrNum(MI.getOperand(2).getReg());
      break;

    // VFMA: vd (= vs) / vs1 / vs2.  The tied register is op[0]/op[3].
    case V_VFMA:
      Word[2] = vrNum(MI.getOperand(0).getReg());  // vd
      Word[3] = vrNum(MI.getOperand(1).getReg());  // vs1
      Word[4] = vrNum(MI.getOperand(2).getReg());  // vs2
      // op[3] is the tied accumulator input (same as op[0]).
      break;

    // VV (unary): vd, vs.
    case V_VABS: case V_VNEG: case V_VMOV:
    case V_VEXP: case V_VLOG: case V_VRSQRT: case V_VSQRT: case V_VRECIP:
    case V_VSCAN:
      Word[2] = vrNum(MI.getOperand(0).getReg());
      Word[5] = vrNum(MI.getOperand(1).getReg());
      break;

    // VCVT: vd, vs, cvt-kind.
    case V_VCVT:
      Word[2]  = vrNum(MI.getOperand(0).getReg());
      Word[5]  = vrNum(MI.getOperand(1).getReg());
      Word[25] = (uint8_t)MI.getOperand(2).getImm(); // cvt
      break;

    // VCMP: md, vs1, vs2, cmp-kind.
    case V_VCMP:
      Word[10] = mrNum(MI.getOperand(0).getReg());
      Word[3]  = vrNum(MI.getOperand(1).getReg());
      Word[4]  = vrNum(MI.getOperand(2).getReg());
      Word[24] = (uint8_t)MI.getOperand(3).getImm(); // cmp kind
      break;

    // VSEL: vd, vs1, vs2, mask.
    case V_VSEL:
      Word[2]  = vrNum(MI.getOperand(0).getReg());
      Word[3]  = vrNum(MI.getOperand(1).getReg());
      Word[4]  = vrNum(MI.getOperand(2).getReg());
      Word[13] = 1;
      Word[14] = mrNum(MI.getOperand(3).getReg()); // mask
      break;

    // VBCAST: vd, rs.
    case V_VBCAST:
      Word[2] = vrNum(MI.getOperand(0).getReg());
      Word[7] = srNum(MI.getOperand(1).getReg());
      break;

    // VSETVL: rd, rs.
    case V_VSETVL:
      Word[6] = srNum(MI.getOperand(0).getReg());
      Word[7] = srNum(MI.getOperand(1).getReg());
      break;

    // VLOAD: vd, rs (base), imm (stride).
    case V_VLOAD: case V_VGATHER:
      Word[2]  = vrNum(MI.getOperand(0).getReg());
      Word[7]  = srNum(MI.getOperand(1).getReg());
      Word[15] = 1;
      putI64LE(Word + 16, MI.getOperand(2).getImm());
      break;

    // VSTORE: vs, rs (base), imm (stride).
    case V_VSTORE: case V_VSCATTER:
      Word[5]  = vrNum(MI.getOperand(0).getReg());
      Word[7]  = srNum(MI.getOperand(1).getReg());
      Word[15] = 1;
      putI64LE(Word + 16, MI.getOperand(2).getImm());
      break;

    // Reductions: rd (engine scalar), vs.
    case V_VREDSUM: case V_VREDMAX: case V_VREDMIN:
    case V_VREDAND: case V_VREDOR:
      Word[6] = srNum(MI.getOperand(0).getReg());
      Word[5] = vrNum(MI.getOperand(1).getReg());
      break;

    // Mask ops: MMM.
    case V_VMASK_AND: case V_VMASK_OR:
      Word[10] = mrNum(MI.getOperand(0).getReg());
      Word[11] = mrNum(MI.getOperand(1).getReg());
      Word[12] = mrNum(MI.getOperand(2).getReg());
      break;

    // Mask ops: MM.
    case V_VMASK_NOT:
      Word[10] = mrNum(MI.getOperand(0).getReg());
      Word[11] = mrNum(MI.getOperand(1).getReg());
      break;

    // Mask set: md, vs.
    case V_VMASK_SET:
      Word[10] = mrNum(MI.getOperand(0).getReg());
      Word[5]  = vrNum(MI.getOperand(1).getReg());
      break;

    // Engine scalar: SLI rd, imm.
    case V_SLI:
      Word[6] = srNum(MI.getOperand(0).getReg());
      putI64LE(Word + 16, MI.getOperand(1).getImm());
      break;

    // Engine scalar: SFI rd, fimm.
    case V_SFI: {
      Word[6] = srNum(MI.getOperand(0).getReg());
      uint64_t bits = MI.getOperand(1).getDFPImm();
      memcpy(Word + 16, &bits, 8);
      break;
    }

    // Engine scalar: SADD/SSUB/SMUL rd, rs1, rs2.
    case V_SADD: case V_SSUB: case V_SMUL:
      Word[6] = srNum(MI.getOperand(0).getReg());
      Word[8] = srNum(MI.getOperand(1).getReg());
      Word[9] = srNum(MI.getOperand(2).getReg());
      break;

    // Engine scalar: SADDI/SMULI rd, rs, imm.
    case V_SADDI: case V_SMULI:
      Word[6] = srNum(MI.getOperand(0).getReg());
      Word[7] = srNum(MI.getOperand(1).getReg());
      putI64LE(Word + 16, MI.getOperand(2).getImm());
      break;

    // Engine scalar loads: rd, imm(rs).
    case V_SLW: case V_SLF:
      Word[6]  = srNum(MI.getOperand(0).getReg());
      Word[7]  = srNum(MI.getOperand(1).getReg());
      Word[15] = 1;
      putI64LE(Word + 16, MI.getOperand(2).getImm());
      break;

    // SRECIP: rd, rs.
    case V_SRECIP:
      Word[6] = srNum(MI.getOperand(0).getReg());
      Word[7] = srNum(MI.getOperand(1).getReg());
      break;

    // KID: rd.
    case V_KID:
      Word[6] = srNum(MI.getOperand(0).getReg());
      break;

    // Vector branch: target only.
    case V_BR:
      if (MI.getOperand(0).isImm())
        putI64LE(Word + 16, MI.getOperand(0).getImm());
      break;

    // Conditional vector branches: rs, target.
    case V_BRZ: case V_BRNZ: case V_LOOP:
      Word[7] = srNum(MI.getOperand(0).getReg());
      if (MI.getOperand(1).isImm())
        putI64LE(Word + 16, MI.getOperand(1).getImm());
      break;

    // KEXIT: no operands.
    case V_KEXIT:
      break;

    default:
      break;
    }
  }

  // Append the 32-byte word to the output buffer.
  CB.append(reinterpret_cast<char *>(Word),
            reinterpret_cast<char *>(Word) + 32);
}

// ── Factory ───────────────────────────────────────────────────────────────────

MCCodeEmitter *llvm::createNPUMCCodeEmitter(const MCInstrInfo &MCII,
                                             MCContext &Ctx) {
  return new NPUMCCodeEmitter(MCII, Ctx);
}
