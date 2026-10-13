//===-- NPUAsmParser.cpp - NPU assembly parser ----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Hand-written assembly parser for the NPU ISA.  Accepts all three layers of
// the NPU instruction set in the syntax produced by the simulator's disassembler
// (npu_asm.h :: npu_disasm_cinst / npu_disasm_vinst), so assembly files can be
// round-tripped through: LLVM IR → llc → .s → NPUAsmParser → object file.
//
// Register naming conventions (matching the simulator's CTAB/VTAB):
//   Control core GPRs : x0..x31
//   Vector registers  : v0..v31
//   Mask registers    : m0..m7
//   Engine scalars    : s0..s31
//   Tile registers    : t0..t7  (written as integer immediates in the assembler)
//
// Supported operand syntax:
//   Registers   : x3, v12, m2, s7
//   Immediates  : 42, -7, 0xFF, 1.5  (float)
//   Memory      : 8(x2)              (immediate offset + base register)
//   Labels      : .Lbb0, func_name
//   Tile numbers: t0..t7             (resolved to integer 0–7)
//   Predication : @m2 VADD v0, v1, v2  (mask prefix on vector instructions)
//
// This is a hand-written parser (not TableGen-generated) so that we can
// handle the three register namespaces and predication prefix without
// ambiguity.  The key entry points are:
//   parseInstruction()          — parses one instruction line
//   matchAndEmitInstruction()   — matches operands and emits MCInst
//
//===----------------------------------------------------------------------===//

#include "MCTargetDesc/NPUMCTargetDesc.h"
#include "TargetInfo/NPUTargetInfo.h"
#include "llvm/ADT/StringRef.h"
#include <cstring>
#include "llvm/ADT/StringSwitch.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCParser/MCAsmParser.h"
#include "llvm/MC/MCParser/MCParsedAsmOperand.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCParser/MCTargetAsmParser.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/SMLoc.h"

// NPU::ADD, NPU::X0, etc. are already defined by NPUMCTargetDesc.h (included
// above) via its GET_INSTRINFO_ENUM / GET_REGINFO_ENUM sections.  Do NOT
// re-define those macros and re-include the .inc files here — the .inc files
// use enum definitions that cannot be opened twice in one translation unit.

using namespace llvm;

#define DEBUG_TYPE "npu-asm-parser"

namespace {

// ── Parsed operand representation ─────────────────────────────────────────────

struct NPUOperand : public MCParsedAsmOperand {
  enum class KindTy { Token, Reg, Imm, Mem, FPImm, Expr };

  KindTy Kind;
  SMLoc StartLoc, EndLoc;

  struct MemOp { unsigned Base; int64_t Offset; };

  union {
    StringRef       Tok;
    unsigned        Reg;
    int64_t         Imm;
    double          FPImm;
    MemOp           Mem;
    const MCExpr   *Expr;
  };

  NPUOperand(KindTy K, SMLoc S, SMLoc E) : Kind(K), StartLoc(S), EndLoc(E) {}

  static std::unique_ptr<NPUOperand> createToken(StringRef T, SMLoc S) {
    auto O = std::make_unique<NPUOperand>(KindTy::Token, S, S);
    O->Tok = T;
    return O;
  }
  static std::unique_ptr<NPUOperand> createReg(unsigned R, SMLoc S, SMLoc E) {
    auto O = std::make_unique<NPUOperand>(KindTy::Reg, S, E);
    O->Reg = R;
    return O;
  }
  static std::unique_ptr<NPUOperand> createImm(int64_t V, SMLoc S, SMLoc E) {
    auto O = std::make_unique<NPUOperand>(KindTy::Imm, S, E);
    O->Imm = V;
    return O;
  }
  static std::unique_ptr<NPUOperand> createFPImm(double V, SMLoc S, SMLoc E) {
    auto O = std::make_unique<NPUOperand>(KindTy::FPImm, S, E);
    O->FPImm = V;
    return O;
  }
  static std::unique_ptr<NPUOperand> createMem(unsigned Base, int64_t Off,
                                               SMLoc S, SMLoc E) {
    auto O = std::make_unique<NPUOperand>(KindTy::Mem, S, E);
    O->Mem = {Base, Off};
    return O;
  }
  static std::unique_ptr<NPUOperand> createExpr(const MCExpr *E,
                                                SMLoc S, SMLoc End) {
    auto O = std::make_unique<NPUOperand>(KindTy::Expr, S, End);
    O->Expr = E;
    return O;
  }

  bool isToken()  const override { return Kind == KindTy::Token; }
  bool isReg()    const override { return Kind == KindTy::Reg;   }
  bool isImm()    const override { return Kind == KindTy::Imm || Kind == KindTy::Expr; }
  bool isMem()    const override { return Kind == KindTy::Mem;   }
  bool isFPImm()  const { return Kind == KindTy::FPImm; }
  bool isExprOp() const { return Kind == KindTy::Expr;  }

  StringRef getToken() const { assert(isToken()); return Tok; }
  MCRegister getReg() const override { return Reg; }
  SMLoc getStartLoc() const override { return StartLoc; }
  SMLoc getEndLoc()   const override { return EndLoc;   }

  void addRegOperands(MCInst &Inst, unsigned N) const {
    assert(N == 1 && isReg());
    Inst.addOperand(MCOperand::createReg(Reg));
  }
  void addImmOperands(MCInst &Inst, unsigned N) const {
    assert(N == 1 && isImm());
    Inst.addOperand(MCOperand::createImm(Imm));
  }
  void addFPImmOperands(MCInst &Inst, unsigned N) const {
    assert(N == 1 && isFPImm());
    // MCOperand stores FP as raw bits; use createDFPImm for double precision.
    uint64_t Bits;
    memcpy(&Bits, &FPImm, sizeof(Bits));
    Inst.addOperand(MCOperand::createDFPImm(Bits));
  }
  void addMemOperands(MCInst &Inst, unsigned N) const {
    // Memory operands are expanded to (base_reg, offset_imm).
    assert(N == 2 && isMem());
    Inst.addOperand(MCOperand::createReg(Mem.Base));
    Inst.addOperand(MCOperand::createImm(Mem.Offset));
  }

  void print(raw_ostream &OS, const MCAsmInfo &) const override {
    switch (Kind) {
    case KindTy::Token: OS << "Tok:" << Tok;        break;
    case KindTy::Reg:   OS << "Reg:" << Reg;        break;
    case KindTy::Imm:   OS << "Imm:" << Imm;        break;
    case KindTy::FPImm: OS << "FP:" << FPImm;       break;
    case KindTy::Mem:   OS << "Mem:" << Mem.Offset
                           << "(" << Mem.Base << ")"; break;
    case KindTy::Expr:  OS << "Expr"; break;
    }
  }
};

// ── Parser class ──────────────────────────────────────────────────────────────

class NPUAsmParser : public MCTargetAsmParser {
  MCAsmParser &Parser;

  // Registers the opcode for a matched instruction.
  bool parseRegister(MCRegister &Reg, SMLoc &StartLoc, SMLoc &EndLoc) override;
  ParseStatus tryParseRegister(MCRegister &Reg, SMLoc &StartLoc,
                               SMLoc &EndLoc) override;

  bool parseInstruction(ParseInstructionInfo &Info, StringRef Name,
                        SMLoc NameLoc, OperandVector &Operands) override;

  bool matchAndEmitInstruction(SMLoc IdLoc, unsigned &Opcode,
                               OperandVector &Operands, MCStreamer &Out,
                               uint64_t &ErrorInfo,
                               bool MatchingInlineAsm) override;

  // ── Low-level helpers ──────────────────────────────────────────────────────

  // Parse a register name (x<n>, v<n>, m<n>, s<n>, t<n>).
  // Returns the LLVM register number, or 0 on failure.
  // For tile registers (t<n>), writes the integer 0–7 to TileNum.
  bool parseRegName(StringRef Name, unsigned &Reg, int &TileNum);

  // Parse one operand: register, immediate, float immediate, or mem(base).
  std::unique_ptr<NPUOperand> parseOperand();

  // Parse a memory operand: "offset(xN)" or bare "label".
  std::unique_ptr<NPUOperand> parseMemOperand();

  // Peek at the current AsmLexer token.
  AsmToken curTok() { return Parser.getTok(); }
  AsmToken eatTok() { return Parser.getTok(); }

public:
  NPUAsmParser(const MCSubtargetInfo &STI, MCAsmParser &P,
               const MCInstrInfo &MII)
      : MCTargetAsmParser(STI, MII), Parser(P) {
    setAvailableFeatures(ComputeAvailableFeatures(STI.getFeatureBits()));
  }

// Auto-generated feature/availability helpers (empty for NPU, but required).
#define GET_ASSEMBLER_HEADER
#include "NPUGenAsmMatcher.inc"
};

} // namespace

// ── Register name parsing ─────────────────────────────────────────────────────

bool NPUAsmParser::parseRegName(StringRef Name, unsigned &Reg, int &TileNum) {
  TileNum = -1;
  Reg = 0;

  if (Name.empty()) return false;

  char Prefix = std::tolower((unsigned char)Name[0]);
  StringRef NumStr = Name.drop_front(1);
  unsigned Num;
  if (NumStr.getAsInteger(10, Num)) return false;

  switch (Prefix) {
  case 'x':
    if (Num > 31) return false;
    Reg = NPU::X0 + Num;
    return true;
  case 'v':
    if (Num > 31) return false;
    Reg = NPU::V0 + Num;
    return true;
  case 'm':
    if (Num > 7) return false;
    Reg = NPU::M0 + Num;
    return true;
  case 's':
    if (Num > 31) return false;
    Reg = NPU::S0 + Num;
    return true;
  case 't':
    // Tile registers are just immediates 0–7 in the encoding.
    if (Num > 7) return false;
    TileNum = (int)Num;
    return true;
  }
  return false;
}

// ── Operand parsing ───────────────────────────────────────────────────────────

std::unique_ptr<NPUOperand> NPUAsmParser::parseOperand() {
  AsmToken Tok = curTok();
  SMLoc Start = Tok.getLoc();

  // Integer immediate.
  if (Tok.is(AsmToken::Integer)) {
    int64_t Val = Tok.getIntVal();
    SMLoc End = Tok.getEndLoc();
    Parser.Lex();
    return NPUOperand::createImm(Val, Start, End);
  }

  // Negative integer immediate.
  if (Tok.is(AsmToken::Minus)) {
    Parser.Lex();
    AsmToken Num = curTok();
    if (!Num.is(AsmToken::Integer)) {
      Parser.Error(Start, "expected integer after '-'");
      return nullptr;
    }
    int64_t Val = -Num.getIntVal();
    SMLoc End = Num.getEndLoc();
    Parser.Lex();
    return NPUOperand::createImm(Val, Start, End);
  }

  // Float immediate.
  if (Tok.is(AsmToken::Real)) {
    double Val;
    Tok.getString().getAsDouble(Val);
    SMLoc End = Tok.getEndLoc();
    Parser.Lex();
    return NPUOperand::createFPImm(Val, Start, End);
  }

  // Identifier: could be a register name, label, or tile prefix.
  if (Tok.is(AsmToken::Identifier)) {
    StringRef Name = Tok.getString();
    unsigned Reg = 0;
    int TileNum = -1;
    SMLoc End = Tok.getEndLoc();
    Parser.Lex();

    if (parseRegName(Name, Reg, TileNum)) {
      if (TileNum >= 0)
        return NPUOperand::createImm(TileNum, Start, End);

      // Check for memory operand: "reg" followed by "(" => memory form.
      // But GPRs with '(' following are memory operands (e.g. "x2(x1)" is wrong
      // — this parser uses "offset(xN)" ordering, so a bare reg token here is
      // just a register).
      return NPUOperand::createReg(Reg, Start, End);
    }

    // Symbol label (branch target, e.g. ".LBB0_1").  Store the MCExpr so
    // matchAndEmitInstruction can emit an MCOperand::createExpr() and the
    // MCCodeEmitter will produce a fixup for it.
    MCSymbol *Sym = Parser.getContext().getOrCreateSymbol(Name);
    const MCExpr *E = MCSymbolRefExpr::create(Sym, Parser.getContext());
    return NPUOperand::createExpr(E, Start, End);
  }

  return nullptr;
}

// Parse "offset(xN)" memory operands.
// The lexer sees: Integer('offset') LParen Identifier('xN') RParen
std::unique_ptr<NPUOperand> NPUAsmParser::parseMemOperand() {
  SMLoc Start = curTok().getLoc();
  int64_t Offset = 0;

  // Optional offset before '('.
  if (curTok().is(AsmToken::Integer)) {
    Offset = curTok().getIntVal();
    Parser.Lex();
  } else if (curTok().is(AsmToken::Minus)) {
    Parser.Lex();
    if (!curTok().is(AsmToken::Integer)) {
      Parser.Error(Start, "expected integer offset");
      return nullptr;
    }
    Offset = -curTok().getIntVal();
    Parser.Lex();
  }

  if (!curTok().is(AsmToken::LParen)) {
    Parser.Error(Start, "expected '(' in memory operand");
    return nullptr;
  }
  Parser.Lex(); // consume '('

  if (!curTok().is(AsmToken::Identifier)) {
    Parser.Error(Start, "expected base register");
    return nullptr;
  }
  StringRef RegName = curTok().getString();
  SMLoc RegLoc = curTok().getLoc();
  Parser.Lex(); // consume register name

  unsigned Base = 0;
  int TileNum = -1;
  if (!parseRegName(RegName, Base, TileNum) || TileNum >= 0) {
    Parser.Error(RegLoc, "expected GPR base register");
    return nullptr;
  }

  if (!curTok().is(AsmToken::RParen)) {
    Parser.Error(curTok().getLoc(), "expected ')' after base register");
    return nullptr;
  }
  SMLoc End = curTok().getEndLoc();
  Parser.Lex(); // consume ')'

  return NPUOperand::createMem(Base, Offset, Start, End);
}

// ── parseInstruction ─────────────────────────────────────────────────────────
//
// Called once per instruction line.  `Name` is the mnemonic (already consumed
// by the MCAsmParser dispatcher).  We push the mnemonic as a token operand
// and then parse all remaining operands separated by commas.

bool NPUAsmParser::parseInstruction(ParseInstructionInfo &Info, StringRef Name,
                                    SMLoc NameLoc, OperandVector &Operands) {
  // Handle predication prefix stored in the mnemonic token by the MCAsmParser.
  // The simulator syntax uses "@m<n> MNEM" but llvm-mc strips the prefix before
  // calling parseInstruction — handle it here if present (indicated by '@' at
  // the start of Name).
  // For now, just push the mnemonic as a token.
  Operands.push_back(NPUOperand::createToken(Name, NameLoc));

  // If the rest of the line is empty (no operands), we're done.
  if (Parser.getTok().is(AsmToken::EndOfStatement))
    return false;

  // Parse comma-separated operands.
  while (true) {
    // Check for memory operand: integer or minus followed by '('.
    auto Tok = curTok();
    std::unique_ptr<NPUOperand> Op;

    // parseMemOperand() consumes the integer before checking for '(', so it
    // cannot safely back out when the integer is a plain immediate.  Peek one
    // token ahead: if the current token is an integer and the NEXT token is '('
    // we have the "offset(base)" pattern; otherwise parse as a plain operand.
    // Negative offsets "-N(xR)" are uncommon in generated output; treat a
    // leading '-' as always starting a plain negative immediate.
    bool LooksLikeMem = Tok.is(AsmToken::Integer) &&
                        Parser.getLexer().peekTok().is(AsmToken::LParen);

    if (LooksLikeMem)
      Op = parseMemOperand();
    else
      Op = parseOperand();

    if (!Op) {
      Parser.eatToEndOfStatement();
      return true;
    }
    Operands.push_back(std::move(Op));

    // Commas separate operands; anything else ends the list.
    if (!curTok().is(AsmToken::Comma))
      break;
    Parser.Lex(); // consume ','
  }

  return false;
}

// ── matchAndEmitInstruction ───────────────────────────────────────────────────
//
// Matches the parsed mnemonic + operands against the NPU instruction set and
// emits the corresponding MCInst.  This is a hand-written switch rather than
// a TableGen-generated matcher, which keeps things readable and avoids the
// complexity of setting up MatchInstructionImpl for a 32-byte instruction word.

bool NPUAsmParser::matchAndEmitInstruction(SMLoc IdLoc, unsigned & /*Opcode*/,
                                            OperandVector &Operands,
                                            MCStreamer &Out,
                                            uint64_t & /*ErrorInfo*/,
                                            bool /*MatchingInlineAsm*/) {
  if (Operands.empty()) return true;
  auto &MnemOp = static_cast<NPUOperand &>(*Operands[0]);
  if (!MnemOp.isToken()) return true;
  // LLVM's base AsmParser lowercases the mnemonic before calling us
  // (AsmParser.cpp line ~2268: "Canonicalize the opcode to lower case").
  // Upper-case it back so our comparisons ("ADD", "JALR", …) work.
  std::string MnemUpper = MnemOp.Tok.upper();
  StringRef Mnem = MnemUpper;

  // Helper lambdas for extracting typed operands.
  auto reg = [&](unsigned i) -> unsigned {
    return static_cast<NPUOperand &>(*Operands[i]).Reg;
  };
  auto imm = [&](unsigned i) -> int64_t {
    return static_cast<NPUOperand &>(*Operands[i]).Imm;
  };
  auto mem = [&](unsigned i) -> NPUOperand::MemOp {
    return static_cast<NPUOperand &>(*Operands[i]).Mem;
  };
  // Branch/jump target: returns createExpr if the operand is a label reference,
  // createImm if it is a plain integer.  This is what NPUMCCodeEmitter needs to
  // produce a relocation fixup for label targets vs. direct index.
  auto brtarget = [&](unsigned i) -> MCOperand {
    auto &Op = static_cast<NPUOperand &>(*Operands[i]);
    if (Op.isExprOp())
      return MCOperand::createExpr(Op.Expr);
    return MCOperand::createImm(Op.Imm);
  };

  MCInst Inst;

  auto emit = [&]() {
    Out.emitInstruction(Inst, getSTI());
    return false;
  };

  // ── Control core arithmetic ───────────────────────────────────────────
  if (Mnem == "ADD") {
    Inst.setOpcode(NPU::ADD);
    Inst.addOperand(MCOperand::createReg(reg(1)));
    Inst.addOperand(MCOperand::createReg(reg(2)));
    Inst.addOperand(MCOperand::createReg(reg(3)));
    return emit();
  }
  if (Mnem == "SUB") {
    Inst.setOpcode(NPU::SUB);
    Inst.addOperand(MCOperand::createReg(reg(1)));
    Inst.addOperand(MCOperand::createReg(reg(2)));
    Inst.addOperand(MCOperand::createReg(reg(3)));
    return emit();
  }
  if (Mnem == "MUL") {
    Inst.setOpcode(NPU::MUL);
    Inst.addOperand(MCOperand::createReg(reg(1)));
    Inst.addOperand(MCOperand::createReg(reg(2)));
    Inst.addOperand(MCOperand::createReg(reg(3)));
    return emit();
  }
  if (Mnem == "AND") {
    Inst.setOpcode(NPU::AND);
    Inst.addOperand(MCOperand::createReg(reg(1)));
    Inst.addOperand(MCOperand::createReg(reg(2)));
    Inst.addOperand(MCOperand::createReg(reg(3)));
    return emit();
  }
  if (Mnem == "OR") {
    Inst.setOpcode(NPU::OR);
    Inst.addOperand(MCOperand::createReg(reg(1)));
    Inst.addOperand(MCOperand::createReg(reg(2)));
    Inst.addOperand(MCOperand::createReg(reg(3)));
    return emit();
  }
  if (Mnem == "XOR") {
    Inst.setOpcode(NPU::XOR);
    Inst.addOperand(MCOperand::createReg(reg(1)));
    Inst.addOperand(MCOperand::createReg(reg(2)));
    Inst.addOperand(MCOperand::createReg(reg(3)));
    return emit();
  }
  if (Mnem == "SLL") {
    Inst.setOpcode(NPU::SLL);
    Inst.addOperand(MCOperand::createReg(reg(1)));
    Inst.addOperand(MCOperand::createReg(reg(2)));
    Inst.addOperand(MCOperand::createReg(reg(3)));
    return emit();
  }
  if (Mnem == "SRL") {
    Inst.setOpcode(NPU::SRL);
    Inst.addOperand(MCOperand::createReg(reg(1)));
    Inst.addOperand(MCOperand::createReg(reg(2)));
    Inst.addOperand(MCOperand::createReg(reg(3)));
    return emit();
  }
  if (Mnem == "SLT") {
    Inst.setOpcode(NPU::SLT);
    Inst.addOperand(MCOperand::createReg(reg(1)));
    Inst.addOperand(MCOperand::createReg(reg(2)));
    Inst.addOperand(MCOperand::createReg(reg(3)));
    return emit();
  }
  if (Mnem == "ADDI") {
    Inst.setOpcode(NPU::ADDI);
    Inst.addOperand(MCOperand::createReg(reg(1)));
    Inst.addOperand(MCOperand::createReg(reg(2)));
    Inst.addOperand(MCOperand::createImm(imm(3)));
    return emit();
  }
  if (Mnem == "LI") {
    Inst.setOpcode(NPU::LI);
    Inst.addOperand(MCOperand::createReg(reg(1)));
    Inst.addOperand(MCOperand::createImm(imm(2)));
    return emit();
  }

  // ── Loads / stores ────────────────────────────────────────────────────
  if (Mnem == "LW") {
    auto M = mem(2);
    Inst.setOpcode(NPU::LW);
    Inst.addOperand(MCOperand::createReg(reg(1)));
    Inst.addOperand(MCOperand::createReg(M.Base));
    Inst.addOperand(MCOperand::createImm(M.Offset));
    return emit();
  }
  if (Mnem == "LH") {
    auto M = mem(2);
    Inst.setOpcode(NPU::LH);
    Inst.addOperand(MCOperand::createReg(reg(1)));
    Inst.addOperand(MCOperand::createReg(M.Base));
    Inst.addOperand(MCOperand::createImm(M.Offset));
    return emit();
  }
  if (Mnem == "LB") {
    auto M = mem(2);
    Inst.setOpcode(NPU::LB);
    Inst.addOperand(MCOperand::createReg(reg(1)));
    Inst.addOperand(MCOperand::createReg(M.Base));
    Inst.addOperand(MCOperand::createImm(M.Offset));
    return emit();
  }
  if (Mnem == "SW") {
    auto M = mem(2);
    Inst.setOpcode(NPU::SW);
    Inst.addOperand(MCOperand::createReg(reg(1)));
    Inst.addOperand(MCOperand::createReg(M.Base));
    Inst.addOperand(MCOperand::createImm(M.Offset));
    return emit();
  }
  if (Mnem == "SH") {
    auto M = mem(2);
    Inst.setOpcode(NPU::SH);
    Inst.addOperand(MCOperand::createReg(reg(1)));
    Inst.addOperand(MCOperand::createReg(M.Base));
    Inst.addOperand(MCOperand::createImm(M.Offset));
    return emit();
  }
  if (Mnem == "SB") {
    auto M = mem(2);
    Inst.setOpcode(NPU::SB);
    Inst.addOperand(MCOperand::createReg(reg(1)));
    Inst.addOperand(MCOperand::createReg(M.Base));
    Inst.addOperand(MCOperand::createImm(M.Offset));
    return emit();
  }

  // ── Branches ─────────────────────────────────────────────────────────
  if (Mnem == "BEQ") {
    Inst.setOpcode(NPU::BEQ);
    Inst.addOperand(MCOperand::createReg(reg(1)));
    Inst.addOperand(MCOperand::createReg(reg(2)));
    Inst.addOperand(brtarget(3));
    return emit();
  }
  if (Mnem == "BNE") {
    Inst.setOpcode(NPU::BNE);
    Inst.addOperand(MCOperand::createReg(reg(1)));
    Inst.addOperand(MCOperand::createReg(reg(2)));
    Inst.addOperand(brtarget(3));
    return emit();
  }
  if (Mnem == "BLT") {
    Inst.setOpcode(NPU::BLT);
    Inst.addOperand(MCOperand::createReg(reg(1)));
    Inst.addOperand(MCOperand::createReg(reg(2)));
    Inst.addOperand(brtarget(3));
    return emit();
  }
  if (Mnem == "BGE") {
    Inst.setOpcode(NPU::BGE);
    Inst.addOperand(MCOperand::createReg(reg(1)));
    Inst.addOperand(MCOperand::createReg(reg(2)));
    Inst.addOperand(brtarget(3));
    return emit();
  }

  // JAL x0, label (unconditional branch) or JAL x1, label (call).
  if (Mnem == "JAL") {
    unsigned RD = reg(1);
    if (RD == NPU::X0) {
      Inst.setOpcode(NPU::BR);
      Inst.addOperand(brtarget(2));
    } else {
      Inst.setOpcode(NPU::JAL);
      Inst.addOperand(brtarget(2));
    }
    return emit();
  }

  // JALR x0, x1, 0 (return) or JALR x1, xN, 0 (indirect call).
  if (Mnem == "JALR") {
    unsigned RD = reg(1);
    if (RD == NPU::X0) {
      Inst.setOpcode(NPU::RET);
    } else {
      Inst.setOpcode(NPU::JALRcall);
      Inst.addOperand(MCOperand::createReg(reg(2)));
    }
    return emit();
  }

  // ── Kernel launch ─────────────────────────────────────────────────────
  if (Mnem == "KLAUNCH") {
    Inst.setOpcode(NPU::KLAUNCH);
    Inst.addOperand(MCOperand::createReg(reg(1)));
    Inst.addOperand(MCOperand::createImm(imm(2)));
    return emit();
  }
  if (Mnem == "KWAIT") {
    Inst.setOpcode(NPU::KWAIT);
    return emit();
  }
  if (Mnem == "HALT") {
    Inst.setOpcode(NPU::HALT);
    return emit();
  }

  // ── Tensor engine ─────────────────────────────────────────────────────
  if (Mnem == "CFG.SHAPE") {
    Inst.setOpcode(NPU::T_CFG_SHAPE);
    Inst.addOperand(MCOperand::createImm(imm(1)));
    Inst.addOperand(MCOperand::createImm(imm(2)));
    Inst.addOperand(MCOperand::createImm(imm(3)));
    return emit();
  }
  if (Mnem == "ACC.CLR") {
    Inst.setOpcode(NPU::T_ACC_CLR);
    Inst.addOperand(MCOperand::createImm(imm(1)));
    return emit();
  }
  if (Mnem == "MMA") {
    Inst.setOpcode(NPU::T_MMA);
    Inst.addOperand(MCOperand::createImm(imm(1)));
    Inst.addOperand(MCOperand::createImm(imm(2)));
    Inst.addOperand(MCOperand::createImm(imm(3)));
    return emit();
  }
  if (Mnem == "MATMUL") {
    Inst.setOpcode(NPU::T_MATMUL);
    Inst.addOperand(MCOperand::createImm(imm(1)));
    Inst.addOperand(MCOperand::createImm(imm(2)));
    Inst.addOperand(MCOperand::createImm(imm(3)));
    return emit();
  }
  if (Mnem == "FENCE") {
    Inst.setOpcode(NPU::T_FENCE);
    return emit();
  }
  if (Mnem == "WAIT") {
    Inst.setOpcode(NPU::T_WAIT);
    return emit();
  }
  if (Mnem == "BARRIER") {
    Inst.setOpcode(NPU::T_BARRIER);
    return emit();
  }
  if (Mnem == "DMA.IN") {
    Inst.setOpcode(NPU::T_DMA_IN);
    Inst.addOperand(MCOperand::createReg(reg(1)));
    Inst.addOperand(MCOperand::createReg(reg(2)));
    Inst.addOperand(MCOperand::createImm(imm(3)));
    return emit();
  }
  if (Mnem == "DMA.OUT") {
    Inst.setOpcode(NPU::T_DMA_OUT);
    Inst.addOperand(MCOperand::createReg(reg(1)));
    Inst.addOperand(MCOperand::createReg(reg(2)));
    Inst.addOperand(MCOperand::createImm(imm(3)));
    return emit();
  }

  // ── Vector core ────────────────────────────────────────────────────────
  if (Mnem == "VADD") {
    Inst.setOpcode(NPU::V_VADD);
    Inst.addOperand(MCOperand::createReg(reg(1)));
    Inst.addOperand(MCOperand::createReg(reg(2)));
    Inst.addOperand(MCOperand::createReg(reg(3)));
    return emit();
  }
  if (Mnem == "VSUB") {
    Inst.setOpcode(NPU::V_VSUB);
    Inst.addOperand(MCOperand::createReg(reg(1)));
    Inst.addOperand(MCOperand::createReg(reg(2)));
    Inst.addOperand(MCOperand::createReg(reg(3)));
    return emit();
  }
  if (Mnem == "VMUL") {
    Inst.setOpcode(NPU::V_VMUL);
    Inst.addOperand(MCOperand::createReg(reg(1)));
    Inst.addOperand(MCOperand::createReg(reg(2)));
    Inst.addOperand(MCOperand::createReg(reg(3)));
    return emit();
  }
  if (Mnem == "VDIV") {
    Inst.setOpcode(NPU::V_VDIV);
    Inst.addOperand(MCOperand::createReg(reg(1)));
    Inst.addOperand(MCOperand::createReg(reg(2)));
    Inst.addOperand(MCOperand::createReg(reg(3)));
    return emit();
  }
  if (Mnem == "VEXP") {
    Inst.setOpcode(NPU::V_VEXP);
    Inst.addOperand(MCOperand::createReg(reg(1)));
    Inst.addOperand(MCOperand::createReg(reg(2)));
    return emit();
  }
  if (Mnem == "VLOG") {
    Inst.setOpcode(NPU::V_VLOG);
    Inst.addOperand(MCOperand::createReg(reg(1)));
    Inst.addOperand(MCOperand::createReg(reg(2)));
    return emit();
  }
  if (Mnem == "VRSQRT") {
    Inst.setOpcode(NPU::V_VRSQRT);
    Inst.addOperand(MCOperand::createReg(reg(1)));
    Inst.addOperand(MCOperand::createReg(reg(2)));
    return emit();
  }
  if (Mnem == "VBCAST") {
    Inst.setOpcode(NPU::V_VBCAST);
    Inst.addOperand(MCOperand::createReg(reg(1)));
    Inst.addOperand(MCOperand::createReg(reg(2)));
    return emit();
  }
  if (Mnem == "VLOAD") {
    auto M = mem(2);
    Inst.setOpcode(NPU::V_VLOAD);
    Inst.addOperand(MCOperand::createReg(reg(1)));
    Inst.addOperand(MCOperand::createReg(M.Base));
    Inst.addOperand(MCOperand::createImm(M.Offset));
    return emit();
  }
  if (Mnem == "VSTORE") {
    auto M = mem(2);
    Inst.setOpcode(NPU::V_VSTORE);
    Inst.addOperand(MCOperand::createReg(reg(1)));
    Inst.addOperand(MCOperand::createReg(M.Base));
    Inst.addOperand(MCOperand::createImm(M.Offset));
    return emit();
  }
  if (Mnem == "VREDSUM") {
    Inst.setOpcode(NPU::V_VREDSUM);
    Inst.addOperand(MCOperand::createReg(reg(1)));
    Inst.addOperand(MCOperand::createReg(reg(2)));
    return emit();
  }
  if (Mnem == "VREDMAX") {
    Inst.setOpcode(NPU::V_VREDMAX);
    Inst.addOperand(MCOperand::createReg(reg(1)));
    Inst.addOperand(MCOperand::createReg(reg(2)));
    return emit();
  }
  if (Mnem == "VREDMIN") {
    Inst.setOpcode(NPU::V_VREDMIN);
    Inst.addOperand(MCOperand::createReg(reg(1)));
    Inst.addOperand(MCOperand::createReg(reg(2)));
    return emit();
  }
  if (Mnem == "VSETVL") {
    Inst.setOpcode(NPU::V_VSETVL);
    Inst.addOperand(MCOperand::createReg(reg(1)));
    Inst.addOperand(MCOperand::createReg(reg(2)));
    return emit();
  }
  if (Mnem == "KID") {
    Inst.setOpcode(NPU::V_KID);
    Inst.addOperand(MCOperand::createReg(reg(1)));
    return emit();
  }
  if (Mnem == "KEXIT") {
    Inst.setOpcode(NPU::V_KEXIT);
    return emit();
  }
  if (Mnem == "VMASK.SET") {
    Inst.setOpcode(NPU::V_VMASK_SET);
    Inst.addOperand(MCOperand::createReg(reg(1)));
    Inst.addOperand(MCOperand::createReg(reg(2)));
    return emit();
  }
  if (Mnem == "VMASK.NOT") {
    Inst.setOpcode(NPU::V_VMASK_NOT);
    Inst.addOperand(MCOperand::createReg(reg(1)));
    Inst.addOperand(MCOperand::createReg(reg(2)));
    return emit();
  }
  if (Mnem == "VMASK.AND") {
    Inst.setOpcode(NPU::V_VMASK_AND);
    Inst.addOperand(MCOperand::createReg(reg(1)));
    Inst.addOperand(MCOperand::createReg(reg(2)));
    Inst.addOperand(MCOperand::createReg(reg(3)));
    return emit();
  }
  if (Mnem == "VMASK.OR") {
    Inst.setOpcode(NPU::V_VMASK_OR);
    Inst.addOperand(MCOperand::createReg(reg(1)));
    Inst.addOperand(MCOperand::createReg(reg(2)));
    Inst.addOperand(MCOperand::createReg(reg(3)));
    return emit();
  }
  if (Mnem == "BRZ") {
    Inst.setOpcode(NPU::V_BRZ);
    Inst.addOperand(MCOperand::createReg(reg(1)));
    Inst.addOperand(MCOperand::createImm(imm(2)));
    return emit();
  }
  if (Mnem == "BRNZ") {
    Inst.setOpcode(NPU::V_BRNZ);
    Inst.addOperand(MCOperand::createReg(reg(1)));
    Inst.addOperand(MCOperand::createImm(imm(2)));
    return emit();
  }
  if (Mnem == "LOOP") {
    Inst.setOpcode(NPU::V_LOOP);
    Inst.addOperand(MCOperand::createReg(reg(1)));
    Inst.addOperand(MCOperand::createImm(imm(2)));
    return emit();
  }
  if (Mnem == "BR") {
    Inst.setOpcode(NPU::V_BR);
    Inst.addOperand(MCOperand::createImm(imm(1)));
    return emit();
  }
  if (Mnem == "SLI") {
    Inst.setOpcode(NPU::V_SLI);
    Inst.addOperand(MCOperand::createReg(reg(1)));
    Inst.addOperand(MCOperand::createImm(imm(2)));
    return emit();
  }
  if (Mnem == "SADD") {
    Inst.setOpcode(NPU::V_SADD);
    Inst.addOperand(MCOperand::createReg(reg(1)));
    Inst.addOperand(MCOperand::createReg(reg(2)));
    Inst.addOperand(MCOperand::createReg(reg(3)));
    return emit();
  }
  if (Mnem == "SADDI") {
    Inst.setOpcode(NPU::V_SADDI);
    Inst.addOperand(MCOperand::createReg(reg(1)));
    Inst.addOperand(MCOperand::createReg(reg(2)));
    Inst.addOperand(MCOperand::createImm(imm(3)));
    return emit();
  }

  Parser.Error(IdLoc, "unknown NPU mnemonic '" + Mnem + "'");
  return true;
}

// ── Required interface overrides ─────────────────────────────────────────────

bool NPUAsmParser::parseRegister(MCRegister &Reg, SMLoc &StartLoc,
                                  SMLoc &EndLoc) {
  StartLoc = curTok().getLoc();
  if (!curTok().is(AsmToken::Identifier)) return true;
  unsigned R = 0;
  int TileNum = -1;
  if (!parseRegName(curTok().getString(), R, TileNum) || TileNum >= 0)
    return true;
  Reg = R;
  EndLoc = curTok().getEndLoc();
  Parser.Lex();
  return false;
}

ParseStatus NPUAsmParser::tryParseRegister(MCRegister &Reg, SMLoc &StartLoc,
                                            SMLoc &EndLoc) {
  if (parseRegister(Reg, StartLoc, EndLoc))
    return ParseStatus::NoMatch;
  return ParseStatus::Success;
}

// ── Generated matcher implementation ─────────────────────────────────────────
// ComputeAvailableFeatures, convertToMapAndConstraints, and
// MatchInstructionImpl bodies all live in this section.

#define GET_MATCHER_IMPLEMENTATION
#include "NPUGenAsmMatcher.inc"

// ── Registration ─────────────────────────────────────────────────────────────
// NPU is a custom target not listed in LLVM_ALL_TARGETS, so llvm-mc never
// calls LLVMInitializeNPUAsmParser() via InitializeAllAsmParsers().  Use a
// file-scope static so the registration happens at library load time.
static llvm::RegisterMCAsmParser<NPUAsmParser> X(getTheNPUTarget());

extern "C" LLVM_ABI LLVM_EXTERNAL_VISIBILITY void
LLVMInitializeNPUAsmParser() {
  // Registration already done by the file-scope static above.
}
