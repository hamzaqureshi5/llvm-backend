//===- NPUInstrInfo.td - NPU Instruction defs --------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// NPU control-core scalar instruction set (Layer 1, Sections 1–3 of
// fully_prog_npu.md).  Mnemonics and operand order match the simulator's text
// assembler (npu-simulator/npu_asm.c CTAB) so emitted assembly round-trips
// through npu_asm_control().
//
// Binary encoding: each instruction is a 32-byte word (NPU_WORD_BYTES).
// The layout is defined in npu_asm.c :: npu_encode_cinst:
//   byte 0  : stream type (0 = cinst, 1 = vinst)
//   byte 1  : opcode (cop_t enum value from npu.h)
//   byte 2  : rd
//   byte 3  : rs1
//   byte 4  : rs2
//   bytes 5–11 : tensor-engine fields (unused here)
//   bytes 12–15: count (u32 LE)
//   bytes 16–23: imm (i64 LE) or fimm (double)
//   bytes 24–31: padding / extended fields
//
// The MCCodeEmitter (NPUMCCodeEmitter.cpp) writes this layout; we do NOT
// use the automatic TableGen instruction encoder because the 32-byte word
// would require 256-bit Inst{} fields and cannot be bit-packed sensibly.
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Instruction format base classes (no binary encoding: text output only).
//===----------------------------------------------------------------------===//

class NPUInst<dag outs, dag ins, string asmstr, list<dag> pattern>
    : Instruction {
  let Namespace = "NPU";
  dag OutOperandList = outs;
  dag InOperandList = ins;
  let AsmString = asmstr;
  let Pattern = pattern;
  // Each instruction occupies one 32-byte binary record (NPU_WORD_BYTES).
  // The MCCodeEmitter writes the full 32-byte encoding; llvm-objdump uses
  // this field to step through a binary object file correctly.
  let Size = 32;
}

class Pseudo<dag outs, dag ins, list<dag> pattern>
    : NPUInst<outs, ins, "", pattern> {
  let isPseudo = 1;
  let isCodeGenOnly = 1;
}

//===----------------------------------------------------------------------===//
// Type profiles and target-specific SDNodes.
//===----------------------------------------------------------------------===//

def SDT_NPUCallSeqStart : SDCallSeqStart<[SDTCisVT<0, i64>, SDTCisVT<1, i64>]>;
def SDT_NPUCallSeqEnd    : SDCallSeqEnd<[SDTCisVT<0, i64>, SDTCisVT<1, i64>]>;
def SDT_NPUCall          : SDTypeProfile<0, -1, [SDTCisVT<0, iPTR>]>;

def callseq_start : SDNode<"ISD::CALLSEQ_START", SDT_NPUCallSeqStart,
                           [SDNPHasChain, SDNPOutGlue]>;
def callseq_end   : SDNode<"ISD::CALLSEQ_END", SDT_NPUCallSeqEnd,
                           [SDNPHasChain, SDNPOptInGlue, SDNPOutGlue]>;

def npucall    : SDNode<"NPUISD::CALL", SDT_NPUCall,
                        [SDNPHasChain, SDNPOutGlue, SDNPOptInGlue,
                         SDNPVariadic]>;
def npuretglue : SDNode<"NPUISD::RET_GLUE", SDTNone,
                        [SDNPHasChain, SDNPOptInGlue, SDNPVariadic]>;

// KLAUNCH: dispatch a kernel.  Operands: kernel_id (i64), grid_size (i64).
def SDT_NPUKlaunch : SDTypeProfile<0, 2, [SDTCisVT<0, i64>,
                                          SDTCisVT<1, i64>]>;
def npuklaunch : SDNode<"NPUISD::KLAUNCH", SDT_NPUKlaunch,
                        [SDNPHasChain, SDNPSideEffect]>;

// KWAIT: barrier — wait for all dispatched kernels to finish.
def npukwait : SDNode<"NPUISD::KWAIT", SDTNone,
                      [SDNPHasChain, SDNPSideEffect]>;

// ── Tensor Engine SDNodes ─────────────────────────────────────────────────────
// Produced by LowerIntrinsicCall; consumed by NPUISelDAGToDAG::Select().

// CFG_SHAPE: configure tile dimensions M×K×N.
def SDT_NPUCfgShape : SDTypeProfile<0, 3, [SDTCisVT<0, i64>, SDTCisVT<1, i64>,
                                           SDTCisVT<2, i64>]>;
def npucfgshape : SDNode<"NPUISD::CFG_SHAPE", SDT_NPUCfgShape,
                         [SDNPHasChain, SDNPSideEffect]>;

// Tile-side-effect ops: emit a tensor-engine instruction and return only the
// chain.  The tile register number (i64 SSA token) is threaded back by
// LowerIntrinsicCall via getMergeValues({Acc_TargetConst, newChain}), so the
// NPUISD node itself only needs to carry the chain.
//
// Rationale: the hardware tile number never changes — the destination tile is
// always the same register after ACC_CLR/MMA/etc. as it was before.  Keeping
// the SSA flow as a TargetConstant (instead of an actual output register)
// avoids the need for a tile register class and allocator in this teaching
// backend.  The variadic operand list allows the number of tile/address args
// to vary per op; NPUISelDAGToDAG validates counts per intrinsic opcode.
def SDT_NPUTileOp : SDTypeProfile<0, -1, []>;
def npuacclr   : SDNode<"NPUISD::ACC_CLR", SDT_NPUTileOp,
                        [SDNPHasChain, SDNPSideEffect]>;
def npumma     : SDNode<"NPUISD::MMA",     SDT_NPUTileOp,
                        [SDNPHasChain, SDNPSideEffect]>;
def npumatmul  : SDNode<"NPUISD::MATMUL",  SDT_NPUTileOp,
                        [SDNPHasChain, SDNPSideEffect]>;
def npudot     : SDNode<"NPUISD::DOT",     SDT_NPUTileOp,
                        [SDNPHasChain, SDNPSideEffect]>;
def nputeload  : SDNode<"NPUISD::TELOAD",  SDT_NPUTileOp,
                        [SDNPHasChain, SDNPSideEffect, SDNPMemOperand]>;
def nputeloadw : SDNode<"NPUISD::TELOADW", SDT_NPUTileOp,
                        [SDNPHasChain, SDNPSideEffect, SDNPMemOperand]>;
def npurelu    : SDNode<"NPUISD::RELU",    SDT_NPUTileOp,
                        [SDNPHasChain, SDNPSideEffect]>;
def npuscale   : SDNode<"NPUISD::SCALE",   SDT_NPUTileOp,
                        [SDNPHasChain, SDNPSideEffect]>;
def npubcast   : SDNode<"NPUISD::BCAST",   SDT_NPUTileOp,
                        [SDNPHasChain, SDNPSideEffect]>;

// Void tile ops (store, DMA).
def SDT_NPUTileStore : SDTypeProfile<0, 3, [SDTCisVT<0, i64>, SDTCisVT<1, i64>,
                                            SDTCisVT<2, i64>]>;
def nputestore : SDNode<"NPUISD::TESTORE", SDT_NPUTileStore,
                        [SDNPHasChain, SDNPSideEffect, SDNPMemOperand]>;

def SDT_NPUDma : SDTypeProfile<0, 3, [SDTCisVT<0, i64>, SDTCisVT<1, i64>,
                                      SDTCisVT<2, i64>]>;
def npudma_in  : SDNode<"NPUISD::DMA_IN",  SDT_NPUDma,
                        [SDNPHasChain, SDNPSideEffect]>;
def npudma_out : SDNode<"NPUISD::DMA_OUT", SDT_NPUDma,
                        [SDNPHasChain, SDNPSideEffect]>;

// ── Vector Core SDNodes ───────────────────────────────────────────────────────

// VLOAD: load <16 x float> from ptr + stride*lane.  Returns v16f32 + chain.
def SDT_NPUVLoad : SDTypeProfile<1, 2, [SDTCisVT<0, v16f32>,
                                        SDTCisVT<1, i64>, SDTCisVT<2, i64>]>;
def npuvload : SDNode<"NPUISD::VLOAD", SDT_NPUVLoad,
                      [SDNPHasChain, SDNPSideEffect, SDNPMemOperand]>;

// VSTORE: store <16 x float> to ptr + stride*lane.  Void.
def SDT_NPUVStore : SDTypeProfile<0, 3, [SDTCisVT<0, i64>, SDTCisVT<1, v16f32>,
                                         SDTCisVT<2, i64>]>;
def npuvstore : SDNode<"NPUISD::VSTORE", SDT_NPUVStore,
                       [SDNPHasChain, SDNPSideEffect, SDNPMemOperand]>;

// VGATHER: indexed load  vd[i] = mem[base + idx[i]].  Returns v16f32 + chain.
def SDT_NPUVGather : SDTypeProfile<1, 2, [SDTCisVT<0, v16f32>,
                                          SDTCisVT<1, i64>, SDTCisVT<2, v16f32>]>;
def npuvgather : SDNode<"NPUISD::VGATHER", SDT_NPUVGather,
                        [SDNPHasChain, SDNPSideEffect, SDNPMemOperand]>;

// VSCATTER: indexed store  mem[base + idx[i]] = data[i].  Void.
def SDT_NPUVScatter : SDTypeProfile<0, 3, [SDTCisVT<0, i64>, SDTCisVT<1, v16f32>,
                                           SDTCisVT<2, v16f32>]>;
def npuvscatter : SDNode<"NPUISD::VSCATTER", SDT_NPUVScatter,
                         [SDNPHasChain, SDNPSideEffect, SDNPMemOperand]>;

// VSETVL: set active vector length.  Takes requested VL (i64), returns actual VL (i64).
// Both the request and result are integer counts that live in GPRs.
def SDT_NPUVsetvl : SDTypeProfile<1, 1, [SDTCisVT<0, i64>, SDTCisVT<1, i64>]>;
def npuvsetvl : SDNode<"NPUISD::VSETVL", SDT_NPUVsetvl,
                       [SDNPHasChain, SDNPSideEffect]>;

//===----------------------------------------------------------------------===//
// Operands.
//===----------------------------------------------------------------------===//

def brtarget : Operand<OtherVT>;
def calltarget : Operand<i64>;

// Condition code immediate carried by NPUISD::BR_CC (one of NPUCC below).
def cc_imm : Operand<i64>;

// Memory operand: offset(base) -> "imm(xN)".
def AddrRI : ComplexPattern<i64, 2, "selectAddrRI", [frameindex]>;
def memri : Operand<i64> {
  let PrintMethod = "printMemRIOperand";
  let MIOperandInfo = (ops GPR, i64imm);
}

//===----------------------------------------------------------------------===//
// Scalar arithmetic (RRR): "OP xd, xa, xb".
//===----------------------------------------------------------------------===//

class ALU_RRR<string mn, SDPatternOperator op>
    : NPUInst<(outs GPR:$rd), (ins GPR:$a, GPR:$b),
              mn # " $rd, $a, $b",
              [(set i64:$rd, (op i64:$a, i64:$b))]> {
  let SchedRW = [WriteALU];
}

def ADD : ALU_RRR<"ADD", add>;
def SUB : ALU_RRR<"SUB", sub>;
def MUL : ALU_RRR<"MUL", mul>;
def AND : ALU_RRR<"AND", and>;
def OR  : ALU_RRR<"OR", or>;
def XOR : ALU_RRR<"XOR", xor>;
def SLL : ALU_RRR<"SLL", shl>;
def SRL : ALU_RRR<"SRL", srl>;

// SLT sets xd = (signed)xa < (signed)xb ? 1 : 0.
def SLT : NPUInst<(outs GPR:$rd), (ins GPR:$a, GPR:$b), "SLT $rd, $a, $b",
                  [(set i64:$rd, (setlt i64:$a, i64:$b))]> {
  let SchedRW = [WriteALU];
}

// SLTU: unsigned less-than comparison.  The simulator has no native SLTU
// but we synthesize it: flip the sign bit of both operands with XOR, then
// perform a signed SLT.  The hardware instruction is still SLT; the
// conversion happens during DAG lowering (NPUISelDAGToDAG.cpp).
// This pseudo is retained as a TableGen def so the MC layer can print it
// during debugging; the real selection emits an XOR+SLT sequence.
def SLTU : NPUInst<(outs GPR:$rd), (ins GPR:$a, GPR:$b), "SLT $rd, $a, $b",
                   [(set i64:$rd, (setult i64:$a, i64:$b))]> {
  let isPseudo = 1;
  let isCodeGenOnly = 1;
  let SchedRW = [WriteALU];
}

//===----------------------------------------------------------------------===//
// Register-immediate.
//===----------------------------------------------------------------------===//

// ADDI xd, xa, imm
def ADDI : NPUInst<(outs GPR:$rd), (ins GPR:$a, i64imm:$imm),
                   "ADDI $rd, $a, $imm",
                   [(set i64:$rd, (add i64:$a, imm:$imm))]> {
  let SchedRW = [WriteALUI];
}

// LI xd, imm : materialize a full 64-bit immediate in one instruction.
let isReMaterializable = 1, isAsCheapAsAMove = 1 in
def LI : NPUInst<(outs GPR:$rd), (ins i64imm:$imm), "LI $rd, $imm",
                 [(set i64:$rd, imm:$imm)]> {
  let SchedRW = [WriteALUI];
}

//===----------------------------------------------------------------------===//
// Loads / stores. The simulator's LW sign-extends a 32-bit word; SW stores the
// low 32 bits. Value register is printed first for stores ("SW xs, off(xa)").
//===----------------------------------------------------------------------===//

let mayLoad = 1 in {
  def LW : NPUInst<(outs GPR:$rd), (ins memri:$addr), "LW $rd, $addr",
                   [(set i64:$rd, (sextloadi32 AddrRI:$addr))]> {
    let SchedRW = [WriteLoad];
  }
  def LH : NPUInst<(outs GPR:$rd), (ins memri:$addr), "LH $rd, $addr",
                   [(set i64:$rd, (sextloadi16 AddrRI:$addr))]> {
    let SchedRW = [WriteLoad];
  }
  def LB : NPUInst<(outs GPR:$rd), (ins memri:$addr), "LB $rd, $addr",
                   [(set i64:$rd, (sextloadi8 AddrRI:$addr))]> {
    let SchedRW = [WriteLoad];
  }
}

// anyext versions of the load instructions. The type legalizer promotes i32
// arithmetic to i64 and uses anyext for loads that feed integer ops where only
// the lower bits matter. LW sign-extends, which satisfies anyext (the caller
// doesn't inspect the upper bits).
def : Pat<(i64 (extloadi32 AddrRI:$addr)), (LW AddrRI:$addr)>;
def : Pat<(i64 (extloadi16 AddrRI:$addr)), (LH AddrRI:$addr)>;
def : Pat<(i64 (extloadi8  AddrRI:$addr)), (LB AddrRI:$addr)>;

let mayStore = 1 in {
  def SW : NPUInst<(outs), (ins GPR:$rs, memri:$addr), "SW $rs, $addr",
                   [(truncstorei32 i64:$rs, AddrRI:$addr)]> {
    let SchedRW = [WriteStore];
  }
  def SH : NPUInst<(outs), (ins GPR:$rs, memri:$addr), "SH $rs, $addr",
                   [(truncstorei16 i64:$rs, AddrRI:$addr)]> {
    let SchedRW = [WriteStore];
  }
  def SB : NPUInst<(outs), (ins GPR:$rs, memri:$addr), "SB $rs, $addr",
                   [(truncstorei8 i64:$rs, AddrRI:$addr)]> {
    let SchedRW = [WriteStore];
  }
}

//===----------------------------------------------------------------------===//
// Control flow.
//===----------------------------------------------------------------------===//

// Unconditional branch: JAL x0, label (link discarded into the zero register).
let isBranch = 1, isBarrier = 1, isTerminator = 1 in
def BR : NPUInst<(outs), (ins brtarget:$dst), "JAL x0, $dst",
                 [(br bb:$dst)]> {
  let SchedRW = [WriteBranch];
}

// Conditional branches: "OP xa, xb, label". Lowered from NPUISD::BR_CC.
// Signed comparisons are native; unsigned ones (BLTU/BGEU) are synthesized by
// NPUISelDAGToDAG by flipping the sign bit of both operands before branching.
let isBranch = 1, isTerminator = 1 in {
  def BEQ : NPUInst<(outs), (ins GPR:$a, GPR:$b, brtarget:$dst),
                    "BEQ $a, $b, $dst", []> {
    let SchedRW = [WriteBranch];
  }
  def BNE : NPUInst<(outs), (ins GPR:$a, GPR:$b, brtarget:$dst),
                    "BNE $a, $b, $dst", []> {
    let SchedRW = [WriteBranch];
  }
  def BLT : NPUInst<(outs), (ins GPR:$a, GPR:$b, brtarget:$dst),
                    "BLT $a, $b, $dst", []> {
    let SchedRW = [WriteBranch];
  }
  def BGE : NPUInst<(outs), (ins GPR:$a, GPR:$b, brtarget:$dst),
                    "BGE $a, $b, $dst", []> {
    let SchedRW = [WriteBranch];
  }
}

// Calls.
let isCall = 1, Defs = [X1], Uses = [X2] in {
  def JAL : NPUInst<(outs), (ins calltarget:$dst), "JAL x1, $dst", []> {
    let SchedRW = [WriteCall];
  }
  def JALRcall : NPUInst<(outs), (ins GPR:$rs), "JALR x1, $rs, 0",
                         [(npucall i64:$rs)]> {
    let SchedRW = [WriteCall];
  }
}

def : Pat<(npucall (i64 tglobaladdr:$dst)), (JAL tglobaladdr:$dst)>;
def : Pat<(npucall (i64 texternalsym:$dst)), (JAL texternalsym:$dst)>;

// Return: jump to the return address in x1.
let isReturn = 1, isTerminator = 1, isBarrier = 1 in
def RET : NPUInst<(outs), (ins), "JALR x0, x1, 0", [(npuretglue)]> {
  let SchedRW = [WriteBranch];
}

// Halt: terminate the flat control program (emitted at end of file).
def HALT : NPUInst<(outs), (ins), "HALT", []> {
  let hasSideEffects = 1;
  let isBarrier = 1;
  let isTerminator = 1;
  let SchedRW = [WriteBranch];
}

//===----------------------------------------------------------------------===//
// Kernel launch and synchronization (Section 3).
//===----------------------------------------------------------------------===//

// KLAUNCH rs1, grid_size — launch `grid_size` instances of kernel[x[rs1]].
// rs1 holds the kernel id (returned by npu_register_kernel).
// imm holds the grid size (number of parallel instances to dispatch).
// This is an asynchronous dispatch; the control core continues execution
// while the vector engine runs.  Use KWAIT to synchronize.
let hasSideEffects = 1, isCall = 1 in
def KLAUNCH : NPUInst<(outs), (ins GPR:$rs1, i64imm:$grid),
    "KLAUNCH $rs1, $grid", []> {
  let SchedRW = [WriteCall];
}

// KWAIT — stall the control core until all dispatched kernels have finished.
// Equivalent to a full barrier on the vector engine.
let hasSideEffects = 1 in
def KWAIT : NPUInst<(outs), (ins), "KWAIT", []> {
  let SchedRW = [WriteBranch];
}

// Patterns that match @llvm.npu.klaunch / @llvm.npu.kwait intrinsic calls
// in the selection DAG.  The intrinsic args (kernel_id i32, grid_size i32)
// map to rs1 (GPR) and imm (i64imm) respectively.
def : Pat<(int_npu_kwait), (KWAIT)>;

//===----------------------------------------------------------------------===//
// Call-frame setup/teardown pseudos.
//===----------------------------------------------------------------------===//

let Defs = [X2], Uses = [X2] in {
  def ADJCALLSTACKDOWN : Pseudo<(outs), (ins i64imm:$amt1, i64imm:$amt2),
                                [(callseq_start timm:$amt1, timm:$amt2)]>;
  def ADJCALLSTACKUP   : Pseudo<(outs), (ins i64imm:$amt1, i64imm:$amt2),
                                [(callseq_end timm:$amt1, timm:$amt2)]>;
}

