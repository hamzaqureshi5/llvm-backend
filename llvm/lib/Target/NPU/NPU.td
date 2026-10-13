//===- NPU.td - Describe the NPU Target Machine ------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Top-level TableGen description of the NPU target.  The NPU is a three-layer
// programmable accelerator (see npu-simulator/fully_prog_npu.md):
//
//   Layer 1 — Control core  : scalar RISC-V-like ISA (NPUInstrInfo.td)
//   Layer 2 — Tensor engine : dense tile matmul fast path (NPUTensorInstrInfo.td)
//   Layer 3 — Vector core   : per-grid-point SIMT kernels (NPUVectorInstrInfo.td)
//
// All three layers share this single LLVM target definition; they are selected
// by different intrinsic / instruction patterns at the ISelDAGToDAG level.
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Target-independent interfaces which we are implementing
//===----------------------------------------------------------------------===//

include "llvm/Target/Target.td"

//===----------------------------------------------------------------------===//
// Register Files
//===----------------------------------------------------------------------===//

// Control core: x0..x31 (64-bit general-purpose integers).
include "NPURegisterInfo.td"

// Vector/tensor engine data planes:
//   v0..v31 (VR, v16f32), m0..m7 (MR, i32), s0..s31 (SR, f32).
include "NPUVectorRegisterInfo.td"

//===----------------------------------------------------------------------===//
// Calling convention (control core ABI)
//===----------------------------------------------------------------------===//

include "NPUCallingConv.td"

//===----------------------------------------------------------------------===//
// Scheduling model (must precede instruction descriptions so that SchedRW
// annotations like [WriteALU] can reference the names defined here).
//===----------------------------------------------------------------------===//

include "NPUSchedule.td"

//===----------------------------------------------------------------------===//
// Instruction descriptions
//===----------------------------------------------------------------------===//

// Layer 1: scalar control-core instructions (ALU, load/store, branches, calls,
// KLAUNCH, KWAIT, HALT).
include "NPUInstrInfo.td"

// Layer 2: tensor-engine instructions (CFG, MMA/MATMUL, LOAD/STORE, DMA, etc.).
include "NPUTensorInstrInfo.td"

// Layer 3: vector-core instructions (VLOAD/VSTORE, VADD/VMUL/etc., reductions,
// transcendentals, predication, kernel branches).
include "NPUVectorInstrInfo.td"

defm : RemapAllTargetPseudoPointerOperands<GPR>;

def NPUInstrInfo : InstrInfo;

//===----------------------------------------------------------------------===//
// NPU processor variants.
//===----------------------------------------------------------------------===//

// "generic" is the only CPU for now; it uses the full NPUSchedModel defined
// in NPUSchedule.td.  Future variants (e.g. "npu-v2") would add feature bits.
def : ProcessorModel<"generic", NPUSchedModel, []>;

//===----------------------------------------------------------------------===//
// Assembly writer
//===----------------------------------------------------------------------===//

def NPUInstPrinter : AsmWriter {
  string AsmWriterClassName = "InstPrinter";
  bit isMCAsmWriter = 1;
}

//===----------------------------------------------------------------------===//
// Assembly parser
//===----------------------------------------------------------------------===//

// The hand-written parser (AsmParser/NPUAsmParser.cpp) accepts both the
// control-core text syntax and the vector-core syntax produced by the
// simulator's disassembler, enabling round-trip verification.
def NPUAsmParser : AsmParser {
  // We do NOT auto-generate register matching because the NPU has three
  // register namespaces (x, v, m, s) that need context-sensitive dispatch.
  bit ShouldEmitMatchRegisterName = 0;
}

//===----------------------------------------------------------------------===//
// Declare the target which we are implementing
//===----------------------------------------------------------------------===//

def NPU : Target {
  let InstructionSet = NPUInstrInfo;
  let AssemblyWriters = [NPUInstPrinter];
  let AssemblyParsers = [NPUAsmParser];
}
