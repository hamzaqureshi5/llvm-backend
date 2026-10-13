//===- NPUCallingConv.td - Calling Conventions NPU ---------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This describes the (invented, RISC-V-style) calling convention for the NPU
// control core. The simulator has no documented ABI; this exists purely so
// LLVM can lower calls/returns into plausible assembly.
//
//===----------------------------------------------------------------------===//

// NPU 64-bit C Calling convention.
def CC_NPU : CallingConv<[
  // Promote sub-word integers to i64 so the control core always operates on
  // native 64-bit values.
  CCIfType<[i8, i16, i32], CCPromoteToType<i64>>,

  // Integer arguments: x10..x17 (RISC-V-style).
  CCIfType<[i64], CCAssignToReg<[X10, X11, X12, X13, X14, X15, X16, X17]>>,

  // Vector-core arguments: <16 x float> values go in v0..v7.
  // These are vector engine registers (VR file), separate from the control core.
  CCIfType<[v16f32], CCAssignToReg<[V0, V1, V2, V3, V4, V5, V6, V7]>>,

  // Scalar float arguments (f32, from reduction results): s2..s9.
  CCIfType<[f32], CCAssignToReg<[S2, S3, S4, S5, S6, S7, S8, S9]>>,

  // Everything else is passed on the stack in 8-byte aligned slots.
  CCAssignToStack<8, 8>
]>;

// NPU 64-bit C return-value convention.
def RetCC_NPU : CallingConv<[
  // Promote sub-word ints to i64.
  CCIfType<[i8, i16, i32], CCPromoteToType<i64>>,

  // Integer return values: x10 (+ x11 for a 128-bit pair).
  CCIfType<[i64], CCAssignToReg<[X10, X11]>>,

  // Vector return value: a single <16 x float> in v0.
  CCIfType<[v16f32], CCAssignToReg<[V0]>>,

  // Scalar float return value: f32 (from reductions) in s2.
  CCIfType<[f32], CCAssignToReg<[S2]>>
]>;

// Callee-saved registers: return address + x18..x27.
def CSR : CalleeSavedRegs<(add X1, (sequence "X%i", 18, 27))>;
