//===- NPUVectorRegisterInfo.td - NPU vector/engine register defs --*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Register definitions for the NPU's two data-plane register files, matching
// npu.h exactly:
//
//   VR0–VR31  — vector registers, each holding VL floats (max VLEN_MAX=256).
//               Modeled here as v16f32 (HW_LANES=16 physical lanes).
//
//   MR0–MR7   — predicate/mask registers, one bit per active lane.
//               Modeled as i32 (bits 0..VL-1 are live; upper bits zero).
//
//   SR0–SR31  — vector-engine scalar registers, 32-bit int or float.
//               Used for SIMT addressing, loop indices, and kernel args.
//               s0 = grid instance id (set by KLAUNCH before vector kernel entry).
//               s1 = kernel arguments base pointer (also set by KLAUNCH).
//
// These registers live entirely within the vector/tensor engine and are
// invisible to the control core.  They are not preserved across kernel
// launches; callee/caller save conventions do not apply.
//
//===----------------------------------------------------------------------===//

// ── NPU base register class ───────────────────────────────────────────────────

class NPUVReg<bits<5> num, string n> : Register<n> {
  field bits<5> Num;
  let Num = num;
  let Namespace = "NPU";
}

// ── Vector registers v0..v31 ──────────────────────────────────────────────────
// Each register holds HW_LANES=16 float elements (the physical SIMD width).
// LLVM represents them as the fixed-width type v16f32.
// DWARF numbering: 64+i (arbitrary; no ABI standard exists for this target).

foreach i = 0-31 in {
  def V#i : NPUVReg<i, "v"#i>, DwarfRegNum<[!add(64, i)]>;
}

// All 32 vector registers in a single register class.
// The allocation order mirrors the engine scalar convention: v0/v1 are
// traditional return/scratch registers, v2–v7 are temporaries, v8–v31
// are caller-saved (the kernel resets all state at entry anyway).
def VReg : RegisterClass<"NPU", [v16f32], 64,
    (add (sequence "V%i", 0, 31))>;

// ── Mask registers m0..m7 ─────────────────────────────────────────────────────
// Each mask register holds one bit per active lane (VL bits, max 256).
// Modeled as i32 to simplify legalization; bits above VL are defined as zero.
// DWARF numbering: 96+i.

foreach i = 0-7 in {
  def M#i : NPUVReg<i, "m"#i>, DwarfRegNum<[!add(96, i)]>;
}

// Predicate register class.
def MReg : RegisterClass<"NPU", [i32], 32,
    (add (sequence "M%i", 0, 7))>;

// ── Engine scalar registers s0..s31 ──────────────────────────────────────────
// 32-bit scalar registers used inside vector kernels for address computation
// and loop control.  Values are 32-bit floats when used with SLF/SRECIP, or
// 32-bit ints when used with SLI/SADD/SADDI/SMUL/SMULI/SLW.
// LLVM uses f32 as the canonical type (the simulator's sreg_t is a union).
//
// Special roles (set by the hardware at kernel entry; read-only by convention):
//   s0 = grid instance id (0 .. grid_size-1)
//   s1 = kernel arguments base address (ptr into NPU memory)
// DWARF numbering: 128+i.

foreach i = 0-31 in {
  def S#i : NPUVReg<i, "s"#i>, DwarfRegNum<[!add(128, i)]>;
}

// Engine scalar register class.
def SReg : RegisterClass<"NPU", [f32], 32,
    (add (sequence "S%i", 2, 31),   // s2..s31 are allocatable
         S0,                         // s0 = grid id, kept last (reserved)
         S1                          // s1 = args ptr, kept last (reserved)
    )>;
