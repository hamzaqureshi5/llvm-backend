//===- NPUVectorInstrInfo.td - NPU vector-core instructions ---*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Instruction definitions for Layer 3: the NPU Vector Core (Sections 9–12 of
// fully_prog_npu.md / npu.h vop_t enum).
//
// The vector core runs as a SIMT (Single Instruction, Multiple Threads) engine:
// one instruction dispatches across HW_LANES=16 physical lanes simultaneously.
// Each grid instance launched by C_KLAUNCH starts executing the vector kernel
// (vinst[]) registered with npu_register_kernel().
//
// Register files available in the vector core:
//   VR   (v0..v31) — vector registers, v16f32 in LLVM
//   MR   (m0..m7)  — predicate mask registers, i32 in LLVM
//   SR   (s0..s31) — engine scalar registers, f32 in LLVM
//
// Assembly syntax mirrors the simulator's VTAB table so output can be
// verified by npu_asm_vector() / npu_disasm_vinst().
//
//===----------------------------------------------------------------------===//

// ── Section 9 : Registers and memory ─────────────────────────────────────────

// Set the active vector length.  All subsequent vector ops use VL active lanes.
// "VSETVL xD, xS" — VL is an integer count; both operands are GPRs (i64).
// Simulator: n->s[I->rd].i = min(n->s[I->rs1].i, VLEN_MAX), accessed via getx()
// which reads the integer reg file — so GPR (i64), not SReg (f32).
def V_VSETVL : NPUInst<(outs GPR:$rd), (ins GPR:$rs),
    "VSETVL $rd, $rs", []> {
  let hasSideEffects = 1;
  let SchedRW = [WriteVectorMisc];
}

// Vector load: vD = mem[base + i*stride], i in [0..VL).
// "VLOAD vD, imm(xBase)" — base address comes from a GPR (i64 pointer).
// Using GPR (not SReg) for the base so that LLVM IR pointers (i64 / ptr)
// lower naturally without a type-mismatch bitcast through the scalar reg file.
def V_VLOAD : NPUInst<(outs VReg:$vd), (ins GPR:$rs, i64imm:$imm),
    "VLOAD $vd, ${imm}(${rs})", []> {
  let mayLoad = 1;
  let SchedRW = [WriteVectorMem];
}

// Vector store: mem[base + i*stride] = vS[i].
def V_VSTORE : NPUInst<(outs), (ins VReg:$vs, GPR:$rs, i64imm:$imm),
    "VSTORE $vs, ${imm}(${rs})", []> {
  let mayStore = 1;
  let SchedRW = [WriteVectorMem];
}

// Gather: vD[i] = mem[base + indexV[i]].
def V_VGATHER : NPUInst<(outs VReg:$vd), (ins GPR:$rs, i64imm:$imm, VReg:$vs1),
    "VGATHER $vd, ${imm}(${rs}), $vs1", []> {
  let mayLoad = 1;
  let SchedRW = [WriteVectorMem];
}

// Scatter: mem[base + indexV[i]] = vS[i].
def V_VSCATTER : NPUInst<(outs), (ins VReg:$vs, GPR:$rs, i64imm:$imm, VReg:$vs1),
    "VSCATTER $vs, ${imm}(${rs}), $vs1", []> {
  let mayStore = 1;
  let SchedRW = [WriteVectorMem];
}

// Move: vD = vS (register-to-register copy).
def V_VMOV : NPUInst<(outs VReg:$vd), (ins VReg:$vs),
    "VMOV $vd, $vs", []> {
  let SchedRW = [WriteVectorMisc];
}

// Broadcast: vD[i] = sS for all i in [0..VL).
// Used to splat a scalar into all lanes before elementwise vector ops.
def V_VBCAST : NPUInst<(outs VReg:$vd), (ins SReg:$rs),
    "VBCAST $vd, $rs", []> {
  let SchedRW = [WriteVectorMisc];
}

// ── Engine scalar register instructions ──────────────────────────────────────
// These scalar ops run inside the vector kernel for address calculation.
// They use SR registers (s0..s31), not the control core's GPR registers.

// Load scalar immediate: sD = imm.
def V_SLI : NPUInst<(outs SReg:$rd), (ins i64imm:$imm),
    "SLI $rd, $imm", []> {
  let SchedRW = [WriteVectorMisc];
}

// Load float immediate: sD = fimm.
def V_SFI : NPUInst<(outs SReg:$rd), (ins f64imm:$fimm),
    "SFI $rd, $fimm", []> {
  let SchedRW = [WriteVectorMisc];
}

// Three-register scalar arithmetic.
def V_SADD : NPUInst<(outs SReg:$rd), (ins SReg:$rs1, SReg:$rs2),
    "SADD $rd, $rs1, $rs2", []> {
  let SchedRW = [WriteVectorMisc];
}
def V_SSUB : NPUInst<(outs SReg:$rd), (ins SReg:$rs1, SReg:$rs2),
    "SSUB $rd, $rs1, $rs2", []> {
  let SchedRW = [WriteVectorMisc];
}
def V_SMUL : NPUInst<(outs SReg:$rd), (ins SReg:$rs1, SReg:$rs2),
    "SMUL $rd, $rs1, $rs2", []> {
  let SchedRW = [WriteVectorMisc];
}

// Register-immediate scalar arithmetic.
def V_SADDI : NPUInst<(outs SReg:$rd), (ins SReg:$rs, i64imm:$imm),
    "SADDI $rd, $rs, $imm", []> {
  let SchedRW = [WriteVectorMisc];
}
def V_SMULI : NPUInst<(outs SReg:$rd), (ins SReg:$rs, i64imm:$imm),
    "SMULI $rd, $rs, $imm", []> {
  let SchedRW = [WriteVectorMisc];
}

// Scalar loads from NPU memory.
def V_SLW : NPUInst<(outs SReg:$rd), (ins SReg:$rs, i64imm:$imm),
    "SLW $rd, ${imm}(${rs})", []> {
  let mayLoad = 1;
  let SchedRW = [WriteVectorMem];
}
def V_SLF : NPUInst<(outs SReg:$rd), (ins SReg:$rs, i64imm:$imm),
    "SLF $rd, ${imm}(${rs})", []> {
  let mayLoad = 1;
  let SchedRW = [WriteVectorMem];
}

// Scalar reciprocal: sD = 1.0 / sS.
def V_SRECIP : NPUInst<(outs SReg:$rd), (ins SReg:$rs),
    "SRECIP $rd, $rs", []> {
  let SchedRW = [WriteVectorMisc];
}

// Read grid instance id into a scalar register: sD = current instance index.
// This is always available as s0 on entry, but KID lets you re-read it later.
def V_KID : NPUInst<(outs SReg:$rd), (ins),
    "KID $rd", []> {
  let SchedRW = [WriteVectorMisc];
}

// ── Section 10 : Compute primitives ──────────────────────────────────────────
// Elementwise float ops.  These have matching patterns against native LLVM
// <16 x float> IR nodes so that vAdd/vMul/etc. in IR lower automatically.

// Three-vector operand form: vD = vS1 OP vS2.
class VALURR<string mn, SDPatternOperator op>
    : NPUInst<(outs VReg:$vd), (ins VReg:$vs1, VReg:$vs2),
              mn # " $vd, $vs1, $vs2",
              [(set v16f32:$vd, (op v16f32:$vs1, v16f32:$vs2))]> {
  let SchedRW = [WriteVectorALU];
}

def V_VADD  : VALURR<"VADD", fadd>;
def V_VSUB  : VALURR<"VSUB", fsub>;
def V_VMUL  : VALURR<"VMUL", fmul>;
def V_VDIV  : VALURR<"VDIV", fdiv>;
def V_VMAX  : VALURR<"VMAX", fmaxnum>;
def V_VMIN  : VALURR<"VMIN", fminnum>;

// Fused multiply-add: vD = vS1 * vS2 + vD (destructive on destination).
def V_VFMA : NPUInst<(outs VReg:$vd), (ins VReg:$vs1, VReg:$vs2, VReg:$vs),
    "VFMA $vd, $vs1, $vs2",
    [(set v16f32:$vd, (fma v16f32:$vs1, v16f32:$vs2, v16f32:$vs))]> {
  let SchedRW = [WriteVectorFMA];
  let Constraints = "$vs = $vd";  // vs is also the accumulator input
}

// Two-vector operand form (unary): vD = OP vS.
class VALUV<string mn, SDPatternOperator op>
    : NPUInst<(outs VReg:$vd), (ins VReg:$vs),
              mn # " $vd, $vs",
              [(set v16f32:$vd, (op v16f32:$vs))]> {
  let SchedRW = [WriteVectorALU];
}

def V_VABS  : VALUV<"VABS",  fabs>;
def V_VNEG  : VALUV<"VNEG",  fneg>;

// Comparison: mD = vS1 cmp vS2 (result is a mask register, one bit per lane).
// The CMP kind is encoded as a 3-bit field; see VCMP_KIND_* in the MCCodeEmitter.
def V_VCMP : NPUInst<(outs MReg:$md), (ins VReg:$vs1, VReg:$vs2, i32imm:$cmp),
    "VCMP $md, $vs1, $vs2", []> {
  let SchedRW = [WriteVectorMisc];
}

// Select: vD[i] = (mask[i]) ? vS1[i] : vS2[i].
// No TableGen pattern: LLVM's vselect expects a vector boolean mask (v16i1),
// but the NPU encodes the mask as a scalar i32 bitmask in MReg.  Lowering
// is handled explicitly in NPUISelLowering.cpp via LowerIntrinsicCall.
def V_VSEL : NPUInst<(outs VReg:$vd), (ins VReg:$vs1, VReg:$vs2, MReg:$mask),
    "VSEL $vd, $vs1, $vs2, $mask", []> {
  let SchedRW = [WriteVectorMisc];
}

// Convert element type: vD = VCVT.MODE vS.
// Mode is encoded as cvtmode_t (CVT_F32=0, CVT_I8=1, CVT_I32=2).
def V_VCVT : NPUInst<(outs VReg:$vd), (ins VReg:$vs, i32imm:$cvt),
    "VCVT $vd, $vs", []> {
  let SchedRW = [WriteVectorMisc];
}

// ── Transcendental intrinsic instructions ─────────────────────────────────────
// These match @llvm.npu.v* intrinsic calls in the SelectionDAG.

class VectorTrans<string mn, Intrinsic intr>
    : NPUInst<(outs VReg:$vd), (ins VReg:$vs),
              mn # " $vd, $vs",
              [(set v16f32:$vd, (intr v16f32:$vs))]> {
  let SchedRW = [WriteVectorTrans];
}

def V_VEXP    : VectorTrans<"VEXP",    int_npu_vexp>;
def V_VLOG    : VectorTrans<"VLOG",    int_npu_vlog>;
def V_VRSQRT  : VectorTrans<"VRSQRT", int_npu_vrsqrt>;
def V_VSQRT   : VectorTrans<"VSQRT",  int_npu_vsqrt>;
def V_VRECIP  : VectorTrans<"VRECIP", int_npu_vrecip>;

// ── Section 11 : Reductions ───────────────────────────────────────────────────
// Reduce all VL active lanes to a single engine-scalar value in sD.
// The result is written to an SR register (f32).

class VectorReduce<string mn, Intrinsic intr>
    : NPUInst<(outs SReg:$rd), (ins VReg:$vs),
              mn # " $rd, $vs",
              [(set f32:$rd, (intr v16f32:$vs))]> {
  let SchedRW = [WriteVectorReduce];
}

def V_VREDSUM : VectorReduce<"VREDSUM", int_npu_vredsum>;
def V_VREDMAX : VectorReduce<"VREDMAX", int_npu_vredmax>;
def V_VREDMIN : VectorReduce<"VREDMIN", int_npu_vredmin>;
def V_VREDAND : VectorReduce<"VREDAND", int_npu_vredand>;
def V_VREDOR  : VectorReduce<"VREDOR",  int_npu_vredor>;

// Inclusive prefix scan: vD[i] = sum(vS[0..i]).  Maps to V_VSCAN.
def V_VSCAN : NPUInst<(outs VReg:$vd), (ins VReg:$vs),
    "VSCAN $vd, $vs", []> {
  let SchedRW = [WriteVectorReduce];
}

// ── Section 12 : Predication and control flow ─────────────────────────────────
// Predicate registers gate per-lane execution.  @m<n> prefix on any vector
// instruction (except VSETVL) enables predicated execution; the MCCodeEmitter
// sets use_mask=1 and mask=n in the vinst encoding.

// Set mask from vector: mD[i] = (vS[i] != 0.0).
def V_VMASK_SET : NPUInst<(outs MReg:$md), (ins VReg:$vs),
    "VMASK.SET $md, $vs", []> {
  let SchedRW = [WriteVectorMask];
}

// Mask AND: mD = m1 & m2.
def V_VMASK_AND : NPUInst<(outs MReg:$md), (ins MReg:$m1, MReg:$m2),
    "VMASK.AND $md, $m1, $m2", []> {
  let SchedRW = [WriteVectorMask];
}

// Mask OR: mD = m1 | m2.
def V_VMASK_OR : NPUInst<(outs MReg:$md), (ins MReg:$m1, MReg:$m2),
    "VMASK.OR $md, $m1, $m2", []> {
  let SchedRW = [WriteVectorMask];
}

// Mask NOT: mD = ~m1.
def V_VMASK_NOT : NPUInst<(outs MReg:$md), (ins MReg:$m1),
    "VMASK.NOT $md, $m1", []> {
  let SchedRW = [WriteVectorMask];
}

// Unconditional kernel branch.  target is a per-kernel instruction index.
let isBranch = 1, isTerminator = 1, isBarrier = 1 in
def V_BR : NPUInst<(outs), (ins brtarget:$dst), "BR $dst", [(br bb:$dst)]> {
  let SchedRW = [WriteVectorMisc];
}

// Branch if engine scalar sS == 0.
let isBranch = 1, isTerminator = 1 in
def V_BRZ : NPUInst<(outs), (ins SReg:$rs, brtarget:$dst),
    "BRZ $rs, $dst", []> {
  let SchedRW = [WriteVectorMisc];
}

// Branch if engine scalar sS != 0.
let isBranch = 1, isTerminator = 1 in
def V_BRNZ : NPUInst<(outs), (ins SReg:$rs, brtarget:$dst),
    "BRNZ $rs, $dst", []> {
  let SchedRW = [WriteVectorMisc];
}

// Counted loop: decrement sS, branch if sS != 0.
// Equivalent to "SADDI sS, sS, -1; BRNZ sS, dst" but atomically.
let isBranch = 1, isTerminator = 1 in
def V_LOOP : NPUInst<(outs), (ins SReg:$rs, brtarget:$dst),
    "LOOP $rs, $dst", []> {
  let SchedRW = [WriteVectorMisc];
}

// Kernel exit: terminate this grid instance.
let isReturn = 1, isTerminator = 1, isBarrier = 1 in
def V_KEXIT : NPUInst<(outs), (ins), "KEXIT", []> {
  let SchedRW = [WriteVectorMisc];
}

// VLOAD / VSTORE DAG patterns are handled via LowerIntrinsicCall in
// NPUISelLowering.cpp (emitting NPUISD::VLOAD / NPUISD::VSTORE), then selected
// in NPUISelDAGToDAG.cpp to the V_VLOAD / V_VSTORE machine instructions.
// The base address is a GPR (i64 pointer), matching the intrinsic ptr arg.
