; RUN: llc -mtriple=npu-unknown-unknown -o - %s | FileCheck %s
;
; Layer 2 (tensor engine) intrinsic lowering tests.
; Each function tests one instruction or a small chain.
; Tile register numbers must be compile-time constants — the T_* instruction
; format encodes them as 3-bit immediate fields, not general-purpose registers.

declare void @llvm.npu.cfg.shape(i64, i64, i64)
declare i64 @llvm.npu.acc.clr(i64)
declare i64 @llvm.npu.mma(i64, i64, i64)
declare i64 @llvm.npu.matmul(i64, i64, i64)
declare i64 @llvm.npu.dot(i64, i64, i64)
declare i64 @llvm.npu.load(i64, ptr, i64)
declare void @llvm.npu.store(ptr, i64, i64)
declare void @llvm.npu.fence()
declare void @llvm.npu.barrier()
declare i64 @llvm.npu.relu(i64)
declare i64 @llvm.npu.scale(i64, float)
declare i64 @llvm.npu.bcast(i64, float)

; --- cfg_shape ----------------------------------------------------------

; CHECK-LABEL: test_cfg_shape:
; CHECK: CFG.SHAPE 4, 4, 4
define void @test_cfg_shape() {
  call void @llvm.npu.cfg.shape(i64 4, i64 4, i64 4)
  ret void
}

; --- acc_clr ------------------------------------------------------------

; CHECK-LABEL: test_acc_clr:
; CHECK: ACC.CLR t0
define i64 @test_acc_clr() {
  %r = call i64 @llvm.npu.acc.clr(i64 0)
  ret i64 %r
}

; --- matmul (load → matmul → store chain) --------------------------------

; CHECK-LABEL: test_matmul:
; CHECK-DAG: CFG.SHAPE 4, 4, 4
; CHECK-DAG: LOAD t0,
; CHECK-DAG: LOAD t1,
; CHECK:     MATMUL t2, t0, t1
; CHECK:     STORE t2,
; CHECK:     FENCE
define void @test_matmul(ptr %a, ptr %b, ptr %c) {
  call void @llvm.npu.cfg.shape(i64 4, i64 4, i64 4)
  %ta = call i64 @llvm.npu.load(i64 0, ptr %a, i64 16)
  %tb = call i64 @llvm.npu.load(i64 1, ptr %b, i64 16)
  %tc = call i64 @llvm.npu.matmul(i64 2, i64 %ta, i64 %tb)
  call void @llvm.npu.store(ptr %c, i64 %tc, i64 16)
  call void @llvm.npu.fence()
  ret void
}

; --- mma chain (two accumulating MMA ops) --------------------------------

; CHECK-LABEL: test_mma_chain:
; CHECK: ACC.CLR t0
; CHECK: MMA t0, t1, t2
; CHECK: MMA t0, t3, t4
define i64 @test_mma_chain() {
  %t0 = call i64 @llvm.npu.acc.clr(i64 0)
  %t1 = call i64 @llvm.npu.mma(i64 %t0, i64 1, i64 2)
  %t2 = call i64 @llvm.npu.mma(i64 %t1, i64 3, i64 4)
  ret i64 %t2
}

; --- dot product (CF_T3 format: dt = destination AND accumulator) --------
; Pass %ta (the acc.clr SSA token = tile 0) as dt to dot.
; getTileConst follows the INTRINSIC_W_CHAIN → extracts constant 0.

; CHECK-LABEL: test_dot:
; CHECK: ACC.CLR t0
; CHECK: DOT t0, t1, t2
define i64 @test_dot() {
  %ta = call i64 @llvm.npu.acc.clr(i64 0)
  %r  = call i64 @llvm.npu.dot(i64 %ta, i64 1, i64 2)
  ret i64 %r
}

; --- relu + scale --------------------------------------------------------

; CHECK-LABEL: test_relu_scale:
; CHECK: RELU t3, t3
; CHECK: SCALE t3, t3, 2.000000e+00
define i64 @test_relu_scale() {
  %r = call i64 @llvm.npu.relu(i64 3)
  %s = call i64 @llvm.npu.scale(i64 %r, float 2.0)
  ret i64 %s
}

; --- bcast ---------------------------------------------------------------
; T_BCAST reads the float bits from a GPR (materialized via LI), NOT an fimm.
; 3.14f = 0x4048F5C3 = 1078523331 decimal.

; CHECK-LABEL: test_bcast:
; CHECK: LI {{x[0-9]+}}, 1078523331
; CHECK: BCAST t2, {{x[0-9]+}}
define i64 @test_bcast() {
  %r = call i64 @llvm.npu.bcast(i64 2, float 3.14)
  ret i64 %r
}
