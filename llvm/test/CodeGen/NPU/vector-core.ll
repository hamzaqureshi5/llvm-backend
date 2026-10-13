; RUN: llc -mtriple=npu-unknown-unknown -o - %s | FileCheck %s
;
; Layer 3 (vector core) intrinsic lowering tests.
; <16 x float> maps to NPU's HW_LANES=16 physical vector registers.

declare <16 x float> @llvm.npu.vload(ptr, i64)
declare void @llvm.npu.vstore(ptr, <16 x float>, i64)
declare <16 x float> @llvm.npu.vexp(<16 x float>)
declare float @llvm.npu.vredsum(<16 x float>)
declare float @llvm.npu.vredmax(<16 x float>)
declare i64 @llvm.npu.vsetvl(i64)

; --- vload / vstore ------------------------------------------------------

; CHECK-LABEL: test_vload_vstore:
; CHECK: VLOAD v0,
; CHECK: VSTORE
define void @test_vload_vstore(ptr %src, ptr %dst) {
  %v = call <16 x float> @llvm.npu.vload(ptr %src, i64 4)
  call void @llvm.npu.vstore(ptr %dst, <16 x float> %v, i64 4)
  ret void
}

; --- vector arithmetic (native LLVM fadd → VADD) ------------------------

; CHECK-LABEL: test_vadd:
; CHECK: VADD v0, v0, v1
define <16 x float> @test_vadd(<16 x float> %a, <16 x float> %b) {
  %r = fadd <16 x float> %a, %b
  ret <16 x float> %r
}

; --- transcendental (vexp) + reduction (vredsum) -------------------------

; CHECK-LABEL: test_softmax_sum:
; CHECK: VLOAD v0,
; CHECK: VEXP v0, v0
; CHECK: VREDSUM
define float @test_softmax_sum(ptr %p) {
  %v = call <16 x float> @llvm.npu.vload(ptr %p, i64 4)
  %e = call <16 x float> @llvm.npu.vexp(<16 x float> %v)
  %s = call float @llvm.npu.vredsum(<16 x float> %e)
  ret float %s
}

; --- vsetvl (constant length) --------------------------------------------

; CHECK-LABEL: test_vsetvl_const:
; CHECK: LI {{x[0-9]+}}, 16
; CHECK: VSETVL {{x[0-9]+}}, {{x[0-9]+}}
define i64 @test_vsetvl_const() {
  %vl = call i64 @llvm.npu.vsetvl(i64 16)
  ret i64 %vl
}

; --- vsetvl (runtime length from argument) --------------------------------

; CHECK-LABEL: test_vsetvl_dynamic:
; CHECK: VSETVL {{x[0-9]+}}, {{x[0-9]+}}
define i64 @test_vsetvl_dynamic(i64 %len) {
  %vl = call i64 @llvm.npu.vsetvl(i64 %len)
  ret i64 %vl
}
