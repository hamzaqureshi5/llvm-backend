; RUN: llc -mtriple=npu -o - %s | FileCheck %s

; Basic integer add — exercises arg passing (CC_NPU), ADD instruction, and return.
define i64 @add_i64(i64 %a, i64 %b) {
; CHECK-LABEL: add_i64:
; CHECK: ADD
  %r = add i64 %a, %b
  ret i64 %r
}

; Constant fold: returning a constant exercises LI (load-immediate).
define i64 @const42() {
; CHECK-LABEL: const42:
; CHECK: LI
  ret i64 42
}
