; RUN: llc -mtriple=npu < %s | FileCheck %s

; Return an immediate: materialize with LI into the return register x10,
; then return via "JALR x0, x1, 0". A trailing HALT terminates the program.

; CHECK-LABEL: f:
; CHECK: LI x10, 42
; CHECK: JALR x0, x1, 0
; CHECK: HALT
define i64 @f() {
  ret i64 42
}
