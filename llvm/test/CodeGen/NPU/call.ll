; RUN: llc -mtriple=npu < %s | FileCheck %s

declare i64 @callee(i64 %x)

; A direct call links via "JAL x1, callee"; the return uses "JALR x0, x1, 0".

; CHECK-LABEL: caller:
; CHECK: JAL x1, callee
; CHECK: JALR x0, x1, 0
define i64 @caller(i64 %x) {
  %r = call i64 @callee(i64 %x)
  ret i64 %r
}
