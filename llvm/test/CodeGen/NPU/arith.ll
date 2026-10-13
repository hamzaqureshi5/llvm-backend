; RUN: llc -mtriple=npu < %s | FileCheck %s

; Arguments arrive in x10, x11; the result is returned in x10.

; CHECK-LABEL: add:
; CHECK: ADD x10, x10, x11
define i64 @add(i64 %a, i64 %b) {
  %c = add i64 %a, %b
  ret i64 %c
}

; CHECK-LABEL: sub:
; CHECK: SUB x10, x10, x11
define i64 @sub(i64 %a, i64 %b) {
  %c = sub i64 %a, %b
  ret i64 %c
}

; CHECK-LABEL: mul:
; CHECK: MUL x10, x10, x11
define i64 @mul(i64 %a, i64 %b) {
  %c = mul i64 %a, %b
  ret i64 %c
}

; CHECK-LABEL: andor:
; CHECK-DAG: AND x{{[0-9]+}}, x10, x11
; CHECK-DAG: OR x{{[0-9]+}}, x10, x11
; CHECK: SUB x10, x{{[0-9]+}}, x{{[0-9]+}}
define i64 @andor(i64 %a, i64 %b) {
  %x = and i64 %a, %b
  %y = or i64 %a, %b
  %z = sub i64 %y, %x
  ret i64 %z
}

; CHECK-LABEL: shifts:
; CHECK: SLL x{{[0-9]+}}, x10, x11
; CHECK: SRL x10, x{{[0-9]+}}, x11
define i64 @shifts(i64 %a, i64 %b) {
  %x = shl i64 %a, %b
  %y = lshr i64 %x, %b
  ret i64 %y
}

; CHECK-LABEL: addimm:
; CHECK: ADDI x10, x10, 7
define i64 @addimm(i64 %a) {
  %c = add i64 %a, 7
  ret i64 %c
}
