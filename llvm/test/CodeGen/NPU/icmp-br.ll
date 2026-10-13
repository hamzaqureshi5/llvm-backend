; RUN: llc -mtriple=npu < %s | FileCheck %s

; Conditional branches fuse the compare and branch into BEQ/BNE/BLT/BGE.
; LLVM inverts the condition so the taken ("true") block falls through, so
; e.g. "icmp eq + br" emits the *inverted* BNE to the false block. "sgt" is
; realized by swapping the operands of BLT/BGE.

; CHECK-LABEL: br_eq:
; CHECK: BNE x10, x11,
define i64 @br_eq(i64 %a, i64 %b) {
entry:
  %c = icmp eq i64 %a, %b
  br i1 %c, label %t, label %f
t:
  ret i64 1
f:
  ret i64 0
}

; CHECK-LABEL: br_slt:
; CHECK: BGE x10, x11,
define i64 @br_slt(i64 %a, i64 %b) {
entry:
  %c = icmp slt i64 %a, %b
  br i1 %c, label %t, label %f
t:
  ret i64 1
f:
  ret i64 0
}

; CHECK-LABEL: br_sgt:
; CHECK: BGE x11, x10,
define i64 @br_sgt(i64 %a, i64 %b) {
entry:
  %c = icmp sgt i64 %a, %b
  br i1 %c, label %t, label %f
t:
  ret i64 1
f:
  ret i64 0
}
