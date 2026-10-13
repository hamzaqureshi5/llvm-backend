; vadd.ll — element-wise int32 vector addition kernel for the NPU.
;
; Compile:
;   llc -mtriple=npu -o vadd.s vadd.ll            (view assembly)
;   llc -mtriple=npu -filetype=obj -o vadd.o vadd.ll
;
; Calling convention: i32* args arrive in x10–x12, i64 count in x13.
; All i32 arithmetic is type-legalized to i64 (the only native width);
; loads are matched as extloadi32 → LW (sign-extends, upper bits unused).

; void vadd(i32 *c, i32 *a, i32 *b, i64 n)
;   c[i] = a[i] + b[i]  for i in [0, n)
define void @vadd(i32* %c, i32* %a, i32* %b, i64 %n) {
entry:
  br label %loop

loop:
  %a_cur  = phi i32* [ %a, %entry ], [ %a_next, %loop ]
  %b_cur  = phi i32* [ %b, %entry ], [ %b_next, %loop ]
  %c_cur  = phi i32* [ %c, %entry ], [ %c_next, %loop ]
  %i      = phi i64  [ 0, %entry  ], [ %i1,     %loop ]
  %av     = load i32, i32* %a_cur
  %bv     = load i32, i32* %b_cur
  %cv     = add i32 %av, %bv
  store i32 %cv, i32* %c_cur
  %a_next = getelementptr i32, i32* %a_cur, i64 1
  %b_next = getelementptr i32, i32* %b_cur, i64 1
  %c_next = getelementptr i32, i32* %c_cur, i64 1
  %i1     = add i64 %i, 1
  %done   = icmp eq i64 %i1, %n
  br i1 %done, label %exit, label %loop

exit:
  ret void
}
