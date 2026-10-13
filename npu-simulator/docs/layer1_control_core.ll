; layer1_control_core.ll — Complete Layer 1 (control-core) LLVM IR reference.
;
; Covers every instruction class the NPU control core supports:
;   §1 Scalar arithmetic : ADD SUB MUL AND OR XOR SLL SRL SLT ADDI LI
;   §2 Memory            : LW LH LB SW SH SB  (i32/i16/i8 only — no i64 load)
;   §3 Control flow      : BEQ BNE BLT BGE (via BR_CC) + loop + function call
;
; Compile:
;   llc -march=npu layer1_control_core.ll -o layer1_control_core.s
;
; ─────────────────────────────────────────────────────────────────────────────
; CALLING CONVENTION (NPUCallingConv.td)
;   Arguments : i64   → x10, x11, x12, x13, x14, x15, x16, x17
;   Return    : i64   → x10
;   Callee-saved : x1 (RA), x18–x27
;   i8/i16/i32 args are promoted to i64 before passing
; ─────────────────────────────────────────────────────────────────────────────
;
; KNOWN LAYER 1 GAPS (do not put these in .ll or llc will fatal-error):
;   load i64     → "Cannot select"   use load i32 + sext instead
;   ashr i64     → not lowered       use lshr (logical) instead
;   select i64   → Expand, no CMOV  use explicit conditional branches
;   sdiv / udiv  → Expand, no divide instruction
; ─────────────────────────────────────────────────────────────────────────────

target triple = "npu"

; ═════════════════════════════════════════════════════════════════════════════
; §1  ARITHMETIC  — every ALU opcode in one function
; ═════════════════════════════════════════════════════════════════════════════
;
; C equivalent:
;   int64_t bitops(int64_t a, int64_t b) {
;       int64_t t = a & b;
;       t = t | (a ^ b);
;       t = t << (b & 7);       // SLL: shift amount in a GPR
;       t = t >> 1;              // SRL: logical right shift
;       return (t < a) ? 1 : 0; // SLT
;   }
;
; Expected assembly:
;   bitops:
;     AND   x12, x10, x11      ; t  = a & b
;     XOR   x13, x10, x11      ; t2 = a ^ b
;     OR    x12, x12, x13      ; t  = t | t2
;     ADDI  x13, x0,  7        ; mask = 7
;     AND   x13, x11, x13      ; shift_amt = b & 7
;     SLL   x12, x12, x13      ; t <<= shift_amt
;     LI    x13, 1             ; constant 1 for SRL
;     SRL   x12, x12, x13      ; t >>= 1
;     SLT   x10, x12, x10      ; return t < a
;     JALR  x0, x1, 0          ; RET
; ─────────────────────────────────────────────────────────────────────────────
define i64 @bitops(i64 %a, i64 %b) {
entry:
  %t0  = and  i64 %a,  %b        ; AND  — bitwise and
  %t1  = xor  i64 %a,  %b        ; XOR  — bitwise xor
  %t2  = or   i64 %t0, %t1       ; OR   — bitwise or
  %sh  = and  i64 %b,  7         ; mask shift amount to 3 bits
  %t3  = shl  i64 %t2, %sh       ; SLL  — shift left (amount in GPR)
  %t4  = lshr i64 %t3, 1         ; SRL  — logical shift right
  ; SLT: set-if-less-than → returns 0 or 1 (i64)
  %cmp = icmp slt i64 %t4, %a    ; SLT  — signed less-than compare
  %r   = zext i1 %cmp to i64     ; zero-extend bool to i64
  ret i64 %r
}

; ─────────────────────────────────────────────────────────────────────────────
; ADD / SUB / MUL with immediates (→ ADDI / LI patterns)
;
; Expected assembly:
;   alu_imm:
;     ADDI  x10, x10, 10        ; x += 10
;     ADDI  x11, x0,  -1        ; y = -1  (LI with small negative)
;     SUB   x10, x10, x11       ; x -= y
;     LI    x12, 1000000        ; large constant → LI
;     MUL   x10, x10, x12       ; x *= 1000000
;     JALR  x0, x1, 0
; ─────────────────────────────────────────────────────────────────────────────
define i64 @alu_imm(i64 %x) {
entry:
  %a = add  i64 %x,  10         ; ADDI xd, xa, 10
  %b = sub  i64 %a,  -1         ; ADDI xd, xa, 1  (sub -1 == add 1)
  %c = mul  i64 %b,  1000000    ; LI + MUL (large imm needs LI first)
  ret i64 %c
}


; ═════════════════════════════════════════════════════════════════════════════
; §2  MEMORY  — LW / LH / LB / SW / SH / SB
; ═════════════════════════════════════════════════════════════════════════════
;
; Rule: loads are i32/i16/i8 → sign-extended to i64.
;       stores truncate i64 → i32/i16/i8.
;       i64 load/store does NOT exist (would fatal-error).
;
; C equivalent:
;   void copy_bytes(int8_t *src, int16_t *mid, int32_t *dst, int64_t n) {
;       for (int i = 0; i < n; i++) {
;           int8_t  b = src[i];       // LB — byte load, sign-extend
;           int16_t h = (int16_t)b;   // SH — store 16-bit
;           int32_t w = (int32_t)h;   // SW — store 32-bit
;           dst[i] = w;
;       }
;   }
;
; Key addressing: GEP computes ptr + (i * sizeof(T)).
; selectAddrRI() in NPUISelDAGToDAG matches this to LW xd, offset(xbase).
;
; Expected assembly (single iteration):
;   LB  x5, 0(x10)      ; b = *src  (sign-extend byte)
;   SH  x5, 0(x11)      ; *mid = (int16_t)b
;   SW  x5, 0(x12)      ; *dst = (int32_t)b
; ─────────────────────────────────────────────────────────────────────────────
define void @memory_ops(ptr %src8, ptr %src16, ptr %dst32, i64 %n) {
entry:
  br label %check

check:
  %i   = phi i64 [0, %entry], [%i_next, %body]
  %cmp = icmp slt i64 %i, %n
  br i1 %cmp, label %body, label %exit

body:
  ; ── loads ─────────────────────────────────────────────────────────────────
  %p8  = getelementptr i8,  ptr %src8,  i64 %i
  %p16 = getelementptr i16, ptr %src16, i64 %i
  %p32 = getelementptr i32, ptr %dst32, i64 %i

  %b = load i8,  ptr %p8        ; → LB  xd, 0(%p8)   (sign-extend to i64)
  %h = load i16, ptr %p16       ; → LH  xd, 0(%p16)  (sign-extend to i64)
  %w = load i32, ptr %p32       ; → LW  xd, 0(%p32)  (sign-extend to i64)

  ; ── compute ───────────────────────────────────────────────────────────────
  %b64 = sext i8  %b to i64     ; sext is a no-op: LB already sign-extends
  %h64 = sext i16 %h to i64
  %sum = add i64 %b64, %h64
  %sum2 = add i64 %sum, %w

  ; ── stores ────────────────────────────────────────────────────────────────
  store i32 %w,  ptr %p32        ; → SW (truncates i64 → i32)
  store i16 %h,  ptr %p16        ; → SH (truncates i64 → i16)
  store i8  %b,  ptr %p8         ; → SB (truncates i64 → i8)

  %i_next = add i64 %i, 1
  br label %check

exit:
  ret void
}


; ═════════════════════════════════════════════════════════════════════════════
; §3a LOOP — sum of int32 array (BLT + JAL back-edge)
; ═════════════════════════════════════════════════════════════════════════════
;
; C equivalent:
;   int64_t sum_array(int32_t *arr, int64_t n) {
;       int64_t acc = 0;
;       for (int64_t i = 0; i < n; i++) acc += arr[i];
;       return acc;
;   }
;
; Expected assembly:
;   sum_array:
;     LI    x12, 0              ; acc = 0
;     LI    x13, 0              ; i   = 0
;   .check:
;     BLT   x13, x11, .loop    ; if i < n goto loop  (→ BLT)
;     ADD   x10, x12, x0       ; return acc
;     JALR  x0, x1, 0
;   .loop:
;     SLL   x14, x13, 2        ; byte_offset = i * 4
;     ADD   x14, x10, x14      ; ptr = arr + byte_offset
;     LW    x15, 0(x14)        ; val = *ptr
;     ADD   x12, x12, x15      ; acc += val
;     ADDI  x13, x13, 1        ; i++
;     JAL   x0, .check         ; (BR pseudo — JAL discards link)
; ─────────────────────────────────────────────────────────────────────────────
define i64 @sum_array(ptr %arr, i64 %n) {
entry:
  br label %check

check:
  %i   = phi i64 [0, %entry], [%i_next, %loop]
  %acc = phi i64 [0, %entry], [%acc_next, %loop]
  %cmp = icmp slt i64 %i, %n    ; → SLT ; fused into BLT by BR_CC lowering
  br i1 %cmp, label %loop, label %exit

loop:
  %ptr     = getelementptr i32, ptr %arr, i64 %i
  %val     = load i32, ptr %ptr  ; → LW (sign-extends i32 → i64)
  %val64   = sext i32 %val to i64
  %acc_next = add i64 %acc, %val64 ; → ADD
  %i_next   = add i64 %i, 1        ; → ADDI xd, xi, 1
  br label %check                   ; → JAL x0, .check  (BR pseudo)

exit:
  ret i64 %acc                      ; → ADD x10, acc_reg, x0  + RET
}


; ═════════════════════════════════════════════════════════════════════════════
; §3b CONDITIONAL — clamp without select
; ═════════════════════════════════════════════════════════════════════════════
;
; The NPU has no CMOV and SELECT is Expand with no fallback.
; Use explicit conditional branches instead of:
;   %r = select i1 %cmp, i64 %a, i64 %b   ← DO NOT USE, will crash
;
; C equivalent:
;   int64_t clamp(int64_t x, int64_t lo, int64_t hi) {
;       if (x < lo) return lo;
;       if (x > hi) return hi;
;       return x;
;   }
;
; Expected assembly:
;   clamp:
;     BLT   x10, x11, .ret_lo   ; if x < lo → return lo
;     BLT   x12, x10, .ret_hi   ; if hi < x → return hi  (x > hi same as hi < x)
;     JALR  x0, x1, 0           ; return x  (already in x10)
;   .ret_lo:
;     ADD   x10, x11, x0        ; x10 = lo
;     JALR  x0, x1, 0
;   .ret_hi:
;     ADD   x10, x12, x0        ; x10 = hi
;     JALR  x0, x1, 0
; ─────────────────────────────────────────────────────────────────────────────
define i64 @clamp(i64 %x, i64 %lo, i64 %hi) {
entry:
  %below = icmp slt i64 %x, %lo    ; x < lo ?
  br i1 %below, label %ret_lo, label %check_hi

check_hi:
  %above = icmp sgt i64 %x, %hi    ; x > hi ? (lowered as: hi < x → BLT hi, x)
  br i1 %above, label %ret_hi, label %ret_x

ret_lo:
  ret i64 %lo

ret_hi:
  ret i64 %hi

ret_x:
  ret i64 %x
}


; ═════════════════════════════════════════════════════════════════════════════
; §3c FUNCTION CALL — recursive factorial (JAL + stack frame)
; ═════════════════════════════════════════════════════════════════════════════
;
; Recursion forces the backend to:
;   1. Build a stack frame  (ADDI x2, x2, -16   — grow stack)
;   2. Spill x1 (RA)        (SW   x1, 8(x2)     — save return address)
;   3. Spill x10 (%n)       (SW   x10, 0(x2)    — save arg across call)
;   4. Issue the call        (JAL  x1, factorial — overwrites x1!)
;   5. Restore x1           (LW   x1, 8(x2))
;   6. Restore %n           (LW   x10, 0(x2)    — then MUL with result)
;   7. Tear down frame      (ADDI x2, x2, 16)
;   8. Return               (JALR x0, x1, 0)
;
; x1 is in the callee-saved list (CSR_NPU in NPUCallingConv.td) so the
; register allocator automatically emits the save/restore prologue/epilogue.
;
; Expected assembly:
;   factorial:
;     ADDI  x2,  x2, -16       ; allocate 16-byte frame
;     SW    x1,  8(x2)         ; save RA
;     SW    x10, 0(x2)         ; save n (need it after recursive call)
;     ADDI  x11, x0, 2
;     BLT   x10, x11, .base   ; if n < 2 goto base
;     ADDI  x10, x10, -1       ; n-1
;     JAL   x1, factorial      ; recursive call — result in x10
;     LW    x11, 0(x2)         ; reload n
;     MUL   x10, x11, x10      ; n * factorial(n-1)
;     JAL   x0, .epilogue
;   .base:
;     LI    x10, 1             ; return 1
;   .epilogue:
;     LW    x1, 8(x2)          ; restore RA
;     ADDI  x2, x2, 16         ; free frame
;     JALR  x0, x1, 0          ; RET
; ─────────────────────────────────────────────────────────────────────────────
define i64 @factorial(i64 %n) {
entry:
  %base_case = icmp slt i64 %n, 2   ; n < 2 ?
  br i1 %base_case, label %base, label %recurse

base:
  ret i64 1

recurse:
  %n1  = sub i64 %n, 1               ; n - 1     → SUB / ADDI -1
  %rec = call i64 @factorial(i64 %n1) ; → JAL x1, factorial
  %res = mul  i64 %n, %rec            ; n * rec   → MUL
  ret i64 %res
}


; ═════════════════════════════════════════════════════════════════════════════
; §3d NON-RECURSIVE CALL — two functions, one calls the other
; ═════════════════════════════════════════════════════════════════════════════
;
; @square(x) is a leaf function (no further calls) — no stack frame needed.
; @sum_of_squares calls @square twice.
;
; Expected assembly for square (leaf — no prologue):
;   square:
;     MUL  x10, x10, x10
;     JALR x0, x1, 0
;
; Expected assembly for sum_of_squares:
;   sum_of_squares:
;     ADDI x2,  x2, -16       ; frame needed: must save x1
;     SW   x1,  8(x2)
;     SW   x11, 0(x2)         ; save b (x11 clobbered by call)
;     JAL  x1, square         ; square(a) — result in x10
;     LW   x11, 0(x2)         ; reload b
;     MUL  x12, x10, x10      ; a^2 is in x10... wait, square(b) next
;   ... (backend may reorder to minimize spills)
; ─────────────────────────────────────────────────────────────────────────────
define i64 @square(i64 %x) {
entry:
  %r = mul i64 %x, %x    ; MUL x10, x10, x10
  ret i64 %r
}

define i64 @sum_of_squares(i64 %a, i64 %b) {
entry:
  %a2 = call i64 @square(i64 %a)   ; JAL x1, square — clobbers x11!
  %b2 = call i64 @square(i64 %b)   ; JAL x1, square
  %r  = add i64 %a2, %b2            ; ADD x10, x10, saved_b2
  ret i64 %r
}


; ═════════════════════════════════════════════════════════════════════════════
; WHAT DOES NOT WORK IN LAYER 1
; ═════════════════════════════════════════════════════════════════════════════
;
; The following will fatal-error or assert during llc — do NOT include them:
;
;   load i64, ptr %p          → "Cannot select" (no 64-bit load instruction)
;   store i64 %v, ptr %p      → same
;
;   %r = ashr i64 %x, 4       → not lowered (use lshr for logical shift)
;
;   %r = select i1 %c, i64 %a, i64 %b
;                             → SELECT is Expand, no CMOV fallback
;                               use explicit br + phi instead (see @clamp above)
;
;   %r = sdiv i64 %a, %b      → Expand, no divide instruction
;   %r = udiv i64 %a, %b      → same
;   %r = srem i64 %a, %b      → same
;
;   fadd float %a, %b         → f32 on control core is Expand
;                               float ops only work inside vector kernel (vinst)
;
; ─────────────────────────────────────────────────────────────────────────────
; BRANCH LOWERING SUMMARY
; ─────────────────────────────────────────────────────────────────────────────
;
; LLVM BR_CC is Legal for i64 (NPUISelLowering.cpp:45).
; NPUISelDAGToDAG::Select fuses icmp + br into a single branch instruction:
;
;   icmp eq  → BEQ  rs1, rs2
;   icmp ne  → BNE  rs1, rs2
;   icmp slt → BLT  rs1, rs2
;   icmp sge → BGE  rs1, rs2
;   icmp sgt → BLT  rs2, rs1   (operands swapped)
;   icmp sle → BGE  rs2, rs1   (operands swapped)
;   icmp ult → XOR+XOR+BLT     (unsigned: flip sign bit of both, then BLT)
;   icmp uge → XOR+XOR+BGE     (same trick)
; ─────────────────────────────────────────────────────────────────────────────
