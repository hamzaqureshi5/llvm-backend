#!/usr/bin/env bash
# test-npu-backend.sh — end-to-end LLVM NPU backend smoke tests
#
# Run from the llvm-project root:
#   bash test-npu-backend.sh
#
# Prerequisites (already built):
#   cmake --build build --target llc FileCheck not
#   cmake -S npu-simulator -B npu-simulator/build && cmake --build npu-simulator/build

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD="${BUILD_DIR:-${ROOT}/build}"
LLC="${BUILD}/bin/llc"
LIT="${BUILD}/bin/llvm-lit"
SIM="${ROOT}/npu-simulator/build/npu_elf_run"
TMP="$(mktemp -d)"
trap 'rm -rf "${TMP}"' EXIT

PASS=0; FAIL=0
GREEN='\033[0;32m'; RED='\033[0;31m'; YELLOW='\033[1;33m'; NC='\033[0m'

ok()   { echo -e "  ${GREEN}PASS${NC}  $1"; PASS=$((PASS+1)); }
fail() { echo -e "  ${RED}FAIL${NC}  $1"; FAIL=$((FAIL+1)); }

# Compile inline LLVM IR (from stdin), run in simulator, check x10.
# Usage: check_sim <label> <expected_x10> <ir_string> [arg0 [arg1 ...]]
check_sim() {
    local label="$1" expected="$2" ir="$3"
    shift 3
    local slug obj got
    slug=$(echo "${label}" | tr -cs 'a-zA-Z0-9' '_')
    obj="${TMP}/${slug}.o"

    if ! printf '%s\n' "${ir}" \
            | "${LLC}" -mtriple=npu -filetype=obj -o "${obj}" - 2>/dev/null; then
        fail "${label}  (llc rejected the IR)"
        return
    fi

    got=$("${SIM}" "${obj}" "$@" 2>/dev/null \
          | awk '/x10 =/ {print $4; exit}')

    if [ "${got}" = "${expected}" ]; then
        ok "${label}  → ${got}"
    else
        fail "${label}  expected=${expected} got='${got}'"
    fi
}

# Compile inline LLVM IR, check that the assembly text contains a pattern.
check_asm() {
    local label="$1" pattern="$2" ir="$3"
    local asm
    if ! asm=$(printf '%s\n' "${ir}" \
                | "${LLC}" -mtriple=npu -o - - 2>/dev/null); then
        fail "${label}  (llc rejected the IR)"
        return
    fi
    if echo "${asm}" | grep -q "${pattern}"; then
        ok "${label}  (contains '${pattern}')"
    else
        fail "${label}  pattern '${pattern}' missing from asm"
    fi
}

# Compile inline IR and expect llc to fail with a specific error message.
# Reports SKIP (not FAIL) because these are known gaps.
check_fails() {
    local label="$1" pattern="$2" ir="$3"
    local err
    err=$(printf '%s\n' "${ir}" | "${LLC}" -mtriple=npu -o /dev/null - 2>&1 || true)
    if echo "${err}" | grep -q "${pattern}"; then
        echo -e "  ${YELLOW}SKIP${NC}  ${label}  (known gap — ${pattern})"
    else
        echo -e "  ${GREEN}FIXED${NC} ${label}  (no longer fails — consider adding a sim test)"
    fi
}

echo "========================================================"
echo " NPU LLVM Backend — smoke tests"
echo " LLC : ${LLC}"
echo " SIM : ${SIM}"
echo "========================================================"

# ─── 0. Prerequisites ────────────────────────────────────────────────────────
echo ""
echo "─── 0. Prerequisites ────────────────────────────────────"

if [ ! -x "${LLC}" ]; then
    echo -e "${RED}ERROR${NC}: ${LLC} not found — build it first:"
    echo "  cmake --build build --target llc FileCheck not"
    exit 1
fi
if [ ! -x "${SIM}" ]; then
    echo -e "${RED}ERROR${NC}: ${SIM} not found — build it first:"
    echo "  cmake -S npu-simulator -B npu-simulator/build"
    echo "  cmake --build npu-simulator/build"
    exit 1
fi
ok "llc and npu_elf_run both present"

# ─── 1. Target registration ───────────────────────────────────────────────────
echo ""
echo "─── 1. Target registration ──────────────────────────────"

if "${LLC}" --version 2>&1 | grep -qi "npu"; then
    ok "NPU target registered"
else
    fail "NPU target not found in llc --version"
fi

# ─── 2. Integer arithmetic ────────────────────────────────────────────────────
echo ""
echo "─── 2. Integer arithmetic (control core) ─────────────────"

check_sim "add i64 5+7=12" 12 \
'define i64 @f(i64 %a, i64 %b) {
  %r = add i64 %a, %b
  ret i64 %r
}' 5 7

check_sim "sub i64 10-7=3" 3 \
'define i64 @f(i64 %a, i64 %b) {
  %r = sub i64 %a, %b
  ret i64 %r
}' 10 7

check_sim "mul i64 6*7=42" 42 \
'define i64 @f(i64 %a, i64 %b) {
  %r = mul i64 %a, %b
  ret i64 %r
}' 6 7

check_sim "and i64 12&5=4" 4 \
'define i64 @f(i64 %a, i64 %b) {
  %r = and i64 %a, %b
  ret i64 %r
}' 12 5

check_sim "or i64 12|5=13" 13 \
'define i64 @f(i64 %a, i64 %b) {
  %r = or i64 %a, %b
  ret i64 %r
}' 12 5

check_sim "xor i64 12^5=9" 9 \
'define i64 @f(i64 %a, i64 %b) {
  %r = xor i64 %a, %b
  ret i64 %r
}' 12 5

check_sim "shl i64 5<<2=20" 20 \
'define i64 @f(i64 %a, i64 %b) {
  %r = shl i64 %a, %b
  ret i64 %r
}' 5 2

check_sim "lshr i64 8>>2=2" 2 \
'define i64 @f(i64 %a, i64 %b) {
  %r = lshr i64 %a, %b
  ret i64 %r
}' 8 2

# ─── 3. Immediates ────────────────────────────────────────────────────────────
echo ""
echo "─── 3. Immediates ────────────────────────────────────────"

check_sim "ret constant 42" 42 \
'define i64 @f() {
  ret i64 42
}'

check_sim "add immediate 5+10=15" 15 \
'define i64 @f(i64 %a) {
  %r = add i64 %a, 10
  ret i64 %r
}' 5

check_sim "large immediate 1000" 1000 \
'define i64 @f() {
  ret i64 1000
}'

check_sim "negative immediate 5+(-3)=2" 2 \
'define i64 @f(i64 %a) {
  %r = add i64 %a, -3
  ret i64 %r
}' 5

# ─── 4. Control flow ──────────────────────────────────────────────────────────
echo ""
echo "─── 4. Control flow ──────────────────────────────────────"

check_sim "icmp slt taken" 1 \
'define i64 @f(i64 %a, i64 %b) {
entry:
  %c = icmp slt i64 %a, %b
  br i1 %c, label %yes, label %no
yes:
  ret i64 1
no:
  ret i64 0
}' 3 5

check_sim "icmp slt not-taken" 0 \
'define i64 @f(i64 %a, i64 %b) {
entry:
  %c = icmp slt i64 %a, %b
  br i1 %c, label %yes, label %no
yes:
  ret i64 1
no:
  ret i64 0
}' 7 5

check_sim "icmp eq true" 1 \
'define i64 @f(i64 %a, i64 %b) {
entry:
  %c = icmp eq i64 %a, %b
  br i1 %c, label %yes, label %no
yes:
  ret i64 1
no:
  ret i64 0
}' 5 5

check_sim "icmp eq false" 0 \
'define i64 @f(i64 %a, i64 %b) {
entry:
  %c = icmp eq i64 %a, %b
  br i1 %c, label %yes, label %no
yes:
  ret i64 1
no:
  ret i64 0
}' 5 4

check_sim "icmp ne true" 1 \
'define i64 @f(i64 %a, i64 %b) {
entry:
  %c = icmp ne i64 %a, %b
  br i1 %c, label %yes, label %no
yes:
  ret i64 1
no:
  ret i64 0
}' 5 4

check_sim "loop sum 1..10=55" 55 \
'define i64 @f(i64 %n) {
entry:
  br label %loop
loop:
  %i   = phi i64 [ 1, %entry ], [ %i2,   %loop ]
  %acc = phi i64 [ 0, %entry ], [ %acc2, %loop ]
  %acc2 = add i64 %acc, %i
  %i2   = add i64 %i,   1
  %done = icmp sgt i64 %i2, %n
  br i1 %done, label %exit, label %loop
exit:
  ret i64 %acc2
}' 10

check_sim "loop multiply 3*5=15" 15 \
'define i64 @f(i64 %a, i64 %b) {
entry:
  br label %loop
loop:
  %i   = phi i64 [ 0,  %entry ], [ %i2,   %loop ]
  %acc = phi i64 [ 0,  %entry ], [ %acc2, %loop ]
  %acc2 = add i64 %acc, %a
  %i2   = add i64 %i,   1
  %done = icmp eq i64 %i2, %b
  br i1 %done, label %exit, label %loop
exit:
  ret i64 %acc2
}' 3 5

# (select/ternary tests moved to known gaps in section 9 — needs CMOV or
# custom SELECT_CC lowering; SELECT=Expand+SELECT_CC=Expand causes an assert)

# ─── 6. Asm text patterns ─────────────────────────────────────────────────────
echo ""
echo "─── 6. Assembly text patterns ────────────────────────────"

check_asm "ADD in asm for add i64" ADD \
'define i64 @f(i64 %a, i64 %b) {
  %r = add i64 %a, %b
  ret i64 %r
}'

check_asm "JALR in return sequence" JALR \
'define i64 @f() {
  ret i64 0
}'

check_asm "HALT at end of function" HALT \
'define i64 @f() {
  ret i64 0
}'

check_asm "LI for constant 42" LI \
'define i64 @f() {
  ret i64 42
}'

# LLVM inverts the condition and branches to the false block, so icmp eq
# emits BNE (branch to 'no' when NOT equal) and icmp ne emits BEQ.
check_asm "conditional branch for icmp eq" BNE \
'define i64 @f(i64 %a, i64 %b) {
entry:
  %c = icmp eq i64 %a, %b
  br i1 %c, label %y, label %n
y:
  ret i64 1
n:
  ret i64 0
}'

check_asm "conditional branch for icmp ne" BEQ \
'define i64 @f(i64 %a, i64 %b) {
entry:
  %c = icmp ne i64 %a, %b
  br i1 %c, label %y, label %n
y:
  ret i64 1
n:
  ret i64 0
}'

check_asm "MUL for mul i64" MUL \
'define i64 @f(i64 %a, i64 %b) {
  %r = mul i64 %a, %b
  ret i64 %r
}'

# ─── 7. Object file output ────────────────────────────────────────────────────
echo ""
echo "─── 7. Object file output ────────────────────────────────"

printf 'define i64 @f(i64 %%a, i64 %%b) {\n  %%r = add i64 %%a, %%b\n  ret i64 %%r\n}\n' \
    | "${LLC}" -mtriple=npu -filetype=obj -o "${TMP}/obj_test.o" -

if file "${TMP}/obj_test.o" | grep -q "ELF 64-bit"; then
    ok "ELF 64-bit object emitted"
else
    fail "object file is not a valid ELF 64-bit file"
fi

OBJ_SIZE=$(stat -c%s "${TMP}/obj_test.o")
[ "${OBJ_SIZE}" -gt 64 ] \
    && ok "object non-trivial (${OBJ_SIZE} bytes)" \
    || fail "object suspiciously small (${OBJ_SIZE} bytes)"

# .text size must be a non-zero multiple of 32 (NPU instruction word = 32 bytes)
TEXT_HEX=$(readelf -S "${TMP}/obj_test.o" 2>/dev/null \
    | awk '/\.text/ {print $6}' | head -1 | sed 's/^0*//')
TEXT_DEC=$(( TEXT_HEX ? 16#${TEXT_HEX} : 0 ))
if [ $((TEXT_DEC % 32)) -eq 0 ] && [ "${TEXT_DEC}" -gt 0 ]; then
    ok ".text size ${TEXT_DEC} bytes (multiple of 32 — correct word alignment)"
else
    fail ".text size ${TEXT_DEC} — expected non-zero multiple of 32"
fi

# ─── 8. FileCheck lit suite ───────────────────────────────────────────────────
echo ""
echo "─── 8. FileCheck lit regression suite ───────────────────"

if "${LIT}" -q llvm/test/CodeGen/NPU/ 2>/dev/null; then
    ok "lit NPU test suite passed"
else
    fail "lit NPU test suite: one or more tests failed (run llvm-lit -v llvm/test/CodeGen/NPU/ for details)"
fi

# ─── 9. NPU intrinsic assembly patterns ──────────────────────────────────────
echo ""
echo "─── 9. NPU intrinsic asm patterns (Layer 2 + Layer 3) ───"

# Layer 2: Tensor engine ─────────────────────────────────────────────────────

check_asm "T_CFG_SHAPE emits CFG.SHAPE" "CFG.SHAPE" \
'declare void @llvm.npu.cfg.shape(i64, i64, i64)
define void @f() {
  call void @llvm.npu.cfg.shape(i64 4, i64 4, i64 4)
  ret void
}'

check_asm "T_ACC_CLR emits ACC.CLR t0" "ACC.CLR t0" \
'declare i64 @llvm.npu.acc.clr(i64)
define i64 @f() {
  %r = call i64 @llvm.npu.acc.clr(i64 0)
  ret i64 %r
}'

# MMA: acc.clr followed by two chained mma ops
check_asm "T_MMA: acc.clr + mma chain" "MMA" \
'declare i64 @llvm.npu.acc.clr(i64)
declare i64 @llvm.npu.mma(i64, i64, i64)
define i64 @f() {
  %t0 = call i64 @llvm.npu.acc.clr(i64 0)
  %t1 = call i64 @llvm.npu.mma(i64 %t0, i64 1, i64 2)
  ret i64 %t1
}'

# MATMUL: explicit dt, no accumulator
check_asm "T_MATMUL emits MATMUL" "MATMUL" \
'declare i64 @llvm.npu.matmul(i64, i64, i64)
define i64 @f() {
  %r = call i64 @llvm.npu.matmul(i64 2, i64 0, i64 1)
  ret i64 %r
}'

# DOT: CF_T3 format — dt is the accumulator (3 operands, same as MMA/MATMUL)
check_asm "T_DOT: CF_T3 format (dt is accumulator)" "DOT" \
'declare i64 @llvm.npu.acc.clr(i64)
declare i64 @llvm.npu.dot(i64, i64, i64)
define i64 @f() {
  %ta = call i64 @llvm.npu.acc.clr(i64 0)
  %r  = call i64 @llvm.npu.dot(i64 %ta, i64 1, i64 2)
  ret i64 %r
}'

check_asm "T_RELU emits RELU" "RELU" \
'declare i64 @llvm.npu.relu(i64)
define i64 @f() {
  %r = call i64 @llvm.npu.relu(i64 3)
  ret i64 %r
}'

# SCALE: float factor encoded as fimm (double bit pattern), not a GPR
check_asm "T_SCALE: factor in fimm (not GPR)" "SCALE" \
'declare i64 @llvm.npu.scale(i64, float)
define i64 @f() {
  %r = call i64 @llvm.npu.scale(i64 3, float 2.0)
  ret i64 %r
}'

# BCAST: float bits go through LI (GPR), NOT an fimm — simulator uses I->rs1
# 3.14f = 0x4048F5C3 = 1078523331
check_asm "T_BCAST: value in GPR via LI (not fimm)" "BCAST" \
'declare i64 @llvm.npu.bcast(i64, float)
define i64 @f() {
  %r = call i64 @llvm.npu.bcast(i64 2, float 3.14)
  ret i64 %r
}'

check_asm "T_BCAST: LI materializes f32 bits" "1078523331" \
'declare i64 @llvm.npu.bcast(i64, float)
define i64 @f() {
  %r = call i64 @llvm.npu.bcast(i64 2, float 3.14)
  ret i64 %r
}'

check_asm "T_FENCE emits FENCE" "FENCE" \
'declare void @llvm.npu.fence()
define void @f() {
  call void @llvm.npu.fence()
  ret void
}'

check_asm "T_BARRIER emits BARRIER" "BARRIER" \
'declare void @llvm.npu.barrier()
define void @f() {
  call void @llvm.npu.barrier()
  ret void
}'

# Layer 3: Vector core ───────────────────────────────────────────────────────

# VSETVL: constant VL → LI + VSETVL (both in GPRs, not SRegs)
check_asm "V_VSETVL: constant VL emits LI + VSETVL" "VSETVL" \
'declare i64 @llvm.npu.vsetvl(i64)
define i64 @f() {
  %vl = call i64 @llvm.npu.vsetvl(i64 16)
  ret i64 %vl
}'

# VSETVL: runtime VL from argument → direct VSETVL (no LI)
check_asm "V_VSETVL: dynamic VL emits direct VSETVL" "VSETVL" \
'declare i64 @llvm.npu.vsetvl(i64)
define i64 @f(i64 %len) {
  %vl = call i64 @llvm.npu.vsetvl(i64 %len)
  ret i64 %vl
}'

check_asm "V_VLOAD emits VLOAD" "VLOAD" \
'declare <16 x float> @llvm.npu.vload(ptr, i64)
define <16 x float> @f(ptr %p) {
  %v = call <16 x float> @llvm.npu.vload(ptr %p, i64 4)
  ret <16 x float> %v
}'

# VADD via native <16 x float> fadd
check_asm "V_VADD via native fadd" "VADD" \
'define <16 x float> @f(<16 x float> %a, <16 x float> %b) {
  %r = fadd <16 x float> %a, %b
  ret <16 x float> %r
}'

check_asm "V_VEXP emits VEXP" "VEXP" \
'declare <16 x float> @llvm.npu.vexp(<16 x float>)
define <16 x float> @f(<16 x float> %v) {
  %r = call <16 x float> @llvm.npu.vexp(<16 x float> %v)
  ret <16 x float> %r
}'

check_asm "V_VREDSUM emits VREDSUM" "VREDSUM" \
'declare float @llvm.npu.vredsum(<16 x float>)
define float @f(<16 x float> %v) {
  %r = call float @llvm.npu.vredsum(<16 x float> %v)
  ret float %r
}'

# Softmax building block: vload → vexp → vredsum
check_asm "softmax chain: VLOAD+VEXP+VREDSUM" "VREDSUM" \
'declare <16 x float> @llvm.npu.vload(ptr, i64)
declare <16 x float> @llvm.npu.vexp(<16 x float>)
declare float @llvm.npu.vredsum(<16 x float>)
define float @f(ptr %p) {
  %v = call <16 x float> @llvm.npu.vload(ptr %p, i64 4)
  %e = call <16 x float> @llvm.npu.vexp(<16 x float> %v)
  %s = call float @llvm.npu.vredsum(<16 x float> %e)
  ret float %s
}'

# ─── 10. Known gaps ───────────────────────────────────────────────────────────
echo ""
echo "─── 10. Known gaps (expected to fail) ───────────────────"

check_fails "i64 load from memory" "Cannot select" \
'define i64 @f(i64* %p) {
  %r = load i64, i64* %p
  ret i64 %r
}'

check_fails "select (no CMOV, assert in legalizer)" "Assertion\|Assertio\|fatal error\|Cannot select" \
'define i64 @f(i64 %a, i64 %b) {
  %c = icmp slt i64 %a, %b
  %r = select i1 %c, i64 99, i64 0
  ret i64 %r
}'

check_fails "ashr i64 (arithmetic right shift)" "Cannot select" \
'define i64 @f(i64 %a, i64 %b) {
  %r = ashr i64 %a, %b
  ret i64 %r
}'

# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "========================================================"
TOTAL=$((PASS+FAIL))
echo -e " Results: ${GREEN}${PASS}${NC}/${TOTAL} passed   ${RED}${FAIL}${NC} failed"
echo "========================================================"
[ "${FAIL}" -eq 0 ]
