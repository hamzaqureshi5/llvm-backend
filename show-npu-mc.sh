#!/usr/bin/env bash
# show-npu-mc.sh — print assembly + decoded MC bytes for every test case
#
# Run from the llvm-project root:
#   bash show-npu-mc.sh [filter]
#
# Optional filter: only show tests whose name contains the string.
#   bash show-npu-mc.sh "loop"
#   bash show-npu-mc.sh "add"

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD="${BUILD_DIR:-${ROOT}/build}"
LLC="${BUILD}/bin/llc"
TMP="$(mktemp -d)"
trap 'rm -rf "${TMP}"' EXIT

FILTER="${1:-}"

BOLD='\033[1m'; CYAN='\033[0;36m'; YELLOW='\033[1;33m'; DIM='\033[2m'; NC='\033[0m'

# ── Python snippet that decodes an NPU ELF .text section ─────────────────────
# Prints one line per instruction: index, mnemonic, fields, raw hex.
DECODE_PY='
import struct, sys

COP = ["ADD","SUB","ADDI","MUL","AND","OR","XOR","SLL","SRL","SLT",
       "LI","LW","LH","LB","SW","SH","SB",
       "BEQ","BNE","BLT","BGE","JAL","JALR","KLAUNCH","KWAIT","HALT",
       "T_CFG_SHAPE","T_CFG_DTYPE","T_CFG_STRIDE","T_CFG_MODE","T_CFG_RND",
       "T_MATMUL","T_MMA","T_DOT","T_CONV","T_ACC_CLR",
       "T_LOAD","T_STORE","T_LOADW","T_DMA_IN","T_DMA_OUT","T_BCAST",
       "T_RELU","T_QUANT","T_DEQUANT","T_SCALE","T_POOL",
       "T_FENCE","T_WAIT","T_BARRIER","T_STATUS"]
VOP = ["VSETVL","VLOAD","VSTORE","VGATHER","VSCATTER","VMOV","VBCAST",
       "SLI","SFI","SADD","SADDI","SSUB","SMUL","SMULI","SLW","SLF","SRECIP","KID",
       "VADD","VSUB","VMUL","VDIV","VMAX","VMIN","VFMA",
       "VCMP","VSEL","VEXP","VLOG","VRSQRT","VSQRT","VRECIP","VCVT","VABS","VNEG",
       "VREDSUM","VREDMAX","VREDMIN","VREDAND","VREDOR","VSCAN",
       "VMASK_SET","VMASK_AND","VMASK_OR","VMASK_NOT",
       "BR","BRZ","BRNZ","LOOP","KEXIT"]

def gpr(n): return f"x{n}"
def vr(n):  return f"v{n}"

fname = sys.argv[1]
data  = open(fname,"rb").read()

e_shoff    = struct.unpack_from("<Q", data, 40)[0]
e_shentsize= struct.unpack_from("<H", data, 58)[0]
e_shnum    = struct.unpack_from("<H", data, 60)[0]
e_shstrndx = struct.unpack_from("<H", data, 62)[0]
shstr_off  = struct.unpack_from("<Q", data, e_shoff + e_shstrndx*e_shentsize + 24)[0]

text_sh = None
for i in range(e_shnum):
    sh   = data[e_shoff + i*e_shentsize : e_shoff + (i+1)*e_shentsize]
    name = data[shstr_off + struct.unpack_from("<I",sh,0)[0]:].split(b"\x00")[0].decode()
    if name == ".text":
        text_sh = sh; break

if not text_sh:
    print("  (no .text section)"); sys.exit(0)

off  = struct.unpack_from("<Q", text_sh, 24)[0]
size = struct.unpack_from("<Q", text_sh, 32)[0]
text = data[off:off+size]
n    = size // 32

for k in range(n):
    w    = text[k*32:(k+1)*32]
    s,op = w[0], w[1]
    rd,rs1,rs2,rs = w[2],w[3],w[4],w[7]
    imm  = struct.unpack_from("<q", w, 16)[0]
    hex5 = " ".join(f"{b:02x}" for b in w[:8]) + " ..."
    if s == 0:
        mn = COP[op] if op < len(COP) else f"cop_{op}"
    else:
        mn = VOP[op] if op < len(VOP) else f"vop_{op}"
    # format fields based on mnemonic
    if mn in ("ADD","SUB","MUL","AND","OR","XOR","SLL","SRL","SLT"):
        flds = f"{gpr(rd)}, {gpr(rs1)}, {gpr(rs2)}"
    elif mn == "ADDI":
        flds = f"{gpr(rd)}, {gpr(rs1)}, {imm}"
    elif mn == "LI":
        flds = f"{gpr(rd)}, {imm}"
    elif mn in ("LW","LH","LB"):
        flds = f"{gpr(rd)}, {imm}({gpr(rs1)})"
    elif mn in ("SW","SH","SB"):
        flds = f"{gpr(rs2)}, {imm}({gpr(rs1)})"
    elif mn in ("BEQ","BNE","BLT","BGE"):
        flds = f"{gpr(rs1)}, {gpr(rs2)}, →{imm}"
    elif mn in ("JAL","BR"):
        flds = f"→{imm}"
    elif mn == "JALR":
        flds = f"{gpr(rd)}, {gpr(rs1)}, {imm}"
    elif mn == "HALT":
        flds = ""
    else:
        flds = f"rd={rd} rs1={rs1} rs2={rs2} imm={imm}"
    print(f"  [{k:2d}]  {mn:<8s} {flds:<30s}   | {hex5}")
'

# ── helper: show one test case ────────────────────────────────────────────────
show() {
    local name="$1" ir="$2"
    # apply filter
    if [ -n "$FILTER" ] && ! echo "$name" | grep -qi "$FILTER"; then
        return
    fi

    local obj="${TMP}/$(echo "$name" | tr -cs 'a-zA-Z0-9' '_').o"

    echo ""
    printf "${BOLD}${CYAN}══ %s ══${NC}\n" "$name"

    echo ""
    printf "${YELLOW}── Assembly ──────────────────────────────────────────────${NC}\n"
    if ! printf '%s\n' "$ir" | "$LLC" -mtriple=npu -o - - 2>/dev/null; then
        printf "${DIM}  (llc failed — known gap)${NC}\n"
        return
    fi

    echo ""
    printf "${YELLOW}── MC (decoded 32-byte instruction words) ────────────────${NC}\n"
    if printf '%s\n' "$ir" | "$LLC" -mtriple=npu -filetype=obj -o "$obj" - 2>/dev/null; then
        python3 -c "$DECODE_PY" "$obj"
    else
        printf "${DIM}  (obj emit failed)${NC}\n"
    fi
}

echo ""
printf "${BOLD}NPU LLC → Assembly + MC dump${NC}"
[ -n "$FILTER" ] && printf "  (filter: '%s')" "$FILTER"
echo ""

# ═══════════════════════════════════════════════════════════════════════════════
# 1. Arithmetic
# ═══════════════════════════════════════════════════════════════════════════════

show "add i64" \
'define i64 @f(i64 %a, i64 %b) {
  %r = add i64 %a, %b
  ret i64 %r
}'

show "sub i64" \
'define i64 @f(i64 %a, i64 %b) {
  %r = sub i64 %a, %b
  ret i64 %r
}'

show "mul i64" \
'define i64 @f(i64 %a, i64 %b) {
  %r = mul i64 %a, %b
  ret i64 %r
}'

show "and i64" \
'define i64 @f(i64 %a, i64 %b) {
  %r = and i64 %a, %b
  ret i64 %r
}'

show "or i64" \
'define i64 @f(i64 %a, i64 %b) {
  %r = or i64 %a, %b
  ret i64 %r
}'

show "xor i64" \
'define i64 @f(i64 %a, i64 %b) {
  %r = xor i64 %a, %b
  ret i64 %r
}'

show "shl i64" \
'define i64 @f(i64 %a, i64 %b) {
  %r = shl i64 %a, %b
  ret i64 %r
}'

show "lshr i64" \
'define i64 @f(i64 %a, i64 %b) {
  %r = lshr i64 %a, %b
  ret i64 %r
}'

# ═══════════════════════════════════════════════════════════════════════════════
# 2. Immediates
# ═══════════════════════════════════════════════════════════════════════════════

show "ret constant 42" \
'define i64 @f() {
  ret i64 42
}'

show "add immediate (a+10)" \
'define i64 @f(i64 %a) {
  %r = add i64 %a, 10
  ret i64 %r
}'

show "large immediate 1000" \
'define i64 @f() {
  ret i64 1000
}'

show "negative immediate (a-3)" \
'define i64 @f(i64 %a) {
  %r = add i64 %a, -3
  ret i64 %r
}'

# ═══════════════════════════════════════════════════════════════════════════════
# 3. Conditional branches
# ═══════════════════════════════════════════════════════════════════════════════

show "icmp slt branch" \
'define i64 @f(i64 %a, i64 %b) {
entry:
  %c = icmp slt i64 %a, %b
  br i1 %c, label %yes, label %no
yes:
  ret i64 1
no:
  ret i64 0
}'

show "icmp eq branch" \
'define i64 @f(i64 %a, i64 %b) {
entry:
  %c = icmp eq i64 %a, %b
  br i1 %c, label %yes, label %no
yes:
  ret i64 1
no:
  ret i64 0
}'

show "icmp ne branch" \
'define i64 @f(i64 %a, i64 %b) {
entry:
  %c = icmp ne i64 %a, %b
  br i1 %c, label %yes, label %no
yes:
  ret i64 1
no:
  ret i64 0
}'

show "icmp sge branch" \
'define i64 @f(i64 %a, i64 %b) {
entry:
  %c = icmp sge i64 %a, %b
  br i1 %c, label %yes, label %no
yes:
  ret i64 1
no:
  ret i64 0
}'

# ═══════════════════════════════════════════════════════════════════════════════
# 4. Loops
# ═══════════════════════════════════════════════════════════════════════════════

show "loop sum 1..N" \
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
}'

show "loop multiply (repeated add)" \
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
}'

# ═══════════════════════════════════════════════════════════════════════════════
# 5. Memory (32-bit loads — LW sign-extends to i64)
# ═══════════════════════════════════════════════════════════════════════════════

show "load i32 (LW sign-ext to i64)" \
'define i64 @f(i32* %p) {
  %r = load i32, i32* %p
  %e = sext i32 %r to i64
  ret i64 %e
}'

show "store i32 (SW)" \
'define void @f(i32* %p, i32 %v) {
  store i32 %v, i32* %p
  ret void
}'

# ═══════════════════════════════════════════════════════════════════════════════
# 6. Known gaps (show the error)
# ═══════════════════════════════════════════════════════════════════════════════

show "i64 load (gap — no LD instruction in ISA)" \
'define i64 @f(i64* %p) {
  %r = load i64, i64* %p
  ret i64 %r
}'

show "select (gap — needs CMOV or custom lowering)" \
'define i64 @f(i64 %a, i64 %b) {
  %c = icmp slt i64 %a, %b
  %r = select i1 %c, i64 1, i64 0
  ret i64 %r
}'

show "ashr (gap — no SRA instruction defined)" \
'define i64 @f(i64 %a, i64 %b) {
  %r = ashr i64 %a, %b
  ret i64 %r
}'

echo ""
printf "${BOLD}Done.${NC}\n"
