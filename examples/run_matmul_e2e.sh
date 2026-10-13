#!/usr/bin/env bash
# run_matmul_e2e.sh — End-to-end NPU matmul pipeline.
#
# Steps:
#   1. Lower matmul.ll → NPU assembly (.s)   — shows the human-readable output
#   2. Lower matmul.ll → ELF object (.o)     — shows the binary artifact
#   3. Inspect the ELF (header + sections + hex dump)
#   4. Build the NPU simulator (npu-simulator/)
#   5. Run demo_matmul_elf — loads the .o, runs it on the simulator, verifies
#
# Run from the llvm-project repo root:
#   bash examples/run_matmul_e2e.sh
#
# Optional flags:
#   --trace    enable per-instruction simulation trace

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUILD_DIR="$REPO_ROOT/build"
SIM_BUILD="$REPO_ROOT/npu-simulator/build"
LLC="$BUILD_DIR/bin/llc"
READELF="$BUILD_DIR/bin/llvm-readelf"
LL_SRC="$REPO_ROOT/examples/matmul.ll"
ASM_OUT="$BUILD_DIR/examples/matmul.s"
OBJ_OUT="$BUILD_DIR/examples/matmul.o"
TRACE="${1:-}"

BOLD=$'\033[1m'; CYAN=$'\033[36m'; GREEN=$'\033[32m'; RESET=$'\033[0m'

banner() { echo; echo "${BOLD}${CYAN}━━━ $* ━━━${RESET}"; echo; }

# ── Preflight ─────────────────────────────────────────────────────────────────
banner "Preflight checks"

if [[ ! -x "$LLC" ]]; then
    echo "  llc not found at $LLC"
    echo "  Build LLVM first:"
    echo "    cmake -G Ninja -S llvm -B build \\"
    echo "      -DCMAKE_BUILD_TYPE=Release \\"
    echo "      -DLLVM_TARGETS_TO_BUILD='X86;NPU' \\"
    echo "      -DLLVM_ENABLE_ASSERTIONS=ON"
    echo "    cmake --build build --target llc llvm-readelf"
    exit 1
fi

echo "  llc:     $LLC"
echo "  readelf: $READELF"
echo "  source:  $LL_SRC"
echo "  asm out: $ASM_OUT"
echo "  obj out: $OBJ_OUT"

mkdir -p "$BUILD_DIR"

# ── Step 1: Lower to assembly ─────────────────────────────────────────────────
banner "Step 1 — Lower matmul.ll → NPU assembly"

"$LLC" -mtriple=npu-unknown-unknown -filetype=asm "$LL_SRC" -o "$ASM_OUT"
echo "${GREEN}Generated: $ASM_OUT${RESET}"
echo
echo "  NPU assembly output:"
echo "  ────────────────────"
sed 's/^/  /' "$ASM_OUT"

# ── Step 2: Lower to ELF object ──────────────────────────────────────────────
banner "Step 2 — Lower matmul.ll → ELF object"

"$LLC" -mtriple=npu-unknown-unknown -filetype=obj "$LL_SRC" -o "$OBJ_OUT"
echo "${GREEN}Generated: $OBJ_OUT ($(wc -c < "$OBJ_OUT") bytes)${RESET}"

# ── Step 3: Inspect ELF ──────────────────────────────────────────────────────
banner "Step 3 — Inspect ELF artifact"

echo "  ELF header:"
"$READELF" -h "$OBJ_OUT" | grep -E 'Class|Data|Machine|Type|Entry' | sed 's/^/    /'

echo
echo "  Sections:"
"$READELF" -S "$OBJ_OUT" | sed 's/^/    /'

echo
echo "  Symbols:"
"$READELF" -s "$OBJ_OUT" | sed 's/^/    /'

echo
echo "  Relocations (.rela.text, if any):"
"$READELF" -r "$OBJ_OUT" 2>&1 | sed 's/^/    /'

echo
echo "  .text hex dump (32-byte instruction words):"
"$READELF" --hex-dump=.text "$OBJ_OUT" | sed 's/^/    /'

# ── Step 4: Build simulator ───────────────────────────────────────────────────
banner "Step 4 — Build NPU simulator"

cmake -S "$REPO_ROOT/npu-simulator" -B "$SIM_BUILD" \
    -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER=clang 2>&1 | tail -5
cmake --build "$SIM_BUILD" --target demo_matmul_elf 2>&1 | tail -5
echo "${GREEN}Built: $SIM_BUILD/demo_matmul_elf${RESET}"

# ── Step 5: Run on simulator ─────────────────────────────────────────────────
banner "Step 5 — Run matmul_4x4 on NPU simulator"

echo "  Loading ELF, executing on simulator, verifying against CPU reference..."
echo
"$SIM_BUILD/demo_matmul_elf" "$OBJ_OUT" $TRACE

echo
banner "Done"
echo "  Assembly:  $ASM_OUT"
echo "  ELF .o:    $OBJ_OUT"
echo "  Simulator: $SIM_BUILD/demo_matmul_elf"
