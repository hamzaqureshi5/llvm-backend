#!/usr/bin/env bash
# build.sh — Build the NPU simulator software stack.
#
# Usage:
#   bash build.sh              # Release build  → build/
#   bash build.sh --debug      # Debug build    → build/
#   bash build.sh --clean      # Wipe build dir then build
#   bash build.sh --test       # Build + run all CTest tests
#   BUILD_DIR=out bash build.sh  # Custom output directory
#
# After building, run individual targets:
#   ./build/npu_demo           # original simulator self-tests
#   ./build/my_driver          # minimalist driver (structs / asm / memory)
#   ./build/npu_driver -h      # full CLI driver
#   ./build/demo_vadd          # Layer-1 runtime: vadd, dot, saxpy
#   ./build/demo_rt            # Layer-2 stack:   GEMM, ReLU, pipeline
#   ./build/npu_elf_run foo.o  # run an NPU ELF .o from llc

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD="${BUILD_DIR:-${ROOT}/build}"
BUILD_TYPE="Release"
RUN_TESTS=0
CLEAN=0

# ── parse arguments ───────────────────────────────────────────────────────────
for arg in "$@"; do
    case "$arg" in
        --debug)  BUILD_TYPE="Debug" ;;
        --clean)  CLEAN=1 ;;
        --test)   RUN_TESTS=1 ;;
        --help|-h)
            sed -n '2,20p' "$0" | sed 's/^# \{0,1\}//'
            exit 0 ;;
        *)
            echo "Unknown option: $arg  (use --help)" >&2
            exit 1 ;;
    esac
done

# ── colours ───────────────────────────────────────────────────────────────────
BOLD='\033[1m'; GREEN='\033[0;32m'; CYAN='\033[0;36m'; NC='\033[0m'

step() { printf "${BOLD}${CYAN}▶ %s${NC}\n" "$*"; }
ok()   { printf "${GREEN}✓ %s${NC}\n" "$*"; }

# ── clean ─────────────────────────────────────────────────────────────────────
if [[ $CLEAN -eq 1 && -d "$BUILD" ]]; then
    step "Cleaning $BUILD"
    rm -rf "$BUILD"
fi

# ── configure ─────────────────────────────────────────────────────────────────
step "Configuring  ($BUILD_TYPE)  →  $BUILD"
cmake -S "$ROOT" -B "$BUILD" \
      -DCMAKE_BUILD_TYPE="$BUILD_TYPE" \
      -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
      --log-level=WARNING \
      -G "$(cmake --help 2>&1 | grep -m1 'Ninja\|Unix' | sed 's/^[^a-zA-Z]*//' | cut -d' ' -f1-2 | sed 's/ $//')"

# ── build ─────────────────────────────────────────────────────────────────────
JOBS=$(nproc 2>/dev/null || sysctl -n hw.logicalcpu 2>/dev/null || echo 4)
step "Building ($JOBS jobs)"
cmake --build "$BUILD" --parallel "$JOBS"

# ── summary ───────────────────────────────────────────────────────────────────
echo ""
printf "${BOLD}Build complete.${NC}  Binaries in ${CYAN}${BUILD}/${NC}\n\n"

print_bin() {
    local name="$1" desc="$2"
    if [[ -f "${BUILD}/${name}" ]]; then
        printf "  ${GREEN}%-18s${NC}  %s\n" "$name" "$desc"
    fi
}

print_bin npu_demo      "original self-checking simulator demos"
print_bin npu_elf_run   "run NPU ELF .o files from llc"
print_bin my_driver     "minimalist driver  (structs / text-asm / memory)"
print_bin npu_driver    "full CLI driver    (--help for all options)"
print_bin demo_vadd     "Layer-1 runtime    (vadd · dot · saxpy)"
print_bin demo_rt       "Layer-2 full stack (GEMM · ReLU · pipeline)"

echo ""
printf "Run all tests:   ${CYAN}ctest --test-dir %s --output-on-failure${NC}\n" "$BUILD"
echo ""

# ── optional test run ─────────────────────────────────────────────────────────
if [[ $RUN_TESTS -eq 1 ]]; then
    step "Running tests"
    ctest --test-dir "$BUILD" --output-on-failure
    ok "All tests passed"
fi
