# Standalone CMake project for the fully-programmable NPU simulator.
# NOT wired into the IREE build — builds independently.
#
# Quick start:
#   bash build.sh           # → build/
#   bash build.sh --test    # build + run all tests
#
# Manual:
#   cmake -S . -B build && cmake --build build
#   ctest --test-dir build --output-on-failure
#
# Directory layout:
#   include/   public headers (npu.h, npu_asm.h, npu_cmd.h, npu_kdrv.h,
#                               npu_rt.h, npu_runtime.h)
#   src/       library sources  (npu.c, npu_asm.c, npu_kdrv.c, npu_rt.c,
#                                npu_runtime.c)
#   tools/     CLI tools        (npu_driver, npu_elf_run)
#   examples/  demos            (npu_demo, my_driver, demo_vadd, demo_rt)
#   docs/      design documents

cmake_minimum_required(VERSION 3.16)
project(npu_simulator LANGUAGES C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(WARN -Wall -Wextra -Wno-missing-field-initializers -Wno-unused-function)

# ── core simulator library ────────────────────────────────────────────────────
# npu + npu_asm: simulator interpreter, text assembler, binary encoder/decoder.
# PUBLIC include/ propagates to every target that links against this library,
# so all source files can write #include "npu.h" without path changes.
add_library(npu STATIC src/npu.c src/npu_asm.c)
target_include_directories(npu PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
if(NOT WIN32)
  target_link_libraries(npu PUBLIC m)   # expf / logf / roundf …
endif()
target_compile_options(npu PRIVATE ${WARN})

# ── Layer 1 runtime: npu_runtime.h ───────────────────────────────────────────
# Synchronous CUDA-style API: npu_malloc / npu_memcpy_h2d / npu_launch.
add_library(npu_runtime STATIC src/npu_runtime.c)
target_link_libraries(npu_runtime PUBLIC npu)
target_compile_options(npu_runtime PRIVATE ${WARN})

# ── Layer 2 runtime: npu_rt.h  (full driver stack) ───────────────────────────
# Kernel driver + firmware dispatch loop + async CUDA-style runtime.
# npuMemcpy_h2d / npuGemm / npuLaunchKernel push commands; npuSynchronize runs them.
add_library(npu_stack STATIC src/npu_kdrv.c src/npu_rt.c)
target_link_libraries(npu_stack PUBLIC npu)
target_compile_options(npu_stack PRIVATE ${WARN})

# ── tools ─────────────────────────────────────────────────────────────────────

# Full CLI driver: load ELF/.s files, inline asm, memory dump, stats.
add_executable(npu_driver tools/npu_driver.c)
target_link_libraries(npu_driver PRIVATE npu)
target_compile_options(npu_driver PRIVATE ${WARN})

# ELF loader: execute NPU .o files produced by:
#   llc -mtriple=npu -filetype=obj -o out.o input.ll
add_executable(npu_elf_run tools/npu_elf_run.c)
target_link_libraries(npu_elf_run PRIVATE npu)
target_compile_options(npu_elf_run PRIVATE ${WARN})

# ── examples ──────────────────────────────────────────────────────────────────

# Original 7 self-checking simulator demos (vadd, softmax, matmul, …).
add_executable(npu_demo examples/examples.c)
target_link_libraries(npu_demo PRIVATE npu)
target_compile_options(npu_demo PRIVATE ${WARN})

# Layer-1 runtime demo: vadd, dot-product, saxpy.
add_executable(demo_vadd examples/demo_vadd.c)
target_link_libraries(demo_vadd PRIVATE npu_runtime)
target_compile_options(demo_vadd PRIVATE ${WARN})

# Layer-2 full-stack demo: GEMM, ReLU, custom kernel, ML pipeline.
add_executable(demo_rt examples/demo_rt.c)
target_link_libraries(demo_rt PRIVATE npu_stack)
target_compile_options(demo_rt PRIVATE ${WARN})

# TE + VC demo: MMA chain, MATMUL+SCALE (via RT), vector add + softmax (via kdrv).
add_executable(demo_rt_te_vc examples/demo_rt_te_vc.c)
target_link_libraries(demo_rt_te_vc PRIVATE npu_stack)
target_compile_options(demo_rt_te_vc PRIVATE ${WARN})

# Demo: load an LLVM-compiled ELF kernel and run it through the runtime.
# Build the kernel separately first:
#   llc -mtriple=npu -filetype=obj -o <build>/vadd.o kernels/vadd.ll
add_executable(demo_elf_load examples/demo_elf_load.c)
target_link_libraries(demo_elf_load PRIVATE npu_stack)
target_compile_options(demo_elf_load PRIVATE ${WARN})

# Demo: load LLVM-compiled matmul_4x4 ELF and verify against CPU reference.
# Build the kernel separately first:
#   llc -mtriple=npu-unknown-unknown -filetype=obj \
#       examples/matmul.ll -o <build>/matmul.o
add_executable(demo_matmul_elf examples/demo_matmul_elf.c)
target_link_libraries(demo_matmul_elf PRIVATE npu_stack m)
target_compile_options(demo_matmul_elf PRIVATE ${WARN})

# ── tests ─────────────────────────────────────────────────────────────────────
enable_testing()
add_test(NAME sim_demos    COMMAND npu_demo)
add_test(NAME runtime_demo COMMAND demo_vadd)
add_test(NAME stack_demo   COMMAND demo_rt)
add_test(NAME te_vc_demo   COMMAND demo_rt_te_vc)
