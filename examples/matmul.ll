; matmul.ll — 4×4 float matrix multiply for the NPU tensor engine.
target triple = "npu-unknown-unknown"
;
; Implements:  C = A × B  (4×4 float matrices, row-major)
;
; Calling convention (NPU ABI):
;   x10 = ptr to C  (output, 16 floats)
;   x11 = ptr to A  (16 floats)
;   x12 = ptr to B  (16 floats)
;
; Tile mapping:
;   t0 = accumulator (C result)
;   t1 = A matrix
;   t2 = B matrix
;
; Compile to assembly:
;   llc -mtriple=npu-unknown-unknown -filetype=asm matmul.ll -o matmul.s
;
; Compile to ELF object:
;   llc -mtriple=npu-unknown-unknown -filetype=obj matmul.ll -o matmul.o
;
; Run end-to-end (see run_matmul_e2e.sh):
;   bash examples/run_matmul_e2e.sh

declare void @llvm.npu.cfg.shape(i64, i64, i64)
declare i64  @llvm.npu.acc.clr(i64)
declare i64  @llvm.npu.load(i64, ptr, i64)
declare i64  @llvm.npu.mma(i64, i64, i64)
declare void @llvm.npu.store(ptr, i64, i64)
declare void @llvm.npu.fence()

define void @matmul_4x4(ptr %C, ptr %A, ptr %B) {
  ; Set tile dimensions: M=4 rows, K=4 shared, N=4 cols.
  call void @llvm.npu.cfg.shape(i64 4, i64 4, i64 4)

  ; Zero accumulator tile 0 (will hold the result C).
  %t0 = call i64 @llvm.npu.acc.clr(i64 0)

  ; Load A (16 floats = 4×4) into tile 1.
  %t1 = call i64 @llvm.npu.load(i64 1, ptr %A, i64 16)

  ; Load B (16 floats = 4×4) into tile 2.
  %t2 = call i64 @llvm.npu.load(i64 2, ptr %B, i64 16)

  ; Accumulate: tile0 += tile1 × tile2  →  C = A × B
  %t0_out = call i64 @llvm.npu.mma(i64 %t0, i64 %t1, i64 %t2)

  ; Store tile 0 (result, 16 floats) to C.
  call void @llvm.npu.store(ptr %C, i64 %t0_out, i64 16)

  ; Fence: flush tile engine before control returns.
  call void @llvm.npu.fence()

  ret void
}
