; layer2_tensor_engine.ll — Complete Layer 2 (tensor engine) LLVM IR reference.
;
; The tensor engine shares the cinst[] stream with the control core.
; You do NOT need KLAUNCH — T_* instructions are issued inline by the control
; core and dispatched to the tensor engine's execution unit automatically.
;
; Compile:
;   llc -march=npu layer2_tensor_engine.ll -o layer2_tensor_engine.s
;
; ─────────────────────────────────────────────────────────────────────────────
; TILE REGISTER ENCODING — the most important design decision
; ─────────────────────────────────────────────────────────────────────────────
;
; The tensor engine has 8 tile registers t0..t7.  In LLVM IR they are NOT a
; register class.  They are i64 SSA constants:
;
;   i64 0  →  t0      (printed as "t0" in assembly)
;   i64 1  →  t1
;   ...
;   i64 7  →  t7
;
; Every tile intrinsic takes and returns i64 tile tokens.  The return value is
; always the same constant as the destination tile — it exists only so that
; LLVM sees a data dependency and keeps instructions in order:
;
;   %t0 = call i64 @llvm.npu.acc_clr(i64 0)   ; clear tile 0, returns i64 0
;   %t0a = call i64 @llvm.npu.mma(i64 %t0,    ; MMA into tile 0 (=%t0=0),
;                                  i64 1,       ; using tile 1
;                                  i64 2)       ; and tile 2
;   ; %t0a is still i64 0 — the same tile register, now with updated HW state.
;
; Using the return value as the next call's first arg creates an SSA chain that
; preserves the program order through LLVM's optimiser (otherwise DCE would
; remove them as "pure" computations that nobody uses).
;
; ─────────────────────────────────────────────────────────────────────────────
; COMPILE-TIME CONSTANT RULES
; ─────────────────────────────────────────────────────────────────────────────
;
;   Tile numbers   : must be i64 literals 0–7.  Runtime values will crash in
;                    getTileConst() in NPUISelLowering.cpp.
;   scale factor   : must be a constant float.  Runtime floats abort in
;                    LowerIntrinsicCall (checked via dyn_cast<ConstantFPSDNode>).
;   bcast value    : same — must be a constant float.
;   cfg_shape dims : M/K/N can be runtime i64 values (held in GPRs).
;   load/store count: runtime i64 OK (held in a GPR).
;
; ─────────────────────────────────────────────────────────────────────────────

target triple = "npu"

; ── Declare all Layer 2 intrinsics ───────────────────────────────────────────
; (these are generated from IntrinsicsNPU.td — shown here for reference)

; §4 Config
declare void @llvm.npu.cfg_shape(i64, i64, i64)   ; M, K, N — all in GPRs

; §5 Compute — all return the destination tile token (same i64 as input dt)
declare i64 @llvm.npu.acc_clr(i64)                ; dt → clears tile, returns dt
declare i64 @llvm.npu.mma    (i64, i64, i64)      ; acc, A, B → acc += A*B
declare i64 @llvm.npu.matmul (i64, i64, i64)      ; dt, A, B  → dt  = A*B
declare i64 @llvm.npu.dot    (i64, i64, i64)      ; dt, A, B  → dt += dot(A,B)

; §6 Data movement
declare i64  @llvm.npu.load   (i64, ptr, i64)     ; dt, ptr, count → dt loaded
declare i64  @llvm.npu.loadw  (i64, ptr, i64)     ; dt, ptr, count → dt (transposed)
declare void @llvm.npu.store  (ptr, i64, i64)     ; ptr, tile, count
declare void @llvm.npu.dma_in (i64, i64, i64)     ; dst_scratch, src_dram, bytes
declare void @llvm.npu.dma_out(i64, i64, i64)     ; dst_dram, src_scratch, bytes

; §7 Activations
declare i64 @llvm.npu.relu (i64)                  ; tile → RELU in-place
declare i64 @llvm.npu.scale(i64, float)           ; tile, CONSTANT factor
declare i64 @llvm.npu.bcast(i64, float)           ; dt,   CONSTANT value → fill tile

; §8 Synchronization
declare void @llvm.npu.fence()                    ; flush pending tensor ops
declare void @llvm.npu.wait()                     ; wait until tensor engine idle
declare void @llvm.npu.barrier()                  ; global barrier across cores


; ═════════════════════════════════════════════════════════════════════════════
; §1  SIMPLE MATMUL — dt = A × B, single shot, no accumulation
; ═════════════════════════════════════════════════════════════════════════════
;
; C equivalent:
;   void simple_matmul(float *A, float *B, float *C, int M, int K, int N) {
;       cfg_shape(M, K, N);
;       tile t0 = load(A, M*K);   // activations
;       tile t1 = loadw(B, K*N);  // weights (transposed)
;       tile t2 = matmul(t2, t0, t1);
;       store(C, t2, M*N);
;   }
;
; Expected assembly:
;   simple_matmul:
;     CFG.SHAPE x12, x13, x14    ; M, K, N from args
;     LOAD  t0, x10, x15         ; load A (count = M*K)
;     LOADW t1, x11, x16         ; load B transposed (count = K*N)
;     MATMUL t2, t0, t1          ; t2 = t0 * t1
;     STORE  x3, t2, x17         ; store to C (count = M*N)
;     JALR x0, x1, 0
;
; Note: MATMUL overwrites dt — no need for ACC.CLR first.
;       Use MMA when you need accumulation across K blocks.
; ─────────────────────────────────────────────────────────────────────────────
define void @simple_matmul(ptr %A, ptr %B, ptr %C,
                            i64 %M, i64 %K, i64 %N) {
entry:
  ; configure tile dimensions (can be runtime values — held in GPRs)
  call void @llvm.npu.cfg_shape(i64 %M, i64 %K, i64 %N)

  ; load activation tile into t0 (tile index MUST be i64 literal 0..7)
  %count_A = mul i64 %M, %K
  %t0 = call i64 @llvm.npu.load(i64 0, ptr %A, i64 %count_A)

  ; load weight tile into t1 (transposed — rows and cols swapped in memory)
  %count_B = mul i64 %K, %N
  %t1 = call i64 @llvm.npu.loadw(i64 1, ptr %B, i64 %count_B)

  ; MATMUL: t2 = t0 × t1  (overwrites t2, no prior ACC.CLR needed)
  %t2 = call i64 @llvm.npu.matmul(i64 2, i64 %t0, i64 %t1)

  ; store result tile t2 to C
  %count_C = mul i64 %M, %N
  call void @llvm.npu.store(ptr %C, i64 %t2, i64 %count_C)

  ret void
}


; ═════════════════════════════════════════════════════════════════════════════
; §2  K-BLOCK MMA LOOP — accumulate over multiple K tiles
; ═════════════════════════════════════════════════════════════════════════════
;
; When K is larger than one tile, split it into k_blocks and accumulate:
;   acc = 0
;   for i in range(k_blocks):
;       acc += A[i] * B[i]
;
; This is the standard tiled matmul inner loop.  ACC.CLR zeros the accumulator
; before the loop; MMA accumulates in-place each iteration.
;
; C equivalent:
;   void mma_kblock(float *A, float *B, float *C,
;                   int M, int tile_K, int N, int k_blocks) {
;       cfg_shape(M, tile_K, N);
;       tile t2 = acc_clr(t2);               // zero the accumulator
;       for (int i = 0; i < k_blocks; i++) {
;           tile t0 = load(A + i*M*tile_K, M*tile_K);
;           tile t1 = loadw(B + i*tile_K*N, tile_K*N);
;           t2 = mma(t2, t0, t1);            // t2 += t0 * t1
;       }
;       store(C, t2, M*N);
;   }
;
; The SSA tile token chain:
;   %t2_init = acc_clr(2)            → t2 = 0
;   iter 0:  %t2_a = mma(%t2_init, %t0, %t1)  → t2 += A[0]*B[0]
;   iter 1:  %t2_b = mma(%t2_a,    %t0, %t1)  → t2 += A[1]*B[1]
;   ...
;   The phi node threads the tile token through the loop so LLVM preserves order.
; ─────────────────────────────────────────────────────────────────────────────
define void @mma_kblock(ptr %A, ptr %B, ptr %C,
                         i64 %M, i64 %tile_K, i64 %N, i64 %k_blocks) {
entry:
  call void @llvm.npu.cfg_shape(i64 %M, i64 %tile_K, i64 %N)

  ; clear accumulator tile 2 before the loop
  %t2_init = call i64 @llvm.npu.acc_clr(i64 2)

  ; precompute per-block strides
  %stride_A = mul i64 %M, %tile_K     ; elements per A tile
  %stride_B = mul i64 %tile_K, %N     ; elements per B tile

  br label %loop

loop:
  ; phi nodes carry: loop counter, current A/B pointer, and tile 2 SSA token
  %i      = phi i64 [0,        %entry], [%i_next, %loop]
  %pA     = phi ptr [%A,       %entry], [%pA_next, %loop]
  %pB     = phi ptr [%B,       %entry], [%pB_next, %loop]
  %t2_cur = phi i64 [%t2_init, %entry], [%t2_new,  %loop]

  ; load current A tile block into t0
  %t0 = call i64 @llvm.npu.load (i64 0, ptr %pA, i64 %stride_A)

  ; load current B tile block into t1 (transposed)
  %t1 = call i64 @llvm.npu.loadw(i64 1, ptr %pB, i64 %stride_B)

  ; MMA: t2 += t0 * t1
  ; pass %t2_cur (the phi token) as the accumulator argument so LLVM sees
  ; the dependency chain and does not reorder iterations
  %t2_new = call i64 @llvm.npu.mma(i64 %t2_cur, i64 %t0, i64 %t1)

  ; advance pointers (GEP by element count, not bytes)
  %pA_next = getelementptr float, ptr %pA, i64 %stride_A
  %pB_next = getelementptr float, ptr %pB, i64 %stride_B
  %i_next  = add i64 %i, 1
  %done    = icmp eq i64 %i_next, %k_blocks
  br i1 %done, label %exit, label %loop

exit:
  ; store final accumulator to C
  %count_C = mul i64 %M, %N
  call void @llvm.npu.store(ptr %C, i64 %t2_new, i64 %count_C)
  ret void
}


; ═════════════════════════════════════════════════════════════════════════════
; §3  ACTIVATIONS — RELU, SCALE, BCAST
; ═════════════════════════════════════════════════════════════════════════════
;
; All three apply element-wise to a tile register in-place.
;
; RELU  : max(0, x)  — no float operand, pure tile op
; SCALE : x *= factor  — factor is a CONSTANT float (burned into fimm field)
; BCAST : tile = fill(val) — val is a CONSTANT float (materialized via LI+BCAST)
;
; SCALE vs BCAST encoding:
;   SCALE factor → encoded as IEEE 754 double bits in cinst.fimm (bytes 16–23)
;                  Control core never sees this as a register — it is immediate.
;   BCAST val    → encoded as IEEE 754 float bits in a GPR (cinst.rs1).
;                  The simulator reads the float from the integer register x[rs1].
;                  LowerIntrinsicCall calls DAG.getConstant (not getTargetConstant)
;                  so the register allocator materializes it via LI automatically.
;
; Expected assembly for activate_tile:
;   activate_tile:
;     LOAD  t0, x10, x11          ; load tile
;     RELU  t0, t0                ; in-place relu
;     SCALE t0, t0, 0.5           ; multiply every element by 0.5 (fimm)
;     LI    x5, 1065353216        ; 0x3F800000 = 1.0f bit pattern
;     BCAST t1, x5                ; fill t1 with 1.0  (GPR source)
;     STORE x3, t0, x11           ; store result
;     JALR x0, x1, 0
; ─────────────────────────────────────────────────────────────────────────────
define void @activate_tile(ptr %src, ptr %dst, i64 %count) {
entry:
  ; load activation tile into t0
  %t0 = call i64 @llvm.npu.load(i64 0, ptr %src, i64 %count)

  ; RELU: element-wise max(0, x)
  %t0_relu = call i64 @llvm.npu.relu(i64 %t0)

  ; SCALE: multiply every element by 0.5
  ; factor MUST be a compile-time float constant — runtime float will abort
  %t0_scaled = call i64 @llvm.npu.scale(i64 %t0_relu, float 5.000000e-01)

  ; BCAST: fill tile 1 with the constant 1.0 (useful for bias addition later)
  ; val MUST be a compile-time float constant
  %t1 = call i64 @llvm.npu.bcast(i64 1, float 1.000000e+00)

  ; store result
  call void @llvm.npu.store(ptr %dst, i64 %t0_scaled, i64 %count)
  ret void
}


; ═════════════════════════════════════════════════════════════════════════════
; §4  DMA STAGING — move data DRAM → SRAM → compute → SRAM → DRAM
; ═════════════════════════════════════════════════════════════════════════════
;
; SRAM (scratchpad) has 1-cycle latency vs 100 cycles for DRAM.
; The DMA engine copies data asynchronously between them.
; FENCE/WAIT ensure the DMA is complete before the tensor engine reads the data.
;
; Address convention:
;   DRAM  : any address with high bit clear (< 0x80000000)
;   SRAM  : SCRATCH_BASE | offset  (SCRATCH_BASE = 0x80000000)
;
; Typical sequence:
;   1. DMA.IN  scratch_A, dram_A, bytes   (async: DRAM → SRAM)
;   2. DMA.IN  scratch_B, dram_B, bytes   (async, overlapped with above)
;   3. FENCE                               (wait for both DMAs to land)
;   4. CFG.SHAPE / LOAD / MATMUL           (now reads from fast SRAM)
;   5. STORE to scratch_C                  (result lands in SRAM)
;   6. DMA.OUT dram_C, scratch_C, bytes    (async: SRAM → DRAM)
;   7. FENCE                               (ensure write-back before returning)
;
; C equivalent:
;   void dma_matmul(int64_t scrA, int64_t scrB, int64_t scrC,
;                   int64_t drmA, int64_t drmB, int64_t drmC,
;                   int64_t bytes, int64_t M, int64_t K, int64_t N) {
;       dma_in(scrA, drmA, bytes);
;       dma_in(scrB, drmB, bytes);
;       fence();                      // wait for DMA before tensor reads
;       cfg_shape(M, K, N);
;       t0 = load (t0, scrA, M*K);   // fast SRAM load
;       t1 = loadw(t1, scrB, K*N);
;       t2 = matmul(t2, t0, t1);
;       store(scrC, t2, M*N);
;       dma_out(drmC, scrC, bytes);
;       fence();
;   }
;
; Note: DMA addresses are i64 raw integers (NPU memory addresses), NOT pointers.
;       dma_in/dma_out take (dst:i64, src:i64, bytes:i64), not ptr.
; ─────────────────────────────────────────────────────────────────────────────
define void @dma_matmul(
    i64 %scrA, i64 %scrB, i64 %scrC,   ; SRAM addresses (high bit set)
    i64 %drmA, i64 %drmB, i64 %drmC,   ; DRAM addresses
    i64 %bytes,
    i64 %M, i64 %K, i64 %N) {
entry:
  ; ── Stage 1: async DMA from DRAM to SRAM ─────────────────────────────────
  ; Both transfers fire concurrently — the DMA engine handles them in parallel.
  call void @llvm.npu.dma_in(i64 %scrA, i64 %drmA, i64 %bytes)
  call void @llvm.npu.dma_in(i64 %scrB, i64 %drmB, i64 %bytes)

  ; ── Stage 2: wait for DMA completion ─────────────────────────────────────
  ; Without FENCE the tensor LOAD might see stale DRAM data still in SRAM.
  call void @llvm.npu.fence()

  ; ── Stage 3: tensor compute (now reading from fast SRAM) ─────────────────
  call void @llvm.npu.cfg_shape(i64 %M, i64 %K, i64 %N)

  ; LOAD and LOADW take a ptr operand — inttoptr converts the i64 SRAM address
  %pA = inttoptr i64 %scrA to ptr
  %pB = inttoptr i64 %scrB to ptr
  %pC = inttoptr i64 %scrC to ptr

  %count_A = mul i64 %M, %K
  %count_B = mul i64 %K, %N
  %count_C = mul i64 %M, %N

  %t0 = call i64 @llvm.npu.load (i64 0, ptr %pA, i64 %count_A)
  %t1 = call i64 @llvm.npu.loadw(i64 1, ptr %pB, i64 %count_B)
  %t2 = call i64 @llvm.npu.matmul(i64 2, i64 %t0, i64 %t1)

  call void @llvm.npu.store(ptr %pC, i64 %t2, i64 %count_C)

  ; ── Stage 4: async DMA write-back SRAM → DRAM ────────────────────────────
  call void @llvm.npu.dma_out(i64 %drmC, i64 %scrC, i64 %bytes)
  call void @llvm.npu.fence()   ; ensure write-back before caller reads DRAM

  ret void
}


; ═════════════════════════════════════════════════════════════════════════════
; §5  SYNCHRONIZATION — FENCE vs WAIT vs BARRIER
; ═════════════════════════════════════════════════════════════════════════════
;
; Three sync primitives, different scopes:
;
;   FENCE   — flush all in-flight tensor-engine ops (DMA + compute).
;              Control core stalls until the tensor engine drains.
;              Use after DMA.IN before LOAD, and after STORE before DMA.OUT.
;
;   WAIT    — wait until the tensor engine is fully IDLE (stronger than FENCE).
;              Use before reading results from SRAM with scalar LW instructions.
;
;   BARRIER — global barrier: all cores must reach this point before any proceeds.
;             Used between KLAUNCH phases when multiple vector-core groups share
;             state (written by one, read by another).
;
; When to use each:
;   DMA.IN → FENCE → LOAD     (FENCE is enough: just need DMA done before LOAD)
;   STORE  → WAIT  → LW       (WAIT: scalar LW reads the same SRAM the TE wrote)
;   phase1 → BARRIER → phase2 (multi-core coordination)
; ─────────────────────────────────────────────────────────────────────────────
define void @sync_demo(ptr %A, ptr %B, ptr %out, i64 %count) {
entry:
  ; FENCE: flush tensor ops before issuing more
  call void @llvm.npu.cfg_shape(i64 4, i64 4, i64 4)
  %t0 = call i64 @llvm.npu.load(i64 0, ptr %A, i64 %count)
  %t1 = call i64 @llvm.npu.load(i64 1, ptr %B, i64 %count)
  %t2 = call i64 @llvm.npu.matmul(i64 2, i64 %t0, i64 %t1)
  call void @llvm.npu.store(ptr %out, i64 %t2, i64 %count)

  ; WAIT: tensor engine must be idle before the control core reads %out with LW
  call void @llvm.npu.wait()

  ; BARRIER: synchronize with other cores at this point
  call void @llvm.npu.barrier()

  ret void
}


; ═════════════════════════════════════════════════════════════════════════════
; LAYER 2 INSTRUCTION → ASSEMBLY QUICK REFERENCE
; ═════════════════════════════════════════════════════════════════════════════
;
; Intrinsic call                            Assembly emitted
; ─────────────────────────────────────────────────────────────────
; cfg_shape(M, K, N)                      → CFG.SHAPE xM, xK, xN
; acc_clr(2)                              → ACC.CLR t2
; matmul(2, t0, t1)                       → MATMUL t2, t0, t1
; mma(t2, t0, t1)                         → MMA t2, t0, t1
; dot(t2, t0, t1)                         → DOT t2, t0, t1
; load(0, ptr, count)                     → LOAD t0, xCount(xPtr)
; loadw(1, ptr, count)                    → LOADW t1, xCount(xPtr)
; store(ptr, t2, count)                   → STORE xPtr, t2, xCount
; dma_in(scratchAddr, dramAddr, bytes)    → DMA.IN xDst, xSrc, xBytes
; dma_out(dramAddr, scratchAddr, bytes)   → DMA.OUT xDst, xSrc, xBytes
; relu(t0)                                → RELU t0, t0
; scale(t0, 0.5f)                         → SCALE t0, t0, 0.5
; bcast(t1, 1.0f)                         → LI xN, 0x3F800000
;                                            BCAST t1, xN
; fence()                                 → FENCE
; wait()                                  → WAIT
; barrier()                               → BARRIER
;
; ─────────────────────────────────────────────────────────────────────────────
; WHAT DOES NOT WORK IN LAYER 2
; ─────────────────────────────────────────────────────────────────────────────
;
;   Tile number as variable:
;     %tile = add i64 0, 0
;     call i64 @llvm.npu.acc_clr(i64 %tile)   ← CRASH in getTileConst()
;     Tile numbers must be i64 literals (0, 1, 2 ... 7).
;
;   Runtime float for scale/bcast:
;     call i64 @llvm.npu.scale(i64 0, float %f)  ← CRASH: needs constant
;     call i64 @llvm.npu.bcast(i64 0, float %f)  ← same
;
;   T_CONV / T_POOL:
;     These opcodes exist in npu.h cop_t enum but are STUBS in npu.c.
;     There are no @llvm.npu.conv / @llvm.npu.pool intrinsics.
;     Build conv from repeated MMA tiles; pool from vector-core primitives.
;
;   Tile register > 7:
;     call i64 @llvm.npu.load(i64 8, ...)   ← out of range, undefined behavior
; ─────────────────────────────────────────────────────────────────────────────
