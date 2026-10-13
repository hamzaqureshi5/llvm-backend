//===- IntrinsicsNPU.td - NPU intrinsic definitions --------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Defines @llvm.npu.* intrinsics — the LLVM IR-level interface to the three
// NPU execution engines described in npu-simulator/llvm_lowering.md.
//
// Each intrinsic maps 1:1 to a simulator instruction (cinst or vinst opcode).
// The mapping is documented on every def.  The intrinsic names intentionally
// match the simulator's text assembler mnemonics (npu_asm.h CTAB / VTAB).
//
// Design notes
// ─────────────
// Tile registers (tensor engine, Section 5–7) are SSA values of type i64.
// The i64 value is the tile register number (0–7); the actual NxM float
// payload lives in hardware and is tracked by SSA data flow.  A production
// backend would use `target("npu.tile")` (a TargetExtType like x86 AMX) for
// stronger type safety, but i64 keeps the teaching backend self-contained.
//
// Vector registers (vector core, Section 9–11) are <16 x float> — matching
// HW_LANES=16 physical lanes defined in npu.h.
//
// Intrinsics with side-effects (memory, engine state) are marked
// IntrHasSideEffects to prevent CSE/DCE from removing them.
//
//===----------------------------------------------------------------------===//

let TargetPrefix = "npu" in {

// ─── Layer 1 : Control-core kernel launch (Section 3) ────────────────────────

// Launch `grid_size` instances of a registered vector kernel.
// kernel_id : i32 — index from npu_register_kernel(); stored in cinst.rd
// grid_size : i32 — number of grid instances; stored in cinst.imm
// Maps to C_KLAUNCH.  Asynchronous: pair with @llvm.npu.kwait.
// NOTE: i32 args are intentional here; LowerIntrinsicCall zero-extends them.
def int_npu_klaunch : DefaultAttrsIntrinsic<
    [],
    [llvm_i32_ty,   // kernel_id
     llvm_i32_ty],  // grid_size
    [IntrHasSideEffects]
>;

// Wait for all outstanding kernel launches to finish.
// Maps to C_KWAIT.  Must follow int_npu_klaunch before reading outputs.
def int_npu_kwait : DefaultAttrsIntrinsic<
    [],
    [],
    [IntrHasSideEffects]
>;

// ─── Layer 2 : Tensor engine – config (Section 4) ────────────────────────────

// Set tile dimensions for subsequent MATMUL / MMA / LOAD / STORE operations.
// M, K, N : i64 — tile edge lengths (max MAX_TILE = 64 each).
// Maps to T_CFG_SHAPE.
// Changed from i32 to i64 so the backend's native i64 type legalizer handles
// these without triggering an unimplemented PromoteIntegerOperand path.
def int_npu_cfg_shape : DefaultAttrsIntrinsic<
    [],
    [llvm_i64_ty,   // M
     llvm_i64_ty,   // K
     llvm_i64_ty],  // N
    [IntrHasSideEffects]
>;

// ─── Layer 2 : Tensor engine – compute (Section 5) ───────────────────────────

// Clear accumulator tile register; returns the tile SSA token.
// dt : i64 — destination tile number (0–7).
// Maps to T_ACC_CLR (cinst.dt = destination tile register).
// Changed to take explicit dt so the instruction selector can emit the
// right tile register number without needing a tile register allocator.
// IntrHasSideEffects: these ops modify hardware tile register state so LLVM
// must NOT DCE/CSE them and must give them a chain (INTRINSIC_W_CHAIN, not
// INTRINSIC_WO_CHAIN).  IntrNoMem was wrong — the chain is required so that
// LowerIntrinsicCall can intercept them and emit T_ACC_CLR / T_MMA / etc.
def int_npu_acc_clr : DefaultAttrsIntrinsic<
    [llvm_i64_ty],  // tile SSA token (same value as dt, for data-flow tracking)
    [llvm_i64_ty],  // dt — destination tile number (0–7)
    [IntrHasSideEffects]
>;

// Multiply-accumulate: accumulator_tile += A * B.
// All tile arguments are SSA tokens (i64 = hardware tile register numbers 0–7).
// The accumulator IS the destination (cinst.dt = acc tile number).
// Maps to T_MMA (cinst.dt = acc, cinst.a = A, cinst.b = B).
def int_npu_mma : DefaultAttrsIntrinsic<
    [llvm_i64_ty],  // result tile token (same as acc — accumulation is in-place)
    [llvm_i64_ty,   // acc  (accumulator tile; also the destination)
     llvm_i64_ty,   // A    (input tile)
     llvm_i64_ty],  // B    (input tile)
    [IntrHasSideEffects]
>;

// Matmul without accumulation: dt = A * B.
// dt : i64 — explicit destination tile (needed because there is no accumulator
// to double as the destination).
// Maps to T_MATMUL.
def int_npu_matmul : DefaultAttrsIntrinsic<
    [llvm_i64_ty],  // result tile token (same value as dt)
    [llvm_i64_ty,   // dt — destination tile (0–7)
     llvm_i64_ty,   // A
     llvm_i64_ty],  // B
    [IntrHasSideEffects]
>;

// Dot product accumulate: dt += dot(A, B).
// Uses CF_T3 format (same as MMA/MATMUL) — dt IS the accumulator.
// dt : i64 — destination tile (also the initial accumulator).
// Maps to T_DOT.
def int_npu_dot : DefaultAttrsIntrinsic<
    [llvm_i64_ty],  // result tile token (same value as dt)
    [llvm_i64_ty,   // dt — destination and accumulator tile (0–7)
     llvm_i64_ty,   // A
     llvm_i64_ty],  // B
    [IntrHasSideEffects]
>;

// ─── Layer 2 : Tensor engine – data movement (Section 6) ─────────────────────

// Load tile from DRAM/SRAM at `ptr` (count elements) into tile register dt.
// dt    : i64 — destination tile number (0–7); explicit to avoid tile allocator.
// count : i64 — element count (changed from i32 to match native i64 type).
// Maps to T_LOAD (cinst.rd = base, cinst.count = count, cinst.dt = dt).
def int_npu_load : DefaultAttrsIntrinsic<
    [llvm_i64_ty],  // tile token (same value as dt)
    [llvm_i64_ty,   // dt — destination tile (0–7)
     llvm_ptr_ty,   // source address
     llvm_i64_ty],  // element count
    [IntrReadMem, IntrArgMemOnly]
>;

// Load weight tile (transposed layout for inner-product access). Maps to T_LOADW.
def int_npu_loadw : DefaultAttrsIntrinsic<
    [llvm_i64_ty],
    [llvm_i64_ty,   // dt — destination tile (0–7)
     llvm_ptr_ty,
     llvm_i64_ty],  // element count (i64)
    [IntrReadMem, IntrArgMemOnly]
>;

// Store tile to memory. Maps to T_STORE.
// count : i64 — element count (changed from i32).
def int_npu_store : DefaultAttrsIntrinsic<
    [],
    [llvm_ptr_ty,   // destination address
     llvm_i64_ty,   // tile token (= tile register number)
     llvm_i64_ty],  // element count
    [IntrWriteMem, IntrArgMemOnly]
>;

// DMA: async DRAM → on-chip SRAM transfer.
// dst   : ptr — scratchpad address (SCRATCH_BASE | offset)
// src   : ptr — DRAM address
// bytes : i64 — byte count (changed from i32 to match native i64 type).
// Maps to T_DMA_IN.
def int_npu_dma_in : DefaultAttrsIntrinsic<
    [],
    [llvm_ptr_ty,   // dst (scratchpad)
     llvm_ptr_ty,   // src (DRAM)
     llvm_i64_ty],  // byte count
    [IntrHasSideEffects]
>;

// DMA: async on-chip SRAM → DRAM transfer. Maps to T_DMA_OUT.
// bytes : i64 — byte count (changed from i32).
def int_npu_dma_out : DefaultAttrsIntrinsic<
    [],
    [llvm_ptr_ty,   // dst (DRAM)
     llvm_ptr_ty,   // src (scratchpad)
     llvm_i64_ty],  // byte count
    [IntrHasSideEffects]
>;

// Broadcast a scalar to all elements of tile dt. Maps to T_BCAST.
// dt : i64 — destination tile number (0–7); explicit for the same reason as load/matmul.
def int_npu_bcast : DefaultAttrsIntrinsic<
    [llvm_i64_ty],  // tile token (same value as dt)
    [llvm_i64_ty,   // dt — destination tile (0–7)
     llvm_float_ty],// scalar value to broadcast
    [IntrHasSideEffects]
>;

// ─── Layer 2 : Tensor engine – activation (Section 7) ────────────────────────

// ReLU applied element-wise to a tile. Maps to T_RELU.
def int_npu_relu : DefaultAttrsIntrinsic<
    [llvm_i64_ty],
    [llvm_i64_ty],  // input tile
    [IntrHasSideEffects]
>;

// Scale all elements of a tile by a scalar. Maps to T_SCALE.
def int_npu_scale : DefaultAttrsIntrinsic<
    [llvm_i64_ty],
    [llvm_i64_ty,      // input tile
     llvm_float_ty],   // scale factor
    [IntrHasSideEffects]
>;

// ─── Layer 2 : Tensor engine – synchronization (Section 8) ───────────────────

// Memory fence: ensure all preceding tensor ops are globally visible.
// Maps to T_FENCE.
def int_npu_fence : DefaultAttrsIntrinsic<
    [],
    [],
    [IntrHasSideEffects]
>;

// Stall until the tensor engine is idle. Maps to T_WAIT.
def int_npu_wait : DefaultAttrsIntrinsic<
    [],
    [],
    [IntrHasSideEffects]
>;

// Stall all cores until every core reaches this point. Maps to T_BARRIER.
def int_npu_barrier : DefaultAttrsIntrinsic<
    [],
    [],
    [IntrHasSideEffects]
>;

// ─── Layer 3 : Vector core – setup (Section 9) ───────────────────────────────

// Set the active vector length for subsequent vector operations.
// length : i64 — desired lane count (clamped to [1, VLEN_MAX=256]).
// Returns the actual VL applied as i64.
// Changed from i32 to i64 to match the native type and avoid type promotion.
// Maps to V_VSETVL.
def int_npu_vsetvl : DefaultAttrsIntrinsic<
    [llvm_i64_ty],  // actual VL set
    [llvm_i64_ty],  // requested VL
    [IntrHasSideEffects]
>;

// Load a vector register from memory at base + stride*lane.
// stride : i64 — byte stride between elements (0 = contiguous, stride=4).
// Changed from i32 to i64 to match the native type.
// Maps to V_VLOAD.
def int_npu_vload : DefaultAttrsIntrinsic<
    [llvm_v16f32_ty],
    [llvm_ptr_ty,   // base address
     llvm_i64_ty],  // byte stride (0 = contiguous)
    [IntrReadMem, IntrArgMemOnly]
>;

// Store a vector register to memory. Maps to V_VSTORE.
// stride : i64 — byte stride (changed from i32).
def int_npu_vstore : DefaultAttrsIntrinsic<
    [],
    [llvm_ptr_ty,     // base address
     llvm_v16f32_ty,  // vector to store
     llvm_i64_ty],    // byte stride
    [IntrWriteMem, IntrArgMemOnly]
>;

// Gather: vd[i] = mem[base + index_vec[i]]. Maps to V_VGATHER.
def int_npu_vgather : DefaultAttrsIntrinsic<
    [llvm_v16f32_ty],
    [llvm_ptr_ty,     // base address
     llvm_v16f32_ty], // index vector (integer indices treated as byte offsets)
    [IntrReadMem, IntrArgMemOnly]
>;

// Scatter: mem[base + index_vec[i]] = vs[i]. Maps to V_VSCATTER.
def int_npu_vscatter : DefaultAttrsIntrinsic<
    [],
    [llvm_ptr_ty,     // base address
     llvm_v16f32_ty,  // index vector
     llvm_v16f32_ty], // data vector
    [IntrWriteMem, IntrArgMemOnly]
>;

// ─── Layer 3 : Vector core – transcendentals (Section 10) ────────────────────
// These ops exist because plain LLVM vector IR has no transcendental opcodes.
// Elementwise math (add/sub/mul/div/min/max) uses native <16 x float> IR.

// e^x element-wise. Maps to V_VEXP.
def int_npu_vexp : DefaultAttrsIntrinsic<
    [llvm_v16f32_ty],
    [llvm_v16f32_ty],
    [IntrNoMem]
>;

// ln(x) element-wise. Maps to V_VLOG.
def int_npu_vlog : DefaultAttrsIntrinsic<
    [llvm_v16f32_ty],
    [llvm_v16f32_ty],
    [IntrNoMem]
>;

// 1/sqrt(x) element-wise. Maps to V_VRSQRT.
def int_npu_vrsqrt : DefaultAttrsIntrinsic<
    [llvm_v16f32_ty],
    [llvm_v16f32_ty],
    [IntrNoMem]
>;

// sqrt(x) element-wise. Maps to V_VSQRT.
def int_npu_vsqrt : DefaultAttrsIntrinsic<
    [llvm_v16f32_ty],
    [llvm_v16f32_ty],
    [IntrNoMem]
>;

// 1/x element-wise. Maps to V_VRECIP.
def int_npu_vrecip : DefaultAttrsIntrinsic<
    [llvm_v16f32_ty],
    [llvm_v16f32_ty],
    [IntrNoMem]
>;

// ─── Layer 3 : Vector core – reductions (Section 11) ─────────────────────────

// Sum all active lanes. Maps to V_VREDSUM.
def int_npu_vredsum : DefaultAttrsIntrinsic<
    [llvm_float_ty],
    [llvm_v16f32_ty],
    [IntrNoMem]
>;

// Maximum across active lanes. Maps to V_VREDMAX.
def int_npu_vredmax : DefaultAttrsIntrinsic<
    [llvm_float_ty],
    [llvm_v16f32_ty],
    [IntrNoMem]
>;

// Minimum across active lanes. Maps to V_VREDMIN.
def int_npu_vredmin : DefaultAttrsIntrinsic<
    [llvm_float_ty],
    [llvm_v16f32_ty],
    [IntrNoMem]
>;

// AND reduction across active lanes. Maps to V_VREDAND.
def int_npu_vredand : DefaultAttrsIntrinsic<
    [llvm_float_ty],
    [llvm_v16f32_ty],
    [IntrNoMem]
>;

// OR reduction across active lanes. Maps to V_VREDOR.
def int_npu_vredor : DefaultAttrsIntrinsic<
    [llvm_float_ty],
    [llvm_v16f32_ty],
    [IntrNoMem]
>;

} // TargetPrefix = "npu"
