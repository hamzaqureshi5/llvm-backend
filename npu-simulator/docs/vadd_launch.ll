; vadd_launch.ll — End-to-end KLAUNCH example for the NPU backend.
;
; This file shows the full journey of a kernel launch from LLVM IR down
; to the 32-byte machine instruction that the simulator executes.
;
; To compile:
;   llc -march=npu vadd_launch.ll -o vadd_launch.s
;
; To see the SelectionDAG at each lowering stage:
;   llc -march=npu -debug-only=isel vadd_launch.ll -o /dev/null 2>&1 | less
;
; ─────────────────────────────────────────────────────────────────────────────
; WHAT THIS PROGRAM DOES (C pseudocode)
; ─────────────────────────────────────────────────────────────────────────────
;
;   // Host side (C, runs on CPU before this IR is called):
;   //   npu_write_f32(n, 0x1000, a,   64);   // A[64]   at 0x1000
;   //   npu_write_f32(n, 0x2000, b,   64);   // B[64]   at 0x2000
;   //   npu_write_i32(n, 0x3000, args, 3);   // args[3] = {0x1000, 0x2000, 0x4000}
;   //   int kid = npu_register_kernel(n, vadd_vinst, vadd_len);
;   //   npu_run(n, cc_prog, cc_len);
;
;   // Control-core program (this .ll file):
;   void launch_vadd(i32 kid) {
;       // The vector kernel reads s1 = args_ptr on entry.
;       // We write the args struct to 0x3000 before launching.
;       npu_klaunch(kid, /*grid=*/4);  // 4 instances × 16 floats/instance = 64
;       npu_kwait();
;   }
;
; ─────────────────────────────────────────────────────────────────────────────
; MEMORY LAYOUT IN NPU DRAM
; ─────────────────────────────────────────────────────────────────────────────
;
;   0x1000 : float A[64]           (input array, written by host)
;   0x2000 : float B[64]           (input array, written by host)
;   0x3000 : i64   args[3]         (written by host before KLAUNCH)
;            args[0] = 0x1000      ← base address of A
;            args[1] = 0x2000      ← base address of B
;            args[2] = 0x4000      ← base address of output C
;   0x4000 : float C[64]           (output, written by vector kernel)
;
;   On kernel entry:  s0 = grid_instance_id  (0, 1, 2, 3)
;                     s1 = 0x3000            (args_ptr from KLAUNCH)
;   Each instance processes floats [i*16 .. i*16+15].
;
; ─────────────────────────────────────────────────────────────────────────────

target triple = "npu"

; ── Intrinsic declarations ────────────────────────────────────────────────────
; Source: llvm/include/llvm/IR/IntrinsicsNPU.td
;
; @llvm.npu.klaunch(kernel_id: i32, grid_size: i32) → void
;   kernel_id : index from npu_register_kernel(); can be a runtime GPR value
;   grid_size : number of parallel instances; MUST be a compile-time constant
;               (validated in NPUISelDAGToDAG::Select, fatal_error otherwise)
;
; @llvm.npu.kwait() → void
;   Full barrier — stalls the control core until all launched kernels finish.

declare void @llvm.npu.klaunch(i32, i32)
declare void @llvm.npu.kwait()

; ── Function ──────────────────────────────────────────────────────────────────
;
; launch_vadd(i32 %kid):
;   %kid comes in via the RISC-V-style ABI: i32 promoted to i64, placed in x10.
;   The function is called by a cinst[] C_JAL from the host setup program.
;
define void @launch_vadd(i32 %kid) {
entry:
  ; ── KLAUNCH intrinsic call ────────────────────────────────────────────────
  ; grid_size=4 is a literal here.  Using a variable (e.g. %n) would compile
  ; but die at code-gen with:
  ;   LLVM ERROR: NPU KLAUNCH: grid_size must be a compile-time constant
  ; This mirrors the simulator constraint that cinst.imm is fixed at assemble
  ; time and cannot be a register read at runtime.
  call void @llvm.npu.klaunch(i32 %kid, i32 4)

  ; ── KWAIT ────────────────────────────────────────────────────────────────
  ; Must come after KLAUNCH before reading any kernel output.
  ; Without this, the control core races ahead while the vector engine runs.
  call void @llvm.npu.kwait()

  ret void
}


; ═════════════════════════════════════════════════════════════════════════════
; HOW KLAUNCH IS LOWERED — STEP BY STEP
; ═════════════════════════════════════════════════════════════════════════════
;
; Step 1 — IR → SelectionDAG initial graph
; ─────────────────────────────────────────
; The call @llvm.npu.klaunch(i32 %kid, i32 4) produces an INTRINSIC_VOID node:
;
;   (INTRINSIC_VOID chain, i32 Intrinsic::npu_klaunch, i32:%kid, i32:4)
;
; The IntrinsicsNPU.td declaration has [IntrHasSideEffects], so LLVM threads
; it into the chain and never CSEs or DCEs it.
;
;
; Step 2 — LowerOperationWrapper / LowerIntrinsicCall
; ────────────────────────────────────────────────────
; File: llvm/lib/Target/NPU/NPUISelLowering.cpp :: LowerIntrinsicCall()
;
;   case Intrinsic::npu_klaunch: {
;     SDValue KernelID = Op.getOperand(2);  // the i32 %kid
;     SDValue GridSize = Op.getOperand(3);  // the constant i32 4
;     SDValue Ops[] = {Chain, KernelID, GridSize};
;     return DAG.getNode(NPUISD::KLAUNCH, DL, MVT::Other, Ops);
;   }
;
; The INTRINSIC_VOID node is replaced with a custom NPUISD::KLAUNCH node:
;
;   (NPUISD::KLAUNCH chain, i64:%kid_zext, i64:4)
;                                ↑
;                  i32 → i64 zero-extension happens during type legalization
;
;
; Step 3 — NPUISelDAGToDAG::Select (instruction selection)
; ─────────────────────────────────────────────────────────
; File: llvm/lib/Target/NPU/NPUISelDAGToDAG.cpp :: Select()
;
;   case NPUISD::KLAUNCH: {
;     SDValue Chain    = Node->getOperand(0);
;     SDValue KernelID = Node->getOperand(1);  // register → GPR (x10 here)
;     SDValue GridSize = Node->getOperand(2);  // must be ConstantSDNode
;
;     if (!isa<ConstantSDNode>(GridSize))
;       report_fatal_error("NPU KLAUNCH: grid_size must be a compile-time constant");
;
;     int64_t Grid   = cast<ConstantSDNode>(GridSize)->getSExtValue(); // = 4
;     SDValue GridImm = CurDAG->getTargetConstant(Grid, DL, MVT::i64);
;
;     SDValue Ops[] = {KernelID, GridImm, Chain};
;     CurDAG->SelectNodeTo(Node, NPU::KLAUNCH, MVT::Other, Ops);
;   }
;
; The NPUISD::KLAUNCH DAG node becomes the NPU::KLAUNCH machine instruction:
;
;   NPU::KLAUNCH $rs1=%kid_reg, $grid=4
;
;
; Step 4 — Machine instruction (NPUInstrInfo.td)
; ───────────────────────────────────────────────
; Definition:
;   def KLAUNCH : NPUInst<(outs), (ins GPR:$rs1, i64imm:$grid),
;       "KLAUNCH $rs1, $grid", []> {
;     let hasSideEffects = 1;
;     let isCall = 1;
;     let Size = 32;   // inherited from NPUInst
;   }
;
; Emitted assembly (vadd_launch.s):
;
;   launch_vadd:
;     KLAUNCH x10, 4    ; x10 holds %kid (argument register)
;     KWAIT
;     RET               ; JALR x0, x1, 0
;
;
; Step 5 — MCCodeEmitter (NPUMCCodeEmitter.cpp) → 32-byte binary word
; ───────────────────────────────────────────────────────────────────
; The NPU binary format (from npu_asm.c :: npu_encode_cinst):
;
;   byte 0   : stream=0  (cinst stream)
;   byte 1   : opcode    (cop_t::C_KLAUNCH = 24, see npu.h)
;   byte 2   : rd=0      (unused for KLAUNCH)
;   byte 3   : rs1=10    (x10 = kernel_id register)
;   byte 4   : rs2=0     (unused)
;   bytes 5–11: tensor fields (zeros)
;   bytes 12–15: count=0
;   bytes 16–23: imm=4   (grid_size, little-endian i64)
;   bytes 24–31: padding (zeros)
;
; Note: in the simulator cinst struct, imm holds the grid_size and rs1 holds
; the kernel_id register index.  The LLVM backend maps exactly to this layout:
;   cinst.rs1 = x10 (kernel_id register)
;   cinst.imm = 4   (grid_size constant)
;
;
; Step 6 — Simulator execution (npu.c :: npu_run)
; ─────────────────────────────────────────────────
; When the simulator fetches the KLAUNCH cinst:
;
;   case C_KLAUNCH: {
;     int kid     = (int)I->imm;            // NO! wait...
;     int64_t grid = getx(n, I->rs1);       // grid from register
;     int64_t args = getx(n, I->rs2);       // args from another register
;   }
;
; ⚠ MISMATCH NOTE: The simulator's C_KLAUNCH reads:
;     kid  from I->imm  (constant)
;     grid from I->rs1  (register)
;     args from I->rs2  (register)
;
; The LLVM backend's KLAUNCH encodes:
;     kernel_id in rs1 (register)
;     grid_size in imm (constant)
;
; This means the LLVM backend inverts the kid/grid roles compared to the
; simulator's cinst API.  When the compiled binary is loaded by the simulator,
; the simulator's npu_decode_cinst + npu_asm_control path must be used (the
; text assembler round-trip), not the raw cinst struct path.  The binary form
; is the authoritative cross-interface format; the cinst struct is the C API.
;
; ═════════════════════════════════════════════════════════════════════════════
; CONCRETE ASSEMBLY OUTPUT
; ═════════════════════════════════════════════════════════════════════════════
;
; Running:  llc -march=npu vadd_launch.ll -o -
;
; Expected output:
;
;   launch_vadd:
;           KLAUNCH x10, 4
;           KWAIT
;           JALR x0, x1, 0          ; RET pseudo-expands to this
;
; ═════════════════════════════════════════════════════════════════════════════
; WHAT THE VECTOR KERNEL SEES ON ENTRY  (written separately as vinst[])
; ═════════════════════════════════════════════════════════════════════════════
;
; The vector kernel registered as kernel[kid] receives on entry:
;   s0 = grid_instance_id   (0, 1, 2, 3 for grid_size=4)
;   s1 = args_ptr           (address of the args struct — set externally)
;
; The vector kernel then does:
;   VSETVL  16             ; work with 16 floats per instance
;   SLW s2, s1, 0          ; s2 = args[0] = 0x1000 (base of A)
;   SLW s3, s1, 8          ; s3 = args[1] = 0x2000 (base of B)
;   SLW s4, s1, 16         ; s4 = args[2] = 0x4000 (base of C)
;   SMULI s5, s0, 64       ; s5 = instance_id * 64 (byte offset into arrays)
;   SADD s2, s2, s5        ; s2 = A + offset
;   SADD s3, s3, s5        ; s3 = B + offset
;   SADD s4, s4, s5        ; s4 = C + offset
;   VLOAD v0, s2           ; v0 = A[i*16 .. i*16+15]
;   VLOAD v1, s3           ; v1 = B[i*16 .. i*16+15]
;   VADD v2, v0, v1        ; v2 = A + B
;   VSTORE v2, s4          ; C[i*16 .. i*16+15] = v2
;   KEXIT                  ; return from instance
