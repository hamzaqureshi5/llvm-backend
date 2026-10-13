# NPU LLVM Backend

LLVM backend for the NPU ISA defined in `npu-simulator/`. Targets the triple
`npu-unknown-unknown`.

## Status

All three ISA layers compile LLVM IR → NPU assembly via `llc`:

| Layer | Engine | Status |
|-------|--------|--------|
| 1 | Control core (C_KLAUNCH / C_KWAIT) | ✅ |
| 2 | Tensor engine (T_CFG_SHAPE, T_ACC_CLR, T_MMA, T_MATMUL, T_DOT, T_LOAD, T_STORE, T_RELU, T_SCALE, T_BCAST, T_FENCE, T_BARRIER) | ✅ |
| 3 | Vector core (V_VSETVL, V_VLOAD, V_VSTORE, V_VADD, V_VEXP, V_VREDSUM, V_VREDMAX, …) | ✅ |

Known gaps (not yet implemented):

- ELF relocations / binary encoding — the encoder/decoder in `npu_asm.h` is not yet wired into the MC layer
- MLIR dialect — no MLIR NPU dialect exists yet

## Files

| File | Purpose |
|------|---------|
| `NPUInstrInfo.td` | Base instruction defs (control-core scalar ISA) |
| `NPUTensorInstrInfo.td` | Layer 2 tensor-engine instruction defs |
| `NPUVectorInstrInfo.td` | Layer 3 vector-core instruction defs |
| `NPUISelLowering.cpp` | `LowerIntrinsicCall` — converts `@llvm.npu.*` INTRINSIC_W_CHAIN nodes to NPUISD:: nodes |
| `NPUISelDAGToDAG.cpp` | `Select` — converts NPUISD:: nodes to real machine instructions |
| `NPUMCInstLower.cpp` | `Lower` — converts `MachineInstr` → `MCInst` (handles `MO_FPImmediate`) |
| `MCTargetDesc/NPUInstPrinter.cpp` | Prints MCInst operands (handles tile `t<n>`, DFPImm) |
| `llvm/include/llvm/IR/IntrinsicsNPU.td` | `@llvm.npu.*` intrinsic declarations |

## Key design decisions

### Tile registers as i64 SSA constants

Tile registers (0–7) are encoded as `i64` constant values in IR. This avoids
needing a tile register allocator. `getTileConst` in `NPUISelLowering.cpp`
extracts the constant from an INTRINSIC_W_CHAIN SSA chain, following through
intermediate nodes recursively.

### IntrHasSideEffects for tile-mutating ops

Tile ops (ACC_CLR, MMA, etc.) are marked `IntrHasSideEffects` so LLVM creates
`INTRINSIC_W_CHAIN` nodes instead of `INTRINSIC_WO_CHAIN`. This is required for
`setOperationAction(ISD::INTRINSIC_W_CHAIN, MVT::i64, Custom)` interception in
`LowerIntrinsicCall`.

### BCAST reads from GPR, not fimm

`T_BCAST dt, xS` — broadcasts float bits stored in integer register `xS` to all tile
elements. Unlike `T_SCALE` which uses the `fimm` field, the simulator reads
`float s = (float)getx(n, I->rs1)` — the value comes from the integer register file.
The f32 bit pattern is materialized into a GPR via `LI` before `BCAST` fires.

### VSETVL uses GPR (i64), not SReg (f32)

`V_VSETVL xD, xS` — VL is an integer count. Both request and result live in GPRs
(i64), not the vector engine's scalar register file (SReg, f32).

### DOT uses CF_T3 (3 operands, same as MMA/MATMUL)

`int_npu_dot(dt, A, B)` — `dt` is both destination and accumulator. This matches
`{"DOT", T_DOT, CF_T3}` in `npu_asm.c`. The accumulator is NOT a separate operand.

### Scale factor encoding

`T_SCALE`'s float factor is encoded as `TargetConstant:i64` (double IEEE 754 bit
pattern) through the NPUISD chain to avoid f64/f32 type-legalization issues.
`NPUISelDAGToDAG::Select` reconstructs `TargetConstantFP:f64` from the bits
when emitting the machine node (machine nodes skip type legalization).

## Running tests

```sh
# Run all NPU lit tests:
build/bin/llvm-lit -v llvm/test/CodeGen/NPU/

# Compile a single .ll file to NPU assembly:
build/bin/llc -mtriple=npu-unknown-unknown -o - <file.ll>

# Run the full Layer 2+3 pipeline manually:
build/bin/llc -mtriple=npu-unknown-unknown -o - /tmp/test_full_pipeline.ll
```

## Intrinsic → instruction mapping

```
@llvm.npu.cfg.shape(M, K, N)           → CFG.SHAPE M, K, N
@llvm.npu.acc.clr(dt)                  → ACC.CLR t<dt>
@llvm.npu.mma(acc, A, B)               → MMA t<acc>, t<A>, t<B>
@llvm.npu.matmul(dt, A, B)             → MATMUL t<dt>, t<A>, t<B>
@llvm.npu.dot(dt, A, B)                → DOT t<dt>, t<A>, t<B>
@llvm.npu.load(dt, ptr, count)         → LOAD t<dt>, count(xBase)
@llvm.npu.store(ptr, tile, count)      → STORE t<tile>, count(xBase)
@llvm.npu.relu(tile)                   → RELU t<tile>, t<tile>
@llvm.npu.scale(tile, factor)          → SCALE t<tile>, t<tile>, <factor>
@llvm.npu.fence()                      → FENCE
@llvm.npu.barrier()                    → BARRIER
@llvm.npu.vload(ptr, stride)           → VLOAD v<d>, stride(xBase)
@llvm.npu.vstore(ptr, vec, stride)     → VSTORE stride(xBase), v<s>
@llvm.npu.vexp(v)                      → VEXP v<d>, v<s>
@llvm.npu.vredsum(v)                   → VREDSUM s<d>, v<s>
```
