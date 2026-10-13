# NPU Architecture — Documentation Index

This directory contains the reference documentation for the three-layer NPU
architecture and the LLVM backend that targets it.

---

## Documents

| File | Contents |
|---|---|
| [layer1_control_core.md](layer1_control_core.md) | Scalar CPU: registers, instruction set, LLVM lowering, branch synthesis, KLAUNCH, known gaps |
| [layer2_tensor_engine.md](layer2_tensor_engine.md) | Matrix accelerator: tile registers, MATMUL vs MMA, DMA staging, activations, worked examples |
| [layer3_vector_core.md](layer3_vector_core.md) | Vector/SIMD engine: register files, instruction set, multi-instance dispatch, softmax, predication |
| [execution_model.md](execution_model.md) | SIMD vs SIMT vs SPMD — which model each layer uses and why it matters |

---

## One-line summary of each layer

**Layer 1 — Control Core**
Scalar RISC-V-like CPU. Runs `cinst[]`. Orchestrates everything: loads data,
configures the tensor engine, launches vector kernels, waits for results.

**Layer 2 — Tensor Engine**
Matrix co-processor. Shares the `cinst[]` stream — no KLAUNCH needed. `t0`–`t7`
tile registers hold up to 64×64 floats each. One `MATMUL` instruction multiplies
two tiles; `MMA` accumulates across K-blocks.

**Layer 3 — Vector Core**
Fully programmable SIMD engine. Runs a separate `vinst[]` program dispatched by
`KLAUNCH`. Handles exp, log, softmax, reductions, gather/scatter — anything the
tensor engine cannot do. Grid of N instances; each instance is SIMD, grid is SPMD.

---

## ISA reference

- `npu-simulator/fully_prog_npu.md` — authoritative ISA specification
- `npu-simulator/llvm_lowering.md` — lowering design (now implemented; read as context)
- `llvm/include/llvm/IR/IntrinsicsNPU.td` — all `@llvm.npu.*` intrinsic signatures
