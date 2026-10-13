# Plan: Minimal `npu` LLVM backend — control-core scalar codegen (text asm)

## Context

This repo is an LLVM fork for NPU compiler work. There is an NPU functional simulator
(`npu-simulator/`) with a defined three-layer ISA (control core / tensor engine / vector core)
and a design sketch (`npu-simulator/llvm_lowering.md`) for lowering LLVM IR to it, but **no LLVM
backend exists yet**. The goal is the first concrete step toward that: a registered experimental
LLVM target named `npu` that compiles simple **scalar** LLVM IR into **control-core text
assembly** matching the simulator's `npu_asm` syntax, so output round-trips through the existing
`npu_asm_control` assembler.

**Scope (decided with user):** control-core scalar layer only; **text assembly output only**
(AsmPrinter/InstPrinter, no MC code emitter / object files); **no AsmParser or Disassembler**.
Tensor-engine and vector-core layers are explicitly out of scope for this milestone.

Target outcome:
```
llc -mtriple=npu add.ll -o add.s    # i64 add/mul/load/store/branch/call/ret -> NPU control-core asm
```

## Verified simulator facts (ground truth — confirmed by reading npu.c / npu_asm.c / npu.h)

- **32 registers `x0..x31`, each 64-bit (`int64_t`).** `x0` is **hardwired zero** (`npu.c` `getx`/`setx`):
  reserve it, non-allocatable; use as canonical zero, move (`OR xd,xa,x0`), uncond jump (`JAL x0,label`).
- **Control-core text syntax** the InstPrinter must emit exactly (from `npu_asm.c` CTAB / `npu_disasm_cinst`):
  - RRR `OP xd, xa, xb` — ADD SUB MUL AND OR XOR SLL SRL SLT
  - RRI `ADDI xd, xa, imm` · RI `LI xd, imm` (LI materializes a full int64 in **one** op)
  - LOAD `LW xd, off(xa)` (also LH, LB) · STORE `SW xs, off(xa)` (also SH, SB) — **value register FIRST**
  - BR2 `BEQ xa, xb, label` (also BNE, BLT, BGE) — label resolves to an **instruction index**
  - `JAL xd, label` · `JALR xd, xa, imm` · `KWAIT` · `HALT` (no operands)
- **Missing instructions** (must expand or diagnose, never emit): no SRA (SRL is logical only), no SLTU,
  no DIV/REM, no multiply-high, no sign/zero-extend op, **no 64-bit load/store** (`LW`=32-bit sign-extending,
  `SW`=stores low 32 bits). Unknown mnemonics are rejected by the simulator's assembler.
- **Branch/jump targets are symbolic labels → instruction indices.** Standard MBB-label emission works;
  text-only means **no MC fixups / PC-relative math at all**.
- **One flat program terminated by `HALT`; no documented stack/ABI.** Calling convention + frame are
  invented for LLVM's benefit (model on RISC-V). Multi-function execution on the sim is NOT a goal;
  emitting plausible single-function asm is the bar.
- Comment char `#` is accepted by the sim (so is `;`); avoid `@`-leading tokens. The sim's assembler
  rejects `.`-directives — filter them only for the round-trip check, keep them for lit.

## Data layout decision

Single i64 GPR class; **64-bit pointers** (simplest legalization — one legal integer type; the 32-bit
address space is enforced physically by the sim at the memory boundary). Data layout:
`e-m:e-p:64:64-i64:64-n64-S64`. Adjustable to `p:32:32` later; note this in the TargetMachine.

## Reference backend

Model every file on **`llvm/lib/Target/Lanai`** (cleanest minimal modern RISC target); Sparc/XCore secondary.

## Registration touchpoints (outside the target dir)

- `llvm/CMakeLists.txt` — add `NPU` to `LLVM_ALL_EXPERIMENTAL_TARGETS` (~line 601-606).
- `llvm/include/llvm/TargetParser/Triple.h` — add `npu` to `ArchType` enum; insert **before `ve`** so
  `LastArchType` need not change.
- `llvm/lib/TargetParser/Triple.cpp` — mirror the `lanai` arms: `getArchTypeName`,
  `getArchTypeForLLVMName`, `parseArch` (`.Case("npu", npu)`), and the object-format/pointer-width arms
  where the `default:` would be wrong (ELF, 64-bit).
- `llvm/include/llvm/Config/*.def` — generated at configure time; **no manual edit**.
- Build: add `-DLLVM_EXPERIMENTAL_TARGETS_TO_BUILD="NPU"` (keep `LLVM_TARGETS_TO_BUILD="X86"`), e.g. via
  `build-llvm.sh`.

## TableGen `-gen-*` set (text-only) in `NPU/CMakeLists.txt`

`-gen-asm-writer`, `-gen-callingconv`, `-gen-dag-isel`, `-gen-instr-info`, `-gen-register-info`,
`-gen-sd-node-info` (mandatory even with zero custom SD nodes — ship a trivial `NPUSelectionDAGInfo`),
`-gen-subtarget`. **Drop** `-gen-asm-matcher`, `-gen-disassembler`, `-gen-emitter`.

## Phased implementation (each phase independently buildable/testable)

**Phase A — Registration; `llc --version` lists `npu`.**
Triple.h/.cpp + llvm/CMakeLists.txt edits. `NPU/TargetInfo/NPUTargetInfo.{h,cpp}` (← Lanai) with
`getTheNPUTarget()` + `LLVMInitializeNPUTargetInfo`. `NPU/TargetInfo/CMakeLists.txt`. `NPU/CMakeLists.txt`
with `add_llvm_component_group(NPU)` + `add_subdirectory(TargetInfo)`. Test: `llc --version | grep npu`.

**Phase B — Register file + MC layer constructs.**
TableGen: `NPU.td`, `NPURegisterInfo.td` (32 X-regs, i64, GPR class, X0 reserved/last), `NPUInstrFormats.td`
(lightweight classes, **omit `Inst{}` bit encodings**), seed `NPUInstrInfo.td` (LI/ADDI/HALT). C++:
`MCTargetDesc/NPUMCTargetDesc.{h,cpp}` (keep MCAsmInfo/MCInstrInfo/MCRegInfo/MCSubtargetInfo/MCInstPrinter;
drop code-emitter/asm-backend/ELF-streamer), `NPUMCAsmInfo.{h,cpp}` (`CommentString="#"`, ELF base),
`NPUInstPrinter.{h,cpp}` (printOperand, printMemOperand `off(xa)`), `MCTargetDesc/CMakeLists.txt`.

**Phase C — Compile `ret i64 42` → `LI x10, 42` / return / `HALT`.**
`NPUCallingConv.td` (RISC-V-style invented: args X10-X17, return X10/X11, RA=X1, SP=X2, CSR=X1+X18-X27).
C++ (← Lanai): `NPUTargetMachine`, `NPUSubtarget`, `NPURegisterInfo` (reserve X0/X1/X2, eliminateFrameIndex
via ADDI), `NPUFrameLowering` (empty when frameSize==0; hasFP=false), `NPUMachineFunctionInfo`,
`NPUISelLowering` (addRegisterClass i64/GPR; lower FormalArguments/Return/Call; `NPUISD` nodes in C++),
`NPUISelDAGToDAG` (Constant→LI, FrameIndex→ADDI, SelectAddr base+imm), `NPUMCInstLower` (MBB→symbol),
`NPUAsmPrinter` (emit instrs; append a trailing `HALT` — documented milestone hack), `NPUSelectionDAGInfo`
(trivial). Full `add_llvm_target(NPUCodeGen …)` in `NPU/CMakeLists.txt`.

**Phase D — Arithmetic / memory / branch patterns** in `NPUInstrInfo.td`:
- RRR patterns: ADD(add) SUB(sub) MUL(mul) AND(and) OR(or) XOR(xor) SLL(shl) SRL(srl) SLT(setlt);
  ADDI(add imm); LI(imm64, one op).
- Mem: `LW xd, off(xa)` for `sextloadi32`, `SW xs, off(xa)` for `truncstorei32` (**value-first AsmString**),
  LB/LH/SB/SH for i8/i16. i64 plain load/store: diagnose unsupported (require i32 test inputs) — never emit `LD`.
- Branches: fuse icmp+br → BEQ/BNE/BLT/BGE with operand swaps (sgt→BLT swapped, sle→BGE swapped);
  uncond `br` → `JAL x0, label`. Calls: `JAL x1, callee` / indirect `JALR x0, rs, 0`; RET pseudo → `JALR x0, x1, 0`.
- `NPUInstrInfo.cpp`: copyPhysReg via `OR xd,xa,x0`; load/store via LW/SW; analyzeBranch; no delay slots.

**Phase E — Unsupported-op handling** in `NPUISelLowering` ctor:
- SDIV/UDIV/SREM/UREM, MULHS/MULHU → `Custom` + `report_fatal_error("NPU: <op> unsupported")` (honest;
  flip to libcall later if needed).
- SRA → custom-lower or diagnose. Zero-extend → `AND` with LI-materialized mask; neg=`SUB xd,x0,xa`;
  not=`XOR xd,xa,-1`. Unsigned compares/branches (no SLTU) → sign-bias trick or diagnose.
- SELECT/SELECT_CC → custom via BR_CC control flow (no conditional move).
- ROTL/ROTR/CTLZ/CTTZ/CTPOP/BSWAP → Expand. GlobalAddress/BlockAddress/etc. → `LI xd, sym`. Discourage jump tables.

## Files to create

`llvm/lib/Target/NPU/`: `NPU.td`, `NPURegisterInfo.td`, `NPUInstrFormats.td`, `NPUInstrInfo.td`,
`NPUCallingConv.td`, `NPU.h`, `NPUTargetMachine.{h,cpp}`, `NPUSubtarget.{h,cpp}`, `NPUInstrInfo.{h,cpp}`,
`NPURegisterInfo.{h,cpp}`, `NPUISelLowering.{h,cpp}`, `NPUISelDAGToDAG.{h,cpp}`, `NPUFrameLowering.{h,cpp}`,
`NPUMachineFunctionInfo.{h,cpp}`, `NPUMCInstLower.{h,cpp}`, `NPUAsmPrinter.cpp`, `NPUSelectionDAGInfo.{h,cpp}`,
`CMakeLists.txt`; `MCTargetDesc/` (`NPUMCTargetDesc.{h,cpp}`, `NPUMCAsmInfo.{h,cpp}`, `NPUInstPrinter.{h,cpp}`,
`CMakeLists.txt`); `TargetInfo/` (`NPUTargetInfo.{h,cpp}`, `CMakeLists.txt`).
**Dropped vs Lanai:** AsmParser/, Disassembler/, MCCodeEmitter, AsmBackend, ELFObjectWriter, FixupKinds,
DelaySlotFiller, MemAluCombiner, TargetObjectFile, all `Inst{}` encodings, CCR/status reg class.

## Verification

New `llvm/test/CodeGen/NPU/` with `lit.local.cfg` gating on `'NPU' in config.root.targets`. Tests
(`RUN: llc -mtriple=npu < %s | FileCheck %s`): `ret-imm.ll` (LI + return + HALT), `arith.ll` (RRR + ADDI),
`icmp-br.ll` (BEQ/BNE/BLT/BGE + swaps + `JAL x0,`), `loadstore.ll` (LW/SW value-first, LB/LH/SB/SH),
`call.ll` (JAL x1 / JALR), `unsupported.ll` (sdiv/udiv → `not llc` / diagnostic).

**Simulator round-trip (authority on syntax):** `llc -mtriple=npu arith.ll -o arith.s`, strip directives
(`grep -vE '^\s*\.'`), feed to `npu_asm_control(...)` via a small harness in `npu-simulator/`; a return ≥0
with empty `err` proves the text matches CTAB exactly (mnemonics, operand order, `off(xN)`, label refs).

## Pitfalls & non-goals

Pitfalls: x0 is zero (reserve); store is value-first; no 64-bit load/store; no SRA/SLTU/DIV/REM/MULH;
branch labels are indices (no byte offsets); `-gen-sd-node-info` mandatory; insert `npu` before `ve` in
the enum; trailing-HALT is a milestone hack. Non-goals: object files / AsmParser / Disassembler /
relocations; tensor & vector layers; KLAUNCH/kernels; floating point; runnable multi-function ABI;
64-bit memory ops; scheduling fidelity.
