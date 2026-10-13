# Layer 1 — The Control Core

The control core is the scalar CPU of the NPU. It runs a sequential program
(`cinst[]`) until it hits `C_HALT`. Every other operation on the chip —
configuring the tensor engine, moving data, launching vector kernels — is
orchestrated from here.

The entire interpreter is `npu_run()` in `npu.c`: a `while(pc < len)` loop
that fetches the current `cinst`, dispatches on `op`, then increments `pc` —
or jumps directly for branches.

---

## 1.1 Binary Encoding — The 32-Byte Word

Every instruction — Layer 1 scalar, Layer 2 tensor, Layer 3 vector — occupies
exactly **32 bytes** on disk.

```
byte  0     : stream type  (0 = cinst / Layer 1+2,  1 = vinst / Layer 3)
byte  1     : opcode       (cop_t enum value from npu.h)
byte  2     : rd           (destination register, x0–x31)
byte  3     : rs1          (source register 1)
byte  4     : rs2          (source register 2)
bytes 5–11  : tensor fields (dt, a, b — tile registers; ignored by Layer 1)
bytes 12–15 : count        (u32 LE — element/byte count for load/store/DMA)
bytes 16–23 : imm          (i64 LE — or fimm / IEEE 754 double for T_SCALE)
bytes 24–31 : padding
```

| Field | Bytes | Description |
|---|---|---|
| `stream` | 0 | `0` = cinst (L1+L2), `1` = vinst (L3) |
| `opcode` | 1 | `cop_t` or `vop_t` enum value |
| `rd`, `rs1`, `rs2` | 2–4 | Control-core register indices (x0–x31) |
| `tensor` | 5–11 | `dt`, `a`, `b` (tile 0–7); ignored by Layer 1 ops |
| `count` | 12–15 | Element or byte count (u32 LE) |
| `imm` | 16–23 | 64-bit signed immediate, or `fimm` (double) for `T_SCALE` |
| `padding` | 24–31 | Reserved |

---

## 1.2 Register File and Calling Convention

The control core has **32 general-purpose 64-bit registers**, `x0`–`x31`.
`x0` is hardwired to zero — reads always return 0, writes are discarded.

| Registers | Role | Allocation priority |
|---|---|---|
| `x10`–`x17` | Arguments / return values | First (caller-saved) |
| `x5`–`x9`, `x28`–`x31` | Caller-saved temporaries | Second |
| `x18`–`x27` | Callee-saved | Third |
| `x1` | Return address | Reserved |
| `x2` | Stack pointer | Reserved |
| `x0` | Hardwired zero | Never allocated |

**Calling convention:**

| Type | Argument registers | Return register |
|---|---|---|
| `i64` | `x10`–`x17` | `x10` |
| `<16 x float>` | `v0`–`v7` | `v0` |
| `f32` (reductions) | `s2`–`s9` | `s2` |

Sub-word integers (`i8`, `i16`, `i32`) are promoted to `i64` before passing.
Callee-saved registers: `x1`, `x18`–`x27`.

---

## 1.3 Instruction Set

### Scalar arithmetic

```
ADD   rd, rs1, rs2    rd = rs1 + rs2
SUB   rd, rs1, rs2    rd = rs1 - rs2
MUL   rd, rs1, rs2    rd = rs1 * rs2
AND   rd, rs1, rs2
OR    rd, rs1, rs2
XOR   rd, rs1, rs2
SLL   rd, rs1, rs2    logical left shift
SRL   rd, rs1, rs2    logical right shift
SLT   rd, rs1, rs2    rd = (signed) rs1 < rs2 ? 1 : 0
ADDI  rd, rs1, imm    rd = rs1 + imm
LI    rd, imm         rd = imm  (full 64-bit immediate)
```

### Memory

Only 32-bit and smaller. All loads sign-extend to 64 bits.
There is **no 64-bit load or store**.

```
LW  rd, imm(rs1)    load 32-bit word, sign-extend → rd
LH  rd, imm(rs1)    load 16-bit, sign-extend → rd
LB  rd, imm(rs1)    load 8-bit, sign-extend → rd
SW  rs, imm(rs1)    store low 32 bits of rs
SH  rs, imm(rs1)
SB  rs, imm(rs1)
```

Address: `use_base=1` → `x[rs1] + imm`, `use_base=0` → `imm` (absolute).

### Control flow

Branch targets are **absolute instruction indices**, not byte offsets.

```
BEQ   rs1, rs2, target    branch if equal
BNE   rs1, rs2, target
BLT   rs1, rs2, target    signed less-than
BGE   rs1, rs2, target    signed greater-or-equal
JAL   rd, target          x[rd] = pc+1; pc = target
JALR  rd, rs1, imm        x[rd] = pc+1; pc = x[rs1]+imm
HALT                       terminate npu_run()
```

No unsigned branches — LLVM synthesizes them with XOR (see §1.5).

### Kernel launch

```
KLAUNCH  rs1, grid    launch grid instances of kernel[x[rs1]]
                      grid is a compile-time constant immediate
KWAIT                 wait for all dispatched kernels to finish
```

---

## 1.4 LLVM Lowering Pipeline

```
LLVM IR
  │  SelectionDAGBuilder
  ▼
SelectionDAG  (ISD::ADD, ISD::BR_CC, ISD::RET …)
  │  LegalizeTypes → sub-word ints promoted to i64
  │  Legalize      → SELECT/SDIV/ROTR/… expanded
  ▼
Legal DAG
  │  NPUTargetLowering::LowerOperation
  │    @llvm.npu.* intrinsics → NPUISD:: nodes
  │  NPUISelDAGToDAG::Select
  │    NPUISD::KLAUNCH → NPU::KLAUNCH
  │    ISD::BR_CC      → BEQ/BNE/BLT/BGE (+ XOR for unsigned)
  │    ISD::FrameIndex → ADDI x2, offset
  │    everything else → SelectCode() → TableGen patterns
  ▼
Machine Instructions  (NPU::ADD, NPU::LW, NPU::BEQ …)
  │  NPUAsmPrinter + NPUMCInstLower
  ▼
MCInst stream
  ├─ NPUInstPrinter   → text assembly (.s)
  └─ NPUMCCodeEmitter → 32-byte ELF object (.o)
```

| Operation | Action | Reason |
|---|---|---|
| `ISD::BR_CC` (i64) | Legal | Used directly in `NPUISelDAGToDAG` |
| `ISD::SELECT` / `SELECT_CC` | Expand | No CMOV |
| `ISD::SDIV` / `UDIV` / `SREM` / `UREM` | Expand | No divide instruction |
| `ISD::FADD/…` (f32) | Expand | Control core has no FPU |

---

## 1.5 Conditional Branch Synthesis

The ISA has four signed conditional branches only. LLVM maps all eight
conditions to these four.

**Signed:**

| LLVM condition | Emitted | Transform |
|---|---|---|
| `SETEQ` | `BEQ a, b, target` | direct |
| `SETNE` | `BNE a, b, target` | direct |
| `SETLT` | `BLT a, b, target` | direct |
| `SETGE` | `BGE a, b, target` | direct |
| `SETGT` | `BLT b, a, target` | swap operands |
| `SETLE` | `BGE b, a, target` | swap operands |

**Unsigned** — synthesized by flipping bit 63 of both operands:

```
; a <u b  ≡  (a ^ MSB) <s (b ^ MSB)

LI   x5, 0x8000000000000000
XOR  tmp_a, a, x5
XOR  tmp_b, b, x5
BLT  tmp_a, tmp_b, target
```

---

## 1.6 KLAUNCH — Full Lowering Chain

`@llvm.npu.klaunch(i32 %kid, i32 %grid)` passes through three stages:

**Stage 1 — LowerIntrinsicCall** (`NPUISelLowering.cpp`)
Converts to `NPUISD::KLAUNCH` DAG node. Kid may be a runtime GPR; grid
must be a compile-time constant.

**Stage 2 — NPUISelDAGToDAG::Select**
Validates grid is a `ConstantSDNode`, emits `NPU::KLAUNCH` with kid in
a GPR and grid as an immediate. Fatal error if grid is not constant.

**Stage 3 — Simulator**
`C_KLAUNCH` runs `grid` instances sequentially. Cycle model then replaces
the serial sum with `max-over-cores` wall time.

---

## 1.7 Frame Index Lowering

Stack slots reference the frame via `ISD::FrameIndex` nodes. These lower to:

```
ADDI xN, x2, <offset>   (frame slot address into a register)
```

`NPUFrameLowering` resolves the actual SP offset after register allocation.
Loads/stores from that address use `LW`/`SW` with zero offset on the result.

---

## 1.8 Known Gaps

| IR operation | Symptom | Root cause |
|---|---|---|
| `load i64, ptr` | Cannot select | No 64-bit load — only `LW`/`LH`/`LB` |
| `ashr i64` | Cannot select | No `SRA` — only logical `SRL` |
| `select i1 %c, i64 %a, i64 %b` | Legalizer assert | `SELECT` and `SELECT_CC` both Expand, no CMOV |
| `sdiv` / `udiv` | Expand fails | No divide instruction, no libcall infrastructure |
| `f32` arithmetic on control core | Expand | No FPU on control core |
| Varargs (`...`) | Fatal error | Rejected in `LowerFormalArguments` |

---

## 1.9 Real Example — Array Sum

```
Input: int32_t arr[8] = { 3, 7, 2, 9, 1, 5, 4, 8 }
       stored at 0x0100

Register plan:
  x1 = current pointer  (walks from 0x0100 to 0x0120)
  x2 = end pointer      (0x0100 + 8×4 = 0x0120)
  x3 = accumulator
  x4 = current element

Program:
  [0]  LI    x1, 0x0100
  [1]  LI    x2, 0x0120
  [2]  LI    x3, 0
  [3]  LW    x4, 0(x1)      ← loop top
  [4]  ADD   x3, x3, x4
  [5]  ADDI  x1, x1, 4
  [6]  BLT   x1, x2, 3      if x1 < 0x0120, jump to [3]
  [7]  SW    x3, 0x0200      store result
  [8]  HALT

Result: mem[0x0200] = 39
```
