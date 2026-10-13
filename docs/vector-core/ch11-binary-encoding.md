# Chapter 11: The 32-Byte Instruction Word and Binary Encoding

Every NPU instruction — control, tensor, or vector — is exactly 32 bytes. This chapter
explains why, details the byte-level layout, and shows how the three representations (text,
struct, binary) relate to each other and to the LLVM backend that generates them.

---

## 11.1 Why 32 Bytes Per Instruction?

Most ISAs use a narrow fixed instruction width: RISC-V is 4 bytes, x86 averages 3–5, ARM64
is exactly 4. The NPU uses 32 bytes per instruction. This is a large word — eight times
bigger than a typical RISC ISA. The reasons are pragmatic given the fields that must coexist
in a single instruction:

### 11.1.1 Field Inventory

Consider what a single tensor engine instruction (`T_SCALE`) needs to encode:

| Field | Type | Size |
|-------|------|------|
| Stream type (cinst vs vinst) | uint8 | 1 byte |
| Opcode (`cop_t`) | uint8 | 1 byte |
| Destination tile register `dt` (0–7) | uint8 | 1 byte |
| Source tile register `a` (0–7) | uint8 | 1 byte |
| Float scale factor `fimm` | IEEE 754 double | **8 bytes** |
| Element count `count` | uint32 | 4 bytes |
| Padding for alignment | — | 16 bytes |

The double-precision float immediate (`fimm`) alone consumes 8 bytes. A 4-byte instruction
word cannot hold an 8-byte float plus opcode and operand fields simultaneously.

### 11.1.2 Other Sizing Pressures

**Tile dimensions M, K, N**: each up to `MAX_TILE=64`, needing 7 bits. Three together = 21
bits beyond the other fields already present in `T_MATMUL`.

**Element count `count`**: A 32-bit count for the load/store instructions that move tile data.
With 64×64 tiles at 4 bytes per float, moving a whole tile requires `count = 4096`.

**Branch target `imm`**: For `V_LOOP` or `C_JAL`, the target is an instruction index (not a
byte offset). A kernel with 256 instructions at 32 bytes each spans 8 KB. An instruction
index needs ~8 bits for small kernels but up to 64 bits for large programs stored in DRAM.
The ISA uses `int64_t imm` so future programs are not constrained.

**Two address spaces**: The 32-bit address range is split between DRAM (high bit = 0) and
SRAM (`SCRATCH_BASE = 0x80000000`, high bit = 1). An absolute address field must be 32 bits.

**Result**: after laying out all mandatory fields with no packing, 32 bytes is the natural
alignment boundary that accommodates every instruction variant without conditional field
reuse. The regularity makes decode trivial — every field is at the same byte position
regardless of opcode.

---

## 11.2 The 32-Byte Layout

### 11.2.1 Control/Tensor Instruction Word (`cinst`, stream type byte = 0)

```
Byte   0       1       2       3       4       5       6       7
     ┌───────┬───────┬───────┬───────┬───────┬───────┬───────┬───────┐
     │  0x00 │ opcode│  rd   │  rs1  │  rs2  │  dt   │   a   │   b   │
     └───────┴───────┴───────┴───────┴───────┴───────┴───────┴───────┘
      stream   cop_t   GPR     GPR     GPR    tile    tile    tile
       type   (0–49)  (0–31) (0–31)  (0–31)  (0–7)  (0–7)   (0–7)

Byte   8       9      10      11      12..15
     ┌───────┬───────┬───────┬───────┬───────────────────────────────┐
     │   M   │   K   │   N   │use_bas│         count (uint32, LE)    │
     └───────┴───────┴───────┴───────┴───────────────────────────────┘
      (0–64)  (0–64)  (0–64) e (0/1)     element/byte count

Byte  16..23                             24..31
     ┌───────────────────────────────────────────────────────────────┐
     │   imm (int64, LE) — OR — fimm (double IEEE 754 bits, LE)      │
     ├───────────────────────────────────────────────────────────────┤
     │                    padding (zeros)                            │
     └───────────────────────────────────────────────────────────────┘
      imm = branch target / kernel id / absolute address
      fimm (double) used only when cop_is_float(op) is true:
           T_SCALE, T_QUANT, T_DEQUANT
```

**Field summary for cinst:**

| Byte(s) | Field | Type | Used by |
|---------|-------|------|---------|
| 0 | stream type | uint8 = 0 | decoder (always 0 for cinst) |
| 1 | opcode | `cop_t` uint8 | all |
| 2 | `rd` | uint8 (0–31) | arithmetic, load, JAL, STATUS |
| 3 | `rs1` | uint8 (0–31) | arithmetic, load/store, branches, KLAUNCH |
| 4 | `rs2` | uint8 (0–31) | arithmetic, store, branches, KLAUNCH |
| 5 | `dt` | uint8 (0–7) | T_MATMUL, T_MMA, T_DOT, T_LOAD, T_STORE, T_ACC_CLR, T_RELU, T_SCALE, T_BCAST |
| 6 | `a` | uint8 (0–7) | T_MATMUL, T_MMA, T_DOT, T_RELU, T_SCALE |
| 7 | `b` | uint8 (0–7) | T_MATMUL, T_MMA, T_DOT |
| 8 | `M` | uint8 (0–64) | T_CFG_SHAPE (and matmul shape) |
| 9 | `K` | uint8 (0–64) | T_CFG_SHAPE |
| 10 | `N` | uint8 (0–64) | T_CFG_SHAPE |
| 11 | `use_base` | uint8 (0/1) | C_LW/LH/LB/SW/SH/SB, T_LOAD, T_STORE, T_DMA_IN/OUT |
| 12–15 | `count` | uint32 LE | T_LOAD, T_STORE, T_DMA_IN/OUT |
| 16–23 | `imm` or `fimm` | int64 or double | branches, LI, addresses; float for SCALE/QUANT/DEQUANT |
| 24–31 | (padding) | zeros | reserved |

### 11.2.2 Vector Instruction Word (`vinst`, stream type byte = 1)

```
Byte   0       1       2       3       4       5       6       7
     ┌───────┬───────┬───────┬───────┬───────┬───────┬───────┬───────┐
     │  0x01 │ opcode│  vd   │  vs1  │  vs2  │  vs   │  rd   │  rs   │
     └───────┴───────┴───────┴───────┴───────┴───────┴───────┴───────┘
      stream   vop_t  vreg    vreg    vreg    vreg   sreg    sreg
       type   (0–49) (0–31) (0–31)  (0–31)  (0–31) (0–31)  (0–31)

Byte   8       9      10      11      12      13      14      15
     ┌───────┬───────┬───────┬───────┬───────┬───────┬───────┬───────┐
     │  rs1  │  rs2  │  md   │  m1   │  m2   │use_msk│ mask  │use_bas│
     └───────┴───────┴───────┴───────┴───────┴───────┴───────┴───────┘
      sreg    sreg   mreg    mreg    mreg     (0/1)  mreg idx  e (0/1)

Byte  16..23                  24      25      26..31
     ┌──────────────────────┬───────┬───────┬──────────────────────────┐
     │  imm (int64, LE)     │  cmp  │  cvt  │  padding (zeros)         │
     │  OR fimm (double)    │(0–5)  │(0–2)  │                          │
     └──────────────────────┴───────┴───────┴──────────────────────────┘
      imm = branch target / address / immediate integer
      fimm (double) used only when vop_is_float(op): V_SFI only.
      cmp = cmpkind_t: LT=0,LE=1,EQ=2,NE=3,GT=4,GE=5 (for V_VCMP)
      cvt = cvtmode_t: F32=0,I8=1,I32=2              (for V_VCVT)
```

**Field summary for vinst:**

| Byte(s) | Field | Type | Used by |
|---------|-------|------|---------|
| 0 | stream type | uint8 = 1 | decoder (always 1 for vinst) |
| 1 | opcode | `vop_t` uint8 | all |
| 2 | `vd` | uint8 (0–31) | VADD, VSUB, VMUL, VDIV, VMAX, VMIN, VFMA, VCVT, VABS, VNEG, VEXP, VLOG, VSQRT, VRSQRT, VRECIP, VSCAN, VBCAST, VMOV, VLOAD, VGATHER, VSEL |
| 3 | `vs1` | uint8 (0–31) | binary vector ops (VADD, VSUB, …), VFMA, VCMP, VSEL |
| 4 | `vs2` | uint8 (0–31) | binary vector ops, VFMA, VCMP, VSEL |
| 5 | `vs` | uint8 (0–31) | unary ops (VEXP, VLOG, VRSQRT, VSQRT, VRECIP, VABS, VNEG, VMOV, VSCAN, VMASK_SET), VSTORE, VSCATTER, VREDSUM/MAX/MIN/AND/OR |
| 6 | `rd` | uint8 (0–31) | VREDSUM, VREDMAX, VREDMIN, VREDAND, VREDOR, VSETVL, KID, SLI, SFI, SADD, SADDI, SSUB, SMUL, SMULI, SLW, SLF, SRECIP |
| 7 | `rs` | uint8 (0–31) | VBCAST, VLOAD, VSTORE, VGATHER, VSCATTER (address base) |
| 8 | `rs1` | uint8 (0–31) | SADD, SSUB, SMUL, SADDI, SMULI, SRECIP, VSETVL, SLW, SLF |
| 9 | `rs2` | uint8 (0–31) | SADD, SSUB, SMUL |
| 10 | `md` | uint8 (0–7) | VCMP, VMASK_SET, VMASK_AND, VMASK_OR, VMASK_NOT |
| 11 | `m1` | uint8 (0–7) | VMASK_AND, VMASK_OR, VMASK_NOT |
| 12 | `m2` | uint8 (0–7) | VMASK_AND, VMASK_OR |
| 13 | `use_mask` | uint8 (0/1) | predication prefix: applies mask to any instruction |
| 14 | `mask` | uint8 (0–7) | which mask register to use when use_mask=1 |
| 15 | `use_base` | uint8 (0/1) | VLOAD, VSTORE, VGATHER, VSCATTER, SLW, SLF |
| 16–23 | `imm` or `fimm` | int64 or double | branch targets, immediates, SFI float constant |
| 24 | `cmp` | uint8 `cmpkind_t` | VCMP: comparison operator |
| 25 | `cvt` | uint8 `cvtmode_t` | VCVT: conversion mode |
| 26–31 | (padding) | zeros | reserved |

---

## 11.3 The Encoding Code in npu_asm.c

The function `npu_encode_vinst(const vinst *I, uint8_t *out)` performs the
struct-to-binary conversion. Here is the logic in full:

```c
/* Each field is written to a fixed byte position.                        */
/* The output buffer `out` must be NPU_WORD_BYTES = 32 bytes.            */

void npu_encode_vinst(const vinst *I, uint8_t *out) {
    memset(out, 0, NPU_WORD_BYTES);
    out[0]  = 1;                        /* stream type: vinst             */
    out[1]  = (uint8_t)I->op;          /* opcode                         */
    out[2]  = (uint8_t)I->vd;
    out[3]  = (uint8_t)I->vs1;
    out[4]  = (uint8_t)I->vs2;
    out[5]  = (uint8_t)I->vs;
    out[6]  = (uint8_t)I->rd;
    out[7]  = (uint8_t)I->rs;
    out[8]  = (uint8_t)I->rs1;
    out[9]  = (uint8_t)I->rs2;
    out[10] = (uint8_t)I->md;
    out[11] = (uint8_t)I->m1;
    out[12] = (uint8_t)I->m2;
    out[13] = (uint8_t)I->use_mask;
    out[14] = (uint8_t)I->mask;
    out[15] = (uint8_t)I->use_base;

    /* Bytes 16–23: either int64 immediate or double float immediate.     */
    if (vop_is_float(I->op)) {
        /* V_SFI: encode as IEEE 754 double.                              */
        uint64_t bits;
        memcpy(&bits, &I->fimm, 8);     /* double → raw bits              */
        for (int k = 0; k < 8; k++)
            out[16 + k] = (bits >> (8 * k)) & 0xFF; /* little-endian     */
    } else {
        /* All other ops: encode imm as int64.                            */
        uint64_t uimm = (uint64_t)I->imm;
        for (int k = 0; k < 8; k++)
            out[16 + k] = (uimm >> (8 * k)) & 0xFF;
    }

    out[24] = (uint8_t)I->cmp;          /* cmpkind_t for VCMP             */
    out[25] = (uint8_t)I->cvt;          /* cvtmode_t for VCVT             */
    /* Bytes 26–31: remain zero (padding).                                */
}
```

Decoding (`npu_decode_vinst`) is the exact inverse: read each byte back into the struct
field. The stream type byte (0 vs 1) tells the decoder which struct to populate.

---

## 11.4 Three Equivalent Forms of an Instruction

Every NPU instruction has three interchangeable representations:

```
┌─────────────────────────────────────────────────────────────────────┐
│  TEXT FORM (assembly)                                               │
│                                                                     │
│   "VADD v2, v0, v1"                                                │
│                                                                     │
│   Human-readable. The assembler (npu_asm_vector) accepts this.      │
│   The disassembler (npu_disasm_vinst) produces it.                  │
└─────────────────────────────┬───────────────────────────────────────┘
                              │  asm / disasm
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│  STRUCT FORM (C struct)                                             │
│                                                                     │
│   (vinst){ .op=V_VADD, .vd=2, .vs1=0, .vs2=1 }                    │
│                                                                     │
│   What the simulator uses internally. What LLVM's NPUMCCodeEmitter  │
│   builds before calling npu_encode_vinst. All other fields are      │
│   implicitly zero (C designated initializer rules).                 │
└─────────────────────────────┬───────────────────────────────────────┘
                              │  encode / decode
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│  BINARY FORM (32-byte word)                                         │
│                                                                     │
│   Byte:  0    1    2    3    4    5    6    7  ...  31              │
│          01   12   02   00   01   00   00   00  ... 00              │
│          ↑    ↑    ↑         ↑                                      │
│        vinst VADD  vd=2    vs2=1                                    │
│              (=18)       vs1=0 (byte 3)                             │
│                                                                     │
│   Wire format. What goes into the ELF .text section. What          │
│   hardware fetches and decodes. Produced by npu_encode_vinst.       │
└─────────────────────────────────────────────────────────────────────┘
```

### 11.4.1 Worked Encoding: VADD v2, v0, v1

`V_VADD` has enum value 18 (the 19th entry in `vop_t`, zero-indexed).

```
Field       Value     Byte pos  Encoded byte
─────────── ───────── ──────── ─────────────
stream type   1         [0]      0x01
op (V_VADD) 18         [1]      0x12
vd            2         [2]      0x02
vs1           0         [3]      0x00
vs2           1         [4]      0x01
vs            0         [5]      0x00  (not used; zero)
rd            0         [6]      0x00
rs            0         [7]      0x00
rs1           0         [8]      0x00
rs2           0         [9]      0x00
md            0         [10]     0x00
m1            0         [11]     0x00
m2            0         [12]     0x00
use_mask      0         [13]     0x00
mask          0         [14]     0x00
use_base      0         [15]     0x00
imm=0     0x0000…    [16–23]    0x00 × 8
cmp           0         [24]     0x00
cvt           0         [25]     0x00
padding       0       [26–31]    0x00 × 6
```

Complete 32-byte hex dump:
```
01 12 02 00 01 00 00 00   00 00 00 00 00 00 00 00
00 00 00 00 00 00 00 00   00 00 00 00 00 00 00 00
```

### 11.4.2 Worked Encoding: SFI s4, 0.0

`V_SFI` has enum value 8. `fimm = 0.0` as IEEE 754 double = all zeros.

```
stream type   1   [0]   0x01
op (V_SFI)    8   [1]   0x08
rd            4   [6]   0x04    ← destination scalar s4
fimm (0.0)    0 [16–23] 0x00 × 8
(all others   0)
```

Complete hex dump:
```
01 08 00 00 00 00 04 00   00 00 00 00 00 00 00 00
00 00 00 00 00 00 00 00   00 00 00 00 00 00 00 00
```

### 11.4.3 Worked Encoding: VCMP.GT m0, v0, v1

`V_VCMP` = 25. `CMP_GT` = 4.

```
stream type   1   [0]   0x01
op (V_VCMP)  25   [1]   0x19
vs1           0   [3]   0x00
vs2           1   [4]   0x01
md            0   [10]  0x00
cmp=CMP_GT    4   [24]  0x04
```

Complete hex dump:
```
01 19 00 00 01 00 00 00   00 00 00 00 00 00 00 00
00 00 00 00 00 00 00 00   04 00 00 00 00 00 00 00
```

### 11.4.4 Worked Encoding: LOOP s2, 3 (branch to instr 3)

`V_LOOP` = 48. `imm = 3` (instruction index). Branch imm is stored as int64 in bytes 16–23.

```
stream type   1       [0]     0x01
op (V_LOOP)  48       [1]     0x30
rs            2       [7]     0x02    ← countdown register s2
imm           3   [16–23]    0x03 0x00 0x00 0x00 0x00 0x00 0x00 0x00
```

Complete hex dump:
```
01 30 00 00 00 00 00 02   00 00 00 00 00 00 00 00
03 00 00 00 00 00 00 00   00 00 00 00 00 00 00 00
```

---

## 11.5 ELF Output from llc

When you compile to an object file with the NPU backend:

```sh
llc -mtriple=npu-unknown-unknown -filetype=obj input.ll -o output.o
```

The LLVM NPU backend emits a standard ELF object file with the following structure:

### 11.5.1 ELF Header

```
ELF Header:
  Magic:     7f 45 4c 46  (ELF magic)
  Class:     ELF32
  Data:      2's complement, little endian
  Version:   1 (current)
  OS/ABI:    UNIX - System V
  Type:      REL (Relocatable file)
  Machine:   <unknown>: 0x13b    ← EM_NPU = 315 = 0x13B
  Version:   0x1
  Entry:     0x0
  Flags:     0x0
```

`EM_NPU = 315` is defined in `NPUTargetMachine.cpp` and registered with the ELF writer.
This value was chosen to not conflict with any existing LLVM target. Hardware toolchains
that consume the ELF would check for this value to confirm they have an NPU object.

### 11.5.2 ELF Sections

| Section | Content | Notes |
|---------|---------|-------|
| `.text` | N × 32 bytes | One 32-byte word per instruction, in program order |
| `.rela.text` | RELA records | One entry per unresolved branch/call target |
| `.symtab` | Symbol table | Function names and their byte offsets in `.text` |
| `.strtab` | String table | Symbol name strings |
| `.shstrtab` | Section name strings | Standard ELF metadata |

**`.text` layout**: Instruction 0 is at offset 0, instruction 1 at offset 32, instruction k
at offset `k * 32`. For a kernel with 10 instructions, `.text` is 320 bytes.

### 11.5.3 Reading the ELF

```sh
# Show ELF header and section headers
llvm-readelf -h -S output.o

# Dump the raw bytes of .text (16 bytes per line for readability)
llvm-readelf -x .text output.o

# Show all relocations
llvm-readelf -r output.o

# Show symbol table
llvm-readelf -s output.o
```

---

## 11.6 Relocation Types

Relocations arise when a branch or call target is a symbol in another compilation unit (or
a forward reference that requires link-time resolution).

### 11.6.1 Relocation Table

| Name | Value | Formula | When Used |
|------|-------|---------|-----------|
| `R_NPU_32` | 1 | `S + A` | 32-bit data reference to a symbol |
| `R_NPU_64` | 2 | `S + A` | 64-bit data reference to a symbol |
| `R_NPU_BRANCH` | 3 | `(S + A) / NPU_WORD_BYTES` | Branch/call target (instruction index) |

Notation: `S` = symbol address (in bytes), `A` = addend, `NPU_WORD_BYTES = 32`.

### 11.6.2 R_NPU_BRANCH in Detail

The NPU branch and call instructions encode their target as an **instruction index**, not a
byte offset. Instruction index 5 means "jump to the 6th instruction in the program" —
which is at byte offset `5 * 32 = 160`.

Why instruction indices? Because 32-byte alignment means the lower 5 bits of any valid
instruction address are always zero. Storing byte addresses in `imm` would waste 5 bits and
constrain the addressable range. Instruction indices double the effective range.

The `applyFixup` function in `NPUAsmBackend.cpp` applies this transformation at link time:

```c
/* NPUAsmBackend::applyFixup — called by the linker for each RELA record */
void NPUAsmBackend::applyFixup(const MCAssembler &Asm,
                               const MCFixup &Fixup,
                               const MCValue &Target,
                               SmallVectorImpl<char> &Data,
                               uint64_t Value,        /* byte address */
                               bool IsResolved) {
    /* Convert byte address to instruction index. */
    uint64_t instr_index = Value / NPU_WORD_BYTES;

    /* Write as little-endian int64 at the fixup offset (byte 16 of word). */
    for (int i = 0; i < 8; i++)
        Data[Fixup.getOffset() + i] = (instr_index >> (8 * i)) & 0xFF;
}
```

The `r_offset` field in the RELA record points to byte 16 of the instruction word (the start
of the `imm` field), because that is where branch targets are stored in the binary encoding.

### 11.6.3 Example: Relocation for a Function Call

Given LLVM IR:

```llvm
declare void @external_relu_pass()

define void @my_kernel() {
  call void @external_relu_pass()
  ret void
}
```

The assembler emits `C_JAL` or `C_JALR` for the call. If `external_relu_pass` is in another
object file, the linker must patch the target address. The ELF `.rela.text` section contains:

```
Relocation section '.rela.text':
  Offset    Info                   Type           Sym. Value  Sym. Name + Addend
  00000010  0000000200000003       R_NPU_BRANCH   00000000    external_relu_pass + 0
```

At offset `0x10 = 16` (byte 16 of instruction 0), the linker writes
`symbol_address / 32` as an `int64`.

---

## 11.7 The LLVM Backend Compilation Pipeline

The complete path from LLVM IR to a linked binary:

```
  LLVM IR (matmul.ll)
  ─────────────────────────────────────────────────────────
  │
  │  llvm::SelectionDAGISel + NPUISelDAGToDAG::Select()
  │  • Lowers intrinsics (@llvm.npu.*) to NPUISD:: nodes
  │  • Pattern-matches to NPUInstrInfo.td instruction defs
  │  • Tile register numbers become TargetConstant:i64 in DAG
  │
  ▼
  NPU Machine Instructions (MachineInstr / MCInst)
  ─────────────────────────────────────────────────────────
  │
  │  NPUAsmPrinter::emitInstruction()
  │  • Converts MachineInstr → MCInst
  │  • Calls MCStreamer::emitInstruction()
  │
  ▼
  MCInst (architecture-neutral instruction representation)
  ─────────────────────────────────────────────────────────
  │
  │  NPUMCCodeEmitter::encodeInstruction()
  │  • Reads opcode and operands from MCInst
  │  • Builds the vinst or cinst struct in memory
  │  • Calls npu_encode_vinst() / npu_encode_cinst()
  │  • Emits NPU_WORD_BYTES=32 bytes to the output stream
  │
  ▼
  32-byte binary words → ELF .text section
  ─────────────────────────────────────────────────────────
  │
  │  NPUAsmBackend::applyFixup()
  │  • For each MCFixup (unresolved symbol reference):
  │    - Computes symbol_address / 32 (instruction index)
  │    - Writes int64 into bytes [16..23] of the instruction
  │  • Creates RELA records via ELFObjectWriter
  │
  ▼
  ELF object file (.o) with:
    .text:      N × 32 bytes (instruction words)
    .rela.text: relocation records for external refs
    .symtab:    kernel function names → instruction offsets
  ─────────────────────────────────────────────────────────
  │
  │  lld or npu-ld: resolves relocations, links .o files
  │
  ▼
  Linked ELF binary (or raw binary for embedded loader)
```

### 11.7.1 Key Design Choices in the LLVM Backend

**Tile registers as SSA i64 constants.** The tensor engine has 8 tile registers (`t0..t7`).
Rather than introducing a `TileReg` register class (which would require a register
allocator), the NPU backend treats tile numbers as compile-time constants in the LLVM DAG.
The function `getTileConst()` in `NPUISelDAGToDAG.cpp` walks the use-def chain through
`MERGE_VALUES` and `INTRINSIC_W_CHAIN` nodes to extract the integer constant. This works
because every tile operation in a typical program has a statically known tile assignment.

**T_BCAST uses GPR, not fimm.** The instruction `T_BCAST` broadcasts a scalar float to all
elements of a tile. In the simulator, it reads `(float)getx(n, I->rs1)` — the float value
is bit-cast from a GPR. The compiler materializes the float constant via a `C_LI` instruction
(which loads a 64-bit integer) and then uses `T_BCAST` to consume the raw bits. This avoids
the complexity of encoding a float in the `fimm` field for BCAST.

**T_SCALE uses double bits as TargetConstant:i64.** The `T_SCALE` factor is a `double`
(`fimm` in the cinst). To pass it through the SelectionDAG without triggering f64
type-legalization (which would try to split it), the backend converts the double to its
`uint64_t` bit pattern and carries it as a `TargetConstant:i64`. The `NPUMCCodeEmitter`
then converts it back: `memcpy(&fimm, &bits, 8)`.

**V_VSETVL uses the GPR file.** Vector length (`vl`) is an integer count. The `VSETVL`
instruction's `rd` field names an engine scalar register (`s0..s31`), but the value is
always treated as `int64_t` (using the `.i` member of the scalar union). The LLVM intrinsic
`@llvm.npu.vsetvl(i64 len)` returns `i64`, and both request and result live in the `SReg`
LLVM register class.

**Data layout: 64-bit pointers.** The NPU data layout string is:
```
e-m:e-p:64:64-i64:64-n64-S64
```
Pointers are 64 bits wide (matching the `int64_t` address fields in the instruction structs),
`i64` is naturally aligned to 8 bytes, and the stack is 64-bit aligned.

---

## 11.8 The Text Assembler

`npu_asm_vector()` and `npu_asm_control()` parse text assembly and populate struct arrays.
The assembler is line-oriented:

```
line := [label ':'] [pred] mnemonic [operands] [';' comment]
pred  := '@' 'm' digit             ; predication: @m0..@m7
```

Operands use standard RISC-style syntax:
- Vector regs: `v0`..`v31`
- Engine scalars: `s0`..`s31`
- Mask regs: `m0`..`m7`
- Memory: `imm(sN)` = `use_base=1, rs=N, imm=value`
- VCMP/VCVT suffixes: `VCMP.GT`, `VCVT.I8`, etc.

**Roundtrip guarantee**: `npu_encode_vinst(decode(encode(I))) == I` (encoding is
information-preserving). `npu_disasm_vinst(encode(I))` produces text that `npu_asm_vector`
re-assembles to the original struct (modulo unused-field normalization).

---

## 11.9 Summary

| Topic | Key fact |
|-------|----------|
| Instruction size | 32 bytes (`NPU_WORD_BYTES`), fixed for all three engines |
| Stream type byte | Byte 0: 0 = cinst (control/tensor), 1 = vinst (vector) |
| Opcode | Byte 1: `cop_t` or `vop_t`, values 0–49 |
| Registers | Bytes 2–9 (vinst) or 2–7 (cinst): register indices, 0–31 or 0–7 |
| Immediate | Bytes 16–23: `int64_t` LE or `double` IEEE 754 LE (for SFI, T_SCALE/QUANT/DEQUANT) |
| Float vs int imm | `vop_is_float(op)` true only for `V_SFI`; `cop_is_float(op)` for T_SCALE/QUANT/DEQUANT |
| Branch targets | Encoded as instruction index = byte_address / 32 |
| ELF machine code | `EM_NPU = 315 = 0x13B` |
| Relocation for branches | `R_NPU_BRANCH`: linker divides symbol byte address by 32 |
| LLVM fixup offset | Byte 16 of the instruction word (start of `imm` field) |
| Assembler roundtrip | encode → decode and asm → disasm are both exact inverses |
