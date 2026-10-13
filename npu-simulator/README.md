# NPU Simulator

A standalone **functional simulator** (instruction-set simulator) for the
fully-programmable NPU described in [`fully_prog_npu.md`](./fully_prog_npu.md).
It lets you write and run NPU programs in C and check their results without real
hardware.

This is an **independent project** — it is not wired into the IREE build.

## What it models

All three layers of the reference ISA, interpreted from C struct instruction
arrays:

| Layer | What | Stream |
|-------|------|--------|
| 1. Control core | Scalar ISA (`ADD`/`LW`/`BEQ`/`JAL`…) + kernel launch (`KLAUNCH`/`KWAIT`) | `cinst[]` |
| 2. Tensor engine | Dense matmul fast path (`CFG`/`MMA`/`LOAD`/`STORE`/activations) | `cinst[]` (`NPU.*` ops) |
| 3. Vector core | Programmable per-grid-point kernels (`VADD`/`VEXP`/`VREDSUM`/`VCMP`/`VSEL`/masks/branches) | `vinst[]` |

Fidelity: **functional** (correct results) **+ rough cycle estimates** per engine.
It is *not* cycle-accurate — the cycle model is a simple cost table for intuition.

It also models:

- **Multiple vector cores** — `npu_set_cores(n, N)` spreads a kernel launch's grid
  instances round-robin across `N` cores. Modeled vector cycles for a launch become
  the **max over cores** (busiest core); `stats.vector_work` keeps the serial sum so
  you can see the speedup.
- **On-chip SRAM scratchpad + DMA** — a separate, capacity-limited address region.
  DRAM accesses are expensive, SRAM accesses are cheap, and `DMA.IN/OUT` move data
  between them. This makes the "stage to on-chip memory, then compute" pattern pay
  off in the cycle estimates (see Demo 6).

## Build & run

```sh
cmake -S . -B build
cmake --build build
./build/npu_demo          # run the self-checking demos
ctest --test-dir build    # or run them through CTest
```

Expected output ends with `ALL DEMOS PASSED`.

## Files

- `npu.h` — ISA opcodes, instruction structs, machine state, public API.
- `npu.c` — the interpreters (control core, tensor engine, vector core) + cycle model.
- `npu_asm.h` / `npu_asm.c` — text **assembler**, binary **encoder/decoder**, and
  **disassembler** for the ISA.
- `examples.c` — seven worked, self-checking demos (also the `npu_demo` driver):
  vadd, softmax, matmul, predication, multi-core, SRAM/DMA staging, and
  assembler + encode/decode round-trip.
- `CMakeLists.txt` — builds `libnpu` (core + assembler) and `npu_demo`.
- `llvm_lowering.md` — design sketch of how a matmul / kernels would be
  represented in **LLVM IR** and lowered to the ISA (the future compiler layer).

## How you write a program

A program is two pieces of plain C data:

1. A **vector kernel** = an array of `vinst` (the programmable engine's instruction
   stream).
2. A **control program** = an array of `cinst` that sets up memory/grid and
   `KLAUNCH`es the kernel.

Launch convention for a kernel instance: on entry `s0` = grid instance id,
`s1` = args base pointer (into NPU memory). The kernel reads tensor base
addresses / shapes from the args block.

### Example: softmax (Section 14 of the reference), built from primitives

```c
static const vinst softmax[] = {
    { .op=V_SLW, .rd=2, .use_base=1, .rs=1, .imm=0 },   /* s2 = x base */
    { .op=V_SLW, .rd=3, .use_base=1, .rs=1, .imm=4 },   /* s3 = y base */
    { .op=V_SLW, .rd=4, .use_base=1, .rs=1, .imm=8 },   /* s4 = n      */
    { .op=V_SMUL,  .rd=5, .rs1=0, .rs2=4 },             /* s5 = id*n   */
    { .op=V_SMULI, .rd=5, .rs1=5, .imm=4 },             /* bytes       */
    { .op=V_SADD,  .rd=6, .rs1=2, .rs2=5 },             /* x row ptr   */
    { .op=V_SADD,  .rd=7, .rs1=3, .rs2=5 },             /* y row ptr   */
    { .op=V_VSETVL, .rd=8, .rs1=4 },                    /* VL = n      */
    { .op=V_VLOAD,  .vd=0, .use_base=1, .rs=6 },
    { .op=V_VREDMAX, .rd=9, .vs=0 },                    /* max(x)      */
    { .op=V_VBCAST, .vd=1, .rs=9 },
    { .op=V_VSUB,   .vd=0, .vs1=0, .vs2=1 },            /* x - max     */
    { .op=V_VEXP,   .vd=0, .vs=0 },
    { .op=V_VREDSUM, .rd=10, .vs=0 },                   /* sum         */
    { .op=V_SRECIP, .rd=11, .rs1=10 },                  /* 1/sum       */
    { .op=V_VBCAST, .vd=1, .rs=11 },
    { .op=V_VMUL,   .vd=0, .vs1=0, .vs2=1 },            /* normalize   */
    { .op=V_VSTORE, .vs=0, .use_base=1, .rs=7 },
    { .op=V_KEXIT },
};
```

Then from the host side:

```c
npu_t *n = npu_create();
npu_register_kernel(n, softmax, sizeof softmax / sizeof softmax[0]);   /* -> kid 0 */
/* ... write inputs + args block into memory with npu_write_f32 / npu_write_i32 ... */
const cinst prog[] = {
    { .op=C_LI, .rd=10, .imm=ROWS },                  /* grid = #rows */
    { .op=C_LI, .rd=11, .imm=ARGS },                  /* args pointer */
    { .op=C_KLAUNCH, .imm=0 /*kid*/, .rs1=10, .rs2=11 },
    { .op=C_KWAIT },
    { .op=C_HALT },
};
npu_run(n, prog, sizeof prog / sizeof prog[0]);
/* ... npu_read_f32 the result ... */
npu_destroy(n);
```

## Writing kernels as assembly (instead of C structs)

`npu_asm.h` lets you reach the same interpreter from **assembly text** or from a
**binary instruction stream** — the three forms of an instruction are:

```
 TEXT  "VADD v2, v0, v1"  <--asm/disasm-->  STRUCT {V_VADD,...}  <--enc/dec-->  BYTES
```

```c
const char *src =
    "  VSETVL s8, s4\n"
    "  VLOAD  v0, 0(s6)\n"
    "  VREDMAX s9, v0\n"
    "  VBCAST v1, s9\n"
    "  VSUB   v0, v0, v1\n"
    "  VEXP   v0, v0\n"
    "  KEXIT\n";

vinst code[64]; char err[128];
int n = npu_asm_vector(src, code, 64, err, sizeof err);   /* text -> structs */
if (n < 0) { /* err holds the message */ }
npu_register_kernel(npu, code, n);                         /* run as usual    */
```

Binary round-trip (what a future LLVM backend would emit, and the sim would load):

```c
unsigned char word[NPU_WORD_BYTES];
npu_encode_vinst(&code[i], word);     /* struct -> 32-byte machine word */
npu_decode_vinst(word, &code2[i]);    /* machine word -> struct         */

char text[80];
npu_disasm_vinst(&code[i], text, sizeof text);   /* struct -> text */
```

Assembly syntax notes:
- Registers: `v0..v31`, `s0..s31` (engine scalars), `m0..m7` (masks),
  `x0..x31` (control), `t0..t7` (tensor tiles).
- Memory operands: `OFF(sREG)` (e.g. `0(s6)`) or a bare absolute address.
- Comments start with `;` or `#`. Labels are `name:` and branches refer to them.
- Predication: prefix an instruction with `@m<idx>` (e.g. `@m0 VADD v2, v2, v3`).
- Suffixed ops: `VCMP.GT`, `VCMP.LE`, …, and `VCVT.I8` / `VCVT.F32` / `VCVT.I32`.
- `npu_asm_vector` assembles a vector kernel; `npu_asm_control` assembles a
  control/tensor program (e.g. `LI`, `KLAUNCH 0, x10, x11`, `CFG.SHAPE 2,3,2`, `MMA t2, t0, t1`).

## Operand conventions (quick reference)

- Vector regs `v0..v31` (`VLEN_MAX` lanes), mask regs `m0..m7`, engine scalars
  `s0..s31`, control regs `x0..x31` (`x0` hardwired 0).
- Two address spaces share the 32-bit space: **DRAM** (high bit clear, up to
  `NPU_MEM_SIZE`) and the **on-chip SRAM scratchpad** (high bit set, i.e.
  `SCRATCH_BASE | offset`, up to `NPU_SCRATCH_SIZE`). Use `DMA.IN/OUT` to move data
  between them; over-running the scratchpad faults.
- Addressing: set `.use_base=1, .rs=<reg>, .imm=<offset>` → `addr = s[rs] + imm`.
  With `.use_base=0`, `.imm` is an absolute address.
- Predication ("@m" in the reference): set `.use_mask=1, .mask=<m-reg>` on a vector
  op so it only updates lanes where that mask is 1.
- Branch/loop targets (`V_BR`/`V_BRZ`/`V_BRNZ`/`V_LOOP`, `C_BEQ`/…/`C_JAL`) are
  **instruction indices** into the same array.

## Current limitations

- `T_CONV` and `T_POOL` are stubs (build them from vector-core primitives instead).
- Vector registers store `float`; integer/quantized lanes are modeled as float
  values (`VCVT` rounds/clamps). Fine for functional checks, not bit-exact int8.
- The cycle model is illustrative (single latency per memory tier, no bank
  conflicts/ports, multi-core via max-over-cores). Grid instances are executed
  sequentially on one host thread; multi-core is a *cycle model*, not real threads.
- Not yet integrated with IREE/LLVM — purely standalone for now.
