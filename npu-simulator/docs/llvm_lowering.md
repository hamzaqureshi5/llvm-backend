# LLVM IR Lowering for the NPU

How a matmul (and other kernels) would be represented in **LLVM IR** on the way
to this NPU's instruction streams — the "compiler/toolchain" layer named in
[`fully_prog_npu.md`](./fully_prog_npu.md) §15.5.

> **Status:** design sketch. No NPU LLVM/MLIR backend exists in this repo yet
> (the simulator is standalone — see the README's *Current limitations*). The
> intrinsic names below are **illustrative**, but each maps 1:1 onto a `cinst` /
> `vinst` the simulator already executes, and onto the 32-byte words produced by
> `npu_encode_cinst` / `npu_encode_vinst`.

---

## The key idea: three layers, three very different IR representations

The ISA's three layers do **not** lower to LLVM IR the same way:

| Layer | LLVM IR representation | Why |
|-------|------------------------|-----|
| **Control core** (§1–3) | Ordinary scalar IR (`add`, `mul`, `br`, `phi`, `getelementptr`) → a RISC-V-style backend | It *is* RISC-V; loops / address math / branches are normal IR |
| **Tensor engine** (§4–8) | **Target intrinsics** (`@llvm.npu.mma`, …) over an opaque `target("npu.tile")` type | A whole-tile matmul is a coarse op, not expressible as scalar/vector IR (cf. x86 AMX's `x86_amx` type) |
| **Vector core** (§9–13) | **Native LLVM vector IR** (`fadd <N x float>`, `fmul`, …) **+** intrinsics for the rest | It mirrors RISC-V Vector (RVV) — elementwise math is native; only loads/reductions/transcendentals need intrinsics |

So "the LLVM IR for this" is really two stories, depending on which engine an op
lands on.

---

## Where it sits in the IREE pipeline

```
linalg.matmul (128x128)
  └─ tile to 64x64 (IREE codegen, K-block reduction loop)
       └─ pick a path:
            ├─ tensor engine: lower tile op → NPU tensor intrinsics
            └─ vector core:   vectorize → vector.* → LLVM vector IR + NPU vector intrinsics
                 └─ Instruction selection → cinst[] / vinst[]
                      → npu_encode_* → 32-byte words → simulator runs it
```

The last mile closes the round-trip the README sketches:

```
LLVM IR  →(isel)→  cinst/vinst struct  →(npu_encode_*)→  32-byte word  →  simulator
```

---

## Path A — Tensor engine (the tiled 128×128 matmul)

The interesting property: **`MMA`'s K-accumulation becomes a phi-chained SSA
value** over the contraction loop. Tile registers are SSA values of an opaque
target type; the address GEPs are the `MUL`/`ADD` index math the control core
runs.

```llvm
; intrinsics  ↔  the cinst T_* ops the simulator runs
declare void                @llvm.npu.cfg.shape(i32, i32, i32)            ; T_CFG_SHAPE
declare target("npu.tile")  @llvm.npu.acc.clr()                           ; T_ACC_CLR
declare target("npu.tile")  @llvm.npu.load(ptr, i32)                      ; T_LOAD  (addr, count)
declare target("npu.tile")  @llvm.npu.loadw(ptr, i32)                     ; T_LOADW
declare target("npu.tile")  @llvm.npu.mma(target("npu.tile"),             ; T_MMA   (acc, a, b)
                                          target("npu.tile"),
                                          target("npu.tile"))
declare void                @llvm.npu.store(ptr, target("npu.tile"), i32) ; T_STORE
declare void                @llvm.npu.dma.in(ptr, ptr, i32)               ; T_DMA_IN

define void @matmul_128(ptr %A, ptr %B, ptr %D, ptr %scrA, ptr %scrB) {
entry:
  call void @llvm.npu.cfg.shape(i32 64, i32 64, i32 64)
  call void @llvm.npu.dma.in(ptr %scrA, ptr %A, i32 65536)
  call void @llvm.npu.dma.in(ptr %scrB, ptr %B, i32 65536)
  br label %jb.body                              ; (ib/jb loops elided)

jb.body:                                         ; one output tile (ib,jb)
  %acc0 = call target("npu.tile") @llvm.npu.acc.clr()
  br label %kb.body

kb.body:                                         ; contraction over K-blocks
  %kb   = phi i32 [ 0, %jb.body ], [ %kb.n, %kb.body ]
  %acc  = phi target("npu.tile") [ %acc0, %jb.body ], [ %acc.n, %kb.body ]
  %aptr = getelementptr float, ptr %scrA, i32 ...   ; (ib*KB+kb)*4096
  %bptr = getelementptr float, ptr %scrB, i32 ...   ; (kb*NB+jb)*4096
  %a    = call target("npu.tile") @llvm.npu.load(ptr %aptr, i32 4096)
  %b    = call target("npu.tile") @llvm.npu.loadw(ptr %bptr, i32 4096)
  %acc.n = call target("npu.tile") @llvm.npu.mma(target("npu.tile") %acc,
                                                 target("npu.tile") %a,
                                                 target("npu.tile") %b)
  %kb.n = add i32 %kb, 1
  %done = icmp eq i32 %kb.n, 2
  br i1 %done, label %store, label %kb.body

store:
  %dptr = getelementptr float, ptr %D, i32 ...
  call void @llvm.npu.store(ptr %dptr, target("npu.tile") %acc.n, i32 4096)
  ret void                                       ; (ib/jb back-edges elided)
}
```

The `%acc` phi is exactly `t2` in the hand-written assembly: cleared once
(`ACC.CLR`), threaded through each `MMA`, stored after the loop. This is why the
ISA has both `MATMUL` (overwrite) and `MMA` (accumulate) — the K-reduction needs
the accumulating form.

---

## Path B — Vector core (e.g. softmax): real LLVM vector IR

Elementwise ops are **native** LLVM and lower straight to `V_VADD` / `V_VSUB` /
`V_VMUL`; only loads, reductions, and transcendentals need intrinsics — exactly
the RVV story.

```llvm
declare <64 x float> @llvm.npu.vload(ptr)
declare float        @llvm.npu.vredmax(<64 x float>)   ; V_VREDMAX
declare float        @llvm.npu.vredsum(<64 x float>)   ; V_VREDSUM
declare <64 x float> @llvm.npu.vexp(<64 x float>)      ; V_VEXP
declare void         @llvm.npu.vstore(ptr, <64 x float>)

define void @softmax_row(ptr %x, ptr %y) {
  %row  = call <64 x float> @llvm.npu.vload(ptr %x)
  %max  = call float @llvm.npu.vredmax(<64 x float> %row)
  %mv   = insertelement <64 x float> poison, float %max, i32 0
  %mb   = shufflevector <64 x float> %mv, <64 x float> poison,
                        <64 x i32> zeroinitializer        ; VBCAST
  %sub  = fsub <64 x float> %row, %mb                      ; <- plain LLVM IR -> V_VSUB
  %e    = call <64 x float> @llvm.npu.vexp(<64 x float> %sub)
  %sum  = call float @llvm.npu.vredsum(<64 x float> %e)
  %rs   = fdiv float 1.0, %sum
  %rsv  = insertelement <64 x float> poison, float %rs, i32 0
  %rsb  = shufflevector <64 x float> %rsv, <64 x float> poison,
                        <64 x i32> zeroinitializer
  %out  = fmul <64 x float> %e, %rsb                       ; <- plain LLVM IR -> V_VMUL
  call void @llvm.npu.vstore(ptr %y, <64 x float> %out)
  ret void
}
```

A production target would use **scalable** vectors `<vscale x N x float>` plus a
`vsetvl` intrinsic to get the length-agnostic `VL` behaviour (§9), exactly like
the upstream RVV backend. `@m` predication (§12) maps to LLVM's masked-vector
intrinsics or `select`.

---

## Intrinsic → ISA mapping (quick reference)

| LLVM construct | NPU instruction | Layer |
|----------------|-----------------|-------|
| `@llvm.npu.cfg.shape` | `T_CFG_SHAPE` | tensor |
| `@llvm.npu.load` / `.loadw` / `.store` | `T_LOAD` / `T_LOADW` / `T_STORE` | tensor |
| `@llvm.npu.mma` / `.matmul` | `T_MMA` / `T_MATMUL` | tensor |
| `@llvm.npu.acc.clr` | `T_ACC_CLR` | tensor |
| `@llvm.npu.dma.in` / `.out` | `T_DMA_IN` / `T_DMA_OUT` | tensor |
| `fadd`/`fsub`/`fmul`/`fdiv` `<N x float>` | `V_VADD`/`V_VSUB`/`V_VMUL`/`V_VDIV` | vector |
| `fmuladd` `<N x float>` | `V_VFMA` | vector |
| `@llvm.npu.vexp` / `.vlog` / `.vrsqrt` … | `V_VEXP` / `V_VLOG` / `V_VRSQRT` … | vector |
| `@llvm.npu.vredsum` / `.vredmax` … | `V_VREDSUM` / `V_VREDMAX` … | vector |
| `shufflevector`(splat) | `V_VBCAST` | vector |
| `select` / masked intrinsics | `@m` predication, `V_VSEL` | vector |
| `add`/`mul`/`br`/`phi`/`icmp` (scalar) | `C_ADD`/`C_MUL`/`C_BEQ`…/`C_JAL` | control |

---

## Realistic build order

1. **MLIR `npu` dialect** mirroring these ops (`npu.mma`, `npu.vexp`, …) + an
   IREE target backend that tiles `linalg.matmul` to 64×64 and bufferizes the
   working set into the SRAM scratchpad.
2. Either **emit the binary stream directly from MLIR**, or go through an
   **LLVM target** whose intrinsics are the table above and whose instruction
   selector produces `cinst` / `vinst`, then `npu_encode_*`.

Either way the simulator in this directory is the executable back-end target:
feed it the encoded stream and it runs the kernel.
