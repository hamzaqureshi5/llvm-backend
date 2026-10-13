# Layer 3 — The Vector Core

The vector core is the fully programmable engine of the NPU. It handles
everything the tensor engine cannot: softmax, layernorm, custom activations,
reductions, gather/scatter, and any per-element operation that requires
arbitrary math.

It runs a **separate program** (`vinst[]`) compiled independently from the
control core's `cinst[]`. The control core registers this program with
`npu_register_kernel()` and dispatches it with `KLAUNCH`.

---

## 3.1 What It Does That Layer 2 Cannot

| Operation | Tensor Engine (L2) | Vector Core (L3) |
|---|---|---|
| Dense matmul | ✓ fast path | ✓ (slow, via VMUL loop) |
| RELU, SCALE | ✓ | ✓ |
| exp, log, sqrt | ✗ | ✓ VEXP, VLOG, VSQRT |
| Softmax | ✗ | ✓ (VEXP + VREDSUM + VDIV) |
| Arbitrary reduction | ✗ | ✓ VREDSUM, VREDMAX, VREDMIN |
| Gather / scatter | ✗ | ✓ VGATHER, VSCATTER |
| Per-lane conditionals | ✗ | ✓ via mask registers |
| Custom activation | ✗ | ✓ any vinst sequence |

---

## 3.2 Three Register Files

Each grid instance starts with **all three completely zeroed** — fresh state
every time, no carry-over between instances.

```
v0–v31    vector registers    each holds vl floats  (vl ≤ VLEN_MAX = 256)
s0–s31    scalar registers    each holds one float  (addressing, reductions)
m0–m7     mask registers      each holds vl bits    (per-lane predication)
```

Two registers are pre-loaded on every instance entry:

```
s0 = grid_instance_id    (0, 1, 2 ... grid_size-1)
s1 = args_ptr            (NPU DRAM address of the args struct)
```

Everything else is zero.

---

## 3.3 Instruction Set

### Memory

```
VSETVL   rd, rs         Set vector length to min(rs, VLEN_MAX). Returns
                         actual vl in rd. Must come before any vector op.

VLOAD    vd, rs         Load vl floats from address s[rs] into vd.
VSTORE   rs, vs         Store vl floats from vs to address s[rs].
VGATHER  vd, rs, vi     vd[l] = mem[ s[rs] + vi[l] ]   (indexed load)
VSCATTER rs, vi, vs     mem[ s[rs] + vi[l] ] = vs[l]   (indexed store)
```

### Compute — element-wise, all lanes in parallel

```
VADD     vd, v0, v1     vd[l] = v0[l] + v1[l]
VSUB     vd, v0, v1     vd[l] = v0[l] - v1[l]
VMUL     vd, v0, v1     vd[l] = v0[l] * v1[l]
VDIV     vd, v0, v1     vd[l] = v0[l] / v1[l]
VFMA     vd, v0, v1, v2 vd[l] = v0[l] * v1[l] + v2[l]
VEXP     vd, vs         vd[l] = exp(vs[l])
VLOG     vd, vs         vd[l] = log(vs[l])
VSQRT    vd, vs         vd[l] = sqrt(vs[l])
VRECIP   vd, vs         vd[l] = 1 / vs[l]
VMAX     vd, v0, v1     vd[l] = max(v0[l], v1[l])
VMIN     vd, v0, v1     vd[l] = min(v0[l], v1[l])
VABS     vd, vs         vd[l] = |vs[l]|
VNEG     vd, vs         vd[l] = -vs[l]
```

### Reductions — vector → scalar

```
VREDSUM  rd, vs         s[rd] = sum of all vs[l]
VREDMAX  rd, vs         s[rd] = max over all lanes
VREDMIN  rd, vs         s[rd] = min over all lanes
```

### Predication

```
VCMP     md, v0, v1, cmp    md[l] = (v0[l] cmp v1[l]) ? 1 : 0
VMASK_NOT md, ms            md[l] = !ms[l]
VMASK_AND md, m0, m1        md[l] = m0[l] & m1[l]
VMASK_OR  md, m0, m1        md[l] = m0[l] | m1[l]

Prefix @m on any compute/memory op → runs only on lanes where m[l] == 1.
```

### Misc

```
VBCAST   vd, rs         vd[l] = s[rs]  for all l   (broadcast scalar)
VMOV     vd, vs         copy vector register
KEXIT                   return from this grid instance
```

---

## 3.4 How Multiple Instances Work

`KLAUNCH kid, grid=N` spawns N instances of the registered kernel.

**What KLAUNCH does internally:**

```c
for (int64_t g = 0; g < N; g++) {
    memset(v, 0, ...);     // all vector registers cleared
    memset(s, 0, ...);     // all scalar registers cleared
    s[0] = g;              // s0 = instance id
    s[1] = args_ptr;       // s1 = args address
    run_kernel_to_kexit(); // runs to completion before next instance starts
}
```

**The only thing that changes between instances is `s0`.**
Same kernel code, same args_ptr, same starting register state (all zero).
The instance id is the single knob the kernel uses to select its data slice.

---

## 3.5 Complete Example — Vector Addition, 4 Instances

**Setup:**

```
DRAM 0x1000 : float A[64]
DRAM 0x2000 : float B[64]
DRAM 0x3000 : float C[64]  (output)
DRAM 0x4000 : args = { 0x1000, 0x2000, 0x3000 }

KLAUNCH  kid, grid=4       4 instances, 16 floats each
KWAIT
```

**The kernel:**

```
VSETVL   16                    set vl = 16 floats

SMULI  s3, s0, 64              s3 = s0 × 64  (byte offset: 16 floats × 4 bytes)

SLW    s4, s1, 0               s4 = args[0] = 0x1000  (A base)
SLW    s5, s1, 8               s5 = args[1] = 0x2000  (B base)
SLW    s6, s1, 16              s6 = args[2] = 0x3000  (C base)

SADD   s4, s4, s3              s4 = A + offset
SADD   s5, s5, s3              s5 = B + offset
SADD   s6, s6, s3              s6 = C + offset

VLOAD  v0, s4                  load 16 floats from A slice
VLOAD  v1, s5                  load 16 floats from B slice
VADD   v2, v0, v1              16 additions in one instruction
VSTORE s6, v2                  write 16 results

KEXIT
```

**What each instance computes:**

```
Instance 0  (s0 = 0):
  offset = 0 × 64 = 0
  reads  A[0..15],  B[0..15]
  writes C[0..15]

Instance 1  (s0 = 1):
  offset = 1 × 64 = 64
  reads  A[16..31], B[16..31]
  writes C[16..31]

Instance 2  (s0 = 2):
  reads  A[32..47], B[32..47]
  writes C[32..47]

Instance 3  (s0 = 3):
  reads  A[48..63], B[48..63]
  writes C[48..63]
```

After `KWAIT`, `C[0..63]` is fully written.

---

## 3.6 Functional vs Cycle Model

```
Functional simulator:
  Instance 0 → runs to KEXIT
  Instance 1 → runs to KEXIT
  Instance 2 → runs to KEXIT
  Instance 3 → runs to KEXIT
  Functional result is always correct regardless of core count.

Cycle model (npu_print_stats):
  Assumes instances spread across physical cores.
  Reported time = max(busiest core), not the sum.
  Set with: npu_set_cores(n, num_cores)
```

---

## 3.7 Softmax Example — Full Kernel

Softmax: `output[i] = exp(input[i]) / sum(exp(input[j]) for all j)`

```
; 16 floats per instance

VSETVL   16

SLW    s2, s1, 0               s2 = input ptr  (from args)
SLW    s3, s1, 8               s3 = output ptr

VLOAD  v0, s2                  load 16 inputs

VEXP   v1, v0                  v1[l] = exp(v0[l])  for all 16 lanes

VREDSUM  s4, v1                s4 = v1[0] + v1[1] + ... + v1[15]

VBCAST v2, s4                  v2[l] = s4  (same value in all lanes)

VDIV   v3, v1, v2              v3[l] = exp(v0[l]) / sum

VSTORE s3, v3

KEXIT
```

The tensor engine cannot do this — `VREDSUM` (16 floats → 1 scalar) followed
by `VDIV` (divide each lane by that scalar) has no tensor-engine equivalent.

---

## 3.8 Predication — Per-Lane Conditionals

The vector core has no branch that splits lanes. Instead, mask registers
control which lanes execute each instruction.

```
; manual ReLU: output[l] = input[l] > 0 ? input[l] : 0

VCMP   m0, v0, v_zero, GT      m0[l] = (v0[l] > 0) ? 1 : 0
VMOV   @m0, v1, v0             v1[l] = v0[l]  only where m0[l] == 1
                                v1[l] unchanged  where m0[l] == 0
```

For full if/else:

```
VCMP      m0, v0, v_zero, GT
VADD   @m0, v1, v0, v_bias     true  branch: add bias  (active lanes only)
VMASK_NOT m1, m0
VMUL   @m1, v1, v0, v_scale    false branch: scale      (inactive lanes only)
```

All lanes always advance through the instruction stream together — the mask
only suppresses the write. This is predicated SIMD (same as AVX-512 `{k1}`).

---

## 3.9 Execution Model — SIMD, not SIMT

The vector core is **SIMD**: one program counter, all 16 lanes execute the
same instruction on different data. No lane has an independent PC.

The grid launch is **SPMD**: N independent instances, each running the full
kernel on its own slice. Not SIMT — no warp, no hardware divergence handling,
no `__syncwarp()`.

| Scope | Model | Analogy |
|---|---|---|
| Within one instance | SIMD | AVX-512, RISC-V V |
| Across instances | SPMD | OpenCL NDRange |

---

## 3.10 Known Gaps

| Gap | Detail |
|---|---|
| No inter-instance communication | Instances cannot read each other's registers. Shared state must go through DRAM/SRAM. |
| Dynamic grid size | `grid_size` in KLAUNCH must be a compile-time constant in LLVM backend. |
| `V_VSCAN` | Defined in `vop_t`, not implemented in simulator. |
| Float-only lanes | Vector registers hold `float`. Integer lanes modeled as float via `VCVT` — not bit-exact for int8. |
| Scalar reg addressing | `s0`–`s31` hold `float`. Addresses stored as float bit patterns reinterpreted as integers — works for 32-bit NPU addresses only. |
