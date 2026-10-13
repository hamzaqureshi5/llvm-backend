/* demo_control_core.c — Layer 1 (control core) worked examples.
 *
 * Two self-contained demos, both run with n->trace = 1 so every instruction
 * prints to stderr as it fires. Read the trace alongside the program to see
 * exactly what the control core is doing cycle-by-cycle.
 *
 * Build (from npu-simulator/):
 *   cc -o demo_cc examples/demo_control_core.c src/npu.c src/npu_asm.c \
 *      -I include -lm && ./demo_cc
 *
 * Expected total output:
 *   Demo 1 result: 39   (sum of {3,7,2,9,1,5,4,8})
 *   Demo 2 result: 9    (max of the same array)
 */

#include "npu.h"
#include "npu_asm.h"
#include <stdio.h>
#include <stdint.h>
#include <string.h>

/* ============================================================
 * Demo 1 — Sum an array using a pointer-walking loop
 *
 * This is the simplest realistic program you can write on the
 * control core: iterate over memory, accumulate into a register,
 * store the result, halt.
 *
 * Memory layout (NPU DRAM):
 *   0x0100 : int32_t arr[8] = { 3, 7, 2, 9, 1, 5, 4, 8 }
 *   0x0200 : int32_t result  (written by the program)
 *
 * Register assignment:
 *   x1 = current pointer  (walks 0x0100 → 0x0120)
 *   x2 = end pointer      (0x0120 = 0x0100 + 8*4, never changes)
 *   x3 = accumulator      (sum so far)
 *   x4 = scratch          (element loaded each iteration)
 *
 * Instruction layout (indices matter — they are the branch targets):
 *
 *   [0]  LI   x1, 0x0100      x1  = array base
 *   [1]  LI   x2, 0x0120      x2  = one-past-end (8 * 4 bytes = 32 = 0x20)
 *   [2]  LI   x3, 0           acc = 0
 *
 *      ┌─── loop top (target of BLT at [6])
 *   [3] │ LW   x4, 0(x1)      x4  = *x1  (load 32-bit, sign-extend)
 *   [4] │ ADD  x3, x3, x4     acc += x4
 *   [5] │ ADDI x1, x1, 4      x1 += 4   (advance by one int32_t)
 *   [6] └─BLT  x1, x2, 3      if x1 < x2: jump back to [3]
 *
 *   [7]  SW   x3, 0x0200      mem[0x0200] = acc  (store result)
 *   [8]  HALT
 *
 * Trace walkthrough (abbreviated):
 *   [  0] LI x1, 256              imm=256          → x1=0x100
 *   [  1] LI x2, 288              imm=288          → x2=0x120
 *   [  2] LI x3, 0                imm=0            → x3=0
 *   --- iteration 1 ---
 *   [  3] LW x4, 0(x1)            addr=0x00000100  → x4=3
 *   [  4] ADD x3, x3, x4          x3=0   x4=3      → x3=3
 *   [  5] ADDI x1, x1, 4          x1=256 imm=4     → x1=260
 *   [  6] BLT x1, x2, 3           260 < 288 → taken
 *   --- (iterations 2-7 omitted) ---
 *   --- iteration 8 (last) ---
 *   [  3] LW x4, 0(x1)            addr=0x0000011c  → x4=8
 *   [  4] ADD x3, x3, x4          x3=31  x4=8      → x3=39
 *   [  5] ADDI x1, x1, 4          x1=284 imm=4     → x1=288
 *   [  6] BLT x1, x2, 3           288 < 288 → NOT taken (fall through)
 *   [  7] SW x3, 512              addr=0x00000200   (store 39)
 *   [  8] HALT
 * ============================================================ */
static void demo_array_sum(npu_t *n)
{
    puts("\n=== Demo 1: array sum (pure control core) ===");

    enum { ADDR_ARR = 0x0100, ADDR_OUT = 0x0200, N = 8 };

    int32_t arr[N] = { 3, 7, 2, 9, 1, 5, 4, 8 };  /* expected sum = 39 */
    npu_write_i32(n, ADDR_ARR, arr, N);

    /*
     * Each cinst is a C struct with designated initializers.
     * Fields not named are zero-initialized, which is safe —
     * the simulator only reads the fields relevant to each opcode.
     *
     * Key fields used here:
     *   .op       — opcode (cop_t)
     *   .rd       — destination register index
     *   .rs1/.rs2 — source register indices
     *   .imm      — immediate value OR absolute branch target (instruction index)
     *   .use_base — addressing mode: 1 = x[rs1]+imm, 0 = imm (absolute)
     */
    const cinst prog[] = {
        /* [0] */ { .op = C_LI,   .rd  = 1,                .imm = ADDR_ARR     },
        /* [1] */ { .op = C_LI,   .rd  = 2,                .imm = ADDR_ARR+N*4 },
        /* [2] */ { .op = C_LI,   .rd  = 3,                .imm = 0            },
        /* [3] */ { .op = C_LW,   .rd  = 4, .rs1 = 1,      .imm = 0, .use_base = 1 },
        /* [4] */ { .op = C_ADD,  .rd  = 3, .rs1 = 3, .rs2 = 4                 },
        /* [5] */ { .op = C_ADDI, .rd  = 1, .rs1 = 1,      .imm = 4            },
        /* [6] */ { .op = C_BLT,            .rs1 = 1, .rs2 = 2, .imm = 3       },
        /* [7] */ { .op = C_SW,             .rs1 = 0, .rs2 = 3, .imm = ADDR_OUT },
        /* [8] */ { .op = C_HALT                                                },
    };

    npu_run(n, prog, sizeof prog / sizeof prog[0]);

    int32_t result = 0;
    npu_read_i32(n, ADDR_OUT, &result, 1);
    printf("\n  result = %d  (expected 39) — %s\n",
           result, result == 39 ? "PASS" : "FAIL");
    npu_print_stats(n);
}

/* ============================================================
 * Demo 2 — Find the maximum value in an array
 *
 * Almost the same structure as Demo 1, but the loop body uses
 * a conditional: only update the running max when the current
 * element is greater.
 *
 * On the control core there is no conditional move (CMOV).
 * So the "update if greater" is implemented with a second branch:
 *   SLT  x5, x3, x4    x5 = (x3 < x4) ? 1 : 0
 *   BEQ  x5, x0, skip  if x5 == 0 (i.e. x3 >= x4): skip the update
 *   ADD  x3, x0, x4    x3 = x4    (x0 is hardwired zero, so this copies x4)
 *                                  ← "skip" lands here
 *
 * Same array, same memory layout as Demo 1.
 *
 * Register assignment:
 *   x1 = current pointer
 *   x2 = end pointer
 *   x3 = running max       (initialized to arr[0], then loop starts at arr[1])
 *   x4 = current element
 *   x5 = scratch for SLT
 *
 * Instruction layout:
 *
 *   [0]  LI   x1, 0x0100    ptr  = arr base
 *   [1]  LI   x2, 0x0120    end  = arr+32
 *   [2]  LW   x3, 0(x1)     max  = arr[0]  (prime the max register)
 *   [3]  ADDI x1, x1, 4     advance past the first element
 *
 *      ┌─── loop top
 *   [4] │ LW   x4, 0(x1)    x4   = *ptr
 *   [5] │ SLT  x5, x3, x4   x5   = (max < x4) ? 1 : 0
 *   [6] │ BEQ  x5, x0, 8    if x5==0 (max >= x4): skip to [8]
 *   [7] │ ADD  x3, x0, x4   max  = x4     ← only reached if x4 > max
 *   [8] │ ADDI x1, x1, 4    ptr += 4       ← skip target
 *   [9] └─BLT  x1, x2, 4    if ptr < end: goto [4]
 *
 *   [10] SW   x3, 0x0200    mem[0x0200] = max
 *   [11] HALT
 *
 * Note: BEQ x5, x0, 8 compares x5 against x0 (hardwired zero).
 * This is the idiomatic way to branch on a boolean result from SLT
 * when there is no dedicated branch-if-false instruction.
 * ============================================================ */
static void demo_array_max(npu_t *n)
{
    puts("\n=== Demo 2: array max (control core conditional branch) ===");

    enum { ADDR_ARR = 0x0100, ADDR_OUT = 0x0200, N = 8 };

    int32_t arr[N] = { 3, 7, 2, 9, 1, 5, 4, 8 };  /* expected max = 9 */
    npu_write_i32(n, ADDR_ARR, arr, N);

    const cinst prog[] = {
        /* [0]  */ { .op = C_LI,   .rd = 1,               .imm = ADDR_ARR         },
        /* [1]  */ { .op = C_LI,   .rd = 2,               .imm = ADDR_ARR + N * 4 },
        /* [2]  */ { .op = C_LW,   .rd = 3, .rs1 = 1,     .imm = 0, .use_base = 1 },
        /* [3]  */ { .op = C_ADDI, .rd = 1, .rs1 = 1,     .imm = 4                },

        /* [4]  */ { .op = C_LW,   .rd = 4, .rs1 = 1,     .imm = 0, .use_base = 1 },
        /* [5]  */ { .op = C_SLT,  .rd = 5, .rs1 = 3, .rs2 = 4                    },
        /* [6]  */ { .op = C_BEQ,           .rs1 = 5, .rs2 = 0, .imm = 8          },
        /* [7]  */ { .op = C_ADD,  .rd = 3, .rs1 = 0, .rs2 = 4                    },
        /* [8]  */ { .op = C_ADDI, .rd = 1, .rs1 = 1,     .imm = 4                },
        /* [9]  */ { .op = C_BLT,           .rs1 = 1, .rs2 = 2, .imm = 4          },

        /* [10] */ { .op = C_SW,            .rs1 = 0, .rs2 = 3, .imm = ADDR_OUT   },
        /* [11] */ { .op = C_HALT                                                   },
    };

    npu_run(n, prog, sizeof prog / sizeof prog[0]);

    int32_t result = 0;
    npu_read_i32(n, ADDR_OUT, &result, 1);
    printf("\n  result = %d  (expected 9) — %s\n",
           result, result == 9 ? "PASS" : "FAIL");
    npu_print_stats(n);
}

/* ============================================================ */

int main(void)
{
    puts("NPU control-core demo — trace output goes to stderr");
    puts("Redirect if you want to separate it:  ./demo_cc 2>trace.txt");

    npu_t *n = npu_create();
    n->trace = 1;   /* print every instruction + register state to stderr */

    demo_array_sum(n);

    /*
     * Reset register state between demos so x1-x5 start clean.
     * Memory is NOT cleared — arr written in demo 1 is still at 0x0100.
     */
    npu_reset(n);
    n->trace = 1;

    demo_array_max(n);

    npu_destroy(n);
    return 0;
}
