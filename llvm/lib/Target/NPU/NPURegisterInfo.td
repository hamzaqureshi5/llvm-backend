//===- NPURegisterInfo.td - NPU Register defs --------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//  Declarations that describe the NPU control-core register file.
//
//  The control core has 32 general-purpose registers x0..x31, each 64-bit.
//  x0 is hardwired to zero (reads return 0, writes are discarded), matching
//  the simulator (npu.c getx/setx).
//===----------------------------------------------------------------------===//

class NPUReg<bits<5> num, string n> : Register<n> {
  field bits<5> Num;
  let Num = num;
  let Namespace = "NPU";
}

// Integer registers x0..x31.
foreach i = 0-31 in {
  def X#i : NPUReg<i, "x"#i>, DwarfRegNum<[i]>;
}

// General-purpose register class. Allocation order keeps the
// caller-saved/argument registers first, then callee-saved, and finally the
// reserved registers (x0 = zero, x1 = return address, x2 = stack pointer),
// which getReservedRegs() removes from allocation.
def GPR : RegisterClass<"NPU", [i64], 64,
    (add (sequence "X%i", 10, 17),   // arguments / return values / caller-saved
         (sequence "X%i", 5, 9),     // caller-saved temporaries
         (sequence "X%i", 28, 31),   // caller-saved temporaries
         (sequence "X%i", 18, 27),   // callee-saved
         X3, X4,                     // reserved-for-future / caller-saved
         X1,                         // return address
         X2,                         // stack pointer
         X0                          // constant zero
    )>;
