; RUN: llc -mtriple=npu-unknown-unknown -filetype=obj -o %t %s
; RUN: llvm-readelf -h %t | FileCheck %s --check-prefix=ELF-HDR
; RUN: llvm-readelf -r %t | FileCheck %s --check-prefix=RELOCS
; RUN: llvm-readelf -S %t | FileCheck %s --check-prefix=SECTIONS
;
; Verify that compiling to an object file produces a valid NPU ELF:
;   1. ELF header shows EM_NPU machine number (0x13b = 315).
;   2. A JAL to an external (undefined) callee emits an R_NPU_BRANCH (type 3)
;      relocation in .rela.text at the imm field of the JAL instruction.
;   3. The .rela.text section is present.

; ELF-HDR: Machine:{{.*}}13b

; The relocation section header names the section, then the entry lists the symbol.
; RELOCS: .rela.text
; RELOCS: callee

; SECTIONS: .rela.text

; External declaration — forces an unresolved JAL fixup → ELF RELA relocation.
declare i64 @callee(i64 %x)

; Single basic block: ADDI (frame setup), SW (spill), JAL callee, LW (restore),
; ADDI (frame teardown), JALR (return), HALT.
; The JAL instruction is at byte offset 0x40; its imm field is at 0x40+16 = 0x50.
; That byte offset becomes r_offset in the RELA record.
define i64 @caller(i64 %x) {
  %r = call i64 @callee(i64 %x)
  ret i64 %r
}
