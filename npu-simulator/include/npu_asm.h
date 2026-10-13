/* npu_asm.h - text assembler + binary encoder/decoder + disassembler for the
 * NPU ISA. These turn instructions between their three forms:
 *
 *   TEXT  "VADD v2, v0, v1"  <--asm/disasm-->  STRUCT {V_VADD,...}  <--enc/dec-->  BYTES
 *
 * The core interpreter (npu.c) runs the STRUCT form. This module lets you reach
 * it from assembly text or from a binary instruction stream (e.g. emitted by a
 * future LLVM backend).
 */
#ifndef NPU_ASM_H
#define NPU_ASM_H

#include "npu.h"

/* Fixed-width binary instruction record. Real silicon would bit-pack a 32-bit
 * word per instruction format; for this teaching ISS we use one fixed record
 * with named byte slots, which is simpler and lossless for the rich structs. */
#define NPU_WORD_BYTES 32

/* ---- Assembler: text -> struct array ---------------------------------- *
 * Parse `src` (newline-separated assembly) into the given array. Returns the
 * instruction count (>=0) on success, or -1 on error with a message written to
 * `err`. Labels (`name:`) are resolved to instruction indices for branches. */
int npu_asm_vector (const char *src, vinst *out, int max_out, char *err, int errsz);
int npu_asm_control(const char *src, cinst *out, int max_out, char *err, int errsz);

/* ---- Encoder / Decoder: struct <-> bytes ------------------------------ *
 * Each returns NPU_WORD_BYTES. Decode reconstructs the struct exactly. */
int npu_encode_vinst(const vinst *in,  unsigned char out[NPU_WORD_BYTES]);
int npu_decode_vinst(const unsigned char in[NPU_WORD_BYTES], vinst *out);
int npu_encode_cinst(const cinst *in,  unsigned char out[NPU_WORD_BYTES]);
int npu_decode_cinst(const unsigned char in[NPU_WORD_BYTES], cinst *out);

/* ---- Disassembler: struct -> text ------------------------------------- */
void npu_disasm_vinst(const vinst *in, char *buf, int bufsz);
void npu_disasm_cinst(const cinst *in, char *buf, int bufsz);

#endif /* NPU_ASM_H */
