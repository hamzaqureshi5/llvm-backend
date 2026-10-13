/*
 * npu_elf_run.c — load a relocatable NPU ELF object file and run each
 * exported function in the NPU simulator.
 *
 * Workflow:
 *   llc -mtriple=npu -filetype=obj -o foo.o foo.ll
 *   ./build/npu_elf_run foo.o [arg0 [arg1 ...]]
 *
 * The loader:
 *   1. Reads the ELF64 .text section (raw 32-byte NPU instruction words).
 *   2. Uses npu_decode_cinst() to turn each word into a cinst struct.
 *   3. Calls npu_run() for each global function symbol.
 *
 * Calling convention (CC_NPU from the LLVM backend):
 *   Arguments  : x10, x11, x12 … (integer)
 *   Return val : x10
 *   Link reg   : x1  (set to the HALT instruction index so JALR returns cleanly)
 */

#include "npu.h"
#include "npu_asm.h"

#include <elf.h>
#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

/* ── helpers ───────────────────────────────────────────────────────────────── */

static void die(const char *msg) {
    perror(msg);
    exit(1);
}

/* Section header accessor */
static const Elf64_Shdr *shdr_at(const uint8_t *elf, const Elf64_Ehdr *eh,
                                  int i) {
    return (const Elf64_Shdr *)(elf + eh->e_shoff + i * eh->e_shentsize);
}

/* Section name lookup via shstrtab */
static const char *shdr_name(const uint8_t *elf, const Elf64_Ehdr *eh,
                              const Elf64_Shdr *sh) {
    const Elf64_Shdr *strtab = shdr_at(elf, eh, eh->e_shstrndx);
    return (const char *)(elf + strtab->sh_offset + sh->sh_name);
}

/* ── main ──────────────────────────────────────────────────────────────────── */

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: npu_elf_run <file.o> [arg0 arg1 ...]\n");
        return 1;
    }

    /* ── mmap the ELF file ───────────────────────────────────────────────── */
    int fd = open(argv[1], O_RDONLY);
    if (fd < 0) die("open");
    struct stat st;
    if (fstat(fd, &st) < 0) die("fstat");
    const uint8_t *elf = mmap(NULL, st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (elf == MAP_FAILED) die("mmap");
    close(fd);

    const Elf64_Ehdr *eh = (const Elf64_Ehdr *)elf;
    if (memcmp(eh->e_ident, ELFMAG, 4) != 0) {
        fprintf(stderr, "not an ELF file\n");
        return 1;
    }
    if (eh->e_ident[EI_CLASS] != ELFCLASS64) {
        fprintf(stderr, "only ELF64 supported\n");
        return 1;
    }

    /* ── locate .text, .symtab, .strtab, .rela.text ─────────────────────── */
    const Elf64_Shdr *text_sh = NULL, *sym_sh = NULL, *str_sh = NULL;
    const Elf64_Shdr *rela_sh = NULL;
    for (int i = 0; i < eh->e_shnum; i++) {
        const Elf64_Shdr *sh = shdr_at(elf, eh, i);
        const char *name = shdr_name(elf, eh, sh);
        if (strcmp(name, ".text") == 0)      text_sh = sh;
        if (sh->sh_type == SHT_SYMTAB)       sym_sh  = sh;
        if (strcmp(name, ".rela.text") == 0) rela_sh = sh;
    }
    if (!text_sh) { fprintf(stderr, "no .text section\n"); return 1; }
    if (!sym_sh)  { fprintf(stderr, "no .symtab section\n"); return 1; }
    str_sh = shdr_at(elf, eh, sym_sh->sh_link);

    /* mutable copy of .text — relocations are patched in before decoding */
    uint8_t *text = malloc(text_sh->sh_size);
    if (!text) die("malloc text");
    memcpy(text, elf + text_sh->sh_offset, text_sh->sh_size);

    /* ── apply R_NPU_BRANCH (type 3) relocations ─────────────────────────
     * Formula: (S + A) / NPU_WORD_BYTES → word index stored at byte 16
     * within the 32-byte instruction (the imm field offset).              */
    if (rela_sh) {
        const Elf64_Rela *relas =
            (const Elf64_Rela *)(elf + rela_sh->sh_offset);
        int nrelas = (int)(rela_sh->sh_size / sizeof(Elf64_Rela));
        const Elf64_Sym *rsyms =
            (const Elf64_Sym *)(elf + shdr_at(elf, eh,
                                              rela_sh->sh_link)->sh_offset);
        for (int i = 0; i < nrelas; i++) {
            uint32_t type    = ELF64_R_TYPE(relas[i].r_info);
            uint32_t sym_idx = ELF64_R_SYM(relas[i].r_info);
            int64_t  S       = (int64_t)rsyms[sym_idx].st_value;
            int64_t  A       = relas[i].r_addend;
            uint64_t roff    = relas[i].r_offset;
            if (type == 3 /* R_NPU_BRANCH */ && roff + 4 <= text_sh->sh_size) {
                int32_t target = (int32_t)((S + A) / NPU_WORD_BYTES);
                memcpy(text + roff, &target, 4);
            }
        }
    }

    const char    *strtab = (const char *)(elf + str_sh->sh_offset);
    const Elf64_Sym *syms = (const Elf64_Sym *)(elf + sym_sh->sh_offset);
    int nsyms = sym_sh->sh_size / sym_sh->sh_entsize;

    /* ── parse optional integer arguments ───────────────────────────────── */
    int64_t user_args[8] = {0};
    int nargs = 0;
    for (int i = 2; i < argc && nargs < 8; i++)
        user_args[nargs++] = (int64_t)strtoll(argv[i], NULL, 0);

    /* ── run each global STT_FUNC symbol ─────────────────────────────────── */
    npu_t *npu = npu_create();

    for (int si = 0; si < nsyms; si++) {
        const Elf64_Sym *sym = &syms[si];
        if (ELF64_ST_BIND(sym->st_info) != STB_GLOBAL) continue;
        if (ELF64_ST_TYPE(sym->st_info) != STT_FUNC)   continue;
        if (sym->st_size == 0) continue;

        const char *fname = strtab + sym->st_name;
        uint64_t   off    = sym->st_value;    /* byte offset in .text */
        uint64_t   size   = sym->st_size;     /* byte size            */

        if (off + size > text_sh->sh_size) {
            fprintf(stderr, "symbol '%s' out of .text bounds\n", fname);
            continue;
        }
        if (size % NPU_WORD_BYTES != 0) {
            fprintf(stderr, "symbol '%s' size %llu not a multiple of %d\n",
                    fname, (unsigned long long)size, NPU_WORD_BYTES);
            continue;
        }

        int ninstr = (int)(size / NPU_WORD_BYTES);
        cinst *prog = malloc(ninstr * sizeof(cinst));
        if (!prog) die("malloc");

        /* Decode each 32-byte word into a cinst struct */
        const uint8_t *p = text + off;
        for (int k = 0; k < ninstr; k++) {
            if (npu_decode_cinst(p + k * NPU_WORD_BYTES, &prog[k]) < 0) {
                fprintf(stderr, "'%s' instr %d: decode error\n", fname, k);
                free(prog);
                goto next;
            }
        }

        /* Reset simulator registers and set up the call */
        memset(npu->x, 0, sizeof(npu->x));
        /* x1 = link register = index of HALT (last instruction) so that
         * "JALR x0, x1, 0" (the generated return sequence) jumps to HALT. */
        npu->x[1] = ninstr - 1;

        /* Load arguments into x10, x11, x12 … (CC_NPU convention) */
        for (int k = 0; k < nargs && k < 8; k++)
            npu->x[10 + k] = user_args[k];

        printf("running  %s(", fname);
        for (int k = 0; k < nargs; k++)
            printf("%s%lld", k ? ", " : "", (long long)user_args[k]);
        printf(")\n");

        npu_run(npu, prog, ninstr);

        printf("  -> x10 = %lld  (0x%llx)\n",
               (long long)npu->x[10], (unsigned long long)(uint64_t)npu->x[10]);
        free(prog);
    next:;
    }

    npu_destroy(npu);
    free(text);
    munmap((void *)elf, st.st_size);
    return 0;
}
