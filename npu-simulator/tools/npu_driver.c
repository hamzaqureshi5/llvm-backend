/*
 * npu_driver.c — full-featured NPU simulator driver
 *
 * Supports three input modes:
 *   1. ELF object file  (.o from  llc -mtriple=npu -filetype=obj)
 *   2. Assembly file    (.s from  llc -mtriple=npu  or  hand-written)
 *   3. Inline assembly  (-e "ADD x10, x10, x11\nJALR x0,x1,0\nHALT")
 *
 * Build:
 *   gcc -O2 -o npu_driver npu_driver.c npu.c npu_asm.c -lm
 *
 * Usage:
 *   npu_driver [options] <file.o | file.s> [arg0 arg1 ...]
 *   npu_driver [options] -e 'asm text'    [arg0 arg1 ...]
 *
 * Examples:
 *   npu_driver out.o 3 5
 *   npu_driver -v -r out.s 10 32
 *   npu_driver -e "ADD x10,x10,x11\nJALR x0,x1,0\nHALT" 7 8
 *   npu_driver -d out.o                        # disassemble only
 *   npu_driver -v -s -r -c 4 kernel.o 1024    # verbose, stats, all regs, 4 cores
 *   npu_driver -w 0 42 -m 0 4 memtest.o       # write DRAM, dump after
 */

#include "npu.h"
#include "npu_asm.h"

#include <elf.h>
#include <errno.h>
#include <fcntl.h>
#include <getopt.h>
#include <inttypes.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

/* ── constants ──────────────────────────────────────────────────────────────── */

#define MAX_PROG    4096   /* max instructions per function */
#define MAX_ARGS    8      /* max user arguments (x10..x17) */
#define MAX_WRITES  64     /* max -w ADDR VAL pairs */
#define MAX_DUMPS   16     /* max -m ADDR LEN pairs */

/* ── option state ───────────────────────────────────────────────────────────── */

typedef struct {
    /* Output control */
    int verbose;      /* -v : print decoded instructions before run */
    int disasm_only;  /* -d : decode and print; do not run */
    int dump_regs;    /* -r : print all 32 GPRs after run */
    int print_stats;  /* -s : print cycle/memory statistics */
    int hex_output;   /* -x : print numbers in hex */
    int trace;        /* -t : enable instruction-level trace in simulator */

    /* Execution */
    int cores;        /* -c N : number of vector cores */
    const char *func; /* -f NAME : only run this symbol (ELF mode) */

    /* Inline assembly */
    const char *inline_asm; /* -e 'text' */

    /* Arguments passed to the function (x10..x17) */
    int64_t args[MAX_ARGS];
    int     nargs;

    /* Pre-run DRAM writes: -w ADDR VAL */
    struct { uint32_t addr; int32_t val; } writes[MAX_WRITES];
    int nwrites;

    /* Post-run DRAM dumps: -m ADDR LEN */
    struct { uint32_t addr; int len; } dumps[MAX_DUMPS];
    int ndumps;
} options_t;

/* ── decoded program ────────────────────────────────────────────────────────── */

typedef struct {
    char    name[128];   /* function symbol name */
    cinst  *prog;        /* decoded instruction array */
    int     len;         /* number of instructions */
} function_t;

/* ── output helpers ─────────────────────────────────────────────────────────── */

static void die(const char *fmt, ...)
{
    va_list ap;
    va_start(ap, fmt);
    fprintf(stderr, "npu_driver: error: ");
    vfprintf(stderr, fmt, ap);
    fprintf(stderr, "\n");
    va_end(ap);
    exit(1);
}

static void info(const char *fmt, ...)
{
    va_list ap;
    va_start(ap, fmt);
    vprintf(fmt, ap);
    va_end(ap);
}

/* Print an integer in decimal or hex depending on -x flag. */
static void print_int(const options_t *opt, int64_t v)
{
    if (opt->hex_output)
        printf("0x%llx", (unsigned long long)(uint64_t)v);
    else
        printf("%" PRId64, v);
}

/* ── disassembler ───────────────────────────────────────────────────────────── */

static void disassemble(const function_t *fn)
{
    char buf[128];
    printf("  Disassembly of %s (%d instructions):\n", fn->name, fn->len);
    for (int i = 0; i < fn->len; i++) {
        npu_disasm_cinst(&fn->prog[i], buf, sizeof(buf));
        printf("    [%3d]  %s\n", i, buf);
    }
}

/* ── register dump ──────────────────────────────────────────────────────────── */

static void dump_registers(const npu_t *n, const options_t *opt)
{
    printf("  Registers:\n");
    for (int i = 0; i < NCREG; i++) {
        printf("    x%-2d = ", i);
        print_int(opt, n->x[i]);
        if (i == 1)  printf("  (link)");
        if (i == 10) printf("  (return value)");
        printf("\n");
        /* Skip uninteresting zero regs in non-hex mode to reduce noise. */
        if (!opt->hex_output && !opt->verbose && n->x[i] == 0 &&
            i != 10 && i > 2) {
            /* Check if the rest are all zero. */
            int allzero = 1;
            for (int j = i + 1; j < NCREG; j++)
                if (n->x[j] != 0) { allzero = 0; break; }
            if (allzero) { printf("    ... (x%d–x31 = 0)\n", i + 1); break; }
        }
    }
}

/* ── stats dump ─────────────────────────────────────────────────────────────── */

static void dump_stats(const npu_t *n)
{
    const npu_stats *s = &n->stats;
    printf("  Cycle model:\n");
    printf("    control  core : %llu cycles, %llu instructions retired\n",
           (unsigned long long)s->ctrl_cycles,
           (unsigned long long)s->ctrl_instret);
    printf("    tensor engine : %llu cycles, %llu instructions retired\n",
           (unsigned long long)s->tensor_cycles,
           (unsigned long long)s->tensor_instret);
    printf("    vector engine : %llu cycles, %llu instructions retired\n",
           (unsigned long long)s->vector_cycles,
           (unsigned long long)s->vector_instret);
    if (s->kernels_launched)
        printf("    kernels: %llu launched, %llu instances run\n",
               (unsigned long long)s->kernels_launched,
               (unsigned long long)s->instances_run);
}

/* ── memory dump ────────────────────────────────────────────────────────────── */

static void dump_memory(const npu_t *n, uint32_t addr, int count,
                        const options_t *opt)
{
    printf("  DRAM @ 0x%08x (%d words):\n", addr, count);
    int32_t buf[256];
    if (count > 256) count = 256;
    npu_read_i32((npu_t *)n, addr, buf, count);
    for (int i = 0; i < count; i++) {
        if (i % 8 == 0) printf("    [%4d]  ", i);
        if (opt->hex_output)
            printf("0x%08x ", (unsigned)buf[i]);
        else
            printf("%10d ", buf[i]);
        if (i % 8 == 7 || i == count - 1) printf("\n");
    }
}

/* ── run one function ───────────────────────────────────────────────────────── */

static void run_function(npu_t *n, const function_t *fn, const options_t *opt)
{
    /* Set up calling convention: x10..x17 = args, x1 = HALT index (link). */
    memset(n->x, 0, sizeof(n->x));
    n->x[1] = fn->len - 1;   /* link register → last instruction (HALT) */
    for (int i = 0; i < opt->nargs && i < MAX_ARGS; i++)
        n->x[10 + i] = opt->args[i];

    /* Separator */
    printf("\n%s", "────────────────────────────────────────────────\n");

    /* Print function name and arguments. */
    printf("Running  %s(", fn->name);
    for (int i = 0; i < opt->nargs; i++) {
        if (i) printf(", ");
        print_int(opt, opt->args[i]);
    }
    printf(")\n");

    /* Optional: show decoded instructions. */
    if (opt->verbose || opt->disasm_only)
        disassemble(fn);

    if (opt->disasm_only)
        return;

    /* Run. */
    npu_run(n, fn->prog, fn->len);

    /* Primary result. */
    printf("  x10 = ");
    print_int(opt, n->x[10]);
    printf("\n");

    /* Optional: all registers. */
    if (opt->dump_regs)
        dump_registers(n, opt);

    /* Optional: stats. */
    if (opt->print_stats)
        dump_stats(n);

    /* Optional: memory dumps. */
    for (int i = 0; i < opt->ndumps; i++)
        dump_memory(n, opt->dumps[i].addr, opt->dumps[i].len, opt);
}

/* ═══════════════════════════════════════════════════════════════════════════════
 * ELF LOADER
 *   Reads an ELF64 relocatable object produced by:
 *     llc -mtriple=npu -filetype=obj -o out.o input.ll
 *   Finds all global STT_FUNC symbols, decodes the .text bytes for each
 *   function using npu_decode_cinst(), and returns the list of functions.
 * ═══════════════════════════════════════════════════════════════════════════════ */

typedef struct {
    function_t *funcs;
    int         count;
} elf_program_t;

static const Elf64_Shdr *shdr_at(const uint8_t *elf, const Elf64_Ehdr *eh, int i)
{
    return (const Elf64_Shdr *)(elf + eh->e_shoff + i * eh->e_shentsize);
}

static const char *shdr_name(const uint8_t *elf, const Elf64_Ehdr *eh,
                              const Elf64_Shdr *sh)
{
    const Elf64_Shdr *strtab = shdr_at(elf, eh, eh->e_shstrndx);
    return (const char *)(elf + strtab->sh_offset + sh->sh_name);
}

static elf_program_t load_elf(const char *path, const options_t *opt)
{
    int fd = open(path, O_RDONLY);
    if (fd < 0) die("cannot open '%s': %s", path, strerror(errno));

    struct stat st;
    if (fstat(fd, &st) < 0) die("fstat '%s': %s", path, strerror(errno));

    const uint8_t *elf = mmap(NULL, st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (elf == MAP_FAILED) die("mmap '%s': %s", path, strerror(errno));
    close(fd);

    const Elf64_Ehdr *eh = (const Elf64_Ehdr *)elf;
    if (memcmp(eh->e_ident, ELFMAG, 4) != 0)
        die("'%s' is not an ELF file", path);
    if (eh->e_ident[EI_CLASS] != ELFCLASS64)
        die("'%s': only ELF64 supported", path);

    /* Find .text, .symtab, .strtab sections. */
    const Elf64_Shdr *text_sh = NULL, *sym_sh = NULL, *str_sh = NULL;
    for (int i = 0; i < eh->e_shnum; i++) {
        const Elf64_Shdr *sh = shdr_at(elf, eh, i);
        const char *name = shdr_name(elf, eh, sh);
        if (strcmp(name, ".text") == 0)  text_sh = sh;
        if (sh->sh_type == SHT_SYMTAB)   sym_sh  = sh;
    }
    if (!text_sh) die("'%s': no .text section", path);
    if (!sym_sh)  die("'%s': no .symtab section", path);
    str_sh = shdr_at(elf, eh, sym_sh->sh_link);

    const uint8_t  *text   = elf + text_sh->sh_offset;
    const char     *strtab = (const char *)(elf + str_sh->sh_offset);
    const Elf64_Sym *syms  = (const Elf64_Sym *)(elf + sym_sh->sh_offset);
    int nsyms = (int)(sym_sh->sh_size / sym_sh->sh_entsize);

    /* Count qualifying symbols. */
    int nfuncs = 0;
    for (int si = 0; si < nsyms; si++) {
        const Elf64_Sym *sym = &syms[si];
        if (ELF64_ST_BIND(sym->st_info) != STB_GLOBAL) continue;
        if (ELF64_ST_TYPE(sym->st_info) != STT_FUNC)   continue;
        if (sym->st_size == 0) continue;
        if (opt->func && strcmp(strtab + sym->st_name, opt->func) != 0) continue;
        nfuncs++;
    }

    if (nfuncs == 0) {
        if (opt->func)
            die("'%s': no global function named '%s'", path, opt->func);
        else
            die("'%s': no global STT_FUNC symbols found", path);
    }

    function_t *funcs = calloc(nfuncs, sizeof(function_t));
    if (!funcs) die("out of memory");
    int fi = 0;

    for (int si = 0; si < nsyms; si++) {
        const Elf64_Sym *sym = &syms[si];
        if (ELF64_ST_BIND(sym->st_info) != STB_GLOBAL) continue;
        if (ELF64_ST_TYPE(sym->st_info) != STT_FUNC)   continue;
        if (sym->st_size == 0) continue;

        const char *fname = strtab + sym->st_name;
        if (opt->func && strcmp(fname, opt->func) != 0) continue;

        uint64_t off  = sym->st_value;
        uint64_t size = sym->st_size;

        if (off + size > text_sh->sh_size)
            die("symbol '%s' extends past .text bounds", fname);
        if (size % NPU_WORD_BYTES != 0)
            die("symbol '%s' size %llu not a multiple of %d (corrupt object?)",
                fname, (unsigned long long)size, NPU_WORD_BYTES);

        int len = (int)(size / NPU_WORD_BYTES);
        cinst *prog = malloc(len * sizeof(cinst));
        if (!prog) die("out of memory");

        const uint8_t *p = text + off;
        for (int k = 0; k < len; k++) {
            if (npu_decode_cinst(p + k * NPU_WORD_BYTES, &prog[k]) < 0)
                die("symbol '%s' instruction %d: decode error", fname, k);
        }

        snprintf(funcs[fi].name, sizeof(funcs[fi].name), "%s", fname);
        funcs[fi].prog = prog;
        funcs[fi].len  = len;
        fi++;
    }

    /* Note: elf mapping kept alive; caller owns prog[] arrays. */
    elf_program_t result = { funcs, nfuncs };
    return result;
}

/* ═══════════════════════════════════════════════════════════════════════════════
 * ASSEMBLY FILE LOADER
 *   Reads a .s file produced by:
 *     llc -mtriple=npu -o out.s input.ll
 *   or hand-written in the same mnemonic style.
 *
 *   The file may contain:
 *     - Multiple label-delimited functions (any label followed by instructions)
 *     - Assembler directives (.globl, .type, .size, .text, .section, .p2align,
 *       .file, comments starting with #)
 *
 *   Strategy: strip directives/comments, collect "label: instrs..." blocks, and
 *   assemble each block with npu_asm_control().  Only .globl-declared labels are
 *   treated as function entry points (same semantics as the ELF loader).
 * ═══════════════════════════════════════════════════════════════════════════════ */

typedef struct {
    function_t *funcs;
    int         count;
} asm_program_t;

/* Returns a heap-allocated copy of the file contents, NUL-terminated. */
static char *read_file(const char *path)
{
    FILE *f = fopen(path, "r");
    if (!f) die("cannot open '%s': %s", path, strerror(errno));
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    rewind(f);
    char *buf = malloc(sz + 1);
    if (!buf) die("out of memory");
    if (fread(buf, 1, sz, f) != (size_t)sz) die("read error on '%s'", path);
    buf[sz] = '\0';
    fclose(f);
    return buf;
}

/*
 * Parse a .s source into functions.
 *
 * Approach:
 *   1. Collect names of .globl-declared labels (these are our functions).
 *   2. Walk line-by-line, collecting lines between globl-labels into per-function
 *      assembler text.  Skip directive lines that npu_asm_control cannot handle.
 *   3. Assemble each collected block.
 */
static asm_program_t load_asm_file(const char *path, const options_t *opt)
{
    char *src = read_file(path);
    (void)opt; /* reserved for future filtering */

    /* Pass 1: find all .globl declarations. */
    #define MAX_GLOBALS 64
    char globals[MAX_GLOBALS][128];
    int  nglobals = 0;

    char *line = src;
    while (*line) {
        char *nl = strchr(line, '\n');
        size_t len = nl ? (size_t)(nl - line) : strlen(line);

        /* Strip leading whitespace. */
        char *p = line;
        while (*p == ' ' || *p == '\t') p++;

        if (strncmp(p, ".globl", 6) == 0 && (p[6] == ' ' || p[6] == '\t')) {
            p += 6;
            while (*p == ' ' || *p == '\t') p++;
            /* Extract symbol name up to whitespace or NUL. */
            int nlen = 0;
            while (p[nlen] && p[nlen] != ' ' && p[nlen] != '\t' &&
                   p[nlen] != '\n' && p[nlen] != '\r')
                nlen++;
            if (nlen > 0 && nglobals < MAX_GLOBALS) {
                snprintf(globals[nglobals], sizeof(globals[0]), "%.*s", nlen, p);
                nglobals++;
            }
        }

        line = nl ? nl + 1 : line + len;
    }

    if (nglobals == 0)
        die("'%s': no .globl declarations found — nothing to run.\n"
            "  Add '.globl f' before each function entry label.", path);

    /* Directives to drop (npu_asm_control doesn't understand them). */
    static const char *drop_prefixes[] = {
        ".file", ".text", ".globl", ".type", ".p2align", ".align",
        ".size", ".section", ".note", "#", NULL
    };

    /* Pass 2: collect per-function assembler text blocks. */
    function_t *funcs = calloc(nglobals, sizeof(function_t));
    if (!funcs) die("out of memory");
    int nfuncs = 0;

    /* We'll build a buffer for the current function's text. */
    char *func_buf = malloc(512 * 1024);
    if (!func_buf) die("out of memory");
    int cur_func = -1;   /* index into funcs[] */
    int func_buf_pos = 0;

    /* flush_func: assemble accumulated text for the current function */
    #define flush_func() do { \
        if (cur_func >= 0 && func_buf_pos > 0) { \
            func_buf[func_buf_pos] = '\0'; \
            char _err[256] = {0}; \
            cinst *_prog = malloc(MAX_PROG * sizeof(cinst)); \
            if (!_prog) die("out of memory"); \
            int _len = npu_asm_control(func_buf, _prog, MAX_PROG, _err, sizeof(_err)); \
            if (_len < 0) \
                die("assembly error in function '%s':\n  %s\n\nSource:\n%s", \
                    funcs[cur_func].name, _err, func_buf); \
            funcs[cur_func].prog = _prog; \
            funcs[cur_func].len  = _len; \
        } \
    } while (0)

    line = src;
    while (*line) {
        char *nl = strchr(line, '\n');
        size_t len = nl ? (size_t)(nl - line) : strlen(line);
        char linebuf[512];
        size_t copy = len < sizeof(linebuf) - 1 ? len : sizeof(linebuf) - 1;
        memcpy(linebuf, line, copy);
        linebuf[copy] = '\0';

        /* Strip trailing whitespace / carriage returns. */
        int ll = (int)strlen(linebuf);
        while (ll > 0 && (linebuf[ll-1] == ' ' || linebuf[ll-1] == '\t' ||
                           linebuf[ll-1] == '\r'))
            linebuf[--ll] = '\0';

        /* Leading whitespace pointer. */
        char *p = linebuf;
        while (*p == ' ' || *p == '\t') p++;

        /* Skip blank lines. */
        if (*p == '\0') {
            line = nl ? nl + 1 : line + len;
            continue;
        }

        /* Drop known directives. */
        int skip = 0;
        for (int i = 0; drop_prefixes[i]; i++) {
            if (strncmp(p, drop_prefixes[i], strlen(drop_prefixes[i])) == 0) {
                skip = 1; break;
            }
        }

        if (!skip) {
            /* Check if line is a label definition "name:" (may have trailing
             * whitespace or a # comment, e.g. "f:  # @f" from llc output). */
            char *colon = strchr(p, ':');
            int is_label_line = 0;
            if (colon) {
                char *after = colon + 1;
                while (*after == ' ' || *after == '\t') after++;
                if (*after == '\0' || *after == '#') is_label_line = 1;
            }
            if (is_label_line) {
                /* Looks like a label. */
                char label[128];
                int lablen = (int)(colon - p);
                if (lablen > 0 && lablen < (int)sizeof(label)) {
                    snprintf(label, sizeof(label), "%.*s", lablen, p);

                    /* Is it a globl label (function entry)? */
                    int is_globl = 0;
                    for (int gi = 0; gi < nglobals; gi++)
                        if (strcmp(globals[gi], label) == 0) { is_globl = 1; break; }

                    if (is_globl) {
                        /* Flush previous function, start new. */
                        flush_func();
                        cur_func = nfuncs;
                        snprintf(funcs[nfuncs].name, 128, "%s", label);
                        nfuncs++;
                        func_buf_pos = 0;
                    }
                    /* Always emit the label into the assembler buffer
                     * (needed for local branch targets). */
                    if (cur_func >= 0) {
                        func_buf_pos += snprintf(func_buf + func_buf_pos,
                                                 512*1024 - func_buf_pos,
                                                 "%s:\n", label);
                    }
                }
            } else {
                /* Regular instruction line — pass to assembler. */
                if (cur_func >= 0) {
                    func_buf_pos += snprintf(func_buf + func_buf_pos,
                                             512*1024 - func_buf_pos,
                                             "%s\n", linebuf);
                }
            }
        }

        line = nl ? nl + 1 : line + len;
    }

    /* Flush the last function. */
    flush_func();
    #undef flush_func

    free(func_buf);
    free(src);

    if (nfuncs == 0)
        die("'%s': no functions assembled successfully", path);

    asm_program_t result = { funcs, nfuncs };
    return result;
}

/* ═══════════════════════════════════════════════════════════════════════════════
 * INLINE ASSEMBLY LOADER
 *   For -e "asm text" — assemble the given string as a single anonymous function.
 * ═══════════════════════════════════════════════════════════════════════════════ */

static function_t load_inline_asm(const char *src)
{
    cinst *prog = malloc(MAX_PROG * sizeof(cinst));
    if (!prog) die("out of memory");

    char err[512] = {0};
    /* Expand literal \n in the string so users can pass multi-line programs. */
    char *expanded = malloc(strlen(src) * 2 + 1);
    if (!expanded) die("out of memory");
    int ei = 0;
    for (int i = 0; src[i]; i++) {
        if (src[i] == '\\' && src[i+1] == 'n') {
            expanded[ei++] = '\n'; i++;
        } else {
            expanded[ei++] = src[i];
        }
    }
    expanded[ei] = '\0';

    int len = npu_asm_control(expanded, prog, MAX_PROG, err, sizeof(err));
    free(expanded);
    if (len < 0) die("inline assembly error: %s", err);

    function_t fn;
    strncpy(fn.name, "<inline>", sizeof(fn.name));
    fn.prog = prog;
    fn.len  = len;
    return fn;
}

/* ═══════════════════════════════════════════════════════════════════════════════
 * ARGUMENT PARSING
 * ═══════════════════════════════════════════════════════════════════════════════ */

static void usage(const char *prog, int full)
{
    printf("Usage: %s [options] <file.o | file.s> [arg0 arg1 ...]\n", prog);
    printf("       %s [options] -e 'asm text'    [arg0 arg1 ...]\n", prog);
    if (!full) { printf("  -h for full help\n"); return; }
    printf("\n");
    printf("Input:\n");
    printf("  file.o      ELF object (llc -mtriple=npu -filetype=obj)\n");
    printf("  file.s      Assembly source (llc -mtriple=npu or hand-written)\n");
    printf("  -e 'text'   Inline assembly string; use \\n between instructions\n");
    printf("\n");
    printf("Arguments (passed to function as x10, x11, ...):\n");
    printf("  arg0 ...    Integer values; 0x prefix for hex. Up to %d args.\n", MAX_ARGS);
    printf("\n");
    printf("Output options:\n");
    printf("  -v          Verbose: show decoded instructions before running\n");
    printf("  -d          Disassemble only; do not run\n");
    printf("  -r          Dump all 32 GPRs after run (default: x10 only)\n");
    printf("  -s          Print cycle statistics after run\n");
    printf("  -x          Print numbers in hex (default: decimal)\n");
    printf("  -t          Trace: print each instruction as the simulator executes it\n");
    printf("\n");
    printf("Execution options:\n");
    printf("  -c N        Number of vector cores (default: 1)\n");
    printf("  -f NAME     Run only the function named NAME (ELF mode)\n");
    printf("\n");
    printf("Memory options (applied before run):\n");
    printf("  -w ADDR VAL Write i32 value VAL to DRAM at byte address ADDR\n");
    printf("              (repeat for multiple writes; use 0x prefix for hex)\n");
    printf("Memory options (applied after run):\n");
    printf("  -m ADDR LEN Dump LEN i32 words from DRAM at byte address ADDR\n");
    printf("              (repeat for multiple dumps)\n");
    printf("\n");
    printf("Examples:\n");
    printf("  # Run compiled object, pass two args, dump all registers:\n");
    printf("  %s -r out.o 3 5\n", prog);
    printf("\n");
    printf("  # Show assembly, then run and print stats:\n");
    printf("  %s -v -s out.s 10 32\n", prog);
    printf("\n");
    printf("  # Inline assembly — add two registers:\n");
    printf("  %s -e \"ADD x10,x10,x11\\nJALR x0,x1,0\\nHALT\" 7 8\n", prog);
    printf("\n");
    printf("  # Write 99 to address 0, run, then dump 4 words from address 0:\n");
    printf("  %s -w 0 99 -m 0 4 memtest.o\n", prog);
    printf("\n");
    printf("  # Disassemble only (no simulation):\n");
    printf("  %s -d out.o\n", prog);
    printf("\n");
}

static int64_t parse_int(const char *s)
{
    char *end;
    long long v = strtoll(s, &end, 0);
    if (*end) die("'%s' is not a valid integer", s);
    return (int64_t)v;
}

static uint32_t parse_addr(const char *s)
{
    char *end;
    unsigned long v = strtoul(s, &end, 0);
    if (*end) die("'%s' is not a valid address", s);
    return (uint32_t)v;
}

static options_t parse_options(int argc, char **argv, const char **file_out)
{
    options_t opt = {0};
    opt.cores = 1;

    static const struct option long_opts[] = {
        {"verbose",  no_argument,       NULL, 'v'},
        {"disasm",   no_argument,       NULL, 'd'},
        {"regs",     no_argument,       NULL, 'r'},
        {"stats",    no_argument,       NULL, 's'},
        {"hex",      no_argument,       NULL, 'x'},
        {"trace",    no_argument,       NULL, 't'},
        {"help",     no_argument,       NULL, 'h'},
        {"cores",    required_argument, NULL, 'c'},
        {"func",     required_argument, NULL, 'f'},
        {"eval",     required_argument, NULL, 'e'},
        {"write",    required_argument, NULL, 'w'},
        {"memdump",  required_argument, NULL, 'm'},
        {NULL, 0, NULL, 0}
    };

    int ch;
    while ((ch = getopt_long(argc, argv, "vdrshtxc:f:e:w:m:", long_opts, NULL)) != -1) {
        switch (ch) {
        case 'v': opt.verbose    = 1; break;
        case 'd': opt.disasm_only= 1; break;
        case 'r': opt.dump_regs  = 1; break;
        case 's': opt.print_stats= 1; break;
        case 'x': opt.hex_output = 1; break;
        case 't': opt.trace      = 1; break;
        case 'h': usage(argv[0], 1); exit(0);
        case 'c':
            opt.cores = (int)strtol(optarg, NULL, 0);
            if (opt.cores < 1 || opt.cores > 256)
                die("-c: core count must be 1–256");
            break;
        case 'f':
            opt.func = optarg;
            break;
        case 'e':
            opt.inline_asm = optarg;
            break;
        case 'w': {
            if (opt.nwrites >= MAX_WRITES) die("too many -w options");
            /* -w ADDR VAL : two separate argv values */
            if (optind >= argc) die("-w requires two arguments: ADDR VAL");
            opt.writes[opt.nwrites].addr = parse_addr(optarg);
            opt.writes[opt.nwrites].val  = (int32_t)parse_int(argv[optind++]);
            opt.nwrites++;
            break;
        }
        case 'm': {
            if (opt.ndumps >= MAX_DUMPS) die("too many -m options");
            if (optind >= argc) die("-m requires two arguments: ADDR LEN");
            opt.dumps[opt.ndumps].addr = parse_addr(optarg);
            opt.dumps[opt.ndumps].len  = (int)parse_int(argv[optind++]);
            opt.ndumps++;
            break;
        }
        default:
            usage(argv[0], 0);
            exit(1);
        }
    }

    /* Remaining positional arguments: optional file path, then integer args. */
    if (!opt.inline_asm) {
        if (optind >= argc) {
            fprintf(stderr, "npu_driver: no input file specified\n");
            usage(argv[0], 0);
            exit(1);
        }
        *file_out = argv[optind++];
    }

    while (optind < argc && opt.nargs < MAX_ARGS)
        opt.args[opt.nargs++] = parse_int(argv[optind++]);

    return opt;
}

/* ═══════════════════════════════════════════════════════════════════════════════
 * MAIN
 * ═══════════════════════════════════════════════════════════════════════════════ */

int main(int argc, char **argv)
{
    const char *file = NULL;
    options_t opt = parse_options(argc, argv, &file);

    /* Determine input mode. */
    function_t  *funcs = NULL;
    int          nfuncs = 0;
    function_t   inline_fn;   /* used only for -e mode */

    if (opt.inline_asm) {
        inline_fn = load_inline_asm(opt.inline_asm);
        funcs  = &inline_fn;
        nfuncs = 1;
    } else {
        /* Detect file type by extension. */
        const char *ext = strrchr(file, '.');
        if (!ext) die("'%s': cannot determine file type (need .o or .s extension)", file);

        if (strcmp(ext, ".o") == 0) {
            elf_program_t ep = load_elf(file, &opt);
            funcs  = ep.funcs;
            nfuncs = ep.count;
        } else if (strcmp(ext, ".s") == 0) {
            asm_program_t ap = load_asm_file(file, &opt);
            funcs  = ap.funcs;
            nfuncs = ap.count;
        } else {
            die("'%s': unknown extension '%s' (expected .o or .s)", file, ext);
        }
    }

    /* Create the simulator. */
    npu_t *n = npu_create();
    npu_set_cores(n, opt.cores);
    if (opt.trace) n->trace = 1;

    /* Apply pre-run DRAM writes. */
    for (int i = 0; i < opt.nwrites; i++) {
        npu_write_i32(n, opt.writes[i].addr, &opt.writes[i].val, 1);
        if (opt.verbose)
            printf("  DRAM[0x%08x] <- %d\n", opt.writes[i].addr, opt.writes[i].val);
    }

    /* Print banner. */
    if (file)
        printf("npu_driver: %s", file);
    else
        printf("npu_driver: <inline>");
    if (opt.cores > 1) printf("  [%d cores]", opt.cores);
    printf("  (%d function%s)\n", nfuncs, nfuncs == 1 ? "" : "s");

    /* Run each function. */
    for (int i = 0; i < nfuncs; i++) {
        /* Reset stats between functions so each gets its own numbers. */
        memset(&n->stats, 0, sizeof(n->stats));
        run_function(n, &funcs[i], &opt);
    }

    printf("%s", "────────────────────────────────────────────────\n");

    /* Cleanup. */
    for (int i = 0; i < nfuncs; i++)
        free(funcs[i].prog);
    if (!opt.inline_asm) free(funcs);
    npu_destroy(n);
    return 0;
}
