/*
 * npu_runtime.c — CUDA-style runtime library for the NPU simulator.
 *
 * Implements npu_runtime.h.  This file is self-contained; link with:
 *   gcc ... npu_runtime.c npu.c npu_asm.c -lm
 */

#include "npu_runtime.h"
#include "npu.h"
#include "npu_asm.h"

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <stdint.h>
#include <assert.h>
#include <ctype.h>
#include <elf.h>       /* Elf64_* types, STB_GLOBAL, STT_FUNC, SHT_* */
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

/* ════════════════════════════════════════════════════════════════════════════
 * Internal limits
 * ════════════════════════════════════════════════════════════════════════════ */

#define MAX_HEAP_BLOCKS    2048  /* host-side allocation table entries        */
#define MAX_MODULES        64    /* modules per context                       */
#define MAX_FNS_PER_MOD    128   /* functions per module                      */
#define MAX_FN_INSTS       4096  /* max instructions per function             */
#define HEAP_ALIGN         64    /* allocation alignment (bytes)              */
#define HEAP_GUARD         64    /* first 64 bytes reserved so 0 == null      */

/* ════════════════════════════════════════════════════════════════════════════
 * Internal structs
 * ════════════════════════════════════════════════════════════════════════════ */

typedef struct {
    uint32_t base;   /* byte address in DRAM                           */
    uint32_t size;   /* size in bytes (always a multiple of HEAP_ALIGN) */
    int      used;   /* 1 = allocated, 0 = free                        */
} heap_block_t;

struct npu_fn_s {
    char   name[128];
    cinst *prog;
    int    len;
};

struct npu_module_s {
    struct npu_fn_s fns[MAX_FNS_PER_MOD];
    int             nfns;
    npu_ctx_t       owner;   /* back-pointer so unload can deregister itself */
};

struct npu_ctx_s {
    npu_t       *npu;
    npu_error_t  last_err;
    int64_t      retval;     /* x10 after npu_run()                     */
    /* heap */
    heap_block_t heap[MAX_HEAP_BLOCKS];
    int          nblocks;
    /* module table (weak refs; ownership is in the npu_module structs) */
    npu_module_t mods[MAX_MODULES];
    int          nmods;
};

/* ════════════════════════════════════════════════════════════════════════════
 * Error helpers
 * ════════════════════════════════════════════════════════════════════════════ */

const char *npu_strerror(npu_error_t err)
{
    switch (err) {
    case NPU_SUCCESS:            return "success";
    case NPU_ERROR_INVALID_CTX:  return "invalid context";
    case NPU_ERROR_INVALID_ARG:  return "invalid argument";
    case NPU_ERROR_OUT_OF_MEM:   return "out of device memory";
    case NPU_ERROR_INVALID_ADDR: return "invalid device address";
    case NPU_ERROR_BAD_MODULE:   return "bad module (ELF or ASM parse error)";
    case NPU_ERROR_NOT_FOUND:    return "function not found in module";
    case NPU_ERROR_LAUNCH:       return "kernel launch failed";
    case NPU_ERROR_IO:           return "I/O error";
    case NPU_ERROR_TOO_MANY:     return "too many modules or allocations";
    default:                     return "unknown error";
    }
}

static npu_error_t ctx_err(npu_ctx_t ctx, npu_error_t e)
{
    if (ctx) ctx->last_err = e;
    return e;
}

/* ════════════════════════════════════════════════════════════════════════════
 * Heap allocator
 *
 * A sorted array of (base, size, used) blocks covering DRAM.
 * First-fit with coalescing free.  All addresses are DRAM byte offsets.
 * ════════════════════════════════════════════════════════════════════════════ */

static void heap_init(npu_ctx_t ctx)
{
    /* One free block covering all available DRAM (after the guard area). */
    ctx->heap[0].base = HEAP_GUARD;
    ctx->heap[0].size = NPU_MEM_SIZE - HEAP_GUARD;
    ctx->heap[0].used = 0;
    ctx->nblocks      = 1;
}

static uint32_t align_up(uint32_t v, uint32_t a)
{
    return (v + a - 1) & ~(a - 1);
}

static npu_error_t heap_alloc(npu_ctx_t ctx, size_t bytes, npu_devptr_t *out)
{
    uint32_t need = align_up((uint32_t)bytes, HEAP_ALIGN);
    if (need == 0) need = HEAP_ALIGN;

    for (int i = 0; i < ctx->nblocks; i++) {
        heap_block_t *b = &ctx->heap[i];
        if (b->used || b->size < need)
            continue;

        /* Found a fit.  Split if the remainder is worth keeping. */
        if (b->size - need >= HEAP_ALIGN * 2 && ctx->nblocks < MAX_HEAP_BLOCKS) {
            /* Insert a new free block after this one. */
            int tail = ctx->nblocks;
            memmove(&ctx->heap[i + 2], &ctx->heap[i + 1],
                    (tail - i - 1) * sizeof(heap_block_t));
            ctx->nblocks++;
            ctx->heap[i + 1].base = b->base + need;
            ctx->heap[i + 1].size = b->size - need;
            ctx->heap[i + 1].used = 0;
            b->size = need;
        }
        b->used = 1;
        *out = b->base;
        return NPU_SUCCESS;
    }
    return ctx_err(ctx, NPU_ERROR_OUT_OF_MEM);
}

static npu_error_t heap_free(npu_ctx_t ctx, npu_devptr_t ptr)
{
    /* Find the block. */
    int found = -1;
    for (int i = 0; i < ctx->nblocks; i++) {
        if (ctx->heap[i].base == ptr && ctx->heap[i].used) {
            found = i; break;
        }
    }
    if (found < 0)
        return ctx_err(ctx, NPU_ERROR_INVALID_ADDR);

    ctx->heap[found].used = 0;

    /* Coalesce with next free block. */
    if (found + 1 < ctx->nblocks && !ctx->heap[found + 1].used) {
        ctx->heap[found].size += ctx->heap[found + 1].size;
        memmove(&ctx->heap[found + 1], &ctx->heap[found + 2],
                (ctx->nblocks - found - 2) * sizeof(heap_block_t));
        ctx->nblocks--;
    }
    /* Coalesce with previous free block. */
    if (found > 0 && !ctx->heap[found - 1].used) {
        ctx->heap[found - 1].size += ctx->heap[found].size;
        memmove(&ctx->heap[found], &ctx->heap[found + 1],
                (ctx->nblocks - found - 1) * sizeof(heap_block_t));
        ctx->nblocks--;
    }
    return NPU_SUCCESS;
}

/* ════════════════════════════════════════════════════════════════════════════
 * Device
 * ════════════════════════════════════════════════════════════════════════════ */

void npu_get_props(npu_props_t *props)
{
    if (!props) return;
    memset(props, 0, sizeof(*props));
    snprintf(props->name, sizeof(props->name), "NPU Simulator");
    props->dram_bytes    = NPU_MEM_SIZE;
    props->scratch_bytes = NPU_SCRATCH_SIZE;
    props->max_cores     = NPU_MAX_CORES;
    props->max_kernels   = MAX_KERNELS;
}

/* ════════════════════════════════════════════════════════════════════════════
 * Context
 * ════════════════════════════════════════════════════════════════════════════ */

npu_error_t npu_ctx_create(npu_ctx_t *out)
{
    if (!out) return NPU_ERROR_INVALID_ARG;
    npu_ctx_t ctx = calloc(1, sizeof(struct npu_ctx_s));
    if (!ctx) return NPU_ERROR_OUT_OF_MEM;
    ctx->npu = npu_create();
    if (!ctx->npu) { free(ctx); return NPU_ERROR_OUT_OF_MEM; }
    heap_init(ctx);
    *out = ctx;
    return NPU_SUCCESS;
}

npu_error_t npu_ctx_destroy(npu_ctx_t ctx)
{
    if (!ctx) return NPU_ERROR_INVALID_CTX;
    while (ctx->nmods > 0)
        npu_module_unload(ctx->mods[ctx->nmods - 1]);
    npu_destroy(ctx->npu);
    free(ctx);
    return NPU_SUCCESS;
}

npu_error_t npu_ctx_set_cores(npu_ctx_t ctx, int cores)
{
    if (!ctx) return NPU_ERROR_INVALID_CTX;
    npu_set_cores(ctx->npu, cores);
    return NPU_SUCCESS;
}

npu_error_t npu_ctx_set_trace(npu_ctx_t ctx, int enable)
{
    if (!ctx) return NPU_ERROR_INVALID_CTX;
    ctx->npu->trace = enable ? 1 : 0;
    return NPU_SUCCESS;
}

npu_error_t npu_ctx_reset_stats(npu_ctx_t ctx)
{
    if (!ctx) return NPU_ERROR_INVALID_CTX;
    memset(&ctx->npu->stats, 0, sizeof(ctx->npu->stats));
    return NPU_SUCCESS;
}

npu_error_t npu_ctx_print_stats(npu_ctx_t ctx)
{
    if (!ctx) return NPU_ERROR_INVALID_CTX;
    npu_print_stats(ctx->npu);
    return NPU_SUCCESS;
}

npu_error_t npu_ctx_last_error(npu_ctx_t ctx)
{
    if (!ctx) return NPU_ERROR_INVALID_CTX;
    return ctx->last_err;
}

int64_t npu_ctx_retval(npu_ctx_t ctx)
{
    if (!ctx) return 0;
    return ctx->retval;
}

/* ════════════════════════════════════════════════════════════════════════════
 * Device memory
 * ════════════════════════════════════════════════════════════════════════════ */

npu_error_t npu_malloc(npu_ctx_t ctx, npu_devptr_t *ptr, size_t bytes)
{
    if (!ctx)        return NPU_ERROR_INVALID_CTX;
    if (!ptr || !bytes) return ctx_err(ctx, NPU_ERROR_INVALID_ARG);
    return heap_alloc(ctx, bytes, ptr);
}

npu_error_t npu_free(npu_ctx_t ctx, npu_devptr_t ptr)
{
    if (!ctx) return NPU_ERROR_INVALID_CTX;
    if (!ptr) return ctx_err(ctx, NPU_ERROR_INVALID_ARG);
    return heap_free(ctx, ptr);
}

npu_error_t npu_memcpy_h2d(npu_ctx_t ctx, npu_devptr_t dst,
                             const void *src, size_t bytes)
{
    if (!ctx)               return NPU_ERROR_INVALID_CTX;
    if (!src || !bytes)     return ctx_err(ctx, NPU_ERROR_INVALID_ARG);
    if (dst + bytes > NPU_MEM_SIZE)
        return ctx_err(ctx, NPU_ERROR_INVALID_ADDR);

    size_t   nwords = (bytes + 3) / 4;
    int32_t *tmp    = malloc(nwords * 4);
    if (!tmp) return ctx_err(ctx, NPU_ERROR_OUT_OF_MEM);
    memset(tmp, 0, nwords * 4);
    memcpy(tmp, src, bytes);
    npu_write_i32(ctx->npu, dst, tmp, (int)nwords);
    free(tmp);
    return NPU_SUCCESS;
}

npu_error_t npu_memcpy_d2h(npu_ctx_t ctx, void *dst,
                             npu_devptr_t src, size_t bytes)
{
    if (!ctx)           return NPU_ERROR_INVALID_CTX;
    if (!dst || !bytes) return ctx_err(ctx, NPU_ERROR_INVALID_ARG);
    if (src + bytes > NPU_MEM_SIZE)
        return ctx_err(ctx, NPU_ERROR_INVALID_ADDR);

    size_t   nwords = (bytes + 3) / 4;
    int32_t *tmp    = malloc(nwords * 4);
    if (!tmp) return ctx_err(ctx, NPU_ERROR_OUT_OF_MEM);
    npu_read_i32(ctx->npu, src, tmp, (int)nwords);
    memcpy(dst, tmp, bytes);
    free(tmp);
    return NPU_SUCCESS;
}

npu_error_t npu_memcpy_d2d(npu_ctx_t ctx, npu_devptr_t dst,
                             npu_devptr_t src, size_t bytes)
{
    if (!ctx)   return NPU_ERROR_INVALID_CTX;
    if (!bytes) return ctx_err(ctx, NPU_ERROR_INVALID_ARG);
    if (src + bytes > NPU_MEM_SIZE || dst + bytes > NPU_MEM_SIZE)
        return ctx_err(ctx, NPU_ERROR_INVALID_ADDR);

    size_t   nwords = (bytes + 3) / 4;
    int32_t *tmp    = malloc(nwords * 4);
    if (!tmp) return ctx_err(ctx, NPU_ERROR_OUT_OF_MEM);
    npu_read_i32(ctx->npu, src, tmp, (int)nwords);
    npu_write_i32(ctx->npu, dst, tmp, (int)nwords);
    free(tmp);
    return NPU_SUCCESS;
}

npu_error_t npu_memset_i32(npu_ctx_t ctx, npu_devptr_t ptr,
                             int32_t value, size_t count)
{
    if (!ctx)   return NPU_ERROR_INVALID_CTX;
    if (!count) return ctx_err(ctx, NPU_ERROR_INVALID_ARG);
    if (ptr + count * 4 > NPU_MEM_SIZE)
        return ctx_err(ctx, NPU_ERROR_INVALID_ADDR);

    int32_t *tmp = malloc(count * 4);
    if (!tmp) return ctx_err(ctx, NPU_ERROR_OUT_OF_MEM);
    for (size_t i = 0; i < count; i++) tmp[i] = value;
    npu_write_i32(ctx->npu, ptr, tmp, (int)count);
    free(tmp);
    return NPU_SUCCESS;
}

/* ════════════════════════════════════════════════════════════════════════════
 * Module internals — function storage
 * ════════════════════════════════════════════════════════════════════════════ */

static npu_module_t alloc_module(void)
{
    npu_module_t m = calloc(1, sizeof(struct npu_module_s));
    return m;
}

static npu_error_t mod_add_fn(npu_module_t mod, const char *name,
                               cinst *prog, int len)
{
    if (mod->nfns >= MAX_FNS_PER_MOD) return NPU_ERROR_TOO_MANY;
    struct npu_fn_s *f = &mod->fns[mod->nfns++];
    snprintf(f->name, sizeof(f->name), "%s", name);
    f->prog = prog;
    f->len  = len;
    return NPU_SUCCESS;
}

/* ════════════════════════════════════════════════════════════════════════════
 * ELF module loader
 *
 * Parses an ELF64 relocatable object (.o) produced by:
 *   llc -mtriple=npu -filetype=obj -o out.o input.ll
 *
 * For each global STT_FUNC symbol it decodes the 32-byte instruction words
 * from the .text section using npu_decode_cinst().
 * ════════════════════════════════════════════════════════════════════════════ */

npu_error_t npu_module_load(npu_ctx_t ctx, npu_module_t *out,
                             const char *elf_path)
{
    if (!ctx)              return NPU_ERROR_INVALID_CTX;
    if (!out || !elf_path) return ctx_err(ctx, NPU_ERROR_INVALID_ARG);
    if (ctx->nmods >= MAX_MODULES)
        return ctx_err(ctx, NPU_ERROR_TOO_MANY);

    /* mmap the file. */
    int fd = open(elf_path, O_RDONLY);
    if (fd < 0) return ctx_err(ctx, NPU_ERROR_IO);

    struct stat st;
    fstat(fd, &st);
    size_t fsz = (size_t)st.st_size;

    uint8_t *data = mmap(NULL, fsz, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    if (data == MAP_FAILED) return ctx_err(ctx, NPU_ERROR_IO);

    npu_error_t rc = NPU_SUCCESS;
    npu_module_t mod = alloc_module();
    if (!mod) { munmap(data, fsz); return ctx_err(ctx, NPU_ERROR_OUT_OF_MEM); }
    mod->owner = ctx;

    /* Validate ELF magic. */
    if (fsz < sizeof(Elf64_Ehdr) ||
        memcmp(data, ELFMAG, SELFMAG) ||
        data[EI_CLASS] != ELFCLASS64) {
        rc = NPU_ERROR_BAD_MODULE;
        goto done;
    }

    Elf64_Ehdr *ehdr     = (Elf64_Ehdr *)data;
    Elf64_Shdr *shdrs    = (Elf64_Shdr *)(data + ehdr->e_shoff);
    uint16_t    shnum    = ehdr->e_shnum;
    uint16_t    shstrndx = ehdr->e_shstrndx;

    /* Section name string table. */
    const char *shstrtab = (const char *)(data + shdrs[shstrndx].sh_offset);

    /* Locate .text, .symtab, .strtab. */
    Elf64_Shdr *sh_text   = NULL;
    Elf64_Shdr *sh_symtab = NULL;
    Elf64_Shdr *sh_strtab = NULL;

    for (uint16_t i = 0; i < shnum; i++) {
        const char *sname = shstrtab + shdrs[i].sh_name;
        if (!strcmp(sname, ".text"))   sh_text   = &shdrs[i];
        if (shdrs[i].sh_type == SHT_SYMTAB) sh_symtab = &shdrs[i];
        if (!strcmp(sname, ".strtab")) sh_strtab = &shdrs[i];
    }

    if (!sh_text || !sh_symtab || !sh_strtab) {
        rc = NPU_ERROR_BAD_MODULE;
        goto done;
    }

    const uint8_t *text    = data + sh_text->sh_offset;
    size_t         text_sz = sh_text->sh_size;
    const char    *strtab  = (const char *)(data + sh_strtab->sh_offset);
    Elf64_Sym     *syms    = (Elf64_Sym *)(data + sh_symtab->sh_offset);
    size_t         nsyms   = sh_symtab->sh_size / sizeof(Elf64_Sym);

    for (size_t si = 0; si < nsyms; si++) {
        Elf64_Sym *sym = &syms[si];
        if (ELF64_ST_BIND(sym->st_info) != STB_GLOBAL) continue;
        if (ELF64_ST_TYPE(sym->st_info) != STT_FUNC)   continue;
        if (sym->st_size == 0 || sym->st_value + sym->st_size > text_sz) continue;

        /* Function name. */
        const char *fname = strtab + sym->st_name;

        /* Decode instructions from .text. */
        size_t fn_sz   = sym->st_size;
        int    fn_insn = (int)(fn_sz / NPU_WORD_BYTES);
        if (fn_insn <= 0 || fn_insn > MAX_FN_INSTS) continue;

        cinst *prog = malloc(fn_insn * sizeof(cinst));
        if (!prog) { rc = NPU_ERROR_OUT_OF_MEM; goto done; }

        const uint8_t *fn_bytes = text + sym->st_value;
        for (int k = 0; k < fn_insn; k++)
            npu_decode_cinst(fn_bytes + k * NPU_WORD_BYTES, &prog[k]);

        if (mod_add_fn(mod, fname, prog, fn_insn) != NPU_SUCCESS) {
            free(prog);
            rc = NPU_ERROR_TOO_MANY;
            goto done;
        }
    }

    if (mod->nfns == 0) {
        rc = NPU_ERROR_BAD_MODULE;  /* no callable functions found */
        goto done;
    }

    ctx->mods[ctx->nmods++] = mod;
    *out = mod;

done:
    munmap(data, fsz);
    if (rc != NPU_SUCCESS) {
        for (int i = 0; i < mod->nfns; i++) free(mod->fns[i].prog);
        free(mod);
        ctx_err(ctx, rc);
    }
    return rc;
}

/* ════════════════════════════════════════════════════════════════════════════
 * ASM module loader
 *
 * Splits the source text at top-level function labels (lines starting with a
 * non-dot identifier immediately followed by ':'), assembles each chunk with
 * npu_asm_control(), and registers the result.
 *
 * Supports both:
 *   • Hand-written assembly with explicit function labels
 *   • A single anonymous function (no label) — stored as "_start"
 * ════════════════════════════════════════════════════════════════════════════ */

/* Returns 1 if `line` (trimmed) is a top-level function label "name:". */
static int is_fn_label(const char *line, char *name_out, int name_sz)
{
    if (!line || !*line) return 0;
    /* Must start with identifier character, not '.' (local label). */
    if (!isalpha((unsigned char)line[0]) && line[0] != '_') return 0;

    const char *colon = strchr(line, ':');
    if (!colon) return 0;

    /* Everything after ':' must be whitespace or comment. */
    for (const char *p = colon + 1; *p; p++) {
        if (!isspace((unsigned char)*p) && *p != '#') return 0;
    }

    int nlen = (int)(colon - line);
    if (name_out && nlen < name_sz) {
        memcpy(name_out, line, nlen);
        name_out[nlen] = '\0';
    }
    return 1;
}

/* Returns a heap-allocated trimmed copy of `line`. */
static char *trim_line(const char *s, int len)
{
    while (len > 0 && isspace((unsigned char)s[0])) { s++; len--; }
    while (len > 0 && isspace((unsigned char)s[len-1])) len--;
    char *out = malloc(len + 1);
    if (!out) return NULL;
    memcpy(out, s, len);
    out[len] = '\0';
    return out;
}

npu_error_t npu_module_load_asm(npu_ctx_t ctx, npu_module_t *out,
                                  const char *src)
{
    if (!ctx)       return NPU_ERROR_INVALID_CTX;
    if (!out || !src) return ctx_err(ctx, NPU_ERROR_INVALID_ARG);
    if (ctx->nmods >= MAX_MODULES)
        return ctx_err(ctx, NPU_ERROR_TOO_MANY);

    npu_module_t mod = alloc_module();
    if (!mod) return ctx_err(ctx, NPU_ERROR_OUT_OF_MEM);
    mod->owner = ctx;

    /* Split source into lines. */
    char *work = strdup(src);
    if (!work) { free(mod); return ctx_err(ctx, NPU_ERROR_OUT_OF_MEM); }

    /* We'll collect per-function chunks.  A chunk is: the function label line
     * plus all following lines until the next function label or end of source.
     * Directives (.text, .globl, .type, .size, .section) are stripped. */

    /* Dynamic buffer for current function's source lines. */
    char  cur_name[128]     = "_start";  /* fallback if no label seen yet */
    char  fn_src[65536]     = "";        /* assembled source for cur fn   */
    int   fn_src_len        = 0;
    int   in_fn             = 0;         /* 1 = we have seen at least one line */

    npu_error_t rc = NPU_SUCCESS;

    /* Helper: flush cur_name / fn_src into the module. */
    #define FLUSH_FN() do {                                              \
        if (in_fn && fn_src_len > 0) {                                  \
            cinst *prog = malloc(MAX_FN_INSTS * sizeof(cinst));          \
            char   err[256] = {0};                                       \
            int    len = npu_asm_control(fn_src, prog, MAX_FN_INSTS,    \
                                         err, sizeof(err));             \
            if (len < 0) {                                               \
                fprintf(stderr, "npu_runtime: asm error in '%s': %s\n", \
                        cur_name, err);                                  \
                free(prog);                                              \
                rc = NPU_ERROR_BAD_MODULE;                               \
            } else {                                                     \
                mod_add_fn(mod, cur_name, prog, len);                   \
            }                                                            \
            fn_src_len = 0; fn_src[0] = '\0';                          \
        }                                                                \
        in_fn = 0;                                                       \
    } while (0)

    char *line = work;
    while (line && *line) {
        char *nl = strchr(line, '\n');
        int   ll = nl ? (int)(nl - line) : (int)strlen(line);
        char *tr = trim_line(line, ll);

        if (tr) {
            char fn_lbl[128];
            if (is_fn_label(tr, fn_lbl, sizeof(fn_lbl))) {
                /* New function starts here. */
                FLUSH_FN();
                snprintf(cur_name, sizeof(cur_name), "%s", fn_lbl);
                in_fn = 1;
            } else if (tr[0] == '\0') {
                /* blank — skip */
            } else if (tr[0] == '.' && strchr(tr, ':') == NULL) {
                /* assembler directive (.text, .globl, .type …) — skip.
                 * Local branch labels (.Lloop:) have a colon and are kept. */
            } else {
                /* Code or local label — append to current function source. */
                in_fn = 1;
                int room = (int)sizeof(fn_src) - fn_src_len - 2;
                if (room > 0) {
                    int n = snprintf(fn_src + fn_src_len, room, "%s\n", tr);
                    if (n > 0) fn_src_len += n;
                }
            }
            free(tr);
        }
        line = nl ? nl + 1 : NULL;
    }
    FLUSH_FN();
    #undef FLUSH_FN

    free(work);

    if (rc != NPU_SUCCESS || mod->nfns == 0) {
        for (int i = 0; i < mod->nfns; i++) free(mod->fns[i].prog);
        free(mod);
        return ctx_err(ctx, rc != NPU_SUCCESS ? rc : NPU_ERROR_BAD_MODULE);
    }

    ctx->mods[ctx->nmods++] = mod;
    *out = mod;
    return NPU_SUCCESS;
}

/* ════════════════════════════════════════════════════════════════════════════
 * Module — unload and function lookup
 * ════════════════════════════════════════════════════════════════════════════ */

npu_error_t npu_module_unload(npu_module_t mod)
{
    if (!mod) return NPU_ERROR_INVALID_ARG;
    /* Deregister from the owning context so ctx_destroy won't double-free. */
    npu_ctx_t ctx = mod->owner;
    if (ctx) {
        for (int i = 0; i < ctx->nmods; i++) {
            if (ctx->mods[i] == mod) {
                ctx->mods[i] = ctx->mods[--ctx->nmods];
                break;
            }
        }
    }
    for (int i = 0; i < mod->nfns; i++) free(mod->fns[i].prog);
    free(mod);
    return NPU_SUCCESS;
}

npu_error_t npu_module_get_fn(npu_module_t mod, npu_fn_t *fn,
                               const char *name)
{
    if (!mod || !fn || !name) return NPU_ERROR_INVALID_ARG;
    for (int i = 0; i < mod->nfns; i++) {
        if (!strcmp(mod->fns[i].name, name)) {
            *fn = &mod->fns[i];
            return NPU_SUCCESS;
        }
    }
    return NPU_ERROR_NOT_FOUND;
}

/* ════════════════════════════════════════════════════════════════════════════
 * Launch
 *
 * Sets up the control-core register file to match the NPU calling convention
 * (x10–x17 = args, x1 = return address pointing at the HALT instruction),
 * then calls npu_run().
 * ════════════════════════════════════════════════════════════════════════════ */

npu_error_t npu_launch(npu_ctx_t ctx, npu_fn_t fn,
                        const npu_arg_t *args, int nargs)
{
    if (!ctx)       return NPU_ERROR_INVALID_CTX;
    if (!fn)        return ctx_err(ctx, NPU_ERROR_INVALID_ARG);
    if (nargs < 0 || nargs > 8)
        return ctx_err(ctx, NPU_ERROR_INVALID_ARG);

    /* Zero all registers. */
    memset(ctx->npu->x, 0, sizeof(ctx->npu->x));

    /* Load arguments into x10–x17. */
    for (int i = 0; i < nargs; i++) {
        int64_t val = 0;
        if (args[i].kind == NPU_ARG_PTR)
            val = (int64_t)(uint64_t)args[i].ptr;
        else
            val = args[i].i64;
        ctx->npu->x[10 + i] = val;
    }

    /* x1 (link register) = index of the last instruction (HALT). */
    ctx->npu->x[1] = fn->len - 1;

    npu_run(ctx->npu, fn->prog, fn->len);

    ctx->retval = ctx->npu->x[10];
    return NPU_SUCCESS;
}

/* ════════════════════════════════════════════════════════════════════════════
 * Debug helpers
 * ════════════════════════════════════════════════════════════════════════════ */

npu_error_t npu_mem_dump(npu_ctx_t ctx, npu_devptr_t ptr,
                          size_t count, FILE *out)
{
    if (!ctx || !out) return NPU_ERROR_INVALID_CTX;
    if (!count) return NPU_ERROR_INVALID_ARG;
    if ((uint64_t)ptr + count * 4 > NPU_MEM_SIZE)
        return ctx_err(ctx, NPU_ERROR_INVALID_ADDR);

    int32_t *buf = malloc(count * 4);
    if (!buf) return ctx_err(ctx, NPU_ERROR_OUT_OF_MEM);
    npu_read_i32(ctx->npu, ptr, buf, (int)count);

    fprintf(out, "Device memory @ 0x%08x  (%zu words):\n", ptr, count);
    for (size_t i = 0; i < count; i++) {
        if (i % 8 == 0) fprintf(out, "  [0x%06x]", (unsigned)(ptr + i * 4));
        fprintf(out, " %11d", buf[i]);
        if (i % 8 == 7 || i == count - 1) fputc('\n', out);
    }
    free(buf);
    return NPU_SUCCESS;
}

npu_error_t npu_reg_dump(npu_ctx_t ctx, FILE *out)
{
    if (!ctx || !out) return NPU_ERROR_INVALID_CTX;
    fprintf(out, "Control-core registers (after last npu_launch):\n");
    for (int i = 0; i < NCREG; i++) {
        fprintf(out, "  x%-2d = %lld", i, (long long)ctx->npu->x[i]);
        if (i % 4 == 3) fputc('\n', out);
    }
    if (NCREG % 4) fputc('\n', out);
    return NPU_SUCCESS;
}

npu_error_t npu_heap_dump(npu_ctx_t ctx, FILE *out)
{
    if (!ctx || !out) return NPU_ERROR_INVALID_CTX;
    fprintf(out, "Device DRAM heap (%d blocks):\n", ctx->nblocks);
    size_t total_used = 0, total_free = 0;
    for (int i = 0; i < ctx->nblocks; i++) {
        heap_block_t *b = &ctx->heap[i];
        fprintf(out, "  [%4d]  0x%08x .. 0x%08x  %6u bytes  %s\n",
                i, b->base, b->base + b->size - 1, b->size,
                b->used ? "USED" : "free");
        if (b->used) total_used += b->size;
        else          total_free += b->size;
    }
    fprintf(out, "  used: %zu bytes  free: %zu bytes  total: %u bytes\n",
            total_used, total_free, NPU_MEM_SIZE - HEAP_GUARD);
    return NPU_SUCCESS;
}
