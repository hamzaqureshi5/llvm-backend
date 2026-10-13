/* npu_rt.c — CUDA-style runtime library layered on the NPU kernel driver.
 *
 * Implements npu_rt.h.  Link with:
 *   gcc -O2 -Wall -o test npu_kdrv.c npu_rt.c npu.c npu_asm.c -lm
 *
 * Design:
 *   - Context owns a kdrv handle and a module table.
 *   - Module loading uses the same split-by-label + npu_asm_control()
 *     approach as npu_runtime.c; ELF loading uses mmap + ELF64 parsing
 *     + npu_decode_cinst() (also matching npu_runtime.c).
 *   - npuLaunchKernel copies the args array into a small internal pool
 *     so the caller's stack allocation is safe to discard.
 *   - npuSynchronize calls npu_kdrv_flush() which runs the firmware
 *     synchronously, so by the time flush returns all work is done.
 */

#include "npu_rt.h"
#include "npu_kdrv.h"
#include "npu.h"
#include "npu_asm.h"

/* Pull in POSIX extensions (strdup, etc.) before any system header. */
#ifndef _POSIX_C_SOURCE
#define _POSIX_C_SOURCE 200809L
#endif
#ifndef _DEFAULT_SOURCE
#define _DEFAULT_SOURCE 1
#endif
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <stdint.h>
#include <assert.h>
#include <ctype.h>
#include <elf.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

/* ════════════════════════════════════════════════════════════════════════════
 * Internal limits
 * ════════════════════════════════════════════════════════════════════════════ */

#define MAX_RT_MODULES   64
#define MAX_RT_FNS       128
#define MAX_FN_INSTS     4096
#define MAX_ARG_SLOTS    1024    /* pool of copied arg arrays                */
#define MAX_ARGS_PER_KERN 8      /* max args per kernel (x10..x17)           */

/* ════════════════════════════════════════════════════════════════════════════
 * Internal structs
 * ════════════════════════════════════════════════════════════════════════════ */

struct npu_module_s {
    npu_fn_s_t  fns[MAX_RT_FNS];
    int         nfns;
    NpuCtx      owner;   /* back-pointer so npuModuleUnload can deregister  */
};

/* One slot in the arg-copy pool: holds a copied args array for a pending
 * CMD_LAUNCH_KERN command.  Slots are allocated FIFO; after npuSynchronize
 * all slots are recycled at once. */
typedef struct {
    npu_arg_t args[MAX_ARGS_PER_KERN];
    int       nargs;
    int       used;
} arg_slot_t;

struct npu_ctx_s {
    npu_kdrv_t  drv;
    NpuModule   mods[MAX_RT_MODULES];
    int         nmods;
    /* arg-copy pool */
    arg_slot_t  arg_pool[MAX_ARG_SLOTS];
    int         arg_next;   /* next free slot index (wraps around)          */
};

/* ════════════════════════════════════════════════════════════════════════════
 * Error strings
 * ════════════════════════════════════════════════════════════════════════════ */

const char *npuGetErrorString(NpuError err)
{
    switch (err) {
    case NPU_OK:             return "success";
    case NPU_ERR_INVALID:    return "invalid argument or handle";
    case NPU_ERR_NO_MEM:     return "out of device memory";
    case NPU_ERR_QUEUE_FULL: return "command queue full";
    case NPU_ERR_IO:         return "I/O error";
    case NPU_ERR_NOT_FOUND:  return "function not found in module";
    case NPU_ERR_BAD_MODULE: return "bad module (ELF or ASM error)";
    default:                 return "unknown error";
    }
}

/* Map kdrv_err_t → NpuError */
static NpuError from_kdrv(kdrv_err_t e)
{
    switch (e) {
    case KDRV_OK:         return NPU_OK;
    case KDRV_NO_MEM:     return NPU_ERR_NO_MEM;
    case KDRV_QUEUE_FULL: return NPU_ERR_QUEUE_FULL;
    case KDRV_INVALID:    return NPU_ERR_INVALID;
    case KDRV_IO_ERR:     return NPU_ERR_IO;
    default:              return NPU_ERR_INVALID;
    }
}

/* ════════════════════════════════════════════════════════════════════════════
 * Device properties
 * ════════════════════════════════════════════════════════════════════════════ */

void npuGetDeviceProps(NpuDeviceProps *props)
{
    if (!props) return;
    memset(props, 0, sizeof(*props));
    snprintf(props->name, sizeof(props->name), "NPU Simulator");
    props->dram_bytes    = NPU_MEM_SIZE;
    props->scratch_bytes = NPU_SCRATCH_SIZE;
    props->max_cores     = NPU_MAX_CORES;
}

/* ════════════════════════════════════════════════════════════════════════════
 * Context
 * ════════════════════════════════════════════════════════════════════════════ */

NpuError npuInit(NpuCtx *out)
{
    if (!out) return NPU_ERR_INVALID;

    NpuCtx ctx = (NpuCtx)calloc(1, sizeof(struct npu_ctx_s));
    if (!ctx) return NPU_ERR_NO_MEM;

    kdrv_err_t ke = npu_kdrv_probe(&ctx->drv);
    if (ke != KDRV_OK) { free(ctx); return from_kdrv(ke); }

    *out = ctx;
    return NPU_OK;
}

NpuError npuFinalize(NpuCtx ctx)
{
    if (!ctx) return NPU_ERR_INVALID;

    /* Unload all modules (also frees their function cinst arrays). */
    while (ctx->nmods > 0)
        npuModuleUnload(ctx->mods[ctx->nmods - 1]);

    npu_kdrv_remove(ctx->drv);
    free(ctx);
    return NPU_OK;
}

NpuError npuSetCores(NpuCtx ctx, int cores)
{
    if (!ctx) return NPU_ERR_INVALID;
    npu_t *sim = npu_kdrv_get_sim(ctx->drv);
    if (!sim) return NPU_ERR_INVALID;
    npu_set_cores(sim, cores);
    return NPU_OK;
}

NpuError npuSetTrace(NpuCtx ctx, int enable)
{
    if (!ctx) return NPU_ERR_INVALID;
    npu_t *sim = npu_kdrv_get_sim(ctx->drv);
    if (!sim) return NPU_ERR_INVALID;
    sim->trace = enable ? 1 : 0;
    return NPU_OK;
}

NpuError npuResetStats(NpuCtx ctx)
{
    if (!ctx) return NPU_ERR_INVALID;
    npu_t *sim = npu_kdrv_get_sim(ctx->drv);
    if (!sim) return NPU_ERR_INVALID;
    memset(&sim->stats, 0, sizeof(sim->stats));
    return NPU_OK;
}

/* ════════════════════════════════════════════════════════════════════════════
 * Memory
 * ════════════════════════════════════════════════════════════════════════════ */

NpuError npuMalloc(NpuCtx ctx, NpuDevPtr *ptr, size_t bytes)
{
    if (!ctx || !ptr || !bytes) return NPU_ERR_INVALID;
    return from_kdrv(npu_kdrv_alloc(ctx->drv, ptr, bytes));
}

NpuError npuFree(NpuCtx ctx, NpuDevPtr ptr)
{
    if (!ctx || !ptr) return NPU_ERR_INVALID;
    return from_kdrv(npu_kdrv_free(ctx->drv, ptr));
}

/* ── async memcpy (pushes to command queue) ──────────────────────────────── */

NpuError npuMemcpy_h2d(NpuCtx ctx, NpuDevPtr dst, const void *src, size_t bytes)
{
    if (!ctx || !src || !bytes) return NPU_ERR_INVALID;
    npu_cmd_t cmd = {0};
    cmd.opcode = NPU_CMD_DMA_H2D;
    cmd.arg0   = (uint64_t)dst;                  /* device destination */
    cmd.arg1   = (uint64_t)(uintptr_t)src;       /* host source        */
    cmd.arg2   = (uint64_t)bytes;
    return from_kdrv(npu_kdrv_submit(ctx->drv, &cmd));
}

NpuError npuMemcpy_d2h(NpuCtx ctx, void *dst, NpuDevPtr src, size_t bytes)
{
    if (!ctx || !dst || !bytes) return NPU_ERR_INVALID;
    npu_cmd_t cmd = {0};
    cmd.opcode = NPU_CMD_DMA_D2H;
    cmd.arg0   = (uint64_t)(uintptr_t)dst;       /* host destination   */
    cmd.arg1   = (uint64_t)src;                  /* device source      */
    cmd.arg2   = (uint64_t)bytes;
    return from_kdrv(npu_kdrv_submit(ctx->drv, &cmd));
}

NpuError npuMemcpy_d2d(NpuCtx ctx, NpuDevPtr dst, NpuDevPtr src, size_t bytes)
{
    if (!ctx || !bytes) return NPU_ERR_INVALID;
    npu_cmd_t cmd = {0};
    cmd.opcode = NPU_CMD_DMA_D2D;
    cmd.arg0   = (uint64_t)dst;
    cmd.arg1   = (uint64_t)src;
    cmd.arg2   = (uint64_t)bytes;
    return from_kdrv(npu_kdrv_submit(ctx->drv, &cmd));
}

NpuError npuMemset(NpuCtx ctx, NpuDevPtr ptr, int32_t val, size_t count)
{
    if (!ctx || !count) return NPU_ERR_INVALID;

    npu_cmd_t cmd;
    memset(&cmd, 0, sizeof(cmd));
    cmd.opcode = NPU_CMD_MEMSET_I32;
    cmd.arg0   = (uint64_t)ptr;
    cmd.arg1   = (uint64_t)(uint32_t)val;
    cmd.arg2   = (uint64_t)count;

    return from_kdrv(npu_kdrv_submit(ctx->drv, &cmd));
}

/* ════════════════════════════════════════════════════════════════════════════
 * Module internals — shared with ELF and ASM loaders
 * ════════════════════════════════════════════════════════════════════════════ */

static NpuModule alloc_module(void)
{
    return (NpuModule)calloc(1, sizeof(struct npu_module_s));
}

static NpuError mod_add_fn(NpuModule mod, const char *name,
                            cinst *prog, int len)
{
    if (mod->nfns >= MAX_RT_FNS) return NPU_ERR_NO_MEM;
    npu_fn_s_t *f = &mod->fns[mod->nfns++];
    snprintf(f->name, sizeof(f->name), "%s", name);
    f->prog = prog;
    f->len  = len;
    return NPU_OK;
}

/* Register the module with the context and set its back-pointer. */
static NpuError ctx_register_mod(NpuCtx ctx, NpuModule mod)
{
    if (ctx->nmods >= MAX_RT_MODULES) return NPU_ERR_NO_MEM;
    mod->owner         = ctx;
    ctx->mods[ctx->nmods++] = mod;
    return NPU_OK;
}

/* ════════════════════════════════════════════════════════════════════════════
 * ELF module loader
 *
 * Parses an ELF64 relocatable object and decodes 32-byte instruction words
 * from .text for every global STT_FUNC symbol.
 * ════════════════════════════════════════════════════════════════════════════ */

NpuError npuModuleLoad(NpuCtx ctx, NpuModule *out, const char *elf_path)
{
    if (!ctx || !out || !elf_path) return NPU_ERR_INVALID;
    if (ctx->nmods >= MAX_RT_MODULES) return NPU_ERR_NO_MEM;

    int fd = open(elf_path, O_RDONLY);
    if (fd < 0) return NPU_ERR_IO;

    struct stat st;
    fstat(fd, &st);
    size_t fsz = (size_t)st.st_size;

    uint8_t *data = (uint8_t *)mmap(NULL, fsz, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    if (data == MAP_FAILED) return NPU_ERR_IO;

    NpuError    rc  = NPU_OK;
    NpuModule   mod = alloc_module();
    if (!mod) { munmap(data, fsz); return NPU_ERR_NO_MEM; }

    /* Validate ELF magic. */
    if (fsz < sizeof(Elf64_Ehdr) ||
        memcmp(data, ELFMAG, SELFMAG) ||
        data[EI_CLASS] != ELFCLASS64) {
        rc = NPU_ERR_BAD_MODULE;
        goto done;
    }

    {
        Elf64_Ehdr *ehdr     = (Elf64_Ehdr *)data;
        Elf64_Shdr *shdrs    = (Elf64_Shdr *)(data + ehdr->e_shoff);
        uint16_t    shnum    = ehdr->e_shnum;
        uint16_t    shstrndx = ehdr->e_shstrndx;
        const char *shstrtab = (const char *)(data + shdrs[shstrndx].sh_offset);

        Elf64_Shdr *sh_text   = NULL;
        Elf64_Shdr *sh_symtab = NULL;
        Elf64_Shdr *sh_strtab = NULL;

        for (uint16_t i = 0; i < shnum; i++) {
            const char *sname = shstrtab + shdrs[i].sh_name;
            if (!strcmp(sname, ".text"))              sh_text   = &shdrs[i];
            if (shdrs[i].sh_type == SHT_SYMTAB)      sh_symtab = &shdrs[i];
            if (!strcmp(sname, ".strtab"))            sh_strtab = &shdrs[i];
        }

        if (!sh_text || !sh_symtab || !sh_strtab) {
            rc = NPU_ERR_BAD_MODULE;
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
            if (sym->st_size == 0 ||
                sym->st_value + sym->st_size > text_sz)     continue;

            const char *fname   = strtab + sym->st_name;
            int         fn_insn = (int)(sym->st_size / NPU_WORD_BYTES);
            if (fn_insn <= 0 || fn_insn > MAX_FN_INSTS) continue;

            cinst *prog = (cinst *)malloc((size_t)fn_insn * sizeof(cinst));
            if (!prog) { rc = NPU_ERR_NO_MEM; goto done; }

            const uint8_t *fn_bytes = text + sym->st_value;
            for (int k = 0; k < fn_insn; k++)
                npu_decode_cinst(fn_bytes + k * NPU_WORD_BYTES, &prog[k]);

            fprintf(stderr, "[npu_rt] ELF: loaded function \"%s\"  (%d instructions, %zu bytes)\n",
                    fname, fn_insn, (size_t)fn_insn * NPU_WORD_BYTES);

            if (mod_add_fn(mod, fname, prog, fn_insn) != NPU_OK) {
                free(prog);
                rc = NPU_ERR_NO_MEM;
                goto done;
            }
        }
    }

    if (mod->nfns == 0) { rc = NPU_ERR_BAD_MODULE; goto done; }

    rc = ctx_register_mod(ctx, mod);
    if (rc == NPU_OK) *out = mod;

done:
    munmap(data, fsz);
    if (rc != NPU_OK) {
        for (int i = 0; i < mod->nfns; i++) free(mod->fns[i].prog);
        free(mod);
    }
    return rc;
}

/* ════════════════════════════════════════════════════════════════════════════
 * ASM module loader
 *
 * Splits source at top-level function labels (identifier + ':'), strips
 * assembler directives (lines starting with '.'), and assembles each chunk
 * with npu_asm_control().  Matches the approach in npu_runtime.c exactly.
 * ════════════════════════════════════════════════════════════════════════════ */

static int is_fn_label(const char *line, char *name_out, int name_sz)
{
    if (!line || !*line) return 0;
    if (!isalpha((unsigned char)line[0]) && line[0] != '_') return 0;
    const char *colon = strchr(line, ':');
    if (!colon) return 0;
    for (const char *p = colon + 1; *p; p++)
        if (!isspace((unsigned char)*p) && *p != '#') return 0;
    int nlen = (int)(colon - line);
    if (name_out && nlen < name_sz) {
        memcpy(name_out, line, (size_t)nlen);
        name_out[nlen] = '\0';
    }
    return 1;
}

static char *rt_trim_line(const char *s, int len)
{
    while (len > 0 && isspace((unsigned char)s[0])) { s++; len--; }
    while (len > 0 && isspace((unsigned char)s[len-1])) len--;
    char *out = (char *)malloc((size_t)len + 1);
    if (!out) return NULL;
    memcpy(out, s, (size_t)len);
    out[len] = '\0';
    return out;
}

NpuError npuModuleLoadAsm(NpuCtx ctx, NpuModule *out, const char *src)
{
    if (!ctx || !out || !src) return NPU_ERR_INVALID;
    if (ctx->nmods >= MAX_RT_MODULES) return NPU_ERR_NO_MEM;

    NpuModule mod = alloc_module();
    if (!mod) return NPU_ERR_NO_MEM;

    char *work = strdup(src);
    if (!work) { free(mod); return NPU_ERR_NO_MEM; }

    char     cur_name[128]  = "_start";
    char    *fn_src         = (char *)calloc(1, 65536);
    if (!fn_src) { free(work); free(mod); return NPU_ERR_NO_MEM; }
    int      fn_src_len     = 0;
    int      in_fn          = 0;
    NpuError rc             = NPU_OK;

    /* Helper: flush current function buffer into module. */
#define FLUSH_FN() do { \
        if (in_fn && fn_src_len > 0) { \
            cinst *prog = (cinst *)malloc(MAX_FN_INSTS * sizeof(cinst)); \
            char   errbuf[256] = {0}; \
            int    flen = npu_asm_control(fn_src, prog, MAX_FN_INSTS, \
                                           errbuf, sizeof(errbuf)); \
            if (flen < 0) { \
                fprintf(stderr, "npu_rt: asm error in '%s': %s\n", \
                        cur_name, errbuf); \
                free(prog); \
                rc = NPU_ERR_BAD_MODULE; \
            } else { \
                mod_add_fn(mod, cur_name, prog, flen); \
            } \
            fn_src_len = 0; fn_src[0] = '\0'; \
        } \
        in_fn = 0; \
    } while (0)

    char *line = work;
    while (line && *line) {
        char *nl = strchr(line, '\n');
        int   ll = nl ? (int)(nl - line) : (int)strlen(line);
        char *tr = rt_trim_line(line, ll);

        if (tr) {
            char fn_lbl[128];
            if (is_fn_label(tr, fn_lbl, sizeof(fn_lbl))) {
                FLUSH_FN();
                snprintf(cur_name, sizeof(cur_name), "%s", fn_lbl);
                in_fn = 1;
            } else if (tr[0] == '\0') {
                /* blank — skip */
            } else if (tr[0] == '.' && !strchr(tr, ':')) {
                /* assembler directive — skip */
            } else {
                in_fn = 1;
                int room = 65536 - fn_src_len - 2;
                if (room > 0) {
                    int n = snprintf(fn_src + fn_src_len, (size_t)room, "%s\n", tr);
                    if (n > 0) fn_src_len += n;
                }
            }
            free(tr);
        }
        line = nl ? nl + 1 : NULL;
    }
    FLUSH_FN();
#undef FLUSH_FN

    free(fn_src);
    free(work);

    if (rc != NPU_OK || mod->nfns == 0) {
        for (int i = 0; i < mod->nfns; i++) free(mod->fns[i].prog);
        free(mod);
        return rc != NPU_OK ? rc : NPU_ERR_BAD_MODULE;
    }

    rc = ctx_register_mod(ctx, mod);
    if (rc == NPU_OK)
        *out = mod;
    else {
        for (int i = 0; i < mod->nfns; i++) free(mod->fns[i].prog);
        free(mod);
    }
    return rc;
}

/* ════════════════════════════════════════════════════════════════════════════
 * Module unload and function lookup
 * ════════════════════════════════════════════════════════════════════════════ */

NpuError npuModuleUnload(NpuModule mod)
{
    if (!mod) return NPU_ERR_INVALID;
    /* Deregister from owning context to prevent double-free on finalize. */
    NpuCtx ctx = mod->owner;
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
    return NPU_OK;
}

NpuError npuGetFunction(NpuModule mod, NpuFunction *fn, const char *name)
{
    if (!mod || !fn || !name) return NPU_ERR_INVALID;
    for (int i = 0; i < mod->nfns; i++) {
        if (!strcmp(mod->fns[i].name, name)) {
            *fn = &mod->fns[i];
            return NPU_OK;
        }
    }
    return NPU_ERR_NOT_FOUND;
}

/* ════════════════════════════════════════════════════════════════════════════
 * Tensor engine helpers (async — push commands)
 * ════════════════════════════════════════════════════════════════════════════ */

NpuError npuGemm(NpuCtx ctx,
                 NpuDevPtr A, NpuDevPtr B, NpuDevPtr C,
                 int M, int K, int N)
{
    if (!ctx) return NPU_ERR_INVALID;
    if (M <= 0 || K <= 0 || N <= 0) return NPU_ERR_INVALID;
    /* Dims are packed into extra as 8-bit fields — clamp to 0..255. */
    if (M > 255 || K > 255 || N > 255) return NPU_ERR_INVALID;

    npu_cmd_t cmd;
    memset(&cmd, 0, sizeof(cmd));
    cmd.opcode = NPU_CMD_TENSOR_GEMM;
    cmd.extra  = (uint32_t)(M | (K << 8) | (N << 16));
    cmd.arg0   = (uint64_t)A;
    cmd.arg1   = (uint64_t)B;
    cmd.arg2   = (uint64_t)C;

    return from_kdrv(npu_kdrv_submit(ctx->drv, &cmd));
}

NpuError npuRelu(NpuCtx ctx, NpuDevPtr src, NpuDevPtr dst, int M, int N)
{
    if (!ctx) return NPU_ERR_INVALID;
    if (M <= 0 || N <= 0) return NPU_ERR_INVALID;
    /* Pack M and N into 16-bit fields inside extra. */
    if (M > 0xFFFF || N > 0xFFFF) return NPU_ERR_INVALID;

    npu_cmd_t cmd;
    memset(&cmd, 0, sizeof(cmd));
    cmd.opcode = NPU_CMD_TENSOR_RELU;
    cmd.extra  = (uint32_t)(M | (N << 16));
    cmd.arg0   = (uint64_t)src;
    cmd.arg1   = (uint64_t)dst;

    return from_kdrv(npu_kdrv_submit(ctx->drv, &cmd));
}

/* ════════════════════════════════════════════════════════════════════════════
 * Kernel launch (async)
 *
 * The args pointer typically points to the caller's stack — it will be gone
 * by the time npuSynchronize runs the firmware.  We copy the args into our
 * internal pool and store a pointer to the pool copy in the command packet.
 * ════════════════════════════════════════════════════════════════════════════ */

NpuError npuLaunchKernel(NpuCtx ctx, NpuFunction fn,
                          const npu_arg_t *args, int nargs)
{
    if (!ctx || !fn)         return NPU_ERR_INVALID;
    if (nargs < 0 || nargs > MAX_ARGS_PER_KERN)
        return NPU_ERR_INVALID;
    if (nargs > 0 && !args) return NPU_ERR_INVALID;

    /* Grab a pool slot for the args copy. */
    int slot = -1;
    for (int i = 0; i < MAX_ARG_SLOTS; i++) {
        int idx = (ctx->arg_next + i) % MAX_ARG_SLOTS;
        if (!ctx->arg_pool[idx].used) {
            slot = idx;
            ctx->arg_next = (idx + 1) % MAX_ARG_SLOTS;
            break;
        }
    }
    if (slot < 0) return NPU_ERR_NO_MEM;

    arg_slot_t *s = &ctx->arg_pool[slot];
    s->used  = 1;
    s->nargs = nargs;
    if (nargs > 0)
        memcpy(s->args, args, (size_t)nargs * sizeof(npu_arg_t));

    npu_cmd_t cmd;
    memset(&cmd, 0, sizeof(cmd));
    cmd.opcode = NPU_CMD_LAUNCH_KERN;
    cmd.extra  = (uint32_t)nargs;
    cmd.arg0   = (uint64_t)(uintptr_t)fn;        /* pointer to npu_fn_s_t  */
    cmd.arg1   = (uint64_t)(uintptr_t)s->args;   /* pointer to pool copy   */

    return from_kdrv(npu_kdrv_submit(ctx->drv, &cmd));
}

/* ════════════════════════════════════════════════════════════════════════════
 * Synchronize — flush the command queue and wait for completion
 * ════════════════════════════════════════════════════════════════════════════ */

NpuError npuSynchronize(NpuCtx ctx)
{
    if (!ctx) return NPU_ERR_INVALID;

    /* Flush is synchronous: firmware runs to completion before returning. */
    NpuError rc = from_kdrv(npu_kdrv_flush(ctx->drv));

    /* All arg-copy pool slots are now safe to recycle. */
    for (int i = 0; i < MAX_ARG_SLOTS; i++)
        ctx->arg_pool[i].used = 0;
    ctx->arg_next = 0;

    return rc;
}

/* ════════════════════════════════════════════════════════════════════════════
 * Fence management
 * ════════════════════════════════════════════════════════════════════════════ */

NpuError npuFenceCreate(NpuCtx ctx, NpuFence *fence)
{
    if (!ctx || !fence) return NPU_ERR_INVALID;
    NpuError e = from_kdrv(npu_kdrv_fence_create(ctx->drv, fence));
    if (e != NPU_OK) return e;
    /* Insert CMD_FENCE_SIGNAL into the command stream at this point.
     * The fence becomes "done" when the firmware processes this command
     * (i.e., after the next npuSynchronize). */
    npu_cmd_t cmd = {0};
    cmd.opcode = NPU_CMD_FENCE_SIGNAL;
    cmd.arg0   = *fence;
    return from_kdrv(npu_kdrv_submit(ctx->drv, &cmd));
}

NpuError npuFenceWait(NpuCtx ctx, NpuFence fence)
{
    if (!ctx) return NPU_ERR_INVALID;
    return from_kdrv(npu_kdrv_fence_wait(ctx->drv, fence));
}

/* ════════════════════════════════════════════════════════════════════════════
 * Result read-back (synchronous, use after npuSynchronize)
 * ════════════════════════════════════════════════════════════════════════════ */

NpuError npuRead_f32(NpuCtx ctx, NpuDevPtr src, float *dst, int count)
{
    if (!ctx || !dst || count <= 0) return NPU_ERR_INVALID;
    npu_t *sim = npu_kdrv_get_sim(ctx->drv);
    if (!sim) return NPU_ERR_INVALID;
    npu_read_f32(sim, src, dst, count);
    return NPU_OK;
}

NpuError npuRead_i32(NpuCtx ctx, NpuDevPtr src, int32_t *dst, int count)
{
    if (!ctx || !dst || count <= 0) return NPU_ERR_INVALID;
    npu_t *sim = npu_kdrv_get_sim(ctx->drv);
    if (!sim) return NPU_ERR_INVALID;
    npu_read_i32(sim, src, dst, count);
    return NPU_OK;
}

/* ════════════════════════════════════════════════════════════════════════════
 * Debug helpers
 * ════════════════════════════════════════════════════════════════════════════ */

NpuError npuMemDump(NpuCtx ctx, NpuDevPtr ptr, size_t count, FILE *out)
{
    if (!ctx || !out || !count) return NPU_ERR_INVALID;
    npu_t *sim = npu_kdrv_get_sim(ctx->drv);
    if (!sim) return NPU_ERR_INVALID;
    if ((uint64_t)ptr + count * 4 > NPU_MEM_SIZE) return NPU_ERR_INVALID;

    int32_t *buf = (int32_t *)malloc(count * 4);
    if (!buf) return NPU_ERR_NO_MEM;
    npu_read_i32(sim, ptr, buf, (int)count);

    fprintf(out, "Device memory @ 0x%08x  (%zu words):\n", ptr, count);
    for (size_t i = 0; i < count; i++) {
        if (i % 8 == 0) fprintf(out, "  [0x%06x]", (unsigned)(ptr + i * 4));
        fprintf(out, " %11d", buf[i]);
        if (i % 8 == 7 || i == count - 1) fputc('\n', out);
    }
    free(buf);
    return NPU_OK;
}

NpuError npuPrintStats(NpuCtx ctx)
{
    if (!ctx) return NPU_ERR_INVALID;
    npu_kdrv_print_stats(ctx->drv);
    return NPU_OK;
}

void npuHeapDump(NpuCtx ctx, FILE *out)
{
    if (!ctx || !out) return;
    /* Access heap metadata via the driver's internal structures.
     * Since the heap is inside npu_kdrv_s (opaque), we expose a summary
     * by printing through kdrv stats and the known DRAM size. */
    fprintf(out, "NPU device DRAM: %u bytes total, guard=%u bytes\n",
            NPU_MEM_SIZE, 64u);
    fprintf(out, "(Use npuPrintStats for detailed allocator info)\n");
}
