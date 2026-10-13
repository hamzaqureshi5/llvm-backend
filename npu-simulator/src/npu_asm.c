/* npu_asm.c - text assembler, binary encoder/decoder, and disassembler.
 * See npu_asm.h. Pure string/byte manipulation; no dependence on npu.c state. */
#include "npu_asm.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

/* ======================================================================= */
/* Operand-format codes                                                    */
/* ======================================================================= */
enum {
    /* vector formats */
    VF_NONE, VF_VVV, VF_VV, VF_VS, VF_RV, VF_LOAD, VF_STORE, VF_GATHER,
    VF_SCATTER, VF_SEL, VF_CMP, VF_MV, VF_MMM, VF_MM, VF_VSETVL,
    VF_S_RI, VF_S_RF, VF_S_RRR, VF_S_RRI, VF_S_MEM, VF_S_RR, VF_S_R,
    VF_BR, VF_BRC,
    /* control / tensor formats */
    CF_NONE, CF_RRR, CF_RRI, CF_RI, CF_LOAD, CF_STORE, CF_BR2, CF_JAL,
    CF_JALR, CF_KL, CF_SHAPE, CF_TILE1, CF_T2, CF_T3, CF_TLOAD, CF_DMA,
    CF_TSCALE, CF_STATUS
};

typedef struct { const char *name; int op; int fmt; } mnem_t;

static const mnem_t VTAB[] = {
    {"VADD",V_VADD,VF_VVV},{"VSUB",V_VSUB,VF_VVV},{"VMUL",V_VMUL,VF_VVV},
    {"VDIV",V_VDIV,VF_VVV},{"VMAX",V_VMAX,VF_VVV},{"VMIN",V_VMIN,VF_VVV},
    {"VFMA",V_VFMA,VF_VVV},
    {"VEXP",V_VEXP,VF_VV},{"VLOG",V_VLOG,VF_VV},{"VSQRT",V_VSQRT,VF_VV},
    {"VRSQRT",V_VRSQRT,VF_VV},{"VRECIP",V_VRECIP,VF_VV},{"VABS",V_VABS,VF_VV},
    {"VNEG",V_VNEG,VF_VV},{"VMOV",V_VMOV,VF_VV},{"VCVT",V_VCVT,VF_VV},
    {"VSCAN",V_VSCAN,VF_VV},
    {"VBCAST",V_VBCAST,VF_VS},
    {"VREDSUM",V_VREDSUM,VF_RV},{"VREDMAX",V_VREDMAX,VF_RV},
    {"VREDMIN",V_VREDMIN,VF_RV},{"VREDAND",V_VREDAND,VF_RV},
    {"VREDOR",V_VREDOR,VF_RV},
    {"VLOAD",V_VLOAD,VF_LOAD},{"VSTORE",V_VSTORE,VF_STORE},
    {"VGATHER",V_VGATHER,VF_GATHER},{"VSCATTER",V_VSCATTER,VF_SCATTER},
    {"VSEL",V_VSEL,VF_SEL},{"VCMP",V_VCMP,VF_CMP},
    {"VMASK.SET",V_VMASK_SET,VF_MV},{"VMASK.AND",V_VMASK_AND,VF_MMM},
    {"VMASK.OR",V_VMASK_OR,VF_MMM},{"VMASK.NOT",V_VMASK_NOT,VF_MM},
    {"VSETVL",V_VSETVL,VF_VSETVL},
    {"SLI",V_SLI,VF_S_RI},{"SFI",V_SFI,VF_S_RF},
    {"SADD",V_SADD,VF_S_RRR},{"SSUB",V_SSUB,VF_S_RRR},{"SMUL",V_SMUL,VF_S_RRR},
    {"SADDI",V_SADDI,VF_S_RRI},{"SMULI",V_SMULI,VF_S_RRI},
    {"SLW",V_SLW,VF_S_MEM},{"SLF",V_SLF,VF_S_MEM},
    {"SRECIP",V_SRECIP,VF_S_RR},{"KID",V_KID,VF_S_R},
    {"BR",V_BR,VF_BR},{"BRZ",V_BRZ,VF_BRC},{"BRNZ",V_BRNZ,VF_BRC},
    {"LOOP",V_LOOP,VF_BRC},{"KEXIT",V_KEXIT,VF_NONE},
};
static const mnem_t CTAB[] = {
    {"ADD",C_ADD,CF_RRR},{"SUB",C_SUB,CF_RRR},{"MUL",C_MUL,CF_RRR},
    {"AND",C_AND,CF_RRR},{"OR",C_OR,CF_RRR},{"XOR",C_XOR,CF_RRR},
    {"SLL",C_SLL,CF_RRR},{"SRL",C_SRL,CF_RRR},{"SLT",C_SLT,CF_RRR},
    {"ADDI",C_ADDI,CF_RRI},{"LI",C_LI,CF_RI},
    {"LW",C_LW,CF_LOAD},{"LH",C_LH,CF_LOAD},{"LB",C_LB,CF_LOAD},
    {"SW",C_SW,CF_STORE},{"SH",C_SH,CF_STORE},{"SB",C_SB,CF_STORE},
    {"BEQ",C_BEQ,CF_BR2},{"BNE",C_BNE,CF_BR2},{"BLT",C_BLT,CF_BR2},
    {"BGE",C_BGE,CF_BR2},{"JAL",C_JAL,CF_JAL},{"JALR",C_JALR,CF_JALR},
    {"KLAUNCH",C_KLAUNCH,CF_KL},{"KWAIT",C_KWAIT,CF_NONE},{"HALT",C_HALT,CF_NONE},
    {"CFG.SHAPE",T_CFG_SHAPE,CF_SHAPE},{"ACC.CLR",T_ACC_CLR,CF_TILE1},
    {"MATMUL",T_MATMUL,CF_T3},{"MMA",T_MMA,CF_T3},{"DOT",T_DOT,CF_T3},
    {"LOAD",T_LOAD,CF_TLOAD},{"LOADW",T_LOADW,CF_TLOAD},{"STORE",T_STORE,CF_TLOAD},
    {"DMA.IN",T_DMA_IN,CF_DMA},{"DMA.OUT",T_DMA_OUT,CF_DMA},
    {"SCALE",T_SCALE,CF_TSCALE},{"RELU",T_RELU,CF_T2},
    {"FENCE",T_FENCE,CF_NONE},{"WAIT",T_WAIT,CF_NONE},{"BARRIER",T_BARRIER,CF_NONE},
    {"STATUS",T_STATUS,CF_STATUS},
};

/* ======================================================================= */
/* Small string helpers                                                    */
/* ======================================================================= */
static char *trim(char *s) {
    while (*s && isspace((unsigned char)*s)) s++;
    if (!*s) return s;
    char *e = s + strlen(s) - 1;
    while (e > s && isspace((unsigned char)*e)) *e-- = 0;
    return s;
}

static int preg(const char *t, char p, int hi, int *out, char *err, int es) {
    if (!t || (char)toupper((unsigned char)t[0]) != p) {
        snprintf(err, es, "expected %c-register, got '%s'", p, t ? t : "(none)");
        return -1;
    }
    char *end; long v = strtol(t + 1, &end, 10);
    if (*end || v < 0 || v > hi) { snprintf(err, es, "bad register '%s'", t); return -1; }
    *out = (int)v; return 0;
}
static int pimm(const char *t, long long *out, char *err, int es) {
    if (!t || !*t) { snprintf(err, es, "missing immediate"); return -1; }
    char *end; long long v = strtoll(t, &end, 0);
    if (*end) { snprintf(err, es, "bad immediate '%s'", t); return -1; }
    *out = v; return 0;
}
static int pflt(const char *t, double *out, char *err, int es) {
    if (!t || !*t) { snprintf(err, es, "missing float"); return -1; }
    char *end; double v = strtod(t, &end);
    if (*end) { snprintf(err, es, "bad float '%s'", t); return -1; }
    *out = v; return 0;
}
/* parse "OFF(pREG)" or bare absolute "ADDR"; p = base register prefix */
static int pmem(char *t, char p, int *off, int *base, int *use_base,
                char *err, int es) {
    char *lp = strchr(t, '(');
    if (lp) {
        char ob[64]; size_t n = (size_t)(lp - t);
        if (n >= sizeof ob) n = sizeof ob - 1;
        memcpy(ob, t, n); ob[n] = 0;
        long long o = 0; char *ot = trim(ob);
        if (*ot && pimm(ot, &o, err, es)) return -1;
        *off = (int)o;
        char *rp = strchr(lp, ')');
        if (!rp) { snprintf(err, es, "missing ')' in '%s'", t); return -1; }
        char rb[64]; size_t m = (size_t)(rp - (lp + 1));
        if (m >= sizeof rb) m = sizeof rb - 1;
        memcpy(rb, lp + 1, m); rb[m] = 0;
        if (preg(trim(rb), p, 31, base, err, es)) return -1;
        *use_base = 1;
    } else {
        long long o; if (pimm(t, &o, err, es)) return -1;
        *off = (int)o; *base = 0; *use_base = 0;
    }
    return 0;
}
static int cmp_kind(const char *s) {
    if (!strcmp(s,"LT")) return CMP_LT;
    if (!strcmp(s,"LE")) return CMP_LE;
    if (!strcmp(s,"EQ")) return CMP_EQ;
    if (!strcmp(s,"NE")) return CMP_NE;
    if (!strcmp(s,"GT")) return CMP_GT;
    if (!strcmp(s,"GE")) return CMP_GE;
    return -1;
}
static int cvt_kind(const char *s) {
    if (!strcmp(s,"F32")) return CVT_F32;
    if (!strcmp(s,"I8"))  return CVT_I8;
    if (!strcmp(s,"I32")) return CVT_I32;
    return -1;
}

/* ======================================================================= */
/* Line parsing (shared by both passes)                                    */
/* ======================================================================= */
typedef struct {
    int   has_label; char label[64];
    int   pred;                 /* mask reg for @m prefix, else -1 */
    int   has_instr; char mnem[32];
    int   nops; char ops[6][64];
} pline;

/* Parse one (mutable) line. Returns 0 ok, -1 on malformed structure. */
static int parse_line(char *line, pline *p, char *err, int es) {
    memset(p, 0, sizeof *p);
    p->pred = -1;
    /* strip comments (';' or '#') */
    char *c = line; while (*c) { if (*c == ';' || *c == '#') { *c = 0; break; } c++; }
    char *s = trim(line);
    if (!*s) return 0;
    /* label: leading "name:" */
    char *colon = strchr(s, ':');
    if (colon) {
        *colon = 0; char *lab = trim(s);
        if (strlen(lab) >= sizeof p->label) { snprintf(err,es,"label too long"); return -1; }
        strcpy(p->label, lab); p->has_label = 1;
        s = trim(colon + 1);
        if (!*s) return 0;
    }
    /* predication prefix: @m<idx> */
    if (*s == '@') {
        char *sp = s; while (*sp && !isspace((unsigned char)*sp)) sp++;
        char save = *sp; *sp = 0;
        int mr;
        if (preg(s + 1, 'M', 7, &mr, err, es)) return -1;
        p->pred = mr;
        *sp = save; s = trim(sp);
        if (!*s) { snprintf(err, es, "predicate with no instruction"); return -1; }
    }
    /* mnemonic */
    char *sp = s; while (*sp && !isspace((unsigned char)*sp)) sp++;
    char save = *sp; *sp = 0;
    if (strlen(s) >= sizeof p->mnem) { snprintf(err,es,"mnemonic too long"); return -1; }
    strcpy(p->mnem, s);
    p->has_instr = 1;
    if (save) s = trim(sp + 1); else s = sp;
    /* operands split on ',' */
    while (*s && p->nops < 6) {
        char *comma = strchr(s, ',');
        if (comma) *comma = 0;
        char *tok = trim(s);
        if (strlen(tok) >= sizeof p->ops[0]) { snprintf(err,es,"operand too long"); return -1; }
        strcpy(p->ops[p->nops++], tok);
        if (!comma) break;
        s = comma + 1;
    }
    return 0;
}

static const mnem_t *lookup(const mnem_t *tab, int n, const char *mn,
                            char *suffix, int sufsz) {
    suffix[0] = 0;
    for (int i = 0; i < n; i++) if (!strcmp(tab[i].name, mn)) return &tab[i];
    /* try splitting a ".SUFFIX" (e.g. VCMP.GT, VCVT.I8) */
    const char *dot = strchr(mn, '.');
    if (dot) {
        char base[32]; size_t b = (size_t)(dot - mn);
        if (b < sizeof base) { memcpy(base, mn, b); base[b] = 0;
            snprintf(suffix, sufsz, "%s", dot + 1);
            for (int i = 0; i < n; i++) if (!strcmp(tab[i].name, base)) return &tab[i];
        }
    }
    return NULL;
}

#define NEED(k) do{ if (p->nops < (k)) { snprintf(err,es,"%s: need %d operands, got %d",p->mnem,(k),p->nops); return -1; } }while(0)

/* ======================================================================= */
/* Pass 1: collect labels                                                  */
/* ======================================================================= */
typedef struct { char name[64]; int index; } label_t;

static int collect_labels(const char *src, label_t *labs, int maxlab, int *nlab,
                          char *err, int es) {
    *nlab = 0;
    char *buf = strdup(src);
    if (!buf) { snprintf(err,es,"oom"); return -1; }
    int idx = 0, rc = 0;
    char *save = NULL;
    for (char *ln = strtok_r(buf, "\n", &save); ln; ln = strtok_r(NULL, "\n", &save)) {
        char tmp[256]; snprintf(tmp, sizeof tmp, "%s", ln);
        pline p;
        if (parse_line(tmp, &p, err, es)) { rc = -1; break; }
        if (p.has_label) {
            if (*nlab >= maxlab) { snprintf(err,es,"too many labels"); rc=-1; break; }
            snprintf(labs[*nlab].name, sizeof labs[0].name, "%s", p.label);
            labs[*nlab].index = idx; (*nlab)++;
        }
        if (p.has_instr) idx++;
    }
    free(buf);
    return rc;
}
static int resolve_label(const label_t *labs, int n, const char *name,
                         char *err, int es) {
    for (int i = 0; i < n; i++) if (!strcmp(labs[i].name, name)) return labs[i].index;
    snprintf(err, es, "undefined label '%s'", name);
    return -1;
}

/* ======================================================================= */
/* Vector assembler                                                        */
/* ======================================================================= */
int npu_asm_vector(const char *src, vinst *out, int max_out, char *err, int errsz) {
    char *es_unused = err; (void)es_unused; int es = errsz;
    label_t labs[256]; int nlab;
    if (collect_labels(src, labs, 256, &nlab, err, es)) return -1;

    char *buf = strdup(src); if (!buf) { snprintf(err,es,"oom"); return -1; }
    int count = 0, rc = 0; char *save = NULL;
    for (char *ln = strtok_r(buf, "\n", &save); ln; ln = strtok_r(NULL, "\n", &save)) {
        char tmp[256]; snprintf(tmp, sizeof tmp, "%s", ln);
        pline pl, *p = &pl;
        if (parse_line(tmp, p, err, es)) { rc = -1; break; }
        if (!p->has_instr) continue;
        if (count >= max_out) { snprintf(err,es,"too many instructions"); rc=-1; break; }

        char suf[16];
        const mnem_t *m = lookup(VTAB, (int)(sizeof VTAB/sizeof VTAB[0]), p->mnem, suf, sizeof suf);
        if (!m) { snprintf(err,es,"unknown vector mnemonic '%s'", p->mnem); rc=-1; break; }

        vinst v; memset(&v, 0, sizeof v); v.op = m->op;
        if (p->pred >= 0) { v.use_mask = 1; v.mask = p->pred; }
        int t;
        switch (m->fmt) {
        case VF_NONE: break;
        case VF_VVV: NEED(3);
            if (preg(p->ops[0],'V',31,&v.vd,err,es)||preg(p->ops[1],'V',31,&v.vs1,err,es)||preg(p->ops[2],'V',31,&v.vs2,err,es)){rc=-1;} break;
        case VF_VV: NEED(2);
            if (preg(p->ops[0],'V',31,&v.vd,err,es)||preg(p->ops[1],'V',31,&v.vs,err,es)){rc=-1;}
            if (m->op==V_VCVT){ int k=cvt_kind(suf); if(k<0){snprintf(err,es,"VCVT needs .F32/.I8/.I32");rc=-1;} else v.cvt=k; } break;
        case VF_VS: NEED(2);
            if (preg(p->ops[0],'V',31,&v.vd,err,es)||preg(p->ops[1],'S',31,&v.rs,err,es)){rc=-1;} break;
        case VF_RV: NEED(2);
            if (preg(p->ops[0],'S',31,&v.rd,err,es)||preg(p->ops[1],'V',31,&v.vs,err,es)){rc=-1;} break;
        case VF_LOAD: NEED(2);
            if (preg(p->ops[0],'V',31,&v.vd,err,es)||pmem(p->ops[1],'S',(int*)&t,&v.rs,&v.use_base,err,es)){rc=-1;} else v.imm=t; break;
        case VF_STORE: NEED(2);
            if (preg(p->ops[0],'V',31,&v.vs,err,es)||pmem(p->ops[1],'S',(int*)&t,&v.rs,&v.use_base,err,es)){rc=-1;} else v.imm=t; break;
        case VF_GATHER: NEED(3);
            if (preg(p->ops[0],'V',31,&v.vd,err,es)||pmem(p->ops[1],'S',(int*)&t,&v.rs,&v.use_base,err,es)||preg(p->ops[2],'V',31,&v.vs1,err,es)){rc=-1;} else v.imm=t; break;
        case VF_SCATTER: NEED(3);
            if (preg(p->ops[0],'V',31,&v.vs,err,es)||pmem(p->ops[1],'S',(int*)&t,&v.rs,&v.use_base,err,es)||preg(p->ops[2],'V',31,&v.vs1,err,es)){rc=-1;} else v.imm=t; break;
        case VF_SEL: NEED(4);
            if (preg(p->ops[0],'V',31,&v.vd,err,es)||preg(p->ops[1],'V',31,&v.vs1,err,es)||preg(p->ops[2],'V',31,&v.vs2,err,es)||preg(p->ops[3],'M',7,&v.mask,err,es)){rc=-1;} break;
        case VF_CMP: NEED(3); {
            int k=cmp_kind(suf); if(k<0){snprintf(err,es,"VCMP needs .LT/.LE/.EQ/.NE/.GT/.GE");rc=-1;break;} v.cmp=k;
            if (preg(p->ops[0],'M',7,&v.md,err,es)||preg(p->ops[1],'V',31,&v.vs1,err,es)||preg(p->ops[2],'V',31,&v.vs2,err,es)){rc=-1;} } break;
        case VF_MV: NEED(2);
            if (preg(p->ops[0],'M',7,&v.md,err,es)||preg(p->ops[1],'V',31,&v.vs,err,es)){rc=-1;} break;
        case VF_MMM: NEED(3);
            if (preg(p->ops[0],'M',7,&v.md,err,es)||preg(p->ops[1],'M',7,&v.m1,err,es)||preg(p->ops[2],'M',7,&v.m2,err,es)){rc=-1;} break;
        case VF_MM: NEED(2);
            if (preg(p->ops[0],'M',7,&v.md,err,es)||preg(p->ops[1],'M',7,&v.m1,err,es)){rc=-1;} break;
        case VF_VSETVL: NEED(2);
            if (preg(p->ops[0],'S',31,&v.rd,err,es)||preg(p->ops[1],'S',31,&v.rs1,err,es)){rc=-1;} break;
        case VF_S_RI: NEED(2); { long long im;
            if (preg(p->ops[0],'S',31,&v.rd,err,es)||pimm(p->ops[1],&im,err,es)){rc=-1;} else v.imm=im; } break;
        case VF_S_RF: NEED(2); { double f;
            if (preg(p->ops[0],'S',31,&v.rd,err,es)||pflt(p->ops[1],&f,err,es)){rc=-1;} else v.fimm=f; } break;
        case VF_S_RRR: NEED(3);
            if (preg(p->ops[0],'S',31,&v.rd,err,es)||preg(p->ops[1],'S',31,&v.rs1,err,es)||preg(p->ops[2],'S',31,&v.rs2,err,es)){rc=-1;} break;
        case VF_S_RRI: NEED(3); { long long im;
            if (preg(p->ops[0],'S',31,&v.rd,err,es)||preg(p->ops[1],'S',31,&v.rs1,err,es)||pimm(p->ops[2],&im,err,es)){rc=-1;} else v.imm=im; } break;
        case VF_S_MEM: NEED(2);
            if (preg(p->ops[0],'S',31,&v.rd,err,es)||pmem(p->ops[1],'S',(int*)&t,&v.rs,&v.use_base,err,es)){rc=-1;} else v.imm=t; break;
        case VF_S_RR: NEED(2);
            if (preg(p->ops[0],'S',31,&v.rd,err,es)||preg(p->ops[1],'S',31,&v.rs1,err,es)){rc=-1;} break;
        case VF_S_R: NEED(1);
            if (preg(p->ops[0],'S',31,&v.rd,err,es)){rc=-1;} break;
        case VF_BR: NEED(1); { int tg=resolve_label(labs,nlab,p->ops[0],err,es); if(tg<0)rc=-1; else v.imm=tg; } break;
        case VF_BRC: NEED(2); { int tg=resolve_label(labs,nlab,p->ops[1],err,es);
            if (preg(p->ops[0],'S',31,&v.rs,err,es)||tg<0){rc=-1;} else v.imm=tg; } break;
        default: snprintf(err,es,"internal: bad vfmt"); rc=-1;
        }
        if (rc) break;
        out[count++] = v;
    }
    free(buf);
    return rc ? -1 : count;
}

/* ======================================================================= */
/* Control assembler                                                       */
/* ======================================================================= */
int npu_asm_control(const char *src, cinst *out, int max_out, char *err, int errsz) {
    int es = errsz;
    label_t labs[256]; int nlab;
    if (collect_labels(src, labs, 256, &nlab, err, es)) return -1;

    char *buf = strdup(src); if (!buf) { snprintf(err,es,"oom"); return -1; }
    int count = 0, rc = 0; char *save = NULL;
    for (char *ln = strtok_r(buf, "\n", &save); ln; ln = strtok_r(NULL, "\n", &save)) {
        char tmp[256]; snprintf(tmp, sizeof tmp, "%s", ln);
        pline pl, *p = &pl;
        if (parse_line(tmp, p, err, es)) { rc = -1; break; }
        if (!p->has_instr) continue;
        if (count >= max_out) { snprintf(err,es,"too many instructions"); rc=-1; break; }

        char suf[16];
        const mnem_t *m = lookup(CTAB, (int)(sizeof CTAB/sizeof CTAB[0]), p->mnem, suf, sizeof suf);
        if (!m) { snprintf(err,es,"unknown control mnemonic '%s'", p->mnem); rc=-1; break; }

        cinst c; memset(&c, 0, sizeof c); c.op = m->op;
        int t; long long im; double f;
        switch (m->fmt) {
        case CF_NONE: break;
        case CF_RRR: NEED(3);
            if (preg(p->ops[0],'X',31,&c.rd,err,es)||preg(p->ops[1],'X',31,&c.rs1,err,es)||preg(p->ops[2],'X',31,&c.rs2,err,es)){rc=-1;} break;
        case CF_RRI: NEED(3);
            if (preg(p->ops[0],'X',31,&c.rd,err,es)||preg(p->ops[1],'X',31,&c.rs1,err,es)||pimm(p->ops[2],&im,err,es)){rc=-1;} else c.imm=im; break;
        case CF_RI: NEED(2);
            if (preg(p->ops[0],'X',31,&c.rd,err,es)||pimm(p->ops[1],&im,err,es)){rc=-1;} else c.imm=im; break;
        case CF_LOAD: NEED(2);
            if (preg(p->ops[0],'X',31,&c.rd,err,es)||pmem(p->ops[1],'X',&t,&c.rs1,&c.use_base,err,es)){rc=-1;} else c.imm=t; break;
        case CF_STORE: NEED(2);
            if (preg(p->ops[0],'X',31,&c.rs2,err,es)||pmem(p->ops[1],'X',&t,&c.rs1,&c.use_base,err,es)){rc=-1;} else c.imm=t; break;
        case CF_BR2: NEED(3); { int tg=resolve_label(labs,nlab,p->ops[2],err,es);
            if (preg(p->ops[0],'X',31,&c.rs1,err,es)||preg(p->ops[1],'X',31,&c.rs2,err,es)||tg<0){rc=-1;} else c.imm=tg; } break;
        case CF_JAL: NEED(2); { int tg=resolve_label(labs,nlab,p->ops[1],err,es);
            if (preg(p->ops[0],'X',31,&c.rd,err,es)||tg<0){rc=-1;} else c.imm=tg; } break;
        case CF_JALR: NEED(3);
            if (preg(p->ops[0],'X',31,&c.rd,err,es)||preg(p->ops[1],'X',31,&c.rs1,err,es)||pimm(p->ops[2],&im,err,es)){rc=-1;} else c.imm=im; break;
        case CF_KL: NEED(3);
            if (pimm(p->ops[0],&im,err,es)||preg(p->ops[1],'X',31,&c.rs1,err,es)||preg(p->ops[2],'X',31,&c.rs2,err,es)){rc=-1;} else c.imm=im; break;
        case CF_SHAPE: NEED(3); {
            long long mm,kk,nn;
            if (pimm(p->ops[0],&mm,err,es)||pimm(p->ops[1],&kk,err,es)||pimm(p->ops[2],&nn,err,es)){rc=-1;} else { c.M=(int)mm;c.K=(int)kk;c.N=(int)nn; } } break;
        case CF_TILE1: NEED(1);
            if (preg(p->ops[0],'T',7,&c.dt,err,es)){rc=-1;} break;
        case CF_T2: NEED(2);
            if (preg(p->ops[0],'T',7,&c.dt,err,es)||preg(p->ops[1],'T',7,&c.a,err,es)){rc=-1;} break;
        case CF_T3: NEED(3);
            if (preg(p->ops[0],'T',7,&c.dt,err,es)||preg(p->ops[1],'T',7,&c.a,err,es)||preg(p->ops[2],'T',7,&c.b,err,es)){rc=-1;} break;
        case CF_TLOAD: NEED(3);
            if (preg(p->ops[0],'T',7,&c.dt,err,es)||pmem(p->ops[1],'X',&t,&c.rs1,&c.use_base,err,es)||pimm(p->ops[2],&im,err,es)){rc=-1;} else { c.imm=t; c.count=(uint32_t)im; } break;
        case CF_DMA: NEED(3);
            if (preg(p->ops[0],'X',31,&c.rd,err,es)||preg(p->ops[1],'X',31,&c.rs1,err,es)||pimm(p->ops[2],&im,err,es)){rc=-1;} else c.count=(uint32_t)im; break;
        case CF_TSCALE: NEED(3);
            if (preg(p->ops[0],'T',7,&c.dt,err,es)||preg(p->ops[1],'T',7,&c.a,err,es)||pflt(p->ops[2],&f,err,es)){rc=-1;} else c.fimm=f; break;
        case CF_STATUS: NEED(1);
            if (preg(p->ops[0],'X',31,&c.rd,err,es)){rc=-1;} break;
        default: snprintf(err,es,"internal: bad cfmt"); rc=-1;
        }
        if (rc) break;
        out[count++] = c;
    }
    free(buf);
    return rc ? -1 : count;
}

/* ======================================================================= */
/* Binary encode / decode (fixed 32-byte record)                           */
/* ======================================================================= */
static void put_u32(unsigned char *b, uint32_t v) {
    b[0]=v; b[1]=v>>8; b[2]=v>>16; b[3]=v>>24;
}
static uint32_t get_u32(const unsigned char *b) {
    return (uint32_t)b[0] | (uint32_t)b[1]<<8 | (uint32_t)b[2]<<16 | (uint32_t)b[3]<<24;
}
static void put_i64(unsigned char *b, int64_t v) {
    uint64_t u=(uint64_t)v; for (int i=0;i<8;i++) b[i]=(unsigned char)(u>>(8*i));
}
static int64_t get_i64(const unsigned char *b) {
    uint64_t u=0; for (int i=0;i<8;i++) u|=(uint64_t)b[i]<<(8*i); return (int64_t)u;
}

static int vop_is_float(int op) { return op == V_SFI; }
static int cop_is_float(int op) { return op==T_SCALE || op==T_QUANT || op==T_DEQUANT; }

int npu_encode_vinst(const vinst *in, unsigned char out[NPU_WORD_BYTES]) {
    memset(out, 0, NPU_WORD_BYTES);
    out[0]=1; out[1]=(unsigned char)in->op;
    out[2]=(unsigned char)in->vd;  out[3]=(unsigned char)in->vs1;
    out[4]=(unsigned char)in->vs2; out[5]=(unsigned char)in->vs;
    out[6]=(unsigned char)in->rd;  out[7]=(unsigned char)in->rs;
    out[8]=(unsigned char)in->rs1; out[9]=(unsigned char)in->rs2;
    out[10]=(unsigned char)in->md; out[11]=(unsigned char)in->m1;
    out[12]=(unsigned char)in->m2; out[13]=(unsigned char)in->use_mask;
    out[14]=(unsigned char)in->mask; out[15]=(unsigned char)in->use_base;
    if (vop_is_float(in->op)) { double d=in->fimm; memcpy(out+16,&d,8); }
    else put_i64(out+16, in->imm);
    out[24]=(unsigned char)in->cmp; out[25]=(unsigned char)in->cvt;
    return NPU_WORD_BYTES;
}
int npu_decode_vinst(const unsigned char in[NPU_WORD_BYTES], vinst *o) {
    memset(o, 0, sizeof *o);
    o->op=in[1];
    o->vd=in[2]; o->vs1=in[3]; o->vs2=in[4]; o->vs=in[5];
    o->rd=in[6]; o->rs=in[7]; o->rs1=in[8]; o->rs2=in[9];
    o->md=in[10]; o->m1=in[11]; o->m2=in[12]; o->use_mask=in[13];
    o->mask=in[14]; o->use_base=in[15];
    if (vop_is_float(o->op)) { double d; memcpy(&d,in+16,8); o->fimm=d; }
    else o->imm=get_i64(in+16);
    o->cmp=in[24]; o->cvt=in[25];
    return NPU_WORD_BYTES;
}
int npu_encode_cinst(const cinst *in, unsigned char out[NPU_WORD_BYTES]) {
    memset(out, 0, NPU_WORD_BYTES);
    out[0]=0; out[1]=(unsigned char)in->op;
    out[2]=(unsigned char)in->rd;  out[3]=(unsigned char)in->rs1;
    out[4]=(unsigned char)in->rs2; out[5]=(unsigned char)in->dt;
    out[6]=(unsigned char)in->a;   out[7]=(unsigned char)in->b;
    out[8]=(unsigned char)in->M;   out[9]=(unsigned char)in->K;
    out[10]=(unsigned char)in->N;  out[11]=(unsigned char)in->use_base;
    put_u32(out+12, in->count);
    if (cop_is_float(in->op)) { double d=in->fimm; memcpy(out+16,&d,8); }
    else put_i64(out+16, in->imm);
    return NPU_WORD_BYTES;
}
int npu_decode_cinst(const unsigned char in[NPU_WORD_BYTES], cinst *o) {
    memset(o, 0, sizeof *o);
    o->op=in[1];
    o->rd=in[2]; o->rs1=in[3]; o->rs2=in[4]; o->dt=in[5];
    o->a=in[6]; o->b=in[7]; o->M=in[8]; o->K=in[9]; o->N=in[10];
    o->use_base=in[11]; o->count=get_u32(in+12);
    if (cop_is_float(o->op)) { double d; memcpy(&d,in+16,8); o->fimm=d; }
    else o->imm=get_i64(in+16);
    return NPU_WORD_BYTES;
}

/* ======================================================================= */
/* Disassembler (struct -> text)                                           */
/* ======================================================================= */
static const char *vname(int op, int *fmt) {
    for (size_t i=0;i<sizeof VTAB/sizeof VTAB[0];i++)
        if (VTAB[i].op==op) { if(fmt)*fmt=VTAB[i].fmt; return VTAB[i].name; }
    return "?";
}
static const char *cname(int op, int *fmt) {
    for (size_t i=0;i<sizeof CTAB/sizeof CTAB[0];i++)
        if (CTAB[i].op==op) { if(fmt)*fmt=CTAB[i].fmt; return CTAB[i].name; }
    return "?";
}
static const char *CMPS[]={"LT","LE","EQ","NE","GT","GE"};
static const char *CVTS[]={"F32","I8","I32"};

void npu_disasm_vinst(const vinst *v, char *b, int bz) {
    int f = VF_NONE; const char *nm = vname(v->op, &f);
    char pre[8]=""; if (v->use_mask) snprintf(pre,sizeof pre,"@m%d ", v->mask);
    switch (f) {
    case VF_NONE: snprintf(b,bz,"%s%s",pre,nm); break;
    case VF_VVV: snprintf(b,bz,"%s%s v%d, v%d, v%d",pre,nm,v->vd,v->vs1,v->vs2); break;
    case VF_VV:
        if (v->op==V_VCVT) snprintf(b,bz,"%s%s.%s v%d, v%d",pre,nm,CVTS[v->cvt],v->vd,v->vs);
        else snprintf(b,bz,"%s%s v%d, v%d",pre,nm,v->vd,v->vs);
        break;
    case VF_VS: snprintf(b,bz,"%s%s v%d, s%d",pre,nm,v->vd,v->rs); break;
    case VF_RV: snprintf(b,bz,"%s%s s%d, v%d",pre,nm,v->rd,v->vs); break;
    case VF_LOAD: snprintf(b,bz,"%s%s v%d, %lld(s%d)",pre,nm,v->vd,(long long)v->imm,v->rs); break;
    case VF_STORE: snprintf(b,bz,"%s%s v%d, %lld(s%d)",pre,nm,v->vs,(long long)v->imm,v->rs); break;
    case VF_GATHER: snprintf(b,bz,"%s%s v%d, %lld(s%d), v%d",pre,nm,v->vd,(long long)v->imm,v->rs,v->vs1); break;
    case VF_SCATTER: snprintf(b,bz,"%s%s v%d, %lld(s%d), v%d",pre,nm,v->vs,(long long)v->imm,v->rs,v->vs1); break;
    case VF_SEL: snprintf(b,bz,"%s%s v%d, v%d, v%d, m%d",pre,nm,v->vd,v->vs1,v->vs2,v->mask); break;
    case VF_CMP: snprintf(b,bz,"%s%s.%s m%d, v%d, v%d",pre,nm,CMPS[v->cmp],v->md,v->vs1,v->vs2); break;
    case VF_MV: snprintf(b,bz,"%s%s m%d, v%d",pre,nm,v->md,v->vs); break;
    case VF_MMM: snprintf(b,bz,"%s%s m%d, m%d, m%d",pre,nm,v->md,v->m1,v->m2); break;
    case VF_MM: snprintf(b,bz,"%s%s m%d, m%d",pre,nm,v->md,v->m1); break;
    case VF_VSETVL: snprintf(b,bz,"%s%s s%d, s%d",pre,nm,v->rd,v->rs1); break;
    case VF_S_RI: snprintf(b,bz,"%s%s s%d, %lld",pre,nm,v->rd,(long long)v->imm); break;
    case VF_S_RF: snprintf(b,bz,"%s%s s%d, %g",pre,nm,v->rd,v->fimm); break;
    case VF_S_RRR: snprintf(b,bz,"%s%s s%d, s%d, s%d",pre,nm,v->rd,v->rs1,v->rs2); break;
    case VF_S_RRI: snprintf(b,bz,"%s%s s%d, s%d, %lld",pre,nm,v->rd,v->rs1,(long long)v->imm); break;
    case VF_S_MEM: snprintf(b,bz,"%s%s s%d, %lld(s%d)",pre,nm,v->rd,(long long)v->imm,v->rs); break;
    case VF_S_RR: snprintf(b,bz,"%s%s s%d, s%d",pre,nm,v->rd,v->rs1); break;
    case VF_S_R: snprintf(b,bz,"%s%s s%d",pre,nm,v->rd); break;
    case VF_BR: snprintf(b,bz,"%s%s %lld",pre,nm,(long long)v->imm); break;
    case VF_BRC: snprintf(b,bz,"%s%s s%d, %lld",pre,nm,v->rs,(long long)v->imm); break;
    default: snprintf(b,bz,"%s%s ?",pre,nm);
    }
}
void npu_disasm_cinst(const cinst *c, char *b, int bz) {
    int f = CF_NONE; const char *nm = cname(c->op, &f);
    switch (f) {
    case CF_NONE: snprintf(b,bz,"%s",nm); break;
    case CF_RRR: snprintf(b,bz,"%s x%d, x%d, x%d",nm,c->rd,c->rs1,c->rs2); break;
    case CF_RRI: snprintf(b,bz,"%s x%d, x%d, %lld",nm,c->rd,c->rs1,(long long)c->imm); break;
    case CF_RI: snprintf(b,bz,"%s x%d, %lld",nm,c->rd,(long long)c->imm); break;
    case CF_LOAD: snprintf(b,bz,"%s x%d, %lld(x%d)",nm,c->rd,(long long)c->imm,c->rs1); break;
    case CF_STORE: snprintf(b,bz,"%s x%d, %lld(x%d)",nm,c->rs2,(long long)c->imm,c->rs1); break;
    case CF_BR2: snprintf(b,bz,"%s x%d, x%d, %lld",nm,c->rs1,c->rs2,(long long)c->imm); break;
    case CF_JAL: snprintf(b,bz,"%s x%d, %lld",nm,c->rd,(long long)c->imm); break;
    case CF_JALR: snprintf(b,bz,"%s x%d, x%d, %lld",nm,c->rd,c->rs1,(long long)c->imm); break;
    case CF_KL: snprintf(b,bz,"%s %lld, x%d, x%d",nm,(long long)c->imm,c->rs1,c->rs2); break;
    case CF_SHAPE: snprintf(b,bz,"%s %d, %d, %d",nm,c->M,c->K,c->N); break;
    case CF_TILE1: snprintf(b,bz,"%s t%d",nm,c->dt); break;
    case CF_T2: snprintf(b,bz,"%s t%d, t%d",nm,c->dt,c->a); break;
    case CF_T3: snprintf(b,bz,"%s t%d, t%d, t%d",nm,c->dt,c->a,c->b); break;
    case CF_TLOAD: snprintf(b,bz,"%s t%d, %lld(x%d), %u",nm,c->dt,(long long)c->imm,c->rs1,c->count); break;
    case CF_DMA: snprintf(b,bz,"%s x%d, x%d, %u",nm,c->rd,c->rs1,c->count); break;
    case CF_TSCALE: snprintf(b,bz,"%s t%d, t%d, %g",nm,c->dt,c->a,c->fimm); break;
    case CF_STATUS: snprintf(b,bz,"%s x%d",nm,c->rd); break;
    default: snprintf(b,bz,"%s ?",nm);
    }
}
