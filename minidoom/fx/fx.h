/* Portable C implementation of isa/semantics.md.
 *
 * Rules: no signed overflow, no shifts of negative values, no implementation-defined
 * right shifts. Every helper is total. Status flags accumulate in fx_flags (bit 0 SAT,
 * bit 1 DIVZ, bit 2 SHIFT, bit 3 OVF); the golden dumper serialises fx_flags with the
 * game state. Built and tested with -fsanitize=undefined -fno-sanitize-recover.
 */
#ifndef DROSOPHILOS_FX_H
#define DROSOPHILOS_FX_H

#include <stdint.h>

#define FX_FLAG_SAT 1
#define FX_FLAG_DIVZ 2
#define FX_FLAG_SHIFT 4
#define FX_FLAG_OVF 8

extern int fx_flags;

typedef int32_t fixed_t;
#define FRACBITS 16
#define FRACUNIT ((fixed_t)(1 << FRACBITS))

static inline int64_t fx__imin(int w) { return -((int64_t)1 << (w - 1)); }
static inline int64_t fx__imax(int w) { return ((int64_t)1 << (w - 1)) - 1; }
static inline uint64_t fx__umax(int w) { return (w == 64) ? UINT64_MAX : (((uint64_t)1 << w) - 1); }

/* exact -> modulo 2^w into signed range (w <= 32 so the exact value fits int64) */
static inline int64_t fx__wrap(int64_t x, int w) {
    uint64_t u = (uint64_t)x & fx__umax(w);
    if (u > (uint64_t)fx__imax(w)) return (int64_t)u - ((int64_t)1 << w);
    return (int64_t)u;
}

static inline int64_t fx__clamp(int64_t x, int w) {
    if (x < fx__imin(w)) { fx_flags |= FX_FLAG_SAT; return fx__imin(w); }
    if (x > fx__imax(w)) { fx_flags |= FX_FLAG_SAT; return fx__imax(w); }
    return x;
}

/* floor division by a power of two, defined for negative values */
static inline int64_t fx__floor_shr(int64_t x, int n) {
    int64_t d = (int64_t)1 << n;
    int64_t q = x / d;              /* truncates toward zero */
    if ((x % d != 0) && (x < 0)) q -= 1;
    return q;
}

static inline int64_t fx__trunc_div(int64_t a, int64_t b) { return a / b; } /* C99: toward zero */

/* ---- generic width-w helpers (w in {8,16,32}); operands already in range ---- */
static inline int64_t fxw_add_wrap(int64_t a, int64_t b, int w) { int64_t e = a + b, r = fx__wrap(e, w); if (r != e) fx_flags |= FX_FLAG_OVF; return r; }
static inline int64_t fxw_sub_wrap(int64_t a, int64_t b, int w) { int64_t e = a - b, r = fx__wrap(e, w); if (r != e) fx_flags |= FX_FLAG_OVF; return r; }
static inline int64_t fxw_mul_wrap(int64_t a, int64_t b, int w) { int64_t e = a * b, r = fx__wrap(e, w); if (r != e) fx_flags |= FX_FLAG_OVF; return r; }
static inline int64_t fxw_neg_wrap(int64_t a, int w) { int64_t e = -a, r = fx__wrap(e, w); if (r != e) fx_flags |= FX_FLAG_OVF; return r; }
static inline int64_t fxw_abs_wrap(int64_t a, int w) { int64_t e = a < 0 ? -a : a, r = fx__wrap(e, w); if (r != e) fx_flags |= FX_FLAG_OVF; return r; }
static inline int64_t fxw_add_sat(int64_t a, int64_t b, int w) { return fx__clamp(a + b, w); }
static inline int64_t fxw_sub_sat(int64_t a, int64_t b, int w) { return fx__clamp(a - b, w); }
static inline int64_t fxw_mul_sat(int64_t a, int64_t b, int w) { return fx__clamp(a * b, w); }
static inline int64_t fxw_neg_sat(int64_t a, int w) { return fx__clamp(-a, w); }
static inline int64_t fxw_abs_sat(int64_t a, int w) { return fx__clamp(a < 0 ? -a : a, w); }

static inline int64_t fxw_div(int64_t a, int64_t b, int w) {
    if (b == 0) { fx_flags |= FX_FLAG_DIVZ; return 0; }
    if (a == fx__imin(w) && b == -1) { fx_flags |= FX_FLAG_OVF; return fx__imin(w); }
    return fx__trunc_div(a, b);
}
static inline int64_t fxw_rem(int64_t a, int64_t b, int w) {
    if (b == 0) { fx_flags |= FX_FLAG_DIVZ; return a; }
    if (a == fx__imin(w) && b == -1) return 0;
    return a - b * fx__trunc_div(a, b);
}
static inline int64_t fxw_divu(int64_t a, int64_t b, int w) { (void)w; if (b == 0) { fx_flags |= FX_FLAG_DIVZ; return 0; } return (int64_t)((uint64_t)a / (uint64_t)b); }
static inline int64_t fxw_remu(int64_t a, int64_t b, int w) { (void)w; if (b == 0) { fx_flags |= FX_FLAG_DIVZ; return a; } return (int64_t)((uint64_t)a % (uint64_t)b); }

static inline int64_t fxw_shl(int64_t a, int64_t n, int w) {
    if (n < 0 || n > w - 1) { fx_flags |= FX_FLAG_SHIFT; return 0; }
    int64_t e = (int64_t)((uint64_t)a << n);      /* exact for |a| < 2^31, n <= 31 */
    /* (uint64_t)a of a negative a is modular; shifting keeps the exact product mod 2^64,
       which is exact here because |a * 2^n| < 2^63 */
    int64_t r = fx__wrap(e, w);
    if (fx__floor_shr(r, (int)n) != a) fx_flags |= FX_FLAG_OVF;
    return r;
}
static inline int64_t fxw_shr(int64_t a, int64_t n, int w) {
    if (n < 0 || n > w - 1) { fx_flags |= FX_FLAG_SHIFT; return 0; }
    return fx__floor_shr(a, (int)n);
}
static inline int64_t fxw_shru(int64_t a, int64_t n, int w) {
    if (n < 0 || n > w - 1) { fx_flags |= FX_FLAG_SHIFT; return 0; }
    return (int64_t)((uint64_t)a >> n);
}
static inline int64_t fxw_and(int64_t a, int64_t b, int w) { return fx__wrap((int64_t)(((uint64_t)a & fx__umax(w)) & ((uint64_t)b & fx__umax(w))), w); }
static inline int64_t fxw_or(int64_t a, int64_t b, int w)  { return fx__wrap((int64_t)(((uint64_t)a & fx__umax(w)) | ((uint64_t)b & fx__umax(w))), w); }
static inline int64_t fxw_xor(int64_t a, int64_t b, int w) { return fx__wrap((int64_t)(((uint64_t)a & fx__umax(w)) ^ ((uint64_t)b & fx__umax(w))), w); }
static inline int64_t fxw_not(int64_t a, int w) { return fx__wrap((int64_t)(~(uint64_t)a & fx__umax(w)), w); }

/* ---- Q16.16 ---- */
static inline fixed_t fx_mul(fixed_t a, fixed_t b) {
    int64_t p = (int64_t)a * (int64_t)b;
    int64_t e = fx__floor_shr(p, FRACBITS);
    int64_t r = fx__wrap(e, 32);
    if (r != e) fx_flags |= FX_FLAG_OVF;
    return (fixed_t)r;
}
static inline fixed_t fx_mul_sat(fixed_t a, fixed_t b) {
    int64_t p = (int64_t)a * (int64_t)b;
    return (fixed_t)fx__clamp(fx__floor_shr(p, FRACBITS), 32);
}
static inline fixed_t fx_div(fixed_t a, fixed_t b) {
    if (b == 0) { fx_flags |= FX_FLAG_DIVZ; return 0; }
    int64_t num = (int64_t)a * (int64_t)FRACUNIT;   /* no left shift of a signed value */
    int64_t q = num / (int64_t)b;                    /* toward zero; |num| < 2^48, no overflow */
    return (fixed_t)fx__clamp(q, 32);
}

/* ---- I32 conveniences used by minidoom ---- */
static inline fixed_t fx_add(fixed_t a, fixed_t b) { return (fixed_t)fxw_add_wrap(a, b, 32); }
static inline fixed_t fx_sub(fixed_t a, fixed_t b) { return (fixed_t)fxw_sub_wrap(a, b, 32); }
static inline fixed_t fx_add_sat(fixed_t a, fixed_t b) { return (fixed_t)fxw_add_sat(a, b, 32); }
static inline fixed_t fx_neg(fixed_t a) { return (fixed_t)fxw_neg_wrap(a, 32); }
static inline fixed_t fx_abs(fixed_t a) { return (fixed_t)fxw_abs_wrap(a, 32); }
static inline fixed_t fx_shl(fixed_t a, int n) { return (fixed_t)fxw_shl(a, n, 32); }
static inline fixed_t fx_shr(fixed_t a, int n) { return (fixed_t)fxw_shr(a, n, 32); }
static inline int32_t i32_div(int32_t a, int32_t b) { return (int32_t)fxw_div(a, b, 32); }
static inline int32_t i32_rem(int32_t a, int32_t b) { return (int32_t)fxw_rem(a, b, 32); }
static inline int16_t i32_to_i16(int32_t a) { int64_t r = fx__wrap(a, 16); if (r != a) fx_flags |= FX_FLAG_OVF; return (int16_t)r; }
static inline fixed_t fx_tofix(int32_t i) { return fx_shl(i, FRACBITS); }
static inline int32_t fx_fromfix(fixed_t x) { return fx_shr(x, FRACBITS); }

#endif /* DROSOPHILOS_FX_H */
