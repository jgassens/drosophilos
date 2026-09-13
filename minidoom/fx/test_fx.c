/* Runs every shared vector through fx.h; exits non-zero on the first mismatch.
 * Build: clang -std=c11 -Wall -Wextra -fsanitize=undefined -fno-sanitize-recover=all -o test_fx test_fx.c
 */
#include <stdio.h>
#include <stdint.h>
#include "fx.h"
#include "vectors_generated.inc"

int fx_flags = 0;

static int64_t run(int op, int w, int64_t a, int64_t b) {
    switch (op) {
    case OP_ADD_WRAP: return fxw_add_wrap(a, b, w);
    case OP_SUB_WRAP: return fxw_sub_wrap(a, b, w);
    case OP_MUL_WRAP: return fxw_mul_wrap(a, b, w);
    case OP_NEG_WRAP: return fxw_neg_wrap(a, w);
    case OP_ABS_WRAP: return fxw_abs_wrap(a, w);
    case OP_ADD_SAT: return fxw_add_sat(a, b, w);
    case OP_SUB_SAT: return fxw_sub_sat(a, b, w);
    case OP_MUL_SAT: return fxw_mul_sat(a, b, w);
    case OP_NEG_SAT: return fxw_neg_sat(a, w);
    case OP_ABS_SAT: return fxw_abs_sat(a, w);
    case OP_MUL_Q16_16_WRAP: return fx_mul((fixed_t)a, (fixed_t)b);
    case OP_MUL_Q16_16_SAT: return fx_mul_sat((fixed_t)a, (fixed_t)b);
    case OP_DIV_Q16_16: return fx_div((fixed_t)a, (fixed_t)b);
    case OP_DIV: return fxw_div(a, b, w);
    case OP_REM: return fxw_rem(a, b, w);
    case OP_DIVU: return fxw_divu(a, b, w);
    case OP_REMU: return fxw_remu(a, b, w);
    case OP_SHL: return fxw_shl(a, b, w);
    case OP_SHR: return fxw_shr(a, b, w);
    case OP_SHRU: return fxw_shru(a, b, w);
    case OP_AND: return fxw_and(a, b, w);
    case OP_OR: return fxw_or(a, b, w);
    case OP_XOR: return fxw_xor(a, b, w);
    case OP_NOT: return fxw_not(a, w);
    default: fprintf(stderr, "unknown op %d\n", op); return 0;
    }
}

int main(void) {
    int failures = 0;
    for (int i = 0; i < N_VECTORS; i++) {
        const struct vec *v = &VECTORS[i];
        fx_flags = 0;
        int64_t got = run(v->op, v->w, v->a, v->b);
        if (got != v->value || fx_flags != v->flags) {
            fprintf(stderr, "vector %d: op %d w %d a %lld b %lld: expected %lld flags %d, got %lld flags %d\n",
                    i, v->op, v->w, (long long)v->a, (long long)v->b, (long long)v->value, v->flags,
                    (long long)got, fx_flags);
            if (++failures > 20) break;
        }
    }
    printf("%d vectors, %d failures\n", N_VECTORS, failures);
    return failures ? 1 : 0;
}
