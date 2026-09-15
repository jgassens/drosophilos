"""Executable arithmetic semantics (semantics.md). Pure Python integers, no C behaviour.

Every function returns `Result(value, flags)`. `generate_vectors()` produces the shared
test vectors that the C helpers (`minidoom/fx/fx.h`) and the neural lowering must match.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, NamedTuple

FRACBITS = 16
FRACUNIT = 1 << FRACBITS


@dataclass(frozen=True)
class Flags:
    sat: int = 0
    divz: int = 0
    shift: int = 0
    ovf: int = 0

    @property
    def word(self) -> int:
        return self.sat | (self.divz << 1) | (self.shift << 2) | (self.ovf << 3)

    @classmethod
    def from_word(cls, w: int) -> "Flags":
        return cls(sat=w & 1, divz=(w >> 1) & 1, shift=(w >> 2) & 1, ovf=(w >> 3) & 1)


NONE = Flags()


class Result(NamedTuple):
    value: int
    flags: Flags


def imin(w: int) -> int:
    return -(1 << (w - 1))


def imax(w: int) -> int:
    return (1 << (w - 1)) - 1


def umax(w: int) -> int:
    return (1 << w) - 1


def _check_signed(a: int, w: int) -> None:
    if not (imin(w) <= a <= imax(w)):
        raise ValueError(f"{a} is not a signed {w}-bit value")


def _check_unsigned(a: int, w: int) -> None:
    if not (0 <= a <= umax(w)):
        raise ValueError(f"{a} is not an unsigned {w}-bit value")


def wrap(x: int, w: int) -> int:
    """Reduce an exact integer modulo 2^w into the signed w-bit range."""
    x &= umax(w)
    return x - (1 << w) if x > imax(w) else x


def clamp(x: int, w: int) -> tuple[int, int]:
    if x < imin(w):
        return imin(w), 1
    if x > imax(w):
        return imax(w), 1
    return x, 0


# ---- wrapping -------------------------------------------------------------------------
def add_wrap(a: int, b: int, w: int = 32) -> Result:
    _check_signed(a, w); _check_signed(b, w)
    exact = a + b
    r = wrap(exact, w)
    return Result(r, Flags(ovf=int(r != exact)))


def sub_wrap(a: int, b: int, w: int = 32) -> Result:
    _check_signed(a, w); _check_signed(b, w)
    exact = a - b
    r = wrap(exact, w)
    return Result(r, Flags(ovf=int(r != exact)))


def mul_wrap(a: int, b: int, w: int = 32) -> Result:
    _check_signed(a, w); _check_signed(b, w)
    exact = a * b
    r = wrap(exact, w)
    return Result(r, Flags(ovf=int(r != exact)))


def neg_wrap(a: int, w: int = 32) -> Result:
    _check_signed(a, w)
    r = wrap(-a, w)
    return Result(r, Flags(ovf=int(r != -a)))


def abs_wrap(a: int, w: int = 32) -> Result:
    _check_signed(a, w)
    r = wrap(abs(a), w)
    return Result(r, Flags(ovf=int(r != abs(a))))


# ---- saturating -----------------------------------------------------------------------
def add_sat(a: int, b: int, w: int = 32) -> Result:
    _check_signed(a, w); _check_signed(b, w)
    r, s = clamp(a + b, w)
    return Result(r, Flags(sat=s))


def sub_sat(a: int, b: int, w: int = 32) -> Result:
    _check_signed(a, w); _check_signed(b, w)
    r, s = clamp(a - b, w)
    return Result(r, Flags(sat=s))


def mul_sat(a: int, b: int, w: int = 32) -> Result:
    _check_signed(a, w); _check_signed(b, w)
    r, s = clamp(a * b, w)
    return Result(r, Flags(sat=s))


def neg_sat(a: int, w: int = 32) -> Result:
    _check_signed(a, w)
    r, s = clamp(-a, w)
    return Result(r, Flags(sat=s))


def abs_sat(a: int, w: int = 32) -> Result:
    _check_signed(a, w)
    r, s = clamp(abs(a), w)
    return Result(r, Flags(sat=s))


# ---- fixed point Q16.16 ---------------------------------------------------------------
def mul_q16_wrap(a: int, b: int) -> Result:
    _check_signed(a, 32); _check_signed(b, 32)
    p = a * b
    r_exact = p >> FRACBITS  # Python >> floors toward -inf on negatives
    r = wrap(r_exact, 32)
    return Result(r, Flags(ovf=int(r != r_exact)))


def mul_q16_sat(a: int, b: int) -> Result:
    _check_signed(a, 32); _check_signed(b, 32)
    r_exact = (a * b) >> FRACBITS
    r, s = clamp(r_exact, 32)
    return Result(r, Flags(sat=s))


def _trunc_div(a: int, b: int) -> int:
    q = abs(a) // abs(b)
    return -q if (a < 0) != (b < 0) else q


def div_q16(a: int, b: int) -> Result:
    _check_signed(a, 32); _check_signed(b, 32)
    if b == 0:
        return Result(0, Flags(divz=1))
    q = _trunc_div(a * FRACUNIT, b)
    r, s = clamp(q, 32)
    return Result(r, Flags(sat=s))


# ---- integer division -----------------------------------------------------------------
def div_i(a: int, b: int, w: int = 32) -> Result:
    _check_signed(a, w); _check_signed(b, w)
    if b == 0:
        return Result(0, Flags(divz=1))
    if a == imin(w) and b == -1:
        return Result(imin(w), Flags(ovf=1))
    return Result(_trunc_div(a, b), NONE)


def rem_i(a: int, b: int, w: int = 32) -> Result:
    _check_signed(a, w); _check_signed(b, w)
    if b == 0:
        return Result(a, Flags(divz=1))
    if a == imin(w) and b == -1:
        return Result(0, NONE)
    return Result(a - b * _trunc_div(a, b), NONE)


def divu(a: int, b: int, w: int = 32) -> Result:
    _check_unsigned(a, w); _check_unsigned(b, w)
    if b == 0:
        return Result(0, Flags(divz=1))
    return Result(a // b, NONE)


def remu(a: int, b: int, w: int = 32) -> Result:
    _check_unsigned(a, w); _check_unsigned(b, w)
    if b == 0:
        return Result(a, Flags(divz=1))
    return Result(a % b, NONE)


# ---- shifts ---------------------------------------------------------------------------
def shl(a: int, n: int, w: int = 32) -> Result:
    _check_signed(a, w)
    if not (0 <= n <= w - 1):
        return Result(0, Flags(shift=1))
    exact = a << n
    r = wrap(exact, w)
    return Result(r, Flags(ovf=int(r != exact)))


def shr(a: int, n: int, w: int = 32) -> Result:
    _check_signed(a, w)
    if not (0 <= n <= w - 1):
        return Result(0, Flags(shift=1))
    return Result(a >> n, NONE)  # floor


def shru(a: int, n: int, w: int = 32) -> Result:
    _check_unsigned(a, w)
    if not (0 <= n <= w - 1):
        return Result(0, Flags(shift=1))
    return Result(a >> n, NONE)


# ---- bitwise ---------------------------------------------------------------------------
def shl(a: int, k: int, w: int = 32) -> Result:
    """Logical shift left by a constant 0 <= k < w on the unsigned pattern; bits shifted out are
    dropped (WRAP). A count >= w is the SHIFT fault."""
    if not 0 <= k < w:
        return Result(0, Flags(shift=1))
    return Result(wrap((a & umax(w)) << k, w), NONE)


def shr(a: int, k: int, w: int = 32) -> Result:
    """Logical shift right by a constant 0 <= k < w on the unsigned pattern (zero fill)."""
    if not 0 <= k < w:
        return Result(0, Flags(shift=1))
    return Result((a & umax(w)) >> k, NONE)


def and_(a: int, b: int, w: int = 32) -> Result:
    return Result(wrap((a & umax(w)) & (b & umax(w)), w), NONE)


def or_(a: int, b: int, w: int = 32) -> Result:
    return Result(wrap((a & umax(w)) | (b & umax(w)), w), NONE)


def xor(a: int, b: int, w: int = 32) -> Result:
    return Result(wrap((a & umax(w)) ^ (b & umax(w)), w), NONE)


def not_(a: int, w: int = 32) -> Result:
    return Result(wrap(~a & umax(w), w), NONE)


# ---- comparison ------------------------------------------------------------------------
def cmp_s(a: int, b: int, w: int = 32) -> tuple[int, int, int]:
    _check_signed(a, w); _check_signed(b, w)
    return (int(a == b), int(a < b), int(a > b))


def cmp_u(a: int, b: int, w: int = 32) -> tuple[int, int, int]:
    _check_unsigned(a, w); _check_unsigned(b, w)
    return (int(a == b), int(a < b), int(a > b))


# ---- width conversion ------------------------------------------------------------------
def trunc(a: int, w: int, v: int) -> Result:
    _check_signed(a, w)
    r = wrap(a, v)
    return Result(r, Flags(ovf=int(r != a)))


def sat_narrow(a: int, w: int, v: int) -> Result:
    _check_signed(a, w)
    r, s = clamp(a, v)
    return Result(r, Flags(sat=s))


def sext(a: int, v: int, w: int) -> Result:
    _check_signed(a, v)
    return Result(a, NONE)


def zext(a: int, v: int, w: int) -> Result:
    _check_unsigned(a, v)
    return Result(a, NONE)


def tofix(i: int) -> Result:
    return shl(i, FRACBITS, 32)


def fromfix(x: int) -> Result:
    return shr(x, FRACBITS, 32)


# ---- vector generation -----------------------------------------------------------------
# op name -> (function, kind). kind: "bb" two signed operands of width w; "b" one operand;
# "q" two Q16.16 operands; "bn" operand + shift count; "uu" unsigned pair; "un" unsigned + count.
OPS: dict[str, tuple[Callable, str]] = {
    "ADD.WRAP": (add_wrap, "bb"),
    "SUB.WRAP": (sub_wrap, "bb"),
    "MUL.WRAP": (mul_wrap, "bb"),
    "NEG.WRAP": (neg_wrap, "b"),
    "ABS.WRAP": (abs_wrap, "b"),
    "ADD.SAT": (add_sat, "bb"),
    "SUB.SAT": (sub_sat, "bb"),
    "MUL.SAT": (mul_sat, "bb"),
    "NEG.SAT": (neg_sat, "b"),
    "ABS.SAT": (abs_sat, "b"),
    "MUL.Q16_16.WRAP": (mul_q16_wrap, "q"),
    "MUL.Q16_16.SAT": (mul_q16_sat, "q"),
    "DIV.Q16_16": (div_q16, "q"),
    "DIV": (div_i, "bb"),
    "REM": (rem_i, "bb"),
    "DIVU": (divu, "uu"),
    "REMU": (remu, "uu"),
    "SHL": (shl, "bn"),
    "SHR": (shr, "bn"),
    "SHRU": (shru, "un"),
    "AND": (and_, "bb"),
    "OR": (or_, "bb"),
    "XOR": (xor, "bb"),
    "NOT": (not_, "b"),
}

WIDTHS = (8, 16, 32)


def _corners(w: int) -> list[int]:
    return sorted({imin(w), imin(w) + 1, -1, 0, 1, imax(w) - 1, imax(w), FRACUNIT if w == 32 else 0,
                   -FRACUNIT if w == 32 else 0, 3 * FRACUNIT // 2 if w == 32 else 0})


def _ucorners(w: int) -> list[int]:
    return sorted({0, 1, umax(w) - 1, umax(w), umax(w) // 2})


def generate_vectors(seed: int = 20260913, n_random: int = 40) -> list[dict]:
    import random

    rng = random.Random(seed)
    out: list[dict] = []

    def emit(op, w, a, b, res):
        out.append({"op": op, "w": w, "a": a, "b": b, "value": res.value, "flags": res.flags.word})

    for op, (fn, kind) in OPS.items():
        widths = (32,) if kind in ("q",) else WIDTHS
        for w in widths:
            if kind in ("bb", "q"):
                pairs = [(a, b) for a in _corners(w) for b in _corners(w)]
                pairs += [(rng.randint(imin(w), imax(w)), rng.randint(imin(w), imax(w))) for _ in range(n_random)]
                for a, b in pairs:
                    emit(op, w, a, b, fn(a, b) if kind == "q" else fn(a, b, w))
            elif kind == "b":
                vals = _corners(w) + [rng.randint(imin(w), imax(w)) for _ in range(n_random)]
                for a in vals:
                    emit(op, w, a, 0, fn(a, w))
            elif kind == "bn":
                vals = _corners(w) + [rng.randint(imin(w), imax(w)) for _ in range(n_random)]
                for a in vals:
                    for n in (0, 1, w // 2, w - 1, w, w + 1, -1):
                        emit(op, w, a, n, fn(a, n, w))
            elif kind == "uu":
                pairs = [(a, b) for a in _ucorners(w) for b in _ucorners(w)]
                pairs += [(rng.randint(0, umax(w)), rng.randint(0, umax(w))) for _ in range(n_random)]
                for a, b in pairs:
                    emit(op, w, a, b, fn(a, b, w))
            elif kind == "un":
                vals = _ucorners(w) + [rng.randint(0, umax(w)) for _ in range(n_random)]
                for a in vals:
                    for n in (0, 1, w // 2, w - 1, w, w + 1, -1):
                        emit(op, w, a, n, fn(a, n, w))
    return out


OP_IDS = {name: i for i, name in enumerate(OPS)}


def write_vectors(json_path: Path, c_inc_path: Path | None = None) -> int:
    vectors = generate_vectors()
    json_path.write_text(json.dumps({"ops": OP_IDS, "vectors": vectors}, indent=0))
    if c_inc_path is not None:
        lines = ["/* generated by drosophilos.isa.semantics.write_vectors; do not edit */"]
        for name, i in OP_IDS.items():
            lines.append(f"#define OP_{name.replace('.', '_')} {i}")
        lines.append("static const struct vec { int op; int w; int64_t a; int64_t b; int64_t value; int flags; } VECTORS[] = {")
        for v in vectors:
            lines.append(f"  {{{OP_IDS[v['op']]}, {v['w']}, {v['a']}LL, {v['b']}LL, {v['value']}LL, {v['flags']}}},")
        lines.append("};")
        lines.append(f"static const int N_VECTORS = {len(vectors)};")
        c_inc_path.write_text("\n".join(lines) + "\n")
    return len(vectors)


if __name__ == "__main__":
    here = Path(__file__).resolve().parent
    n = write_vectors(here / "vectors.json", here.parents[1] / "minidoom" / "fx" / "vectors_generated.inc")
    print(f"wrote {n} vectors")
