# Arithmetic semantics (normative)

One specification, three implementations that must agree on shared test vectors:
`isa/semantics.py` (Python, used by the IR interpreter), `minidoom/fx/fx.h` (portable C,
used by the golden reference, built with UBSan), and the FlyISA lowering (neural circuits).
Ordinary C operators on signed values are **not** used for anything below; every operation
goes through a named helper, so there is no undefined behaviour to inherit.

## 1. Types

| name | representation | range |
|---|---|---|
| `I8`, `I16`, `I32` | two's complement, width *w* | `[−2^(w−1), 2^(w−1) − 1]` |
| `U8`, `U16`, `U32` | unsigned, width *w* | `[0, 2^w − 1]` |
| `Q16_16` | `I32` interpreted with 16 fractional bits (`FRACUNIT = 65536`) | `[−32768.0, 32767.99998]` |
| `B1` | boolean, one exact bit | `{0, 1}` |
| `I64` | intermediate only, never stored in game state | `[−2^63, 2^63 − 1]` |

`MIN_w = −2^(w−1)`, `MAX_w = 2^(w−1) − 1`.

## 2. Status flags

Every operation returns a value and a status word `{SAT, DIVZ, SHIFT, OVF}` of exact bits.
A flag that is not mentioned for an operation is 0. Flags are values, not traps; the
program decides what to do with them. `CHECK.RANGE` and `QUANTIZE` also produce `RANGE`
and `INDET` respectively (see FlyISA).

## 3. Operations

### Wrapping (modular) integer arithmetic
- `ADD.WRAP.Iw(a, b)`, `SUB.WRAP.Iw(a, b)`, `MUL.WRAP.Iw(a, b)`: exact result reduced
  modulo `2^w` into the signed range. `OVF = 1` if the exact result was outside
  `[MIN_w, MAX_w]`. (Doom's fixed-point convention.)
- `NEG.WRAP.Iw(a)`: `−a` modulo `2^w`; `NEG(MIN_w) = MIN_w`, `OVF = 1`.
- `ABS.WRAP.Iw(a)`: `|a|` modulo `2^w`; `ABS(MIN_w) = MIN_w`, `OVF = 1`.

### Saturating integer arithmetic
- `ADD.SAT.Iw`, `SUB.SAT.Iw`, `MUL.SAT.Iw`, `NEG.SAT.Iw`, `ABS.SAT.Iw`: exact result
  clamped to `[MIN_w, MAX_w]`; `SAT = 1` when clamping changed the value.

### Fixed point (`Q16_16`)
- `MUL.Q16_16.WRAP(a, b)`: `p = a·b` as exact `I64`; `r = p >> 16` (arithmetic shift,
  i.e. `floor(p / 65536)`, rounds toward −∞ for negative products); result `r mod 2^32`
  into `I32`; `OVF = 1` if `r` outside `I32`. This is Doom's `FixedMul`.
- `MUL.Q16_16.SAT(a, b)`: same `r`, clamped to `I32`; `SAT` on clamp.
- `DIV.Q16_16(a, b)`: if `b == 0`: result `0`, `DIVZ = 1`. Else `q = (a·65536) / b` with
  the exact `I64` numerator and **truncation toward zero**; if `q` outside `I32`: clamp,
  `SAT = 1`. (Doom's `FixedDiv` saturates when `|a| >> 14 ≥ |b|`; this spec saturates on
  the exact same overflow condition and otherwise agrees with 64-bit `FixedDiv` ports.)

### Integer division
- `DIV.Iw(a, b)`: truncation toward zero (C99). `b == 0` → result `0`, `DIVZ = 1`.
  `MIN_w / −1` → result `MIN_w`, `OVF = 1`.
- `REM.Iw(a, b)`: `a − b·DIV.Iw(a, b)` (sign follows the dividend). `b == 0` → result
  `a`, `DIVZ = 1`. `MIN_w REM −1` → `0`.
- `DIVU.Uw`, `REMU.Uw`: unsigned; `b == 0` → `0` / `a`, `DIVZ = 1`.

### Shifts
- `SHL.Iw(a, n)`: valid for `0 ≤ n ≤ w − 1`: `(a · 2^n) mod 2^w` into signed; `OVF = 1`
  if bits were lost. `n` outside range: result `0`, `SHIFT = 1`.
- `SHR.Iw(a, n)` (arithmetic): valid `0 ≤ n ≤ w − 1`: `floor(a / 2^n)`. Invalid `n`:
  result `0`, `SHIFT = 1`.
- `SHRU.Uw(a, n)` (logical): valid `0 ≤ n ≤ w − 1`; invalid → `0`, `SHIFT = 1`.

### Bitwise
- `AND`, `OR`, `XOR`, `NOT` on the *w*-bit pattern; no flags.

### Comparison
- `CMP.SIGNED.Iw(a, b)` → `(EQ, LT, GT)` as three exact bits, exactly one set.
- `CMP.UNSIGNED.Uw(a, b)` → same, on unsigned interpretation.

### Width conversion
- `TRUNC.Iw→Iv(a)` (`v < w`): low *v* bits reinterpreted as signed; `OVF = 1` if the value
  changed.
- `SAT.Iw→Iv(a)`: clamp to `[MIN_v, MAX_v]`; `SAT` on clamp.
- `SEXT.Iv→Iw(a)`, `ZEXT.Uv→Uw(a)`: exact, no flags.
- `TOFIX(i) = SHL.I32(i, 16)` semantics (`OVF` if `|i| ≥ 32768`); `FROMFIX(x) = SHR.I32(x, 16)`
  (floor).

### Fixed-point helpers used by `minidoom` (names in C and in DrosoC)
| helper | operation |
|---|---|
| `fx_add(a,b)` | `ADD.WRAP.I32` |
| `fx_sub(a,b)` | `SUB.WRAP.I32` |
| `fx_add_sat(a,b)` | `ADD.SAT.I32` |
| `fx_mul(a,b)` | `MUL.Q16_16.WRAP` |
| `fx_mul_sat(a,b)` | `MUL.Q16_16.SAT` |
| `fx_div(a,b)` | `DIV.Q16_16` |
| `fx_neg(a)` | `NEG.WRAP.I32` |
| `fx_abs(a)` | `ABS.WRAP.I32` |
| `fx_shl(a,n)` / `fx_shr(a,n)` | `SHL.I32` / `SHR.I32` |
| `i32_div(a,b)` / `i32_rem(a,b)` | `DIV.I32` / `REM.I32` |
| `i32_to_i16(a)` | `TRUNC.I32→I16` |

The C versions carry the status word in a thread-local `fx_flags` accumulator that the
golden dumper serialises with the game state, so flag behaviour is part of the comparison.

## 4. Test vectors

`isa/vectors.json` is generated from `semantics.py` and covers, for every operation: all
corner values (`MIN`, `MIN+1`, `−1`, `0`, `1`, `MAX−1`, `MAX`, and `±FRACUNIT` multiples for
`Q16_16`), random pairs, and every flag path. The C helpers are compiled with
`-fsanitize=undefined -fno-sanitize-recover` and must reproduce every vector. The neural
lowering is certified against the same vectors at the transaction level.

## 5. Canonical state serialisation

Comparisons between the C reference, the IR interpreter, and neural execution use
`compiler/canon.py`: fields in declared order, each written at its declared width as
little-endian two's complement, no padding, no pointers, followed by the flag accumulator.
Host memory layout never enters a comparison.
