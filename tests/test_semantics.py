"""Arithmetic semantics: Python spec self-checks and the C helpers against shared vectors."""

import shutil
import subprocess
from pathlib import Path

import pytest

from drosophilos.isa import semantics as S

ROOT = Path(__file__).resolve().parents[1]


def test_doom_fixedmul_convention():
    # 1.5 * 2.0 = 3.0 ; (-1.5) * 2.0 = -3.0 ; floor toward -inf on inexact negative products
    one_half = S.FRACUNIT * 3 // 2
    assert S.mul_q16_wrap(one_half, 2 * S.FRACUNIT).value == 3 * S.FRACUNIT
    assert S.mul_q16_wrap(-one_half, 2 * S.FRACUNIT).value == -3 * S.FRACUNIT
    assert S.mul_q16_wrap(-1, 1).value == -1  # -2^-32 floors to -1 quantum
    assert S.mul_q16_wrap(S.imax(32), S.FRACUNIT).value == S.imax(32)
    r = S.mul_q16_wrap(S.imax(32), 2 * S.FRACUNIT)
    assert r.flags.ovf == 1 and r.value == S.wrap(2 * S.imax(32), 32)


def test_div_q16_saturates_and_flags():
    assert S.div_q16(3 * S.FRACUNIT, 2 * S.FRACUNIT).value == S.FRACUNIT * 3 // 2
    assert S.div_q16(-7 * S.FRACUNIT, 2 * S.FRACUNIT).value == -(7 * S.FRACUNIT) // 2
    r = S.div_q16(5, 0)
    assert r.value == 0 and r.flags.divz == 1
    r = S.div_q16(S.imax(32), 1)
    assert r.value == S.imax(32) and r.flags.sat == 1


def test_wrap_and_sat_corners():
    assert S.add_wrap(S.imax(32), 1) == (S.imin(32), S.Flags(ovf=1))
    assert S.add_sat(S.imax(32), 1) == (S.imax(32), S.Flags(sat=1))
    assert S.neg_wrap(S.imin(16), 16) == (S.imin(16), S.Flags(ovf=1))
    assert S.div_i(S.imin(8), -1, 8) == (S.imin(8), S.Flags(ovf=1))
    assert S.rem_i(S.imin(8), -1, 8) == (0, S.NONE)
    assert S.shr(-5, 1).value == -3  # floor
    assert S.shl(1, 32).flags.shift == 1
    assert S.trunc(0x1234_5678, 32, 16) == (0x5678, S.Flags(ovf=1))


def test_vectors_are_deterministic():
    a = S.generate_vectors()
    b = S.generate_vectors()
    assert a == b and len(a) > 3000


@pytest.mark.skipif(shutil.which("clang") is None, reason="clang not available")
def test_c_helpers_match_vectors(tmp_path):
    inc = tmp_path / "vectors_generated.inc"
    S.write_vectors(tmp_path / "vectors.json", inc)
    src_dir = ROOT / "minidoom" / "fx"
    exe = tmp_path / "test_fx"
    subprocess.run(
        [
            "clang", "-std=c11", "-Wall", "-Wextra", "-Werror",
            "-fsanitize=undefined", "-fno-sanitize-recover=all",
            "-I", str(tmp_path), "-I", str(src_dir),
            "-o", str(exe), str(src_dir / "test_fx.c"),
        ],
        check=True,
    )
    out = subprocess.run([str(exe)], capture_output=True, text=True)
    assert out.returncode == 0, out.stdout + out.stderr
    assert "0 failures" in out.stdout
