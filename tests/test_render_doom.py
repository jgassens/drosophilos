"""--reference-only checks for the generalised Doom bench driver.

No neural pipeline is built or simulated here: these runs exercise ``render_doom.main``'s
own reference-frame computation (the ``kernel_outputs`` oracle) and stop before
``build_pipeline``.
"""

from pathlib import Path

import pytest

from drosophilos.bench import render_doom


def _run(monkeypatch, tmp_path, argv, name="doom"):
    out = tmp_path / name
    monkeypatch.setattr("sys.argv", ["render_doom.py", *argv, "--out", str(out), "--reference-only"])
    render_doom.main()
    return out


def test_doom1_reference_frames_are_byte_identical_to_docs_img(monkeypatch, tmp_path):
    out = _run(monkeypatch, tmp_path, [
        "--source", "examples/doom1.c", "--width", "40", "--height", "25", "--frames", "2",
        "--inputs", "2,258",  # the H200 run's inputs (Juno job 404188): docs/img/doom40_*_reference.png are its references
    ])
    from PIL import Image
    for f in range(2):  # pixel-identical (the PNG bytes differ by the zlib that wrote them)
        got = Image.open(f"{out}_{f}_reference.png").convert("RGB")
        want = Image.open(f"docs/img/doom40_{f}_reference.png").convert("RGB")
        assert got.size == want.size and list(got.getdata()) == list(want.getdata()), f


def test_doom4_reference_runs_through_the_driver_with_its_four_streams(monkeypatch, tmp_path):
    out = _run(monkeypatch, tmp_path, [
        "--source", "examples/doom4.c", "--width", "8", "--height", "5", "--frames", "3",
        "--inputs", "6,256,0",
    ])

    frames = []
    for f in range(3):
        png_path = Path(f"{out}_{f}_reference.png")
        assert png_path.exists()
        from PIL import Image
        img = Image.open(png_path)
        w, h = img.size
        px = img.load()
        colours = {(x, y): px[x, y] for x in range(w) for y in range(h)}
        frames.append(colours)

    # the imp's sprite colours (15, 10, 9) never occur on walls, floor or sky; it must appear
    # in every frame and cover more pixels as it approaches (matches tests/test_doom4.py).
    from drosophilos.display.frame import PALETTE as _P
    imp_rgb = {_P[c] for c in (15, 10, 9)}
    counts = [sum(1 for rgb in colours.values() if rgb in imp_rgb) for colours in frames]
    assert all(counts), counts
    assert counts[0] < counts[1] < counts[2], counts


def test_doom4_compiled_streams_are_input_f_s_p():
    from drosophilos.compiler.frontend_c import compile_c
    from drosophilos.compiler.kernel import compile_program

    prog = compile_c(open("examples/doom4.c").read())
    ks = compile_program(prog, params=None, pacing="host")
    assert ks.streams == ["input", "f", "s", "p"]


def test_params_json_overrides_a_single_variable(monkeypatch, tmp_path):
    out_default = _run(monkeypatch, tmp_path, [
        "--source", "examples/doom1.c", "--width", "8", "--height", "5", "--frames", "1",
    ], name="default")
    default_bytes = Path(f"{out_default}_0_reference.png").read_bytes()

    out_overridden = _run(monkeypatch, tmp_path, [
        "--source", "examples/doom1.c", "--width", "8", "--height", "5", "--frames", "1",
        "--params", '{"heading": 16}',
    ], name="overridden")
    overridden_bytes = Path(f"{out_overridden}_0_reference.png").read_bytes()
    assert overridden_bytes != default_bytes
