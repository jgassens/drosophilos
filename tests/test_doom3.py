"""Compile-only and host-reference checks for the Doom sprite renderer.

No neural pipeline is run here: ``kernel_outputs`` is the resident-kernel oracle, and the
small C-golden schedule stays within golden.py's fixed 64-word input shim.
"""

from collections import Counter
from pathlib import Path
import tempfile

from drosophilos.compiler.frontend_c import compile_c
from drosophilos.compiler.golden import run_golden
from drosophilos.compiler.kernel import compile_program, kernel_outputs
from drosophilos.display.frame import write_png
from drosophilos.isa.ir import interpret


SOURCE_PATH = Path("examples/doom3.c")
PARAMS = {
    "px": 120,
    "py": 120,
    "heading": 48,
    "sdepth": 3,
    "scol": 81,
    "hw": 19,
    "sh": 39,
    "recip2w": 54,
}
TICKS = (6, 0)
SPRITE_COLOURS = {9, 10, 15}


def _schedule(pixel_tokens):
    schedule = []
    for tick, pixels in zip(TICKS, pixel_tokens):
        schedule += [("input", col) for col in range(160)]
        schedule += [("s", col) for col in range(160)]
        schedule += [("p", token) for token in pixels]
        schedule += [("f", tick)]
    return schedule


def test_doom3_compiles_and_reference_frames_match_c_golden(monkeypatch):
    source = SOURCE_PATH.read_text()
    prog = compile_c(source)
    spec = compile_program(prog, params=PARAMS, pacing="host")

    assert prog.width == 16
    assert spec.streams == ["input", "f", "s", "p"]
    assert spec.rams == {"hbuf", "dbuf", "ubuf", "sbuf", "stex"}
    assert Counter(cell["stream"] for cell in spec.cells) == {
        "input:f": 60,
        "input": 67,
        "input:s": 18,
        "input:p": 52,
    }

    # This is the render_doom.py host schedule, with the added sprite-column phase.
    rows, cols = range(100), range(160)
    pixels = [(row << 8) | col for row in rows for col in cols]
    ref_schedule = []
    for tick in TICKS:
        ref_schedule += [("input", col) for col in cols]
        ref_schedule += [("s", col) for col in cols]
        ref_schedule += [("p", token) for token in pixels]
        ref_schedule += [("f", tick)]
    ref_out = kernel_outputs(spec, ref_schedule)

    frames = []
    cursor = 0
    for frame_number in range(2):
        column_out = ref_out[cursor : cursor + 160]
        cursor += 160
        sprite_out = ref_out[cursor : cursor + 160]
        cursor += 160
        frame = {
            (col, row): ref_out[cursor + j][0]
            for j, (row, col) in enumerate((row, col) for row in rows for col in cols)
        }
        cursor += 16000
        cursor += 1  # the tick token; its state-carrier outputs are not pixels
        frames.append(frame)
        write_png(frame, 160, 100, f"docs/img/doom3_{frame_number}_reference.png", scale=4)

        if frame_number == 0:
            dbuf = [values[1] for values in column_out]
            sbuf = [values[0] for values in sprite_out]
            assert any(sbuf[col] == 39 for col in range(62, 100))
            assert any(sbuf[col] == 0 for col in range(62, 100))
            assert all(sbuf[col] == (39 if 3 < dbuf[col] else 0) for col in range(62, 100))
            assert all(sbuf[col] == 0 for col in range(160) if not 62 <= col < 100)

    sprite_pixels = [
        (col, row, colour)
        for (col, row), colour in frames[0].items()
        if colour in SPRITE_COLOURS
    ]
    assert sprite_pixels
    assert all(62 <= col < 100 for col, _, _ in sprite_pixels)
    assert all(31 <= row < 70 for _, row, _ in sprite_pixels)
    frame1_sprite = [(col, row) for (col, row), colour in frames[1].items() if colour in SPRITE_COLOURS]
    assert frame1_sprite and all(19 <= col < 57 for col, _ in frame1_sprite)

    # golden.py's runtime has 64 input slots.  Two 31-pixel passes plus their two ticks fit
    # exactly and still exercise both camera states against clang and the IR interpreter.
    golden_source = source.replace("p = 16000;", "p = 31;")
    sample_pixels = [pixels[j * 503 % len(pixels)] for j in range(31)]
    golden_inputs = sample_pixels + [TICKS[0]] + sample_pixels + [TICKS[1]]
    # The managed macOS test sandbox has no writable /tmp for xcrun's cache or clang's
    # intermediates; both locations remain inside this test's owned doom3 image path.
    developer_dir = Path("/Applications/Xcode.app/Contents/Developer")
    if developer_dir.exists():
        monkeypatch.setenv("DEVELOPER_DIR", str(developer_dir))
    with tempfile.TemporaryDirectory(prefix="doom3_tmp_", dir="docs/img") as compiler_tmp:
        monkeypatch.setenv("TMPDIR", str(Path(compiler_tmp).resolve()))
        golden = run_golden(golden_source, golden_inputs, [], 16)["outs"]
    ir_result = interpret(compile_c(golden_source), golden_inputs, max_steps=1_000_000)
    assert ir_result["halted"]
    ir = ir_result["outs"]
    sample_spec = compile_program(compile_c(golden_source), params=PARAMS, pacing="host")
    sample_out = kernel_outputs(sample_spec, _schedule((sample_pixels, sample_pixels)))
    kernel_pixels = [values[0] for (stream, _), values in zip(_schedule((sample_pixels, sample_pixels)), sample_out) if stream == "p"]
    assert kernel_pixels == ir == golden
