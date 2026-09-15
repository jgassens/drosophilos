"""Compile-only reference checks for the chasing Doom sprite renderer.

``kernel_outputs`` is the resident-kernel oracle.  This test deliberately never builds or
simulates a neural pipeline.
"""

from collections import Counter
from pathlib import Path
import re
import tempfile

from drosophilos.compiler.frontend_c import compile_c
from drosophilos.compiler.golden import run_golden
from drosophilos.compiler.kernel import compile_program, kernel_outputs
from drosophilos.display.frame import write_png
from drosophilos.isa.ir import interpret


SOURCE_PATH = Path("examples/doom4.c")
PARAMS = {
    "px": 120, "py": 120, "tx": 168, "ty": 88, "heading": 48,
    "sdepth": 3, "scol": 81, "hw": 19, "sh": 39, "recip2w": 54,
}
# Turn east, then move once.  The imp takes three collision-tested chase steps and its
# projected billboard shifts left and widens on each frame.
TICKS = (6, 256, 0)


def _schedule(pixel_tokens):
    schedule = []
    for tick, pixels in zip(TICKS, pixel_tokens):
        schedule += [("input", col) for col in range(160)]
        schedule += [("s", col) for col in range(160)]
        schedule += [("p", token) for token in pixels]
        schedule += [("f", tick)]
    return schedule


def _image_values(source, name):
    body = re.search(rf"static u16 {name}\[\d+\] = \{{([^}}]+)\}}", source).group(1)
    return [int(value.strip()) for value in body.split(",")]


def _expected_thing_positions(source):
    """The P_Move-shaped chase in plain Python, including the same grid-cell collision."""
    grid, cost, sint = (_image_values(source, name) for name in ("grid", "cost", "sint"))
    px, py, tx, ty, heading = 120, 120, 168, 88, 48
    positions = [(tx, ty)]
    for token in TICKS:
        heading = (heading + (token & 63)) & 63
        fwd = (token >> 8) & 3
        nx, ny = px, py
        dx = cost[heading] if cost[heading] < 32768 else cost[heading] - 65536
        dy = sint[heading] if sint[heading] < 32768 else sint[heading] - 65536
        if fwd == 1:
            nx, ny = (px + dx) & 65535, (py + dy) & 65535
        if fwd == 2:
            nx, ny = (px - dx) & 65535, (py - dy) & 65535
        if grid[((ny >> 4) << 4) | (nx >> 4)] == 0:
            px, py = nx, ny
        nx, ny = tx, ty
        if abs(tx - px) >= abs(ty - py):
            nx += 8 if tx < px else -8
        else:
            ny += 8 if ty < py else -8
        destination = ((ny >> 4) << 4) | (nx >> 4)
        if grid[destination] == 0 and destination != (((py >> 4) << 4) | (px >> 4)):
            tx, ty = nx, ny
        positions.append((tx, ty))
    return positions


def test_doom4_moving_thing_matches_kernel_and_c_golden(monkeypatch):
    source = SOURCE_PATH.read_text()
    expected_positions = _expected_thing_positions(source)
    assert expected_positions == [(168, 88), (160, 88), (152, 88), (144, 88)]

    prog = compile_c(source)
    spec = compile_program(prog, params=PARAMS, pacing="host")
    assert prog.width == 16
    assert spec.streams == ["input", "f", "s", "p"]
    assert spec.rams == {"hbuf", "dbuf", "ubuf", "sbuf", "stex"}
    assert Counter(cell["stream"] for cell in spec.cells) == {
        "input:f": 96, "input": 67, "input:s": 18, "input:p": 55,
    }

    rows, cols = range(100), range(160)
    pixels = [(row << 8) | col for row in rows for col in cols]
    ref_out = kernel_outputs(spec, _schedule((pixels, pixels, pixels)))
    frames, bands = [], []
    cursor = 0
    for frame_number in range(3):
        cursor += 160  # column outputs
        sprite_out = ref_out[cursor : cursor + 160]
        cursor += 160
        frame = {(col, row): ref_out[cursor + j][0]
                 for j, (row, col) in enumerate((row, col) for row in rows for col in cols)}
        cursor += 16000
        tick_state = ref_out[cursor]
        cursor += 1
        states = {name: tick_state[spec.outputs.index(cell)] for name, cell in spec.state_cells.items()}
        assert (states["tx"], states["ty"]) == expected_positions[frame_number + 1]
        band = [col for col, values in enumerate(sprite_out) if values[0]]
        assert band
        bands.append(band)
        frames.append(frame)
        write_png(frame, 160, 100, f"docs/img/doom4_{frame_number}_reference.png", scale=4)

    # The wall clips the first billboard, while the closer later positions widen and shift it.
    assert [len(band) for band in bands] == [26, 40, 64]
    assert [min(band) for band in bands] == [62, 0, 0]
    assert all(len(bands[n]) < len(bands[n + 1]) for n in range(2))

    # golden.py has 64 input slots: two 30-pixel passes and their ticks fit exactly.
    golden_source = source.replace("f = 3;", "f = 2;").replace("p = 16000;", "p = 30;")
    sample_pixels = [pixels[j * 503 % len(pixels)] for j in range(30)]
    golden_inputs = sum((sample_pixels + [tick] for tick in TICKS[:2]), [])
    developer_dir = Path("/Applications/Xcode.app/Contents/Developer")
    if developer_dir.exists():
        monkeypatch.setenv("DEVELOPER_DIR", str(developer_dir))
    with tempfile.TemporaryDirectory(prefix="doom4_tmp_", dir="docs/img") as compiler_tmp:
        monkeypatch.setenv("TMPDIR", str(Path(compiler_tmp).resolve()))
        golden = run_golden(golden_source, golden_inputs, [], 16)["outs"]
    ir_result = interpret(compile_c(golden_source), golden_inputs, max_steps=1_000_000)
    assert ir_result["halted"]
    sample_spec = compile_program(compile_c(golden_source), params=PARAMS, pacing="host")
    sample_schedule = _schedule((sample_pixels, sample_pixels))
    sample_out = kernel_outputs(sample_spec, sample_schedule)
    kernel_pixels = [values[0] for (stream, _), values in zip(sample_schedule, sample_out)
                     if stream == "p"]
    assert kernel_pixels == ir_result["outs"] == golden
