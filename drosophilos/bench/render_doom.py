"""The Doom-shaped program (`examples/doom1.c`: a tick kernel moving the player, a column pass
casting rays into RAM buffers, a pixel pass) rendered frame by frame. Columns are dealt to the
copies, and each copy renders the pixels of its own columns (its buffers hold their heights);
every copy gets the same tick token (replicated state). Frames go to PNGs beside the
references.

    python -m drosophilos.bench.render_doom --width 40 --height 25 --frames 2 --nodes 8 --device cpu
"""

import argparse
import json
import time

from ..compiler.frontend_c import compile_c
from ..compiler.kernel import compile_program, kernel_outputs
from ..display.frame import write_png
from ..lib.kernel import build_pipeline, run_pipeline, run_pipeline_batched
from ..sim.model import Params


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="examples/doom1.c")
    ap.add_argument("--width", type=int, default=40)
    ap.add_argument("--height", type=int, default=25)
    ap.add_argument("--frames", type=int, default=2)
    ap.add_argument("--inputs", default="0,2,258,256", help="tick inputs per frame: turn in bits 0-5, forward 256 / back 512")
    ap.add_argument("--nodes", type=int, default=8)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--max-ms", type=float, default=3600000)
    ap.add_argument("--out", default="data/a2/doom1")
    ap.add_argument("--json", default=None)
    ap.add_argument("--pacing", default="host", choices=["host", "neural"], help="neural: phase gates in the substrate, the host deals tokens in order with no barrier")
    ap.add_argument("--fp32", action="store_true", help="single precision (Apple GPU always; GeForce cards are slow at float64)")
    a = ap.parse_args()
    prog = compile_c(open(a.source).read())
    W, H, B, F = a.width, a.height, a.nodes, a.frames
    per_node_cols = -(-W // B)  # neural pacing needs the same token counts on every copy: W must divide by B
    if a.pacing == "neural" and W % B:
        raise SystemExit(f"neural pacing: the width {W} must be a multiple of the copies {B}")
    ks = compile_program(prog, params={"px": 128, "py": 128, "heading": 0}, pacing=a.pacing,
                         counts={"input": per_node_cols, "p": per_node_cols * H} if a.pacing == "neural" else None)
    col_s, tick_s, pix_s = ks.streams[:3]  # "input" (columns), the frame counter (tick), the pixel loop
    sx, sy = 160 // W, 100 // H
    cols = [x * sx for x in range(W)]
    rows = [y * sy for y in range(H)]
    ticks = [int(v) for v in a.inputs.split(",")][:F]
    at = lambda t: ((t & 255) // sx, (t >> 8) // sy)
    pixel_cell = [o for o in ks.outputs if next(c for c in ks.cells if c["name"] == o)["stream"] == f"input:{pix_s}"][0]
    store_cells = [o for o in ks.outputs if next(c for c in ks.cells if c["name"] == o)["stream"] == "input"]
    n_tick = sum(1 for o in ks.outputs if next(c for c in ks.cells if c["name"] == o)["stream"] == f"input:{tick_s}")
    # reference frames (the whole picture, every column)
    ref_sched = []
    for f in range(F):
        ref_sched += [("input", c) for c in cols] + [(pix_s, (r << 8) | c) for r in rows for c in cols] + [(tick_s, ticks[f])]
    ref_out = kernel_outputs(ks, ref_sched)
    ref_frames, k = [], 0
    for f in range(F):
        k += len(cols)
        ref_frames.append({at((r << 8) | c): ref_out[k + j][0] for j, (r, c) in enumerate((r, c) for r in rows for c in cols)})
        k += len(rows) * len(cols) + 1
        write_png(ref_frames[-1], W, H, f"{a.out}_{f}_reference.png")
    # per-node schedules: the node's columns, then its pixels once its stores are out, then the tick once its pixels are out
    deal = [cols[b::B] for b in range(B)]
    scheds, expect = [], []
    for b in range(B):
        sc, owed = [], 0
        for f in range(F):
            bar = (lambda x: x) if a.pacing == "host" else (lambda x: 0)  # neural pacing: no host barrier
            sc += [("input", c, bar(owed)) for c in deal[b]]
            owed += len(deal[b]) * len(store_cells)
            sc += [(pix_s, (r << 8) | c, bar(owed)) for c in deal[b] for r in rows]
            owed += len(deal[b]) * len(rows)
            sc.append((tick_s, ticks[f], bar(owed)))
            owed += n_tick
        scheds.append(sc)
        expect.append(owed)
    P = Params()
    pl = build_pipeline(P, prog.width, ks.cells, consts=ks.consts, mems=ks.mems, outputs=ks.outputs, streams=ks.streams, phases=ks.phases)
    print(f"kernels: {len(ks.cells)} cells, {pl.net.n} neurons per node, {B} nodes, {W}x{H} pixels x {F} frames, pacing {a.pacing}", flush=True)
    t0 = time.time()
    if B == 1:
        outs1, sim, st = run_pipeline(pl, P, scheds[0], max_ms=a.max_ms, expect_outputs=expect[0])
        outs = [st["outputs_by_cell"]]
        st = {"neural_ms": sim.step_index * P.dt, "faults": st["faults"], "timeouts": st["timeouts"], "bad_outputs": st["bad_outputs"]}
    else:
        import torch
        dtype = torch.float32 if (a.device == "mps" or a.fp32) else None
        outs, sim, st = run_pipeline_batched(pl, P, scheds, max_ms=a.max_ms, device=a.device, expect_outputs=expect, dtype=dtype)
    wall = time.time() - t0
    wrong = missing = 0
    for f in range(F):
        got = {}
        for b in range(B):
            per_frame = len(deal[b]) * len(rows)
            vals = [v for _, v in outs[b][pixel_cell]][f * per_frame:(f + 1) * per_frame]
            for (c, r), v in zip(((c, r) for c in deal[b] for r in rows), vals):
                got[at((r << 8) | c)] = v
        write_png(got, W, H, f"{a.out}_{f}_neural.png")
        wrong += sum(1 for xy, v in ref_frames[f].items() if got.get(xy) != v)
        missing += sum(1 for xy in ref_frames[f] if xy not in got)
    print(f"neural doom: {F} frames of {W}x{H} in {st['neural_ms'] / 1000:.1f} s of neural time ({wall:.0f} s wall on {a.device}); "
          f"wrong {wrong}, missing {missing}; {a.out}_<k>_neural.png", flush=True)
    if a.json:
        json.dump({**{k: v for k, v in st.items() if k in ("neurons", "nodes", "neural_ms", "faults", "timeouts", "bad_outputs")},
                   "wall_s": wall, "wrong": wrong, "missing": missing, "frames": F, "width": W, "height": H, "neurons_per_node": pl.net.n},
                  open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
