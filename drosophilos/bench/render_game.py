"""The slideshow: `examples/game.c` (a tick loop that turns the view around a frame loop that
renders it pixel by pixel) as two kernels on B copies, the host dealing each frame's pixels to
the copies and then the same tick token to every copy (the game state is replicated in every
brain). Writes frame_<k>_neural.png beside frame_<k>_reference.png for each frame.

    python -m drosophilos.bench.render_game --width 16 --height 10 --frames 3 --turn 10 --nodes 8 --device cpu
"""

import argparse
import json
import time

from ..compiler.frontend_c import compile_c
from ..compiler.kernel import compile_program, kernel_outputs
from ..display.frame import write_png
from ..lib.kernel import build_pipeline, run_pipeline_batched
from ..sim.model import Params


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="examples/game.c")
    ap.add_argument("--width", type=int, default=16)
    ap.add_argument("--height", type=int, default=10)
    ap.add_argument("--frames", type=int, default=3)
    ap.add_argument("--turn", type=int, default=10)
    ap.add_argument("--nodes", type=int, default=8)
    ap.add_argument("--heading", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--max-ms", type=float, default=1800000)
    ap.add_argument("--out", default="data/a2/game")
    ap.add_argument("--json", default=None)
    ap.add_argument("--fp32", action="store_true", help="single precision (Apple GPU always; GeForce cards are slow at float64)")
    a = ap.parse_args()
    prog = compile_c(open(a.source).read())
    ks = compile_program(prog, params={"heading": a.heading})
    tick_stream = ks.streams[1]
    W, H, B, F = a.width, a.height, a.nodes, a.frames
    sx, sy = 160 // W, 100 // H
    toks = [((y * sy) << 8) | (x * sx) for y in range(H) for x in range(W)]
    at = lambda t: ((t & 255) // sx, (t >> 8) // sy)
    pixel_cell = next(o for o in ks.outputs if next(c for c in ks.cells if c["name"] == o)["stream"] == "input")
    n_tick_outs = sum(1 for o in ks.outputs if o != pixel_cell)
    # reference frames
    ref_sched = []
    for f in range(F):
        ref_sched += [("input", t) for t in toks] + [(tick_stream, a.turn)]
    ref_outs = kernel_outputs(ks, ref_sched)
    ref_frames = []
    k = 0
    for f in range(F):
        ref_frames.append({at(t): ref_outs[k + j][0] for j, t in enumerate(toks)})
        k += len(toks) + 1
        write_png(ref_frames[-1], W, H, f"{a.out}_{f}_reference.png")
    # per-node host schedules: the node's share of each frame, then the tick, each waiting for what the node owes
    deal = [toks[b::B] for b in range(B)]
    scheds, expect = [], []
    for b in range(B):
        sc, owed = [], 0
        for f in range(F):
            sc += [("input", t, owed) for t in deal[b]]
            owed += len(deal[b])
            sc.append((tick_stream, a.turn, owed))
            owed += n_tick_outs
        scheds.append(sc)
        expect.append(owed)
    P = Params()
    pl = build_pipeline(P, prog.width, ks.cells, consts=ks.consts, mems=ks.mems, outputs=ks.outputs, streams=ks.streams)
    print(f"kernels: {len(ks.cells)} cells, {pl.net.n} neurons per node, {B} nodes, {len(toks)} pixels x {F} frames", flush=True)
    import torch
    dtype = torch.float32 if (a.device == "mps" or a.fp32) else None
    t0 = time.time()
    outs, sim, st = run_pipeline_batched(pl, P, scheds, max_ms=a.max_ms, device=a.device, expect_outputs=expect, dtype=dtype)
    wall = time.time() - t0
    wrong = missing = 0
    for f in range(F):
        got = {}
        for b in range(B):
            vals = [v for _, v in outs[b][pixel_cell]][f * len(deal[b]):(f + 1) * len(deal[b])]
            for t, v in zip(deal[b], vals):
                got[at(t)] = v
        write_png(got, W, H, f"{a.out}_{f}_neural.png")
        wrong += sum(1 for xy, v in ref_frames[f].items() if got.get(xy) != v)
        missing += sum(1 for xy in ref_frames[f] if xy not in got)
    print(f"neural slideshow: {F} frames of {len(toks)} pixels in {st['neural_ms'] / 1000:.1f} s of neural time ({wall:.0f} s wall on {a.device}); "
          f"wrong {wrong}, missing {missing}; {a.out}_<k>_neural.png", flush=True)
    if a.json:
        json.dump({**st, "wall_s": wall, "wrong": wrong, "missing": missing, "frames": F, "width": W, "height": H}, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
