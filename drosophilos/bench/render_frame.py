"""Render a frame of `examples/frame.c` in the substrate: one token per pixel, dealt to B
copies of the pixel kernel on the batched simulator (B brains), the pixel records assembled
into a PNG by the host beside the reference image.

    python -m drosophilos.bench.render_frame --width 32 --height 20 --nodes 8 --heading 5 --device cpu
"""

import argparse
import json
import time

from ..compiler.frontend_c import compile_c
from ..compiler.kernel import compile_kernel, kernel_outputs, loop_body
from ..display.frame import write_png
from ..lib.kernel import build_pipeline, run_pipeline_batched
from ..sim.model import Params


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="examples/frame.c")
    ap.add_argument("--width", type=int, default=32)
    ap.add_argument("--height", type=int, default=20)
    ap.add_argument("--nodes", type=int, default=8)
    ap.add_argument("--heading", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--max-ms", type=float, default=600000)
    ap.add_argument("--out", default="data/a2/frame")
    ap.add_argument("--json", default=None)
    ap.add_argument("--fp32", action="store_true", help="single precision (Apple GPU always; GeForce cards are slow at float64)")
    a = ap.parse_args()
    prog = compile_c(open(a.source).read())
    ks = compile_kernel(prog, loop_body(prog), "i", params={"heading": a.heading})
    W, H = a.width, a.height
    sx, sy = 160 // W, 100 // H  # a W x H picture samples the 160 x 100 screen
    toks = [((y * sy) << 8) | (x * sx) for y in range(H) for x in range(W)]
    at = lambda t: ((t & 255) // sx, (t >> 8) // sy)
    ref = [v[0] for v in kernel_outputs(ks, toks)]
    write_png({at(t): v for t, v in zip(toks, ref)}, W, H, f"{a.out}_reference.png")
    P = Params()
    t0 = time.time()
    pl = build_pipeline(P, prog.width, ks.cells, consts=ks.consts, mems=ks.mems, outputs=ks.outputs)
    print(f"kernel: {len(ks.cells)} cells, {pl.net.n} neurons per node, {a.nodes} nodes, {len(toks)} pixels, built in {time.time() - t0:.1f} s", flush=True)
    deal = [toks[b::a.nodes] for b in range(a.nodes)]  # round-robin: node b renders pixels b, b+B, ...
    t0 = time.time()
    import torch
    dtype = torch.float32 if (a.device == "mps" or a.fp32) else None  # Apple's GPU has no float64
    outs, sim, st = run_pipeline_batched(pl, P, deal, max_ms=a.max_ms, device=a.device, dtype=dtype)
    wall = time.time() - t0
    out_cell = ks.outputs[0]
    got = {}
    for b in range(a.nodes):
        for t, (_, v) in zip(deal[b], outs[b][out_cell]):
            got[at(t)] = v
    write_png(got, W, H, f"{a.out}_neural.png")
    wrong = sum(1 for t, r in zip(toks, ref) if got.get(at(t)) != r)
    missing = sum(1 for t in toks if at(t) not in got)
    print(f"neural frame: {len(got)} of {len(toks)} pixels in {st['neural_ms'] / 1000:.1f} s of neural time "
          f"({wall:.0f} s wall on {a.device}); wrong {wrong}, missing {missing}; {a.out}_neural.png vs {a.out}_reference.png", flush=True)
    if a.json:
        json.dump({**st, "wall_s": wall, "wrong": wrong, "missing": missing, "width": W, "height": H, "heading": a.heading}, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
