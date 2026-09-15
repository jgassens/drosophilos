"""Compile one loop of a DrosoC program into a resident kernel and run it neurally: the
references (IR interpreter, kernel reference) and the pipeline's outputs and throughput.

    python -m drosophilos.bench.run_kernel examples/render.c --loop loop3 --stream col --param heading=3 --tokens 0-7
"""

import argparse
import json
import time

from ..compiler.frontend_c import compile_c
from ..compiler.kernel import compile_kernel, kernel_outputs, loop_body
from ..isa.ir import interpret
from ..lib.kernel import build_pipeline, run_pipeline
from ..sim.model import Params


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("source")
    ap.add_argument("--loop", required=True, help="the loop's label in main (e.g. loop3)")
    ap.add_argument("--stream", required=True, help="the induction variable the host streams")
    ap.add_argument("--param", action="append", default=[], help="name=value kernel parameters")
    ap.add_argument("--tokens", default="0-7")
    ap.add_argument("--max-ms", type=float, default=60000)
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    params = {k: int(v, 0) for k, v in (kv.split("=") for kv in a.param)}
    lo, hi = a.tokens.split("-")
    tokens = list(range(int(lo), int(hi) + 1))
    prog = compile_c(open(a.source).read())
    ks = compile_kernel(prog, loop_body(prog, "main", a.loop), a.stream, params=params)
    ref = kernel_outputs(ks, tokens)  # per token, one value per output cell
    print("cells       :", [(c["name"], c["op"]) for c in ks.cells], "outputs", ks.outputs, "state", ks.state_cells)
    print("kernel ref  :", ref)
    print("IR interp   :", interpret(prog, list(params.values()))["outs"][: len(tokens)], "(first outputs of the whole program)")
    P = Params()
    t0 = time.time()
    pl = build_pipeline(P, prog.width, ks.cells, consts=ks.consts, mems=ks.mems, outputs=ks.outputs)
    print(f"pipeline    : {pl.net.n} neurons, built in {time.time() - t0:.1f} s", flush=True)
    t0 = time.time()
    outs, sim, st = run_pipeline(pl, P, tokens, max_ms=a.max_ms)
    by_cell = st.pop("outputs_by_cell")
    got = [list(t) for t in zip(*[[v for _, v in by_cell[o]] for o in ks.outputs])]
    st.update(wall_s=round(time.time() - t0, 1), match=(got == ref), outputs_values=got, reference=ref, source=a.source, loop=a.loop)
    print("neural      :", got)
    print("match       :", st["match"], "| first output", st["first_output_ms"], "ms | per token", st["per_token_ms"], "ms | wall", st["wall_s"], "s")
    if a.json:
        json.dump(st, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
