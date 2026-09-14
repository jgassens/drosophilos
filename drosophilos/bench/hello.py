"""Hello, World on the neural machine: compile examples/hello.c through the DrosoC path,
check the C reference and the IR interpreter, then run it on the neural machine and print
each character as its port-word completion lands.

    python -m drosophilos.bench.hello [--device cuda]
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from ..compiler.frontend_c import compile_c
from ..compiler.golden import run_golden
from ..compiler.lower import lower
from ..isa.ir import interpret
from ..lib.control import build_machine, load_image, reference_run
from ..protocol.token import decode_at
from ..sim.model import Params


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--source", default=str(Path(__file__).resolve().parents[2] / "examples" / "hello.c"))
    a = ap.parse_args()
    params = Params()
    src = Path(a.source).read_text()
    prog = compile_c(src)
    words = lower(prog)
    gold = run_golden(src, [], [], prog.width)
    ir = interpret(prog, [])
    text = lambda outs: "".join(chr(v) for v in outs)
    print(f"C reference : {text(gold['outs'])!r}")
    print(f"IR interp   : {text(ir['outs'])!r}")
    n_prog = 1 << max(0, (len(words) - 1).bit_length())
    n_data = 1 << max(0, (len(prog.variables) - 1).bit_length())
    m = build_machine(params, n=prog.width, n_prog=n_prog, n_data=n_data)
    ref = reference_run(words, m.n, m.a, {}, max_steps=500)
    print(f"machine ref : {text(ref_outs(ref, prog, words))!r}  ({len(words)} words, {len(ref['trace'])} instructions, {m.net.n} neurons)")
    if a.device == "cpu":
        from ..sim.ref64 import RefSim
        sim = RefSim(m.net.topology(), params)
    else:
        from ..sim.lif_torch import TorchSim
        sim = TorchSim(m.net.topology(), params, n_nodes=1, device=a.device)
    load_image(sim, m, words, {})
    out = m.dmem.words[prog.ports["out"]]
    wm = out.completion.u
    window = 2 * m.drive.loop_period_steps
    period = m.drive.loop_period_steps
    last, got, t0 = None, [], time.time()
    max_steps = int((len(ref["trace"]) * 1300 + 3000) / params.dt)
    print("neural      : ", end="", flush=True)
    chunk = 2000
    while sim.step_index < max_steps:
        sim.run(chunk)
        ev = sim.trace.events
        st = ev["step"][(ev["neuron"] == wm) & (ev["step"] >= sim.step_index - chunk - 1)]
        for s_ in st:
            if last is None or s_ - last > 3 * period:
                v = decode_at(sim.trace, out.rail_taps, int(s_), window)[0]
                got.append(v)
                print(chr(v) if v is not None else "?", end="", flush=True)
            last = int(s_)
        if len(got) >= len(gold["outs"]):
            break
    print(f"\nneural machine emitted {text([v for v in got if v is not None])!r} in {sim.step_index * params.dt / 1000:.1f} s of neural time "
          f"({time.time() - t0:.0f} s wall on {a.device}); match: {got == gold['outs']}")


def ref_outs(ref, prog, words):
    """The port writes of the machine reference in execution order: the accumulator's value
    at every STORE to the output port."""
    mask = (1 << prog.width) - 1
    port = prog.ports["out"]
    return [acc & mask for pc, acc in ref["trace"] if words[pc] == ("STORE", port)]


if __name__ == "__main__":
    main()
