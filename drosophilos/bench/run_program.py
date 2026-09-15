"""Run a DrosoC program on the neural machine and print its output records as they land.

    python -m drosophilos.bench.run_program examples/tick.c --inputs 5 [--device cuda] [--chars]

Prints the C reference's outputs, the IR interpreter's, the machine reference's, then the
neural machine's decoded output-port completions in order, and the match verdict.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from ..compiler.frontend_c import compile_c
from ..compiler.golden import run_golden
from ..compiler.lower import lower
from ..isa.ir import interpret
from ..lib.control import build_machine, load_image, reference_run
from ..protocol.token import decode_recent, recent_spikes
from ..sim.model import Params


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("source")
    ap.add_argument("--inputs", type=int, nargs="*", default=[])
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--chars", action="store_true", help="print outputs as characters")
    ap.add_argument("--ms-per-instruction", type=float, default=1300)
    a = ap.parse_args()
    params = Params()
    src = Path(a.source).read_text()
    prog = compile_c(src)
    words = lower(prog)
    names = [v for v in prog.variables if not v.startswith("__")]
    gold = run_golden(src, a.inputs, names, prog.width)
    ir = interpret(prog, a.inputs)
    fmt = (lambda outs: "".join(chr(v) for v in outs)) if a.chars else (lambda outs: outs)
    n_prog = 1 << max(0, (len(words) - 1).bit_length())
    n_data = 1 << max(0, (len(prog.variables) - 1).bit_length())
    x_word = prog.ports.get("x")
    uses_mul = any(op.startswith("MUL") for op, _ in words)
    m = build_machine(params, n=prog.width, n_prog=n_prog, n_data=n_data, x_word=x_word, mul=uses_mul,
                      watchdog_hops=300 if uses_mul else 170)
    dmem = {prog.ports["in"]: a.inputs[0]} if a.inputs else {}
    for v, addr in prog.variables.items():  # array initialisers live in the image
        pass
    ref = reference_run(words, m.n, m.a, dmem, max_steps=20000, x_word=x_word)
    mouts = [acc & ((1 << prog.width) - 1) for pc, acc in ref["trace"] if words[pc] == ("STORE", prog.ports["out"])]
    print(f"program     : {len(words)} words, {len(prog.variables)} data words; machine {m.net.n} neurons")
    print(f"C reference : {fmt(gold['outs'])!r}")
    print(f"IR interp   : {fmt(ir['outs'])!r}")
    print(f"machine ref : {fmt(mouts)!r}  ({len(ref['trace'])} instructions, halted={ref['halted']}, fault={ref['fault']})")
    if a.device == "cpu":
        from ..sim.ref64 import RefSim
        sim = RefSim(m.net.topology(), params)
    else:
        from ..sim.lif_torch import TorchSim
        sim = TorchSim(m.net.topology(), params, n_nodes=1, device=a.device)
    load_image(sim, m, words, dmem)
    out = m.dmem.words[prog.ports["out"]]
    wm = out.completion.u
    window, period = 2 * m.drive.loop_period_steps, m.drive.loop_period_steps
    last, got, t0 = None, [], time.time()
    max_steps = int((len(ref["trace"]) * a.ms_per_instruction + 3000) / params.dt)
    print("neural      : ", end="", flush=True)
    chunk = 2000
    while sim.step_index < max_steps and len(got) < len(gold["outs"]):
        sim.run(chunk)
        if len(sim._spk_step) > 4 * chunk:  # keep the recent spikes only: the whole trace of a long run does not fit (a 517-instruction run was OOM-killed at 16 GB)
            del sim._spk_step[: -2 * chunk]; del sim._spk_neuron[: -2 * chunk]
            if hasattr(sim, "_spk_node"): del sim._spk_node[: -2 * chunk]
        for s_ in recent_spikes(sim, wm, sim.step_index - chunk - 1):  # never poll sim.trace: it sorts everything
            if last is None or s_ - last > 3 * period:
                v = decode_recent(sim, out.rail_taps, int(s_), window)[0]
                got.append(v)
                print((chr(v) if a.chars else f"{v} ") if v is not None else "? ", end="", flush=True)
            last = int(s_)
    print(f"\nneural machine emitted {fmt([v for v in got if v is not None])!r} in {sim.step_index * params.dt / 1000:.1f} s of neural "
          f"time ({time.time() - t0:.0f} s wall on {a.device}); match: {got == gold['outs']}")


if __name__ == "__main__":
    main()
