"""Mix-B perturbation campaigns for the A2 datapath blocks: ALU channel, staged register,
accumulator. These are composition checks with a recorded trial count, not attempts to
push the silent-error bound (M1 review: no further campaigns for that).

    python -m drosophilos.bench.a2_campaigns alu --n 5000
    python -m drosophilos.bench.a2_campaigns register --n 5000
    python -m drosophilos.bench.a2_campaigns accumulator --n 2000
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ..lib.alu import OPS, alu_operand_word, alu_reference, alu_word, build_alu_channel
from ..lib.campaign import Perturbation, run_campaign
from ..lib.staged import build_accumulator, build_staged_register
from ..protocol.token import rails_for
from ..sim.model import Params

OPLIST = list(OPS)
MIXES = {
    "B": Perturbation(0.04, 0.2, 0.2, 5.0, 150, 100),  # every M1 / A2 campaign so far
    "B+": Perturbation(0.05, 0.25, 0.25, 5.0, 150, 100),  # the failure-hunting probes of step 6
}


def alu_word_fn(width):
    def f(rng):
        return alu_word(int(rng.integers(0, 1 << width)), int(rng.integers(0, 1 << width)), OPLIST[int(rng.integers(0, len(OPLIST)))], width)
    return f


def alu_expected_fn(width):
    from ..lib.alu import decode_alu_word

    def f(w):
        a, b, op = decode_alu_word(w, width)
        return alu_reference(a, b, op, width)["word"]
    return f


def register_program_fn(sc, width, commit_delay_steps):
    """Random words; one COMMIT token per word, `commit_delay_steps` after the load."""
    def f(rng, loads):
        words = [int(rng.integers(0, 1 << width)) for _ in loads]
        events = [(t + commit_delay_steps, sc.reg.commit.u, sc.drive.ignite) for t in loads]
        return words, list(words), events
    return f


def accumulator_program_fn(sc, width, commit_delay_steps):
    """Random program whose first instruction is a MOV (the host never touches the master: a
    direct load would raise W_M and reset the stage while the first instruction is in flight);
    one COMMIT token `commit_delay_steps` after each load. The precomputed expectations are
    only a record; the decoder re-derives each expectation from the master it actually finds
    (see accumulator_chain_fn), so a discarded instruction does not cascade."""
    def f(rng, loads):
        acc, events, words, expected = 0, [], [], []
        for k, t in enumerate(loads):
            op = "MOV" if k == 0 else OPLIST[int(rng.integers(0, len(OPLIST)))]
            b = int(rng.integers(0, 1 << width))
            ref = alu_reference(acc, b, op, width)
            acc = ref["r"]
            words.append(alu_operand_word(b, op, width))
            expected.append(ref["word"])
            events.append((t + commit_delay_steps, sc.reg.commit.u, sc.drive.ignite))
        return words, expected, events
    return f


def accumulator_chain_fn(width):
    from ..lib.alu import decode_alu_word
    mask = (1 << width) - 1

    def f(prev_master, word):
        _, b, op = decode_alu_word(word, width, with_a=False)
        if prev_master is None and op != "MOV":
            return None
        return alu_reference(0 if prev_master is None else prev_master & mask, b, op, width)["word"]
    return f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("block", choices=["alu", "register", "accumulator"])
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--width", type=int, default=4)
    ap.add_argument("--batch", type=int, default=100)
    ap.add_argument("--tx-per-chunk", type=int, default=10)
    ap.add_argument("--period", type=int, default=None)
    ap.add_argument("--commit-delay", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--mix", default="B", choices=["B", "B+"],
                    help="B: the M1/A2 mix (w 4 %%, Vth/bias +-0.2 mV); B+: w 5 %%, +-0.25 mV (harsher)")
    a = ap.parse_args()
    params = Params()
    pert = MIXES[a.mix]
    out = Path(a.out or f"data/a2/{a.block}_{a.width}bit_B.jsonl")
    if a.block == "alu":
        build = lambda: build_alu_channel(params, a.width)
        period = a.period or 10000
        s = run_campaign(build, params, a.n, batch=a.batch, tx_per_chunk=a.tx_per_chunk, tx_period_steps=period, pert=pert,
                         seed=a.seed, out_path=out, word_fn=alu_word_fn(a.width), expected_fn=alu_expected_fn(a.width), debug=a.debug)
    else:
        holder = {}

        def build():
            sc = build_staged_register(params, a.width) if a.block == "register" else build_accumulator(params, a.width)
            holder["sc"] = sc
            return sc
        period = a.period or (7000 if a.block == "register" else 12000)
        # run_campaign calls build_fn first, so the program_fn can close over the built channel
        prog_holder = {}

        def program_fn(rng, loads):
            if "f" not in prog_holder:
                sc = holder["sc"]
                prog_holder["f"] = (register_program_fn if a.block == "register" else accumulator_program_fn)(sc, a.width, a.commit_delay)
            return prog_holder["f"](rng, loads)
        chain = accumulator_chain_fn(a.width) if a.block == "accumulator" else None
        s = run_campaign(build, params, a.n, batch=a.batch, tx_per_chunk=a.tx_per_chunk, tx_period_steps=period, pert=pert,
                         seed=a.seed, out_path=out, program_fn=program_fn, chain_fn=chain, debug=a.debug)
    print(json.dumps({k: v for k, v in s.items() if k != "perturbation"}, indent=1))


if __name__ == "__main__":
    main()
