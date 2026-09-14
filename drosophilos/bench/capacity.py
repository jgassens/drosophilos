"""Whole-program capacity report (plan §Compiler: required before any run above 4 nodes).

For a DrosoC program compiled to the control machine: live state bits, program-image bits,
data words, ALU and control neurons, handshake overhead, and the critical path in neural
time, from the machine's own netlist and the lowered program, not from source size.

    python -m drosophilos.bench.capacity path/to/prog.c
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

from ..compiler.frontend_c import compile_c
from ..compiler.lower import lower
from ..isa.ir import interpret
from ..lib.control import build_machine, reference_run, word_bits
from ..sim.model import Params


def capacity_report(source: str, inputs: list[int] | None = None, ms_per_instruction: float = 1050.0) -> dict:
    prog = compile_c(source)
    words = lower(prog)
    n_prog = 1 << max(0, (len(words) - 1).bit_length())
    n_data = 1 << max(0, (len(prog.variables) - 1).bit_length())
    m = build_machine(Params(), n=prog.width, n_prog=n_prog, n_data=n_data)
    roles = m.net.roles
    classes = Counter()
    for r in roles:
        head = r.split(".")[0]
        if head == "IM":
            classes["program image (latches + fetch relays)"] += 1
        elif head == "DM" or head in ("LD", "ST"):
            classes["data memory (words + ports)"] += 1
        elif head in ("PC", "FSM", "IR", "JT", "SP", "NEXT", "FETCH", "JTkill", "SPkill"):
            classes["control (rings, IR, PC update)"] += 1
        elif head == "alu":
            classes["ALU datapath"] += 1
        elif head in ("P", "Q", "R", "out", "data"):
            classes["registers, stage, master, handshake"] += 1
        else:
            classes["other"] += 1
    ir_ops = sum(1 for f in prog.functions.values() for i in f if not isinstance(i, str))
    ref = reference_run(words, prog.width, m.a, {prog.ports["in"]: (inputs or [0])[0]}, max_steps=5000)
    n_exec = len(ref["trace"])
    return {
        "width_bits": prog.width,
        "ir_instructions": ir_ops,
        "machine_words": len(words),
        "program_words_allocated": n_prog,
        "program_image_bits": len(words) * word_bits(prog.width, m.a),
        "data_words": len(prog.variables),
        "data_words_allocated": n_data,
        "live_state_bits": len(prog.variables) * prog.width + prog.width + 3,  # variables + accumulator + flags
        "neurons_total": m.net.n,
        "synapses_total": m.net.nnz,
        "neurons_by_class": dict(classes),
        "alu_count": 1,
        "instructions_executed_for_inputs": n_exec,
        "critical_path_ms": n_exec * ms_per_instruction,
        "halted": ref["halted"],
        "note": "one accumulator machine, one ALU, no sharing yet; ~1.05 s per instruction on the clean model",
    }


if __name__ == "__main__":
    src = Path(sys.argv[1]).read_text()
    print(json.dumps(capacity_report(src, [int(x) for x in sys.argv[2:]]), indent=1))
