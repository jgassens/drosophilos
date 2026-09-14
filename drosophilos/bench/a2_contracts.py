"""Regenerate every primitive contract in docs/contracts/ from the current library build,
attaching the latest campaign summary for each block (docs/a2/*_summary.json).

    python -m drosophilos.bench.a2_contracts
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..lib.adder import build_adder_channel, operand_word
from ..lib.alu import OPS, alu_operand_word, alu_reference, alu_word, build_alu_channel
from ..lib.contracts import measure_contract, write_contract
from ..lib.staged import build_accumulator, build_staged_register, run_commits
from ..protocol.handshake import build_channel
from ..sim.model import Params

OUT = Path("docs/contracts")


def _summary(path: str) -> dict | None:
    p = Path(path)
    if not p.exists():
        return None
    s = json.loads(p.read_text())
    return {k: s[k] for k in ("transactions", "counts", "errors", "detected_by", "observed_non_ok_rate", "non_ok_upper_95",
                              "silent_wrong_value_upper_95", "accept_latency_ms", "neurons", "perturbation_text", "tx_period_ms",
                              "wall_minutes") if k in s}


STAGED_TIMING = [
    "READY = 15-hop delay chain (~80 ms) after a reset trigger; must exceed reset settling (~21 ms) plus member recovery",
    "veto relays: a veto rail is live >= 15 ms before its driver rises (by construction, see docs/a2_alu_register.md section 2.4)",
    "relay recovery budget: hold interneuron ~40 ms, veto ~55 ms, source inhibition ~86 ms, against gaps of >= 85, >= 155, >= 170 ms",
    "COMMIT is issued at most once per transaction, any time after DATA; readers sample the master only while W_M holds",
    "loads (upstream DATA) arrive only after READY",
]


def main() -> None:
    params = Params()
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(3)

    # channel (re-measured after the relay-hold fix; the 2026-09-14 freeze is re-issued on the re-run campaign)
    ch = build_channel(params, 4, watchdog_hops=55)
    c = measure_contract("channel_4bit", ch, params, [0b1010, 0b0101, 0b1111, 0b0000, 0b0110, 0b1001],
                         campaign_summary={"final_build_1e5_mix_B": _summary("docs/a2/campaign_channel_final_summary.json"),
                                           "relay_hold_build_1e5_mix_B": _summary("docs/a2/channel_4bit_B_rehold_summary.json")},
                         notes=["4-bit transport channel, watchdog 55 hops; relay-hold build (each gate-fed latch holds its ignition relay)",
                                "status: FROZEN for A2/B (2026-09-14), re-issued the same day on the relay-hold build after a 1e5 mix-B re-run with 0 non-ok (docs/a2_alu_register.md section 3.1)"])
    write_contract(c, OUT / "channel_4bit.yaml")

    fa = build_adder_channel(params, 1)
    cases = [(a, b, ci) for a in (0, 1) for b in (0, 1) for ci in (0, 1)]
    c = measure_contract("full_adder_1bit", fa, params, [operand_word(a, b, ci, 1) for a, b, ci in cases], [a + b + ci for a, b, ci in cases],
                         notes=["dual-rail full adder as a channel; relay-hold build"])
    write_contract(c, OUT / "full_adder_1bit.yaml")

    ad = build_adder_channel(params, 4)
    cases = [(15, 15, 1), (3, 9, 0), (7, 8, 1), (0, 0, 0), (12, 5, 1), (9, 6, 0)]
    c = measure_contract("ripple_adder_4bit", ad, params, [operand_word(a, b, ci, 4) for a, b, ci in cases], [a + b + ci for a, b, ci in cases],
                         campaign_summary={"final_build_1e5_random_additions_mix_B": _summary("docs/a2/campaign_adder4_summary.json")},
                         notes=["4-bit ripple-carry adder as a channel; relay-hold build (the campaign predates that fix)",
                                "campaign classifier of that run filed FAULT-ACCEPTs without completion as no_accept (see a2_liveness.md)"])
    write_contract(c, OUT / "ripple_adder_4bit.yaml")

    alu = build_alu_channel(params, 4)
    cases = [(15, 15, "ADD"), (3, 9, "SUB"), (9, 3, "SUB"), (6, 5, "AND"), (6, 5, "OR"), (10, 5, "XOR"), (0, 11, "MOV"), (8, 8, "SUB")]
    c = measure_contract("alu_4bit", alu, params, [alu_word(a, b, op, 4) for a, b, op in cases],
                         [alu_reference(a, b, op, 4)["word"] for a, b, op in cases],
                         campaign_summary={"veto_build_mix_B": _summary("docs/a2/alu_4bit_B_summary.json")},
                         notes=["4-bit ALU: units ADDER/AND/OR/XOR/PASSB, flags C Z V; veto-relay mux, logic units and first XOR; watchdog 150 hops",
                                "latency depends on the unit: MOV/logic ~300 ms, ADD/SUB ~400-520 ms (carry ripple)"],
                         timing_assumptions=STAGED_TIMING[:3] + [STAGED_TIMING[4]])
    write_contract(c, OUT / "alu_4bit.yaml")

    reg = build_staged_register(params, 4)
    words = [0b1010, 0b0101, 0b1111, 0b0000, 0b0110, 0b1001]
    c = measure_contract("staged_register_4bit", reg, params, words, words,
                         runner=lambda: run_commits(reg, params, words, words, commit_delay_steps=0),
                         campaign_summary={"veto_build_mix_B": _summary("docs/a2/register_4bit_B_summary.json")},
                         notes=["word register with staged commit: stage (channel consumer) + master with completion; COMMIT token; veto-relay copy",
                                "retention: the stage held a word 2 s under the clean model before commit (tests/test_register.py); "
                                "maximum supported hold is not bounded by the model (self-sustaining loops) and is set by the one-input "
                                "exposure of the stage's fault gates (0.55 x 2)",
                                "commit latency (grant -> W_M) ~247 ms; READY ~93 ms after commit-done"],
                         timing_assumptions=STAGED_TIMING)
    write_contract(c, OUT / "staged_register_4bit.yaml")

    acc = build_accumulator(params, 4)
    instrs = [("MOV", 5), ("ADD", 3), ("SUB", 9), ("AND", 6), ("OR", 1), ("XOR", 15), ("ADD", 15)]
    a0, words, exp = 0, [], []
    for op, b in instrs:
        ref = alu_reference(a0, b, op, 4)
        a0 = ref["r"]
        words.append(alu_operand_word(b, op, 4))
        exp.append(ref["word"])
    c = measure_contract("accumulator_4bit", acc, params, words, exp,
                         runner=lambda: run_commits(acc, params, words, exp, commit_delay_steps=0, init_master=0),
                         campaign_summary={"veto_build_mix_B": _summary("docs/a2/accumulator_4bit_B_summary.json")},
                         notes=["ALU -> staged register -> ALU operand A through the operand gate (ACTIVE-driven veto relays)",
                                "the host loads (B, op) and one COMMIT per instruction and reads the master; A is never touched by the host",
                                "cycle per instruction 720-910 ms (clean model): ALU 300-480 ms, commit ~293 ms, READY ~93 ms"],
                         timing_assumptions=STAGED_TIMING)
    write_contract(c, OUT / "accumulator_4bit.yaml")
    print("contracts written:", sorted(p.name for p in OUT.glob("*.yaml")))


if __name__ == "__main__":
    main()
