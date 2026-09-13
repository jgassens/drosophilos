"""Render docs/h0_report.md from data/h0/results.json (Stage H0 exit deliverable)."""

from __future__ import annotations

import json
import sys
from pathlib import Path


def fmt_ms(x):
    return "—" if x is None else f"{x:.1f} ms"


def truth_rows(rows):
    out = ["| a | b | y1 fired | y0 fired | completion | correct | completion latency | reset | ready latency | surround spikes | recruited neurons |",
           "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        out.append(f"| {r['a']} | {r['b']} | {r['fired_y1']} | {r['fired_y0']} | {r['completion']} | "
                   f"{'✓' if r['correct'] else '✗'} | {fmt_ms(r['completion_latency_ms'])} | "
                   f"{'✓' if r['reset_ok'] else '✗'} | {fmt_ms(r['ready_latency_ms'])} | "
                   f"{r['surround_spikes']} | {r['surround_neurons_recruited']} |")
    return "\n".join(out)


def main(results_path: Path, out_path: Path) -> None:
    R = json.loads(results_path.read_text())
    if R.get("status") == "no embedding found":
        out_path.write_text("# Stage H0 report\n\nNo embedding found.\n\n```\n" + json.dumps(R["search"], indent=1) + "\n```\n")
        return
    S, T, P, E = R["search"], R["targets"], R["policy"], R["embedding"]
    L = []
    L.append("# Stage H0 report — a minimal circuit in real MCNS wiring (Profile 2)\n")
    L.append("Conclusions and consequences are written up in `docs/h0_findings.md`.\n")
    L.append("Generated from `data/h0/results.json`. Labels: **profile 2**, isolated **and** full-graph, "
             "hybrid orchestration (host injects DATA and reads spikes), external compilation (hand-designed circuit).\n")
    L.append("## Circuit\n")
    L.append("Dual-rail 1-bit AND (the carry of a half adder) with register, completion detector, and reset return path. "
             "Values are held by two-neuron excitatory loops (a circulating spike); the AND is a threshold on the "
             "sustained drive of two loops; OR, completion, and reset work in single-pulse mode.\n")
    L.append("| role | bodyId | type | superclass | transmitter | side |\n|---|---|---|---|---|---|")
    for c in E["circuit"]:
        L.append(f"| {c['role']} | {c['bodyId']} | {c['type']} | {c['superclass']} | {c['nt']} | {c['side']} |")
    L.append("\n### Designed edges (all anatomical; weight = quanta, bound 0 ≤ q ≤ k_max·count·16)\n")
    L.append("| role | pre → post | anatomical synapses | quanta | scale vs anatomical |\n|---|---|---|---|---|")
    for e in E["designed_edges"]:
        L.append(f"| {e['role']} | {e['pre']} → {e['post']} | {e['count']} | {e['quanta']} | {e['scale']}× |")
    L.append(f"\nParasitic anatomical edges among circuit neurons (zeroed, documented): {len(E['parasitic_edges'])}; "
             f"minimum weight margin (count·16 / |q|, ≥ 1/k_max required): {E['min_margin']:.2f}.\n")
    L.append("## Motif availability\n")
    rc = S["required_counts"]
    L.append(f"- Physics targets (quanta): loop {T['loop']}, AND input {T['and_in']} (rate mode), OR input {T['or_in']}, "
             f"completion / reset drive {T['completion']} (single pulse), reset {T['reset']} per hit member; "
             f"loop period {T['period_steps']} steps.")
    L.append(f"- Required anatomical synapse counts at k_max = {P['k_max']}: loop ≥ {rc['loop']}, AND ≥ {rc['and_in']}, "
             f"OR ≥ {rc['or_in']}, completion ≥ {rc['completion']}, reset ≥ {rc['reset_quanta_per_member']} quanta per member.")
    L.append(f"- Candidate latches (mutual cholinergic pairs, both directions ≥ {rc['loop']}): **{S['n_loops']}**.")
    L.append(f"- Completion candidates (cholinergic, ≥ 2 cholinergic inputs ≥ {rc['completion']}): **{S['n_E_candidates']}**; "
             f"of these, with four resettable latches: **{S['n_E_with_four_resettable_loops']}**; with usable AND+OR gate pairs: "
             f"**{S['n_E_with_gate_pairs']}**.")
    L.append(f"- Complete embeddings found: **{S['n_solutions']}** in {S['search_seconds']} s "
             f"({'search truncated' if S['truncated'] else 'exhaustive'}).")
    L.append(f"- Relay overhead: reset uses {sum(1 for e in E['designed_edges'] if e['role'].startswith('reset_') and not e['role'].startswith('reset_drive'))} "
             f"inhibitory edges from {len({e['post'] for e in E['designed_edges'] if e['role'].startswith('reset_drive')})} inhibitory neurons; "
             f"no relay neurons were needed beyond the designed roles.\n")
    try:
        from ..connectome.embed_h0 import isolation_cost
        from ..connectome.mcns import load_mcns

        m = load_mcns()
        ic = isolation_cost(m, [c["neuron_idx"] for c in E["circuit"]])
        L.append("## Isolation cost (what Profile 2 silencing must zero to stop the circuit broadcasting)\n")
        L.append(f"- Anatomical edges from the {len(E['circuit'])} circuit neurons to the surround: **{ic['out_edges']}** "
                 f"({ic['out_synapses']} synapses; {ic['out_edges_ge_24']} edges of ≥ 24 synapses, each strong enough "
                 f"to fire its target on its own under a 213 Hz latch train).")
        L.append(f"- Anatomical edges from the surround into circuit neurons: **{ic['in_edges']}** ({ic['in_synapses']} synapses; "
                 f"{ic['in_edges_ge_24']} of ≥ 24).\n")
    except Exception as exc:  # data not present
        L.append(f"## Isolation cost\n\n(not computed: {exc})\n")
    L.append("## Results by condition\n")
    for cond, C in R["conditions"].items():
        L.append(f"### {cond}\n")
        L.append(truth_rows(C["truth_table"]))
        if "offset_sweep_11" in C:
            L.append("\nTiming tolerance (a=1, b=1 with b delayed):\n")
            L.append("| b offset | y1 fired | correct | completion latency | reset |\n|---|---|---|---|---|")
            for s in C["offset_sweep_11"]:
                L.append(f"| {s['offset_ms']} ms | {s['fired_y1']} | {'✓' if s['correct'] else '✗'} | "
                         f"{fmt_ms(s['completion_latency_ms'])} | {'✓' if s['reset_ok'] else '✗'} |")
        if "envelope" in C:
            L.append("\nBoundary-input envelope (quanta into circuit neurons from non-circuit spikes):\n")
            L.append("| a | b | max +quanta / step | max −quanta / step | total |quanta| | steps with input |\n|---|---|---|---|---|---|")
            for r, env in zip(C["truth_table"], C["envelope"]):
                L.append(f"| {r['a']} | {r['b']} | {env['max_pos_quanta_per_step']} | {env['max_neg_quanta_per_step']} | "
                         f"{env['total_abs_quanta']} | {env['steps_with_any_input']} |")
            for r in C["truth_table"]:
                if r.get("top_recruited"):
                    L.append(f"\nMost recruited surround neurons for a={r['a']}, b={r['b']}: " +
                             ", ".join(f"{t['type']} ({t['superclass']}, {t['spikes']} spikes)" for t in r["top_recruited"]))
        if "n_output_edges_zeroed" in C and C["n_output_edges_zeroed"]:
            L.append(f"\nOutgoing anatomical edges from circuit neurons to the surround zeroed (documented silencing): {C['n_output_edges_zeroed']}.")
        L.append("")
    out_path.write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    main(Path(sys.argv[1] if len(sys.argv) > 1 else "data/h0/results.json"),
         Path(sys.argv[2] if len(sys.argv) > 2 else "docs/h0_report.md"))
