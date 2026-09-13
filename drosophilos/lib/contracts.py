"""Measure a primitive's contract (plan §Circuit library) on a built channel and write it
as YAML. Fields that the model cannot exercise are stated as such rather than invented:
fan-out has no effect on a presynaptic neuron in this model (no load), so "fan-out
tolerated" is unbounded-by-model; per-synapse delay jitter is not supported by the shared
topology, so it is listed as not measured.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import yaml

from ..protocol.run import run_transactions
from ..sim.model import Params


def measure_contract(name: str, ch, params: Params, words: list[int], expected: list[int] | None = None,
                     campaign_summary: dict | None = None, sweep: dict | None = None, notes: list[str] = ()) -> dict:
    recs, sim, st = run_transactions(ch, params, words, expected=expected, max_steps_per_tx=20000)
    net, drive = ch.net, ch.drive
    roles = net.roles
    ev = sim.trace.events
    n_tx = len(recs)
    assert st["completed"] == n_tx and st["correct"] == n_tx, st
    # role classes
    def cls(role: str) -> str:
        if ".L." in role or (role.split(".")[1].startswith("b") and role.endswith((".u", ".v"))):
            return "latch"
        if role.endswith((".and", ".or", ".maj")):
            return "gate"
        return "control"  # relays, reset, ready chains, edge inhibitors
    counts = {"latch": 0, "gate": 0, "control": 0}
    for n in ev["neuron"]:
        counts[cls(roles[n])] += 1
    latch_members = [x for x, r in enumerate(roles) if cls(r) == "latch"]
    # duty cycle: mean rate of a latch member while it holds (spikes per hold / hold time)
    hold_ms = float(np.mean([(r.accept_step - r.load_step) * params.dt for r in recs]))
    active_latch_spikes = counts["latch"] / n_tx
    # output port spikes per transaction: consumer rail taps + completion + READY + CLEARED
    out_ports = set(t for pair in ch.consumer.rail_taps for t in pair) | {ch.consumer.completion.u, ch.consumer.ready, ch.producer.ready}
    out_spikes = int(np.isin(ev["neuron"], list(out_ports)).sum()) / n_tx
    # reset energy: inhibitory quanta delivered per reset of each register
    reset_energy = {}
    for reg_name, reg in (("producer", ch.producer), ("consumer", ch.consumer)):
        inh = reg.reset_inh
        q = sum(abs(net.quanta[k]) for k in range(net.nnz) if net.src[k] == inh)
        prefix = roles[reg.reset_trigger][: -len(".reset")]
        pulses = 1 + sum(1 for r in roles if r.startswith(f"{prefix}.reset_relay"))
        reset_energy[reg_name] = {"pulses": pulses, "quanta_per_pulse": int(q), "quanta_per_reset": int(q * pulses),
                                  "mv_equivalent_per_reset": round(q * pulses * params.w_unit, 1)}
    # quiescent activity: spikes between READY and the next load
    quiet = 0
    for k, r in enumerate(recs[:-1]):
        nxt = recs[k + 1].load_step
        quiet += int(((ev["step"] > r.ready_step) & (ev["step"] < nxt)).sum())
    contract = {
        "primitive": name,
        "profile": 3,
        "model": {"dt_ms": params.dt, "tau_m_ms": params.tau_m, "tau_s_ms": params.tau_s, "delay_ms": params.default_delay_ms,
                  "w_syn_mv": params.w_syn},
        "drive_quanta": asdict(drive),
        "resources": {"neurons": net.n, "synapses": net.nnz, **net.summary()},
        "latency": {"accept_ms_mean": round(float(np.mean(st["accept_latency_ms"])), 1),
                    "accept_ms_max": round(float(np.max(st["accept_latency_ms"])), 1)},
        "initiation_interval": {"cycle_ms_mean": round(float(np.mean(st["cycle_ms"])), 1),
                                "cycle_ms_max": round(float(np.max(st["cycle_ms"])), 1)},
        "activity": {"spikes_per_transaction_total": round(st["spikes_per_tx"], 1),
                     "spikes_per_transaction_by_class": {k: round(v / n_tx, 1) for k, v in counts.items()},
                     "output_port_spikes_per_transaction": round(out_spikes, 1),
                     "latch_member_rate_hz_while_holding": round(1000.0 / (drive.loop_period_steps * params.dt), 1),
                     "sustained_duty_cycle": "a held latch fires continuously at the rate above until reset",
                     "quiescent_spikes_between_ready_and_next_load": quiet},
        "reset_energy": reset_energy,
        "fan_out_tolerated": "unbounded by the model: a presynaptic spike costs the source nothing (no load); measured on the real connectome as isolation cost instead (H0)",
        "delay_jitter_tolerance": "not measured: per-synapse delays are shared topology in this backend",
        "stray_input_tolerance": sweep,
        "campaign": campaign_summary,
        "timing_assumptions": [
            "READY/CLEARED = 11-hop delay chain (~58 ms) after the reset trigger; must exceed reset settling (~21 ms, 4 pulses) plus member recovery from ~-45 mV total inhibition",
            "edge relays: relay fires ~2.5 ms after the source's first spike, its inhibitor's pulse lands ~5.3 ms after; ordering margin ~2.8 ms nominal",
            "loads (upstream DATA) arrive only after READY",
        ],
        "notes": list(notes),
    }
    return contract


def write_contract(contract: dict, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(json.loads(json.dumps(contract, default=float)), sort_keys=False, allow_unicode=True))
