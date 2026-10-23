"""Bounded integration-rule comparison; every simulated replica has <=15 neurons.

Run with ``python -m drosophilos.bench.model_compare --mcns-dir PATH``.
No full-graph topology or full-graph simulator is constructed.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ..connectome import mcns
from ..connectome.embed_h0 import Embedding, Policy, Targets, best_embedding, find_embeddings
from ..connectome.h0_run import Transaction, isolated_topology
from ..connectome.mcns import MCNS
from ..sim import ModelRules, Params, RefSim, Topology

CONFIGS = {
    "M0_current": ModelRules(),
    "M1_clear_g": ModelRules(clear_g_on_spike=True),
    "M2_freeze_g": ModelRules(freeze_g_in_refractory=True),
    "M3_stim_v": ModelRules(stim_target="v"),
    "M4_stim_norefrac": ModelRules(stim_disables_refractory=True),
    "M5_shiu": ModelRules(True, True, "v", True),
}
PAIR_QUANTA = 2300
FORCE_QUANTA = 100_000
SHIU_QUANTA = 4000
RATES = (2, 10, 50)


def load_mcns(*, data_dir: Path) -> MCNS:
    """Read existing caches or derive in memory, never cache into the source."""
    cache_path = data_dir / "mcns_v1.0_edges_minsyn1.npz"
    neurons_path = data_dir / "mcns_v1.0_neurons.parquet"
    try:
        # Read directly: cache=True could write if a file disappeared after a check.
        with np.load(cache_path) as edges:
            neurons = pd.read_parquet(neurons_path)
            return MCNS(neurons, edges["pre"], edges["post"], edges["count"], min_syn=1)
    except FileNotFoundError:
        return mcns.load_mcns(data_dir=data_dir, cache=False)


@dataclass
class Fixture:
    name: str
    topo: Topology
    target: int = 0
    embedding: Embedding | None = None
    index: dict[int, int] | None = None
    provenance: dict | None = None

    def __post_init__(self) -> None:
        if not 1 <= self.topo.n <= 15:
            raise ValueError("comparison fixtures must contain 1..15 neurons")


def small_fixtures(params: Params = Params()) -> list[Fixture]:
    pair = Topology.from_edges(2, [0, 1], [1, 0], [PAIR_QUANTA] * 2,
                               [params.default_delay_steps] * 2)
    return [Fixture("F1_single", Topology.empty(1)), Fixture("F2_pair", pair)]


def h0_fixture(m: MCNS, params: Params = Params(), *, source: str = "provided") -> Fixture:
    policy = Policy()
    solutions, stats = find_embeddings(m, params, policy, max_solutions=50,
                                      time_limit_s=300, verbose=False)
    if not solutions:
        raise ValueError(f"no H0 embedding: {stats}")
    emb = best_embedding(solutions)
    topo, idx = isolated_topology(m, emb, params, policy)
    provenance = {
        "source": source, "search": stats, "policy": asdict(policy),
        "body_ids": m.bodies_of(emb.neurons).tolist(),
        "roles": {str(int(m.bodies_of([n])[0])): role for n, role in emb.roles().items()},
        "edges": [asdict(edge) for edge in emb.edges],
    }
    return Fixture("F3_h0", topo, idx[emb.loops["a1"][0]], emb, idx, provenance)


def periodic_steps(rate_hz: float, duration_ms: float, phase: float, params: Params) -> np.ndarray:
    """phase in [0, 1): evenly spaced replicas integrate over periodic event phase."""
    if rate_hz == 0:
        return np.empty(0, np.int64)
    period_ms = 1000 / rate_hz
    times = np.arange(phase * period_ms, duration_ms, period_ms)
    # Floor keeps the entire left-inclusive drive interval on the simulation grid.
    return np.floor(times / params.dt + 1e-9).astype(np.int64)


def run_protocol(fixture: Fixture, rules: ModelRules, *, quanta: int = FORCE_QUANTA,
                 rate_hz: float = 2, drive_ms: float = 300, off_ms: float = 300,
                 replicas: int = 50, single: bool = False, probe: bool = False,
                 params: Params = Params()) -> tuple[dict, RefSim]:
    """Fresh replicas at rest; rates and counts are ensemble means, never pooled Hz."""
    if fixture.topo.n > 15:
        raise ValueError("comparison is limited to 15 neurons")
    sim = RefSim(fixture.topo, params, n_nodes=replicas, rules=rules)
    stop = int(round(drive_ms / params.dt))
    total = int(round((drive_ms + off_ms) / params.dt))
    probe_step = stop + int(round(50 / params.dt))
    event_count = 0
    for b in range(replicas):
        steps = (np.array([int(round(5 / params.dt))]) if single else
                 periodic_steps(rate_hz, drive_ms, (b + .5) / replicas, params))
        event_count += len(steps)
        if probe:
            steps = np.append(steps, probe_step)
        sim.add_events(b, steps, np.full(len(steps), fixture.target),
                       np.full(len(steps), quanta))
    sim.run(stop)
    residual = sim.g.copy()
    sim.run(total - stop)
    ev = sim.trace.events
    during = ev["step"] < stop
    after = ev["step"] >= stop
    target = ev["neuron"] == fixture.target
    counts = np.bincount(ev["neuron"], minlength=fixture.topo.n) / replicas
    drive_counts = np.bincount(ev["neuron"][during], minlength=fixture.topo.n) / replicas
    off_steps = ev["step"][after]
    per_replica_last = []
    for b in range(replicas):
        ss = ev["step"][after & (ev["node"] == b)]
        per_replica_last.append(float((ss[-1] - stop) * params.dt) if len(ss) else None)
    result = {
        "quanta": int(quanta), "amplitude_mv": quanta * params.w_unit,
        "nominal_event_rate_hz": None if single else rate_hz,
        "observed_event_rate_hz": event_count / replicas / (drive_ms / 1000),
        "events_per_replica": event_count / replicas,
        "replicas": replicas, "drive_ms": drive_ms, "off_ms": off_ms,
        "spike_count_mean": len(ev) / replicas,
        "mean_firing_rate_hz": len(ev) / replicas / fixture.topo.n / ((drive_ms + off_ms) / 1000),
        "target_drive_rate_hz": int(np.sum(during & target)) / replicas / (drive_ms / 1000),
        "spikes_per_neuron_mean": counts.tolist(),
        "drive_rates_per_neuron_hz": (drive_counts / (drive_ms / 1000)).tolist(),
        "residual_g_mv_per_neuron_mean": residual.mean(axis=0).tolist(),
        "residual_g_mv_abs_max": float(np.abs(residual).max()),
        "off_spike_count_mean": int(after.sum()) / replicas,
        "last_spike_after_stop_ms": float((off_steps.max() - stop) * params.dt) if len(off_steps) else None,
        "last_spike_after_stop_ms_per_replica": per_replica_last,
        "tail_spikes_final_50ms_mean": int(np.sum(ev["step"] >= total - round(50 / params.dt))) / replicas,
    }
    return result, sim


def probe_metrics(control: RefSim, probed: RefSim, target: int, *, drive_ms: float = 300) -> dict:
    """A causal proxy: extra target spike within 30 ms versus the paired no-probe run."""
    probe_ms = drive_ms + 50
    start = int(round(probe_ms / probed.params.dt))
    end = start + int(round(30 / probed.params.dt))
    extra_latencies, raw_latencies = [], []
    for b in range(probed.B):
        def steps(sim):
            ev = sim.trace.events
            return ev["step"][(ev["node"] == b) & (ev["neuron"] == target)
                              & (ev["step"] >= start) & (ev["step"] < end)]
        baseline, observed = steps(control), steps(probed)
        extra = np.setdiff1d(observed, baseline)
        extra_latencies.append(float((extra[0] - start) * probed.params.dt) if len(extra) else None)
        raw_latencies.append(float((observed[0] - start) * probed.params.dt) if len(observed) else None)
    detected = [x for x in extra_latencies if x is not None]
    return {
        "probe_ms": probe_ms, "window_ms": 30,
        "extra_spike_fraction": len(detected) / probed.B,
        "extra_spike_latency_ms_mean": float(np.mean(detected)) if detected else None,
        "extra_spike_latency_ms_per_replica": extra_latencies,
        "raw_first_spike_latency_ms_per_replica": raw_latencies,
    }


def matched_measurement(fixture: Fixture, rules: ModelRules, quanta: int, rate_hz: int,
                        replicas: int, params: Params) -> dict:
    """Drive/off metrics plus paired responsiveness at the calibrated amplitude."""
    metrics, control = run_protocol(fixture, rules, quanta=quanta, rate_hz=rate_hz,
                                    drive_ms=1000, off_ms=300, replicas=replicas, params=params)
    probe, probed = run_protocol(fixture, rules, quanta=quanta, rate_hz=rate_hz,
                                 drive_ms=1000, off_ms=300, replicas=replicas, probe=True, params=params)
    probe["probe"] = probe_metrics(control, probed, fixture.target, drive_ms=1000)
    metrics["probe_trial"] = probe
    return metrics


def calibrate(fixture: Fixture, rules: ModelRules, target_hz: int, *, replicas: int = 10,
              params: Params = Params()) -> dict:
    """Integer bisection on amplitude, fresh 1-s trials at the target event rate.

    Count plateaus and recurrence can make matching impossible. Keep the closest
    measured candidate, record the error, and never label a mismatched run matched.
    """
    lo, hi = 0, FORCE_QUANTA
    best = None
    trials = []
    while lo <= hi:
        q = (lo + hi) // 2
        metrics, _ = run_protocol(fixture, rules, quanta=q, rate_hz=target_hz,
                                  drive_ms=1000, off_ms=0, replicas=replicas, params=params)
        rate = metrics["target_drive_rate_hz"]
        error = abs(rate - target_hz)
        trials.append({"quanta": q, "rate_hz": rate})
        if best is None or (error, q) < (best[0], best[1]):
            best = (error, q, rate)
        if rate >= target_hz:
            hi = q - 1
        else:
            lo = q + 1
    error, q, rate = best
    metrics = matched_measurement(fixture, rules, q, target_hz, replicas, params)
    return {
        "target_hz": target_hz, "calibrated_quanta": q,
        "calibrated_amplitude_mv": q * params.w_unit,
        "achieved_hz": rate, "absolute_error_hz": error,
        "matched_within_5_percent": bool(error <= .05 * target_hz),
        "status": "matched" if error <= .05 * target_hz else "no_match_found",
        "search": "integer bisection, 0..100000 quanta; no monotonicity guarantee in recurrent circuits",
        "trials": trials, "measurement": metrics,
    }


def h0_transactions(fixture: Fixture, rules: ModelRules, params: Params = Params()) -> list[dict]:
    """Same stimulus, decoder and reset/latency checks as h0_run.run_transaction.

    Kept local because h0_run is outside this task's edit scope and has no rules
    argument. Tests compare this decoder with the existing function under M0.
    No weight, event amplitude or completion criteria are retuned for another model.
    """
    emb, idx = fixture.embedding, fixture.index
    targets = Targets.from_policy(params, Policy())
    rows = []
    for a, b in itertools.product((0, 1), repeat=2):
        tx = Transaction(a, b)
        sim = RefSim(fixture.topo, params, rules=rules)
        for rail, on in {"a1": a == 1, "a0": a == 0, "b1": b == 1, "b0": b == 0}.items():
            if on:
                sim.add_events(0, [tx.t_data], [idx[emb.loops[rail][0]]], [targets.loop])
        sim.run(tx.n_steps)
        roles = {role: sim.trace.neuron_steps(idx[x]).tolist() for x, role in emb.roles().items()}
        y1, y0, comp = roles["and_y1"], roles["or_y0"], roles["completion"]
        loops = [s for role, ss in roles.items() if role.startswith("loop_") for s in ss]
        last = max(loops) if loops else None
        correct = bool(y1) == bool(a and b) and bool(y0) == (not bool(a and b)) and bool(comp)
        reset = last is not None and bool(comp) and last < tx.n_steps - round(30 / params.dt)
        latency = (comp[0] - tx.t_data) * params.dt if comp else None
        rows.append({"a": a, "b": b, "correct": correct, "reset_ok": reset,
                     "fired_y1": bool(y1), "fired_y0": bool(y0), "completion": bool(comp),
                     "completion_latency_ms": latency,
                     "passes_existing_checks": bool(correct and reset and latency is not None and latency < 60),
                     "spikes_per_role": {role: len(ss) for role, ss in roles.items()}})
    return rows


def campaign(fixtures: list[Fixture], *, replicas: int = 50, calibration_replicas: int = 10,
             factorial: bool = True, params: Params = Params()) -> dict:
    if not fixtures or fixtures[0].name != "F1_single":
        raise ValueError("F1_single must be first: its calibration defines matched-input controls")
    results = []
    isolated_calibration = {}
    for fixture in fixtures:
        for name, rules in CONFIGS.items():
            print(f"[compare] {fixture.name} {name}", flush=True)
            single, _ = run_protocol(fixture, rules, drive_ms=5.1, off_ms=94.9,
                                      replicas=1, single=True, params=params)
            drive, control = run_protocol(fixture, rules, replicas=replicas, params=params)
            probe, probed = run_protocol(fixture, rules, replicas=replicas, probe=True, params=params)
            probe["probe"] = probe_metrics(control, probed, fixture.target)
            # Equal amplitudes isolate the four switches. A second amplitude is the
            # actual released stimulus, so M5 is not mislabeled at 1718.75 mV.
            released, _ = run_protocol(fixture, rules, quanta=SHIU_QUANTA,
                                       replicas=replicas, params=params)
            row = {"fixture": fixture.name, "configuration": name, "rules": asdict(rules),
                   "P_A": single, "P_B": drive, "P_C": probe,
                   "P_B_68_75mv": released,
                   "P_D": [calibrate(fixture, rules, rate, replicas=calibration_replicas, params=params)
                           for rate in RATES]}
            if fixture.name == "F1_single":
                isolated_calibration[name] = {r["target_hz"]: r["calibrated_quanta"] for r in row["P_D"]}
            for calibration in row["P_D"]:
                rate = calibration["target_hz"]
                q = isolated_calibration[name][rate]
                calibration["F1_matched_input_quanta"] = q
                calibration["F1_matched_input_measurement"] = (
                    calibration["measurement"] if fixture.name == "F1_single" else matched_measurement(
                        fixture, rules, q, rate, calibration_replicas, params))
            if fixture.name == "F1_single":
                row["transfer"] = []
                for rate in (0, *RATES):
                    metrics = drive if rate == 2 else run_protocol(
                        fixture, rules, rate_hz=rate, replicas=replicas, params=params)[0]
                    row["transfer"].append({"event_rate_hz": rate,
                                            "firing_rate_hz": metrics["target_drive_rate_hz"]})
            if fixture.name == "F3_h0":
                row["and_transactions"] = h0_transactions(fixture, rules, params)
            results.append(row)
    appendix = []
    if factorial:
        for clear, freeze, target, norefrac in itertools.product((False, True), (False, True),
                                                                ("g", "v"), (False, True)):
            rules = ModelRules(clear, freeze, target, norefrac)
            metrics, _ = run_protocol(small_fixtures(params)[0], rules, drive_ms=5.1,
                                      off_ms=94.9, replicas=1, single=True, params=params)
            appendix.append({"rules": asdict(rules), "P_A": metrics})
    return {
        "schema_version": 1, "params": asdict(params),
        "protocol": {"replicas": replicas, "calibration_replicas": calibration_replicas,
                     "drive": "periodic, uniformly stratified phases (b+0.5)/replicas",
                     "main_quanta": FORCE_QUANTA, "released_quanta": SHIU_QUANTA,
                     "matched": "per fixture/configuration, event rates 2/10/50 Hz for 1 s; amplitude calibrated separately",
                     "matched_input_control": "replay each configuration's F1-calibrated amplitude into F2/F3; recurrent firing may differ"},
        "fixtures": [{"id": f.name, "n": f.topo.n, "target": f.target,
                      "topology_sha256": hashlib.sha256(b"".join(
                          a.tobytes() for a in (f.topo.src, f.topo.dst, f.topo.quanta, f.topo.delay))).hexdigest(),
                      "provenance": f.provenance} for f in fixtures],
        "pair_quanta": PAIR_QUANTA, "conditions": results,
        "F1_factorial_single_event_appendix": appendix,
    }


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--mcns-dir", type=Path)
    parser.add_argument("--synthetic", action="store_true", help="use the repository test_embed_h0 fixture")
    parser.add_argument("--out", type=Path, default=Path("docs/stage_h/model_comparison.json"))
    parser.add_argument("--replicas", type=int, default=50)
    parser.add_argument("--calibration-replicas", type=int, default=10)
    parser.add_argument("--no-factorial", action="store_true")
    args = parser.parse_args(argv)
    if args.synthetic and args.mcns_dir:
        parser.error("choose --synthetic or --mcns-dir")
    if args.replicas < 1 or args.calibration_replicas < 1:
        parser.error("replica counts must be positive")
    fixtures = small_fixtures()
    h0_status = {"status": "not_run", "reason": "no --mcns-dir or --synthetic supplied"}
    try:
        if args.synthetic:
            # Development-only path, deliberately reuse the existing planted fixture.
            import runpy
            factory = runpy.run_path(str(Path(__file__).resolve().parents[2] / "tests/test_embed_h0.py"))["synthetic_mcns"]
            fixtures.append(h0_fixture(factory(), source="tests/test_embed_h0.py:synthetic_mcns(seed=0)"))
            h0_status = {"status": "synthetic_only"}
        elif args.mcns_dir:
            # Cached data are read in place. No download, symlink or external write.
            m = load_mcns(data_dir=args.mcns_dir)
            fixtures.append(h0_fixture(m, source=str(args.mcns_dir)))
            del m
            h0_status = {"status": "real_data", "mcns_dir": str(args.mcns_dir)}
    except (OSError, ValueError) as exc:
        h0_status = {"status": "not_run", "reason": f"{type(exc).__name__}: {exc}",
                     "mcns_dir": str(args.mcns_dir)}
        print(f"[compare] F3 unavailable: {h0_status['reason']}", flush=True)
    result = campaign(fixtures, replicas=args.replicas, calibration_replicas=args.calibration_replicas,
                      factorial=not args.no_factorial)
    result["h0_status"] = h0_status
    if h0_status["status"] == "not_run":
        result["conditions"].extend({"fixture": "F3_h0", "configuration": name,
                                     "status": "not_run", "reason": h0_status["reason"]} for name in CONFIGS)
    result["findings"] = {
        "h0_transactions_passing_existing_checks": {
            row["configuration"]: sum(tx["passes_existing_checks"] for tx in row["and_transactions"])
            for row in result["conditions"] if "and_transactions" in row},
        "local_firing_calibrations_matched": sum(
            trial["matched_within_5_percent"] for row in result["conditions"]
            for trial in row.get("P_D", [])),
        "recommendation": "M5_shiu for new biological/connectome experiments, subject to owner choice and circuit recertification; preserve M0_current as a labelled engineering/historical comparison",
        "defaults_changed": False,
        "limitations": [
            "Only 1-, 2- and 15-neuron fixtures; no whole-brain or full-graph simulation",
            "Does not establish that surround runaway or 0/50 adder outcomes were stimulation artefacts",
            "Periodic phase ensemble, not independent Poisson stimulation",
            "Finite observation windows; recurrence can prevent local firing-rate matching",
            "Amplitude bisection has no global monotonicity guarantee in recurrent circuits",
            "An extra probe-window spike may be a phase shift of existing recurrent activity",
        ],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(f"[compare] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
