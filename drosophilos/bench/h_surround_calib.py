"""Calibrate the circuit-free MCNS surround; cluster CPU runs, compact JSONL output.

Each configuration starts fresh P1 sensory, P2 point-source replicates, and P3 rest trials.
Opt-in P4 tests labellar-bristle responsiveness (a proxy, not sugar-specific validation).
No spike traces are retained: per-step spike batches are reduced and immediately discarded.
See docs/stage_h/surround_calibration.md for metric definitions and cluster commands.
"""

from __future__ import annotations

import argparse
import dataclasses
import itertools
import json
from pathlib import Path

import numpy as np

from ..connectome import mcns_download
from ..connectome.embed_image import (Image, full_graph_topology, surround_events,
                                      surround_populations, validate_surround_settings)
from ..connectome.mcns import MCNS, load_mcns
from ..sim.model import Params, Topology
from ..sim.ref64 import RefSim

MCNS_NEURONS = 166_700
BIN_MS = 10
DRIVE_MS = 300
OFF_MS = 300
REST_MS = 200
POINT_RATE_HZ = 213
P4_OFF_MS = 200
P4_SEED = 0


def parse_grid(value: str) -> list[dict]:
    """Cartesian grid; omitted axes use defaults. Reject typos rather than skipping axes."""
    axes = {"exc": [1.0], "inh": [1.0], "cap": [None]}
    seen = set()
    try:
        for item in value.split(";"):
            key, values = (s.strip() for s in item.split("="))
            if key not in axes or key in seen:
                raise ValueError("grid axes must be unique exc, inh, cap")
            seen.add(key)
            axes[key] = [None if key == "cap" and s.strip().lower() == "none" else float(s)
                         for s in values.split(",")]
        configs = []
        for exc, inh, cap in itertools.product(axes["exc"], axes["inh"], axes["cap"]):
            validate_surround_settings(exc, inh, cap)
            config = dict(surround_scale_exc=exc, surround_scale_inh=inh, surround_input_cap=cap)
            if config not in configs:
                configs.append(config)
        return configs
    except (ValueError, TypeError) as exc:
        raise ValueError(f"invalid grid {value!r}: {exc}") from exc


def empty_image() -> Image:
    """An image with no hosts, designed edges, synthetic neurons, or bias edits."""
    return Image(0, Topology.empty(0), np.empty(0, np.int64), np.empty(0, np.int64),
                 [], [], {}, [], [], [], True)


class ActivityMetrics:
    """Small streaming accumulator; every interval is left-inclusive, right-exclusive.

    Population rates are total spikes / second (divide by n for the per-neuron rate).
    The reference peak is the drive's maximum 10-ms bin. Decay requires staying below
    1% of that peak through the remaining observation, to avoid transient troughs.
    Persistence uses the off tail versus the mean drive bin; runaway/rising_end_drive
    retain their deprecated definitions solely for output compatibility.
    """

    def __init__(self, n: int, params: Params, drive_ms: int = DRIVE_MS,
                 total_ms: int = DRIVE_MS + OFF_MS, driven_sensory=()):
        self.n = n
        self.dt = params.dt
        self.bin_steps = int(round(BIN_MS / params.dt))
        self.drive_steps = int(round(drive_ms / params.dt))
        self.total_steps = int(round(total_ms / params.dt))
        self.drive_ms = drive_ms
        self.total_ms = total_ms
        self.bins = np.zeros(int(np.ceil(total_ms / BIN_MS)), np.int64)
        self.counts = np.zeros(n, np.int64)
        self.drive_counts = np.zeros(n, np.int64)
        self.tail_counts = np.zeros(n, np.int64)
        self.driven_sensory = np.asarray(driven_sensory, np.int64)

    def observe(self, step: int, neurons: np.ndarray) -> None:
        self.bins[step // self.bin_steps] += len(neurons)
        np.add.at(self.counts, neurons, 1)
        if step < self.drive_steps:
            np.add.at(self.drive_counts, neurons, 1)
        if step >= self.total_steps - int(round(100 / self.dt)):
            np.add.at(self.tail_counts, neurons, 1)

    def finish(self, superclasses=None) -> dict:
        active = self.counts > 0
        n_active = int(active.sum())
        n_drive_bins = self.drive_ms // BIN_MS
        during = self.bins[:n_drive_bins]
        off = self.bins[n_drive_bins:] if self.drive_ms else np.empty(0, np.int64)
        peak = int(during.max()) if len(during) else 0
        decay = None
        rising = lingering = False
        if self.drive_ms:
            if peak:
                below = off < .01 * peak
                # First bin of the final continuously-below-threshold suffix.
                sustained = np.logical_and.accumulate(below[::-1])[::-1]
                hits = np.flatnonzero(sustained)
                decay = int(hits[0] * BIN_MS) if len(hits) else None
            elif not off.any():
                decay = 0
            final = during[-5:]
            if len(final) >= 2:
                x = np.arange(len(final)) - (len(final) - 1) / 2
                rising = bool(np.dot(x, final) > 0)  # sign of least-squares slope
            at_100 = 100 // BIN_MS
            lingering = bool(len(off) > at_100 and off[at_100] > .1 * peak)
        rates = self.drive_counts[self.drive_counts > 0] * (1000 / self.drive_ms) if self.drive_ms else []
        tail_spikes = int(self.tail_counts.sum())
        # Use only off-period bins, never the end of the drive. P3 has no off period.
        off_tail = off[-(100 // BIN_MS):]
        off_mean = float(off_tail.mean()) if len(off_tail) else None
        drive_mean = float(during.mean()) if len(during) else 0.0
        recruited = active.copy()
        recruited[self.driven_sensory] = False
        result = {
            "n_neurons": self.n, "fraction_denominator_mcns": MCNS_NEURONS,
            "active_neurons": n_active, "active_fraction_mcns": n_active / MCNS_NEURONS,
            "recruited_nonsensory": int(recruited.sum()),
            "off_tail_spikes_per_bin": off_mean,
            "decays_to_zero": off_mean < 1 if off_mean is not None else None,
            # A completely silent trial is not self-sustaining (zero reference rate).
            "self_sustained": bool(off_mean is not None and off_mean > 0
                                   and off_mean >= .1 * drive_mean),
            "active_fraction_simulated": n_active / self.n if self.n else 0.0,
            "total_spikes": int(self.counts.sum()), "bin_ms": BIN_MS,
            "spikes_per_bin": self.bins.tolist(), "drive_ms": self.drive_ms,
            "total_ms": self.total_ms, "drive_active_neurons": int((self.drive_counts > 0).sum()),
            "mean_active_drive_hz": float(np.mean(rates)) if len(rates) else None,
            "median_active_drive_hz": float(np.median(rates)) if len(rates) else None,
            "peak_drive_spikes_per_bin": peak,
            "last_100ms": {"spikes": tail_spikes, "active_neurons": int((self.tail_counts > 0).sum()),
                           "population_hz": tail_spikes * 10,
                           "mean_per_neuron_hz": tail_spikes * 10 / self.n if self.n else 0.0},
            "decay_ms": decay, "rising_end_drive": rising,
            "above_10pct_peak_at_100ms_off": lingering, "runaway": rising or lingering,
        }
        if superclasses is not None:
            classes = np.asarray(superclasses, dtype=str)
            breakdown = {}
            for label in np.unique(classes):
                mask = classes == label
                count = int((active & mask).sum())
                breakdown[str(label)] = {"active_neurons": count, "neurons": int(mask.sum()),
                                         "active_fraction": count / int(mask.sum())}
            result["by_superclass"] = breakdown
        return result


def simulate_protocol(topology: Topology, params: Params, *, drive_ms: int, total_ms: int,
                      events=None, sources=(), superclasses=None, driven_sensory=(),
                      readout_neurons=(), readout_groups=None) -> dict:
    """Run fresh, reducing and dropping trace batches each step.

    Point sources are clamped to one spike on each rounded 1/213-s tick during drive.
    Clearing source current and resetting voltage between ticks prevents the very large
    H0 forcing pulse from generating extra source spikes. After drive they evolve freely.
    """
    sim = RefSim(topology, params)
    metrics = ActivityMetrics(topology.n, params, drive_ms, total_ms, driven_sensory)
    if events is not None and len(events[0]):
        sim.add_events(0, *events)
    sources = np.asarray(sources, np.int64)
    ticks = set(np.rint(np.arange(0, drive_ms, 1000 / POINT_RATE_HZ) / params.dt).astype(int))
    for step in range(metrics.total_steps):
        if len(sources) and step < metrics.drive_steps:
            sim.V[0, sources] = params.V_th + 1 if step in ticks else params.E_L
            sim.g[0, sources] = 0
            sim.r[0, sources] = 0
        sim.step()
        for neurons in sim._spk_neuron:
            metrics.observe(step, neurons)
        sim._spk_step.clear()
        sim._spk_node.clear()
        sim._spk_neuron.clear()
    result = metrics.finish(superclasses)
    if len(readout_neurons):
        result["readout_neurons"] = [
            {"spikes": int(metrics.counts[i]),
             "rate_hz": float(metrics.counts[i] * 1000 / total_ms),
             "drive_spikes": int(metrics.drive_counts[i]),
             "drive_rate_hz": float(metrics.drive_counts[i] * 1000 / drive_ms) if drive_ms else None}
            for i in readout_neurons]
    if readout_groups is not None:
        result["readout_active_neurons"] = {
            label: int(np.count_nonzero(metrics.counts[indices]))
            for label, indices in readout_groups.items()}
    if len(sources):
        result["source_spikes_during_drive"] = metrics.drive_counts[sources].tolist()
        result["recruited_non_sources"] = int((metrics.counts > 0).sum() - (metrics.counts[sources] > 0).sum())
    return result


def p4_populations(m: MCNS) -> tuple[dict[str | None, np.ndarray], np.ndarray, dict[str, np.ndarray]]:
    """Resolve typed drives and the all-bristle drive (None), aligning annotations by body ID."""
    neurons = m.neurons.copy()
    missing = [column for column in ("type", "subclass") if column not in neurons]
    if missing:
        import pandas as pd

        annotations = pd.read_feather(mcns_download.path_of("annotations"),
                                      columns=["bodyId", *missing]).set_index("bodyId")
        if not neurons["bodyId"].isin(annotations.index).all():
            raise ValueError("P4 annotations are missing MCNS neuron body IDs")
        aligned = annotations.reindex(neurons["bodyId"])
        for column in missing:
            neurons[column] = aligned[column].to_numpy()
    labellar = neurons["subclass"].eq("labellar bristle").fillna(False).to_numpy(bool)
    if not labellar.any():
        raise ValueError("P4 requires neurons with subclass 'labellar bristle'")
    types = neurons["type"]
    populations: dict[str | None, np.ndarray] = {
        str(label): np.flatnonzero(labellar & types.eq(label).fillna(False).to_numpy(bool))
        for label in sorted(types[labellar].dropna().unique())}
    populations[None] = np.flatnonzero(labellar)
    mn9 = np.flatnonzero(types.eq("MN9").fillna(False).to_numpy(bool))
    if not len(mn9):
        raise ValueError("P4 requires type 'MN9' readout neurons")
    classes = neurons["superclass"]
    groups = {"descending": np.flatnonzero(classes.eq("descending_neuron").fillna(False).to_numpy(bool)),
              "motor": np.flatnonzero(classes.isin(["cb_motor", "vnc_motor"]).to_numpy())}
    return populations, mn9, groups


def responsiveness_protocol(m: MCNS, topology: Topology, params: Params, *,
                            rate: float = 100, drive_ms: int = 300) -> dict:
    """P4: each labellar-bristle type, then all together; no other external drive."""
    populations, mn9, groups = p4_populations(m)
    drives = []
    for label, sensory in populations.items():
        events = surround_events("poisson", sensory, params, 0, int(round(drive_ms / params.dt)),
                                 np.random.default_rng(P4_SEED), rate_hz=rate)
        result = simulate_protocol(topology, params, drive_ms=drive_ms,
                                   total_ms=drive_ms + P4_OFF_MS, events=events,
                                   driven_sensory=sensory, readout_neurons=mn9,
                                   readout_groups=groups)
        drives.append({"type": label, "driven_neurons": int(len(sensory)),
                       "forcing_events": int(len(events[0])),
                       "mn9": [{"body_id": int(body), **counts}
                               for body, counts in zip(m.bodies_of(mn9), result["readout_neurons"])],
                       **{f"{group}_active_neurons": count
                          for group, count in result["readout_active_neurons"].items()},
                       **{key: result[key] for key in ("recruited_nonsensory", "decays_to_zero",
                                                       "off_tail_spikes_per_bin", "self_sustained")}})
    return {"seed": P4_SEED, "rate_hz": rate, "drive_ms": drive_ms, "off_ms": P4_OFF_MS,
            "driven_type_counts": {label: int(len(indices)) for label, indices in populations.items()
                                   if label is not None},
            "untyped_labellar_neurons": int(len(populations[None]) - sum(
                len(indices) for label, indices in populations.items() if label is not None)),
            "mn9_neurons": int(len(mn9)),
            **{f"{label}_neurons": int(len(indices)) for label, indices in groups.items()},
            "drives": drives}


def calibrate_config(m: MCNS, base: Topology, params: Params, config: dict, *, rate: float = 2,
                     n_sources: int = 1, seeds=(0, 1, 2), sensory_seed: int = 0,
                     p4: bool = False, p4_only: bool = False,
                     p4_rate: float = 100, p4_ms: int = 300) -> dict:
    fg = full_graph_topology(m, params, empty_image(), base=base, **config)
    record = {"schema": "surround_calibration_v1", "settings": config, "connectome": m.version,
              "params": dataclasses.asdict(params), "full_graph_counts": fg.counts}
    if p4 or p4_only:
        record["P4_responsiveness"] = responsiveness_protocol(m, fg.topology, params,
                                                            rate=p4_rate, drive_ms=p4_ms)
    if p4_only:
        return record
    sensory = surround_populations(m)["sensory"]
    candidates = np.setdiff1d(np.arange(m.n), sensory)
    if n_sources > len(candidates):
        raise ValueError("n_sources exceeds the non-sensory population")
    events = surround_events("poisson", sensory, params, 0, int(round(DRIVE_MS / params.dt)),
                             np.random.default_rng(sensory_seed), rate_hz=rate)
    p1 = simulate_protocol(fg.topology, params, drive_ms=DRIVE_MS, total_ms=DRIVE_MS + OFF_MS,
                           events=events, superclasses=m.neurons["superclass"].to_numpy(),
                           driven_sensory=sensory)
    p1.update(seed=sensory_seed, rate_hz=rate, sensory_neurons=int(len(sensory)),
              forcing_events=int(len(events[0])))
    p2 = []
    for seed in seeds:
        sources = np.random.default_rng(seed).choice(candidates, n_sources, replace=False)
        result = simulate_protocol(fg.topology, params, drive_ms=DRIVE_MS, total_ms=DRIVE_MS + OFF_MS,
                                    sources=sources)
        result.update(seed=int(seed), source_body_ids=m.bodies_of(sources).astype(int).tolist(),
                      source_rate_hz=POINT_RATE_HZ)
        p2.append(result)
    p3 = simulate_protocol(fg.topology, params, drive_ms=0, total_ms=REST_MS)
    p3["any_activity"] = bool(p3["total_spikes"])
    record.update(P1_sensory=p1, P2_point_sources=p2, P3_rest=p3)
    return record


def parse(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--grid", default="exc=1;inh=1;cap=none",
                        help="Cartesian axes, e.g. 'exc=1,0.5;inh=1,2;cap=none,500'")
    parser.add_argument("--rate", type=float, default=2, help="P1 sensory drive Hz")
    parser.add_argument("--n-sources", type=int, default=1)
    parser.add_argument("--seeds", default="0,1,2", help="comma-separated P2 source-selection seeds")
    parser.add_argument("--sensory-seed", type=int, default=0)
    parser.add_argument("--p4", action="store_true", help="also run labellar-bristle responsiveness proxy")
    parser.add_argument("--p4-only", action="store_true", help="run only P4; implies --p4")
    parser.add_argument("--p4-rate", type=float, default=100, help="P4 independent Poisson input Hz")
    parser.add_argument("--p4-ms", type=int, default=300, help="P4 drive duration in ms (multiple of 10)")
    parser.add_argument("--out", type=Path, required=True, help="one compact JSON record per config (JSONL)")
    args = parser.parse_args(argv)
    try:
        args.configs = parse_grid(args.grid)
        args.seeds = tuple(dict.fromkeys(int(s.strip()) for s in args.seeds.split(",")))
        if not args.seeds or min(args.seeds) < 0 or args.sensory_seed < 0:
            raise ValueError("seeds must be nonnegative integers")
        if args.n_sources < 1:
            raise ValueError("n-sources must be positive")
        for name, rate in (("rate", args.rate), ("p4-rate", args.p4_rate)):
            if not np.isfinite(rate) or not 0 <= rate <= 1000 / Params().dt:
                raise ValueError(f"{name} must be finite and in [0, 1000/dt]")
        if args.p4_ms <= 0 or args.p4_ms % BIN_MS:
            raise ValueError("p4-ms must be a positive multiple of 10 ms")
    except ValueError as exc:
        parser.error(str(exc))
    return args


def run(args) -> int:
    """Load MCNS and its anatomical topology once; stream a grid to disk, flushing per config."""
    params = Params()
    m = load_mcns()
    base = m.topology(params)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as out:
        for index, config in enumerate(args.configs):
            record = calibrate_config(m, base, params, config, rate=args.rate, n_sources=args.n_sources,
                                      seeds=args.seeds, sensory_seed=args.sensory_seed,
                                      p4=args.p4, p4_only=args.p4_only,
                                      p4_rate=args.p4_rate, p4_ms=args.p4_ms)
            out.write(json.dumps(record, separators=(",", ":"), allow_nan=False) + "\n")
            out.flush()
            print(f"[surround-calib] {index + 1}/{len(args.configs)} {config}", flush=True)
    return len(args.configs)


if __name__ == "__main__":
    run(parse())
