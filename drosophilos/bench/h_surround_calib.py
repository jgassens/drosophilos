"""Calibrate the circuit-free MCNS surround; cluster CPU runs, compact JSONL output.

Each configuration starts fresh P1 sensory, P2 point-source replicates, and P3 rest trials.
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
    """

    def __init__(self, n: int, params: Params, drive_ms: int = DRIVE_MS,
                 total_ms: int = DRIVE_MS + OFF_MS):
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
        result = {
            "n_neurons": self.n, "fraction_denominator_mcns": MCNS_NEURONS,
            "active_neurons": n_active, "active_fraction_mcns": n_active / MCNS_NEURONS,
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
                      events=None, sources=(), superclasses=None) -> dict:
    """Run fresh, reducing and dropping trace batches each step.

    Point sources are clamped to one spike on each rounded 1/213-s tick during drive.
    Clearing source current and resetting voltage between ticks prevents the very large
    H0 forcing pulse from generating extra source spikes. After drive they evolve freely.
    """
    sim = RefSim(topology, params)
    metrics = ActivityMetrics(topology.n, params, drive_ms, total_ms)
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
    if len(sources):
        result["source_spikes_during_drive"] = metrics.drive_counts[sources].tolist()
        result["recruited_non_sources"] = int((metrics.counts > 0).sum() - (metrics.counts[sources] > 0).sum())
    return result


def calibrate_config(m: MCNS, base: Topology, params: Params, config: dict, *, rate: float = 2,
                     n_sources: int = 1, seeds=(0, 1, 2), sensory_seed: int = 0) -> dict:
    fg = full_graph_topology(m, params, empty_image(), base=base, **config)
    sensory = surround_populations(m)["sensory"]
    candidates = np.setdiff1d(np.arange(m.n), sensory)
    if n_sources > len(candidates):
        raise ValueError("n_sources exceeds the non-sensory population")
    events = surround_events("poisson", sensory, params, 0, int(round(DRIVE_MS / params.dt)),
                             np.random.default_rng(sensory_seed), rate_hz=rate)
    p1 = simulate_protocol(fg.topology, params, drive_ms=DRIVE_MS, total_ms=DRIVE_MS + OFF_MS,
                           events=events, superclasses=m.neurons["superclass"].to_numpy())
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
    return {"schema": "surround_calibration_v1", "settings": config, "connectome": m.version,
            "params": dataclasses.asdict(params), "full_graph_counts": fg.counts,
            "P1_sensory": p1, "P2_point_sources": p2, "P3_rest": p3}


def parse(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--grid", default="exc=1;inh=1;cap=none",
                        help="Cartesian axes, e.g. 'exc=1,0.5;inh=1,2;cap=none,500'")
    parser.add_argument("--rate", type=float, default=2, help="P1 sensory drive Hz")
    parser.add_argument("--n-sources", type=int, default=1)
    parser.add_argument("--seeds", default="0,1,2", help="comma-separated P2 source-selection seeds")
    parser.add_argument("--sensory-seed", type=int, default=0)
    parser.add_argument("--out", type=Path, required=True, help="one compact JSON record per config (JSONL)")
    args = parser.parse_args(argv)
    try:
        args.configs = parse_grid(args.grid)
        args.seeds = tuple(dict.fromkeys(int(s.strip()) for s in args.seeds.split(",")))
        if not args.seeds or min(args.seeds) < 0 or args.sensory_seed < 0:
            raise ValueError("seeds must be nonnegative integers")
        if args.n_sources < 1:
            raise ValueError("n-sources must be positive")
        if not np.isfinite(args.rate) or not 0 <= args.rate <= 1000 / Params().dt:
            raise ValueError("rate must be finite and in [0, 1000/dt]")
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
                                      seeds=args.seeds, sensory_seed=args.sensory_seed)
            out.write(json.dumps(record, separators=(",", ":"), allow_nan=False) + "\n")
            out.flush()
            print(f"[surround-calib] {index + 1}/{len(args.configs)} {config}", flush=True)
    return len(args.configs)


if __name__ == "__main__":
    run(parse())
