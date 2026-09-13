"""M1 design rule check: compare storage primitives by broadcast cost, robustness and reset
cost, so the 213 Hz two-neuron loop is a measured choice, not an inherited one.

Candidates (all Profile 3, default neuron parameters unless stated):
  loop2_1.4x   two-neuron loop, 1.4x drive (the H0/M1 latch)
  loop2_1.05x  two-neuron loop, marginal drive (slower regeneration, lower rate)
  loop2_d10    two-neuron loop, 1.4x, 10 ms synaptic delays (Profile 3 only)
  ring4_1.4x   four-neuron ring, 1.4x
  ring8_1.4x   eight-neuron ring, 1.4x
Metrics per candidate: neurons, period, member rate, total spikes per 100 ms hold (the
broadcast cost), survival under weight noise (100 trials at 5 % and 10 %), minimal reset
(pulses x strength on both members of one link) that kills at every phase, recovery time
before a fresh ignition succeeds.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..sim.model import Params, Topology
from ..sim.ref64 import RefSim
from .netlist import Drive


def ring_topology(k: int, quanta: int, delay: int) -> Topology:
    src = list(range(k)); dst = [(i + 1) % k for i in range(k)]
    return Topology.from_edges(k, src, dst, [quanta] * k, [delay] * k)


def measure(params: Params, drive: Drive, k: int, scale: float, delay: int, rng: np.random.Generator) -> dict:
    q = int(round(scale * drive.single_need))
    topo = ring_topology(k, q, delay)
    sim = RefSim(topo, params)
    sim.add_events(0, [50], [0], [drive.ignite])
    sim.run(3000)
    st0 = sim.trace.neuron_steps(0)
    alive = len(st0) >= 3 and st0[-1] > 2500
    period = int(np.median(np.diff(st0)[-5:])) if len(st0) > 6 else None
    total = int(((sim.trace.events["step"] > 1000) & (sim.trace.events["step"] <= 2000)).sum())  # spikes per 100 ms
    # survival under weight noise
    surv = {}
    for sigma in (0.05, 0.10):
        ok = 0
        for _ in range(100):
            qs = np.rint(q * np.exp(rng.normal(0, sigma, size=k))).astype(int)
            t2 = Topology.from_edges(k, list(range(k)), [(i + 1) % k for i in range(k)], qs.tolist(), [delay] * k)
            s2 = RefSim(t2, params); s2.add_events(0, [50], [0], [drive.ignite]); s2.run(3000)
            st = s2.trace.neuron_steps(0)
            ok += int(len(st) >= 3 and st[-1] > 2500)
        surv[f"survives_100ms_sigma{int(sigma*100)}pct"] = ok
    # minimal reset: pulses x strength (fraction of loop drive) on members 0 and 1, all phases
    reset = None
    for pulses in (1, 4):
        for strength in (0.5, 0.75, 1.0, 1.5, 2.0, 3.0):
            kills = 0
            for tF in range(1000, 1000 + (period or 47), max(1, (period or 47) // 12)):
                qr = -int(round(strength * drive.loop))
                src = list(range(k)) + [k, k]; dst = [(i + 1) % k for i in range(k)] + [0, 1 % k]
                t3 = Topology.from_edges(k + 1, src, dst, [q] * k + [qr, qr], [delay] * k + [params.default_delay_steps] * 2)
                s3 = RefSim(t3, params); s3.add_events(0, [50], [0], [drive.ignite])
                s3.add_events(0, [tF + 53 * j for j in range(pulses)], [k] * pulses, [drive.loop] * pulses)
                s3.run(tF + 1500)
                last = max(s3.trace.neuron_steps(i).max() if len(s3.trace.neuron_steps(i)) else 0 for i in range(k))
                kills += int(last < tF + 53 * pulses + 400)
            n_ph = len(range(1000, 1000 + (period or 47), max(1, (period or 47) // 12)))
            if kills == n_ph:
                reset = {"pulses": pulses, "strength_x_loop": strength, "quanta_per_member_total": int(pulses * strength * drive.loop)}
                break
        if reset:
            break
    # recovery: earliest time after that reset at which a fresh 1.4x ignition succeeds
    recovery_ms = None
    if reset:
        for wait in range(0, 120, 5):
            qr = -int(round(reset["strength_x_loop"] * drive.loop)); pulses = reset["pulses"]
            src = list(range(k)) + [k, k]; dst = [(i + 1) % k for i in range(k)] + [0, 1 % k]
            t4 = Topology.from_edges(k + 1, src, dst, [q] * k + [qr, qr], [delay] * k + [params.default_delay_steps] * 2)
            s4 = RefSim(t4, params); s4.add_events(0, [50], [0], [drive.ignite])
            s4.add_events(0, [1000 + 53 * j for j in range(pulses)], [k] * pulses, [drive.loop] * pulses)
            t_re = 1000 + 53 * pulses + 100 + wait * 10
            s4.add_events(0, [t_re], [0], [drive.loop]); s4.run(t_re + 1500)
            st = s4.trace.neuron_steps(0)
            if len(st) and st[-1] > t_re + 1000:
                recovery_ms = (t_re - (1000 + 53 * pulses)) * params.dt
                break
    return {"neurons": k, "drive_x_need": scale, "delay_ms": delay * params.dt, "self_sustaining": bool(alive),
            "period_ms": None if period is None else period * params.dt,
            "member_rate_hz": None if not period else round(1000 / (period * params.dt), 1),
            "spikes_per_100ms_hold": total, **surv, "minimal_reset": reset,
            "recovery_after_reset_ms_1.4x_ignition": recovery_ms}


def main(out: Path = Path("data/m1/latch_alternatives.json")) -> dict:
    params = Params(); drive = Drive.from_params(params); rng = np.random.default_rng(0)
    d = params.default_delay_steps
    cands = {"loop2_1.4x": (2, 1.4, d), "loop2_1.05x": (2, 1.05, d), "loop2_1.4x_delay10ms": (2, 1.4, 100),
             "ring4_1.4x": (4, 1.4, d), "ring8_1.4x": (8, 1.4, d)}
    res = {name: measure(params, drive, k, s, dl, rng) for name, (k, s, dl) in cands.items()}
    out.parent.mkdir(parents=True, exist_ok=True); out.write_text(json.dumps(res, indent=1))
    for name, r in res.items():
        print(f"{name:>22}: alive {r['self_sustaining']!s:5} period {r['period_ms']} ms rate {r['member_rate_hz']} Hz spikes/100ms {r['spikes_per_100ms_hold']:>3} "
              f"survive5% {r['survives_100ms_sigma5pct']:>3} survive10% {r['survives_100ms_sigma10pct']:>3} reset {r['minimal_reset']} recovery {r['recovery_after_reset_ms_1.4x_ignition']} ms")
    return res


if __name__ == "__main__":
    main()
