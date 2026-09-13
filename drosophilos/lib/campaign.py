"""Perturbation campaign: many independent copies of a channel (batch dimension), each with
its own weight noise, threshold drift, bias, stray background input, operand arrival
jitter and random words, transacting on a fixed schedule. Every transaction is decoded
post hoc from the spike trace and classified. Results stream to a JSONL file per chunk so
a long run can be resumed and inspected while it runs.

Error classes:
    wrong_value     decoded value != expected (the only silent failure class)
    fault           fault gate fired (both rails on a bit)
    no_accept       completion never fired in the window
    no_cleared      CLEARED never fired after ACCEPT
    no_ready        READY never fired before the next load
    late_activity   consumer spikes between READY and the next load (stale state)
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from ..sim.lif_torch import TorchSim
from ..sim.model import Params
from ..sim.trace import SpikeTrace
from ..protocol.handshake import Channel
from ..protocol.token import rails_for


@dataclass(frozen=True)
class Perturbation:
    weight_sigma: float = 0.05  # multiplicative log-normal noise on every synapse
    th_sigma_mv: float = 0.25  # per-neuron threshold drift
    bias_sigma_mv: float = 0.25  # per-neuron bias
    stray_rate_hz: float = 5.0  # background Poisson input on every neuron
    stray_quanta: int = 150  # ~2.6 mV per stray pulse
    arrival_jitter_steps: int = 100  # per-bit load offset in [0, jitter]

    def describe(self) -> str:
        return (f"w~LN(0,{self.weight_sigma}), Vth±{self.th_sigma_mv} mV, bias±{self.bias_sigma_mv} mV, "
                f"stray {self.stray_rate_hz} Hz×{self.stray_quanta} q, arrival jitter ≤{self.arrival_jitter_steps} steps")


def _decode_node(steps, neurons, taps, comp, cleared, ready, faults, consumer_set, loads, expected, window):
    """Classify every transaction of one node from its (sorted) spike arrays."""
    out = []
    n_tx = len(loads)
    for k in range(n_tx):
        lo = loads[k]
        hi = loads[k + 1] if k + 1 < n_tx else steps[-1] + 1 if len(steps) else lo + 1
        m = (steps >= lo) & (steps < hi)
        st, nu = steps[m], neurons[m]
        cls, acc_lat = None, None
        acc = st[nu == comp]
        if len(acc) == 0:
            cls = "no_accept"
        else:
            s_acc = int(acc[0])
            acc_lat = s_acc - lo
            recent = nu[(st > s_acc - window) & (st <= s_acc)]
            active = set(recent.tolist())
            value, status = 0, "valid"
            for i, (r0, r1) in enumerate(taps):
                a0, a1 = r0 in active, r1 in active
                if a0 and a1:
                    status = "fault"
                elif not (a0 or a1):
                    status = "incomplete"
                elif a1:
                    value |= 1 << i
            if status != "valid":
                cls = "fault" if status == "fault" else "no_accept"
            elif value != expected[k]:
                cls = "wrong_value"
            elif (nu[st >= s_acc] == cleared).sum() == 0:
                cls = "no_cleared"
            else:
                s_cl = int(st[(st >= s_acc) & (nu == cleared)][0])
                rd = st[(st >= s_cl) & (nu == ready)]
                if len(rd) == 0:
                    cls = "no_ready"
                else:
                    s_rd = int(rd[0])
                    late = np.isin(nu[st > s_rd], list(consumer_set)).sum()
                    cls = "late_activity" if late else "ok"
        if cls == "ok" and (nu[(st >= lo)] == -1).any():
            pass
        n_fault = int(np.isin(nu, faults).sum())
        if cls == "ok" and n_fault:
            cls = "fault"
        out.append((cls, acc_lat, int(m.sum())))
    return out


def run_campaign(build_fn, params: Params, n_transactions: int, *, batch: int = 200, tx_per_chunk: int = 20,
                 tx_period_steps: int = 3200, pert: Perturbation = Perturbation(), seed: int = 0,
                 out_path: Path | None = None, expected_fn=None, word_fn=None, device: str = "cpu",
                 dtype=torch.float64, verbose: bool = True) -> dict:
    ch: Channel = build_fn()
    net, drive = ch.net, ch.drive
    topo = net.topology()
    P, Q = ch.producer, ch.consumer
    taps = Q.rail_taps
    comp, cleared, ready = Q.completion.u, P.ready, Q.ready
    faults = np.array(Q.fault, dtype=np.int64)
    consumer_set = {x for x, role in enumerate(net.roles) if role.startswith("Q.") and not role.startswith("Q.ready")}
    window = 2 * drive.loop_period_steps
    rng = np.random.default_rng(seed)
    base_q = topo.quanta.astype(np.float64)
    totals = {"ok": 0, "wrong_value": 0, "fault": 0, "no_accept": 0, "no_cleared": 0, "no_ready": 0, "late_activity": 0}
    lat_all, spikes_all = [], []
    done, chunk_id = 0, 0
    t_start = time.time()
    if out_path:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
    while done < n_transactions:
        B = min(batch, int(np.ceil((n_transactions - done) / tx_per_chunk)))
        K = tx_per_chunk
        # per-node perturbations
        q = np.rint(base_q[None, :] * np.exp(rng.normal(0, pert.weight_sigma, size=(B, topo.nnz)))).astype(np.int32)
        vth = params.V_th + rng.normal(0, pert.th_sigma_mv, size=(B, topo.n))
        bias = rng.normal(0, pert.bias_sigma_mv, size=(B, topo.n))
        sim = TorchSim(topo, params, n_nodes=B, V_th=vth, bias=bias, quanta=q, device=device, dtype=dtype)
        n_steps = K * tx_period_steps + 200
        # loads
        loads = np.arange(K) * tx_period_steps + 50
        words = np.zeros((B, K), dtype=np.int64)
        expected = np.zeros((B, K), dtype=np.int64)
        for b in range(B):
            st_, ne_, qu_ = [], [], []
            for k in range(K):
                w = int(word_fn(rng)) if word_fn else int(rng.integers(0, 1 << ch.width))
                words[b, k] = w
                expected[b, k] = expected_fn(w) if expected_fn else w
                for i, r in rails_for(w, ch.width):
                    off = int(rng.integers(0, pert.arrival_jitter_steps + 1)) if pert.arrival_jitter_steps else 0
                    st_.append(loads[k] + off); ne_.append(P.rails[i][r].u); qu_.append(drive.ignite)
            # stray background: Poisson on every neuron
            if pert.stray_rate_hz > 0:
                p = pert.stray_rate_hz * params.dt / 1000.0
                n_ev = rng.binomial(n_steps * topo.n, p)
                st_.extend(rng.integers(0, n_steps, size=n_ev).tolist())
                ne_.extend(rng.integers(0, topo.n, size=n_ev).tolist())
                qu_.extend([pert.stray_quanta] * n_ev)
            sim.add_events(b, st_, ne_, qu_)
        t0 = time.time()
        sim.run(n_steps)
        wall = time.time() - t0
        tr = sim.trace
        ev = tr.events
        order = np.lexsort((ev["step"], ev["node"]))
        ev = ev[order]
        node_bounds = np.searchsorted(ev["node"], np.arange(B + 1))
        chunk_counts = dict.fromkeys(totals, 0)
        lats, spk = [], []
        for b in range(B):
            sl = slice(node_bounds[b], node_bounds[b + 1])
            res = _decode_node(ev["step"][sl], ev["neuron"][sl], taps, comp, cleared, ready, faults, consumer_set,
                               loads.tolist(), expected[b].tolist(), window)
            for cls, lat, nsp in res:
                chunk_counts[cls] += 1
                if lat is not None:
                    lats.append(lat)
                spk.append(nsp)
        n_chunk = B * K
        done += n_chunk
        for k_, v in chunk_counts.items():
            totals[k_] += v
        lat_all.extend(lats); spikes_all.extend(spk)
        rec = {"chunk": chunk_id, "transactions": n_chunk, "counts": chunk_counts, "wall_s": round(wall, 1),
               "accept_latency_ms_mean": float(np.mean(lats) * params.dt) if lats else None,
               "accept_latency_ms_max": float(np.max(lats) * params.dt) if lats else None,
               "spikes_per_tx_mean": float(np.mean(spk)) if spk else None, "done": done}
        if out_path:
            with open(out_path, "a") as f:
                f.write(json.dumps(rec) + "\n")
        if verbose:
            errs = {k_: v for k_, v in chunk_counts.items() if k_ != "ok" and v}
            print(f"[campaign] chunk {chunk_id}: {n_chunk} tx in {wall:.0f} s, errors {errs or 'none'}, "
                  f"done {done}/{n_transactions}, elapsed {(time.time()-t_start)/60:.1f} min", flush=True)
        chunk_id += 1
    n_err = sum(v for k_, v in totals.items() if k_ != "ok")
    summary = {
        "transactions": done, "counts": totals, "errors": n_err,
        "error_rate_upper_95": (3.0 / done) if n_err == 0 else n_err / done,
        "accept_latency_ms": {"mean": float(np.mean(lat_all) * params.dt), "max": float(np.max(lat_all) * params.dt),
                              "p99": float(np.percentile(lat_all, 99) * params.dt)} if lat_all else None,
        "spikes_per_tx": {"mean": float(np.mean(spikes_all)), "max": int(np.max(spikes_all))} if spikes_all else None,
        "perturbation": asdict(pert), "perturbation_text": pert.describe(),
        "neurons": net.n, "synapses": net.nnz, "tx_period_ms": tx_period_steps * params.dt,
        "wall_minutes": (time.time() - t_start) / 60, "device": device, "dtype": str(dtype),
    }
    if out_path:
        Path(str(out_path).replace(".jsonl", "_summary.json")).write_text(json.dumps(summary, indent=1))
    return summary


if __name__ == "__main__":
    import argparse

    from ..protocol.handshake import build_channel

    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--width", type=int, default=4)
    ap.add_argument("--batch", type=int, default=200)
    ap.add_argument("--out", default="data/m1/campaign_4bit.jsonl")
    ap.add_argument("--weight-sigma", type=float, default=0.05)
    ap.add_argument("--stray-hz", type=float, default=5.0)
    ap.add_argument("--th-sigma", type=float, default=0.25)
    ap.add_argument("--bias-sigma", type=float, default=0.25)
    ap.add_argument("--jitter", type=int, default=100)
    ap.add_argument("--period", type=int, default=3200)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    params = Params()
    pert = Perturbation(a.weight_sigma, a.th_sigma, a.bias_sigma, a.stray_hz, 150, a.jitter)
    s = run_campaign(lambda: build_channel(params, a.width), params, a.n, batch=a.batch, tx_period_steps=a.period,
                     pert=pert, seed=a.seed, out_path=Path(a.out))
    print(json.dumps({k: v for k, v in s.items() if k != "perturbation"}, indent=1))
