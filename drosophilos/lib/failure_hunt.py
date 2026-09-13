"""Reproduce campaign failures and dump their anatomy: which consumer stage did not fire."""

from __future__ import annotations

import sys
from collections import Counter

import numpy as np

from ..protocol.handshake import build_channel
from ..protocol.token import decode_at, rails_for
from ..sim.lif_torch import TorchSim
from ..sim.model import Params
from .campaign import Perturbation


def hunt(pert: Perturbation, seed: int = 0, B: int = 400, K: int = 3, T: int = 3200, max_dump: int = 6):
    params = Params()
    ch = build_channel(params, 4); net = ch.net; topo = net.topology(); P, Q = ch.producer, ch.consumer; roles = net.roles
    rng = np.random.default_rng(seed)
    q = np.rint(topo.quanta.astype(float)[None, :] * np.exp(rng.normal(0, pert.weight_sigma, size=(B, topo.nnz)))).astype(np.int32)
    vth = params.V_th + rng.normal(0, pert.th_sigma_mv, size=(B, topo.n)); bias = rng.normal(0, pert.bias_sigma_mv, size=(B, topo.n))
    sim = TorchSim(topo, params, n_nodes=B, V_th=vth, bias=bias, quanta=q)
    n_steps = K * T + 200
    words = rng.integers(0, 16, size=(B, K)); offs = {}
    for b in range(B):
        st_, ne_, qu_ = [], [], []
        for k in range(K):
            for i, r in rails_for(int(words[b, k]), 4):
                off = int(rng.integers(0, pert.arrival_jitter_steps + 1)) if pert.arrival_jitter_steps else 0
                offs[(b, k, i)] = off
                st_.append(k * T + 50 + off); ne_.append(P.rails[i][r].u); qu_.append(ch.drive.ignite)
        if pert.stray_rate_hz > 0:
            p = pert.stray_rate_hz * params.dt / 1000.0; n_ev = rng.binomial(n_steps * topo.n, p)
            st_.extend(rng.integers(0, n_steps, size=n_ev).tolist()); ne_.extend(rng.integers(0, topo.n, size=n_ev).tolist()); qu_.extend([pert.stray_quanta] * n_ev)
        sim.add_events(b, st_, ne_, qu_)
    sim.run(n_steps)
    ev = sim.trace.events; comp, cleared, ready = Q.completion.u, P.ready, Q.ready; taps = Q.rail_taps
    bad = []
    for b in range(B):
        for k in range(K):
            m = (ev["node"] == b) & (ev["step"] >= k * T) & (ev["step"] < (k + 1) * T)
            st, nu = ev["step"][m], ev["neuron"][m]
            acc = st[nu == comp]
            if len(acc) == 0:
                bad.append((b, k, "no_accept")); continue
            val, status = decode_at(sim.trace, taps, int(acc[0]), 2 * ch.drive.loop_period_steps, node=b)
            if status != "valid" or val != words[b, k]:
                bad.append((b, k, f"{status}:{val}!={words[b,k]}")); continue
            if (nu[st >= acc[0]] == cleared).sum() == 0:
                bad.append((b, k, "no_cleared")); continue
            s_cl = st[(st >= acc[0]) & (nu == cleared)][0]
            if (nu[st >= s_cl] == ready).sum() == 0:
                bad.append((b, k, "no_ready"))
    print(f"[hunt] {len(bad)} failures in {B*K} transactions at {pert.describe()}")
    for b, k, why in bad[:max_dump]:
        m = (ev["node"] == b) & (ev["step"] >= k * T) & (ev["step"] < (k + 1) * T)
        st, nu = ev["step"][m], ev["neuron"][m]
        keep = lambda r: r.startswith(("Q.b", "Q.valid", "Q.comp", "Q.fault", "Q.reset", "Q.ready", "P.reset", "P.ready", "data.")) and not r.endswith((".v", "_inh"))
        first = {}
        for s_, n_ in zip(st, nu):
            r = roles[n_]
            if keep(r): first.setdefault(r, (0, s_)); first[r] = (first[r][0] + 1, first[r][1])
        want = [f"P.b{i}r{(words[b,k]>>i)&1}.u" for i in range(4)] + [f"data.b{i}r{(words[b,k]>>i)&1}.edge" for i in range(4)] + [f"Q.b{i}r{(words[b,k]>>i)&1}.u" for i in range(4)]
        missing = [w for w in want if w not in first]
        print(f"  node {b} tx {k} word {words[b,k]:04b} -> {why}; offsets {[offs[(b,k,i)] for i in range(4)]}; missing on the DATA path: {missing}")
        print("     " + ", ".join(f"{r}×{c}@{(s_-k*T)*0.1:.0f}ms" for r, (c, s_) in sorted(first.items(), key=lambda kv: kv[1][1])[:24]))
    return bad


if __name__ == "__main__":
    pert = Perturbation(0.04, 0.2, 0.2, 5.0, 150, 100)
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    hunt(pert, seed=seed)
