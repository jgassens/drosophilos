"""Stage H0 experiments: run the embedded circuit isolated and inside the full MCNS graph,
with a silent, a background-driven, and a burst-driven surround. Produces the H0 report."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..sim.model import QUANTA_PER_SYNAPSE, Params, Topology
from ..sim.ref64 import RefSim
from ..sim.trace import SpikeTrace
from .embed_h0 import Embedding, Policy, Targets, best_embedding, find_embeddings
from .manifest import Manifest, ParameterEdit, params_dict
from .mcns import MCNS, load_mcns


# ---- topologies ---------------------------------------------------------------------
def isolated_topology(m: MCNS, emb: Embedding, params: Params, policy: Policy):
    """Circuit neurons only, re-indexed; designed edges at target quanta; parasitic edges
    zeroed (policy) or anatomical."""
    ns = emb.neurons
    idx = {x: i for i, x in enumerate(ns)}
    src, dst, q = [], [], []
    for e in emb.edges:
        src.append(idx[e.pre]); dst.append(idx[e.post]); q.append(e.quanta)
    if not policy.zero_parasitic:
        for p in emb.parasitic:
            src.append(idx[p.pre]); dst.append(idx[p.post]); q.append(int(m.sign[p.pre]) * p.count * QUANTA_PER_SYNAPSE)
    d = np.full(len(src), params.default_delay_steps)
    return Topology.from_edges(len(ns), src, dst, q, d), idx


def full_topology(m: MCNS, emb: Embedding, params: Params, policy: Policy, zero_outputs: bool = False):
    """Whole MCNS graph with the Profile 2 edits applied in place."""
    base = m.topology(params)
    quanta = base.quanta.copy()

    def locate(pre, post):
        lo, hi = base.indptr[pre], base.indptr[pre + 1]
        sub = base.dst[lo:hi]
        k = np.searchsorted(sub, post)
        assert k < len(sub) and sub[k] == post, (pre, post)
        return lo + k

    edits = []
    for e in emb.edges:
        k = locate(e.pre, e.post)
        edits.append((k, e.quanta, e))
        quanta[k] = e.quanta
    if policy.zero_parasitic:
        for p in emb.parasitic:
            k = locate(p.pre, p.post)
            quanta[k] = 0
    n_zeroed_out = 0
    if zero_outputs:
        circ = set(emb.neurons)
        for x in emb.neurons:
            lo, hi = base.indptr[x], base.indptr[x + 1]
            for k in range(lo, hi):
                if base.dst[k] not in circ:
                    quanta[k] = 0
                    n_zeroed_out += 1
    topo = Topology(base.n, base.src, base.dst, quanta.astype(np.int32), base.delay, base.indptr)
    return topo, n_zeroed_out


# ---- one transaction -----------------------------------------------------------------
@dataclass
class Transaction:
    a: int
    b: int
    offset_steps: int = 0  # b's DATA relative to a's
    n_steps: int = 1500  # 150 ms
    t_data: int = 50


def run_transaction(
    topo: Topology,
    params: Params,
    emb: Embedding,
    idx: dict[int, int],
    tx: Transaction,
    targets: Targets,
    background=None,  # (neuron indices in topo, rate_hz, rng)
    burst=None,  # (neuron indices in topo, step)
) -> dict:
    sim = RefSim(topo, params)
    rails = {"a1": tx.a == 1, "a0": tx.a == 0, "b1": tx.b == 1, "b0": tx.b == 0}
    for rail, on in rails.items():
        if on:
            u = idx[emb.loops[rail][0]]
            t = tx.t_data + (tx.offset_steps if rail.startswith("b") else 0)
            sim.add_events(0, [t], [u], [targets.loop])
    big = 100_000  # forces a spike on the next step (schedule.md implication)
    if background is not None:
        neurons, rate_hz, rng = background
        p = rate_hz * params.dt / 1000.0
        hits = rng.random((tx.n_steps, len(neurons))) < p
        s, j = np.nonzero(hits)
        sim.add_events(0, s, neurons[j], np.full(len(s), big))
    if burst is not None:
        neurons, step = burst
        sim.add_events(0, np.full(len(neurons), step), neurons, np.full(len(neurons), big))
    t0 = time.time()
    sim.run(tx.n_steps)
    wall = time.time() - t0
    tr = sim.trace
    roles = emb.roles()
    role_steps = {}
    for x, role in roles.items():
        role_steps[role] = tr.neuron_steps(idx[x]).tolist()
    y1 = role_steps["and_y1"]
    y0 = role_steps["or_y0"]
    comp = role_steps["completion"]
    expect_y1 = tx.a == 1 and tx.b == 1
    loop_steps = [s for r, ss in role_steps.items() if r.startswith("loop_") for s in ss]
    last_loop = max(loop_steps) if loop_steps else None
    fired_y1, fired_y0 = len(y1) > 0, len(y0) > 0
    correct = (fired_y1 == expect_y1) and (fired_y0 == (not expect_y1)) and len(comp) > 0
    data_step = tx.t_data + max(0, tx.offset_steps)
    completion_latency = (comp[0] - data_step) * params.dt if comp else None
    reset_ok = last_loop is not None and comp and last_loop < tx.n_steps - int(round(30.0 / params.dt))
    ready_latency = ((last_loop - data_step) * params.dt) if (reset_ok and last_loop is not None) else None
    circ = set(idx.values())
    ev = tr.events
    outside = ~np.isin(ev["neuron"], list(circ))
    out_neurons = ev["neuron"][outside]
    res = {
        "a": tx.a, "b": tx.b, "offset_ms": tx.offset_steps * params.dt,
        "expect_y1": expect_y1, "fired_y1": fired_y1, "fired_y0": fired_y0,
        "completion": len(comp) > 0, "correct": bool(correct),
        "completion_latency_ms": completion_latency, "reset_ok": bool(reset_ok),
        "ready_latency_ms": ready_latency,
        "spikes_per_role": {r: len(s) for r, s in role_steps.items()},
        "surround_spikes": int(outside.sum()),
        "surround_neurons_recruited": int(len(np.unique(out_neurons))),
        "wall_s": round(wall, 2),
    }
    return res, tr


def boundary_envelope(topo: Topology, tr: SpikeTrace, circuit_idx: list[int], n_steps: int) -> dict:
    """Quanta delivered into circuit neurons from non-circuit spikes, per neuron per step."""
    circ = np.array(sorted(circuit_idx))
    pos = {x: i for i, x in enumerate(circ)}
    env = np.zeros((n_steps + 200, len(circ)), dtype=np.int64)
    ev = tr.events
    mask = ~np.isin(ev["neuron"], circ)
    for s, j in zip(ev["step"][mask].tolist(), ev["neuron"][mask].tolist()):
        lo, hi = topo.indptr[j], topo.indptr[j + 1]
        d = topo.dst[lo:hi]
        hit = np.isin(d, circ)
        for k in np.nonzero(hit)[0]:
            env[s + topo.delay[lo + k], pos[d[k]]] += topo.quanta[lo + k]
    env = env[:n_steps]
    return {
        "max_pos_quanta_per_step": int(env.max()) if env.size else 0,
        "max_neg_quanta_per_step": int(env.min()) if env.size else 0,
        "total_abs_quanta": int(np.abs(env).sum()),
        "per_neuron_max_abs": [int(v) for v in np.abs(env).max(axis=0)] if env.size else [],
        "steps_with_any_input": int((env != 0).any(axis=1).sum()),
    }


# ---- the whole H0 protocol ---------------------------------------------------------------
def run_h0(out_dir: Path, params: Params = Params(), policy: Policy = Policy(), seed: int = 1,
           full_graph: bool = True, offsets_ms=(0, 1, 2, 3, 4, 5, 6, 8, 10, 12), bg_rate_hz: float = 2.0) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    m = load_mcns()
    targets = Targets.from_policy(params, policy)
    sols, sstats = find_embeddings(m, params, policy, max_solutions=50, time_limit_s=900)
    if not sols:
        report = {"status": "no embedding found", "search": sstats}
        (out_dir / "results.json").write_text(json.dumps(report, indent=1, default=str))
        return report
    emb = best_embedding(sols)
    bodies = m.bodies_of(emb.neurons)
    roles = emb.roles()
    circuit_table = [
        {"neuron_idx": int(x), "bodyId": int(m.bodies_of([x])[0]), "role": roles[x],
         "type": str(m.neurons["type"].iat[x]), "superclass": str(m.neurons["superclass"].iat[x]),
         "nt": str(m.neurons["nt"].iat[x]), "side": str(m.neurons["somaSide"].iat[x])}
        for x in emb.neurons
    ]
    edges_table = [{"pre": int(m.bodies_of([e.pre])[0]), "post": int(m.bodies_of([e.post])[0]), "count": e.count,
                    "quanta": e.quanta, "scale": round(e.quanta / (e.count * QUANTA_PER_SYNAPSE), 2), "role": e.role}
                   for e in emb.edges]
    results = {"search": sstats, "targets": targets.__dict__, "policy": policy.__dict__,
               "embedding": {"E": emb.completion, "min_margin": emb.min_margin(), "circuit": circuit_table,
                             "designed_edges": edges_table,
                             "parasitic_edges": [{"pre": int(m.bodies_of([p.pre])[0]), "post": int(m.bodies_of([p.post])[0]),
                                                  "count": p.count, "sign": int(m.sign[p.pre])} for p in emb.parasitic]},
               "conditions": {}}

    # isolated
    for zero in (True, False):
        pol = Policy(**{**policy.__dict__, "zero_parasitic": zero})
        topo, idx = isolated_topology(m, emb, params, pol)
        cond = f"isolated_parasitic_{'zeroed' if zero else 'anatomical'}"
        rows = []
        for a, b in ((0, 0), (0, 1), (1, 0), (1, 1)):
            r, _ = run_transaction(topo, params, emb, idx, Transaction(a, b), targets)
            rows.append(r)
        sweep = []
        for off in offsets_ms:
            r, _ = run_transaction(topo, params, emb, idx, Transaction(1, 1, int(round(off / params.dt))), targets)
            sweep.append({"offset_ms": off, "correct": r["correct"], "fired_y1": r["fired_y1"],
                          "completion_latency_ms": r["completion_latency_ms"], "reset_ok": r["reset_ok"]})
        results["conditions"][cond] = {"truth_table": rows, "offset_sweep_11": sweep}
        print(f"[h0] {cond}: correct {[r['correct'] for r in rows]}, reset {[r['reset_ok'] for r in rows]}")

    if full_graph:
        rng = np.random.default_rng(seed)
        idx_full = {x: x for x in emb.neurons}
        sensory = np.nonzero(m.neurons["superclass"].str.contains("sensory", na=False).to_numpy())[0]
        exc_all = np.nonzero((m.neurons["nt"] == "acetylcholine").to_numpy())[0]
        for zero_out in (False, True):
            topo, n_zeroed = full_topology(m, emb, params, policy, zero_outputs=zero_out)
            tag = "outputs_zeroed" if zero_out else "surround_live"
            for bg in ("silent", "background", "burst"):
                cond = f"full_graph_{tag}_{bg}"
                rows, envs = [], []
                for a, b in ((0, 0), (0, 1), (1, 0), (1, 1)):
                    tx = Transaction(a, b)
                    kw = {}
                    if bg == "background":
                        kw["background"] = (sensory, bg_rate_hz, np.random.default_rng(seed + a * 2 + b))
                    if bg == "burst":
                        kw["burst"] = (rng.choice(exc_all, 1000, replace=False), tx.t_data + 50)
                    r, tr = run_transaction(topo, params, emb, idx_full, tx, targets, **kw)
                    env = boundary_envelope(topo, tr, list(idx_full.values()), tx.n_steps)
                    rows.append(r); envs.append(env)
                    # who got recruited
                    ev = tr.events
                    out = ev["neuron"][~np.isin(ev["neuron"], emb.neurons)]
                    if len(out):
                        ids, cnts = np.unique(out, return_counts=True)
                        top = np.argsort(-cnts)[:8]
                        r["top_recruited"] = [{"type": str(m.neurons["type"].iat[int(ids[k])]),
                                               "superclass": str(m.neurons["superclass"].iat[int(ids[k])]),
                                               "spikes": int(cnts[k])} for k in top]
                results["conditions"][cond] = {"truth_table": rows, "envelope": envs, "n_output_edges_zeroed": n_zeroed}
                print(f"[h0] {cond}: correct {[r['correct'] for r in rows]}, reset {[r['reset_ok'] for r in rows]}, "
                      f"surround spikes {[r['surround_spikes'] for r in rows]}")

        (out_dir / "results.json").write_text(json.dumps(results, indent=1, default=str))
        # manifest for the full-graph, surround-live run
        # anatomical value of a weight edit = signed anatomical quanta (sign from presynaptic NT)
        edits = [ParameterEdit("weight", int(bodies[emb.neurons.index(e.pre)]), int(bodies[emb.neurons.index(e.post)]),
                               int(m.sign[e.pre]) * e.count * QUANTA_PER_SYNAPSE, e.quanta,
                               f"0 <= |q| <= {policy.k_max} * count * {QUANTA_PER_SYNAPSE}, sign preserved", e.role)
                 for e in emb.edges]
        edits += [ParameterEdit("weight", int(bodies[emb.neurons.index(p.pre)]), int(bodies[emb.neurons.index(p.post)]),
                                int(m.sign[p.pre]) * p.count * QUANTA_PER_SYNAPSE, 0, "documented zero weight",
                                "parasitic edge among circuit neurons")
                  for p in emb.parasitic]
        man = Manifest(
            connectome={"id": "MCNS", "version": "v1.0", **{k: v for k, v in m.summary().items() if k.endswith("_rule")}},
            profile=policy.profile, params=params_dict(params), circuit_bodies=[int(b) for b in bodies],
            graphs={"original": m.graph_summary(), "retained": m.graph_summary(),
                    "active_nonzero": {**m.graph_summary(), "n_edges": m.graph_summary()["n_edges"] - len(emb.parasitic),
                                       "note": "original minus zeroed parasitic edges; designed edges rescaled"}},
            silencing=[], parameter_edits=edits, structural_edits=[],
            ports={"inputs": [int(bodies[emb.neurons.index(emb.loops[r][0])]) for r in ("a1", "a0", "b1", "b0")],
                   "outputs": [int(bodies[emb.neurons.index(x)]) for x in (emb.gate_and, emb.gate_or, emb.completion)]},
            execution_mode="full-graph",
            isolation_classification="robust within a measured boundary-input envelope (see results.json)",
        )
        man.save(out_dir / "manifest_full_graph_surround_live.json")

    (out_dir / "results.json").write_text(json.dumps(results, indent=1, default=str))
    return results


if __name__ == "__main__":
    import sys
    full = "--isolated-only" not in sys.argv
    run_h0(Path("data/h0"), full_graph=full)
