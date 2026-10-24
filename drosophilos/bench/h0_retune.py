"""Recertify isolated anatomical H0 after retuning under the released Shiu rules.

Only graph searches read the full MCNS graph; every simulation is a small circuit.
The source data directory is read-only. See docs/stage_h/h0_under_shiu.md.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

from ..connectome.embed_h0 import DesignedEdge, Embedding, Policy, Targets, best_embedding, find_embeddings
from ..connectome.h0_run import Transaction, isolated_topology, run_transaction
from ..sim.model import ModelRules, Params, QUANTA_PER_SYNAPSE, Topology
from ..sim.ref64 import RefSim
from .model_compare import load_mcns

SHIU = ModelRules(True, True, "v", True)
BODY_IDS = [10643, 10864, 10881, 11127, 11429, 14160, 14210, 26764, 32461,
            37111, 512079, 520816, 523040, 523983, 555296]
PUBLISHED = Path(__file__).resolve().parents[2] / "docs/h0/results.json"
# A smaller OR safety factor fits H0's 24-synapse OR edge at the slower M5 rate.
# The AND fraction and all other policy fields, especially k_max, stay fixed.
RETUNED_POLICY = replace(Policy(), or_margin=1.5)
DRIVE_QUANTA = 410  # 7.046875 mV into V: one ignition spike, no g reservoir
INPUTS = ((0, 0), (0, 1), (1, 0), (1, 1))


def published_embedding(m, path: Path = PUBLISHED) -> Embedding:
    """Resolve the published roles by bodyId, checking counts against the real graph."""
    record = json.loads(path.read_text())["embedding"]
    bodies = [row["bodyId"] for row in record["circuit"]]
    if sorted(bodies) != BODY_IDS:
        raise ValueError("published circuit differs from the specified 15 H0 bodyIds")
    lookup = dict(zip(bodies, m.index_of(bodies).tolist()))
    roles = {row["role"]: lookup[row["bodyId"]] for row in record["circuit"]}
    ns = list(lookup.values())
    keep = np.isin(m.pre, ns) & np.isin(m.post, ns)
    counts = {(int(u), int(v)): int(c) for u, v, c in
              zip(m.pre[keep], m.post[keep], m.count[keep])}
    edges = []
    for row in record["designed_edges"]:
        u, v = lookup[row["pre"]], lookup[row["post"]]
        if counts.get((u, v)) != row["count"]:
            raise ValueError(f"anatomical count mismatch: {row}")
        edges.append(DesignedEdge(u, v, row["count"], row["quanta"], row["role"]))
    loops = {r: (roles[f"loop_{r}_driver"], roles[f"loop_{r}_partner"])
             for r in ("a1", "b1", "a0", "b0")}
    resets = {r: (sorted({e.post for e in edges if e.role == f"reset_drive_{r}"}),
                  sorted({e.post for e in edges if e.role == f"reset_{r}"})) for r in loops}
    designed = {(e.pre, e.post) for e in edges}
    parasitic = [DesignedEdge(u, v, c, int(m.sign[u]) * c * QUANTA_PER_SYNAPSE, "parasitic")
                 for (u, v), c in sorted(counts.items()) if (u, v) not in designed and u != v]
    return Embedding(loops, roles["and_y1"], roles["or_y0"], roles["completion"],
                     resets, edges, parasitic)


def retune_embedding(emb: Embedding, targets: Targets) -> Embedding:
    """Reweight the same physical edges. Shared E->F drives occur only once.

    The historical emitter lists a shared reset-drive edge once per rail it serves.
    Those entries must not sum to an effective anatomical weight above k_max.
    Reset demand is apportioned across the original inhibitory edges by count.
    """
    physical = {}
    for e in emb.edges:
        key = e.pre, e.post
        if key in physical:
            if not (e.role.startswith("reset_drive_") and
                    physical[key].role.startswith("reset_drive_")):
                raise ValueError(f"unexpected repeated designed edge: {key}")
            continue
        if e.role.startswith("loop_"):
            q = targets.loop
        elif e.role.startswith("and_"):
            q = targets.and_in
        elif e.role.startswith("or_"):
            q = targets.or_in
        elif e.role.startswith(("completion_", "reset_drive_")):
            q = targets.completion
        elif e.role.startswith("reset_"):
            q = 0  # apportioned below
        else:
            raise ValueError(f"unknown H0 edge role: {e.role}")
        physical[key] = replace(e, quanta=q)
    reset_posts = sorted({e.post for e in physical.values()
                          if e.role.startswith("reset_") and not e.role.startswith("reset_drive_")})
    for post in reset_posts:
        es = [e for e in physical.values() if e.post == post and e.quanta == 0]
        total_count = sum(e.count for e in es)
        remaining = abs(targets.reset)
        for i, e in enumerate(es):
            take = remaining if i == len(es) - 1 else abs(targets.reset) * e.count // total_count
            physical[e.pre, e.post] = replace(e, quanta=-take)
            remaining -= take
    return replace(emb, edges=list(physical.values()))


def audit_embedding(m, emb: Embedding, policy: Policy) -> dict:
    """Check the bound per physical anatomical edge, including parallel entries."""
    totals, counts = {}, {}
    for e in emb.edges:
        key = e.pre, e.post
        totals[key] = totals.get(key, 0) + e.quanta
        counts[key] = e.count
    ns = emb.neurons
    keep = np.isin(m.pre, ns) & np.isin(m.post, ns)
    anatomy = {(int(u), int(v)): int(c) for u, v, c in
               zip(m.pre[keep], m.post[keep], m.count[keep])}
    rows = []
    for (u, v), q in totals.items():
        count = anatomy.get((u, v), 0)
        scale = abs(q) / (count * QUANTA_PER_SYNAPSE) if count else None
        rows.append({"pre_body_id": int(m.bodies_of([u])[0]),
                     "post_body_id": int(m.bodies_of([v])[0]), "count": count,
                     "quanta": q, "required_multiplier": scale,
                     "anatomical_edge": count > 0 and count == counts[u, v],
                     "sign_preserved": bool(q == 0 or np.sign(q) == m.sign[u]),
                     "within_bound": scale is not None and scale <= policy.k_max})
    return {"valid": all(r["anatomical_edge"] and r["sign_preserved"] and r["within_bound"]
                         for r in rows),
            "k_max": policy.k_max,
            "required_multiplier": max((r["required_multiplier"] or 0 for r in rows), default=0),
            "physical_edges": len(rows), "listed_edges": len(emb.edges), "edges": rows}


def arrival_stats(topo, trace, edges, idx, params, rules, stimulated, n_steps) -> dict:
    """Count arrivals rejected by freeze-g, including postsynaptic spike steps."""
    total, discarded = 0, 0
    for e in edges:
        pre, post = idx[e.pre], idx[e.post]
        arrivals = trace.neuron_steps(pre) + params.default_delay_steps
        arrivals = arrivals[arrivals < n_steps]
        spikes = trace.neuron_steps(post)
        last = np.searchsorted(spikes, arrivals, side="right") - 1
        valid = last >= 0
        ages = np.full(len(arrivals), params.n_ref + 1, dtype=int)
        ages[valid] = arrivals[valid] - spikes[last[valid]]
        blocked = (ages == 0) | ((ages < params.n_ref) & (post not in stimulated))
        total += len(arrivals)
        discarded += int(blocked.sum()) if rules.freeze_g_in_refractory else 0
    return {"total": total, "discarded": discarded,
            "discarded_fraction": discarded / total if total else 0.0}


def transactions(m, emb, targets, policy, rules, *, drive_quanta=None, n_steps=1500) -> list[dict]:
    topo, idx = isolated_topology(m, emb, Params(), policy)
    rows = []
    for a, b in INPUTS:
        tx = Transaction(a, b, n_steps=n_steps, drive_quanta=drive_quanta)
        row, trace = run_transaction(topo, Params(), emb, idx, tx, targets, rules=rules)
        row.pop("wall_s")
        latency = row["completion_latency_ms"]
        row["passes_existing_checks"] = bool(row["correct"] and row["reset_ok"] and
                                               latency is not None and latency < 60)
        stimulated = {idx[emb.loops[f"{name}{value}"][0]] for name, value in (("a", a), ("b", b))}
        row["loop_arrivals"] = arrival_stats(topo, trace,
            [e for e in emb.edges if e.role.startswith("loop_")], idx, Params(), rules, stimulated, n_steps)
        row["reset_arrivals"] = arrival_stats(topo, trace,
            [e for e in emb.edges if e.role.startswith("reset_") and not e.role.startswith("reset_drive_")],
            idx, Params(), rules, stimulated, n_steps)
        row["spike_steps_by_role"] = {r: trace.neuron_steps(idx[x]).tolist() for x, r in emb.roles().items()}
        row["trace_sha256"] = hashlib.sha256(trace.events.tobytes()).hexdigest()
        rows.append(row)
    return rows


def dynamics_probe(targets: Targets, *, rules=SHIU, n_steps=10000) -> dict:
    """Latch persistence and one-versus-two gate inputs, without completion/reset."""
    p = Params()
    result = {}
    for active_inputs in (1, 2):
        topo = Topology.from_edges(6, [0, 1, 2, 3, 0, 2, 0, 2], [1, 0, 3, 2, 4, 4, 5, 5],
                                  [targets.loop] * 4 + [targets.and_in] * 2 + [targets.or_in] * 2,
                                  [p.default_delay_steps] * 8)
        sim = RefSim(topo, p, rules=rules, record=[(0, 4), (0, 5)])
        sim.add_events(0, [50] * active_inputs, [0, 2][:active_inputs], [DRIVE_QUANTA] * active_inputs)
        sim.run(n_steps)
        driver = sim.trace.neuron_steps(0)
        gate = sim.trace.neuron_steps(4)
        gate_or = sim.trace.neuron_steps(5)
        recorded_v = np.asarray(sim.rec_V)
        result[str(active_inputs)] = {
            "driver_spikes": len(driver), "last_driver_spike_ms": float(driver[-1] * p.dt) if len(driver) else None,
            "driver_final_50ms_spikes": int((driver >= n_steps - 500).sum()),
            "steady_period_steps": int(driver[-1] - driver[-2]) if len(driver) >= 2 else None,
            "gate_spikes": len(gate),
            "first_gate_latency_ms": float((gate[0] - 50) * p.dt) if len(gate) else None,
            "gate_max_recorded_v_mv": float(recorded_v[:, 0].max()),
            "or_gate_spikes": len(gate_or),
            "or_first_gate_latency_ms": float((gate_or[0] - 50) * p.dt) if len(gate_or) else None,
            "or_gate_max_recorded_v_mv": float(recorded_v[:, 1].max()),
        }
    return {"duration_ms": n_steps * p.dt, "threshold_mv": p.V_th, "inputs": result}


def all_pass(rows) -> bool:
    return all(r["passes_existing_checks"] for r in rows)


def run(m, original: Embedding, *, source: str, max_solutions=50, time_limit_s=300,
        test_best=10, probe_steps=10000) -> dict:
    p, policy = Params(), RETUNED_POLICY
    old = Targets.from_policy(p, Policy())
    derived = Targets.from_policy(p, Policy(), rules=SHIU)
    tuned = Targets.from_policy(p, policy, rules=SHIU)
    emb = retune_embedding(original, tuned)
    audit = audit_embedding(m, emb, policy)
    record = {
        "schema_version": 1, "source": source, "graph": m.graph_summary(),
        "params": asdict(p), "model_rules": asdict(SHIU),
        "original_policy": asdict(Policy()), "retuned_policy": asdict(policy),
        "original_targets": asdict(old), "m5_targets_original_margins": asdict(derived),
        "retuned_targets": asdict(tuned),
        "protocol": {"drive_quanta": DRIVE_QUANTA, "drive_mv": DRIVE_QUANTA * p.w_unit,
                     "t_data_ms": 5, "b_offset_ms": 0, "observation_ms": 150,
                     "completion_limit_ms_exclusive": 60, "reset_quiet_tail_ms": 30,
                     "external_targets": "selected operand rail drivers only; no external reset"},
        "tier_a": {"body_ids": m.bodies_of(emb.neurons).tolist(), "policy_audit": audit,
                   "original_policy_audit": audit_embedding(m, original, Policy()),
                   "m5_original_margins_audit": audit_embedding(m, retune_embedding(original, derived), policy),
                   "roles": {str(int(m.bodies_of([x])[0])): r for x, r in emb.roles().items()},
                   "parasitic_edges_zeroed": len(emb.parasitic),
                   "M0_original": transactions(m, original, old, Policy(), ModelRules()),
                   "M5_stale": transactions(m, original, old, Policy(), SHIU)},
        "diagnosis": {"M5_stale": dynamics_probe(old, n_steps=probe_steps),
                      "M5_retuned": dynamics_probe(tuned, n_steps=probe_steps)},
    }
    tier = record["tier_a"]
    tier["transactions"] = transactions(m, emb, tuned, policy, SHIU, drive_quanta=DRIVE_QUANTA) if audit["valid"] else []
    tier["pass"] = bool(audit["valid"] and all_pass(tier["transactions"]))
    if tier["pass"]:
        record["verdict"] = "Tier A PASS: the same anatomical H0 computes AND under M5 after bounded retuning."
        record["tier_b"] = {"status": "not_run", "reason": "Tier A passed"}
    else:
        solutions, stats = find_embeddings(m, p, policy, max_solutions=max_solutions,
                                           time_limit_s=time_limit_s, verbose=False, rules=SHIU)
        candidates = []
        remaining = list(solutions)
        for _ in range(min(test_best, len(remaining))):
            candidate = best_embedding(remaining)
            remaining = [e for e in remaining if e is not candidate]
            candidate = retune_embedding(candidate, tuned)
            check = audit_embedding(m, candidate, policy)
            rows = transactions(m, candidate, tuned, policy, SHIU, drive_quanta=DRIVE_QUANTA) if check["valid"] else []
            candidates.append({"body_ids": m.bodies_of(candidate.neurons).tolist(),
                               "policy_audit": check, "transactions": rows,
                               "pass": bool(check["valid"] and all_pass(rows))})
        passed = any(c["pass"] for c in candidates)
        record["tier_b"] = {"status": "pass" if passed else "fail", "search": stats,
                            "caps": {"max_solutions": max_solutions, "time_limit_s": time_limit_s,
                                     "test_best": test_best, "assignments_per_gate_pair": 400},
                            "candidates": candidates}
        record["verdict"] = ("Tier A FAIL; Tier B " + ("PASS: another embedding computes AND under M5."
                              if passed else "FAIL: no tested embedding computes AND within the declared search."))
    return record


def parse(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mcns-dir", type=Path)
    parser.add_argument("--synthetic", action="store_true", help="development only: tests/test_embed_h0.py fixture")
    parser.add_argument("--out", type=Path, default=Path("docs/stage_h/h0_under_shiu.json"))
    parser.add_argument("--max-solutions", type=int, default=50)
    parser.add_argument("--time-limit-s", type=float, default=300)
    parser.add_argument("--test-best", type=int, default=10)
    args = parser.parse_args(argv)
    if not args.synthetic and args.mcns_dir is None:
        parser.error("provide --mcns-dir or --synthetic")
    if args.synthetic and args.mcns_dir is not None:
        parser.error("--synthetic and --mcns-dir are mutually exclusive")
    if min(args.max_solutions, args.time_limit_s, args.test_best) <= 0:
        parser.error("search caps must be positive")
    return args


def main(argv=None):
    args = parse(argv)
    if args.synthetic:
        # Match the existing development fixture without copying it into production.
        import importlib.util
        path = Path(__file__).resolve().parents[2] / "tests/test_embed_h0.py"
        spec = importlib.util.spec_from_file_location("h0_synthetic_fixture", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        m = module.synthetic_mcns()
        solutions, _ = find_embeddings(m, max_solutions=5, verbose=False)
        original = best_embedding(solutions)
        source = "synthetic_mcns development fixture; not anatomical evidence"
    else:
        try:
            m = load_mcns(data_dir=args.mcns_dir)
        except (OSError, PermissionError) as exc:
            record = {"verdict": "NOT RUN: real MCNS data unavailable", "error": str(exc),
                      "source": str(args.mcns_dir), "tier_a": {"status": "not_run"},
                      "tier_b": {"status": "not_run"}}
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(json.dumps(record, indent=2) + "\n")
            print(record["verdict"])
            return record
        original = published_embedding(m)
        source = str(args.mcns_dir)
    record = run(m, original, source=source, max_solutions=args.max_solutions,
                 time_limit_s=args.time_limit_s, test_best=args.test_best)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
    print(record["verdict"])
    return record


if __name__ == "__main__":
    main()
