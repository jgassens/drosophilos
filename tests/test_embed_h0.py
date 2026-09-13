"""H0 pipeline on a synthetic connectome: the search must find the planted motif and the
embedded circuit must compute a dual-rail AND with completion and reset, isolated."""

import numpy as np
import pandas as pd
import pytest

from drosophilos.connectome.embed_h0 import Policy, Targets, best_embedding, find_embeddings
from drosophilos.connectome.h0_run import Transaction, isolated_topology, run_transaction
from drosophilos.connectome.mcns import MCNS
from drosophilos.sim.model import Params


def synthetic_mcns(seed=0, n_distractors=40):
    rng = np.random.default_rng(seed)
    names = ["a1u", "a1v", "a0u", "a0v", "b1u", "b1v", "b0u", "b0v", "C", "D", "E", "F1", "F2", "F3", "F4"]
    nt = ["acetylcholine"] * 11 + ["gaba"] * 4
    idx = {nm: i for i, nm in enumerate(names)}
    edges = []

    def add(a, b, c):
        edges.append((idx[a], idx[b], c))

    for r in ("a1", "a0", "b1", "b0"):
        add(r + "u", r + "v", 70)
        add(r + "v", r + "u", 66)
    add("a1u", "C", 30); add("b1u", "C", 33)
    add("a0u", "D", 60); add("b0u", "D", 58)
    add("C", "E", 61); add("D", "E", 64)
    for k, r in enumerate(("a1", "b1", "a0", "b0"), start=1):
        add("E", f"F{k}", 62)
        add(f"F{k}", r + "u", 90)
        add(f"F{k}", r + "v", 88)
    n0 = len(names)
    n = n0 + n_distractors
    nts = nt + list(rng.choice(["acetylcholine", "gaba", "glutamate"], n_distractors))
    # weak random background edges everywhere (below every requirement)
    for _ in range(300):
        a, b = rng.integers(0, n, 2)
        if a != b:
            edges.append((int(a), int(b), int(rng.integers(1, 6))))
    pre = np.array([e[0] for e in edges], np.int32)
    post = np.array([e[1] for e in edges], np.int32)
    cnt = np.array([e[2] for e in edges], np.int32)
    # aggregate duplicates
    df = pd.DataFrame({"pre": pre, "post": post, "count": cnt}).groupby(["pre", "post"], as_index=False)["count"].sum()
    df = df.sort_values(["pre", "post"])
    neurons = pd.DataFrame({
        "bodyId": 1000 + np.arange(n), "type": [f"T{i}" for i in range(n)], "superclass": "cb_intrinsic",
        "class": None, "somaSide": "R", "nt": nts, "nt_conf": 0.9,
    })
    neurons["sign"] = np.where(neurons["nt"].isin(["gaba", "glutamate", "histamine"]), -1, 1).astype(np.int8)
    return MCNS(neurons, df["pre"].to_numpy(np.int32), df["post"].to_numpy(np.int32), df["count"].to_numpy(np.int32), min_syn=1)


def test_targets_are_rate_mode():
    tg = Targets.from_policy(Params(), Policy())
    assert 40 <= tg.period_steps <= 60
    assert tg.and_in < tg.needed_rate < 2 * tg.and_in  # one input below, two above
    assert tg.or_in >= tg.needed_rate
    assert tg.reset < 0


def test_search_finds_planted_motif_and_circuit_computes_and():
    m = synthetic_mcns()
    params, policy = Params(), Policy()
    sols, stats = find_embeddings(m, params, policy, max_solutions=5, verbose=False)
    assert sols, stats
    emb = best_embedding(sols)
    assert emb.gate_and == 8 and emb.gate_or == 9 and emb.completion == 10
    tg = Targets.from_policy(params, policy)
    topo, idx = isolated_topology(m, emb, params, policy)
    results = {}
    for a, b in ((0, 0), (0, 1), (1, 0), (1, 1)):
        r, _ = run_transaction(topo, params, emb, idx, Transaction(a, b), tg)
        results[(a, b)] = r
    for (a, b), r in results.items():
        assert r["correct"], (a, b, r)
        assert r["reset_ok"], (a, b, r)
        assert r["completion_latency_ms"] < 60, (a, b, r)


def test_timestep_refinement_leaves_transactions_unchanged():
    """Stage 0 exit criterion: dt 0.1 ms -> 0.02 ms must not change decoded transactions.
    Uses the H0-style circuit (latches, rate-mode AND, OR, completion, reset)."""
    m = synthetic_mcns()
    policy = Policy()
    coarse, fine = Params(), Params().with_dt(0.02)
    outcomes = {}
    for params in (coarse, fine):
        sols, _ = find_embeddings(m, params, policy, max_solutions=1, verbose=False)
        assert sols
        emb = best_embedding(sols)
        tg = Targets.from_policy(params, policy)
        topo, idx = isolated_topology(m, emb, params, policy)
        steps_per_ms = int(round(1.0 / params.dt))
        row = []
        for a, b in ((0, 0), (0, 1), (1, 0), (1, 1)):
            tx = Transaction(a, b, n_steps=150 * steps_per_ms, t_data=5 * steps_per_ms)
            r, _ = run_transaction(topo, params, emb, idx, tx, tg)
            row.append((r["fired_y1"], r["fired_y0"], r["completion"], r["reset_ok"]))
        outcomes[params.dt] = row
    assert outcomes[0.1] == outcomes[0.02], outcomes
    assert all(r[2] for r in outcomes[0.1])
