"""Motif placement of a compiled netlist onto connectome wiring (Profile 2).

Fast: a single latch with its edge relay and the relay's inhibitor, built with the protocol
helpers, placed completely onto a synthetic connectome that contains one planted instance
among random weak wiring. The search must find the planted neurons and carry all six edges,
with the mixed-sign, hub and chain bookkeeping exercised on the way.

Flip-flops (protocol.flipflop): two flip-flops joined by one edge relay, placed on a synthetic
connectome with two planted driven mutual inhibitory pairs among undriven and unreachable
decoys. The pairs are `ffpair` motifs, placed as units on the planted pairs; the relay's edges
FROM a flip-flop member (`u -> edge`, `u -> edge_inh`, excitatory) are of the wrong sign for a
GABA host under Dale's law on any connectome and must be reported as such, not carried.

Flip-flops with a proxy (add_flipflop(proxy=True)): the pair plus p, a biased excitatory neuron
v inhibits, is the `fftriple` motif, placed as a unit on a planted (driven pair, inhibited
excitatory neuron); p's outputs are ordinary excitatory edges, so the relay read off p and the
set chain it fires into the second flip-flop are placed completely -- the edges the pair-only
motif could never carry.

MCNS (skipped without the data): the chain toy of docs/h1_placement.md "Placing flip-flops", at
a small size, with and without proxies.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from drosophilos.connectome.embed_h0 import Policy
from drosophilos.connectome.embed_netlist import Placement, _Anatomy, _Design, audit, place_netlist, place_netlist_greedy
from drosophilos.connectome.mcns import MCNS
from drosophilos.connectome.mcns_download import DEFAULT_DIR, path_of
from drosophilos.lib.netlist import Drive, Netlist
from drosophilos.protocol.flipflop import BIAS_213HZ_MV, add_flipflop, add_set_chain
from drosophilos.protocol.latch import add_edge_relay, add_latch
from drosophilos.sim.model import Params


def latch_relay_netlist():
    """src -> [edge relay: edge, edge_inh] -> latch (u, v): 5 neurons, 6 designed edges."""
    params = Params()
    drive = Drive.from_params(params)
    net = Netlist(params)
    src = net.neuron("src")
    latch = add_latch(net, drive, "L")
    relay = add_edge_relay(net, drive, "in", src)
    net.synapse(relay, latch.u, drive.ignite)
    return net, drive, {"src": src, "u": latch.u, "v": latch.v, "edge": relay, "edge_inh": relay + 1}


def synthetic_mcns(policy: Policy, drive: Drive, seed: int = 0, n_distractors: int = 80, n_weak: int = 600,
                   twin_latch: bool = False, decoy_inputs: int = 50) -> tuple[MCNS, dict]:
    """A planted instance (S, E, I, U, V) with counts just above the requirements, plus random
    neurons with weak edges (all below every requirement) and a few strong decoys that do not
    close the motif (a mutual pair with no relay, a relay with no inhibitor).
    `twin_latch`: a second latch pair (U2, V2), edge for edge as good as (U, V) -- the same
    counts, reached from E by the same count -- but U2 receives `decoy_inputs` extra weak edges
    from distinct distractor neurons, so it is the noisier of two equally good hosts."""
    rng = np.random.default_rng(seed)
    req = policy.req_count
    names = ["S", "E", "I", "U", "V", "P", "Q", "S2", "E2"] + (["U2", "V2"] if twin_latch else [])
    nts = ["acetylcholine"] * len(names)
    nts[2] = "gaba"
    idx = {nm: i for i, nm in enumerate(names)}
    edges = [
        (idx["S"], idx["E"], req(drive.relay_in) + 2), (idx["S"], idx["I"], req(drive.pulse) + 1),
        (idx["I"], idx["E"], req(int(round(2.2 * drive.loop))) + 3), (idx["E"], idx["U"], req(drive.ignite) + 1),
        (idx["U"], idx["V"], req(drive.loop) + 4), (idx["V"], idx["U"], req(drive.loop)),
        (idx["P"], idx["Q"], req(drive.loop) + 10), (idx["Q"], idx["P"], req(drive.loop) + 10),  # decoy pair, no relay
        (idx["S2"], idx["E2"], req(drive.relay_in) + 5), (idx["E2"], idx["P"], req(drive.ignite) + 5),  # decoy relay, no inhibitor
    ]
    n0 = len(names)
    n = n0 + n_distractors
    nts += list(rng.choice(["acetylcholine", "gaba", "glutamate"], n_distractors))
    if twin_latch:
        edges += [(idx["E"], idx["U2"], req(drive.ignite) + 1),
                  (idx["U2"], idx["V2"], req(drive.loop) + 4), (idx["V2"], idx["U2"], req(drive.loop))]
        for a in rng.choice(np.arange(n0, n), decoy_inputs, replace=False):  # 50 distinct distractors, 2 synapses each
            edges.append((int(a), idx["U2"], 2))
    for _ in range(n_weak):
        a, b = rng.integers(0, n, 2)
        if a != b:
            edges.append((int(a), int(b), int(rng.integers(1, 4))))
    df = pd.DataFrame(edges, columns=["pre", "post", "count"]).groupby(["pre", "post"], as_index=False)["count"].sum()
    df = df.sort_values(["pre", "post"])
    neurons = pd.DataFrame({"bodyId": 1000 + np.arange(n), "type": [f"T{i}" for i in range(n)], "superclass": "cb",
                            "class": None, "somaSide": "R", "nt": nts, "nt_conf": 0.9})
    neurons["sign"] = np.where(neurons["nt"].isin(["gaba", "glutamate", "histamine"]), -1, 1).astype(np.int8)
    m = MCNS(neurons, df["pre"].to_numpy(np.int32), df["post"].to_numpy(np.int32), df["count"].to_numpy(np.int32), min_syn=1)
    return m, idx


def synthetic_autapses(too_weak: bool = False):
    """The planted relay's inhibitory host, an excitatory host and a host without feedback."""
    net, drive, ids = latch_relay_netlist()
    policy = Policy(k_max=4)
    m, idx = synthetic_mcns(policy, drive)
    q = -round(.2 * drive.loop)
    for name in ("edge_inh", "edge", "v"):
        net.synapse(ids[name], ids[name], q, 0)
    cnt = policy.req_count(q)
    m = MCNS(m.neurons, np.append(m.pre, [idx["I"], idx["E"]]).astype(np.int32),
             np.append(m.post, [idx["I"], idx["E"]]).astype(np.int32),
             np.append(m.count, [cnt - 1 if too_weak else cnt, cnt]).astype(np.int32), min_syn=1)
    real = np.array([idx["S"], idx["U"], idx["V"], idx["E"], idx["I"]])
    return net, drive, ids, policy, m, idx, real


@pytest.mark.parametrize("too_weak", [False, True])
def test_autapses_use_only_their_host_self_edge(too_weak):
    net, drive, ids, policy, m, idx, real = synthetic_autapses(too_weak)
    design = _Design(net, policy, hub_deg=20, hard_hubs=range(net.n))
    assert not design.hard[design.self_edge].any()
    assert design.sign[ids["edge"]] == design.sign[ids["v"]] == 1
    pl = audit(net, design, _Anatomy(m, policy), real, 0, {}, "planted", policy)
    assert pl.carried == 6 + (not too_weak)
    reasons = {(s, d): reason for s, d, _, reason in pl.missing}
    assert reasons[ids["edge"], ids["edge"]] == "autapse (wrong sign)"
    assert reasons[ids["v"], ids["v"]] == "autapse (no host self-edge)"
    if too_weak:
        assert reasons[ids["edge_inh"], ids["edge_inh"]] == "autapse (too weak)"
    else:
        assert (ids["edge_inh"], ids["edge_inh"]) not in reasons
    for reason in reasons.values():
        assert pl.missing_by_class[reason] == [1, 1]
    assert pl.summary(net)["missing_by_class"] == pl.missing_by_class


def test_autapses_preserve_motifs_and_never_constrain_host_choice():
    net, drive, ids, policy, m, idx, real = synthetic_autapses(too_weak=True)
    baseline, _, _ = latch_relay_netlist()
    old = _Design(baseline, policy, hub_deg=20)
    new = _Design(net, policy, hub_deg=20)
    assert [(mo.kind, mo.nodes, mo.edges) for mo in old.motifs] == [(mo.kind, mo.nodes, mo.edges) for mo in new.motifs]
    a = place_netlist(baseline, m, policy, restarts=1, seed=0, time_limit_s=20)
    b = place_netlist(net, m, policy, restarts=1, seed=0, time_limit_s=20)
    assert a.mapping == b.mapping and not b.unplaced
    assert b.carried == a.carried == 6
    assert len(b.missing) == 3


def test_carriable_autapse_on_decoy_cannot_override_hard_edges():
    net, drive, ids = latch_relay_netlist()
    policy = Policy(k_max=4)
    m, idx = synthetic_mcns(policy, drive)
    # P has a carriable excitatory autapse, but only E fits the relay's hard edges.
    q = round(.2 * drive.loop)
    net.synapse(ids["edge"], ids["edge"], q, 0)
    m = MCNS(m.neurons, np.append(m.pre, idx["P"]).astype(np.int32),
             np.append(m.post, idx["P"]).astype(np.int32),
             np.append(m.count, policy.req_count(q)).astype(np.int32), min_syn=1)
    pl = place_netlist(net, m, policy, restarts=1, time_limit_s=20)
    assert pl.mapping[ids["edge"]] == idx["E"]
    assert pl.carried == 6 and not pl.unplaced
    assert pl.missing == [(ids["edge"], ids["edge"], q, "autapse (no host self-edge)")]


@pytest.mark.parametrize("isolation_weight", [0.0, 1.0])
def test_allowed_hosts_none_and_all_true_preserve_placement(isolation_weight):
    net, drive, _ = latch_relay_netlist()
    policy = Policy(k_max=4)
    m, _ = synthetic_mcns(policy, drive, twin_latch=True)
    kwargs = dict(policy=policy, restarts=2, time_limit_s=20, seed=7, isolation_weight=isolation_weight)
    baseline = place_netlist(net, m, **kwargs)
    unrestricted = place_netlist(net, m, allowed_hosts=None, **kwargs)
    all_hosts = place_netlist(net, m, allowed_hosts=np.ones(m.n, bool), **kwargs)
    for pl in (unrestricted, all_hosts):
        assert pl.mapping == baseline.mapping
        assert pl.carried == baseline.carried and pl.missing == baseline.missing
        assert pl.exposure == baseline.exposure and pl.objective == baseline.objective
        assert pl.motifs_search == baseline.motifs_search and pl.strategy == baseline.strategy


@pytest.mark.parametrize("indices", [False, True])
@pytest.mark.parametrize("exclude_inhibitor", [False, True])
@pytest.mark.parametrize("hub_deg", [1, 20])
def test_allowed_hosts_restrict_all_search_paths_and_keep_whole_brain_exposure(indices, exclude_inhibitor, hub_deg):
    net, drive, _ = latch_relay_netlist()
    policy = Policy(k_max=4)
    m, idx = synthetic_mcns(policy, drive)
    mask = np.zeros(m.n, bool)
    mask[[idx[k] for k in ("S", "E", "U", "V")]] = True
    mask[idx["I"]] = not exclude_inhibitor
    pl = place_netlist(net, m, policy, allowed_hosts=np.flatnonzero(mask) if indices else mask,
                       region="planted", restarts=2, time_limit_s=20, isolation_weight=1, hub_deg=hub_deg)
    assert pl.mapping and all(mask[r] for r in pl.mapping.values())
    assert pl.summary(net)["allowed_hosts"] == int(mask.sum())
    assert pl.summary(net)["region"] == "planted"
    hosts = set(pl.mapping.values())
    incoming = [int(c) for s, d, c in zip(m.pre, m.post, m.count) if d in hosts and s not in hosts]
    outgoing = [int(c) for s, d, c in zip(m.pre, m.post, m.count) if s in hosts and d not in hosts]
    assert incoming and outgoing  # real exposure exists beyond the permitted pool
    assert pl.exposure["in_total"] == sum(incoming)
    assert pl.exposure["out_total"] == sum(outgoing)
    assert pl.exposure["in_edges"] == len(incoming) and pl.exposure["out_edges"] == len(outgoing)


@pytest.mark.parametrize("pool", [np.array([], dtype=int), np.zeros(89, bool)])
def test_empty_allowed_hosts_leaves_every_neuron_unplaced(pool):
    net, drive, _ = latch_relay_netlist()
    m, _ = synthetic_mcns(Policy(), drive)
    pl = place_netlist(net, m, allowed_hosts=pool, restarts=1, time_limit_s=20)
    assert pl.mapping == {} and pl.unplaced == list(range(net.n))


@pytest.mark.parametrize("pool", [np.array([True]), np.array([[0]]), np.array([-1]), np.array([10000]), np.array([1.5])])
def test_invalid_allowed_hosts(pool):
    net, drive, _ = latch_relay_netlist()
    m, _ = synthetic_mcns(Policy(), drive)
    with pytest.raises(ValueError, match="allowed_hosts"):
        place_netlist(net, m, allowed_hosts=pool)


def test_self_only_neuron_scores_hosts_without_a_hard_sign_constraint():
    _, drive, _, policy, m, idx, _ = synthetic_autapses()
    net = Netlist(Params())
    node = net.neuron("feedback")
    net.synapse(node, node, -round(.2 * drive.loop), 0)
    design = _Design(net, policy, hub_deg=20)
    assert design.sign[node] == 0 and not design.hard.any()
    pl = place_netlist(net, m, policy, restarts=1, cand_cap=m.n, time_limit_s=20)
    assert pl.mapping == {node: idx["I"]} and pl.carried == 1
    # Strong edges between OTHER hosts cannot carry this node's self-edge.
    keep = m.pre != m.post
    absent = MCNS(m.neurons, m.pre[keep], m.post[keep], m.count[keep], min_syn=1)
    pl = place_netlist(net, absent, policy, restarts=1, cand_cap=m.n, time_limit_s=20)
    assert len(pl.mapping) == 1 and not pl.unplaced and pl.carried == 0
    assert pl.missing_by_class == {"autapse (no host self-edge)": [1, 1]}


def chain_toy_netlist(n_pairs: int, kind: str = "ffpair", proxy: bool = False):
    """`n_pairs` flip-flops (or excitatory latches, kind="latch") in a line, each joined to the
    next by one edge relay read off the previous pair's `u` and igniting the next pair's `u`:
    the toy of docs/h1_placement.md "Placing flip-flops". Returns (net, drive, pairs, relays).
    For flip-flops the two edges from `u` into each relay (excitatory, from an inhibitory
    neuron) cannot be carried by any host: `possible_edges` counts the rest. With `proxy`, each
    flip-flop gets its excitatory proxy p and the relay reads p (`ff.rail`) instead of u: every
    edge is then of a carriable sign."""
    params = Params()
    drive = Drive.from_params(params)
    net = Netlist(params)
    pairs, relays = [], []
    for k in range(n_pairs):
        pairs.append(add_flipflop(net, drive, f"F{k}", proxy=proxy) if kind == "ffpair" else add_latch(net, drive, f"L{k}"))
        if k:
            relay = add_edge_relay(net, drive, f"r{k}", pairs[k - 1].rail if kind == "ffpair" else pairs[k - 1].u)
            net.synapse(relay, pairs[k].u, drive.ignite)
            relays.append(relay)
    return net, drive, pairs, relays


def possible_edges(net: Netlist) -> int:
    """Designed edges whose sign agrees with their source's (a flip-flop member is inhibitory)."""
    D = _Design(net, Policy(), hub_deg=20)
    return int((~D.impossible).sum())


def two_flipflop_netlist():
    """F0 (u0, v0) -> [edge relay: edge, edge_inh] -> F1 (u1, v1): 6 neurons, 8 designed edges,
    of which 6 are of a carriable sign."""
    net, drive, pairs, relays = chain_toy_netlist(2, "ffpair")
    ids = {"u0": pairs[0].u, "v0": pairs[0].v, "u1": pairs[1].u, "v1": pairs[1].v, "edge": relays[0], "edge_inh": relays[0] + 1}
    return net, drive, ids


def synthetic_ff_mcns(policy: Policy, drive: Drive, seed: int = 0, n_distractors: int = 80, n_weak: int = 600) -> tuple[MCNS, dict]:
    """Two planted driven mutual inhibitory pairs (U1, V1) and (U2, V2), each member with a
    cholinergic driver at the loop threshold, and a relay (E, I) with I -> E and E -> U2 at their
    requirements; decoys: an undriven inhibitory mutual pair (P, Q), a driven inhibitory pair (R, T)
    that no relay reaches, and an excitatory mutual pair (X, Y); random weak wiring around them."""
    rng = np.random.default_rng(seed)
    req = policy.req_count
    names = ["U1", "V1", "U2", "V2", "E", "I", "D1", "D2", "D3", "D4", "P", "Q", "R", "T", "D5", "X", "Y"]
    nts = {"U1": "gaba", "V1": "glutamate", "U2": "gaba", "V2": "gaba", "I": "gaba", "P": "gaba", "Q": "gaba", "R": "gaba", "T": "glutamate"}
    idx = {nm: i for i, nm in enumerate(names)}
    thr = req(drive.loop)
    edges = [
        (idx["U1"], idx["V1"], thr + 3), (idx["V1"], idx["U1"], thr), (idx["U2"], idx["V2"], thr + 1), (idx["V2"], idx["U2"], thr + 5),
        (idx["D1"], idx["U1"], thr + 2), (idx["D2"], idx["V1"], thr), (idx["D3"], idx["U2"], thr + 9), (idx["D4"], idx["V2"], thr),
        (idx["I"], idx["E"], req(int(round(2.2 * drive.loop))) + 2), (idx["E"], idx["U2"], req(drive.ignite) + 1),
        (idx["P"], idx["Q"], thr + 20), (idx["Q"], idx["P"], thr + 20),  # undriven decoy pair
        (idx["R"], idx["T"], thr + 4), (idx["T"], idx["R"], thr + 4), (idx["D5"], idx["R"], thr + 1), (idx["D5"], idx["T"], thr + 1),  # driven, unreachable
        (idx["X"], idx["Y"], thr + 2), (idx["Y"], idx["X"], thr + 2),  # excitatory pair (a latch host, not a flip-flop's)
    ]
    n0 = len(names)
    n = n0 + n_distractors
    nt_list = [nts.get(nm, "acetylcholine") for nm in names] + list(rng.choice(["acetylcholine", "gaba", "glutamate"], n_distractors))
    for _ in range(n_weak):
        a, b = rng.integers(0, n, 2)
        if a != b:
            edges.append((int(a), int(b), int(rng.integers(1, 4))))
    df = pd.DataFrame(edges, columns=["pre", "post", "count"]).groupby(["pre", "post"], as_index=False)["count"].sum()
    df = df.sort_values(["pre", "post"])
    neurons = pd.DataFrame({"bodyId": 2000 + np.arange(n), "type": [f"T{i}" for i in range(n)], "superclass": "cb",
                            "class": None, "somaSide": "R", "nt": nt_list, "nt_conf": 0.9})
    neurons["sign"] = np.where(neurons["nt"].isin(["gaba", "glutamate", "histamine"]), -1, 1).astype(np.int8)
    m = MCNS(neurons, df["pre"].to_numpy(np.int32), df["post"].to_numpy(np.int32), df["count"].to_numpy(np.int32), min_syn=1)
    return m, idx


def two_proxied_flipflop_netlist():
    """F0 (u0, v0, p0) -> [edge relay read off p0: edge, edge_inh] -> F1's set chain (set,
    set_relay1, set_relay2, three ignite pulses into u1) -> F1 (u1, v1, p1): 11 neurons, 15
    designed edges, every one of a carriable sign (the readout leaves p, an excitatory neuron)."""
    params = Params()
    drive = Drive.from_params(params)
    net = Netlist(params)
    f0 = add_flipflop(net, drive, "F0", proxy=True)
    f1 = add_flipflop(net, drive, "F1", proxy=True)
    relay = add_edge_relay(net, drive, "r1", f0.rail)
    trigger = add_set_chain(net, drive, "F1", f1, pulses=3)  # the synthetic connectome plants a 3-relay chain (T, R1, R2); the default train is 4 pulses since the lockstep fix
    net.synapse(relay, trigger, drive.ignite)
    ids = {"u0": f0.u, "v0": f0.v, "p0": f0.p, "u1": f1.u, "v1": f1.v, "p1": f1.p, "edge": relay, "edge_inh": relay + 1,
           "set": trigger, "set_relay1": trigger + 1, "set_relay2": trigger + 2}
    assert net.roles[ids["set_relay2"]] == "F1.set_relay2" and net.roles[ids["p0"]] == "F0.p"
    return net, drive, ids


def synthetic_triple_mcns(policy: Policy, drive: Drive, seed: int = 0, n_distractors: int = 80, n_weak: int = 600) -> tuple[MCNS, dict]:
    """Two planted triples: driven mutual inhibitory pairs (U1, V1) and (U2, V2) with V1 -> P1
    and V2 -> P2 at the proxy threshold (P1, P2 cholinergic; P2 -> Z its excitatory output), a
    relay (E, I) read off P1 (P1 -> E, P1 -> I, I -> E), E -> T and the set chain T, R1, R2 into
    U2. Decoys: an undriven inhibitory pair (PA, PB) with a proxy X1, a driven pair (QA, QB)
    with no proxy, an excitatory neuron X2 inhibited by a non-pair GABA neuron N, an excitatory
    mutual pair (XA, XB); random weak wiring around them."""
    rng = np.random.default_rng(seed)
    req = policy.req_count
    names = ["U1", "V1", "P1", "D1", "D2", "E", "I", "T", "R1", "R2", "U2", "V2", "P2", "D3", "D4", "Z",
             "PA", "PB", "X1", "QA", "QB", "D5", "N", "X2", "XA", "XB"]
    gaba = {"U1", "V1", "I", "U2", "V2", "PA", "PB", "QA", "QB", "N"}
    idx = {nm: i for i, nm in enumerate(names)}
    thr = req(drive.loop)
    ig, pu = req(drive.ignite), req(drive.pulse)
    edges = [
        (idx["U1"], idx["V1"], thr + 3), (idx["V1"], idx["U1"], thr), (idx["D1"], idx["U1"], thr + 2), (idx["D2"], idx["V1"], thr),
        (idx["V1"], idx["P1"], thr + 1),
        (idx["P1"], idx["E"], req(drive.relay_in) + 2), (idx["P1"], idx["I"], pu + 1), (idx["I"], idx["E"], req(int(round(2.2 * drive.loop))) + 3),
        (idx["E"], idx["T"], ig + 1),
        (idx["T"], idx["U2"], ig + 1), (idx["T"], idx["R1"], pu + 1), (idx["R1"], idx["U2"], ig), (idx["R1"], idx["R2"], pu + 2), (idx["R2"], idx["U2"], ig + 3),
        (idx["U2"], idx["V2"], thr + 1), (idx["V2"], idx["U2"], thr + 5), (idx["D3"], idx["U2"], thr + 9), (idx["D4"], idx["V2"], thr),
        (idx["V2"], idx["P2"], thr + 2), (idx["P2"], idx["Z"], thr + 1),
        (idx["PA"], idx["PB"], thr + 20), (idx["PB"], idx["PA"], thr + 20), (idx["PA"], idx["X1"], thr + 5),  # undriven pair with a proxy
        (idx["QA"], idx["QB"], thr + 4), (idx["QB"], idx["QA"], thr + 4), (idx["D5"], idx["QA"], thr + 1), (idx["D5"], idx["QB"], thr + 1),  # driven, no proxy
        (idx["N"], idx["X2"], thr + 3),  # an inhibited excitatory neuron whose inhibitor is in no pair
        (idx["XA"], idx["XB"], thr + 2), (idx["XB"], idx["XA"], thr + 2),  # excitatory pair
    ]
    n0 = len(names)
    n = n0 + n_distractors
    nt_list = ["gaba" if nm in gaba else "acetylcholine" for nm in names] + list(rng.choice(["acetylcholine", "gaba", "glutamate"], n_distractors))
    for _ in range(n_weak):
        a, b = rng.integers(0, n, 2)
        if a != b:
            edges.append((int(a), int(b), int(rng.integers(1, 4))))
    df = pd.DataFrame(edges, columns=["pre", "post", "count"]).groupby(["pre", "post"], as_index=False)["count"].sum()
    df = df.sort_values(["pre", "post"])
    neurons = pd.DataFrame({"bodyId": 3000 + np.arange(n), "type": [f"T{i}" for i in range(n)], "superclass": "cb",
                            "class": None, "somaSide": "R", "nt": nt_list, "nt_conf": 0.9})
    neurons["sign"] = np.where(neurons["nt"].isin(["gaba", "glutamate", "histamine"]), -1, 1).astype(np.int8)
    m = MCNS(neurons, df["pre"].to_numpy(np.int32), df["post"].to_numpy(np.int32), df["count"].to_numpy(np.int32), min_syn=1)
    return m, idx


def test_motifs_are_found_from_structure():
    net, _, ids = latch_relay_netlist()
    D = _Design(net, Policy(), hub_deg=20)
    kinds = {mo.kind: mo.nodes for mo in D.motifs}
    assert set(kinds) == {"latch", "relay", "single"}
    assert set(kinds["latch"]) == {ids["u"], ids["v"]}
    assert kinds["relay"] == (ids["edge"], ids["edge_inh"])  # E first, then its inhibitor
    assert kinds["single"] == (ids["src"],)
    assert D.hard.all() and not D.hub.any()


def test_single_latch_with_relay_is_placed_completely():
    net, drive, ids = latch_relay_netlist()
    policy = Policy()
    m, idx = synthetic_mcns(policy, drive)
    pl = place_netlist(net, m, policy, time_limit_s=20, restarts=2, verbose=False)
    assert isinstance(pl, Placement)
    assert pl.carried == net.nnz == 6, (pl.summary(net), pl.missing)
    assert not pl.unplaced
    # the planted neurons, in their roles
    assert pl.mapping[ids["src"]] == idx["S"]
    assert pl.mapping[ids["edge"]] == idx["E"]
    assert pl.mapping[ids["edge_inh"]] == idx["I"]
    assert {pl.mapping[ids["u"]], pl.mapping[ids["v"]]} == {idx["U"], idx["V"]}
    assert pl.motifs == {"latch": {"complete": 1}, "relay": {"complete": 1}, "single": {"complete": 1}}
    # parasitic = every anatomical edge among the five hosts that is not a designed edge (weak ones too)
    hosts = set(pl.mapping.values())
    among = sum(1 for a, b in zip(m.pre, m.post) if a in hosts and b in hosts)
    assert pl.parasitic == among - 6


def test_missing_inhibitor_is_reported_not_faked():
    """Remove the planted I -> E edge: no complete relay exists any more. The search must still
    place every neuron (the latch as a real pair, the relay by best fit) and the audit must
    list exactly the designed edges the chosen hosts do not carry, checked against the data."""
    net, drive, ids = latch_relay_netlist()
    policy = Policy()
    m, idx = synthetic_mcns(policy, drive)
    keep = ~((m.pre == idx["I"]) & (m.post == idx["E"]))
    m2 = MCNS(m.neurons, m.pre[keep], m.post[keep], m.count[keep], min_syn=1)
    pl = place_netlist(net, m2, policy, time_limit_s=20, restarts=2, verbose=False)
    assert not pl.unplaced
    assert 4 <= pl.carried <= 5, (pl.summary(net), pl.missing)  # the optimum is 5; a decoy pair gives 4
    assert pl.carried + len(pl.missing) == net.nnz
    assert pl.motifs["latch"] == {"complete": 1} and (ids["edge_inh"], ids["edge"]) in [(s, d) for s, d, _, _ in pl.missing]
    counts = {(int(a), int(b)): int(c) for a, b, c in zip(m2.pre, m2.post, m2.count)}
    for s, d, q in zip(net.src, net.dst, net.quanta):
        carried_really = counts.get((pl.mapping[s], pl.mapping[d]), 0) >= policy.req_count(q)
        assert carried_really == ((s, d) not in [(a, b) for a, b, _, _ in pl.missing]), (s, d)


def _external_synapses(m: MCNS, hosts: set) -> tuple[int, int]:
    """By hand: synapses into the hosts from non-hosts, and from the hosts to non-hosts."""
    ins = outs = 0
    for a, b, c in zip(m.pre.tolist(), m.post.tolist(), m.count.tolist()):
        if b in hosts and a not in hosts:
            ins += c
        if a in hosts and b not in hosts:
            outs += c
    return ins, outs


def test_isolation_weight_prefers_the_quiet_host():
    """Two latch pairs carry the circuit's edges equally well; U2 has 50 extra decoy inputs.
    With isolation_weight = 0 the search may take either (coverage is a tie); with
    isolation_weight = 2 it must take the quiet pair. The summary's exposure figures must match
    a by-hand count of the hosts' synapses from and to the rest of the synthetic connectome."""
    net, drive, ids = latch_relay_netlist()
    policy = Policy()
    m, idx = synthetic_mcns(policy, drive, twin_latch=True)
    quiet, noisy = {idx["U"], idx["V"]}, {idx["U2"], idx["V2"]}
    picked = set()
    for seed in range(3):
        pl0 = place_netlist(net, m, policy, time_limit_s=20, restarts=2, seed=seed, verbose=False)
        assert pl0.carried == net.nnz == 6, (pl0.summary(net), pl0.missing)
        hosts = {pl0.mapping[ids["u"]], pl0.mapping[ids["v"]]}
        assert hosts in (quiet, noisy), hosts
        picked.add(frozenset(hosts))
        assert pl0.summary(net)["isolation_weight"] == 0.0
    assert picked  # either pair is a legal answer at weight 0
    for seed in range(3):
        pl2 = place_netlist(net, m, policy, time_limit_s=20, restarts=2, seed=seed, verbose=False, isolation_weight=2.0)
        assert pl2.carried == net.nnz == 6, (pl2.summary(net), pl2.missing)
        assert {pl2.mapping[ids["u"]], pl2.mapping[ids["v"]]} == quiet, pl2.mapping
        s = pl2.summary(net)
        ins, outs = _external_synapses(m, set(pl2.mapping.values()))
        assert s["external_in_synapses"] == ins and s["external_out_synapses"] == outs, (s, ins, outs)
        assert s["external_in_mean"] == round(ins / 5, 1) and s["external_out_mean"] == round(outs / 5, 1)
        assert s["external_in_max"] <= ins and s["external_in_p90"] <= s["external_in_max"]
        assert s["isolation_weight"] == 2.0
        assert pl2.objective < pl2.carried  # the penalty is charged
    # the noisy pair really is noisier: its hosts' external input count exceeds the quiet pair's
    ins_q, _ = _external_synapses(m, (set(pl2.mapping.values()) - quiet) | quiet)
    ins_n, _ = _external_synapses(m, (set(pl2.mapping.values()) - quiet) | noisy)
    assert ins_n >= ins_q + 100, (ins_q, ins_n)


def _one_edge_mcns(nt_pre: str, nt_post: str, count: int) -> MCNS:
    neurons = pd.DataFrame({"bodyId": [1000, 1001], "type": ["Ta", "Tb"], "superclass": "cb",
                            "class": None, "somaSide": "R", "nt": [nt_pre, nt_post], "nt_conf": 0.9})
    neurons["sign"] = np.where(neurons["nt"].isin(["gaba", "glutamate", "histamine"]), -1, 1).astype(np.int8)
    return MCNS(neurons, np.array([0], np.int32), np.array([1], np.int32), np.array([count], np.int32), min_syn=1)


def test_duplicate_designed_edges_are_merged_by_audit():
    """The netlist does not merge synapses: the same (src, dst) pair can appear twice (as it
    does in the adder). One anatomical edge carries both at once, so the audit must count them
    as one distinct edge with quanta summed, not as two carried designed edges."""
    params = Params()
    net = Netlist(params)
    a, b = net.neuron("a"), net.neuron("b")
    net.synapse(a, b, 50)
    net.synapse(a, b, 50)  # duplicate designed edge, same pair
    policy = Policy()
    m = _one_edge_mcns("acetylcholine", "acetylcholine", policy.req_count(100))  # just enough for the summed 100
    D = _Design(net, policy, hub_deg=20)
    A = _Anatomy(m, policy)
    real = np.array([0, 1], dtype=np.int64)
    pl = audit(net, D, A, real, 0.0, {}, "test", policy)
    s = pl.summary(net)
    assert s["synapses"] == net.nnz == 2
    assert s["edges"] == pl.edges == 1
    assert pl.carried == 1
    assert pl.missing == []
    assert pl.parasitic == 0


def test_audit_catches_a_wrong_sign_host():
    """The search only ever draws from sign-masked domains, but the audit itself must still
    catch a host whose own transmitter does not match the designed neuron's sign -- planted
    here through a hand-made mapping the search would never produce."""
    params = Params()
    net = Netlist(params)
    a, b = net.neuron("a"), net.neuron("b")
    net.synapse(a, b, 100)  # a is excitatory-designed (positive quanta)
    policy = Policy()
    req = policy.req_count(100)
    m = _one_edge_mcns("gaba", "acetylcholine", req + 10)  # host of a is gaba: wrong sign, plenty strong
    D = _Design(net, policy, hub_deg=20)
    A = _Anatomy(m, policy)
    real = np.array([0, 1], dtype=np.int64)
    pl = audit(net, D, A, real, 0.0, {}, "test", policy)
    assert pl.carried == 0
    assert len(pl.missing) == 1
    assert pl.missing[0][3] == "sign"


def test_greedy_baseline_runs_on_synthetic():
    net, drive, _ = latch_relay_netlist()
    m, _ = synthetic_mcns(Policy(), drive)
    pl = place_netlist_greedy(net, m, Policy())
    assert 0 <= pl.carried <= net.nnz


# ----------------------------------------------------------------------------------------
# flip-flops (protocol.flipflop): the ffpair motif
# ----------------------------------------------------------------------------------------
def test_flipflop_motifs_are_found_from_structure():
    net, drive, ids = two_flipflop_netlist()
    assert net.bias[ids["u0"]] == net.bias[ids["v1"]] == BIAS_213HZ_MV and net.bias[ids["edge"]] == 0.0
    D = _Design(net, Policy(), hub_deg=20)
    kinds: dict = {}
    for mo in D.motifs:
        kinds.setdefault(mo.kind, []).append(mo)
    assert set(kinds) == {"ffpair", "single"}, {k: len(v) for k, v in kinds.items()}
    assert sorted(mo.nodes for mo in kinds["ffpair"]) == [(ids["u0"], ids["v0"]), (ids["u1"], ids["v1"])]
    thr = Policy().req_count(drive.loop)
    for mo in kinds["ffpair"]:
        assert mo.reqs["r_uv"] == mo.reqs["r_vu"] == thr and mo.reqs["driver"] and mo.reqs["bias"] == (BIAS_213HZ_MV, BIAS_213HZ_MV)
        assert len(mo.edges) == 2 and all(D.q[e] == -drive.loop for e in mo.edges)
    # a flip-flop member is inhibitory even though its two readout edges (u0 -> edge, u0 -> edge_inh) are excitatory
    assert D.sign[ids["u0"]] == D.sign[ids["v0"]] == -1 and D.sign[ids["edge"]] == 1 and D.sign[ids["edge_inh"]] == -1
    assert D.mixed == [ids["u0"]]
    imp = {(int(D.src[e]), int(D.dst[e])) for e in np.flatnonzero(D.impossible)}
    assert imp == {(ids["u0"], ids["edge"]), (ids["u0"], ids["edge_inh"])}
    assert int(D.hard.sum()) == 6 == possible_edges(net)
    # the relay is not a relay motif here (its source edges are impossible): E and I are singles joined by a hard edge
    assert sorted(mo.nodes[0] for mo in kinds["single"]) == sorted([ids["edge"], ids["edge_inh"]])
    # the excitatory latch motif is unchanged: a latch netlist has no ffpair, and its members are excitatory
    net_l, _, ids_l = latch_relay_netlist()
    D_l = _Design(net_l, Policy(), hub_deg=20)
    assert [mo.kind for mo in D_l.motifs if mo.kind in ("latch", "ffpair")] == ["latch"] and not D_l.ff_members
    assert D_l.sign[ids_l["u"]] == 1


def test_real_mutual_inhibitory_pairs_need_a_driver():
    _, drive, _ = two_flipflop_netlist()
    policy = Policy()
    m, idx = synthetic_ff_mcns(policy, drive)
    A = _Anatomy(m, policy)
    thr = policy.req_count(drive.loop)
    mask, pairs = A.mutual_inh(thr, driven=True)
    assert {tuple(sorted(p)) for p in pairs.tolist()} == {(idx["U1"], idx["V1"]), (idx["U2"], idx["V2"]), (idx["R"], idx["T"])}
    assert not mask[idx["P"]] and not mask[idx["X"]] and mask[idx["U1"]]
    mask_all, pairs_all = A.mutual_inh(thr, driven=False)
    assert len(pairs_all) == 4 and mask_all[idx["P"]] and mask_all[idx["Q"]] and not mask_all[idx["X"]]
    mm, _ = A.mutual(thr)  # the excitatory pairs are a different pool
    assert mm[idx["X"]] and mm[idx["Y"]] and not mm[idx["U1"]]


def test_two_flipflops_with_relay_are_placed_completely_on_synthetic():
    net, drive, ids = two_flipflop_netlist()
    policy = Policy()
    m, idx = synthetic_ff_mcns(policy, drive)
    pl = place_netlist(net, m, policy, time_limit_s=20, restarts=2, verbose=False)
    assert not pl.unplaced and len(pl.mapping) == net.n == 6
    # both pairs on the planted driven pairs, the reachable one under the relay
    assert {pl.mapping[ids["u1"]], pl.mapping[ids["v1"]]} == {idx["U2"], idx["V2"]}
    assert {pl.mapping[ids["u0"]], pl.mapping[ids["v0"]]} == {idx["U1"], idx["V1"]}
    assert pl.mapping[ids["edge"]] == idx["E"] and pl.mapping[ids["edge_inh"]] == idx["I"] and pl.mapping[ids["u1"]] == idx["U2"]
    # every edge of a carriable sign is carried; the two excitatory readout edges from u0 are reported as wrong sign
    assert pl.carried == possible_edges(net) == 6 and pl.edges == net.nnz == 8
    assert sorted((s, d, r) for s, d, _, r in pl.missing) == sorted([(ids["u0"], ids["edge"], "wrong sign (mixed-sign designed neuron)"),
                                                                      (ids["u0"], ids["edge_inh"], "wrong sign (mixed-sign designed neuron)")])
    assert pl.motifs == {"ffpair": {"complete": 2}, "single": {"complete": 2}}
    assert pl.ffpair_drivers == {"hosts": 4, "with_driver": 4}
    s = pl.summary(net)
    assert s["ffpair_hosts"] == 4 and s["ffpair_hosts_with_driver"] == 4 and s["carried_fraction"] == 0.75
    # without the driver requirement the undriven decoy pair is a legal host too; the drivers are still reported
    A = _Anatomy(m, policy)
    thr = policy.req_count(drive.loop)
    assert A.mutual_inh(thr, driven=False)[0][idx["P"]]
    pl2 = place_netlist(net, m, policy, time_limit_s=20, restarts=2, verbose=False, ff_driver=False)
    assert pl2.carried == 6 and pl2.ffpair_drivers["hosts"] == 4 and 2 <= pl2.ffpair_drivers["with_driver"] <= 4


# ----------------------------------------------------------------------------------------
# flip-flops with a proxy: the fftriple motif
# ----------------------------------------------------------------------------------------
def test_fftriple_motifs_are_found_from_structure():
    net, drive, ids = two_proxied_flipflop_netlist()
    assert net.n == 11 and net.nnz == 15
    policy = Policy()
    D = _Design(net, policy, hub_deg=20)
    kinds: dict = {}
    for mo in D.motifs:
        kinds.setdefault(mo.kind, []).append(mo)
    assert set(kinds) == {"fftriple", "relay", "single"}, {k: len(v) for k, v in kinds.items()}
    # nodes ordered (u, v, p): nodes[1] is the member that inhibits the proxy
    assert sorted(mo.nodes for mo in kinds["fftriple"]) == [(ids["u0"], ids["v0"], ids["p0"]), (ids["u1"], ids["v1"], ids["p1"])]
    thr = policy.req_count(drive.loop)
    for mo in kinds["fftriple"]:
        assert mo.reqs["r_uv"] == mo.reqs["r_vu"] == mo.reqs["r_vp"] == thr and mo.reqs["driver"]
        assert mo.reqs["bias"] == (BIAS_213HZ_MV, BIAS_213HZ_MV, BIAS_213HZ_MV)
        u, v, p = mo.nodes
        assert [(int(D.src[e]), int(D.dst[e])) for e in mo.edges] == [(u, v), (v, u), (v, p)] and all(D.q[e] < 0 for e in mo.edges)
    assert D.ff_proxy_of == {ids["p0"]: ids["v0"], ids["p1"]: ids["v1"]}
    assert D.ff_members == {ids["u0"], ids["v0"], ids["u1"], ids["v1"]}
    # the members are inhibitory, the proxies excitatory (p1 drives nothing and is still excitatory), nothing is mixed
    assert D.sign[ids["v0"]] == -1 and D.sign[ids["p0"]] == D.sign[ids["p1"]] == 1 and D.mixed == []
    assert not D.impossible.any() and D.hard.all() and possible_edges(net) == 15
    # the relay is read off p0: a relay motif whose source is the proxy
    (relay,) = kinds["relay"]
    assert relay.nodes == (ids["edge"], ids["edge_inh"]) and relay.reqs["inhibitors"][0][3] == ids["p0"]
    assert sorted(mo.nodes[0] for mo in kinds["single"]) == sorted([ids["set"], ids["set_relay1"], ids["set_relay2"]])
    # a flip-flop without a proxy stays an ffpair; one with both proxies is a triple on p plus q as a single;
    # one with only the CLEAR proxy is a triple whose nodes[1] is u (the member inhibiting q)
    params = Params()
    d2 = Drive.from_params(params)
    net2 = Netlist(params)
    plain = add_flipflop(net2, d2, "A")
    both = add_flipflop(net2, d2, "B", proxy=True, clear_proxy=True)
    clear_only = add_flipflop(net2, d2, "C", clear_proxy=True)
    D2 = _Design(net2, policy, hub_deg=20)
    by_kind = {}
    for mo in D2.motifs:
        by_kind.setdefault(mo.kind, []).append(mo.nodes)
    assert by_kind["ffpair"] == [(plain.u, plain.v)]
    assert sorted(by_kind["fftriple"]) == sorted([(both.u, both.v, both.p), (clear_only.v, clear_only.u, clear_only.q)])
    assert by_kind["single"] == [(both.q,)]
    assert D2.ff_proxy_of == {both.p: both.v, both.q: both.u, clear_only.q: clear_only.u}


def test_real_triple_hosts():
    _, drive, _ = two_proxied_flipflop_netlist()
    policy = Policy()
    m, idx = synthetic_triple_mcns(policy, drive)
    A = _Anatomy(m, policy)
    thr = policy.req_count(drive.loop)
    has_proxy, proxy = A.proxy_hosts(thr, thr, driven=True)
    assert {int(x) for x in np.flatnonzero(has_proxy)} == {idx["V1"], idx["V2"]}
    assert {int(x) for x in np.flatnonzero(proxy)} == {idx["P1"], idx["P2"]}
    # without the driver requirement the undriven pair's proxy joins the pool; N's target never does
    has_proxy2, proxy2 = A.proxy_hosts(thr, thr, driven=False)
    assert {int(x) for x in np.flatnonzero(has_proxy2)} == {idx["V1"], idx["V2"], idx["PA"]}
    assert {int(x) for x in np.flatnonzero(proxy2)} == {idx["P1"], idx["P2"], idx["X1"]} and not proxy2[idx["X2"]]


def test_two_proxied_flipflops_are_placed_completely_including_the_readout():
    """The p -> relay edges (and everything downstream) are placed: the readout that the
    pair-only motif reports as "wrong sign" is carried once it leaves the proxy."""
    net, drive, ids = two_proxied_flipflop_netlist()
    policy = Policy()
    m, idx = synthetic_triple_mcns(policy, drive)
    pl = place_netlist(net, m, policy, time_limit_s=30, restarts=2, verbose=False)
    assert not pl.unplaced and len(pl.mapping) == net.n == 11
    want = {"u0": "U1", "v0": "V1", "p0": "P1", "edge": "E", "edge_inh": "I", "set": "T", "set_relay1": "R1", "set_relay2": "R2",
            "u1": "U2", "v1": "V2", "p1": "P2"}
    assert {k: pl.mapping[ids[k]] for k in want} == {k: idx[v] for k, v in want.items()}, pl.mapping
    assert pl.carried == pl.edges == net.nnz == 15 and pl.missing == [], (pl.summary(net), pl.missing)
    assert pl.motifs == {"fftriple": {"complete": 2}, "relay": {"complete": 1}, "single": {"complete": 3}}
    assert pl.motifs_search == {"fftriple": {"complete": 2}, "relay": {"complete": 1}, "single": {"complete": 3}}
    assert pl.ffpair_drivers == {"hosts": 4, "with_driver": 4}
    assert pl.proxy_readout == {"proxies": 2, "placed": 2, "edges": 2, "carried": 2}
    s = pl.summary(net)
    assert s["carried_fraction"] == 1.0 and s["proxies"] == 2 and s["proxy_readout_edges"] == 2 and s["proxy_readout_carried"] == 2
    # the readout edges themselves, checked against the data
    counts = {(int(a), int(b)): int(c) for a, b, c in zip(m.pre, m.post, m.count)}
    for d in ("edge", "edge_inh"):
        e = [k for k, (s_, d_) in enumerate(zip(net.src, net.dst)) if s_ == ids["p0"] and d_ == ids[d]][0]
        assert counts[(pl.mapping[ids["p0"]], pl.mapping[ids[d]])] >= policy.req_count(net.quanta[e])
    # the same two flip-flops read off u instead: the pair-only motif, 2 readout edges impossible
    net_u, _, ids_u = two_flipflop_netlist()
    assert possible_edges(net_u) == 6 == net_u.nnz - 2
    pl_u = place_netlist(net_u, m, policy, time_limit_s=20, restarts=1, verbose=False)
    assert pl_u.proxy_readout == {} and pl_u.carried <= 6 and pl_u.summary(net_u)["proxies"] == 0


def test_triple_without_a_proxy_host_falls_back_to_the_pair():
    """Remove V2 -> P2: F1's triple has no complete instance (u1 must take U2, the only neuron
    with three ignite inputs, and V2 inhibits no excitatory neuron). The search must still
    land F1's pair on the real driven pair, leave p1 unplaced rather than fake it, and place
    F0's triple and the whole readout as before: 14 of 15."""
    net, drive, ids = two_proxied_flipflop_netlist()
    policy = Policy()
    m, idx = synthetic_triple_mcns(policy, drive)
    keep = ~((m.pre == idx["V2"]) & (m.post == idx["P2"]))
    m2 = MCNS(m.neurons, m.pre[keep], m.post[keep], m.count[keep], min_syn=1)
    pl = place_netlist(net, m2, policy, time_limit_s=30, restarts=2, verbose=False)
    assert pl.carried == 14 and pl.unplaced == [ids["p1"]], (pl.summary(net), pl.missing)
    assert pl.missing == [(ids["v1"], ids["p1"], -drive.loop, "endpoint unplaced")]
    assert pl.mapping[ids["u1"]] == idx["U2"] and pl.mapping[ids["v1"]] == idx["V2"]
    assert (pl.mapping[ids["u0"]], pl.mapping[ids["v0"]], pl.mapping[ids["p0"]]) == (idx["U1"], idx["V1"], idx["P1"])
    assert pl.motifs["fftriple"] == {"complete": 1, "partial": 1}
    assert pl.ffpair_drivers == {"hosts": 4, "with_driver": 4}
    assert pl.proxy_readout == {"proxies": 2, "placed": 1, "edges": 2, "carried": 2}
    counts = {(int(a), int(b)): int(c) for a, b, c in zip(m2.pre, m2.post, m2.count)}
    for s_, d_, q in zip(net.src, net.dst, net.quanta):
        if s_ in pl.mapping and d_ in pl.mapping:
            assert counts.get((pl.mapping[s_], pl.mapping[d_]), 0) >= policy.req_count(q), (s_, d_)


@pytest.mark.skipif(not path_of("weights", DEFAULT_DIR).exists(), reason="MCNS data not downloaded")
def test_proxied_flipflop_chain_toy_on_mcns():
    """The chain toy with proxies at 6 flip-flops: every triple on a real (driven pair, proxy),
    the proxies' readout edges carried (the 32-pair figures are in docs/h1_placement.md)."""
    from drosophilos.connectome.mcns import load_mcns

    m = load_mcns()
    net, _, pairs, relays = chain_toy_netlist(6, "ffpair", proxy=True)
    assert possible_edges(net) == net.nnz == 6 * 3 + 5 * 4
    pl = place_netlist(net, m, Policy(), time_limit_s=90, restarts=1, verbose=False)
    assert pl.motifs.get("fftriple", {}).get("complete", 0) == 6, pl.motifs
    assert pl.ffpair_drivers["hosts"] == 12 and pl.proxy_readout["proxies"] == 6 and pl.proxy_readout["placed"] == 6
    assert pl.proxy_readout["edges"] == 10 and pl.proxy_readout["carried"] >= 8, pl.proxy_readout
    assert pl.carried + len(pl.missing) == pl.edges and pl.carried >= 30
    # every proxy host is cholinergic and inhibited by its v host at the loop threshold
    A = _Anatomy(m, Policy())
    thr = Policy().req_count(Drive.from_params(Params()).loop)
    for f in pairs:
        assert A.exc[pl.mapping[f.p]] and A.inh[pl.mapping[f.v]] and A.count(pl.mapping[f.v], pl.mapping[f.p]) >= thr


@pytest.mark.skipif(not path_of("weights", DEFAULT_DIR).exists(), reason="MCNS data not downloaded")
def test_flipflop_chain_toy_on_mcns():
    """A short flip-flop chain and the matching latch chain on the real connectome (the 32-pair
    figures are in docs/h1_placement.md "Placing flip-flops"; this is the same builder at 6)."""
    from drosophilos.connectome.mcns import load_mcns

    m = load_mcns()
    net_f, _, _, _ = chain_toy_netlist(6, "ffpair")
    pl_f = place_netlist(net_f, m, Policy(), time_limit_s=60, restarts=1, verbose=False)
    assert pl_f.motifs.get("ffpair", {}).get("complete", 0) == 6, pl_f.motifs
    assert pl_f.ffpair_drivers["hosts"] == 12 and pl_f.ffpair_drivers["with_driver"] == 12
    assert pl_f.carried + len(pl_f.missing) == pl_f.edges and pl_f.carried >= 12
    net_l, _, _, _ = chain_toy_netlist(6, "latch")
    pl_l = place_netlist(net_l, m, Policy(), time_limit_s=60, restarts=1, verbose=False)
    assert pl_l.motifs.get("latch", {}).get("complete", 0) == 6, pl_l.motifs
    assert pl_l.ffpair_drivers == {}


@pytest.mark.skipif(not path_of("weights", DEFAULT_DIR).exists(), reason="MCNS data not downloaded")
def test_single_latch_with_relay_on_mcns():
    """The same 5-neuron circuit on the real connectome: MCNS has thousands of instances."""
    from drosophilos.connectome.mcns import load_mcns

    net, _, _ = latch_relay_netlist()
    m = load_mcns()
    pl = place_netlist(net, m, Policy(), time_limit_s=60, restarts=1, verbose=False)
    assert pl.carried == net.nnz, pl.summary(net)
