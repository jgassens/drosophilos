"""Netlist.split_hubs: broadcast neurons become trees of identical copies, and nothing else
changes (lib/netlist.py).

(a) The 4-bit adder channel (614 neurons; Q.reset_inh fans out to 115 targets) split at
max_fanout 8: every out-degree is <= 8, and over 20 random additions driven as
tests/test_embed_image.py's condition A drives the placed adder (the same harness,
simulate_channel, on the netlist's own topology, chained through one simulator) every original
neuron's spike train is identical step for step before and after the split, and the sums are
right. (b) A synthetic source feeding three hubs of 30 targets each at max_fanout 4: the hubs
split into trees, the source's out-degree grows past the bound and it is split in turn; a hub
that feeds itself through the tree is refused and the netlist left untouched.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from drosophilos.connectome.embed_image import adder_cases, simulate_channel
from drosophilos.lib.adder import build_adder_channel
from drosophilos.lib.netlist import Netlist
from drosophilos.sim.model import Params

N_CASES = 20


def _trains(sim, neurons):
    ev = sim.trace.events
    steps, who = np.asarray(ev["step"]), np.asarray(ev["neuron"])
    return {x: steps[who == x].tolist() for x in neurons}


def _run(ch, words, expected, params):
    """Condition A's own driver: simulate_channel on a topology that carries every designed
    edge at its designed quanta (here the netlist's), the words chained through one simulator."""
    image = SimpleNamespace(topology=ch.net.topology())
    res = simulate_channel(ch, image, words, expected, params, fresh_each=False)
    return res


def _in_degree_inputs(net: Netlist, x: int):
    return sorted((s, q, d) for s, t, q, d in zip(net.src, net.dst, net.quanta, net.delay) if t == x)


def test_adder_split_at_8_keeps_every_original_spike_train():
    params = Params()
    ch = build_adder_channel(params, 4, ordered=True, watchdog_hops=100)
    ch2 = build_adder_channel(params, 4, ordered=True, watchdog_hops=100)
    n0, nnz0 = ch.net.n, ch.net.nnz
    assert ch2.net.out_degrees().max() > 8
    copies = ch2.net.split_hubs(8)
    net2 = ch2.net
    assert net2.out_degrees().max() <= 8
    assert net2.n > n0 and net2.nnz > nnz0
    # the originals keep their indices, roles and synapse indices; every copy carries the root's
    # role plus a .cN suffix and the root's own inputs
    assert net2.roles[:n0] == ch.net.roles
    assert len(set(net2.roles)) == net2.n  # a neuron split twice numbers its copies on
    assert net2.dst[:nnz0] == ch.net.dst and net2.quanta[:nnz0] == ch.net.quanta and net2.delay[:nnz0] == ch.net.delay
    hub = ch.net.roles.index("Q.reset_inh")
    assert hub in copies and len(copies[hub]) >= 14
    for root, cs in copies.items():
        for c in cs:
            assert net2.roles[c].startswith(ch.net.roles[root] + ".c")
    # a first-level copy has exactly the original's inputs (same pre, quanta, delay); the
    # union of the tree's outputs is the original's output set
    c1 = net2.roles.index("Q.reset_inh.c1")
    assert _in_degree_inputs(net2, c1) == _in_degree_inputs(ch.net, hub)
    tree = [hub] + copies[hub]
    outs = sorted((net2.dst[e], net2.quanta[e]) for e in range(nnz0) if net2.src[e] in tree)
    assert outs == sorted((ch.net.dst[e], ch.net.quanta[e]) for e in range(nnz0) if ch.net.src[e] == hub)

    _, words, expected = adder_cases(4, N_CASES, seed=0)
    r1 = simulate_channel(ch, SimpleNamespace(topology=ch.net.topology()), words, expected, params, fresh_each=False)
    r2 = simulate_channel(ch2, SimpleNamespace(topology=net2.topology()), words, expected, params, fresh_each=False)
    for r in (r1, r2):
        assert r["correct"] == N_CASES and r["faults"] == 0 and r["timeouts"] == 0, {k: v for k, v in r.items() if k != "records"}
    for a, b in zip(r1["records"], r2["records"]):
        assert (a.load_step, a.accept_step, a.ready_step, a.decoded) == (b.load_step, b.accept_step, b.ready_step, b.decoded)


def test_adder_split_spike_trains_identical_step_for_step():
    from drosophilos.protocol.run import run_transactions
    from drosophilos.sim.ref64 import RefSim

    params = Params()
    ch = build_adder_channel(params, 4, ordered=True, watchdog_hops=100)
    ch2 = build_adder_channel(params, 4, ordered=True, watchdog_hops=100)
    n0 = ch.net.n
    copies = ch2.net.split_hubs(8)
    _, words, expected = adder_cases(4, N_CASES, seed=0)
    # the same driver simulate_channel uses (run_transactions), one simulator per netlist
    _, sim1, st1 = run_transactions(ch, params, words, expected=expected, sim=RefSim(ch.net.topology(), params), max_steps_per_tx=12000)
    _, sim2, st2 = run_transactions(ch2, params, words, expected=expected, sim=RefSim(ch2.net.topology(), params), max_steps_per_tx=12000)
    assert sim1.step_index == sim2.step_index
    t1 = _trains(sim1, range(n0))
    t2 = _trains(sim2, range(n0))
    assert t1 == t2
    assert sum(len(v) for v in t1.values()) == st1["total_spikes"] > 0
    # and every copy spikes exactly when its root spikes
    t_copies = _trains(sim2, [c for cs in copies.values() for c in cs])
    for root, cs in copies.items():
        for c in cs:
            assert t_copies[c] == t1[root], ch2.net.roles[c]


def _source_and_hubs(n_hubs: int = 3, n_targets: int = 30) -> tuple[Netlist, int, list[int]]:
    net = Netlist(Params())
    S = net.neuron("S")
    hubs = []
    for h in range(n_hubs):
        H = net.neuron(f"H{h}")
        net.synapse(S, H, 100)
        for t in range(n_targets):
            net.synapse(H, net.neuron(f"H{h}.t{t}"), -50, 3)
        hubs.append(H)
    return net, S, hubs


def test_iterative_split_reaches_the_source():
    net, S, hubs = _source_and_hubs()
    n0, nnz0 = net.n, net.nnz
    copies = net.split_hubs(4)
    deg = net.out_degrees()
    assert deg.max() <= 4
    # each hub: 30 targets at 4 per neuron -> 7 copies; the source then has 3 + 21 = 24 outputs
    # and is split into 5 copies of its own
    for H in hubs:
        assert len(copies[H]) == 7
        assert all(net.roles[c] == f"{net.roles[H]}.c{i + 1}" for i, c in enumerate(copies[H]))
        tree = [H] + copies[H]
        assert sum(int(deg[x]) for x in tree) == 30
        for c in copies[H]:  # the copy's inputs are the hub's: one synapse of 100 quanta from S or one of S's copies
            ins = [(net.roles[s].split(".")[0], q, d) for s, t, q, d in zip(net.src, net.dst, net.quanta, net.delay) if t == c]
            assert ins == [("S", 100, net.params.default_delay_steps)]
    assert S in copies and len(copies[S]) == 5
    assert deg[S] == 4 and all(deg[c] == 4 for c in copies[S])
    assert net.n == n0 + 3 * 7 + 5 and net.nnz == nnz0 + 3 * 7  # 21 hub copies get one input each; S has none to copy
    assert net.out_degrees().max() <= 4
    # idempotent: nothing is above the bound now
    assert net.split_hubs(4) == {}


def test_roles_filter_restricts_the_hubs_split_but_not_the_induced_ones():
    net, S, hubs = _source_and_hubs()
    copies = net.split_hubs(4, roles=["H0"])
    # H0 alone is split on its own account; S's out-degree, 3 -> 10, was raised by the transform
    # itself, so S is split too; H1 and H2 are left as they are
    deg = net.out_degrees()
    assert set(copies) == {hubs[0], S}
    assert deg[hubs[1]] == 30 and deg[hubs[2]] == 30 and deg[hubs[0]] == 4 and deg[S] == 4
    # the same with a predicate
    net, S, hubs = _source_and_hubs()
    assert set(net.split_hubs(4, roles=lambda r: r == "H1")) == {hubs[1], S}


def test_cycle_through_the_tree_is_refused_and_leaves_the_netlist_untouched():
    net = Netlist(Params())
    H = net.neuron("H")
    net.synapse(H, H, 10)  # a self-loop H keeps: every copy's input comes from H, which grows past the bound again
    for t in range(6):
        net.synapse(H, net.neuron(f"t{t}"), 10)
    before = (list(net.roles), list(net.src), list(net.dst), list(net.quanta), list(net.delay))
    with pytest.raises(ValueError, match="cycle"):
        net.split_hubs(4)
    assert (net.roles, net.src, net.dst, net.quanta, net.delay) == before
    # a two-neuron cycle X -> Y -> X where each is a hub at the bound
    net = Netlist(Params())
    X, Y = net.neuron("X"), net.neuron("Y")
    net.synapse(X, Y, 10)
    net.synapse(Y, X, 10)
    for t in range(4):
        net.synapse(X, net.neuron(f"x{t}"), 10)
        net.synapse(Y, net.neuron(f"y{t}"), 10)
    with pytest.raises(ValueError, match="cycle"):
        net.split_hubs(4)
    # a self-loop that is handed to the copy converges: the copy feeds itself, the original
    # is fed by the copy, and both still spike exactly as the original did
    net = Netlist(Params())
    H = net.neuron("H")
    for t in range(6):
        net.synapse(H, net.neuron(f"t{t}"), 10)
    net.synapse(H, H, 10)
    copies = net.split_hubs(4)
    assert copies == {H: [7]} and net.out_degrees().max() <= 4
    edges = set(zip(net.src, net.dst))
    assert (7, 0) in edges and (7, 7) in edges and (0, 0) not in edges
    # a feedback source with room to spare converges
    net = Netlist(Params())
    X, Y = net.neuron("X"), net.neuron("Y")
    net.synapse(Y, X, 10)
    for t in range(6):
        net.synapse(X, net.neuron(f"x{t}"), 10)
    copies = net.split_hubs(4)
    assert set(copies) == {X} and net.out_degrees().max() <= 4 and net.out_degrees()[Y] == 2


def test_bad_bound():
    net, _, _ = _source_and_hubs()
    with pytest.raises(ValueError):
        net.split_hubs(0)
