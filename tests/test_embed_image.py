"""Placement -> simulable image on real neuron ids (connectome/embed_image.py).

Fast: the planted latch + edge relay + inhibitor of test_embed_netlist, placed on its synthetic
connectome, turned into an image (every edge carried at the designed quanta, parasitic edges
zeroed) and simulated: one source spike ignites the latch through the relay, the relay fires
once, and the latch holds for a few hundred ms of neural time.

The image inside the whole graph: full_graph_topology on the synthetic connectome (edits in place,
index map, inputs/outputs zeroed), the latch running inside it with the decoys driven, and
simulate_channel through an index map.

Slow (skipped without the MCNS data): the 4-bit adder on the saved best-of-8 placement,
condition A (every missing edge added as Profile 3, parasitic zeroed), 5 additions, must compute;
and that image applied in place on the whole connectome, built but not simulated.
"""

from __future__ import annotations

import numpy as np
import pytest
from test_embed_netlist import latch_relay_netlist, synthetic_mcns

from drosophilos.connectome.embed_h0 import Policy
from drosophilos.connectome.embed_image import (all_missing, build_image, from_sources, none_missing, of_class,
                                                simulate_channel)
from drosophilos.connectome.embed_netlist import Placement, place_netlist
from drosophilos.connectome.mcns_download import DEFAULT_DIR, path_of
from drosophilos.sim.model import QUANTA_PER_SYNAPSE, Params
from drosophilos.sim.ref64 import RefSim


def _placed_latch_relay():
    net, drive, ids = latch_relay_netlist()
    policy = Policy()
    m, idx = synthetic_mcns(policy, drive)
    pl = place_netlist(net, m, policy, time_limit_s=20, restarts=2, verbose=False)
    assert pl.carried == net.nnz == 6, pl.summary(net)
    return net, drive, ids, policy, m, idx, pl


def test_image_carries_every_edge_at_designed_quanta_with_scale():
    net, drive, ids, policy, m, idx, pl = _placed_latch_relay()
    img = build_image(net, m, pl, policy)
    assert img.profile == 2 and not img.profile3 and not img.synthetic
    assert img.counts["carried"] == 6 and img.topology.nnz == 6
    # image index == designed index; hosts and bodyIds recorded
    assert img.real[ids["src"]] == idx["S"] and img.bodies[ids["src"]] == int(m.neurons["bodyId"].iat[idx["S"]])
    # every designed edge present at exactly its quanta, with its scale k = |q| / (count * 16) <= k_max
    counts = {(int(a), int(b)): int(c) for a, b, c in zip(m.pre, m.post, m.count)}
    topo = {(int(s), int(d)): int(q) for s, d, q in zip(img.topology.src, img.topology.dst, img.topology.quanta)}
    for e, (s, d, q) in enumerate(zip(net.src, net.dst, net.quanta)):
        assert topo[(s, d)] == q
        cnt = counts[(pl.mapping[s], pl.mapping[d])]
        assert img.scales[e] == pytest.approx(abs(q) / (cnt * QUANTA_PER_SYNAPSE)) and img.scales[e] <= policy.k_max
        edit = img.carried[e]
        assert edit.new == q and abs(edit.anatomical) == cnt * QUANTA_PER_SYNAPSE and edit.new * edit.anatomical > 0
    assert sum(img.counts["scales"].values()) == 6
    # parasitic: every other anatomical edge among the hosts, zeroed
    hosts = set(pl.mapping.values())
    among = sum(1 for a, b in zip(m.pre, m.post) if a in hosts and b in hosts and a != b)
    assert len(img.parasitic) == among - 6 == img.counts["parasitic_zeroed"]
    assert img.manifest["profile"] == 2 and not img.manifest["structural_edits"]
    assert len(img.manifest["parameter_edits"]) == 6 + len(img.parasitic)
    # kept instead of zeroed: they are in the topology at their anatomical quanta
    img2 = build_image(net, m, pl, Policy(zero_parasitic=False))
    assert img2.topology.nnz == 6 + len(img.parasitic) and img2.counts["parasitic_kept"] == len(img.parasitic)


def test_missing_edges_become_labelled_profile3_edges():
    """Knock the planted I -> E edge out of the mapping's data: the image must add it as a
    Profile 3 edge (labelled) when asked, and leave it out (counted as omitted) when not."""
    net, drive, ids, policy, m, idx, pl = _placed_latch_relay()
    from drosophilos.connectome.mcns import MCNS

    keep = ~((m.pre == idx["I"]) & (m.post == idx["E"]))
    m2 = MCNS(m.neurons, m.pre[keep], m.post[keep], m.count[keep], min_syn=1)
    pl2 = Placement(dict(pl.mapping), [], 0, [], 0, 0.0)  # the same hosts, audited by the image itself
    img = build_image(net, m2, pl2, policy)
    assert img.profile == 3 and img.counts["carried"] == 5 and len(img.profile3) == 1
    e = img.profile3[0]
    assert (e.src, e.dst) == (ids["edge_inh"], ids["edge"]) and e.cls == "edge_inh -> edge" and "no anatomical edge" in e.reason
    assert e.quanta == net.quanta[net.src.index(ids["edge_inh"])]
    assert img.manifest["structural_edits"][0]["post_role"] == net.roles[ids["edge"]]
    assert img.counts["profile3_by_class"] == {"edge_inh -> edge": 1}
    img_b = build_image(net, m2, pl2, policy, profile3=none_missing)
    assert img_b.topology.nnz == 5 and img_b.counts["omitted_missing_edges"] == 1 and not img_b.profile3
    assert build_image(net, m2, pl2, policy, profile3=of_class("edge_inh -> edge")).topology.nnz == 6
    assert build_image(net, m2, pl2, policy, profile3=from_sources([net.roles[ids["src"]]])).topology.nnz == 5
    # an unplaced neuron becomes a synthetic (Profile 3) neuron; its edges are Profile 3
    pl3 = Placement({k: v for k, v in pl.mapping.items() if k != ids["edge_inh"]}, [ids["edge_inh"]], 0, [], 0, 0.0)
    img3 = build_image(net, m, pl3, policy, profile3=all_missing)
    assert img3.synthetic == [ids["edge_inh"]] and img3.bodies[ids["edge_inh"]] == -1
    assert len(img3.profile3) == 2 and all("endpoint unplaced" in e.reason for e in img3.profile3)
    assert img3.topology.nnz == 6


def test_latch_ignites_through_relay_and_holds_on_the_image():
    net, drive, ids, policy, m, idx, pl = _placed_latch_relay()
    img = build_image(net, m, pl, policy)
    params = Params()
    sim = RefSim(img.topology, params)
    t_src = 100
    sim.add_events(0, [t_src], [ids["src"]], [drive.ignite])
    n_steps = 4000  # 400 ms
    sim.run(n_steps)
    tr = sim.trace
    u, v, edge = (tr.neuron_steps(ids[k]) for k in ("u", "v", "edge"))
    assert len(edge) == 1 and edge[0] > t_src  # the relay fires exactly once (its inhibitor holds it)
    assert len(u) >= 40 and len(v) >= 40  # a running latch: ~200 Hz on each member
    assert u[0] > edge[0] and max(u.max(), v.max()) >= n_steps - 2 * drive.loop_period_steps  # ignited by the relay, still running at the end
    periods = np.diff(np.sort(np.concatenate([u, v])))
    assert 0.5 * drive.loop_period_steps <= np.median(periods) <= drive.loop_period_steps
    assert len(tr.neuron_steps(ids["src"])) == 1


@pytest.mark.skipif(not path_of("weights", DEFAULT_DIR).exists(), reason="MCNS data not downloaded")
def test_placed_adder_condition_A_computes_on_mcns():
    from pathlib import Path

    from drosophilos.connectome.embed_image import adder_cases, load_mapping
    from drosophilos.connectome.mcns import load_mcns
    from drosophilos.lib.adder import build_adder_channel

    params = Params()
    ch = build_adder_channel(params, 4, ordered=True, watchdog_hops=100)
    m = load_mcns()
    mapping = Path(__file__).resolve().parents[1] / "docs" / "h1_placement_mapping.json"
    if mapping.exists():
        pl = load_mapping(mapping, ch.net, m)
    else:
        pl = place_netlist(ch.net, m, time_limit_s=120, restarts=1)
    img = build_image(ch.net, m, pl, Policy(), profile3=all_missing)
    assert img.counts["carried"] + img.counts["profile3_edges"] == ch.net.nnz and img.counts["omitted_missing_edges"] == 0
    assert img.counts["carried"] >= 600, img.counts  # the H1 placement: 704 of 1,146
    cases, words, expected = adder_cases(4, 5, seed=0)
    res = simulate_channel(ch, img, words, expected, params, fresh_each=False)
    assert res["correct"] == 5 and res["faults"] == 0 and res["timeouts"] == 0, {k: v for k, v in res.items() if k != "records"}


# ----------------------------------------------------------------------------------------
# the image inside the whole graph (embed_image.full_graph_topology, CircuitView)
# ----------------------------------------------------------------------------------------
def _run_latch_in_full_graph(fg, ids, drive, decoys, params, rate_hz, seed=0, n_steps=4000, t_src=100):
    """Ignite the planted latch through the relay while the decoy neurons fire from a Poisson drive."""
    from drosophilos.connectome.embed_image import circuit_sim, surround_events, underlying

    sim = circuit_sim(fg.topology, params, fg.index_map)
    steps, neurons, quanta = surround_events("poisson", decoys, params, 0, n_steps, np.random.default_rng(seed), rate_hz=rate_hz)
    underlying(sim).add_events(0, steps, neurons, quanta)
    sim.add_events(0, [t_src], [ids["src"]], [drive.ignite])  # designed index: the view maps it to the host
    sim.run(n_steps)
    return sim


def test_full_graph_topology_embeds_the_image_and_the_index_map_round_trips():
    from drosophilos.connectome.embed_image import CircuitView, full_graph_topology

    net, drive, ids, policy, m, idx, pl = _placed_latch_relay()
    params = Params()
    img = build_image(net, m, pl, policy)
    fg = full_graph_topology(m, params, img)
    base = m.topology(params)
    # index map: designed -> the host chosen by the placement; every real neuron keeps its index
    assert fg.topology.n == m.n and fg.counts["synthetic_neurons"] == 0
    assert fg.index_map[ids["src"]] == idx["S"] and fg.index_map[ids["edge"]] == idx["E"]
    assert all(fg.index_map[d] == r for d, r in pl.mapping.items())
    inv = fg.designed_of()
    assert all(inv[fg.index_map[d]] == d for d in range(net.n)) and (inv >= 0).sum() == net.n
    view = CircuitView(RefSim(fg.topology, params), fg.index_map)
    assert np.array_equal(view.inverse, inv)
    # the edits, in place: carried edges at the designed quanta, parasitic edges zeroed, everything else untouched
    full = {(int(s), int(d)): int(q) for s, d, q in zip(fg.topology.src, fg.topology.dst, fg.topology.quanta)}
    anat = {(int(s), int(d)): int(q) for s, d, q in zip(base.src, base.dst, base.quanta)}
    for s, d, q in zip(net.src, net.dst, net.quanta):
        assert full[(pl.mapping[s], pl.mapping[d])] == q
    for e in img.parasitic:
        assert full[(pl.mapping[e.src], pl.mapping[e.dst])] == 0 and anat[(pl.mapping[e.src], pl.mapping[e.dst])] == e.quanta
    edited = {(pl.mapping[s], pl.mapping[d]) for s, d in zip(net.src, net.dst)} | {(pl.mapping[e.src], pl.mapping[e.dst]) for e in img.parasitic}
    assert all(full[k] == v for k, v in anat.items() if k not in edited)
    assert fg.counts["carried_edges_rescaled"] == 6 and fg.counts["parasitic_zeroed"] == len(img.parasitic) == img.counts["parasitic_zeroed"]
    assert fg.counts["profile3_added"] == 0 and fg.counts["topology_edges"] == base.nnz
    # outputs / inputs zeroed: exactly the anatomical edges across the boundary, and only those
    hosts = set(pl.mapping.values())
    n_out = sum(1 for a, b in zip(m.pre, m.post) if a in hosts and b not in hosts)
    n_in = sum(1 for a, b in zip(m.pre, m.post) if a not in hosts and b in hosts)
    fg_o = full_graph_topology(m, params, img, zero_outputs=True, base=base)
    fg_i = full_graph_topology(m, params, img, zero_inputs=True, base=base)
    assert fg_o.counts["outputs_zeroed"] == n_out and fg_i.counts["inputs_zeroed"] == n_in
    assert fg_o.counts["nonzero_edges"] == fg.counts["nonzero_edges"] - n_out
    assert fg_i.counts["nonzero_edges"] == fg.counts["nonzero_edges"] - n_in
    # a Profile 3 edge and a synthetic neuron: the unplaced inhibitor is appended after the m.n real neurons
    pl3 = Placement({k: v for k, v in pl.mapping.items() if k != ids["edge_inh"]}, [ids["edge_inh"]], 0, [], 0, 0.0)
    img3 = build_image(net, m, pl3, policy, profile3=all_missing)
    fg3 = full_graph_topology(m, params, img3, base=base)
    assert fg3.topology.n == m.n + 1 and fg3.index_map[ids["edge_inh"]] == m.n and fg3.counts["profile3_added"] == 2
    assert fg3.counts["topology_edges"] == base.nnz + 2
    full3 = {(int(s), int(d)): int(q) for s, d, q in zip(fg3.topology.src, fg3.topology.dst, fg3.topology.quanta)}
    assert full3[(m.n, pl.mapping[ids["edge"]])] == net.quanta[net.src.index(ids["edge_inh"])]


def test_latch_ignites_and_holds_inside_the_synthetic_full_graph():
    """The planted latch + relay inside the synthetic connectome with the decoy neurons active: it
    ignites and holds when the parasitic edges are zeroed (the anatomy's weak stray edges into the
    hosts are what the surround delivers); with the inputs zeroed the circuit's spikes are the
    isolated run's, spike for spike, and the boundary envelope is empty; with them live the
    envelope counts the decoys' arrivals by sign."""
    from drosophilos.connectome.embed_image import boundary_envelope_ms, full_graph_topology, underlying

    net, drive, ids, policy, m, idx, pl = _placed_latch_relay()
    params = Params()
    img = build_image(net, m, pl, policy)
    base = m.topology(params)
    hosts = np.array(sorted(pl.mapping.values()))
    decoys = np.setdiff1d(np.arange(m.n), hosts)
    n_steps, t_src, rate = 4000, 100, 50.0
    # isolated reference: the same drive on the image alone
    iso = RefSim(img.topology, params)
    iso.add_events(0, [t_src], [ids["src"]], [drive.ignite])
    iso.run(n_steps)
    ref = iso.trace

    def check_latch(tr):
        u, v, edge = (tr.neuron_steps(ids[k]) for k in ("u", "v", "edge"))
        assert len(edge) == 1 and edge[0] > t_src
        assert len(u) >= 40 and len(v) >= 40 and u[0] > edge[0]
        assert max(u.max(), v.max()) >= n_steps - 2 * drive.loop_period_steps
        assert len(tr.neuron_steps(ids["src"])) == 1

    check_latch(ref)
    # inputs zeroed: the isolated case embedded; identical spikes, nothing crosses the boundary
    fg_i = full_graph_topology(m, params, img, zero_inputs=True, base=base)
    sim = _run_latch_in_full_graph(fg_i, ids, drive, decoys, params, rate, n_steps=n_steps, t_src=t_src)
    assert sim.trace == ref
    whole = underlying(sim).trace
    assert len(whole) > len(ref)  # the decoys did fire
    assert np.isin(whole.events["neuron"], hosts).sum() == len(ref)
    env = boundary_envelope_ms(fg_i.topology, whole, fg_i.index_map, 0, n_steps, params)
    assert env["surround_spikes"] == len(whole) - len(ref) and env["exc_spikes_in"] == 0 and env["inh_spikes_in"] == 0
    # surround live: the weak anatomical edges into the hosts deliver the decoys' spikes; the latch still holds
    fg = full_graph_topology(m, params, img, base=base)
    sim = _run_latch_in_full_graph(fg, ids, drive, decoys, params, rate, n_steps=n_steps, t_src=t_src)
    check_latch(sim.trace)
    whole = underlying(sim).trace
    env = boundary_envelope_ms(fg.topology, whole, fg.index_map, 0, n_steps, params)
    assert env["exc_spikes_in"] + env["inh_spikes_in"] > 0
    assert len(env["exc_per_ms"]) == int(n_steps * params.dt) and sum(env["exc_per_ms"]) == env["exc_spikes_in"]
    assert env["max_exc_quanta_per_ms"] >= 0 >= env["max_inh_quanta_per_ms"]
    # by hand: every decoy spike through each non-zero synapse onto a host, split by the sign of its quanta
    ev = whole.events
    q = {(int(s), int(d)): int(w) for s, d, w in zip(fg.topology.src, fg.topology.dst, fg.topology.quanta)}
    exc = inh = 0
    for j in ev["neuron"][~np.isin(ev["neuron"], hosts)].tolist():
        for h in hosts.tolist():
            w = q.get((j, h), 0)
            exc += w > 0
            inh += w < 0
    assert (env["exc_spikes_in"], env["inh_spikes_in"]) == (exc, inh)
    # outputs zeroed: the surround no longer hears the circuit; the circuit is unchanged by that
    fg_o = full_graph_topology(m, params, img, zero_outputs=True, base=base)
    sim_o = _run_latch_in_full_graph(fg_o, ids, drive, decoys, params, rate, n_steps=n_steps, t_src=t_src)
    check_latch(sim_o.trace)


def test_simulate_channel_takes_an_index_map():
    """simulate_channel on a permuted copy of the isolated topology through its index map gives the
    isolated result; the isolated path (no map, or the identity) is the RefSim itself."""
    from drosophilos.connectome.embed_image import CircuitView, circuit_sim
    from drosophilos.lib.adder import build_adder_channel
    from drosophilos.connectome.embed_image import adder_cases, simulate_channel

    params = Params()
    ch = build_adder_channel(params, 2, ordered=True, watchdog_hops=100)
    net = ch.net
    topo = net.topology()
    assert isinstance(circuit_sim(topo, params), RefSim) and isinstance(circuit_sim(topo, params, np.arange(net.n)), RefSim)
    # the same circuit at shuffled indices, with 5 idle extra neurons
    rng = np.random.default_rng(1)
    index_map = rng.permutation(net.n + 5)[: net.n]
    big = topo.__class__.from_edges(net.n + 5, index_map[topo.src], index_map[topo.dst], topo.quanta, topo.delay)
    assert isinstance(circuit_sim(big, params, index_map), CircuitView)
    from drosophilos.connectome.embed_image import Image

    img = Image(net.n, topo, np.arange(net.n), np.arange(net.n), [], [], {}, [], [], [], True)
    cases, words, expected = adder_cases(2, 3, seed=0)
    a = simulate_channel(ch, img, words, expected, params, fresh_each=True)
    b = simulate_channel(ch, img, words, expected, params, fresh_each=True, topology=big, index_map=index_map)
    for k in ("correct", "accept_ms", "cycle_ms", "total_spikes", "statuses"):
        assert a[k] == b[k], k
    assert a["correct"] == 3 and a["reached"] == b["reached"]


@pytest.mark.skipif(not path_of("weights", DEFAULT_DIR).exists(), reason="MCNS data not downloaded")
def test_full_graph_topology_builds_on_mcns():
    """The placed adder's condition-A image applied in place on the whole connectome: built, not
    simulated (166,700 neurons; bench/h1_fullgraph.py runs it on a cluster node)."""
    from pathlib import Path

    from drosophilos.connectome.embed_image import full_graph_topology, load_mapping
    from drosophilos.connectome.mcns import load_mcns
    from drosophilos.lib.adder import build_adder_channel

    mapping = Path(__file__).resolve().parents[1] / "docs" / "h1_placement_mapping.json"
    if not mapping.exists():
        pytest.skip("no saved placement")
    params = Params()
    ch = build_adder_channel(params, 4, ordered=True, watchdog_hops=100)
    m = load_mcns()
    pl = load_mapping(mapping, ch.net, m)
    img = build_image(ch.net, m, pl, Policy(), profile3=all_missing)
    base = m.topology(params)
    fg = full_graph_topology(m, params, img, zero_outputs=True, base=base)
    c = fg.counts
    assert c["n"] == m.n + len(img.synthetic) and c["synthetic_neurons"] == len(img.synthetic) == 2
    assert c["carried_synapses"] == img.counts["carried"] and c["profile3_added"] == img.counts["profile3_edges"]
    assert c["parasitic_zeroed"] == img.counts["parasitic_zeroed"] and c["outputs_zeroed"] > 0 and c["inputs_zeroed"] == 0
    assert c["topology_edges"] == base.nnz + c["profile3_added"]
    # the index map: every placed designed neuron on its bodyId's index, the synthetic ones after m.n
    for d, r in pl.mapping.items():
        assert fg.index_map[d] == r
    assert sorted(fg.index_map[img.synthetic].tolist()) == [m.n, m.n + 1]
    # spot-check: a carried edge sits at its designed quanta on the hosts' anatomical edge
    s, d, q, _ = img.carried_syn[0]
    lo, hi = fg.topology.indptr[fg.index_map[s]], fg.topology.indptr[fg.index_map[s] + 1]
    k = lo + np.searchsorted(fg.topology.dst[lo:hi], fg.index_map[d])
    dup = sum(qq for ss, dd, qq, _ in img.carried_syn if (ss, dd) == (s, d))
    assert fg.topology.dst[k] == fg.index_map[d] and fg.topology.quanta[k] == dup
