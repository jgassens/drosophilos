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
from test_embed_netlist import (latch_relay_netlist, synthetic_autapses, synthetic_ff_mcns, synthetic_mcns, synthetic_triple_mcns, two_flipflop_netlist,
                                two_proxied_flipflop_netlist)

from drosophilos.connectome.embed_h0 import Policy
from drosophilos.connectome.embed_image import (all_missing, build_image, from_sources, none_missing, of_class,
                                                simulate_channel)
from drosophilos.connectome.embed_netlist import Placement, place_netlist
from drosophilos.connectome.mcns_download import DEFAULT_DIR, path_of
from drosophilos.protocol.flipflop import BIAS_213HZ_MV
from drosophilos.sim.model import QUANTA_PER_SYNAPSE, Params
from drosophilos.sim.ref64 import RefSim


def _placed_latch_relay():
    net, drive, ids = latch_relay_netlist()
    policy = Policy()
    m, idx = synthetic_mcns(policy, drive)
    pl = place_netlist(net, m, policy, time_limit_s=20, restarts=2, verbose=False)
    assert pl.carried == net.nnz == 6, pl.summary(net)
    return net, drive, ids, policy, m, idx, pl


def test_full_graph_defaults_frozen_before_surround_calibration():
    """Digest captured from pre-change full_graph_topology, including legacy counts JSON."""
    import hashlib
    import json
    from drosophilos.connectome.embed_image import full_graph_topology

    net, _, _, policy, m, _, real = synthetic_autapses(False)
    pl = Placement(dict(enumerate(real.tolist())), [], 0, [], 0, 0)
    image = build_image(net, m, pl, policy)
    for kwargs in ({}, {"surround_scale_exc": 1, "surround_scale_inh": 1, "surround_input_cap": None}):
        fg = full_graph_topology(m, Params(), image, **kwargs)
        digest = hashlib.sha256()
        for arr in (fg.topology.src, fg.topology.dst, fg.topology.quanta, fg.topology.delay,
                    fg.topology.indptr, fg.index_map):
            digest.update(arr.astype(arr.dtype.newbyteorder("<")).tobytes())
        digest.update(b"none" if fg.topology.bias is None else fg.topology.bias.astype("<f8").tobytes())
        digest.update(json.dumps(fg.counts, sort_keys=True, separators=(",", ":")).encode())
        assert digest.hexdigest() == "f9e2d09cd9a66ea6591763e3a0d4d06f81165029b5824ad28f48c114eb0b34f4"


def _surround_fixture():
    import pandas as pd
    from drosophilos.connectome.mcns import MCNS
    from drosophilos.lib.netlist import Netlist

    neurons = pd.DataFrame({"bodyId": np.arange(10, 16), "type": "test", "superclass": "central",
                            "nt": ["acetylcholine", "gaba", "acetylcholine", "histamine",
                                   "acetylcholine", "glutamate"],
                            "sign": np.array([1, -1, 1, -1, 1, -1], np.int8)})
    pre = np.array([0, 1, 0, 2, 2, 3, 4, 4, 5], np.int32)
    post = np.array([1, 0, 2, 0, 3, 2, 0, 3, 0], np.int32)
    count = np.array([3, 1, 4, 2, 6, 3, 6, 2, 1], np.int32)
    m = MCNS(neurons, pre, post, count, min_syn=1)
    net = Netlist(Params())
    for role in ("host_e", "host_i", "synthetic"):
        net.neuron(role)
    net.synapse(0, 1, 32, 5)  # carried
    net.synapse(1, 2, -7, 3)  # Profile 3 to a synthetic neuron
    net.synapse(2, 0, 9, 2)  # Profile 3 from a synthetic neuron
    image = build_image(net, m, Placement({0: 0, 1: 1}, [2], 0, [], 0, 0), Policy(zero_parasitic=False))
    return m, image


def _edge_weights(topo):
    return {(int(s), int(d)): int(q) for s, d, q in zip(topo.src, topo.dst, topo.quanta)}


def test_surround_gains_cap_and_image_exclusion():
    from drosophilos.connectome.embed_image import full_graph_topology

    m, image = _surround_fixture()
    base = m.topology()
    frozen_base = base.quanta.copy()
    before = full_graph_topology(m, Params(), image, base=base)
    fg = full_graph_topology(m, Params(), image, base=base, surround_scale_exc=.5,
                             surround_scale_inh=2, surround_input_cap=2)
    q = _edge_weights(fg.topology)
    # Two eligible excitatory inputs to host 0: 2*.5 + 6*.5 = 4, capped to 2 synapse weights.
    assert q[2, 0] == 8 and q[4, 0] == 24
    # The same cap includes outgoing host edges to surround and surround-to-surround anatomy.
    assert q[0, 2] == 32 and q[2, 3] == 24 and q[4, 3] == 8
    assert q[3, 2] == -96 and q[5, 0] == -32
    # Carried, kept parasitic, and Profile 3 edges (including synthetic endpoints) are immutable.
    old = _edge_weights(before.topology)
    for pair in ((0, 1), (1, 0), (1, 6), (6, 0)):
        assert q[pair] == old[pair]
    assert fg.topology.delay.tolist() == before.topology.delay.tolist()
    assert np.array_equal(base.quanta, frozen_base)
    assert image.carried_syn == [(0, 1, 32, 5)]
    assert fg.counts["surround_capped_neurons"] == 2
    assert fg.counts["surround_eligible_edges"] == 7
    assert fg.counts["surround_scale_exc"] == .5 and fg.counts["surround_scale_inh"] == 2
    # A boundary cut is applied AFTER scaling and still ranks original anatomical counts.
    cut = full_graph_topology(m, Params(), image, base=base, surround_scale_exc=.5,
                              surround_input_cap=2, cut_inputs_frac=.2)
    assert _edge_weights(cut.topology)[4, 0] == 0
    assert _edge_weights(cut.topology)[2, 0] == 8


def test_surround_rounds_once_ties_to_even_and_reports_zeroes():
    from drosophilos.connectome.embed_image import full_graph_topology

    m, image = _surround_fixture()
    fg = full_graph_topology(m, Params(), image, surround_scale_exc=1 / 64, surround_scale_inh=1 / 32)
    q = _edge_weights(fg.topology)
    assert q[0, 2] == 1  # 64 / 64
    assert q[2, 0] == 0  # +.5 -> 0
    assert q[4, 0] == 2  # +1.5 -> 2
    assert q[5, 0] == 0  # -.5 -> 0
    assert q[3, 2] == -2  # -1.5 -> -2
    assert q[4, 3] == 0  # another +.5 -> 0
    assert fg.counts["surround_zeroed_by_rounding"] == 3
    zero = full_graph_topology(m, Params(), image, surround_scale_exc=0, surround_scale_inh=0)
    assert zero.counts["surround_zeroed_edges"] == 7
    assert zero.counts["surround_zeroed_by_rounding"] == 0
    # Gain gives .75 and 2.25 quanta into host 0; cap at 2 quanta gives .5, 1.5 -> 0, 2.
    # Rounding gains first would instead cap [1, 2] to [2/3, 4/3] and wrongly yield [1, 1].
    capped = full_graph_topology(m, Params(), image, surround_scale_exc=3 / 128, surround_input_cap=1 / 8)
    assert _edge_weights(capped.topology)[2, 0] == 0
    assert _edge_weights(capped.topology)[4, 0] == 2


@pytest.mark.parametrize("kwargs", [{"surround_scale_exc": -1}, {"surround_scale_inh": np.nan},
                                     {"surround_input_cap": np.inf}, {"surround_input_cap": -1},
                                     {"surround_scale_exc": 1e300}])
def test_surround_rejects_invalid_settings_and_overflow(kwargs):
    from drosophilos.connectome.embed_image import full_graph_topology

    m, image = _surround_fixture()
    with pytest.raises(ValueError, match="surround"):
        full_graph_topology(m, Params(), image, **kwargs)


@pytest.mark.parametrize("too_weak", [False, True])
def test_autapses_in_conditions_and_whole_graph(too_weak):
    from drosophilos.connectome.embed_image import full_graph_topology, h1_conditions

    net, drive, ids, policy, m, idx, real = synthetic_autapses(too_weak)
    pl = Placement(dict(enumerate(real.tolist())), [], 0, [], 0, 0)
    conditions = h1_conditions(policy)
    a = build_image(net, m, pl, conditions["A"][2], profile3=conditions["A"][1])
    b = build_image(net, m, pl, conditions["B"][2], profile3=conditions["B"][1])
    assert a.counts["carried"] == 6 + (not too_weak)
    classes = {"autapse (wrong sign)": 1, "autapse (no host self-edge)": 1}
    if too_weak:
        classes["autapse (too weak)"] = 1
    assert a.counts["profile3_by_class"] == b.counts["omitted_by_class"] == classes
    assert all(e.src == e.dst and e.delay == 0 and e.quanta == -round(.2 * drive.loop) for e in a.profile3)
    assert all(e["delay"] == 0 for e in a.manifest["structural_edits"])
    assert not b.profile3
    for img in (a, b):
        # Existing host self-edges that cannot carry feedback are zeroed before Profile 3 is added.
        assert len([e for e in img.parasitic if e.src == e.dst]) == 1 + too_weak
        fg = full_graph_topology(m, Params(), img)
        for name in ("edge_inh", "edge", "v"):
            s = ids[name]
            present = img is a or (name == "edge_inh" and not too_weak)
            take = (img.topology.src == s) & (img.topology.dst == s)
            assert take.sum() == int(present)
            if present:
                assert img.topology.delay[take].tolist() == [0]
                # The designed delay reaches RefSim unchanged, too.
                assert RefSim(img.topology, Params()).topo.delay[take].tolist() == [0]
            host = fg.index_map[s]
            full = (fg.topology.src == host) & (fg.topology.dst == host) & (fg.topology.quanta != 0)
            assert full.sum() == int(present)
            if present:
                assert fg.topology.quanta[full].tolist() == [-round(.2 * drive.loop)]
                assert fg.topology.delay[full].tolist() == [0]


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


def test_image_carries_the_flipflop_biases():
    """Two flip-flops joined by an edge relay, placed on the synthetic connectome: the image's
    topology carries the designed biases at the image indices, the manifest lists one "bias"
    parameter edit per biased host (counted as `biases`), and full_graph_topology puts each bias
    on the host's full-graph index (a synthetic neuron keeps its bias after the m.n real ones)."""
    from drosophilos.connectome.embed_image import full_graph_topology

    net, drive, ids = two_flipflop_netlist()
    policy = Policy()
    m, idx = synthetic_ff_mcns(policy, drive)
    pl = place_netlist(net, m, policy, time_limit_s=20, restarts=2, verbose=False)
    assert pl.carried == 6 and not pl.unplaced
    img = build_image(net, m, pl, policy)
    members = [ids["u0"], ids["v0"], ids["u1"], ids["v1"]]
    assert img.topology.bias is not None and img.topology.bias.shape == (net.n,)
    assert all(img.topology.bias[d] == BIAS_213HZ_MV for d in members) and img.topology.bias[ids["edge"]] == 0.0
    assert sorted(img.topology.biased_neurons().tolist()) == sorted(members)
    assert img.counts["biases"] == 4 and img.counts["biases_synthetic"] == 0 and len(img.biases) == 4
    edits = [e for e in img.manifest["parameter_edits"] if e["kind"] == "bias"]
    assert len(edits) == 4 and {e["post"] for e in edits} == {int(img.bodies[d]) for d in members}
    assert all(e["pre"] is None and e["anatomical"] == 0.0 and e["new"] == BIAS_213HZ_MV for e in edits)
    assert len(img.manifest["parameter_edits"]) == 6 + 4 + len(img.parasitic)
    # the two wrong-sign readout edges are Profile 3 here (added), so the image is Profile 3; without them, Profile 2 with the biases
    assert img.profile == 3 and len(img.profile3) == 2 and all("wrong sign" in e.reason for e in img.profile3)
    img2 = build_image(net, m, pl, policy, profile3=none_missing)
    assert img2.profile == 2 and img2.manifest["profile"] == 2 and img2.counts["biases"] == 4 and img2.topology.bias is not None
    # the whole graph: biases at the hosts' indices, nowhere else
    params = Params()
    base = m.topology(params)
    assert base.bias is None
    fg = full_graph_topology(m, params, img2, base=base)
    assert fg.topology.bias is not None and fg.counts["biases"] == 4 and fg.counts["biases_synthetic"] == 0
    assert sorted(fg.topology.biased_neurons().tolist()) == sorted(pl.mapping[d] for d in members)
    assert all(fg.topology.bias[pl.mapping[d]] == BIAS_213HZ_MV for d in members)
    # one member unplaced: a synthetic neuron after the m.n real ones, its bias with it
    pl3 = Placement({k: v for k, v in pl.mapping.items() if k != ids["v1"]}, [ids["v1"]], 0, [], 0, 0.0)
    img3 = build_image(net, m, pl3, policy, profile3=all_missing)
    assert img3.counts["biases"] == 3 and img3.counts["biases_synthetic"] == 1 and len(img3.biases) == 3
    fg3 = full_graph_topology(m, params, img3, base=base)
    assert fg3.topology.n == m.n + 1 and fg3.topology.bias[m.n] == BIAS_213HZ_MV and fg3.counts == {**fg3.counts, "biases": 3, "biases_synthetic": 1}
    # a netlist without biases leaves every topology's bias None, as before
    net_l, _, ids_l, policy_l, m_l, _, pl_l = _placed_latch_relay()
    img_l = build_image(net_l, m_l, pl_l, policy_l)
    assert img_l.topology.bias is None and img_l.counts["biases"] == 0 and not img_l.biases
    assert full_graph_topology(m_l, params, img_l).topology.bias is None


def test_image_carries_the_proxy_biases_onto_their_hosts():
    """Two flip-flops with proxies placed as triples: the proxies' biases (the same 58 mV as the
    members') land on the proxy hosts -- six bias edits, one per biased host, the p hosts among
    them -- and the image is Profile 2 with every one of the 15 edges carried (the readout leaves
    p, so nothing is wrong-sign). full_graph_topology puts each bias on the host's index."""
    from drosophilos.connectome.embed_image import full_graph_topology

    net, drive, ids = two_proxied_flipflop_netlist()
    policy = Policy()
    m, idx = synthetic_triple_mcns(policy, drive)
    pl = place_netlist(net, m, policy, time_limit_s=30, restarts=2, verbose=False)
    assert pl.carried == net.nnz == 15 and not pl.unplaced
    img = build_image(net, m, pl, policy)
    biased = [ids[k] for k in ("u0", "v0", "p0", "u1", "v1", "p1")]
    assert sorted(img.topology.biased_neurons().tolist()) == sorted(biased)
    assert all(img.topology.bias[d] == BIAS_213HZ_MV for d in biased) and img.topology.bias[ids["edge"]] == img.topology.bias[ids["set"]] == 0.0
    assert img.counts["biases"] == 6 and img.counts["biases_synthetic"] == 0 and len(img.biases) == 6
    edits = {e["post"]: e for e in img.manifest["parameter_edits"] if e["kind"] == "bias"}
    for k in ("p0", "p1"):
        e = edits[int(img.bodies[ids[k]])]
        assert e["pre"] is None and e["anatomical"] == 0.0 and e["new"] == BIAS_213HZ_MV and e["reason"].startswith(f"F{k[1]}.p bias")
    assert img.profile == 2 and img.counts["carried"] == 15 and not img.profile3
    assert img.real[ids["p0"]] == idx["P1"] and img.real[ids["p1"]] == idx["P2"]
    params = Params()
    fg = full_graph_topology(m, params, img, base=m.topology(params))
    assert fg.counts["biases"] == 6 and fg.counts["biases_synthetic"] == 0
    assert sorted(fg.topology.biased_neurons().tolist()) == sorted(pl.mapping[d] for d in biased)
    assert fg.topology.bias[idx["P1"]] == fg.topology.bias[idx["P2"]] == BIAS_213HZ_MV and fg.topology.bias[idx["E"]] == 0.0


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
    assert img.counts["carried"] + img.counts["profile3_edges"] == img.counts["designed_pairs"] and img.counts["omitted_missing_edges"] == 0  # per distinct (src, dst) pair: the adder has 1,146 entries, 1,144 pairs
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


def test_full_graph_partial_boundary_cuts_are_strongest_first_and_leave_designed_edges_alone():
    """The cut ladder works entirely on a small synthetic MCNS graph, including its tie rule."""
    from drosophilos.connectome.embed_image import full_graph_topology
    from drosophilos.connectome.mcns import MCNS

    net, _, ids, policy, m, _, pl = _placed_latch_relay()
    hosts = set(pl.mapping.values())
    outside = [i for i in range(m.n) if i not in hosts]
    pairs = set(zip(m.pre.tolist(), m.post.tolist()))

    def available(pre, post_candidates):
        found = []
        for post in post_candidates:
            if (pre, post) not in pairs:
                found.append((pre, post))
                pairs.add((pre, post))
        return found

    # These large tied groups dominate the random weak edges.  Each is bigger than half of
    # its boundary, so f=.5 cuts a deterministic *prefix* of equal-count edges by (post, pre).
    out_pairs = available(pl.mapping[ids["src"]], outside)
    in_pairs = [(pre, pl.mapping[ids["u"]]) for pre in outside if (pre, pl.mapping[ids["u"]]) not in pairs]
    pairs.update(in_pairs)
    assert out_pairs and in_pairs
    extra_pairs = out_pairs + in_pairs
    extra_counts = [100_000] * len(out_pairs) + [90_000] * len(in_pairs)
    m2 = MCNS(m.neurons, np.append(m.pre, [a for a, _ in extra_pairs]).astype(np.int32),
              np.append(m.post, [b for _, b in extra_pairs]).astype(np.int32),
              np.append(m.count, extra_counts).astype(np.int32), min_syn=1)
    img = build_image(net, m2, pl, policy)
    params, base = Params(), m2.topology(Params())
    plain = full_graph_topology(m2, params, img, base=base)
    zero_o = full_graph_topology(m2, params, img, zero_outputs=True, base=base)
    zero_i = full_graph_topology(m2, params, img, zero_inputs=True, base=base)
    cut0 = full_graph_topology(m2, params, img, cut_outputs_frac=0, cut_inputs_frac=0, base=base)
    cut_o = full_graph_topology(m2, params, img, cut_outputs_frac=.5, base=base)
    cut_i = full_graph_topology(m2, params, img, cut_inputs_frac=.5, base=base)
    cut1_o = full_graph_topology(m2, params, img, cut_outputs_frac=1, base=base)
    cut1_i = full_graph_topology(m2, params, img, cut_inputs_frac=1, base=base)

    assert np.array_equal(cut0.topology.quanta, plain.topology.quanta)  # f=0 is today
    assert np.array_equal(cut1_o.topology.quanta, zero_o.topology.quanta)  # f=1 is zero_outputs
    assert np.array_equal(cut1_i.topology.quanta, zero_i.topology.quanta)  # f=1 is zero_inputs

    anatomy = {(int(a), int(b)): int(c) for a, b, c in zip(m2.pre, m2.post, m2.count)}

    def expected(direction):
        if direction == "out":
            edges = [j for j, (a, b) in enumerate(zip(base.src, base.dst)) if a in hosts and b not in hosts]
        else:
            edges = [j for j, (a, b) in enumerate(zip(base.src, base.dst)) if a not in hosts and b in hosts]
        ranked = sorted(edges, key=lambda j: (-anatomy[int(base.src[j]), int(base.dst[j])], int(base.dst[j]), int(base.src[j])))
        return ranked, ranked[: (len(ranked) + 1) // 2]

    for direction, cut in (("out", cut_o), ("in", cut_i)):
        boundary, expected_cut = expected(direction)
        actual_cut = [j for j in boundary if cut.topology.quanta[j] == 0 and plain.topology.quanta[j] != 0]
        assert set(actual_cut) == set(expected_cut)
        prefix = "outputs" if direction == "out" else "inputs"
        assert cut.counts[f"{prefix}_cut_edges"] == len(expected_cut)
        assert cut.counts[f"{prefix}_cut_synapses"] == sum(anatomy[int(base.src[j]), int(base.dst[j])] for j in expected_cut)
        assert cut.counts[f"{prefix}_remaining_edges"] == len(boundary) - len(expected_cut)
        assert cut.counts[f"{prefix}_remaining_synapses"] == sum(
            anatomy[int(base.src[j]), int(base.dst[j])] for j in boundary if j not in expected_cut)
        assert cut.counts[f"{prefix}_weakest_cut_synapses"] == min(anatomy[int(base.src[j]), int(base.dst[j])] for j in expected_cut)
        tied = out_pairs if direction == "out" else in_pairs
        assert 0 < sum((int(base.src[j]), int(base.dst[j])) in tied for j in expected_cut) < len(tied)

    # Designed host-to-host image edges are outside both boundary selections.
    for s, d, q in zip(net.src, net.dst, net.quanta):
        pair = (pl.mapping[s], pl.mapping[d])
        j = np.flatnonzero((base.src == pair[0]) & (base.dst == pair[1]))
        assert len(j) == 1
        assert cut_o.topology.quanta[j[0]] == cut_i.topology.quanta[j[0]] == plain.topology.quanta[j[0]] == q
    with pytest.raises(ValueError, match="cannot be combined"):
        full_graph_topology(m2, params, img, zero_outputs=True, cut_outputs_frac=.5, base=base)
    with pytest.raises(ValueError, match="must be a finite float"):
        full_graph_topology(m2, params, img, cut_inputs_frac=1.1, base=base)


def test_full_graph_cuts_all_inputs_from_requested_source_types_only():
    """A type cut is an all-or-nothing synthetic input boundary cut, never an image edit."""
    import hashlib
    import json

    from drosophilos.connectome.embed_image import full_graph_topology
    from drosophilos.connectome.mcns import MCNS

    net, _, ids, policy, m, idx, pl = _placed_latch_relay()
    # Remove a designed edge so build_image has to restore it as a real Profile 3 edge.
    keep = ~((m.pre == idx["I"]) & (m.post == idx["E"]))
    m = MCNS(m.neurons, m.pre[keep], m.post[keep], m.count[keep], min_syn=1)
    hosts = set(pl.mapping.values())
    base = m.topology(Params())
    incoming = [j for j, (pre, post) in enumerate(zip(base.src, base.dst)) if pre not in hosts and post in hosts]
    assert len(incoming) > 1
    cut_pre = int(base.src[incoming[0]])
    keep_pre = next(int(base.src[j]) for j in incoming if int(base.src[j]) != cut_pre)
    other_pre = next(int(base.src[j]) for j in incoming
                     if int(base.src[j]) not in (cut_pre, keep_pre))
    neurons = m.neurons.copy()
    neurons.loc[:, "type"] = "other"
    neurons.loc[cut_pre, "type"] = "cut_me"
    neurons.loc[keep_pre, "type"] = "keep_me"
    neurons.loc[other_pre, "type"] = "also_cut"
    m2 = MCNS(neurons, m.pre, m.post, m.count, min_syn=1)
    image = build_image(net, m2, pl, policy)
    assert image.profile3 and any(e.src == ids["edge_inh"] and e.dst == ids["edge"] for e in image.profile3)
    baseline = full_graph_topology(m2, Params(), image)
    # Frozen from full_graph_topology loaded from commit 7b4ae68 via git show,
    # before cut_inputs_from_types existed, on this same fixture. Hash dtype strings
    # and raw array bytes in this order, followed by the sorted compact JSON counts.
    digest = hashlib.sha256()
    for field in ("src", "dst", "quanta", "delay"):
        array = getattr(baseline.topology, field)
        digest.update(array.dtype.str.encode("ascii"))
        digest.update(array.tobytes())
    digest.update(json.dumps(baseline.counts, sort_keys=True, separators=(",", ":")).encode("utf-8"))
    assert digest.hexdigest() == "2961c74335ff19dc692000f8fa0c0ba08c7d448c07fff6f7e6b8996b39a6073e"

    def weights(graph):
        topo = graph.topology
        pairs = list(zip(topo.src, topo.dst))
        assert len(set(pairs)) == topo.nnz  # this fixture has one edge per pair
        return dict(zip(pairs, topo.quanta))

    # Adding Profile 3 edges re-sorts the full topology; anatomical positions no
    # longer identify the same edges. Compare all full-graph weights by endpoints.
    plain_weights = weights(baseline)
    incoming = {(int(base.src[j]), int(base.dst[j])) for j in incoming}
    anatomy = {(int(s), int(d)): abs(int(q)) // QUANTA_PER_SYNAPSE
               for s, d, q in zip(base.src, base.dst, base.quanta)}
    cut = full_graph_topology(m2, Params(), image, cut_inputs_from_types=("cut_me",))
    cut_weights = weights(cut)
    expected = {pair for pair in incoming if pair[0] == cut_pre}
    assert expected
    assert cut_weights == {pair: 0 if pair in expected else q for pair, q in plain_weights.items()}
    assert cut.counts["cut_inputs_from_types"] == ["cut_me"]
    assert cut.counts["cut_inputs_from_types_matched_neurons"] == 1
    assert cut.counts["inputs_cut_edges"] == len(expected)
    assert cut.counts["inputs_cut_synapses"] == sum(anatomy[pair] for pair in expected)
    assert cut.counts["inputs_remaining_edges"] == len(incoming) - len(expected)
    remaining = incoming - expected
    assert cut.counts["inputs_remaining_synapses"] == sum(anatomy[pair] for pair in remaining)
    assert cut.counts["inputs_remaining_edges"] == len(remaining)
    # All host outputs remain unchanged by an input-only cut.
    outgoing = [(pre, post) for pre, post in zip(base.src, base.dst) if pre in hosts and post not in hosts]
    assert all(cut_weights[pair] == plain_weights[pair] for pair in outgoing)
    multi = full_graph_topology(m2, Params(), image, cut_inputs_from_types=("cut_me", "also_cut"))
    multi_expected = {pair for pair in incoming if pair[0] in (cut_pre, other_pre)}
    assert weights(multi) == {pair: 0 if pair in multi_expected else q for pair, q in plain_weights.items()}
    # Designed host-to-host and Profile 3 image edges are not anatomical input-boundary edges.
    for s, d, q in zip(net.src, net.dst, net.quanta):
        pair = (baseline.index_map[s], baseline.index_map[d])
        assert cut_weights[pair] == plain_weights[pair] == q

    unknown = full_graph_topology(m2, Params(), image, cut_inputs_from_types=("not_a_type",))
    assert np.array_equal(unknown.topology.quanta, baseline.topology.quanta)
    assert unknown.counts["inputs_cut_edges"] == unknown.counts["inputs_cut_synapses"] == 0
    assert unknown.counts["warnings"] == ["requested input source type matched no outside neuron: not_a_type"]
    with pytest.raises(ValueError, match="cannot be combined"):
        full_graph_topology(m2, Params(), image, cut_inputs_from_types=("cut_me",), cut_inputs_frac=.1)
    with pytest.raises(ValueError, match="cannot be combined"):
        full_graph_topology(m2, Params(), image, cut_inputs_from_types=("cut_me",), zero_inputs=True)


def test_full_graph_cut_uses_exact_decimal_ceil():
    """0.07 of exactly 100 boundary edges cuts seven, not eight from float roundoff."""
    from drosophilos.connectome.embed_image import full_graph_topology
    from drosophilos.connectome.mcns import MCNS

    net, drive, ids = latch_relay_netlist()
    policy = Policy()
    m, idx = synthetic_mcns(policy, drive, n_distractors=101, n_weak=0)
    pl = Placement({ids["src"]: idx["S"], ids["edge"]: idx["E"], ids["edge_inh"]: idx["I"],
                    ids["u"]: idx["U"], ids["v"]: idx["V"]}, [], 0, [], 0, 0.0)
    outside = np.array([i for i in range(m.n) if i not in pl.mapping.values()][:100], np.int32)
    assert len(outside) == 100
    pre = np.append(m.pre, np.full(100, idx["S"], np.int32))
    post = np.append(m.post, outside)
    count = np.append(m.count, np.ones(100, np.int32))
    order = np.lexsort((post, pre))
    m2 = MCNS(m.neurons, pre[order], post[order], count[order], min_syn=1)
    img = build_image(net, m2, pl, policy)
    fg = full_graph_topology(m2, Params(), img, cut_outputs_frac=.07)
    assert fg.counts["outputs_edges"] == 100
    assert fg.counts["outputs_cut_edges"] == 7


def test_full_graph_attribution_cuts_scores_then_counts_then_post_pre(tmp_path):
    from drosophilos.connectome.embed_image import full_graph_topology

    net, _, _, policy, m, _, pl = _placed_latch_relay()
    img = build_image(net, m, pl, policy)
    base = m.topology(Params())
    plain = full_graph_topology(m, Params(), img, base=base)
    hosts = set(pl.mapping.values())
    boundary = [j for j, (pre, post) in enumerate(zip(base.src, base.dst)) if pre not in hosts and post in hosts]
    assert len(boundary) > 10
    counts = {j: abs(int(base.quanta[j])) // QUANTA_PER_SYNAPSE for j in boundary}
    scores = {j: i % 3 for i, j in enumerate(boundary)}
    # A weak edge outranks every unscored strong edge. Equal scores include count AND pair ties.
    scores[min(boundary, key=counts.get)] = 10
    assert len({(scores[j], counts[j]) for j in boundary}) < len(boundary)
    ranked = sorted(boundary, key=lambda j: (-scores[j], -counts[j], int(base.dst[j]), int(base.src[j])))
    bodies = m.neurons["bodyId"].to_numpy()
    path = tmp_path / "attr.npz"
    shuffled = np.random.default_rng(0).permutation(boundary)
    np.savez_compressed(path, pre_bodyId=bodies[base.src[shuffled]], post_bodyId=bodies[base.dst[shuffled]],
                        score=[scores[j] for j in shuffled])
    for fraction in (0, .05, .1, .2, .3, .7, 1):
        fg = full_graph_topology(m, Params(), img, base=base, cut_inputs_by="attribution",
                                 attribution_in=path, cut_inputs_frac=fraction)
        n_cut = int(np.ceil(fraction * len(boundary)))
        cut = ranked[:n_cut]
        assert {j for j in boundary if fg.topology.quanta[j] == 0} == set(cut)
        assert fg.counts["inputs_cut_edges"] == n_cut
        assert fg.counts["inputs_cut_synapses"] == sum(counts[j] for j in cut)
        assert fg.counts["inputs_remaining_synapses"] == sum(counts[j] for j in ranked[n_cut:])
        assert fg.counts["inputs_weakest_cut_synapses"] == (min(counts[j] for j in cut) if cut else None)
        if fraction == 0:
            assert np.array_equal(fg.topology.quanta, plain.topology.quanta)
        if fraction == 1:
            zero = full_graph_topology(m, Params(), img, zero_inputs=True, base=base)
            assert np.array_equal(fg.topology.quanta, zero.topology.quanta)
            assert fg.counts["inputs_cut_synapses"] == zero.counts["inputs_cut_synapses"]
    # Explicit legacy ranking has exactly the original topology arrays and count JSON.
    legacy = full_graph_topology(m, Params(), img, base=base, cut_inputs_by="synapses")
    assert legacy.counts == plain.counts and "cut_inputs_by" not in plain.counts
    for name in ("src", "dst", "quanta", "delay", "indptr"):
        assert getattr(legacy.topology, name).tobytes() == getattr(plain.topology, name).tobytes()
    # Empty scores mean zero throughout, reducing exactly to the old synapse ordering.
    np.savez_compressed(path, pre_bodyId=np.empty(0, np.int64), post_bodyId=np.empty(0, np.int64),
                        score=np.empty(0, np.int64))
    zero_scores = full_graph_topology(m, Params(), img, base=base, cut_inputs_by="attribution",
                                     attribution_in=path, cut_inputs_frac=.3)
    synapses = full_graph_topology(m, Params(), img, base=base, cut_inputs_frac=.3)
    assert zero_scores.topology.quanta.tobytes() == synapses.topology.quanta.tobytes()
    # Adjacent integer scores above float64's exact range must not collapse into a tie.
    weak = min(boundary, key=counts.get)
    strong = max((j for j in boundary if j != weak), key=counts.get)
    assert counts[weak] < counts[strong]
    np.savez_compressed(path, pre_bodyId=bodies[base.src[[strong, weak]]],
                        post_bodyId=bodies[base.dst[[strong, weak]]],
                        score=np.array([2**53, 2**53 + 1], np.int64))
    exact = full_graph_topology(m, Params(), img, base=base, cut_inputs_by="attribution",
                               attribution_in=path, cut_inputs_frac=.0001)
    assert exact.topology.quanta[weak] == 0 and exact.topology.quanta[strong] != 0


@pytest.mark.parametrize("bad", ["negative", "nan", "duplicate", "foreign", "host_host"])
def test_full_graph_rejects_invalid_attribution(tmp_path, bad):
    from drosophilos.connectome.embed_image import full_graph_topology

    net, _, _, policy, m, _, pl = _placed_latch_relay()
    image = build_image(net, m, pl, policy)
    hosts = set(pl.mapping.values())
    j = next(i for i, (a, b) in enumerate(zip(m.pre, m.post)) if a not in hosts and b in hosts)
    pre, post = int(m.pre[j]), int(m.post[j])
    bodies = m.neurons["bodyId"].to_numpy()
    pre_body, post_body, score = [bodies[pre]], [bodies[post]], [1.0]
    if bad in ("negative", "nan"):
        score = [-1 if bad == "negative" else np.nan]
    elif bad == "duplicate":
        pre_body *= 2; post_body *= 2; score *= 2
    elif bad == "foreign":
        pre_body = [999_999]
    elif bad == "host_host":
        pre_body = [bodies[min(hosts)]]
    path = tmp_path / "bad.npz"
    np.savez_compressed(path, pre_bodyId=pre_body, post_bodyId=post_body, score=score)
    with pytest.raises(ValueError, match="attribution"):
        full_graph_topology(m, Params(), image, cut_inputs_by="attribution", attribution_in=path, cut_inputs_frac=.1)


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


def test_simulate_channel_powers_on_a_flipflop_channel():
    """simulate_channel hands the harness a made simulator, so run_transactions skips its own
    power-on injection; the image harness must inject the flip-flop pulse itself (the flip-flop
    adder's condition A faulted on every word until it did, docs/h1_placement.md)."""
    from types import SimpleNamespace
    from drosophilos.connectome.embed_image import simulate_channel
    from drosophilos.protocol.handshake import build_channel
    ch = build_channel(Params(), 2, storage="flipflop", watchdog_hops=40)
    image = SimpleNamespace(topology=ch.net.topology())  # the netlist itself as the "image"
    res = simulate_channel(ch, image, [1, 2, 3], [1, 2, 3], fresh_each=False, max_steps_per_tx=8000)
    assert res["correct"] == 3 and res["faults"] == 0, {k: v for k, v in res.items() if k != "records"}
