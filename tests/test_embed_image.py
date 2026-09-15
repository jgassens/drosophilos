"""Placement -> simulable image on real neuron ids (connectome/embed_image.py).

Fast: the planted latch + edge relay + inhibitor of test_embed_netlist, placed on its synthetic
connectome, turned into an image (every edge carried at the designed quanta, parasitic edges
zeroed) and simulated: one source spike ignites the latch through the relay, the relay fires
once, and the latch holds for a few hundred ms of neural time.

Slow (skipped without the MCNS data): the 4-bit adder on the saved best-of-8 placement,
condition A (every missing edge added as Profile 3, parasitic zeroed), 5 additions, must compute.
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
