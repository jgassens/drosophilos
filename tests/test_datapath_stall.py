"""Synthetic spike evidence: distinguish a missing rail from an unrecorded one.

These traces test the diagnostic, not a proposed neural failure mechanism. No external
campaign artifacts or GPU replay are needed.
"""

import json
from types import SimpleNamespace

import numpy as np
import pytest

from drosophilos.bench import stall_diag as diag
from drosophilos.lib.kernel import build_pipeline
from drosophilos.sim.model import Params


@pytest.fixture
def pipeline():
    return build_pipeline(
        Params(), 2,
        [{"name": "source", "op": "XOR", "a": "input", "b": ("const", "one")},
         {"name": "blocked", "op": "SUB", "a": "source", "b": ("const", "one")}],
        consts={"one": 1}, outputs=["blocked"],
    )


def make_dump(tmp_path, pl, spikes, *, capture_ids=None, end=30_000, begin=0):
    spikes = sorted(spikes)
    path = tmp_path / "synthetic.npz"
    diag.write_compact_dump(
        path, [s for s, _ in spikes], [n for _, n in spikes], pl.net.roles,
        capture_ids=list(range(pl.net.n)) if capture_ids is None else capture_ids,
        window={"start_step": begin, "end_step": end},
    )
    return path, diag.load_dump(path, pl.net.roles)


def incoming_driver(net, latch):
    return next(net.src[e] for e in net.incoming[latch.u]
                if net.quanta[e] > 0 and net.src[e] not in latch.members)


def test_full_capture_traces_fault_reset_and_stopped_latches(tmp_path, pipeline):
    pl = pipeline
    source, cell = pl.cells
    actd = diag._actd_id(pl.net.roles, cell.name)
    z = pl.n + 1
    output = incoming_driver(pl.net, cell.stage.rails[0][1])
    # The second bit's output relay fires, but its stage latch never does. An ordinary
    # single-railed Z then has a fault-gate spike: the diagnostic must not invent Z1.
    missing_driver = incoming_driver(pl.net, cell.stage.rails[1][0])
    spikes = [(1000, cell.start), (1100, actd), (1180, output), (1210, missing_driver),
              (1600, cell.stage.fault[z]), (1640, cell.stage.fault_latch.u)]
    for t in range(900, 1651, 50):
        spikes.extend((t, source.master.rails[i][0].u) for i in range(pl.n))
        spikes.extend((t, pl.consts["one"][i][int(i == 0)].u) for i in range(pl.n))
    for t in range(1200, 1651, 50):
        spikes.extend([(t, cell.stage.rails[0][1].u), (t, cell.stage.rails[z][0].u)])
    for t in range(1300, 1701, 50):
        spikes.extend([(t, cell.stage.valid[0].u), (t, cell.stage.valid[z].u)])
    reset_steps = [1700, 1750, 1800, 1850]
    spikes.extend((t, cell.stage.reset_inh) for t in reset_steps)
    _, dump = make_dump(tmp_path, pl, spikes)
    view = diag.datapath_view(pl, dump, cell.name, 1000)
    assert view["valid_bits_rose"] == [0, z]
    assert view["valid_bits_silent"] == [1, 2, 4]
    assert view["valid_bits_unavailable"] == []
    signals = view["signals"]
    assert signals[cell.stage.valid[0].u]["trains"] == [(1300, 1700, 9)]
    assert signals[cell.stage.rails[0][1].u]["median_isi"] == 50
    assert signals[cell.stage.rails[0][1].u]["live_at_end"] is False
    assert signals[cell.stage.rails[z][1].u]["state"] == "silent"
    assert signals[cell.stage.fault[z]]["sample_steps"] == [1600]
    assert signals[cell.stage.reset_inh]["sample_steps"] == reset_steps
    assert cell.stage.rails[0][1].u in view["reset_targets"]
    assert cell.stage.valid[0].u in view["reset_targets"]
    assert cell.stage.reset_trigger in view["fault_targets"]
    assert cell.master.rails[0][0].u not in view["reset_targets"]
    assert view["operands"][0]["bits"][0]["state_at_actd"] == "r0"
    assert view["operands"][1]["bits"][0]["state_at_actd"] == "r1"
    assert signals[missing_driver]["sample_steps"] == [1210]
    assert signals[cell.stage.rails[1][0].u]["state"] == "silent"
    assert any(e["source"] == cell.stage.reset_inh and
               e["target"] == cell.stage.rails[1][0].u and e["quanta"] < 0 for e in view["edges"])
    text = diag.render_datapath(view)
    assert "does not by itself prove a double rail" in text
    assert "1,700, 1,750, 1,800, 1,850" in text
    assert f"{missing_driver}" in text and "1,210" in text


def test_filtered_silence_is_not_missing_data_or_a_kill(tmp_path, pipeline):
    pl, cell = pipeline, pipeline.cells[-1]
    captured = [n for n, r in enumerate(pl.net.roles) if diag.REPLAY_ROLE_FILTER.search(r)]
    _, dump = make_dump(tmp_path, pl, [(1000, cell.start), (1300, cell.stage.valid[0].u)],
                        capture_ids=captured)
    view = diag.datapath_view(pl, dump, cell.name, 1000)
    assert view["valid_bits_rose"] == [0]
    assert view["valid_bits_silent"] == [1, 2, 3, 4]
    assert view["signals"][cell.stage.reset_inh]["state"] == "unavailable"
    assert view["signals"][cell.stage.rails[0][0].u]["state"] == "unavailable"
    assert view["operands"][0]["bits"][0]["state_at_actd"] == "unavailable"
    assert view["signals"][cell.stage.fault[0]]["state"] == "silent"
    text = diag.render_datapath(view)
    assert "unavailable" in text and "silent" in text
    assert "No kill mechanism is inferred" in text
    # A narrower explicit capture overrides even the usual valid/fault role filter.
    _, narrow = make_dump(tmp_path, pl, [(1000, cell.start)], capture_ids=[cell.start])
    assert diag.datapath_view(pl, narrow, cell.name, 1000)["valid_bits_unavailable"] == list(range(5))


def test_live_valid_gate_without_ignition_is_visible(tmp_path, pipeline):
    pl, cell = pipeline, pipeline.cells[-1]
    prefix = f"{cell.name}.Q.valid3"
    gate = pl.net.roles.index(prefix + ".or")
    edge = pl.net.roles.index(prefix + ".ign.edge")
    inhibitor = pl.net.roles.index(prefix + ".ign.edge_inh")
    spikes = [(1000, cell.start)]
    spikes += [(t, gate) for t in range(900, 2000, 100)]
    spikes += [(t + 20, inhibitor) for t in range(900, 2000, 100)]
    captured = [n for n, r in enumerate(pl.net.roles) if diag.REPLAY_ROLE_FILTER.search(r)]
    _, dump = make_dump(tmp_path, pl, spikes, capture_ids=captured)
    view = diag.datapath_view(pl, dump, cell.name, 1000)
    assert {gate, edge, inhibitor} <= set(view["bits"][3]["valid_probes"])
    assert view["signals"][gate]["state"] == "held"
    assert view["signals"][edge]["state"] == "silent"
    assert view["signals"][cell.stage.valid[3].u]["state"] == "silent"
    text = diag.render_datapath(view)
    assert prefix + ".or" in text and prefix + ".ign.edge_inh" in text


def test_topology_disambiguates_repeated_output_and_zero_roles(tmp_path, pipeline):
    pl = pipeline
    source, cell = pl.cells
    _, dump = make_dump(tmp_path, pl, [(1000, cell.start)])
    view = diag.datapath_view(pl, dump, cell.name, 1000)
    for bit in (0, pl.n + 1):
        target_driver = incoming_driver(pl.net, cell.stage.rails[bit][0])
        other_driver = incoming_driver(pl.net, source.stage.rails[bit][0])
        assert pl.net.roles[target_driver] == pl.net.roles[other_driver]
        assert target_driver != other_driver
        assert target_driver in view["bits"][bit]["drivers"][0]
        assert other_driver not in view["bits"][bit]["cone_ids"]
    # A boundary rail is included; the producing cell's transaction is not traversed.
    assert source.master.rails[0][0].u in view["signals"]
    assert source.start not in view["signals"]


def test_transaction_bounds_and_preexisting_valid_train(tmp_path, pipeline):
    pl, cell = pipeline, pipeline.cells[-1]
    valid = cell.stage.valid[0].u
    spikes = [(1000, cell.start), (3000, cell.start), (3100, cell.stage.valid[1].u)]
    spikes += [(t, valid) for t in range(900, 1201, 50)]
    _, dump = make_dump(tmp_path, pl, spikes, begin=800)
    view = diag.datapath_view(pl, dump, cell.name, 1000, stop=3000)
    assert view["end"] == 2999
    assert view["signals"][valid]["state"] == "held"
    assert view["valid_bits_held"] == [0]
    assert view["signals"][valid]["rises"] == []
    assert view["signals"][valid]["last_before_start"] == 950
    assert view["signals"][valid]["trains"] == [(900, 1200, 7)]
    assert view["signals"][cell.stage.valid[1].u]["count"] == 0
    with pytest.raises(ValueError, match="captured window"):
        diag.datapath_view(pl, dump, cell.name, 799)


def test_legacy_partial_dump_retains_observed_nondefault_rail(tmp_path, pipeline):
    pl, cell = pipeline, pipeline.cells[-1]
    rail = cell.stage.rails[0][0].u
    path = tmp_path / "old.npz"
    ids = np.array([cell.start, rail])
    np.savez(path, step=[1000, 1200], neuron=ids, role=np.asarray(pl.net.roles)[ids])
    view = diag.datapath_view(pl, diag.load_dump(path, pl.net.roles), cell.name, 1000)
    assert view["signals"][rail]["state"] == "rose"
    assert view["signals"][cell.stage.rails[0][1].u]["state"] == "unavailable"


def test_analyze_dump_adds_view_without_changing_classification(tmp_path, monkeypatch, pipeline):
    pl, cell = pipeline, pipeline.cells[-1]
    actd = diag._actd_id(pl.net.roles, cell.name)
    path, _ = make_dump(tmp_path, pl, [(1000, cell.start), (1100, actd),
                                     (1300, cell.stage.valid[0].u)])
    record = tmp_path / "campaign.json"
    record.write_text(json.dumps({"copies": 1}))
    monkeypatch.setattr(diag, "build_tick_pipeline",
                        lambda _: (Params(), SimpleNamespace(outputs=[cell.name]), pl))
    report = diag.analyze_dump(path, record, node=0)
    assert report["classification"] == "datapath"
    assert report["anomaly"]["cell"] == cell.name
    assert report["datapath"]["actd"] == 1100
    assert report["datapath"]["valid_bits_rose"] == [0]
    text = diag.render_markdown(report)
    for section in ("## Classification", "## Blocked cell and neighbour timeline",
                    "## Bit-level datapath evidence", "## Campaign accounting for this copy"):
        assert section in text


@pytest.mark.parametrize("rate_robust", [False, True])
def test_specialized_and_conditioned_topologies_remain_traceable(tmp_path, rate_robust):
    pl = build_pipeline(Params(), 2,
                        [{"name": "logic", "op": "XOR", "a": "input", "b": ("const", "one")}],
                        consts={"one": 1}, datapath="specialized", rate_robust=rate_robust)
    cell = pl.cells[0]
    _, dump = make_dump(tmp_path, pl, [(1000, cell.start)])
    view = diag.datapath_view(pl, dump, cell.name, 1000)
    assert len(view["bits"]) == 5
    assert cell.stage.rails[0][0].u in view["bits"][0]["cone_ids"]
    assert pl.input_reg.master.rails[0][0].u in view["bits"][0]["cone_ids"]
    assert view["unavailable_ids"] == []
