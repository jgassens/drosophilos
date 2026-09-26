"""Synthetic, simulation-free tests for the datapath fault diagnostic."""

from types import SimpleNamespace

import numpy as np

from drosophilos.bench.fault_diag import analyze_faults, render_text


def _fake_pipeline():
    roles = [
        "cell.Q.valid0.L.u",
        "cell.Q.valid1.L.u",
        "cell.Q.fault0.and",
        "cell.Q.fault1.and",
        "cell.faultL.u",
        "cell.start",
        "cell.actd.d0",
        "cell.Q.comp.L.u",
        "cell.commit_pulse",
        "cell.done.edge",
    ]
    stage = SimpleNamespace(
        valid=[SimpleNamespace(u=0), SimpleNamespace(u=1)],
        fault=[2, 3],
        fault_latch=SimpleNamespace(u=4),
        completion=SimpleNamespace(u=7),
    )
    cell = SimpleNamespace(
        name="cell",
        op="SEL",
        datapath="generic",
        stage=stage,
        start=5,
        commit_pulse=8,
        reg=SimpleNamespace(done_relay=9),
    )
    return SimpleNamespace(
        net=SimpleNamespace(roles=roles, n=len(roles), nnz=17),
        cells=[cell],
        inputs={},
        drive=SimpleNamespace(loop_period_steps=10),
        build_options={"datapath": "specialized"},
    )


def test_reports_each_fault_rise_with_transaction_and_valid_state(tmp_path, monkeypatch):
    pipeline = _fake_pipeline()
    monkeypatch.setattr(
        "drosophilos.bench.fault_diag.build_tick_pipeline",
        lambda _campaign: (None, None, pipeline),
    )

    # Two transactions and two different fault gates.  Repeated valid-latch spikes keep the
    # latches observably active at each fault; the arrays are a dump fixture, not a simulation.
    events = [
        (100, 5),
        (110, 6),
        (120, 0),
        (125, 1),
        (130, 7),
        (140, 8),
        (142, 0),
        (145, 1),
        (150, 3),
        (160, 4),
        (170, 9),
        (300, 5),
        (310, 6),
        (320, 0),
        (325, 1),
        (330, 7),
        (340, 8),
        (342, 0),
        (345, 1),
        (350, 2),
        (360, 4),
        (370, 9),
    ]
    step = np.asarray([item[0] for item in events], dtype=np.int64)
    neuron = np.asarray([item[1] for item in events], dtype=np.int64)
    role = np.asarray([pipeline.net.roles[item[1]] for item in events])
    dump_path = tmp_path / "faults.npz"
    np.savez(dump_path, step=step, neuron=neuron, role=role)
    campaign_path = tmp_path / "campaign.json"
    campaign_path.write_text('{"copies": 2}')

    report = analyze_faults(dump_path, campaign_path, node=1)

    assert report["netlist"] == {"neurons": 10, "synapses": 17}
    assert report["captured_neurons"] == 10
    assert report["neuron_roles"] == [
        {"neuron": neuron_id, "role": role_name}
        for neuron_id, role_name in enumerate(pipeline.net.roles)
    ]
    assert [(fault["step"], fault["bit"]) for fault in report["faults"]] == [
        (150, 1),
        (350, 0),
    ]

    first, second = report["faults"]
    assert first["gate_role"] == "cell.Q.fault1.and"
    assert first["operation"] == "SEL"
    assert first["datapath"] == "generic"
    assert first["fault_latch_step"] == 160
    assert first["transaction"] == {
        "number": 1,
        "start": 100,
        "actd": 110,
        "completion": 130,
        "commit": 140,
        "done": 170,
    }
    assert first["valid_bits_active"] == [0, 1]
    assert first["valid_bits_rose"] == [0, 1]
    assert [item["rise_step"] for item in first["valid_latches"]] == [120, 125]

    assert second["transaction"]["number"] == 2
    assert second["transaction"]["start"] == 300
    assert second["fault_latch_step"] == 360
    assert second["valid_bits_active"] == [0, 1]
    text = render_text(report)
    assert "2 stage fault-gate rise(s)" in text
    assert "step 150: cell.Q.fault1.and" in text
    assert "transaction 2" in text


def test_rejects_node_outside_campaign(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "drosophilos.bench.fault_diag.build_tick_pipeline",
        lambda _campaign: (None, None, _fake_pipeline()),
    )
    campaign_path = tmp_path / "campaign.json"
    campaign_path.write_text('{"copies": 1}')

    import pytest

    with pytest.raises(ValueError, match=r"node must be in \[0, 1\)"):
        analyze_faults(tmp_path / "unused.npz", campaign_path, node=1)
