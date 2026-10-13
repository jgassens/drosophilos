"""Small, no-MCNS regressions for the H1 full-graph bench's mapping contract."""

from __future__ import annotations

import json

import pytest

from drosophilos.bench.h1_fullgraph import adder_build_kwargs
from drosophilos.connectome.embed_image import load_mapping_meta
from drosophilos.lib.adder import build_adder_channel
from drosophilos.sim.model import Params


def test_stage_h_mapping_meta_rebuilds_configuration_f_and_legacy_is_latch(tmp_path):
    meta = {
        "builder": "build_adder_channel", "width": 4, "ordered": True, "watchdog_hops": 100,
        "storage": "flipflop", "rate_robust": True, "experimental_autapse": True,
        "ordered_xor_oneshot": True, "k_max": 4.0, "isolation_weight": 2.0,
        "region": "whole_brain", "restarts": 4, "seed": 0,
    }
    path = tmp_path / "mapping.json"
    path.write_text(json.dumps(meta | {"mapping": {}}))
    loaded = load_mapping_meta(path)
    assert loaded == meta
    f_options = adder_build_kwargs(loaded)
    assert f_options == {key: meta[key] for key in (
        "width", "ordered", "watchdog_hops", "storage", "rate_robust", "experimental_autapse", "ordered_xor_oneshot")}
    f = build_adder_channel(Params(), **f_options)
    assert f.net.n == 740 and f.drive.rate_robust

    legacy_path = tmp_path / "legacy-mapping.json"
    legacy_path.write_text(json.dumps({"mapping": {}}))
    legacy_options = adder_build_kwargs(load_mapping_meta(legacy_path))
    legacy = build_adder_channel(Params(), **legacy_options)
    assert legacy_options == {"width": 4, "ordered": True, "watchdog_hops": 100, "storage": "latch",
                              "rate_robust": False, "experimental_autapse": False, "ordered_xor_oneshot": False}
    assert legacy.net.n == 614
    assert adder_build_kwargs({"storage": "flipflop"})["storage"] == "flipflop"
    with pytest.raises(ValueError, match="disagrees"):
        adder_build_kwargs(loaded, storage_override="latch")
    with pytest.raises(ValueError, match="disagrees"):
        adder_build_kwargs(loaded, width_override=3)
