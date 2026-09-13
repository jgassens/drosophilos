"""MCNS v1.0 loader: rules reproduce the published counts (skipped without the data)."""

import pytest

from drosophilos.connectome.mcns_download import DEFAULT_DIR, path_of

pytestmark = pytest.mark.skipif(not path_of("weights", DEFAULT_DIR).exists(), reason="MCNS data not downloaded")


@pytest.fixture(scope="module")
def mcns():
    from drosophilos.connectome.mcns import load_mcns

    return load_mcns()


def test_published_counts(mcns):
    assert mcns.n == 166_700  # Codex: MCNS v1.0 neurons
    g1 = mcns.graph_summary(1)
    g5 = mcns.graph_summary(5)
    assert g1["n_edges"] == 25_582_938  # "25.5 M connections" at min confidence 0.5
    assert g5["n_edges"] == 6_242_118  # Codex's displayed connection count = edges with >= 5 synapses
    assert 124_000_000 < g1["total_synapses"] < 125_000_000


def test_signs_and_lookups(mcns):
    assert 0.34 < (mcns.sign < 0).mean() < 0.37
    gf = mcns.select(type="DNp01")  # giant fiber, both sides
    assert len(gf) == 2
    bodies = mcns.bodies_of(gf)
    assert list(mcns.index_of(bodies)) == list(gf)
    topo = mcns.topology(min_syn=5)
    assert topo.nnz == 6_242_118
    assert set(topo.delay.tolist()) == {18}
