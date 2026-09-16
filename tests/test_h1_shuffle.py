"""Each shuffle mode preserves what it claims and changes what it claims, on a small synthetic
MCNS-like graph (200 neurons) with the same fields as the real MCNS."""

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from drosophilos.bench.h1_shuffle import configuration_drift, shuffle_mcns
from drosophilos.connectome.mcns import INHIBITORY, MCNS

N = 200
N_SUPERCLASSES = 4
N_RAW_EDGES = 2000


def _synthetic_mcns(seed: int = 42) -> MCNS:
    rng = np.random.default_rng(seed)
    superclass = np.array([f"sc{i % N_SUPERCLASSES}" for i in range(N)])
    nts = np.array(["acetylcholine", "gaba", "glutamate", "histamine", "unclear"])
    nt = rng.choice(nts, size=N)
    sign = np.where(np.isin(nt, list(INHIBITORY)), -1, 1).astype(np.int8)
    neurons = pd.DataFrame({"bodyId": np.arange(N), "superclass": superclass, "nt": nt, "sign": sign})

    pre = rng.integers(0, N, size=N_RAW_EDGES).astype(np.int32)
    post = rng.integers(0, N, size=N_RAW_EDGES).astype(np.int32)
    count = rng.choice([1, 2, 3, 5, 8, 13], size=N_RAW_EDGES).astype(np.int32)
    C = sp.csr_matrix((count, (pre, post)), shape=(N, N))
    C.sum_duplicates()
    coo = C.tocoo()
    return MCNS(neurons, coo.row.astype(np.int32), coo.col.astype(np.int32), coo.data.astype(np.int32), min_syn=1)


@pytest.fixture(scope="module")
def m() -> MCNS:
    return _synthetic_mcns()


def _strengths(mm: MCNS):
    out = np.bincount(mm.pre, weights=mm.count, minlength=N)
    inn = np.bincount(mm.post, weights=mm.count, minlength=N)
    return out, inn


def test_configuration_preserves_synapse_degree_sequences(m):
    sm = shuffle_mcns(m, seed=0, mode="configuration")
    out_before, in_before = _strengths(m)
    out_after, in_after = _strengths(sm)
    assert np.array_equal(out_before, out_after)
    assert np.array_equal(in_before, in_after)
    assert int(sm.count.sum()) == int(m.count.sum())


def test_configuration_changes_the_graph(m):
    sm = shuffle_mcns(m, seed=0, mode="configuration")
    before = set(zip(m.pre.tolist(), m.post.tolist()))
    after = set(zip(sm.pre.tolist(), sm.post.tolist()))
    assert before != after


def test_configuration_different_seeds_differ(m):
    a = shuffle_mcns(m, seed=0, mode="configuration")
    b = shuffle_mcns(m, seed=1, mode="configuration")
    assert set(zip(a.pre.tolist(), a.post.tolist())) != set(zip(b.pre.tolist(), b.post.tolist()))


def test_configuration_drift_report_is_self_consistent(m):
    sm = shuffle_mcns(m, seed=0, mode="configuration")
    report = configuration_drift(m, seed=0)
    out_before, in_before = _strengths(m)
    out_after, in_after = _strengths(sm)
    assert report["out_synapse_l1_drift"] == float(np.abs(out_after - out_before).sum())
    assert report["in_synapse_l1_drift"] == float(np.abs(in_after - in_before).sum())
    assert report["self_loops_dropped"] >= 0
    assert report["multi_edges_merged"] >= 0


def test_weights_preserves_adjacency_and_total(m):
    sm = shuffle_mcns(m, seed=0, mode="weights")
    assert np.array_equal(m.pre, sm.pre)
    assert np.array_equal(m.post, sm.post)
    assert int(sm.count.sum()) == int(m.count.sum())
    assert len(sm.pre) == len(m.pre)


def test_weights_changes_the_counts(m):
    sm = shuffle_mcns(m, seed=0, mode="weights")
    assert not np.array_equal(m.count, sm.count)


def test_weights_keeps_neurons_untouched(m):
    sm = shuffle_mcns(m, seed=0, mode="weights")
    assert (sm.neurons["nt"].to_numpy() == m.neurons["nt"].to_numpy()).all()


def test_transmitter_preserves_graph_and_weights(m):
    sm = shuffle_mcns(m, seed=0, mode="transmitter")
    assert np.array_equal(m.pre, sm.pre)
    assert np.array_equal(m.post, sm.post)
    assert np.array_equal(m.count, sm.count)


def test_transmitter_preserves_per_superclass_nt_counts(m):
    sm = shuffle_mcns(m, seed=0, mode="transmitter")
    for sc in m.neurons["superclass"].unique():
        before = m.neurons.loc[m.neurons["superclass"] == sc, "nt"].value_counts().sort_index()
        after = sm.neurons.loc[sm.neurons["superclass"] == sc, "nt"].value_counts().sort_index()
        assert before.equals(after)


def test_transmitter_changes_individual_assignments(m):
    sm = shuffle_mcns(m, seed=0, mode="transmitter")
    assert not (sm.neurons["nt"].to_numpy() == m.neurons["nt"].to_numpy()).all()


def test_transmitter_sign_follows_nt(m):
    sm = shuffle_mcns(m, seed=0, mode="transmitter")
    expected_sign = np.where(np.isin(sm.neurons["nt"].to_numpy(), list(INHIBITORY)), -1, 1).astype(np.int8)
    assert np.array_equal(sm.neurons["sign"].to_numpy(), expected_sign)


def test_unknown_mode_raises(m):
    with pytest.raises(ValueError):
        shuffle_mcns(m, seed=0, mode="bogus")
