import numpy as np
import pytest

from drosophilos.sim import Params, Topology


def random_topology(n: int, k: int, rng: np.random.Generator, max_delay: int = 30) -> Topology:
    """n neurons, ~k outgoing synapses each, signed integer-synapse weights, random delays."""
    src = np.repeat(np.arange(n), k)
    dst = rng.integers(0, n, size=len(src))
    keep = dst != src
    src, dst = src[keep], dst[keep]
    # strong fly-like connections: 10-80 anatomical synapses per pair, 30% inhibitory
    syn_count = rng.integers(10, 80, size=len(src))
    sign = np.where(rng.random(len(src)) < 0.3, -1, 1)
    quanta = sign * syn_count * 16
    delay = rng.integers(0, max_delay + 1, size=len(src))
    return Topology.from_edges(n, src, dst, quanta, delay)


def random_events(n: int, n_steps: int, rng: np.random.Generator, rate_per_step: float = 0.03, n_inputs: int = 12):
    """Poisson-like external drive onto the first `n_inputs` neurons (120 synapses' worth per event)."""
    steps, neurons, quanta = [], [], []
    for i in range(n_inputs):
        hits = np.nonzero(rng.random(n_steps) < rate_per_step)[0]
        steps.append(hits)
        neurons.append(np.full(len(hits), i))
        quanta.append(np.full(len(hits), 120 * 16))
    return np.concatenate(steps), np.concatenate(neurons), np.concatenate(quanta)


@pytest.fixture
def small_circuit():
    rng = np.random.default_rng(1234)
    topo = random_topology(60, 8, rng)
    n_steps = 3000
    ev0 = random_events(60, n_steps, rng)
    ev1 = random_events(60, n_steps, rng)
    return topo, Params(), n_steps, (ev0, ev1)
