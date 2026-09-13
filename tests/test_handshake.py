"""One neural channel, repeated four-phase transactions, decoded exactly."""

import numpy as np
import pytest

from drosophilos.protocol.handshake import build_channel
from drosophilos.protocol.run import run_transactions
from drosophilos.sim.model import Params


@pytest.mark.parametrize("width,words", [(1, [1, 0, 1, 1, 0, 0, 1, 0]), (4, [0b1010, 0b0101, 0b1111, 0b0000, 0b0110, 0b1001]), (8, [0xA5, 0x3C, 0xFF, 0x00, 0x81])])
def test_channel_transacts_repeatedly(width, words):
    params = Params()
    ch = build_channel(params, width)
    records, sim, stats = run_transactions(ch, params, words)
    assert stats["completed"] == len(words), stats
    assert stats["correct"] == len(words), [(r.word, r.decoded, r.status) for r in records]
    assert all(r.fault_spikes == 0 for r in records)
    for r in records:
        assert r.accept_step < r.cleared_step < r.ready_step
    # quiescence: no consumer latch fires in the 20 ms before READY, and nothing of the
    # consumer fires between READY and the next load (stale completion would accept early)
    ev = sim.trace.events
    roles = ch.net.roles
    for k, r in enumerate(records):
        nxt = records[k + 1].load_step if k + 1 < len(records) else r.ready_step + 300
        pre = (ev["step"] > r.ready_step - 200) & (ev["step"] <= r.ready_step)
        post = (ev["step"] > r.ready_step) & (ev["step"] < nxt)
        late = [roles[n] for n in ev["neuron"][pre] if roles[n].startswith("Q.") and (".L." in roles[n] or roles[n].startswith("Q.b"))]
        after = [roles[n] for n in ev["neuron"][post] if roles[n].startswith("Q.") and not roles[n].startswith("Q.ready")]
        assert not late, (k, r.word, sorted(set(late)))
        assert not after, (k, r.word, sorted(set(after)))
    print(width, {k: v for k, v in stats.items() if k != "accept_latency_ms"})
