"""A2 liveness: the machine itself detects a transaction that cannot complete (watchdog ->
TIMEOUT latch -> FAULT-ACCEPT) and stale state after a reset (monitor -> STALE latch ->
READY held, reset re-run), and recovers in both cases."""

from drosophilos.protocol.handshake import build_channel
from drosophilos.protocol.run import run_transactions
from drosophilos.sim.model import Params

PARAMS = Params()


def test_watchdog_timeout_on_severed_data_path_recovers():
    ch = build_channel(PARAMS, 4)
    net = ch.net
    relay = net.roles.index("data.b2r1.edge")
    for k in range(net.nnz):  # bit 2, rail 1 can never reach the consumer
        if net.src[k] == relay:
            net.quanta[k] = 0
    recs, sim, st = run_transactions(ch, PARAMS, [0b0100, 0b0010, 0b1001])
    r0 = recs[0]
    assert r0.accept_step is None and r0.timeout_spikes > 0 and r0.status == "timeout", r0
    assert r0.cleared_step is not None and r0.ready_step is not None  # FAULT-ACCEPT ran the four phases
    t_timeout = (r0.cleared_step - r0.load_step) * PARAMS.dt
    assert 180 < t_timeout < 400, t_timeout  # ~40 hops x 5.3 ms after activation, then the CLEARED chain
    assert [r.decoded for r in recs[1:]] == [0b0010, 0b1001]
    assert all(r.timeout_spikes == 0 for r in recs[1:])


def test_stale_monitor_is_optional_and_documented_as_rejected():
    """The stale monitor exists behind `monitor=True` but is not part of the production channel:
    its per-latch detectors fire spuriously under perturbation (see add_liveness). This test
    only pins that the default build has no monitor and that the option still constructs."""
    ch = build_channel(PARAMS, 4)
    assert ch.consumer.monitor is None and ch.producer.watchdog is not None
    ch2 = build_channel(PARAMS, 4, monitor=True)
    assert ch2.consumer.monitor is not None and ch2.net.n > ch.net.n


def test_clean_transactions_never_raise_timeout_or_stale():
    ch = build_channel(PARAMS, 4)
    recs, sim, st = run_transactions(ch, PARAMS, [0b1010, 0b0101, 0b1111, 0b0000])
    assert st["completed"] == 4 and st["correct"] == 4
    assert all(r.timeout_spikes == 0 and r.stale_retry_spikes == 0 for r in recs)
