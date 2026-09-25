"""The ACT^d repair of a request's dark false rail after a failed START re-light (§10.5).

START re-lights each request's false ("consumed") rail `start_relight_hops` (5) hops late. If
that ignition fails it can leave stray spikes on the rail, and each one fires the repair
relay's 0.5-loop veto exactly like a live rail. With the 14-hop tap the first stray came only
~41 ms before the repair's driver: two strays, or one at a 1-sigma adverse corner, blocked the
repair, both rails stayed dark, and under true guards the producer's commit gate never passed
again: a stall with no fault and no timeout, the likeliest cause of seed 110's copy 14
(docs/tick_stalls.md). The shipped build (`relight_repair_delay=True`) moves the tap
start_relight_hops + 2 hops later.

A 1-bit MOV reads the input register; all copies of a test share one RefSim batch and differ
only in the repair relay's corner, the removed START ignition and the injected spikes. Strays
are injected as the reviewer measured them: the START ignition of false is removed and k
spikes reach the repair's veto interneuron at the healthy ignition's time, a loop period
apart. A k-sigma corner moves the relay's driver -4k %, its feed-forward inhibitor and veto
+4k % and its threshold +0.2k mV (mix B's sigmas); a negative k favours the repair.
"""

import numpy as np
import pytest

from drosophilos.lib.kernel import build_pipeline, run_pipeline_batched
from drosophilos.sim.model import Params
from drosophilos.sim.ref64 import RefSim

PARAMS = Params()
SPEC = [{"name": "out", "op": "MOV", "a": "input", "b": ("const", "zero")}]


def _copy(k=0, strays=None, cut=None, request_at=None):
    """strays: None = healthy ignition, else the ignition removed and `strays` veto spikes
    injected; cut: the ignition kept and the latch loop cut after `cut` false spikes (every
    reader of the rail sees them); request_at: the ignition removed and a new request's DONE
    delivered to the trigger `request_at` steps after START."""
    return {"k": k, "strays": strays, "cut": cut, "request_at": request_at}


def _run(relight_repair_delay, copies, tokens=(1, 1), max_ms=3700):
    pl = build_pipeline(PARAMS, 1, SPEC, consts={"zero": 0}, relight_repair_delay=relight_repair_delay)
    cell = pl.cells[0]
    req = cell.reqs["input"]
    false_u, false_v = req[0].members
    roles = pl.net.roles
    hops = pl.build_options["start_relight_hops"]
    ignition = roles.index(f"out.start.fd.d{hops - 1}")
    chain = [i for i, role in enumerate(roles) if role.startswith("out.actd2.d")]
    assert len(chain) == 3 + (hops + 2 if relight_repair_delay else 0)
    tap = roles.index(f"out.actd2.d{len(chain) - 1}")
    repair = roles.index("out.relight.input.edge")
    inhibitor = roles.index("out.relight.input.edge_inh")
    veto = roles.index("out.relight.input.veto")
    trigger = roles.index("out.trigger.input")
    topo = pl.net.topology()

    def edge(src, dst):
        mask = (topo.src == src) & (topo.dst == dst)
        assert mask.sum() == 1
        return mask

    B = len(copies)
    quanta = np.broadcast_to(topo.quanta, (B, topo.nnz)).astype(float).copy()
    vth = np.full((B, topo.n), PARAMS.V_th)
    for b, c in enumerate(copies):
        if c["strays"] is not None or c["request_at"] is not None:
            quanta[b, edge(ignition, false_u)] = 0
        quanta[b, edge(tap, repair)] *= 1 - 0.04 * c["k"]
        quanta[b, edge(tap, inhibitor)] *= 1 + 0.04 * c["k"]
        quanta[b, edge(veto, repair)] *= 1 + 0.04 * c["k"]
        vth[b, repair] += 0.2 * c["k"]
    w_veto, d_veto = int(topo.quanta[edge(false_u, veto)][0]), int(topo.delay[edge(false_u, veto)][0])
    d_loop = int(topo.delay[edge(false_u, false_v)][0])
    d_tap = int(topo.delay[edge(tap, repair)][0])
    period = pl.drive.loop_period_steps
    assert copies[0]["strays"] is None and copies[0]["cut"] is None and copies[0]["request_at"] is None
    watch = {cell.start: "start", false_u: "false", req[1].u: "true", tap: "tap", repair: "repair"}

    class Recording(RefSim):
        def __init__(self):
            super().__init__(topo, PARAMS, n_nodes=B, quanta=np.rint(quanta).astype(np.int32), V_th=vth,
                             record=[(b, repair) for b in range(B)])
            self.events = [{kind: [] for kind in watch.values()} for _ in range(B)]
            self.strays_sent = False

        def step(self):
            before = len(self._spk_step)
            super().step()
            for chunk in range(before, len(self._spk_step)):
                step = int(self._spk_step[chunk][0])
                for node, neuron in zip(self._spk_node[chunk].tolist(), self._spk_neuron[chunk].tolist()):
                    kind = watch.get(neuron)
                    if kind is None:
                        continue
                    events = self.events[node]
                    events[kind].append(step)
                    c = copies[node]
                    if kind == "start" and len(events["start"]) == 1 and c["request_at"] is not None:
                        self.add_events(node, [step + c["request_at"]], [trigger], [pl.drive.ignite])
                    if kind != "false" or len(events["start"]) != 1:
                        continue
                    lit = len([s for s in events["false"] if s > events["start"][0]])
                    if lit == c["cut"]:  # the cut: false_v does not answer this spike
                        self.add_events(node, [step + d_loop], [false_v], [-2 * pl.drive.loop])
                    if node == 0 and not self.strays_sent:  # the healthy ignition's first spike
                        self.strays_sent = True
                        for b, other in enumerate(copies):
                            n = other["strays"] or 0
                            self.add_events(b, [step + d_veto + j * period for j in range(n)], [veto] * n, [w_veto] * n)

    sim = Recording()
    expect = [len(tokens) + (c["request_at"] is not None) for c in copies]
    outs, sim, stats = run_pipeline_batched(pl, PARAMS, [list(tokens)] * B, max_ms=max_ms, gap_ms=1400,
                                            expect_outputs=expect, sim=sim, progress=0)
    V = np.array(sim.rec_V)
    results = []
    for b, events in enumerate(sim.events):
        start = events["start"][0]
        T = next(step for step in events["tap"] if step > start)
        second = events["start"][1] if len(events["start"]) > 1 else len(V)
        c = copies[b]
        rise = None if c["request_at"] is None else next(
            step for step in events["true"] if step > start + c["request_at"])
        results.append({
            "tap_ms": (T - start) / 10,
            "relay_mv": float(V[T + d_tap, b] - PARAMS.E_L),  # when the driver's pulse arrives
            "repairs_ms": [(step - start) / 10 for step in events["repair"] if start < step < second],
            "false_before_tap": len([step for step in events["false"] if start < step < T]),
            "false_after_tap": len([step for step in events["false"] if T < step < min(second, T + 400)]),
            "request_lead_ms": None if rise is None else (T - rise) / 10,  # true's rise before the tap
            "starts": len(events["start"]),
            "outputs": [value for _, value in outs[b]["out"]],
        })
    return pl, results, stats


def test_pre_fix_tap_loses_the_repair_to_two_stray_spikes_and_stalls_silently():
    """Documented pre-fix behaviour (`relight_repair_delay=False`, the merged guards line):
    the tap at START + 74 ms. One nominal stray still repairs, with ~0.7 ms to spare in the
    relay's race against its own inhibitor; two strays leave the relay ~6.3 mV below rest when
    the driver arrives and it stays silent, as does one stray at a 1-sigma corner. The pair
    stays dark, the input register never commits the second token, and nothing reports it."""
    copies = [_copy(), _copy(strays=0), _copy(strays=1), _copy(strays=2), _copy(k=1, strays=1)]
    pl, r, stats = _run(False, copies)
    assert pl.build_options["relight_repair_delay"] is False
    assert all(73 < x["tap_ms"] < 76 for x in r), [x["tap_ms"] for x in r]
    healthy, zero, one, two, corner = r
    assert healthy["repairs_ms"] == [] and healthy["outputs"] == [0, 0]
    for x in (zero, one):
        assert len(x["repairs_ms"]) == 1 and x["false_after_tap"] >= 3 and x["outputs"] == [0, 0], x
    for x in (two, corner):
        assert x["repairs_ms"] == [] and x["false_after_tap"] == 0, x
        assert x["false_before_tap"] == 0  # the strays reached only the veto: the rail stayed dark
        assert x["outputs"] == [0] and x["starts"] == 1, x  # the second token never came through
    assert two["relay_mv"] < -6.0 < one["relay_mv"], (one, two)
    assert stats["faults"] == 0 and stats["timeouts"] == 0  # a silent stall


def test_shipped_tap_repairs_after_strays_and_respects_live_rails():
    """The default build: the tap at START + ~111 ms. A failed ignition with 0-3 stray spikes
    is repaired nominally and at 1- and 2-sigma adverse corners, the relay within 2.5 mV of
    rest when the driver arrives (the pre-fix tap left 6.3 mV after two). A live false rail is
    never repaired, even at a 2-sigma corner favouring the repair (the seed-107 hazard: a
    repair into a live latch accelerated it), and a new request whose true rail rises 15-25 ms
    before the tap vetoes the repair (also at the favouring corner); the next START then
    serves that request."""
    strays = [_copy(strays=s) for s in (0, 1, 2, 3)] + [_copy(k=k, strays=s) for k in (1, 2) for s in (1, 2, 3)]
    live = [_copy(), _copy(k=-2), _copy(request_at=840), _copy(k=-2, request_at=880)]
    copies = live[:2] + strays + live[2:]
    pl, r, stats = _run(True, copies)
    assert pl.build_options["relight_repair_delay"] is True
    assert all(105 < x["tap_ms"] < 118 for x in r), [x["tap_ms"] for x in r]
    for x in r[:2]:  # live false rail
        assert x["repairs_ms"] == [] and x["outputs"] == [0, 0], x
    for c, x in zip(copies[2:-2], r[2:-2]):
        assert len(x["repairs_ms"]) == 1 and x["tap_ms"] < x["repairs_ms"][0] < x["tap_ms"] + 10, (c, x)
        assert x["relay_mv"] > -2.5, (c, x)
        assert x["false_after_tap"] >= 3 and x["outputs"] == [0, 0], (c, x)
    for x in r[-2:]:  # live true rail: vetoed; the pair reads "pending" and the cell starts again
        assert 15 <= x["request_lead_ms"] <= 25, x
        assert x["repairs_ms"] == [] and x["false_after_tap"] == 0, x
        assert x["starts"] >= 2 and x["outputs"] == [0, 0, 0], x
    assert stats["faults"] == 0 and stats["timeouts"] == 0


@pytest.mark.slow
def test_repair_margin_grid_before_and_after_the_fix():
    """0-3 injected strays at 0-3 sigma: the pre-fix tap loses 12 of the 16 corners, the shipped
    tap none. Two knife edges, each decided by <=0.1 ms, are not asserted: 2 sigma without
    strays before the fix (repaired) and 3 sigma with three strays after it (repaired).
    With the latch loop cut instead (every reader sees the spikes; the three-spike cut fires a
    fourth from the residual current at ~+55 ms), 1-2 spikes repair up to 3 sigma and the
    three-spike cut up to 1 sigma."""
    grid = [_copy(k=k, strays=s) for k in (0, 1, 2, 3) for s in (0, 1, 2, 3)]
    _, before, _ = _run(False, [_copy()] + grid)
    lost = {(c["k"], c["strays"]) for c, x in zip(grid, before[1:]) if not x["repairs_ms"]}
    assert lost - {(2, 0)} == {(0, 2), (0, 3), (1, 1), (1, 2), (1, 3), (2, 1), (2, 2), (2, 3),
                               (3, 0), (3, 1), (3, 2), (3, 3)}, sorted(lost)
    cuts = [_copy(k=k, cut=s) for k in (0, 1, 2, 3) for s in (1, 2)] + [_copy(k=k, cut=3) for k in (0, 1)]
    _, after, stats = _run(True, [_copy()] + grid + cuts)
    for c, x in zip(grid + cuts, after[1:]):
        if (c["k"], c["strays"]) == (3, 3):
            continue
        assert len(x["repairs_ms"]) == 1 and x["outputs"] == [0, 0], (c, x)
    assert [x["false_before_tap"] for x in after[1 + len(grid):]] == [1, 2] * 4 + [4, 4]
    assert stats["faults"] == 0 and stats["timeouts"] == 0
