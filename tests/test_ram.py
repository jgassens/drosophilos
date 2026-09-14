"""Data RAM on veto relays: exact decoding, overwrite, initial image, unwritten-read refusal."""

from drosophilos.lib.ram import build_ram_block, run_ram
from drosophilos.sim.model import Params

PARAMS = Params()


def test_write_read_overwrite_every_word():
    rb = build_ram_block(PARAMS, 8, 4)
    vals = [9, 3, 14, 0, 15, 6, 1, 10]
    ops = [("WRITE", a, v) for a, v in enumerate(vals)] + [("READ", a, v) for a, v in enumerate(vals)]
    ops += [("WRITE", 3, 7), ("READ", 3, 7), ("READ", 2, 14)]
    recs, sim, st = run_ram(rb, PARAMS, ops)
    assert st["ok"] == len(ops), [(r.op, r.addr, r.expected, r.decoded, r.status) for r in recs]
    assert all(r.fault_spikes == 0 and r.timeout_spikes == 0 for r in recs)
    print("RAM 8x4:", rb.net.n, "neurons; write done ms", round(st["latency_ms"][0]), "read done ms", round(st["latency_ms"][8]),
          "cycle ms", round(st["cycle_ms"][0]))


def test_initial_image_and_unwritten_read_is_refused():
    rb = build_ram_block(PARAMS, 8, 4)
    ops = [("READ", 1, 12), ("READ", 6, None), ("WRITE", 6, 5), ("READ", 6, 5), ("READ", 4, 2)]
    recs, sim, st = run_ram(rb, PARAMS, ops, init={1: 12, 4: 2})
    r0, r1, r2, r3, r4 = recs
    assert r0.status == "ok" and r0.decoded == 12  # loaded image
    assert r1.status == "refused" and r1.fault_spikes > 0 and r1.done_step is None  # both rails: fault, recovered
    assert r2.status == "ok" and r3.status == "ok" and r3.decoded == 5
    assert r4.status == "ok" and r4.decoded == 2
