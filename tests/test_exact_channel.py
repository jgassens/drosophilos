"""Stage C: an exact channel over FlyLink with the alternating-bit protocol on two neural
machines. A sends (seq | payload); B acknowledges at once on the reverse link, then checks
the sequence bit, consumes the payload, flips its expected bit and clears its port; A's send
timer retransmits when the ACK is late. A dropped rail event is repaired by the retransmit;
a duplicate is rejected by the sequence bit. Word layout (4 bits): bit 3 = sequence,
bits 0-2 = payload.

    A: 0 payload, 1 seq, 2 acked, 3 saved acc, 4 retries, 5 status (timer), 6 port in, 7 port out
    B: 0 expected seq, 1 received payload, 2 message, 3 saved acc, 4 deliveries, 5 pixel, 6 port in, 7 port out
"""

import numpy as np
import pytest

from drosophilos.cluster.flylink import run_linked, word_link
from drosophilos.lib.control import build_machine, load_image
from drosophilos.protocol.token import decode_at
from drosophilos.sim.model import Params
from drosophilos.sim.ref64 import RefSim

PARAMS = Params()


def sender():
    P = [None] * 64
    P[0:4] = [("MOV", 0), ("LOAD", 0), ("ORM", 1), ("STORE", 7)]  # send payload | seq
    P[4:8] = [("LOAD", 2), ("JNZ", 7), ("JMP", 4), ("HALT", 7)]  # wait until acked, then halt
    h = {16: ("STORE", 3), 17: ("LOAD", 5), 18: ("JZ", 28),  # save; timeout?
         19: ("MOV", 0), 20: ("STORE", 5), 21: ("LOAD", 4), 22: ("ADD", 1), 23: ("STORE", 4),  # status <- 0; retries++
         24: ("LOAD", 0), 25: ("ORM", 1), 26: ("STORE", 7), 27: ("JMP", 35),  # resend
         28: ("LOAD", 6), 29: ("AND", 8), 30: ("XORM", 1), 31: ("JNZ", 34),  # ack's seq == seq ?
         32: ("MOV", 1), 33: ("STORE", 2),  # acked
         34: ("CLR", 6), 35: ("LOAD", 3), 36: ("IRET", 0)}
    for k, v in h.items():
        P[k] = v
    return [P[k] if P[k] is not None else ("HALT", k) for k in range(64)]


def receiver():
    P = [None] * 32
    P[0:3] = [("MOV", 0), ("JMP", 2), ("JMP", 1)]  # idle loop (safe points keep coming)
    h = {8: ("STORE", 3), 9: ("LOAD", 6), 10: ("STORE", 2), 11: ("AND", 8), 12: ("STORE", 7),  # ack at once
         13: ("XORM", 0), 14: ("JNZ", 25),  # duplicate -> no delivery
         15: ("LOAD", 2), 16: ("AND", 7), 17: ("STORE", 1), 18: ("STORE", 5),  # payload, pixel
         19: ("LOAD", 4), 20: ("ADD", 1), 21: ("STORE", 4),  # deliveries++
         22: ("LOAD", 0), 23: ("XOR", 8), 24: ("STORE", 0),  # expected ^= 8
         25: ("CLR", 6), 26: ("LOAD", 3), 27: ("IRET", 0)}
    for k, v in h.items():
        P[k] = v
    return [P[k] if P[k] is not None else ("HALT", k) for k in range(32)]


def _word_value(sim, word, at_end=True):
    ev = sim.trace.events
    st = ev["step"][ev["neuron"] == word.completion.u]
    if not len(st):
        return None
    return decode_at(sim.trace, word.rail_taps, int(st[-1]), 94)[0]


def _run(faults_ab=None, payload=5, seq=8, max_ms=45000, delay_ba_steps=200):
    mA = build_machine(PARAMS, n=4, n_prog=64, n_data=8, handler_pc=16, port_in_word=6, port_out_word=7, timer_hops=1900, status_word=5)
    mB = build_machine(PARAMS, n=4, n_prog=32, n_data=8, handler_pc=8, port_in_word=6, port_out_word=7)
    simA, simB = RefSim(mA.net.topology(), PARAMS), RefSim(mB.net.topology(), PARAMS)
    load_image(simA, mA, sender(), {0: payload, 1: seq, 2: 0, 4: 0})
    load_image(simB, mB, receiver(), {0: seq, 4: 0})
    ab = word_link(mA, mB, 200, 0, 1, faults=faults_ab)
    ba = word_link(mB, mA, delay_ba_steps, 1, 0)
    run_linked([simA, simB], [ab, ba], int(max_ms / PARAMS.dt))
    got = {name: _word_value(simA, mA.dmem.words[k]) for name, k in (("acked", 2), ("retries", 4))}
    got.update({name: _word_value(simB, mB.dmem.words[k]) for name, k in (("received", 1), ("deliveries", 4), ("pixel", 5), ("expected", 0))})
    return got, ab, ba, mA, mB, simA, simB


@pytest.mark.slow
def test_clean_link_delivers_once_and_acks():
    got, ab, ba, *_ = _run()
    assert got["received"] == 5 and got["pixel"] == 5 and got["deliveries"] == 1, got
    assert got["acked"] == 1 and (got["retries"] or 0) == 0, got
    assert got["expected"] == 0  # flipped from 8
    print("exact channel, clean:", got, "events A->B", len(ab.log), "B->A", len(ba.log))


@pytest.mark.slow
def test_dropped_event_is_repaired_by_retransmit():
    got, ab, ba, *_ = _run(faults_ab={"drop_first": 1})
    assert got["received"] == 5 and got["deliveries"] == 1 and got["acked"] == 1, got
    assert got["retries"] == 1, got  # one timeout, one resend
    print("exact channel, one dropped rail event:", got, "injected", ab.injected)


@pytest.mark.slow
def test_late_ack_after_timeout_is_queued_and_cancels_only_when_consumed():
    """The ACK lands after the original send timer expires but before its interrupt is
    taken. The timeout remains the first handler invocation and the arrival is queued for a
    second invocation; merely holding the ACK word must not cancel the retransmit timer."""
    got, ab, ba, mA, _, simA, _ = _run(delay_ba_steps=45000)
    ev = simA.trace.events

    def rises(role):
        steps = ev["step"][ev["neuron"] == mA.net.roles.index(role)]
        gap = 3 * mA.drive.loop_period_steps
        return [int(s) for i, s in enumerate(steps) if i == 0 or s - steps[i - 1] > gap]

    timeout = rises("TIMER.h1899")[0]
    arrival = rises("LINK.in.arrive.edge")[0]
    first_clear = rises("INTP.clear.edge")[0]
    handler_entries = rises("PC.16.u")
    assert timeout + 600 < arrival < first_clear, (timeout, arrival, first_clear)
    assert len(handler_entries) == 2, handler_entries
    assert got["received"] == 5 and got["deliveries"] == 1, got
    assert got["acked"] == 1 and got["retries"] == 1, got
    print("exact channel, late ACK:", got, "timer/arrival/clear", timeout, arrival, first_clear,
          "events A->B", len(ab.log), "B->A", len(ba.log))
