"""Executable abstract state machine of spec.md (no neurons; pure protocol logic).

Global state is immutable; `step(state, event)` returns the next state or raises
IllegalTransition. `enabled(state)` lists the events that may fire. Faults and losses are
modelled as explicit events so the explorer can inject them.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import FrozenSet

E, ZERO, ONE, F = "E", "0", "1", "F"


class IllegalTransition(Exception):
    pass


@dataclass(frozen=True)
class Token:
    kind: str  # "DATA" | "ACCEPT" | "CLEARED" | "READY"
    seq: int  # word number the token belongs to
    bit: int = -1  # DATA only
    rail: int = -1  # DATA only
    phase: int = -1  # DATA only, when phase rails are used (seq % 2)


@dataclass(frozen=True)
class State:
    width: int
    p_state: str  # IDLE, LOADED, SENT, CLEARING, AWAIT_READY
    p_word: tuple  # bits of the loaded word, or ()
    p_seq: int  # number of the word P is handling
    q_state: str  # EMPTY, PARTIAL, COMPLETE, AWAIT_CLEARED, RESETTING
    latches: tuple  # per bit: E/0/1/F
    q_seq: int  # number of the word Q expects next
    inflight: FrozenSet[Token]
    consumed: tuple  # (seq, value-tuple) in order
    faults: int
    stale_accepted: int  # saturating flag: a token from word n acted during word m != n
    words: tuple  # the words P will send, in order
    phase_rails: bool
    timing_bound: bool

    def enabled(self) -> list[tuple]:
        ev = []
        if self.p_state == "IDLE" and self.p_seq < len(self.words):
            ev.append(("load",))
        if self.p_state == "LOADED":
            ev.append(("send",))
        if self.p_state == "CLEARING":
            ev.append(("p_cleared",))
        if self.q_state == "COMPLETE":
            ev.append(("consume",))
        if self.q_state == "RESETTING":
            ev.append(("q_ready",))
        for t in sorted(self.inflight, key=lambda t: (t.kind, t.seq, t.bit, t.rail)):
            ev.append(("deliver", t))
        return ev


def initial(words: tuple, width: int, phase_rails: bool = False, timing_bound: bool = True) -> State:
    return State(width, "IDLE", (), 0, "EMPTY", tuple([E] * width), 0, frozenset(), (), 0, 0,
                 tuple(tuple(w) for w in words), phase_rails, timing_bound)


def step(s: State, event: tuple) -> State:
    kind = event[0]
    if kind == "load":
        if s.p_state != "IDLE":
            raise IllegalTransition("load outside IDLE")
        return replace(s, p_state="LOADED", p_word=s.words[s.p_seq])
    if kind == "send":
        if s.p_state != "LOADED":
            raise IllegalTransition("send outside LOADED")
        toks = {Token("DATA", s.p_seq, i, s.p_word[i], s.p_seq % 2) for i in range(s.width)}
        return replace(s, p_state="SENT", inflight=s.inflight | toks)
    if kind == "p_cleared":
        if s.p_state != "CLEARING":
            raise IllegalTransition("p_cleared outside CLEARING")
        return replace(s, p_state="AWAIT_READY", p_word=(), inflight=s.inflight | {Token("CLEARED", s.p_seq)})
    if kind == "consume":
        if s.q_state != "COMPLETE":
            raise IllegalTransition("consume outside COMPLETE")
        value = tuple(int(v) for v in s.latches)
        return replace(s, q_state="AWAIT_CLEARED", consumed=s.consumed + ((s.q_seq, value),),
                       inflight=s.inflight | {Token("ACCEPT", s.q_seq)})
    if kind == "q_ready":
        if s.q_state != "RESETTING":
            raise IllegalTransition("q_ready outside RESETTING")
        return replace(s, q_state="EMPTY", latches=tuple([E] * s.width), q_seq=s.q_seq + 1,
                       inflight=s.inflight | {Token("READY", s.q_seq + 1)})
    if kind == "deliver":
        t: Token = event[1]
        rest = s.inflight - {t}
        if t.kind == "DATA":
            return _deliver_data(replace(s, inflight=rest), t)
        return _deliver_control(replace(s, inflight=rest), t)
    raise IllegalTransition(f"unknown event {event}")


def _stale_control(s: State, t: Token, expected: int) -> State | None:
    """Control token from an earlier transaction. Dropped under a protection mechanism;
    otherwise indistinguishable from a current token (returns None: act as current)."""
    if t.seq > expected:
        raise IllegalTransition(f"{t.kind} seq {t.seq} from the future (expected {expected})")
    if t.seq < expected:
        if s.phase_rails or s.timing_bound:
            return s
        return None
    return None


def _deliver_control(s: State, t: Token) -> State:
    if t.kind == "ACCEPT":
        dropped = _stale_control(s, t, s.p_seq)
        if dropped is not None:
            return dropped
        if s.p_state == "SENT":
            return replace(s, p_state="CLEARING", stale_accepted=min(s.stale_accepted + int(t.seq < s.p_seq), 1))
        if s.p_state in ("CLEARING", "AWAIT_READY", "IDLE", "LOADED"):
            # duplicate (idempotent) or, unprotected and stale, harmless here
            return s
        raise IllegalTransition(f"ACCEPT in P state {s.p_state}")
    if t.kind == "CLEARED":
        dropped = _stale_control(s, t, s.q_seq)
        if dropped is not None:
            return dropped
        if s.q_state == "AWAIT_CLEARED":
            return replace(s, q_state="RESETTING", stale_accepted=min(s.stale_accepted + int(t.seq < s.q_seq), 1))
        if s.q_state in ("RESETTING", "EMPTY", "PARTIAL", "COMPLETE"):
            return s  # duplicate / stale without effect on state
        raise IllegalTransition(f"CLEARED in Q state {s.q_state}")
    if t.kind == "READY":
        expected = s.p_seq + 1 if s.p_state == "AWAIT_READY" else s.p_seq
        dropped = _stale_control(s, t, expected)
        if dropped is not None:
            return dropped
        if s.p_state == "AWAIT_READY":
            return replace(s, p_state="IDLE", p_seq=s.p_seq + 1)
        return s  # duplicate READY
    raise IllegalTransition(f"unknown token {t.kind}")


def _deliver_data(s: State, t: Token) -> State:
    # DATA during RESETTING is dropped (latches inhibited)
    if s.q_state == "RESETTING":
        return s
    stale = t.seq != s.q_seq
    if stale:
        if t.seq > s.q_seq:
            raise IllegalTransition(f"DATA seq {t.seq} from the future (q_seq {s.q_seq})")
        if s.phase_rails and t.phase != s.q_seq % 2:
            return s  # wrong phase rail is disabled: dropped
        if s.timing_bound:
            return s  # cannot happen by assumption: dropped
        s = replace(s, stale_accepted=1)  # unprotected: taken as current data (saturating flag)
    if s.q_state in ("COMPLETE", "AWAIT_CLEARED"):
        # late spike after apparent completion
        held = s.latches[t.bit]
        if held in (ZERO, ONE) and held != str(t.rail):
            latches = list(s.latches)
            latches[t.bit] = F
            if s.q_state == "COMPLETE":  # not yet consumed: never will be; FAULT-ACCEPT instead
                return replace(s, latches=tuple(latches), faults=min(s.faults + 1, 3), q_state="AWAIT_CLEARED",
                               inflight=s.inflight | {Token("ACCEPT", s.q_seq)})
            return replace(s, latches=tuple(latches), faults=min(s.faults + 1, 3))  # consumed already: flag only
        return s  # same-rail duplicate: absorbed
    return _latch(s, t)


def _latch(s: State, t: Token) -> State:
    latches = list(s.latches)
    held = latches[t.bit]
    if held == E:
        latches[t.bit] = str(t.rail)
    elif held == str(t.rail):
        pass  # same-rail duplicate
    elif held == F:
        pass
    else:
        latches[t.bit] = F
    s2 = replace(s, latches=tuple(latches))
    if F in s2.latches:
        # fault: never consumed; proceed to reset via FAULT-ACCEPT
        return replace(s2, q_state="AWAIT_CLEARED", faults=min(s2.faults + 1, 3),
                       inflight=s2.inflight | {Token("ACCEPT", s2.q_seq)})
    if all(v in (ZERO, ONE) for v in s2.latches):
        return replace(s2, q_state="COMPLETE")
    return replace(s2, q_state="PARTIAL")


def terminal(s: State) -> bool:
    return s.p_state == "IDLE" and s.q_state == "EMPTY" and not s.inflight and s.p_seq == len(s.words)
