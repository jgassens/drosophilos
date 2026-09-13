"""Exhaustive exploration of the abstract protocol machine (spec.md §5).

Breadth-first over every interleaving of enabled events, optionally injecting faults:
  stale DATA (a copy of a word-n bit delivered later), duplicate tokens, late opposite-rail
  spikes, lost tokens (with and without retransmission).
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, replace

from .machine import E, F, IllegalTransition, State, Token, initial, step, terminal


@dataclass
class Report:
    states: int
    terminal_states: int
    deadlocks: list
    double_consumes: int
    stale_accepted_max: int
    illegal: list
    faults_max: int
    consumed_sets: set


def explore(words, width, *, phase_rails=False, timing_bound=True, inject_stale=False,
            inject_duplicates=False, inject_late_opposite=False, lose=None, max_states=200_000) -> Report:
    s0 = initial(words, width, phase_rails, timing_bound)
    seen = {s0}
    queue = deque([s0])
    deadlocks, illegal = [], []
    double = 0
    stale_max = 0
    faults_max = 0
    terminals = 0
    consumed_sets = set()
    while queue:
        s = queue.popleft()
        if len(seen) > max_states:
            raise RuntimeError("state space too large")
        events = s.enabled()
        # fault injections (each a possible extra event)
        extra = []
        if inject_stale:
            # a delayed copy of a DATA bit from the previous word may arrive any time
            for (seq, val) in s.consumed:
                if seq == s.q_seq - 1:
                    extra.append(("deliver", Token("DATA", seq, 0, val[0], seq % 2)))
        if inject_duplicates:
            for t in s.inflight:
                extra.append(("dup", t))
        if inject_late_opposite and s.q_state in ("COMPLETE", "AWAIT_CLEARED"):
            held = s.latches[0]
            if held in ("0", "1"):
                extra.append(("deliver", Token("DATA", s.q_seq, 0, 1 - int(held), s.q_seq % 2)))
        if lose:
            for t in s.inflight:
                if t.kind in lose:
                    extra.append(("lose", t))
        if terminal(s):
            terminals += 1
            consumed_sets.add(s.consumed)
        elif not events and not extra:
            deadlocks.append(s)
        for ev in events + extra:
            try:
                if ev[0] == "dup":
                    t = ev[1]
                    nxt = step(replace(s, inflight=s.inflight - {t}), ("deliver", t))
                    nxt = replace(nxt, inflight=nxt.inflight | {t})  # original still in flight
                elif ev[0] == "lose":
                    nxt = replace(s, inflight=s.inflight - {ev[1]})
                else:
                    nxt = step(s, ev)
            except IllegalTransition as exc:
                illegal.append((s, ev, str(exc)))
                continue
            seqs = [c[0] for c in nxt.consumed]
            if len(seqs) != len(set(seqs)):
                double += 1
            stale_max = max(stale_max, nxt.stale_accepted)
            faults_max = max(faults_max, nxt.faults)
            if nxt not in seen:
                seen.add(nxt)
                queue.append(nxt)
    return Report(len(seen), terminals, deadlocks, double, stale_max, illegal, faults_max, consumed_sets)
