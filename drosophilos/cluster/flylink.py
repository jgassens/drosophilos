"""FlyLink v0: address-event transport between simulated nodes (plan §Cluster, Stage C).

The host's role is transport only: after every simulation step it reads the spikes of a
node's export neurons (port relays that fire once per rail rise of an output word) and
schedules each as an event on the destination node's mapped neuron after the link delay,
with the mapped drive. The packet is (source node, source neuron, step); the receiver's
mapping B_ji is fixed at link creation. Nothing in the packet is a computed game value: the
destination word is rebuilt from rail events and validated by its own completion tree.

v0 carries one message per input word (the exact-channel machinery, credits, epochs and
retransmit, is the rest of Stage C).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Link:
    src: int  # node index in `sims`
    dst: int
    mapping: dict  # source neuron -> (destination neuron, quanta)
    delay_steps: int
    log: list = field(default_factory=list)  # (step, src neuron, dst neuron) packets


def run_linked(sims: list, links: list[Link], n_steps: int, on_step=None) -> None:
    """Steps every simulator in lockstep and transports events across the links. Modeled
    positive delays give the conservative ordering the plan requires: an event scheduled at
    step s arrives at s + delay, never earlier than the sender's own clock."""
    for _ in range(n_steps):
        for sim in sims:
            sim.step()
        s = sims[0].step_index - 1
        for ln in links:
            src = sims[ln.src]
            if not src._spk_step or src._spk_step[-1][0] != s:
                continue
            fired = src._spk_neuron[-1].tolist()
            for n_ in fired:
                m = ln.mapping.get(int(n_))
                if m is None:
                    continue
                dn, q = m
                sims[ln.dst].add_events(0, [s + ln.delay_steps], [dn], [q])
                ln.log.append((s, int(n_), dn))
        if on_step is not None:
            on_step(s)


def word_link(src_machine, dst_machine, delay_steps: int) -> Link:
    """Map the source machine's output-port relays onto the destination's input-port word
    rails, rail for rail, at the ignition drive."""
    assert src_machine.link_out_taps and dst_machine.link_in_word is not None
    word = dst_machine.dmem.words[dst_machine.link_in_word]
    mapping = {}
    for i, pair in enumerate(src_machine.link_out_taps):
        for r in (0, 1):
            mapping[pair[r]] = (word.rails[i][r].u, dst_machine.drive.ignite)
    return Link(0, 1, mapping, delay_steps)
