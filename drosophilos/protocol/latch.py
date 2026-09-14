"""Bit latch: a two-neuron excitatory loop holding a circulating spike.

Ignite with one pulse of >= drive.loop quanta into `u`; the value is held as a 213 Hz
train on both members until reset inhibition (drive.reset on both members) stops it.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..lib.netlist import Drive, Netlist


@dataclass(frozen=True)
class Latch:
    u: int  # input member and output tap
    v: int  # partner

    @property
    def members(self) -> tuple[int, int]:
        return (self.u, self.v)


def add_latch(net: Netlist, drive: Drive, name: str) -> Latch:
    u = net.neuron(f"{name}.u")
    v = net.neuron(f"{name}.v")
    net.synapse(u, v, drive.loop)
    net.synapse(v, u, drive.loop)
    return Latch(u, v)


def add_reset(net: Netlist, drive: Drive, name: str, latches: list[Latch], gates: list[int] = (),
              pulses: int = 4, strength: float = 0.75) -> tuple[int, int, int]:
    """Reset controller. `trigger` fires once per activation of its source (edge detector,
    see connect_trigger) and starts a short relay chain; the inhibitory neuron `inh` fires
    once per relay, i.e. `pulses` times ~5.3 ms apart, each delivering `strength` x loop
    drive of inhibition to both members of every latch and to every gate neuron.

    Why a train and not one strong pulse: one pulse blocks re-ignition for only ~5 ms
    (tau_s), but a latch -> rate-mode gate -> latch chain keeps settling for 10-15 ms, so a
    tree latch ignited 8 ms after a single pulse survives (observed). Why half strength:
    the members' after-hyperpolarisation, which the membrane sheds with tau_m = 20 ms, grows
    with the total inhibitory charge; four half pulses cost about what one 1.5x pulse costs
    and kill at every phase (measured: both members, 4 pulses, 0.5x -> 50/50).
    Returns (trigger, inh, edge)."""
    trigger = net.neuron(f"{name}.reset")
    inh = net.neuron(f"{name}.reset_inh")
    edge = net.neuron(f"{name}.reset_edge")
    prev = trigger
    net.synapse(trigger, inh, drive.pulse)
    for k in range(1, pulses):
        r = net.neuron(f"{name}.reset_relay{k}")
        net.synapse(prev, r, drive.pulse)
        net.synapse(r, inh, drive.pulse)
        prev = r
    q = -int(round(strength * drive.loop))
    for l in latches:
        for x in l.members:
            net.synapse(inh, x, q)
    for g in gates:
        net.synapse(inh, g, q)
    # the edge inhibitor must outweigh the trigger's (head-started, 1.29x loop) drive for the
    # whole source train under +-8 % weight noise: 2.2x loop, as for edge relays. At 1.5x the
    # trigger fired at the train's rate in ~1 node per 1000 and CLEARED became a train.
    net.synapse(edge, trigger, -int(round(2.2 * drive.loop)))
    return trigger, inh, edge


def add_edge_relay(net: Netlist, drive: Drive, name: str, source: int, strength: float = 2.2, hold_from=(),
                   hold_strength: float = 0.25) -> int:
    """A relay that fires exactly once per activation of `source`, however long the source's
    train lasts (feed-forward inhibition from the same source). Used so that DATA and gate
    outputs ignite a latch once instead of driving it continuously, which would push the
    latch above the standard train rate that every rate-mode gate assumes.

    The relay gets a head start over its inhibitor (relay_in = 1.8x single-pulse need, the
    largest doublet-free drive, fires ~1.8 ms after the source's first spike; the inhibitor
    gets the plain 1.4x pulse drive and its inhibition lands ~5.3 ms after). With equal
    drives the two raced and 5 % weight noise made the relay lose ~1 % of the time; at 1.15x
    loop it still lost ~0.1 % of the time at 4 % noise (campaign), dropping a bit each time.
    The inhibition (2.2x loop per spike) then outweighs the relay's drive for the rest of the
    train; the relay recovers from the resulting after-hyperpolarisation in ~50 ms, which
    every relay here gets (its source restarts >= 100 ms later).

    `hold_from`: extra trains that drive the inhibitor. The source-train inhibition only holds
    the relay for a fast source (a 213 Hz latch train); a rate-mode gate fires at ~25-40 Hz,
    the inhibition (peak ~22 mV, tau_m) has decayed to ~5 mV by the next gate spike, and the
    relay fires again: measured every ~43 ms for a whole transaction. Each re-fire is an extra
    ignition pulse into a running latch, which then reads ~10 % fast to every rate-mode gate
    downstream and tips one-input ANDs (the ALU's 1 % fault rate; likely the adder's residual
    3e-4). A gate -> latch relay therefore also takes the ignited latch's own train, through a
    separate light interneuron (hold_strength x loop per spike, 0.25x: ~17 mV sustained, which
    blocks the 12.6 mV re-fire pulse with ~11 mV to spare). It must be light: the relay's head
    start over its own inhibitor is ~5 ms, so a residual hyperpolarisation above ~2 mV at the
    next activation loses that race. Through the 2.2x inhibitor the hold sat at ~146 mV and
    needed ~86 ms to decay, and a consumer reset is only ~85 ms before the next load: every
    relay whose target had held in the previous transaction then failed (measured)."""
    relay = net.neuron(f"{name}.edge")
    inh = net.neuron(f"{name}.edge_inh")
    net.synapse(source, relay, drive.relay_in)
    net.synapse(source, inh, drive.pulse)
    net.synapse(inh, relay, -int(round(strength * drive.loop)))
    if hold_from:
        hinh = net.neuron(f"{name}.hold_inh")
        for h in hold_from:
            net.synapse(h, hinh, drive.pulse)
        net.synapse(hinh, relay, -int(round(hold_strength * drive.loop)))
    return relay


def connect_trigger(net: Netlist, drive: Drive, source: int, trigger: int, edge: int) -> None:
    """Edge detector: the source's first spike fires the trigger; from the second spike on,
    `edge` (driven by the same source) holds the trigger down for as long as the train lasts.
    The trigger gets the relay_in head start (1.8x need) so that it always beats its own
    inhibitor: with equal drives, a trigger whose threshold came out +0.5 mV lost the race
    and the consumer never reset (campaign failure class no_ready)."""
    net.synapse(source, trigger, drive.relay_in)
    net.synapse(source, edge, drive.pulse)


def add_ready(net: Netlist, drive: Drive, name: str, trigger: int, veto_latches: list[Latch] | None = None,
              hops: int = 11, veto_by_trigger: bool = False) -> int:
    """READY / CLEARED generator: a pure delay chain from the reset trigger (~5.3 ms per hop,
    ~48 ms at 9 hops). Bounded-delay assumption: the chain delay exceeds the reset settling
    time (~5 ms) plus the latch members' recovery from one reset pulse (a -93 mV pulse dips
    the membrane ~15 mV; ~33 ms later a fresh ignition pulse reaches threshold again).
    An output-side veto ("block READY while any latch is active") was tried and rejected:
    its early pulses hyperpolarise the READY neuron itself and the chain pulse then misses
    threshold. Reset failure is instead detected by the fault gates and the decoder."""
    prev = trigger
    for k in range(hops):
        d = net.neuron(f"{name}.ready_delay{k}")
        net.synapse(prev, d, drive.pulse)
        prev = d
    out = net.neuron(f"{name}.ready")
    net.synapse(prev, out, drive.pulse)
    return out
