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
    net.synapse(edge, trigger, drive.reset)
    return trigger, inh, edge


def add_edge_relay(net: Netlist, drive: Drive, name: str, source: int, strength: float = 2.2) -> int:
    """A relay that fires exactly once per activation of `source`, however long the source's
    train lasts (feed-forward inhibition from the same source). Used so that DATA and gate
    outputs ignite a latch once instead of driving it continuously, which would push the
    latch above the standard train rate that every rate-mode gate assumes.

    The relay gets a head start over its inhibitor (relay_in = 1.15x loop drive, still
    doublet-free, fires ~2.5 ms after the source's first spike; the inhibitor gets the plain
    pulse drive and its inhibition lands ~5.3 ms after). With equal drives the two raced and
    5 % weight noise made the relay lose about 1 % of the time, dropping a bit (measured).
    The inhibition (2.2x loop per spike) then outweighs the relay's drive for the rest of the
    train; the relay recovers from the resulting after-hyperpolarisation in ~50 ms, which
    every relay here gets (its source restarts >= 100 ms later)."""
    relay = net.neuron(f"{name}.edge")
    inh = net.neuron(f"{name}.edge_inh")
    net.synapse(source, relay, drive.relay_in)
    net.synapse(source, inh, drive.pulse)
    net.synapse(inh, relay, -int(round(strength * drive.loop)))
    return relay


def connect_trigger(net: Netlist, drive: Drive, source: int, trigger: int, edge: int) -> None:
    """Edge detector: the source's first spike fires the trigger; from the second spike on,
    `edge` (driven by the same source) holds the trigger down for as long as the train lasts."""
    net.synapse(source, trigger, drive.pulse)
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
