"""Flip-flop latch: two INHIBITORY neurons, each driven above threshold by a tonic bias,
inhibiting each other so that exactly one fires at a time.

    SET   : u fires (a ~213 Hz train, the excitatory Latch's rail statistics), v silent
    CLEAR : v fires, u silent

Why it exists: the fly has 358 / 1,105 / 3,717 disjoint strong mutually inhibiting pairs
with drivers (docs/capacity_doom.md §5), on quiet neurons the excitatory latch cannot use.
Bias is the tonic driver (mV added to the resting potential; Netlist.neuron(bias=...)).

Measured contract: docs/contracts/flipflop.yaml, docs/a1_flipflop.md. The numbers below
are all-phase minima at the default inhibition (1.0x loop per spike):

  bias 58 mV       -> 212.8 Hz tonic (period 47 steps, the latch's loop period exactly)
  inhibition       : bistable from 0.8x loop; 1.0x is the default (a 25 % margin)
  parked member    : ~15 mV below threshold (mean); one standard ignite pulse (12.6 mV) into
                     it does NOT fire it, and one spike of it does not stop the other, so
                     the flip-flop switches on TRAINS or on pulses 2.25-4x the standard:
  SET              : 3 x 0.75 ignite 5.3 ms apart, or one 2.25x ignite pulse into u
  CLEAR            : 3 x 1.0 loop 5.3 ms apart, or one 4x loop pulse into u
  power-on         : both members start at rest and, left alone, fire in LOCKSTEP for ever
                     (a stable third state, ~100 Hz each); one 1.0x loop inhibitory pulse
                     into u at step 0 starts it CLEAR with u never firing
  margins          : silent member survives a stray 1.8x ignite pulse (latch: ignites at
                     1.1x need); firing member survives a stray 2.25x loop pulse (latch's
                     single member: survives 2x, dies at 3x)

The defaults of set_pulse / clear_pulse carry a margin over those minima: SET is three
standard ignite pulses (1.0x, minimum 0.75x) or one 3x ignite pulse (minimum 2.25x); CLEAR
is three standard reset pulses (1.5x loop, minimum 1.0x) or one 5x loop pulse (minimum 4x).
"""

from __future__ import annotations

from dataclasses import dataclass

from ..lib.netlist import Drive, Netlist

#: Bias (mV above rest) at which a lone neuron fires with the latch's 47-step period (212.8 Hz).
BIAS_213HZ_MV = 58.0
#: Spacing of the pulses of a switching train, steps (the reset controller's relay hop, ~5.3 ms).
TRAIN_SPACING_STEPS = 53


@dataclass(frozen=True)
class FlipFlop:
    u: int  # fires while SET; the rail every reader taps (as Latch.u)
    v: int  # fires while CLEAR
    bias_mv: float
    inh_quanta: int

    @property
    def members(self) -> tuple[int, int]:
        return (self.u, self.v)


def add_flipflop(net: Netlist, drive: Drive, name: str, bias_mv: float = BIAS_213HZ_MV,
                 inh_quanta: int | None = None) -> FlipFlop:
    """Two biased inhibitory neurons in mutual inhibition. `inh_quanta` (positive; default
    drive.loop) is the strength of each member's inhibition of the other per spike."""
    q = drive.loop if inh_quanta is None else int(inh_quanta)
    u = net.neuron(f"{name}.u", bias=bias_mv)
    v = net.neuron(f"{name}.v", bias=bias_mv)
    net.synapse(u, v, -q)
    net.synapse(v, u, -q)
    return FlipFlop(u, v, float(bias_mv), q)


def set_pulse(ff: FlipFlop, drive: Drive, pulses: int = 3) -> list[tuple[int, int, int]]:
    """What to inject to SET: a list of (offset_steps, neuron, quanta). `pulses` = 3: three
    standard ignite pulses into u, 5.3 ms apart (all-phase minimum 3 x 0.75 ignite);
    `pulses` = 1: one 3x ignite pulse (minimum 2.25x). Excitation into u alone; nothing into
    v (u's own spikes stop v)."""
    if pulses == 1:
        return [(0, ff.u, 3 * drive.ignite)]
    return [(k * TRAIN_SPACING_STEPS, ff.u, drive.ignite) for k in range(int(pulses))]


def clear_pulse(ff: FlipFlop, drive: Drive, pulses: int = 3) -> list[tuple[int, int, int]]:
    """What to inject to CLEAR: `pulses` = 3: three standard reset pulses (drive.reset, -1.5x
    loop) into u 5.3 ms apart (all-phase minimum 3 x 1.0 loop; the standard 4 x 0.75 loop
    reset train also clears it, with ~7 % margin); `pulses` = 1: one 5x loop pulse (minimum
    4x). Inhibition into u alone: the same train into BOTH members (what add_reset sends to
    a latch's members) leaves the flip-flop in the state it was in (measured: SET stays SET
    at 12 of 12 phases, for 4 x 0.75 and 3 x 1.5 loop)."""
    if pulses == 1:
        return [(0, ff.u, -5 * drive.loop)]
    return [(k * TRAIN_SPACING_STEPS, ff.u, drive.reset) for k in range(int(pulses))]


def power_on_pulse(ff: FlipFlop, drive: Drive) -> list[tuple[int, int, int]]:
    """Inject at step 0 to start CLEAR: one loop-strength inhibitory pulse into u. Without it
    both members fire together from step 25 on and stay in lockstep (measured, 300 ms)."""
    return [(0, ff.u, -drive.loop)]


def schedule(sim, ff_events: list[tuple[int, int, int]], at_step: int, node: int = 0) -> None:
    """Add a set/clear/power-on pulse list to a simulator at `at_step`."""
    for off, neuron, q in ff_events:
        sim.add_events(node, [at_step + off], [neuron], [q])


def add_set_chain(net: Netlist, drive: Drive, name: str, ff: FlipFlop, pulses: int = 3,
                  strength: float = 1.0) -> int:
    """Adapter from the protocol's one-shot sources to the flip-flop's train: a trigger
    neuron `name.set` that, fired once (an ignite pulse from an edge or veto relay), starts
    a relay chain whose every member delivers `strength` x ignite to u, `pulses` pulses
    ~5.3 ms apart (the reset controller's construction, add_reset, mirrored to excitation).
    Returns the trigger. `net.synapse(relay, chain, drive.ignite)` is the flip-flop's
    replacement for `net.synapse(relay, latch.u, drive.ignite)`."""
    trigger = net.neuron(f"{name}.set")
    q = int(round(strength * drive.ignite))
    net.synapse(trigger, ff.u, q)
    prev = trigger
    for k in range(1, pulses):
        r = net.neuron(f"{name}.set_relay{k}")
        net.synapse(prev, r, drive.pulse)
        net.synapse(r, ff.u, q)
        prev = r
    return trigger


def add_clear_chain(net: Netlist, drive: Drive, name: str, flipflops: list[FlipFlop], pulses: int = 3,
                    strength: float = 1.5) -> tuple[int, int]:
    """add_reset's controller aimed at the flip-flops' u members only: `trigger` fires once,
    `inh` fires once per relay (`pulses` times ~5.3 ms apart) and delivers `strength` x loop
    of inhibition to every flip-flop's u. Returns (trigger, inh). Not to both members: a
    train into both leaves the flip-flop where it was (measured), which is why add_reset
    itself cannot clear one."""
    trigger = net.neuron(f"{name}.clear")
    inh = net.neuron(f"{name}.clear_inh")
    prev = trigger
    net.synapse(trigger, inh, drive.pulse)
    for k in range(1, pulses):
        r = net.neuron(f"{name}.clear_relay{k}")
        net.synapse(prev, r, drive.pulse)
        net.synapse(r, inh, drive.pulse)
        prev = r
    q = -int(round(strength * drive.loop))
    for f in flipflops:
        net.synapse(inh, f.u, q)
    return trigger, inh


def connect_clear(net: Netlist, drive: Drive, inh: int, flipflops: list[FlipFlop], strength: float = 1.5) -> None:
    """Aim an existing inhibitory train neuron (a register's reset controller `inh`, which
    fires once per reset relay) at the flip-flops' u members only, `strength` x loop per
    spike. This is add_clear_chain without its own controller: the register's reset train
    (4 pulses) delivers 4 x 1.5x loop into u, above the 3 x 1.0x all-phase minimum, and
    nothing into v (a train into both members leaves the flip-flop where it was)."""
    q = -int(round(strength * drive.loop))
    for f in flipflops:
        net.synapse(inh, f.u, q)


def flipflop_taps(net: Netlist) -> list[int]:
    """The u members of every flip-flop in the netlist: biased neurons whose role ends in
    `.u` (add_flipflop names them so; nothing else in the library carries a bias)."""
    return [i for i, (role, b) in enumerate(zip(net.roles, net.bias)) if b != 0.0 and role.endswith(".u")]


def power_on_events(net: Netlist, drive: Drive, at_step: int = 0) -> list[tuple[int, int, int]]:
    """What a harness injects when it loads an image: one loop-strength inhibitory pulse into
    every flip-flop's u at `at_step`, so each starts CLEAR instead of in lockstep (see
    power_on_pulse). Returns (step, neuron, quanta) triples; `schedule`-compatible after
    subtracting `at_step`, or fed straight to `sim.add_events`."""
    return [(at_step, u, -drive.loop) for u in flipflop_taps(net)]
