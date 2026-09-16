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

Readout on the connectome (add_flipflop(proxy=True)): u is inhibitory, so no host can carry an
excitatory read of it. p, an EXCITATORY neuron on the same bias inhibited by v at 1.0x loop,
fires the same 212.9 Hz train exactly while SET and is what a reader connects to (ff.p /
ff.rail): edge relay once per SET, veto held while SET. It costs one park recovery each way:
p rises 18.5-23.1 ms after the first SET pulse (u: 6.0-6.4) and stops <= 17.8 ms after the
first CLEAR pulse (u: at once); a veto through p needs the SET train >= 22 ms before the
driver (u: 10) and re-arms 60 ms after CLEAR (u: 45). Stray 1.1x ignite into p while CLEAR
never fires it; 1.15x gives one p spike (one relay pulse) and leaves the pair untouched.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..lib.netlist import Drive, Netlist

#: Bias (mV above rest) at which a lone neuron fires with the latch's 47-step period (212.8 Hz).
BIAS_213HZ_MV = 58.0
#: Spacing of the pulses of a switching train, steps (the reset controller's relay hop, ~5.3 ms).
TRAIN_SPACING_STEPS = 53
#: v -> p inhibition per spike as a multiple of drive.loop: the pair's own loop strength. p is
#: silent while CLEAR from 0.8x up (0.75x leaks); 1.0x parks it 15 mV down like u, restarts
#: 10.5-14 ms after v stops; heavier inhibition only slows the restart (1.5x: 20-23 ms).
PROXY_QUANTA_X = 1.0


@dataclass(frozen=True)
class FlipFlop:
    u: int  # fires while SET (inhibitory: readable in a simulation, never on the connectome)
    v: int  # fires while CLEAR
    bias_mv: float
    inh_quanta: int
    p: int | None = None  # excitatory SET proxy: fires while SET (silenced by v); what a reader taps on the fly
    q: int | None = None  # excitatory CLEAR proxy: fires while CLEAR (silenced by u)
    proxy_quanta: int = 0  # v -> p (and u -> q) inhibition per spike

    @property
    def members(self) -> tuple[int, int]:
        return (self.u, self.v)

    @property
    def rail(self) -> int:
        """The SET rail a reader connects to: the excitatory proxy p when built, else u. On a
        placed circuit only p is sign-correct (u is inhibitory, so `u -> reader` would be an
        inhibitory synapse); in a bare simulation either carries the same 213 Hz train."""
        return self.u if self.p is None else self.p

    @property
    def proxies(self) -> tuple[int, ...]:
        return tuple(x for x in (self.p, self.q) if x is not None)


def add_flipflop(net: Netlist, drive: Drive, name: str, bias_mv: float = BIAS_213HZ_MV,
                 inh_quanta: int | None = None, proxy: bool = False, clear_proxy: bool = False,
                 proxy_quanta: int | None = None) -> FlipFlop:
    """Two biased inhibitory neurons in mutual inhibition. `inh_quanta` (positive; default
    drive.loop) is the strength of each member's inhibition of the other per spike.

    `proxy`: add p, an EXCITATORY neuron on the same bias, inhibited by v (`proxy_quanta` per
    spike, default drive.loop = PROXY_QUANTA_X x loop). p fires the same 212.8 Hz train as u
    while SET and is parked ~15 mV below threshold by v's train while CLEAR: it is u's
    sign-correct readout. Readers connect to `ff.p` (`ff.rail`) exactly as they would to a
    latch's u: add_edge_relay(source=ff.p), add_veto_relay(vetoes=[ff.p]). u's own outputs are
    inhibitory and cannot be carried by any host (docs/h1_placement.md, Placing flip-flops).
    What it costs (docs/contracts/flipflop.yaml): p rises 10.5-14 ms after v's last spike
    (18.7-23.1 ms after the first SET pulse, against u's 6.0-6.5), because a parked neuron
    climbs 15 mV with tau_m as v does on CLEAR; it stops within 17.8 ms of the first CLEAR
    pulse. `clear_proxy`: add q, the same neuron inhibited by u (fires while CLEAR).
    Neither proxy feeds back: a stray spike of p never touches the pair's state.
    power_on_pulse / power_on_events include p (it would otherwise fire at 2.5 ms, before v's
    first inhibition lands, and an edge relay on it would fire at power-on)."""
    q = drive.loop if inh_quanta is None else int(inh_quanta)
    u = net.neuron(f"{name}.u", bias=bias_mv)
    v = net.neuron(f"{name}.v", bias=bias_mv)
    net.synapse(u, v, -q)
    net.synapse(v, u, -q)
    p = qq = None
    pq = 0
    if proxy or clear_proxy:
        pq = int(round(PROXY_QUANTA_X * drive.loop)) if proxy_quanta is None else int(proxy_quanta)
    if proxy:
        p = net.neuron(f"{name}.p", bias=bias_mv)
        net.synapse(v, p, -pq)
    if clear_proxy:
        qq = net.neuron(f"{name}.q", bias=bias_mv)
        net.synapse(u, qq, -pq)
    return FlipFlop(u, v, float(bias_mv), q, p, qq, pq)


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
    """Inject at step 0 to start CLEAR: one loop-strength inhibitory pulse into u, and one
    into the SET proxy p when there is one (every biased neuron starts at rest and would fire
    at 2.5 ms, before any inhibition lands). Without it both members fire together from step
    25 on and stay in lockstep (measured, 300 ms). q, the CLEAR proxy, is left alone: it
    starts firing with v, which is the state it reports."""
    ev = [(0, ff.u, -drive.loop)]
    if ff.p is not None:
        ev.append((0, ff.p, -drive.loop))
    return ev


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


def flipflop_taps(net: Netlist, suffix: str = ".u") -> list[int]:
    """The u members of every flip-flop in the netlist: biased neurons whose role ends in
    `.u` (add_flipflop names them so; nothing else in the library carries a bias). With
    `suffix=".p"`, the SET proxies instead."""
    return [i for i, (role, b) in enumerate(zip(net.roles, net.bias)) if b != 0.0 and role.endswith(suffix)]


def power_on_events(net: Netlist, drive: Drive, at_step: int = 0) -> list[tuple[int, int, int]]:
    """What a harness injects when it loads an image: one loop-strength inhibitory pulse into
    every flip-flop's u, and into every SET proxy p, at `at_step`, so each starts CLEAR
    instead of in lockstep and no proxy fires before v parks it (see power_on_pulse).
    Returns (step, neuron, quanta) triples; `schedule`-compatible after subtracting
    `at_step`, or fed straight to `sim.add_events`."""
    return [(at_step, n, -drive.loop) for n in flipflop_taps(net, ".u") + flipflop_taps(net, ".p")]
