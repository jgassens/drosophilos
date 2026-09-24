"""Latched completion elements.

Every gate here reads only latch trains (a standard 213 Hz rate), and every gate output
that feeds another gate is latched, so rate-mode weights never depend on an upstream
gate's firing rate. A completion element therefore is: gate neuron -> latch.
  bit_valid(i)  : OR of the bit's two rails, latched      (rate-mode OR, or_in per rail)
  fault(i)      : AND of the bit's two rails              (rate-mode AND, and_in per rail)
  c2(a, b)      : AND of two latched valids, latched      (rate-mode AND, and_in each)
A word's completion is a binary tree of c2 over its bit_valid latches; its root latch is
the state-holding element: set when all inputs agree (all valid), cleared only by RESET.
"""

from __future__ import annotations

from ..lib.netlist import Drive, Netlist
from .flipflop import FlipFlop, add_flipflop, add_set_chain
from .latch import Latch, add_edge_relay, add_latch


def add_state(net: Netlist, drive: Drive, name: str, storage: str = "latch", proxy: bool = False):
    """A one-bit state element by `storage`: "latch" (the excitatory two-neuron loop, ignited
    by one pulse) or "flipflop" (two biased inhibitory neurons, set by a train through
    add_set_chain). Both expose `.u`, a 213 Hz train while set. `proxy` (flip-flop only) adds
    the excitatory readout p (add_flipflop(proxy=True)); readers then take `rail_of(state)`,
    which is p for a proxied flip-flop and u otherwise, so every reader is the same."""
    if storage == "latch":
        return add_latch(net, drive, name)
    if storage == "flipflop":
        return add_flipflop(net, drive, name, proxy=proxy)
    raise ValueError(f"unknown storage {storage!r}: 'latch' or 'flipflop'")


def rail_of(state) -> int:
    """The neuron a reader of `state` connects to: a latch's u; a flip-flop's excitatory proxy
    p when it has one (its u is inhibitory, so `u -> reader` is the wrong sign on any host),
    else its u. Writers never use this: they go through set_input and the clear train."""
    return state.rail if isinstance(state, FlipFlop) else state.u


def set_input(net: Netlist, drive: Drive, name: str, state) -> int:
    """The neuron one ignition pulse goes into to set `state`: a latch's own `u`; for a
    flip-flop the trigger of a fresh set chain (a train of SET_TRAIN_PULSES pulses into u, four since the lockstep fix), since one pulse into
    its u never sets it (docs/a1_flipflop.md)."""
    if isinstance(state, FlipFlop):
        return add_set_chain(net, drive, name, state)
    return state.u


def _ignite_from(net: Netlist, drive: Drive, name: str, gate: int, latch) -> None:
    """Gate -> latch through an edge relay: one ignition pulse per gate activation. A gate
    that kept pulsing the latch would add a second drive on top of the loop and let the
    latch ride through the reset train (observed on valid and tree latches). The ignited
    latch's train holds the relay down (hold_from): a gate source is too slow to do that
    itself, and a re-firing relay re-ignites the latch every ~43 ms (see add_edge_relay).
    A flip-flop target takes the pulse through its set chain (set_input)."""
    relay = add_edge_relay(net, drive, f"{name}.ign", gate, hold_from=[rail_of(latch)])
    net.synapse(relay, set_input(net, drive, name, latch), drive.ignite)


def add_or_latched(net: Netlist, drive: Drive, name: str, inputs: list[int], storage: str = "latch") -> tuple[int, Latch]:
    g = net.neuron(f"{name}.or")
    for x in inputs:
        net.synapse(x, g, drive.or_in)
    l = add_state(net, drive, f"{name}.L", storage)
    _ignite_from(net, drive, name, g, l)
    return g, l


def add_and_latched(net: Netlist, drive: Drive, name: str, inputs: list[int], storage: str = "latch") -> tuple[int, Latch]:
    assert len(inputs) == 2, "rate-mode AND with margin is a 2-input gate; build a tree"
    g = net.neuron(f"{name}.and")
    for x in inputs:
        net.synapse(x, g, drive.and_in)
    l = add_state(net, drive, f"{name}.L", storage)
    _ignite_from(net, drive, name, g, l)
    return g, l


def add_and_gate(net: Netlist, drive: Drive, name: str, inputs: list[int], fraction: float | None = None) -> int:
    """Unlatched rate-mode AND. `fraction` overrides drive.and_in (as a fraction of the
    sustained-train need); the fault gates use 0.6 so that one rail plus noise never trips
    them, while completion ANDs use 0.75 so that two inputs fire under -10 % weights."""
    assert len(inputs) == 2
    g = net.neuron(f"{name}.and")
    q = drive.and_in if fraction is None else int(round(fraction * drive.rate_need))
    for x in inputs:
        net.synapse(x, g, q)
    return g


def add_completion_tree(net: Netlist, drive: Drive, name: str, valid_latches: list[Latch],
                        storage: str = "latch") -> tuple[Latch, list[Latch]]:
    """Binary tree of latched 2-input ANDs. Returns (root latch, all internal latches)."""
    level = list(valid_latches)
    internal: list[Latch] = []
    depth = 0
    while len(level) > 1:
        nxt = []
        for k in range(0, len(level) - 1, 2):
            _, l = add_and_latched(net, drive, f"{name}.c{depth}_{k // 2}", [rail_of(level[k]), rail_of(level[k + 1])], storage)
            internal.append(l)
            nxt.append(l)
        if len(level) % 2:
            nxt.append(level[-1])
        level = nxt
        depth += 1
    return level[0], internal


def add_veto_neuron(net: Netlist, drive: Drive, name: str, taps: list[int]) -> int:
    """An inhibitory interneuron that fires with every spike of any of `taps`; shared by
    several veto relays through `add_veto_relay(..., veto_neurons=[...])` (e.g. one
    "address is not w" neuron per memory word for all of that word's relays)."""
    v = net.neuron(f"{name}.veto")
    for t in taps:
        net.synapse(t, v, drive.pulse)
    return v


def add_veto_relay(net: Netlist, drive: Drive, name: str, driver: int, vetoes: list[int], target: Latch,
                   veto_strength: float | None = None, veto_neurons: list[int] = (),
                   require: list[int] = ()) -> int:
    """driver AND NOT(any veto), evaluated once at the driver's rise, as a one-shot ignition of
    `target`. The driver is a latch train (so the relay's own feed-forward inhibition and the
    target's train keep it to one pulse); each veto is a latch train that, while live, holds
    the relay ~33 mV below rest through one inhibitory interneuron, against which the 12.6 mV
    driver pulse cannot fire it. There is no exposure window: unlike a rate-mode AND, the
    relay never integrates one input towards threshold.

    Ordering assumption (bounded delay, stated per user): every veto rail must be established
    >= ~15 ms before the driver rises (0.5x loop per veto spike: ~33 mV sustained, ~3 spikes
    to block; it recovers in ~55 ms once the veto rail dies, so a vetoed relay may be driven
    again 55 ms later. 2.2x would block on one spike but paralyse the relay for ~90 ms).
    Dual-rail supplies NOT for free: to compute a AND b with b the earlier operand, veto with
    b's other rail. No hold from the target is needed: the driver is a latch train, which the
    relay's own source inhibition already turns into one shot.

    With `require`, first turn the driver train into one full-strength, fast-inhibited edge.
    Each required tap then qualifies that single pulse in a separate coincidence neuron;
    the checks are serial ANDs, not an additive vote in which another rail can stand in
    for a missing input. The coincidence neurons have -2 mV bias and take 0.92 single_need
    from the pulse and 0.76 rate_need from their one required train. The bias increases
    their distance from stray input without changing any other neuron or the kill train.

    At the default Params, V_th - E_L is 7 mV, and the biased coincidence gap is 9 mV.
    Thus the nominal pulse/rail/sum fractions are 0.92*7/9 = 0.716, 0.76*7/9 = 0.591,
    and 1.68*7/9 = 1.307. At +12% weights/-1 mV threshold the isolated pulse is
    0.92*1.12*7/8 = 0.902; at -12%/+1 mV the coincidence is
    1.68*0.88*7/10 = 1.035. A 47/39-times faster permanent rail at +12%/-1 mV is
    0.76*(47/39)*1.12*7/8 = 0.898. These are mean/peak estimates, not a timing proof;
    tests measure the full trains, veto recovery, inhibitor race, and stray input.
    Mix B uses Gaussian/log-normal sigmas (4%, 0.2 mV threshold and bias), not bounds.

    The required form defaults to a lighter 0.15-loop veto (~10 mV sustained), since its
    ~2.8 mV coincidence surplus cannot recover from the old 0.5-loop veto in the available
    delay. The non-required form retains the original 0.5-loop veto and edge circuit.
    An explicit `veto_strength` overrides either default."""
    if veto_strength is None:
        veto_strength = 0.15 if require else 0.5
    if require:
        pulse = add_edge_relay(net, drive, f"{name}.driver", driver, fast_inhibitor=True)
        for k, tap in enumerate(require):
            relay = net.neuron(f"{name}.edge" if k == len(require) - 1 else f"{name}.require{k}", bias=-2.0)
            net.synapse(pulse, relay, int(round(0.92 * drive.single_need)))
            net.synapse(tap, relay, int(round(0.76 * drive.rate_need)))
            pulse = relay
    else:
        relay = add_edge_relay(net, drive, name, driver)
    if vetoes:
        v = net.neuron(f"{name}.veto")
        for t in vetoes:
            net.synapse(t, v, drive.pulse)
        net.synapse(v, relay, -int(round(veto_strength * drive.loop)))
    for v in veto_neurons:
        net.synapse(v, relay, -int(round(veto_strength * drive.loop)))
    net.synapse(relay, set_input(net, drive, name, target), drive.ignite)  # a flip-flop target gets a set chain
    return relay


def add_delay_chain(net: Netlist, drive: Drive, name: str, source: int, hops: int) -> int:
    """Delays the rise of a train by `hops` relays (~5.3 ms per hop at pulse drive). Used to
    fix the arrival order that veto relays depend on. Only the rise is faithful: fed a 213 Hz
    latch train, a hop is over-driven and fires near its refractory limit, which is harmless
    for a veto relay's driver (its own source inhibition holds it after the first spike). Not
    in any reset domain: it drains within its delay once the source stops, and a train with
    no gap cannot re-fire a relay (its inhibition needs ~86 ms of silence to decay)."""
    prev = source
    for k in range(hops):
        h = net.neuron(f"{name}.d{k}")
        net.synapse(prev, h, drive.pulse)
        prev = h
    return prev
