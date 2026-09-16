"""One four-phase dual-rail channel: producer register -> consumer register.

    DATA     producer rail latch trains drive consumer rail latches (pulse-mode ignition)
    VALIDATE consumer bit_valid latches -> completion tree -> root latch W (state-holding)
    ACCEPT   W drives the producer's reset trigger
    CLEARED  producer's READY generator (vetoed by its rail latches) drives the consumer's reset
    READY    consumer's READY generator (vetoed by its rail latches and W) -> output port

The harness loads the producer by injecting one pulse into each active rail's `u`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..lib.netlist import Drive, Netlist
from ..sim.model import Params
from .celement import add_and_gate, add_completion_tree, add_or_latched, add_state, rail_of, set_input
from .flipflop import FlipFlop, connect_clear, power_on_events
from .latch import Latch, add_edge_relay, add_latch, add_ready, add_reset, connect_trigger
from .watchdog import StaleMonitor, Watchdog, add_stale_monitor, add_watchdog


@dataclass
class Register:
    rails: list  # rails[i][r] -> Latch (storage "latch", read at `.u`) or FlipFlop (storage "flipflop", read at `.p`)
    reset_trigger: int
    reset_inh: int
    ready: int
    valid: list = field(default_factory=list)  # bit_valid latches (consumer only)
    fault: list = field(default_factory=list)  # fault gate neurons (consumer only)
    completion: Latch | None = None
    internal: list = field(default_factory=list)
    reset_edge: int = -1
    fault_latch: Latch | None = None
    last_reset_relay: int = -1
    ready_chain: list = field(default_factory=list)
    monitor: StaleMonitor | None = None
    watchdog: Watchdog | None = None
    storage: str = "latch"
    flag_storage: str = "latch"
    rail_inputs: list = field(default_factory=list)  # rail_inputs[i][r]: where one ignition pulse sets rail (i, r)

    @property
    def rail_taps(self) -> list[list[int]]:
        """What a reader (the decoder, a downstream gate) taps per rail: a latch's u, a
        flip-flop's excitatory proxy p (celement.rail_of)."""
        return [[rail_of(self.rails[i][0]), rail_of(self.rails[i][1])] for i in range(len(self.rails))]

    def all_states(self) -> list:
        """Every state element (Latch or FlipFlop): rails, valids, tree, completion."""
        out = [l for pair in self.rails for l in pair] + list(self.valid) + list(self.internal)
        if self.completion is not None:
            out.append(self.completion)
        return out

    def all_latches(self) -> list[Latch]:
        """The excitatory latches only: what the reset train hits on both members."""
        return [l for l in self.all_states() if not isinstance(l, FlipFlop)]

    def all_flipflops(self) -> list[FlipFlop]:
        """The flip-flops: cleared by the reset train through their u member only."""
        return [l for l in self.all_states() if isinstance(l, FlipFlop)]


def add_register(net: Netlist, drive: Drive, name: str, width: int, with_completion: bool,
                 storage: str = "latch", flag_storage: str = "latch", rail_proxy: bool = True) -> Register:
    """`storage` is what holds the rails: "latch" (two-neuron excitatory loop, set by one
    ignition pulse into `.u`, read at `.u`) or "flipflop" (two biased inhibitory neurons in
    mutual inhibition plus the excitatory proxy p; set by a train of three pulses through a
    per-rail set chain, whose trigger is `rail_inputs[i][r]`; cleared by the reset train
    aimed at `.u` only; READ at `.p` by every consumer of the rail - the valid ORs, the fault
    ANDs, the decode taps `rail_taps` - because u is inhibitory and no host can carry an
    excitatory read of it; needs one power-on pulse, `power_on_events`). `flag_storage` is
    the same choice for the valid, tree and completion elements (unproxied: a flag
    flip-flop is read at its u, an option kept for measurement only). `rail_proxy=False`
    builds the flip-flop rails without p and reads them at u (the 2026-09-16 first build;
    kept for the before/after in docs/contracts/ffregister.yaml, not for placement). The
    default build is unchanged neuron for neuron."""
    proxy = storage == "flipflop" and rail_proxy
    rails = [[add_state(net, drive, f"{name}.b{i}r{r}", storage, proxy=proxy) for r in (0, 1)] for i in range(width)]
    rail_inputs = [[set_input(net, drive, f"{name}.b{i}r{r}", rails[i][r]) for r in (0, 1)] for i in range(width)]
    valid, fault, internal, completion, gates = [], [], [], None, []
    if with_completion:
        for i in range(width):
            taps = [rail_of(rails[i][0]), rail_of(rails[i][1])]  # a latch's u; a flip-flop's proxy p
            g, lv = add_or_latched(net, drive, f"{name}.valid{i}", taps, flag_storage)
            valid.append(lv)
            gates.append(g)
            fault.append(add_and_gate(net, drive, f"{name}.fault{i}", taps, fraction=0.55))
        completion, internal = add_completion_tree(net, drive, f"{name}.comp", valid, flag_storage)
        gates += [x for x, role in enumerate(net.roles) if role.startswith(f"{name}.comp.") and role.endswith(".and")]
        if completion in valid:  # width 1: the valid latch is the completion latch
            internal = []
    reg = Register(rails, -1, -1, -1, valid, fault, completion, internal, storage=storage, flag_storage=flag_storage,
                   rail_inputs=rail_inputs)
    latches = reg.all_latches()
    trig, inh, edge = add_reset(net, drive, name, latches, gates)
    connect_clear(net, drive, inh, reg.all_flipflops())  # CLEAR: the same 4-pulse train, into u only, 1.5x loop
    ready = add_ready(net, drive, name, trig, hops=15)  # ~80 ms: leaves the stale monitor ~19 ms to land its block
    reg.reset_trigger, reg.reset_inh, reg.ready, reg.reset_edge = trig, inh, ready, edge
    reg.last_reset_relay = max(x for x, r in enumerate(net.roles) if r.startswith(f"{name}.reset_relay"))
    reg.ready_chain = [x for x, r in enumerate(net.roles) if r.startswith(f"{name}.ready_delay")]
    net.group(f"{name}.rail_taps", [t for pair in reg.rail_taps for t in pair])
    return reg


@dataclass
class Channel:
    net: Netlist
    drive: Drive
    width: int
    producer: Register
    consumer: Register

    @property
    def accept(self) -> int:
        return self.consumer.completion.u

    @property
    def cleared(self) -> int:
        return self.producer.ready

    @property
    def ready(self) -> int:
        return self.consumer.ready

    def power_on_events(self, at_step: int = 0) -> list[tuple[int, int, int]]:
        """(step, neuron, quanta) the harness injects with the image: one inhibitory pulse
        into every flip-flop's u so it starts CLEAR (none for an all-latch channel)."""
        return power_on_events(self.net, self.drive, at_step)


def wire_fault_path(net: Netlist, drive: Drive, P: Register, Q: Register, storage: str = "latch") -> Latch:
    """FAULT as a state (spec.md §3): any fault gate ignites one shared fault latch F, which
    (a) holds the completion latch, its AND gate and its ignition relay down so the word is
    never consumed, and (b) raises FAULT-ACCEPT once through the producer's edge-detected
    reset trigger so the four phases still complete. F is reset with the consumer.
    Unlatched fault-gate spikes (~every 36 ms while both rails hold) re-armed the edge
    detector and fired a second reset that wiped the next word (observed)."""
    F = add_state(net, drive, "Q.faultL", storage)
    f_in = set_input(net, drive, "Q.faultL", F)
    for f in Q.fault:
        net.synapse(f, f_in, drive.ignite)
    root = net.roles[Q.completion.u][: -len(".L.u")]
    held = list(Q.completion.members) + [x for x, role in enumerate(net.roles) if role in (f"{root}.and", f"{root}.ign.edge")]
    if isinstance(Q.completion, FlipFlop):  # a flip-flop root is set through its chain: hold that too, not its v
        held = [Q.completion.u] + [x for x, role in enumerate(net.roles)
                                   if role in (f"{root}.and", f"{root}.ign.edge") or role.startswith(f"{root}.set")]
    for x in held:
        net.synapse(F.u, x, drive.reset)
    connect_trigger(net, drive, F.u, P.reset_trigger, P.reset_edge)
    if isinstance(F, FlipFlop):
        connect_clear(net, drive, Q.reset_inh, [F])
    else:
        q = -int(round(0.75 * drive.loop))
        for x in F.members:
            net.synapse(Q.reset_inh, x, q)
    Q.fault_latch = F
    return F


def add_liveness(net: Netlist, drive: Drive, P: Register, Q: Register, watchdog_hops: int, monitor: bool = False) -> None:
    """A2: watchdog on the producer (started by its rails, cancelled by ACCEPT / FAULT-ACCEPT).
    The stale-state monitor is available but off by default: rejected on measurement. Its
    per-latch detector (a two-input rate-mode AND, tap + ENABLE) sits at 75 % of threshold on
    the tap alone for ~100 ms per transaction across a dozen latches, and one spurious spike
    ignites STALE; at mix B that produced false retries in ~2 % of transactions, READY hangs,
    and, through mid-transaction resets, wrong values. A safe fraction (0.55 + 0.55) detects
    too slowly to beat READY. Stale state stays harness-observed (and rare: 8 per 10^6)."""
    P.watchdog = add_watchdog(net, drive, "P.wd", [rail_of(l) for pair in P.rails for l in pair],  # a flip-flop producer's rails read through their proxies
                              [Q.completion.u, Q.fault_latch.u], watchdog_hops, P.reset_trigger, P.reset_edge, P.reset_inh)
    if monitor:
        for name, reg in (("P", P), ("Q", Q)):
            taps = [rail_of(l) for l in reg.all_latches()] + ([reg.fault_latch.u] if reg.fault_latch is not None else [])
            reg.monitor = add_stale_monitor(net, drive, f"{name}.mon", taps, reg.last_reset_relay, reg.ready,
                                            reg.ready_chain[-3:], reg.reset_trigger, reg.reset_edge, reg.reset_inh)


def build_channel(params: Params, width: int, drive: Drive | None = None, liveness: bool = True,
                  watchdog_hops: int = 40, monitor: bool = False, storage: str = "latch",
                  flag_storage: str | None = None, producer_storage: str = "latch", rail_proxy: bool = True) -> Channel:
    """`storage` is the consumer register's rail storage (see add_register); `flag_storage`
    that of its valid / tree / completion / fault elements (default: "latch" whatever the
    rails are, chosen by measurement, docs/a1_flipflop.md). Flip-flop rails carry the
    excitatory proxy p and are read there (`rail_proxy`, default True: the sign-correct
    build; False is the u-readout build kept for comparison). The producer stays a latch
    register by default because every harness (run_transactions, the campaigns) loads it
    with one ignition pulse per rail into `P.rails[i][r].u`, which cannot set a flip-flop;
    a flip-flop producer (`producer_storage`) must be loaded through `P.rail_inputs`."""
    drive = drive or Drive.from_params(params)
    flag_storage = flag_storage or "latch"
    net = Netlist(params)
    P = add_register(net, drive, "P", width, with_completion=False, storage=producer_storage, rail_proxy=rail_proxy)
    Q = add_register(net, drive, "Q", width, with_completion=True, storage=storage, flag_storage=flag_storage,
                     rail_proxy=rail_proxy)
    for i in range(width):
        for r in (0, 1):
            relay = add_edge_relay(net, drive, f"data.b{i}r{r}", P.rails[i][r].u)
            net.synapse(relay, Q.rail_inputs[i][r], drive.ignite)  # DATA: one ignition pulse per rail (a flip-flop's set chain makes it three)
    connect_trigger(net, drive, Q.completion.u, P.reset_trigger, P.reset_edge)  # ACCEPT (edge)
    connect_trigger(net, drive, P.ready, Q.reset_trigger, Q.reset_edge)  # CLEARED (edge)
    # FAULT (both rails of a bit active): block the completion latch so the word is never
    # consumed, and raise FAULT-ACCEPT so the four phases still complete and the channel
    # does not deadlock (spec.md §3). A fault after completion is a flag only.
    wire_fault_path(net, drive, P, Q, flag_storage)
    if liveness:
        add_liveness(net, drive, P, Q, watchdog_hops, monitor)
    net.group("accept", [Q.completion.u])
    net.group("cleared", [P.ready])
    net.group("ready", [Q.ready])
    return Channel(net, drive, width, P, Q)
