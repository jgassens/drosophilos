"""Bit latch: a two-neuron excitatory loop holding a circulating spike.

Ignite with one pulse of >= drive.loop quanta into `u`; the value is held as a 213 Hz
train on both members until reset inhibition (drive.reset on both members) stops it.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..lib.netlist import Drive, Netlist
from ..sim.model import D_MAX


# Version 1 is the circuit built by f71538c..63c41f8. Bump on retuning.
EXPERIMENTAL_AUTAPSE_VERSION = 1
EXPERIMENTAL_AUTAPSE_GAIN = 0.2
EXPERIMENTAL_AUTAPSE_DELAY_STEPS = 0


# Physical timing choices, independent of the simulator's delay-ring capacity.
COMPACT_READY_DELAY_STEPS = 100
COMPACT_READY_TAIL_LINKS = 8

# Qualification prototype, NOT a released register policy. See the measured
# frontier in docs/stage_d_completion_stall.md before enabling it in a builder.
VERIFY_WINDOW_LINKS = 4
VERIFY_DELAY_STEPS = 100
VERIFY_MAX_RETRIES = 3


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


def add_experimental_autapses(net: Netlist, drive: Drive) -> None:
    """Finalize kernel storage with the stable-latch probe's 0.2-loop feedback.

    Call after all readers/qualifiers and controllers are wired. Local feedback
    is not a reset: it must never be mirrored onto consumers. The mirror registry
    remains available for subsequent genuine clears, but no new reader may be
    installed after this transform. This is unqualified on the 44-step rate
    floor and reload margin; only the experimental kernel builder calls it.
    """
    edges = {(s, d): q for s, d, q in zip(net.src, net.dst, net.quanta)}
    latches = [Latch(u, v) for (u, v), q in edges.items()
               if q == drive.loop and edges.get((v, u)) == drive.loop
               and net.roles[u].endswith(".u")
               and net.roles[v] == net.roles[u].removesuffix(".u") + ".v"]
    mirrors, net.inhibition_mirrors = net.inhibition_mirrors, {}
    try:
        for latch in latches:
            for member in latch.members:
                net.synapse(member, member, -round(drive.loop * EXPERIMENTAL_AUTAPSE_GAIN),
                            EXPERIMENTAL_AUTAPSE_DELAY_STEPS)
    finally:
        net.inhibition_mirrors = mirrors


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
    Configured tap counts are not emitted spike counts (four ordinary taps emit
    five nominal spikes).
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


def compact_reset_domains(net: Netlist, drive: Drive, controllers: list[tuple[int, int]],
                          *, strength: float = 1.75,
                          ready_tail_links: int = COMPACT_READY_TAIL_LINKS) -> None:
    """Compile experimental register domains to four compact 1.75-loop taps.

    Call after all reset extensions and rate-reader mirrors are installed.
    Working on the completed fan-out includes ALU state, FAULT, ACT, COMMIT,
    grant/COPY, power-up vetoes, zero_once and mirrored readers. Scale from the
    original integer weights, rounding once, including compiled reader gains.
    Retaining those originals makes repeated/overlapping calls idempotent.
    Producer, request, RAM and machine controllers are not implicitly selected.
    READY retains its fifteen-hop chain; its final eight synaptic delays become
    100 steps each (+656 steps at the default physics). The optional tail length
    is for qualification of the recovery frontier, including the rejected policy.
    """
    inhibitors = {inh for _, inh in controllers}
    links = set()
    ready_links = set()
    delayed_ready_links = set()
    roles = {role: i for i, role in enumerate(net.roles)}
    if not 0 <= ready_tail_links <= 16 or not 0 <= COMPACT_READY_DELAY_STEPS <= D_MAX:
        raise ValueError("invalid compact register READY delay")
    for trigger, _ in controllers:
        name = net.roles[trigger].removesuffix(".reset")
        taps = [f"{name}.reset_relay{k}" for k in (1, 2, 3)]
        if any(t not in roles for t in taps) or f"{name}.reset_relay4" in roles:
            raise ValueError("compact register reset requires exactly four taps")
        chain = [trigger] + [roles[t] for t in taps]
        links.update(zip(chain, chain[1:]))
        if f"{name}.ready" in roles:
            names = [f"{name}.ready_delay{k}" for k in range(15)]
            if any(n not in roles for n in names) or f"{name}.ready_delay15" in roles:
                raise ValueError("compact register reset requires a fifteen-hop READY chain")
            chain = [trigger] + [roles[n] for n in names] + [roles[f"{name}.ready"]]
            pairs = list(zip(chain, chain[1:]))
            ready_links.update(pairs)
            if ready_tail_links:
                delayed_ready_links.update(pairs[-ready_tail_links:])
    # Private compilation metadata only: it is never serialized in the topology.
    baseline = getattr(net, "_compact_reset_baseline", {})
    net._compact_reset_baseline = baseline
    scale = round(strength * drive.loop) / round(0.75 * drive.loop)
    for e, (src, dst) in enumerate(zip(net.src, net.dst)):
        if (src, dst) in links:
            net.delay[e] = 0
        if (src, dst) in ready_links:
            net.delay[e] = (COMPACT_READY_DELAY_STEPS if (src, dst) in delayed_ready_links
                            else net.params.default_delay_steps)
        if src in inhibitors and net.quanta[e] < 0:
            net.quanta[e] = round(baseline.setdefault(e, net.quanta[e]) * scale)


@dataclass(frozen=True)
class ResetVerification:
    domain: tuple[int, ...]
    busy: int
    attempts: tuple[int, ...]
    checks: tuple[int, ...]
    exhausted: Latch
    added_neurons: int
    added_synapses: int


def add_reset_verification(net: Netlist, drive: Drive, name: str, trigger: int,
                           inh: int, ready: int, ready_chain: list[int],
                           fault_target: int | None = None) -> ResetVerification:
    """Experimental silence certificate with three statically unrolled retries.

    Finalize AFTER every reset extension and mirrored reader. The domain is the
    complete negative fan-out of ``inh``; even a gate or a mirrored reader can
    veto the certificate. A refractory-limited busy neuron bounds the aggregate
    veto charge independently of register width. Four 100-step links following
    the ordinary recovery chain must propagate without that veto to emit READY.

    A parallel seven-link deadline is cancelled by a successful certificate.
    Otherwise it starts another ordinary four-tap train through the SAME inh,
    preserving every compiled reset weight (including rate-reader gains). No
    feedback edge restarts an attempt. After three retries a sticky exhaustion
    latch inhibits all attempt entries and READY and raises the existing FAULT
    when supplied. Only a new simulator/image clears exhaustion.

    This is a spiking circuit, not a host observer. Neither silence nor a bounded
    retry count proves analogue recovery. It is retained for falsifiable tests;
    ``verified_register_reset=True`` is rejected by the public kernel builder.
    """
    if len(ready_chain) != 15:
        raise ValueError("reset verification requires a fifteen-hop recovery chain")
    if not 0 <= VERIFY_DELAY_STEPS <= D_MAX:
        raise ValueError("invalid reset verification delay")
    old_edge = [e for e in net.incoming[ready] if net.src[e] == ready_chain[-1]]
    if len(old_edge) != 1 or len(net.incoming[ready]) != 1:
        raise ValueError("reset verification requires an unmodified READY output")
    if any(role.startswith(f"{name}.verify.") for role in net.roles):
        raise ValueError("reset verification already installed")
    n0, e0 = net.n, net.nnz
    domain = tuple(sorted({d for s, d, q in zip(net.src, net.dst, net.quanta)
                           if s == inh and q < 0}))
    busy = net.neuron(f"{name}.verify.busy")
    for tap in domain:
        net.synapse(tap, busy, drive.pulse)
    exhausted = add_latch(net, drive, f"{name}.verify.exhausted")
    attempts, checks = [], []
    start = trigger
    previous_deadline = None
    for attempt in range(VERIFY_MAX_RETRIES + 1):
        prefix = f"{name}.verify.a{attempt}"
        if attempt:
            start = net.neuron(f"{prefix}.reset")
            net.synapse(previous_deadline, start, drive.pulse)
            prev = start
            net.synapse(start, inh, drive.pulse)
            for k in range(1, 4):
                relay = net.neuron(f"{prefix}.reset_relay{k}")
                net.synapse(prev, relay, drive.pulse)
                net.synapse(relay, inh, drive.pulse)
                prev = relay
            prev = start
            for k in range(15):
                recovery = net.neuron(f"{prefix}.recovery{k}")
                net.synapse(prev, recovery, drive.pulse)
                prev = recovery
        else:
            prev = ready_chain[-1]
        attempts.append(start)
        recovery_end = prev
        for k in range(VERIFY_WINDOW_LINKS):
            timer = net.neuron(f"{prefix}.quiet{k}")
            if attempt == k == 0:
                # Preserve existing consumers of READY and the recovery prefix.
                edge = old_edge[0]
                net.incoming[ready].remove(edge)
                net.dst[edge] = timer
                net.delay[edge] = VERIFY_DELAY_STEPS
                net.incoming[timer].append(edge)
            else:
                net.synapse(prev, timer, drive.pulse, VERIFY_DELAY_STEPS)
            net.synapse(busy, timer, -int(round(0.75 * drive.loop)))
            prev = timer
        checks.append(prev)
        net.synapse(prev, ready, drive.pulse)
        certificate = prev
        prev = recovery_end
        for k in range(7):
            deadline = net.neuron(f"{prefix}.deadline{k}")
            net.synapse(prev, deadline, drive.pulse, VERIFY_DELAY_STEPS)
            if k >= 4:
                net.synapse(certificate, deadline, -int(round(2.2 * drive.loop)))
            prev = deadline
        previous_deadline = prev
    net.synapse(previous_deadline, exhausted.u, drive.ignite)
    for entry in [*attempts, ready]:
        net.synapse(exhausted.u, entry, -int(round(2.2 * drive.loop)))
    if fault_target is not None:
        net.synapse(exhausted.u, fault_target, drive.ignite)
    net.group(f"{name}.verify.exhausted", exhausted.members)
    return ResetVerification(domain, busy, tuple(attempts), tuple(checks), exhausted,
                             net.n - n0, net.nnz - e0)


def add_edge_relay(net: Netlist, drive: Drive, name: str, source: int, strength: float = 2.2, hold_from=(),
                   hold_strength: float = 0.25, fast_inhibitor: bool = False) -> int:
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
    # `fast_inhibitor`: the inhibitor gets the relay's own head-start drive, so its pulse lands
    # ~3.6 ms after the source's first spike instead of ~5.3 ms. With the plain pulse drive the
    # source train's second spike (4.7 ms, less at fast nodes) can cross before the inhibition
    # lands and the relay fires a doublet (measured on a RAM word-select: two pulses 3.6 ms
    # apart into an edge-detected reset trigger stacked three reset trains and the word could
    # not hold the copy). A doublet into a latch is harmless, so this is used where the target
    # is a trigger. The relay still wins the head-start race by ~1.8 ms nominal.
    net.synapse(source, inh, drive.relay_in if fast_inhibitor else drive.pulse)
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
