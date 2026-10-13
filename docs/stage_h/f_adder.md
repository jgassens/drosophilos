# Configuration F on the H1 flip-flop adder (2026-10-13)

`build_adder_channel` now accepts the experimental Configuration F flags:
`rate_robust=True, experimental_autapse=True, ordered_xor_oneshot=True`, with
`ordered=True`. The sum XORs consume their late rail once, and latch feedback is
installed after every reader, controller and reset. The kernel's
`kill_pulses=4` override is its general kill-train policy, not an F-specific
transform; the adder keeps its existing drive and register reset timing.
The one-shot option retains the kernel's default-physics/drive and width-1..8
restriction. Channel has no build-metadata field, so no new metadata structure
was introduced; `channel.drive.rate_robust` records the reader choice.

The four-bit ordered flip-flop build has **740 neurons, 1,518 synapses and
1,516 distinct edge pairs**, compared with 664 neurons and 1,218 synapses
before F. The finalizer matches **70 excitatory latch pairs**, adding **140
autapses** at -724 quanta and zero delay. The sum XORs add **8 one-shot
self-edges** at -15,932 quanta and zero delay. None of the output register's
10 inhibitory flip-flop pairs receives an experimental latch autapse.
Frozen SHA-256 tests of src, dst, quanta, delays and roles, captured before
editing, preserve both default storage builds in ordered and rate mode.

The isolated float64 RefSim/H1 harness computed **8/8 fresh and 8/8 chained**
random additions (operand seed 110), with no wrong sums, faults or timeouts.
The opt-in mix-B probe also passed **8/8 chained** additions on seed 110:
4% lognormal weight noise, threshold/bias noise, 5 Hz background at 150 quanta,
and up to 10 ms of independent operand-rail jitter. This small probe is not
a broad noise qualification. A separate **50 fresh + 50 chained** test is
available with `DROSO_F_ADDER=1`; it was not run here. Set
`DROSO_F_ADDER_NOISE=1` for the noisy probe. Both use the existing `slow` marker.

On four fresh seed-0 additions, reported ACCEPT latencies were
398, 342, 344, 398 ms without F and 412, 354, 358, 412 ms with F:
means **370.5 versus 384 ms**, an increase of **13.5 ms**.

Placement treats self-edges as scoring opportunities, never hard constraints
or inputs to transmitter voting or motif degrees. Only the neuron's own host
self-edge, with matching transmitter sign and enough synapses under
`Policy.req_count`, can carry one. Audit and image counts distinguish absent,
weak, wrong-sign and unplaced-host autapses. Condition A restores missing
self-edges at their designed weights and zero delays; condition B omits them.
An existing incompatible host self-edge is zeroed as parasitic under A/B.
Synthetic tests check carried and missing cases, simulator delays, and both
carried and restored self-edges in `full_graph_topology`; its existing delay
and edge-copy paths already support them. Existing embed tests pass.

## Real MCNS smoke run

Loaded `/Users/jeremiahgassensmith/programming/drosophilos/data/mcns`:
166,700 neurons, 25,582,938 anatomical edges, including 101 self-edges.
Ran placement once with `Policy(k_max=4)`, four restarts, seed 0,
600-second time limit and the H1 `watchdog_hops=100`. Search took 44.6 seconds;
the winning restart was `hub_last seed 3`. It placed **739/740 neurons** and
carried **809/1,516 edge pairs (53.36%)**. **707** missing pairs became Profile 3
edges, and 16,206 parasitic anatomical edges were zeroed. No designed self-edge
was carried: **147 lacked a host self-edge and one had the wrong sign**;
none was classed as too weak or endpoint unplaced.

Condition A computed **10/10 fresh and 10/10 chained** seed-0 additions,
with zero wrong sums, faults, timeouts or missing completions. Its fresh ACCEPT
latencies were 412, 354, 358, 412, 358, 358, 403, 412, 379, 371 ms.
This is an isolated condition-A image run; whole-brain simulation and condition B
were not run on real MCNS.

Profile 3 counts below are per distinct edge pair. When several classes share
a row, the count applies to **each** listed class.

| Class | Missing pairs per class |
| --- | ---: |
| autapse (no host self-edge) | 147 |
| cancel_inh -> hop | 96 |
| reset_inh -> u | 75 |
| reset_inh -> v | 64 |
| edge -> u | 36 |
| edge_inh -> edge | 30 |
| reset_inh -> rate | 26 |
| veto -> edge | 22 |
| u -> rate | 20 |
| u -> veto | 14 |
| d -> edge | 12 |
| u -> edge | 11 |
| p -> rate; v -> rate | 10 |
| set -> set_relay; edge -> edge | 9 |
| rate -> and; d -> d; d -> edge_inh | 8 |
| set_relay -> set_relay; u -> v; edge -> set | 7 |
| v -> u; reset_relay -> reset_inh; set_relay -> u; u -> hold_inh; reset_inh -> or | 6 |
| u -> edge_inh | 5 |
| reset_inh -> and; and -> u | 4 |
| set -> u; u -> d | 3 |
| reset -> reset_inh; d -> veto; edge -> edge_inh; u -> reset | 2 |
| reset -> reset_relay; reset_edge -> reset; v -> p; and -> edge_inh; or -> edge; hold_inh -> edge; u -> u; u -> and; start -> edge_inh; u -> cancel_inh; cancel_inh -> u; cancel_inh -> v; u -> reset_edge; autapse (wrong sign) | 1 |

Validation used the existing repository virtual environment with `PYTHONPATH=.`
and uv dependency syncing disabled: the sandbox's restricted network prevented
installing dependencies into the new worktree environment. The default F suite
passed 12 tests (three opt-in cases skipped); the noisy probe passed separately.
The embed suites passed, with MCNS tests requiring worktree-local data skipped.
