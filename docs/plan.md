# DrosophilOS — a connectome-constrained asynchronous neural computer that runs Doom

## Context

Target (your spec, unchanged): a distributed neural dataflow machine in which **both the
game update F and the renderer R execute in the neural substrate**. The host may only
integrate neuron equations, transport spike events, transduce input, load the initial
program image, and display already-computed pixels. Never collisions, textures, branches,
or rasterization. Steering an external Doom is *playing* it; this machine must *run* it.

Revisions in this version, from your review:
1. A minimal MCNS-constrained circuit (**Stage H0**) is built beside the first free
   primitives, so the connectome premise is tested before the compiler and renderer exist.
2. "Netlist mode" becomes **isolated execution** with a stated relationship to the full graph.
3. The handshake is a **four-phase, bounded-delay, self-timed spike protocol** with a
   state-holding completion element, built and verified before any arithmetic.
4. Simulator validation compares trajectories, spike events, and decoded transactions,
   against a float64 reference with a specified within-step order.
5. DrosoC gets an arithmetic semantics spec and an executable IR interpreter as the oracle.
6. Memory uses hierarchical decoding; contracts gain fan-in/out, load, envelope, reset
   latency, outstanding transactions; shared interfaces are fixed before parallel work.
7. Renderer is exact first (E1); population coding is an optimization (E2).
8. A whole-program capacity report gates any cluster scale-up; resource sharing is a
   first-class compiler decision.
9. **Stage F2** removes the Python control plane for the supported configuration; FlyLink
   gets a wire format that states where every field originates.
10. Fault model: what converts neural faults into detected rejection; protected voter and
    commit controller; idempotent commits; simulator snapshots kept, distinct from
    architectural checkpoints.
11. 1e-10 is an objective with a reporting discipline, not a claim.
12. Data access starts from Janelia's bulk files; connection counts follow the loader's
    filter rules; Juno inventory is read from the scheduler; topology is shared, not copied.

Fixed decisions: name DrosophilOS (sub-names FlyISA, FlyASM, FlyLink); hardware image
**MCNS v1.0** (male brain + nerve cord, 166,700 neurons); `minidoom` written from scratch
in DrosoC, Doom-structured; dev on the M1 Pro, scale on UTD Juno.

## Vocabulary (defined once)

- **Connectome**: every neuron and every synapse between them.
- **LIF neuron**: voltage rises with input, leaks, spikes at threshold, resets.
- **Dual-rail bit**: two channels; spike on rail 0 = valid zero, rail 1 = valid one,
  neither = not here yet, both = fault. Silence is never zero.
- **Token**: one valid value in flight. **Four-phase handshake**: data, accept, reset,
  ready-for-next; a new transfer starts only after all four.
- **C-element**: a gate whose output changes only when all its inputs agree and otherwise
  holds its previous state. The building block of completion detection.
- **Isolated execution**: simulating only the compiled circuit; its fidelity to the full
  brain is a separate, labeled claim.
- **Epoch**: a numbered round; events from an old epoch are rejected.
- **TMR**: three copies, vote on the decoded result.

## Machine definition

### Neuron model and within-step order
Current-based LIF, exponential synaptic kernels, per-synapse integer-step delay (max
10 ms), absolute refractory, per-neuron bias. dt = 0.1 ms, one versioned parameter file.
Within-step order is fixed and matches Brian2's default slots so cross-checks mean
something: **integrate → threshold → synaptic delivery (delayed and external port events
land here, affecting the next step) → reset/refractory**. Full-graph runs declare a model
for every non-spiking cell class present (graded compartment or documented silence);
"only when a primitive needs one" is not a model.

### Substrate profiles, sharpened
| Profile | Structural edits (add / reroute anatomical edges) | Parameter edits (weights within sign+bounds, thresholds, gains, documented zero weights, silenced cells) |
|---|---|---|
| 3 connectome-inspired | allowed, logged | allowed |
| 2 connectome-constrained | **forbidden** | allowed, each intervention listed |
| 1 native-function | forbidden | forbidden beyond defined inputs |

The manifest reports four graphs separately: original, retained, active-nonzero, and the
silencing list. Zeroing most inconvenient edges and operating inside a largely active
surround are different achievements; both stay visible.

### Program image manifest
```text
connectome id + version; neuron IDs; cell types; contacts vs aggregated adjacency + filter rule
NT sign assumptions + confidence; dynamical models + parameter file hash
profile; the four graphs above; ports (permitted I/O neurons)
execution mode (isolated / full-graph) and isolation classification (see below)
```

### Execution modes
- **Isolated**: compiled circuit only. Classified as one of: *equivalent under stated
  isolation assumptions* / *robust within a measured boundary-input envelope* / *not yet
  assessed*. For Profile 2, the envelope is measured in full-graph runs and replayed into the
  isolated circuit, including coordinated bursts and structured activity, not only Poisson.
- **Full-graph**: whole connectome per node with the same overrides; topology shared across
  nodes, per-node state and overrides separate; contacts with identical source, target,
  kernel, and delay merged by summing weights (different delays are not merged).

### Data classes
| Type | Physical form | Uses |
|---|---|---|
| Exact | dual-rail tokens + completion + ack | instructions, addresses, counters, branches, game state |
| Bounded scalar | rate or inter-spike interval | lighting, interpolation |
| Angle / vector | population phase + amplitude | heading, transforms (E2 only) |
| Sparse signature | sparse pattern | associative nomination, then exact tag compare |
Conversions are explicit instructions; `QUANTIZE` returns a margin flag and an
*indeterminate* status when the resultant magnitude is near zero.

### Token protocol (bounded-delay, self-timed; not claimed delay-insensitive)
```text
DATA(n): producer spikes rails → receiver bit latches capture
VALIDATE(n): C-element tree over latched bit-valid states → word complete
ACCEPT(n): consumer runs; emits ACCEPT token to producer
RESET: producer clears transaction storage; receiver latches + completion tree return to empty
READY(n+1): receiver signals empty; producer may send again
```
The contract must settle, by test: a late opposite-rail spike after apparent completion
(→ FAULT, transaction discarded, counter incremented); a duplicate after reset (rejected by
sequence); a lost ACCEPT (producer timeout → retransmit, receiver idempotent); a fork whose
consumers finish at different times (per-branch ACCEPT, join by C-element); a branch whose
unselected path never yields a token (merge is a one-of-N select, never a wait-for-all).
Timing assumptions (evaluation windows, retention, timeouts) are stated per macro.

## FlyISA v0 and arithmetic semantics
Instruction families as in your spec. Exact ops name width, sign, overflow, rounding,
fault: `ADD.WRAP.I32`, `ADD.SAT.I16`, `MUL.Q16_16`, `CMP.SIGNED.I32`.
A separate **arithmetic semantics spec** (`isa/semantics.md` + `isa/semantics.py`) fixes:
wrapping and saturating add/sub; multiply as 64-bit product then arithmetic shift with
stated rounding for negatives (floor) and saturation on out-of-range; shift counts ≥ width
(fault); division by zero (fault flag, result 0); narrowing (wrap or saturate by opcode).
The portable C helpers used by `minidoom` (`fx_add_wrap`, `fx_mul_q16`, …) and the FlyISA
lowering implement this one spec; the C reference is built with UBSan to prove it never
relies on undefined C behaviour. Execution is spatial dataflow first; a bytecode interpreter
(neural VM) is Stage G. Every primitive ships a measured contract (below).

## Circuit library
Primitive = threshold computation + input buffering + timing control + output restoration
+ completion detection + reset. Contract fields:
`resources, input/output code, latency, initiation interval, jitter tolerance, error rate
(with bound and trial count), retention and maximum supported hold, fan-in, fan-out,
supported load, background-input envelope, reset latency, maximum outstanding transactions`.

Order of construction (A1 before anything wide):
1. Bit latch (token storage) — candidates: bistable pair of recurrent assemblies with
   cross-inhibition; circulating-spike delay loop.
2. C-element on latched valid states; completion tree; reset path; READY generation.
3. Dual-rail NOT/AND/OR/XOR (thresholds 0.5 / 1.5; XOR two-stage).
4. Full adder; 4-bit ripple adder with full handshake at both ends.
Then (A2): bit-serial and ripple ALU; word register with staged commit; RAM bank with
**hierarchical predecoding** (flat one-neuron-per-word decoding is a 64k-neuron bank at
16 address bits; the address word is a namespace and only allocated capacity is
instantiated); ROM by decoder synapses onto data rails; associative index + exact tag
compare; interval codec; port codec; interrupt block; one-hot control FSM; population ring
(E2). Decoder margin note: threshold b − 0.5 on b matching inputs tolerates a uniform
input reduction of only ε < 0.5/b (3.1% at b = 16) before leakage and skew; measure, and
prefer small b per stage.

## Compiler (DrosoC → neural image)
```text
DrosoC (C subset: fixed-point ints via the semantics helpers, static pools, bounded arrays,
  no recursion, no function pointers)
  → front end: pycparser + our semantic analysis (integer promotions, symbol resolution,
    bounds validation, rejection of unsupported constructs)
  → IR (SSA; explicit width/sign/overflow; ownership; completion dependencies)
  → IR interpreter (executable oracle; dev-only, never part of the reported runtime)
  → resource plan: which loops share an ALU, which calls share a resident kernel, which
    workers share a multiplier; spill storage; emitted as the capacity report
  → macro expansion from contract-verified primitives
  → placement & routing: Profile 3 free synthesis / Profile 2 constrained embedding
    (w_ij = A_ij · w̃_ij within bounds; permitted paths; delay matching; buffering)
  → program image + manifest + provenance
```
Three comparison points for every program: `portable C reference ↔ IR interpreter ↔ neural
execution`. State comparisons use **canonical serialization** (declared field order, widths,
byte order, no padding), not host memory dumps. **Capacity report** (required before any
run above 4 nodes): live state bits, asset ROM bits, ALU/multiplier counts, control and
handshake overhead, spill storage, per-frame communication, critical-path latency.

## Cluster and hypervisor
- Logical topology: 32 pods × 31 nodes + 8 service nodes; address `node:10 | bank:6 |
  word:16` is a namespace. Physical: nodes are a batch dimension per GPU; pods map to GPUs.
- **FlyLink wire format, with field origin stated:**
  | field | produced by |
  |---|---|
  | physical source neuron id, simulator timestamp, route | external transport (allowed) |
  | destination/multicast group, port, epoch, sequence, flags, payload word(s) | neural sender's port codec, as tokens |
  | transport checksum | external transport; detects packet corruption only |
  Rule: the production simulator and transport never import game objects or ISA operation
  handlers; diagnostic observers may, in tests only.
- Exact channels: ordered, credit-based, ACCEPT/timeout/retransmit, epoch-staged receive,
  duplicate rejection. Approximate channels: bounded jitter, exposed uncertainty.
- Three clocks (neural time, transaction count, game tick); conservative synchronization.
- Fault model: neural computational faults are converted to detected rejection or
  fail-stop by dual-rail FAULT detection, sequence/epoch checks, TMR on decoded results with
  diverse placements, and a **protected** voter and commit controller (themselves
  replicated). Authoritative state uses a majority-backed commit log (crash model, Raft-
  shaped) that persists request identity and last-applied transaction, so a retried commit
  is applied once. Lose the majority → stop committing.
- Two checkpoint kinds, never conflated: **architectural** (committed objects, registers,
  continuations; the basis of any recovery claim) and **simulator snapshots** (membrane,
  synaptic, delayed-event queues; for debugging and resuming HPC runs only).
- Control plane: Python for Stages C–F, labeled hybrid; **Stage F2** replaces it with neural
  readiness, dispatch, commit, and application-level recovery for a static assignment.
- Every benchmark carries four labels: profile (1/2/3), isolated vs full-graph, hybrid vs
  neural orchestration, external vs self-hosted compilation.

## `minidoom`: a neural Doom-like engine (not a port of Doom)
- From scratch in DrosoC (~2–3 k lines of source; the instantiated circuit is what the
  capacity report measures, since expanded loops can dwarf source size). Keeps Doom's
  structures and conventions: vertexes, linedefs, sidedefs, sectors, subsectors, BSP nodes,
  things, Q16.16, 35 Hz tick, fine sine/tangent tables; module split mirrors `p_tick`,
  `p_mobj`, `p_map`, `r_bsp`, `r_segs`, `r_plane`, `r_things`, `i_video`.
- Content: small sector level with variable heights, doors, collision, one weapon, one
  enemy type (idle/pursue/attack/damaged/dead), sprites, health, exit; 160×100, 16 colours.
- Asset compiler: lumps → neural ROM at compile time; runtime texture lookup is neural.
- Framebuffer policy is explicit: **scanline/tile streaming** of pixel records
  `(frame, x, y, colour)`, not a stored double buffer (64,000 bits × 2 × 16 neurons/bit
  ≈ 2 M neurons under the unvalidated cell cost).
- E1 renderer is exact integer Q16.16 throughout. E2 adds the population accelerator with
  propagated uncertainty (|δu| ≲ f(|δy_c|/|x_c| + |y_c||δx_c|/x_c²)), exact fallback near
  boundaries, and reports fallback frequency and net resource cost.
- Benchmark wording: "neural Doom-like engine"; "neural port of Doom" is reserved for a
  future restricted-C transformation of the id source.

## Repository layout — new repo `~/programming/drosophilos` (git, uv, Python 3.12)
```
drosophilos/
  sim/        schedule.md (within-step order), ref64.py (float64 CPU reference, the oracle),
              lif_torch.py (batched CPU/CUDA/MPS production stepper), snapshot.py,
              modes.py (isolated / full-graph, envelope replay), perturb.py, brian2_check.py
  connectome/ mcns_download.py (Janelia bulk files first; Codex API fallback), mcns.py
              (loader with explicit neuron-selection + edge-filter rules), manifest.py
              (four graphs, silencing list), locate.py, embed_h0.py (Stage H0)
  protocol/   spec.md (four-phase protocol, edge cases), latch.py, celement.py,
              completion.py, tests_exhaustive/ (small FSM state-space checks)
  lib/        primitives/*.yaml + *.py, harness.py (all contract fields, envelope replay)
  isa/        flyisa.py, semantics.md, semantics.py, flyasm.py, interp.py (IR interpreter)
  compiler/   frontend_c.py, sema.py, ir.py, resources.py (sharing + capacity report),
              expand.py, place_free.py, embed_constrained.py, image.py, canon.py (state ser.)
  cluster/    flylink.py (wire format), portcodec.py, node.py, pods.py, sync.py, tmr.py,
              commitlog.py, control_plane_py.py (hybrid, labeled), neural_control.py (F2),
              slurm/
  minidoom/   src/*.c, fx/*.h (semantics helpers), assets/, golden/ (clang + UBSan build,
              canonical state + pixel dumper), asset_compiler.py
  display/    pixel record sink → PNG/mp4/live window
  bench/      report.py (four labels; wall vs simulated vs logical time; neuron use; density;
              traffic; error/recovery counts; external-computation inventory)
docs/         your spec (versioned), contracts, manifests, isolation classifications
```

## Milestones (revised order)

### Stage 0 — semantics, reference simulator, loader
- Repo + env + smoke tests: 1–2 days. Simulator validation has its own exit, no date.
- Write `sim/schedule.md` and `isa/semantics.md` first.
- `ref64.py` float64 CPU reference; `lif_torch.py`; snapshots.
- Exit: three-level agreement on a fixed circuit set — trajectories (float tolerance),
  spike-event traces (exact on CPU vs reference; CUDA/MPS certified at the transaction
  level), decoded transactions (exact); Brian2 spike-trace agreement on small circuits with
  matched scheduling; timestep-refinement test (0.1 → 0.02 ms) leaves transactions unchanged.
- Loader: verify Janelia bulk route; state filter rules; report counts under those rules
  (Codex shows 166,700 neurons / 6,242,118 connections at its threshold; the 25.5 M figure
  is unverified until reproduced).

### Stage A1 — token storage, completion, reset, register, small adder (Profile 3)
- Exit: a 4-bit word crosses a full four-phase handshake 10⁶ times under perturbation with
  zero decoded errors (observed bound ≈ 3/n); every protocol edge case above passes; the
  small protocol FSMs are checked exhaustively; contracts filed with all fields.

### Stage H0 — minimal circuit in MCNS wiring (concurrent with A1) — DONE 2026-09-13
- In real MCNS neuron IDs, Profile 2 rules: input buffer → register → add or compare →
  completion detector → return path, with surrounding circuitry present (full-graph mode).
- Exit: it works or it does not, and either way a report on motif availability, relay
  overhead, and interference, plus the measured boundary-input envelope. Findings feed the
  compiler and allocation strategy immediately.
- **Result (`docs/h0_report.md`):** motifs are plentiful (1,223 latch pairs; 141 completion
  neurons with four resettable latches; 50 complete embeddings in 1.8 s); the circuit
  computes a dual-rail AND with completion and reset in real wiring when its parasitic edges
  are zeroed; with the surround live it fails and recruits the brain (10⁵ spikes / 150 ms);
  with its outgoing edges zeroed (documented silencing) it works inside the full graph.
  Consequences for A1/A2: latches broadcast at ~213 Hz, so Profile 2 circuits need silenced
  outputs or low-out-degree neurons (isolation cost is now a search criterion); gates on
  latch trains work in rate mode (AND ≈ 30–60 ms latency), so stage latency is tens of ms,
  not 2 ms; reset needs ≥ 1.5× loop drive on both latch members.

### Stage A2 / B — input-dependent control-and-memory program
- ALU, word register, hierarchical RAM bank, ROM, control FSM, bounded stack, safe-point
  interrupts; IR interpreter; resource sharing; capacity report.
- Exit: a DrosoC program that reads runtime input, stores neurally, computes, branches,
  loops, calls, and emits a pixel matches C reference and IR interpreter on canonical state
  for fresh inputs; interrupt lands at a safe point and resumes.

### Stage C — two-node communication
- FlyLink + neural port codecs; reset, retries, stale-epoch rejection, duplicates,
  reordering, corruption, backpressure, recovery.
- **Demo 1 (first decisive demonstration):** an input-dependent program stores a value
  neurally, performs arithmetic, takes a neural branch, sends a validated result to a second
  node, and emits a neurally computed pixel; beside it, H0's circuit doing its step inside
  MCNS-constrained wiring.

### Stage D — world-update kernel
- `minidoom` `p_*` on control nodes with TMR + commit log.
- Exit: 1,000 ticks of scripted and fresh random input, canonical state equal to the
  reference every tick; a retried commit is applied once.

### Stage E1 — exact renderer
- Geometry + raster kernels, ROM assets, streamed pixel records.
- Exit: pixels identical to the reference renderer for the same committed state.

### Stage F — cluster execution on Juno (hybrid control plane)
- Scale by the capacity report; Slurm launchers; node-kill recovery from architectural
  checkpoints with no repeated world update; benchmark report with the four labels.

### Stage F2 — neural execution control
- Static assignments; neural readiness, dispatch, commit decision, recovery.
- Exit: the Python control plane is absent from the runtime for the supported configuration
  and the benchmark label flips to "neural orchestration".

### Stage E2 / G / H — distinct results
- E2 population acceleration with uncertainty and fallback accounting.
- G self-hosted compiler via neural VM bytecode.
- H full-workload Profile 2 embedding, full-graph interference tests, matched shuffled and
  free-synthesis comparisons, and a measured statement of what the fly wiring contributed.

## Work split
Interfaces first, then parallel internals: the token protocol, reset, timing, and loading
interfaces (`protocol/spec.md`, contract schema) are fixed inline before anything is
dispatched. After that: sol/opus → latch + C-element + adder, port codec, H0 embedding;
terra/sonnet → decoders, ROM, harness plumbing, display, Slurm; one Kimi review of the
simulator + protocol spec after Stage A1. `minidoom` source and the compiler front end start
in parallel once the IR and semantics spec are frozen.

## Verification and reporting discipline
- Simulator: three levels (trajectory, spike events, transactions); float64 oracle; backend
  certification; timestep refinement; matched Brian2 scheduling.
- Circuits: contracts with trial counts; bounded perturbation analysis; envelope replay;
  common-mode fault injection (shared compiler or timing defects are not independent trials).
- Reliability: 1e-10 per protected operation is the objective. Reports separate observed
  bounds (zero failures in n trials → p ≤ 3/n at 95%, so 1e-10 needs ~3×10¹⁰ trials and
  will not be "measured"), analytically predicted residual error, and fault-injection
  results. Small protocol machines get exhaustive checks.
- Retention: each memory contract states a maximum supported hold; timeouts and
  backpressure policy must keep every transaction within it.
- Golden comparisons: canonical serialized state; fresh inputs and a newly compiled level.

## Open items (need you)
1. **Data**: try Janelia's bulk downloads at male-cns.janelia.org/download first; Codex API
   token only if the bulk route lacks a needed table. Stages A1 and A2 need no data; H0 does.
2. **Juno**: read the real inventory (`sinfo`, GPU memory, allocation) before Stage F; the
   H200 claim is unverified.
