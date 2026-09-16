# DrosophilOS — results record, Stages 0 and H0

**Date:** 2026-09-13. **Repository:** https://github.com/jgassens/drosophilos (commit `5e1869e` and
this file). **Hardware:** Apple M1 Pro, 16 GB, macOS 26; Python 3.12, PyTorch 2.14, NumPy 2.5,
Brian2 2.10, SciPy. **Data:** MCNS v1.0 (male *Drosophila* central nervous system connectome,
HHMI Janelia, CC BY 4.0), bulk tables `body-annotations`, `body-neurotransmitters`,
`connectome-weights` (min confidence 0.5).

This is a record of what was built, what was measured, what worked, and what did not,
including the dead ends. Every number below is reproducible from the repository with the
commands in §7.

---

## 1. Claim boundary

DrosophilOS treats a simulated spiking network built on a real fly connectome as computer
hardware. The long-term target is a Doom-like engine whose game update **and** renderer both
execute in the neural substrate. Nothing in this record claims that. What is claimed:

1. A spiking simulator whose model is written down normatively and certified three ways
   (against a float64 reference, against Brian2, and by internal consistency tests).
2. An arithmetic specification implemented identically in Python and C, checked on 7,989
   shared vectors under Clang's undefined-behaviour sanitizer.
3. A loader for MCNS v1.0 whose stated rules reproduce the published neuron and connection
   counts exactly.
4. **Stage H0:** a minimal circuit (dual-rail 1-bit AND with a register, a completion
   detector, and a reset path) embedded in real MCNS neuron IDs under "Profile 2" rules
   (anatomical graph preserved; weights scaled within a bound; documented zeroing allowed;
   no added or rerouted edges). It computes correctly when isolated or when its outputs are
   silenced, and fails when the surrounding brain is live.

All benchmarks carry four labels: substrate profile (**2**), isolated **and** full-graph
execution, **hybrid** orchestration (the host injects input tokens and reads spikes), and
**external** compilation (the circuit was designed by hand and by a search program, not by
the machine).

---

## 2. Methods

### 2.1 Neuron and synapse model (normative text: `drosophilos/sim/schedule.md`)

Current-based leaky integrate-and-fire neuron with an exponential synaptic kernel:

```
tau_m dV/dt = (E_L + b − V) + g        tau_s dg/dt = −g
```

Defaults are the Shiu et al. (2024) whole-brain model values: `E_L = −52 mV`,
`V_th = −45 mV`, `V_reset = −52 mV`, `tau_m = 20 ms`, `tau_s = 5 ms`, refractory `2.2 ms`,
one anatomical synapse `= 0.275 mV`, delay `1.8 ms` on every edge, `dt = 0.1 ms`.
Sign comes from the presynaptic transmitter (GABA, glutamate, histamine negative; all
others including "unclear" positive). Integration is the exact closed-form solution of the
linear system over one step; the within-step order is fixed to Brian2's default slot order
(integrate → threshold → synaptic delivery → reset). Synaptic weights are **integers**
(16 quanta per anatomical synapse, `0.0171875 mV` per quantum) so that delivery sums are
order-independent and spike traces are bit-identical across backends and batch sizes.

### 2.2 Simulators

- `sim/ref64.py`: NumPy float64 reference. The oracle.
- `sim/lif_torch.py`: PyTorch, batched over nodes, CPU/CUDA/MPS.
- Comparison objects: membrane trajectories, the sorted spike-event trace `(step, node,
  neuron)`, and decoded transactions. Snapshots (state, delay queue, pending events) restore
  bit-exactly.

### 2.3 Arithmetic semantics (`drosophilos/isa/semantics.md`)

Wrapping and saturating integer arithmetic at 8/16/32 bits, Q16.16 fixed point matching
Doom's `FixedMul` (64-bit product, arithmetic shift, floor for negatives), division with
truncation toward zero and defined division-by-zero, shifts with defined out-of-range
behaviour, width conversions, and a four-bit status word (SAT, DIVZ, SHIFT, OVF).
Implemented in `isa/semantics.py` (Python integers) and `minidoom/fx/fx.h` (C, no signed
overflow, no shifts of negative values).

### 2.4 Connectome loader (`drosophilos/connectome/mcns.py`)

Rules, recorded verbatim in every manifest:

| rule | statement |
|---|---|
| neuron | a body is a neuron iff its `superclass` annotation is non-null |
| edge | both endpoints are neurons and synapse count ≥ `min_syn` |
| sign | presynaptic `consensus_nt` ∈ {gaba, glutamate, histamine} → −1, else +1 |
| delay | every edge 1.8 ms |

### 2.5 The H0 circuit

Dual-rail encoding: each bit has two rails; a spike on rail 1 means "valid one", on rail 0
"valid zero"; silence is "not here yet"; both is a fault.

| role | construction |
|---|---|
| register (four rails a1, a0, b1, b0) | two-neuron mutual excitatory loop; a circulating spike holds the value |
| y1 = a1 AND b1 | one neuron receiving sub-threshold drive from each of two latches; fires only on both |
| y0 = a0 OR b0 | one neuron receiving supra-threshold drive from either latch |
| completion | one neuron driven by y1 or y0 (single pulse suffices) |
| reset / return path | completion → inhibitory neurons → both members of every latch |

Design targets, all derived from the model's measured physics rather than assumed:

| quantity | value | how obtained |
|---|---|---|
| membrane peak per mV of synaptic jump | 0.1575, at 9.2 ms | simulated impulse response |
| single synchronous input that just reaches threshold | 2,586 quanta (161.6 synapses) | 7 mV gap ÷ peak |
| latch drive (1.4×) | 3,621 quanta | margin |
| latch loop period at that drive | 4.7 ms (213 Hz), first spike after 3.5 ms | simulated isolated loop |
| sustained-train drive that just reaches threshold | 383 quanta per pulse | mean of periodic drive |
| AND input per latch (0.65×) | 249 quanta | one latch 4.6 mV, two 9.1 mV |
| OR input (2×) | 766 quanta | margin |
| completion and reset-drive edges | 3,621 quanta (single-pulse mode) | see §4.3 |
| reset per latch member (1.5× loop drive) | −5,432 quanta | measured, see §3.4 |

Profile 2 policy: `0 ≤ |q| ≤ 4 × count × 16` per anatomical edge (k_max = 4), sign
preserved; anatomical edges among circuit neurons that are not part of the design
("parasitic") may be zeroed and are listed; outgoing edges to the surround may be zeroed
in a separately labeled condition. Required anatomical synapse counts follow from the
targets: latch ≥ 57 both directions, AND ≥ 4, OR ≥ 12, completion ≥ 57, reset ≥ 85 per
member (several inhibitory neurons may add up).

### 2.6 Search (`drosophilos/connectome/embed_h0.py`)

Sparse-matrix enumeration: candidate latches = mutual cholinergic pairs with both
directions ≥ 57 synapses; completion candidates = cholinergic neurons with ≥ 2
cholinergic inputs ≥ 57; reset feasibility precomputed as a sparse product (completion →
inhibitory partner → latch member) so that latches are filtered before gate assignment;
gate candidates from the completion neuron's strong inputs; four pairwise-disjoint latches;
solutions scored by weight margin, then isolation cost.

### 2.7 Experimental conditions (`drosophilos/connectome/h0_run.py`)

Each transaction: inject one DATA event per active rail at 5 ms (b optionally delayed),
run 150 ms, read the y1/y0/completion/reset spikes, judge correctness (exactly the expected
rail fired and completion fired), reset (all latches silent for the last 30 ms), latencies,
surround spikes and recruited neurons, and the boundary-input envelope (quanta delivered
into circuit neurons from non-circuit spikes, per neuron per step).

| condition | topology | surround |
|---|---|---|
| isolated, parasitic zeroed | 15 circuit neurons only | — |
| isolated, parasitic anatomical | 15 neurons + their 107 mutual anatomical edges | — |
| full graph, surround live | all 166,700 neurons, 25.58 M edges, designed edits applied | silent / 2 Hz on the 17,937 sensory neurons / one synchronous burst of 1,000 cholinergic neurons |
| full graph, outputs zeroed | as above, plus every circuit→surround edge zeroed (4,830 edges) | same three |

---

## 3. Results

### 3.1 Simulator certification (Stage 0)

| test | result |
|---|---|
| PyTorch float64 CPU vs NumPy reference, random 60-neuron recurrent circuit, 2 nodes, 3,000 steps | identical spike traces and identical recorded trajectories |
| batch independence (node simulated alone vs in a batch) | identical |
| snapshot at step 1,000, run 500, restore, continue | identical to uninterrupted run |
| Brian2 (`method='exact'`, matched scheduling), same circuit, 3,000 steps | identical spike trace, spike for spike |
| refractory semantics, delay semantics, silenced neurons, zero-delay self-consistency | pass |
| arithmetic: 7,989 vectors, Python spec vs C helpers under `-fsanitize=undefined` | 0 failures |
| MCNS loader vs published counts | 166,700 neurons; 25,582,938 edges at ≥ 1 synapse; 6,242,118 at ≥ 5; 124,177,617 contacts |

The loader result resolves an open point from the plan review: Codex's displayed
"6,242,118 connections" is the ≥ 5-synapse graph, and the "25.5 M" figure is the ≥ 1-synapse
graph at confidence 0.5, both between the 166,700 annotated neurons.

Physics of the default model that everything downstream depends on: a synaptic jump moves
the membrane by only 0.49 % of its size in the first 0.1 ms step and peaks 9.2 ms later at
15.75 % of it. Crossing the 7 mV threshold gap therefore needs about 160 synapse-equivalents
arriving together, and about 5,150 to fire on the very next step. Gate latencies in this
substrate are milliseconds, not steps.

### 3.2 Motif availability in MCNS v1.0

| motif class | count |
|---|---|
| cholinergic edges ≥ 24 / ≥ 41 / ≥ 57 / ≥ 80 / ≥ 120 synapses | 486,764 / 187,456 / 99,501 / 50,524 / 20,622 |
| GABA/glutamate edges ≥ 24 / ≥ 57 / ≥ 120 | 297,742 / 75,486 / 15,417 |
| mutual cholinergic pairs, both directions ≥ 24 / ≥ 41 / ≥ 57 / ≥ 80 | 5,793 / 2,354 / 1,223 / 589 |
| cholinergic 3-cycles, all edges ≥ 57 | 8,402 |
| neurons with ≥ 2 cholinergic inputs ≥ 57 | 10,782 (5,004 of them cholinergic) |
| inhibitory neurons with ≥ 4 strong (≥ 57) outputs and a strong cholinergic input | 3,796 |
| completion candidates with four resettable latches (final policy) | 141 |
| completion candidates with usable AND + OR gate pairs | 610 |
| complete embeddings found | 50 in 1.8 s (search stopped at 50; not exhausted) |

### 3.3 The chosen embedding

| role | bodyId | type | superclass | transmitter | side |
|---|---|---|---|---|---|
| latch a1 driver / partner | 32461 / 10864 | GNG457 / GNG014 | cb_intrinsic | ACh | R / R |
| latch b1 driver / partner | 523040 / 11127 | GNG108 / DNge059 | cb_intrinsic / descending | ACh | L / L |
| latch a0 driver / partner | 26764 / 10643 | GNG108 / DNge059 | cb_intrinsic / descending | ACh | R / R |
| latch b0 driver / partner | 37111 / 555296 | GNG457 / GNG014 | cb_intrinsic | ACh | L / L |
| y1 (AND) | 10881 | GNG120 | cb_intrinsic | ACh | R |
| y0 (OR) | 512079 | GNG169 | cb_intrinsic | ACh | L |
| completion | 11429 | GNG236 | cb_intrinsic | ACh | L |
| reset ×4 | 14210, 520816, 14160, 523983 | GNG048, GNG298, (untyped), GNG048 | cb_intrinsic | GABA | R, M, L, L |

The search is agnostic to biology; it picked gnathal-ganglion local neurons and, for two of
the latch partners, the descending neuron DNge059 on each side. Minimum weight margin is
0.27 against the 0.25 floor (the tightest designed edge is scaled 3.65× its anatomical
count). 107 parasitic anatomical edges among the 15 neurons were zeroed. Full neuron and
edge tables: `docs/h0_report.md`; manifest with every edit: `docs/h0/manifest_full_graph_surround_live.json`.

### 3.4 Reset requirement (measured on an isolated latch, 50 phases across one period)

| inhibitory drive per pulse | one member, 1 pulse | both members, 1 pulse | one member, 4 pulses 8 ms apart | both members, 4 pulses |
|---|---|---|---|---|
| 1,811 quanta (28 syn at k=4) | 0/50 | 0/50 | 0/50 | 50/50 |
| 3,621 (57) | 0/50 | 0/50 | 50/50 | 50/50 |
| 5,432 (85) | 0/50 | **50/50** | 50/50 | 50/50 |
| 7,242 (113) | 0/50 | 50/50 | 50/50 | 50/50 |
| 10,863 (170) | 41/50 | 50/50 | 50/50 | 50/50 |

Policy adopted: 5,432 quanta on both members, since a gate that barely exceeds threshold
fires only every ≈ 30 ms and cannot be relied on for a pulse train.

### 3.5 Transactions

Isolated, parasitic edges zeroed:

| a | b | y1 | y0 | completion | correct | completion latency | ready latency |
|---|---|---|---|---|---|---|---|
| 0 | 0 | – | ✓ | ✓ | ✓ | 22.8 ms | 26.9 ms |
| 0 | 1 | – | ✓ | ✓ | ✓ | 36.9 ms | 41.7 ms |
| 1 | 0 | – | ✓ | ✓ | ✓ | 36.9 ms | 41.7 ms |
| 1 | 1 | ✓ | – | ✓ | ✓ | 56.3 ms | 60.8 ms |

Delaying operand b by 0, 1, 2, 3, 4, 5, 6, 8, 10, or 12 ms: all correct. The gates
integrate sustained trains, so operand arrival phase does not matter.

Isolated, parasitic edges at anatomical strength: (0,0), (0,1), (1,0) correct; **(1,1)
fails** (y0 fires instead of y1) at every operand offset.

Full graph:

| condition | correct | reset | surround neurons recruited | surround spikes / 150 ms | envelope into circuit, max per step (quanta) |
|---|---|---|---|---|---|
| surround live, silent | 0/4 | 2/4 | 19,680 – 23,917 | 98,908 – 217,118 | +5,104 / −6,176 |
| surround live, 2 Hz sensory background | 1/4 | 2/4 | 29,862 – 30,045 | 457,642 – 469,185 | +6,784 / −5,920 |
| surround live, 1,000-neuron burst | 1/4 | 2/4 | 26,694 – 27,835 | 400,878 – 419,051 | +6,656 / −7,008 |
| outputs zeroed, silent | **4/4** | **4/4** | 0 | 0 | 0 |
| outputs zeroed, background | 1/4 | 1/4 | 29,730 – 30,172 | 456,967 – 468,281 | +6,784 / −7,168 |
| outputs zeroed, burst | 1/4 | 2/4 | 26,393 – 27,163 | 402,754 – 420,968 | +6,848 / −7,552 |

With outputs zeroed and a silent surround, the spike times equal the isolated run's
exactly, so that configuration is classified *equivalent under stated isolation
assumptions* (the assumption being a silent surround). Total stray input per 150 ms
transaction in the live conditions: 1.2 – 5.2 million quanta. Most active recruited
neurons: octopaminergic VUM neurons (OA-VUMa1/a8), ExR6, olfactory local neurons
(lLN2, il3LN6), APL.

Isolation cost of the chosen circuit: 4,830 anatomical edges out to the surround
(72,274 synapses; 697 edges ≥ 24 synapses) and 5,493 edges in from it (71,364 synapses;
714 ≥ 24).

Wall-clock: one 150 ms transaction takes 0.02 s isolated, 2.9 s in the full graph with a
quiet surround, 4.4 – 7.8 s with the brain erupting. The complete H0 protocol (search plus
36 transactions) runs in about five minutes on the laptop.

---

## 4. What did not work, in the order it happened

### 4.1 First search design: zero embeddings in 398 s
Reset was specified as one inhibitory neuron with a single edge of ≥ 170 synapses onto a
latch member (a 3× reset factor from a back-of-envelope estimate), and the enumeration
tested reset only after choosing four latches. 263 completion candidates had usable gate
pairs and 757,718 latch assignments were tried; none had a reset. Both the requirement and
the search order were wrong.

### 4.2 Gates designed for single pulses
The original AND assumed two coincident single spikes. A latch is not a single spike: it
emits a 213 Hz train. Under a train the mean synaptic drive is `q × w_unit × tau_s / period`,
so a single latch alone (≈ 29 mV) would have fired an AND designed for coincidence. The
gate inputs were redesigned in rate mode (one latch 4.6 mV, two 9.1 mV against a 7 mV gap).

### 4.3 Rate-mode chain starvation
With every downstream edge in rate mode, the AND (which sits barely above threshold) fired
only every ≈ 34 ms, the completion neuron received too little drive and fired twice in
150 ms, and the reset neurons never fired. Fix: completion and reset-drive edges were moved
to single-pulse mode (3,621 quanta), where each gate spike fires its target alone.

### 4.4 A misleading reset probe
The first reset probe forced the inhibitory neuron with a 100,000-quanta injection. That
much drive made it burst seven times in 18 ms, so even −1,811 quanta on one member appeared
to stop the latch at every phase. Inside the circuit the same neuron fires once per
completion spike and the latch survived sixteen of them. Re-probing with realistic
single-spike drive gave the table in §3.4. Lesson recorded as a harness rule: never
characterise a primitive with unrealistic injection.

### 4.5 Search grind from an over-strict constraint
The reset selection forbade reusing an inhibitory neuron across latches. Assignments that
passed the vectorised feasibility check then failed late, and the enumeration ran for
15 minutes at 100 % CPU without output. Allowing reuse (a strong inhibitory neuron serving
several latches is exactly what one wants) and capping tries per gate pair brought the
search to 1.8 s.

### 4.6 A column-indexing bug caught by the synthetic test
Reading a column of a CSR sparse matrix returns column indices of a one-column matrix (all
zero), not row indices. The reset candidate lookup silently found nothing until the
synthetic-connectome test (a planted motif in a 55-neuron graph) failed. The synthetic test
is now the gate before any run on the real data.

### 4.7 Manifest validation crashed the first full run
The sign-flip check compared a negative inhibitory target against the unsigned anatomical
synapse count. The run had finished all conditions and crashed while saving the manifest,
before writing results. Fixed by storing signed anatomical quanta and writing results before
the manifest. All conditions were rerun.

### 4.8 Test drives too weak for the model
The first simulator tests injected "eight synapses' worth" of drive and expected spikes.
Nothing fired: the physics in §3.1 was not yet understood. The tests were rewritten around
measured requirements, and the physics was written into the normative model file.

### 4.9 What the surround does on its own
This is the finding that matters most and it is not a bug. Under the default parameters
the whole-brain model has two states: silence and runaway. A 2 Hz drive on the sensory
neurons, or one synchronous burst of 1,000 cholinergic neurons, recruits ≈ 30,000 neurons
into sustained firing with no circuit involved. Uniform 0.275 mV synapses without
adaptation, synaptic depression, or tonic inhibition make the connectome bistable. The
review's demand — "operate within a largely active surround" — cannot be tested for any
circuit until the surround model has a stable active regime. That is a modelling task, now
on the Stage H list, and it applies to every published whole-brain LIF model with these
parameters, not just to this project.

---

## 5. What the results mean for the machine

| finding | consequence |
|---|---|
| Latches broadcast at 213 Hz through thousands of anatomical edges | Profile 2 circuits need documented silencing of their outputs, or placement on neurons with few strong partners. Isolation cost is now a search objective and a manifest line. |
| Stray input of thousands of quanta per step destroys any designed drive | Incoming surround edges must be silenced too unless the surround is provably quiet; stray-input tolerance becomes a primitive contract field. |
| Gates on latch trains work in rate mode | Stage latency is 20 – 60 ms, not the 2 ms assumed in the architecture document; a 100-stage critical path is seconds, not 200 ms. Single-pulse coincidence gates on quiet inputs and bounded threshold edits are the levers to measure in Stage A1. |
| Operand arrival phase does not matter | The self-timed protocol gets timing slack for free, paid for in latency. |
| Reset needs 1.5× loop drive on both members from a single spike | 85-synapse inhibitory edges, or several inhibitory neurons summing to it; plentiful (75,486 inhibitory edges ≥ 57). |
| The register bits landed in descending neurons | Placement policy should carry a no-touch list (sensory, motor, descending) for Profile 1/2 hygiene. |

---

## 6. Limitations and things not done

- The circuit is 15 neurons and one bit wide. Nothing here shows that thousands of such
  circuits compose; Stage A1/A2 will measure composition contracts.
- "Works in real wiring" means: with 30 designed edges rescaled up to 3.65×, 107 parasitic
  edges zeroed, and (in the only working full-graph condition) 4,830 outgoing edges zeroed.
  Preserving an adjacency matrix while zeroing inconvenient edges is the weaker of the two
  achievements the review distinguished; the stronger one was not achieved and, given §4.9,
  is not currently testable.
- Non-spiking cells (APL and others) are simulated as LIF neurons in full-graph mode; the
  declared model for them is "same as everything else", which is a documented gap.
- All delays are 1.8 ms; MCNS provides no per-synapse delay data.
- The k_max = 4 weight bound is a policy choice, not a physiological measurement.
- The search was stopped at 50 embeddings; the count of all embeddings is unknown.
- No matched-shuffled-graph control has been run (Stage H), so nothing here says the fly
  wiring is better or worse than a random graph with the same degree structure.
- Simulator certification on CUDA/MPS at the transaction level, and the timestep-refinement
  test, are specified but not yet run (no primitives exist yet to run them on).
- Brian2 parity was checked on one random circuit family; the circuit-library harness will
  extend it.

---

## 7. Reproduction

```bash
git clone https://github.com/jgassens/drosophilos && cd drosophilos
uv sync                                                    # Python 3.12 environment
uv run pytest                                              # 22 tests, ~2 min (Brian2 included)
uv run python -m drosophilos.connectome.mcns_download      # 1.1 GB from Janelia, no login
uv run python -m drosophilos.connectome.mcns               # builds the cache, prints the counts
uv run python -m drosophilos.connectome.h0_run             # search + 36 transactions, ~5 min
uv run python -m drosophilos.bench.h0_report               # regenerates docs/h0_report.md
uv run python -m drosophilos.isa.semantics                 # regenerates the arithmetic vectors
```

Files: normative model `drosophilos/sim/schedule.md`; semantics
`drosophilos/isa/semantics.md`; plan `docs/plan.md`; architecture specification
`docs/spec.md`; the review that shaped the plan `docs/review-2026-09-13.md`; H0 data
tables `docs/h0_report.md`; H0 conclusions `docs/h0_findings.md`; H0 raw results and manifest
`docs/h0/`.

## 8. Attribution

Architecture specification and review: Jeremiah Gassensmith. Implementation, experiments,
and this record: Claude Fable 5.1 (Anthropic) working in Claude Code, directed by the
author. Connectome: MCNS v1.0, HHMI Janelia FlyEM project team and collaborators
(CC BY 4.0). Model parameters: Shiu et al., *Nature* 2024.

---

# M1 (Stage A1) — reliable neural word transport and arithmetic

**Date:** 2026-09-13/14. Full report with all tables: `docs/m1_report.md`; contracts:
`docs/contracts/`; storage comparison: `docs/m1_latches.md`; raw data: `docs/m1/`.

## What was built

- A four-phase, dual-rail, self-timed token protocol written as a spec and as an
  executable abstract machine, with an exhaustive checker (every interleaving, with fault
  injection). It proves the protocol's properties and demonstrates the one hazard it
  cannot remove by itself (stale tokens need a timing bound or phase rails).
- The neural implementation of that protocol: two-neuron loop latches, latched rate-mode
  gates, a latched completion tree (a state-holding C-element with explicit
  return-to-empty), edge relays for one-pulse ignition, an edge-detected reset trigger
  with a 4-pulse inhibitory train, an 11-hop READY/CLEARED delay chain, and a latched
  FAULT path that refuses corrupted words without deadlocking the channel.
- Gates (NOT, AND, OR, XOR, 2-of-3 majority), a 1-bit full adder and a 4-bit ripple-carry
  adder, all exact.
- A perturbation-campaign harness (batched independent channel copies; weight noise,
  threshold and bias drift, stray input, arrival jitter; post-hoc decode into error
  classes), a failure hunter that dumps the failing stage, and contract measurement.

## What worked

| circuit | neurons | ACCEPT latency | cycle | spikes / transaction |
|---|---|---|---|---|
| 4-bit channel | 125 | 112 ms | 247 ms | 1,221 |
| 1-bit full adder channel | 153 | 169 ms | 305 ms | ~1,300 |
| 4-bit ripple adder channel | 444 | 244–249 ms | 380–384 ms | 5,288 |

- 10⁶ four-bit transactions under perturbation (4 % weight noise, ±0.2 mV threshold and
  bias, 5 Hz × 2.6 mV stray input on every neuron, ≤ 10 ms arrival jitter): **0 wrong
  values consumed** (exact 95 % upper limit 3.0 × 10⁻⁶); 22 transactions (observed
  2.2 × 10⁻⁵, upper limit 3.1 × 10⁻⁵) did not complete — 3 false faults raised by the
  neural machine, 11 no-accept hangs and 8 stale-activity cases noticed only by the
  benchmark harness, since no neural timeout exists yet. ACCEPT latency p99 119 ms.
  The campaign ran on the transport channel; the adder has not had its own.
- Stress sweep (10⁴ per level): arrival jitter to 40 ms is free; stray input to 20 Hz costs
  2 × 10⁻⁴ (false faults only); silent wrong values appear only at ≥ 8 % weight noise or
  ±1 mV threshold drift.
- Fault injection: duplicates absorbed; corrupted bits detected, refused, recovered; late
  opposite rail flagged only; stale spikes never consumed wrong.

## What did not work

Ten designs were rejected on measurement, in this order: an output-side READY veto (its
own pulses hyperpolarised READY); a self-inhibiting one-shot (deep after-hyperpolarisation);
a single strong reset pulse (5 ms window, tree latches re-ignited); half-strength reset
pulses (+10 % loops survived); continuous DATA drive (consumer latches above the standard
rate, false faults); gates driving latches directly (re-ignition during reset); a 0.65×
rate-mode AND (failed at −10 % weights); edge relays and reset triggers racing their own
inhibitor with equal drives (bits dropped, then CLEARED became a train); unlatched fault
gates (a second reset wiped the next word). Each is in `docs/m1_report.md` §6 with the
measurement that condemned it.

Still open: the residual 2 × 10⁻⁵ availability failures at mix B (one caught timeline shows a
completion latch surviving the reset train), which the machine itself does not yet detect;
timeouts for lost tokens; a composition campaign on the adder; the 10⁻¹⁰ objective remains
analytical (the exact 95 % upper limit on silent errors from 10⁶ trials is 3.0 × 10⁻⁶).
Verdict: M1 closed for functional correctness and safety, with a liveness/availability
defect carried into A2/B.

## The design rules M1 settled

1. Every gate reads only latch trains (a standard 213 Hz rate); every gate output that
   feeds another gate is latched through an edge relay.
2. The membrane's 20 ms recovery from inhibition is the dominant timing constraint; the
   READY chain length and the total reset charge are set by it.
3. A relay or trigger and its own inhibitor must never race: the relay gets 1.8× the
   single-pulse need (the largest doublet-free drive), the inhibitor 2.2× the loop drive.
4. FAULT is a state, held in a latch, not a pulse.
5. There is no low-activity register in this neuron model (`docs/m1_latches.md`); the
   two-neuron loop at 1.4× is the production register, at 44 spikes per 100 ms per bit.

---

# A2 opening — liveness, and what composition exposed

**Date:** 2026-09-14. Full note: `docs/a2_liveness.md`; contracts: `docs/contracts/`; raw
campaign data: `docs/a2/`.

## What was built

- A neural **watchdog**: a delay chain started when the producer's register activates and
  cancelled by ACCEPT or FAULT-ACCEPT; if it completes it ignites a TIMEOUT latch that raises
  FAULT-ACCEPT itself. A transaction that starts and cannot complete is now refused by the
  machine (severed data path: refusal at 227 ms, channel recovered, next words correct).
- A **stale-state monitor** (ENABLE armed after the reset train, one AND per latch, STALE
  latch holding READY and re-running the reset). Built, measured, rejected: its detectors
  fire spuriously under perturbation and a single spike ignites STALE.
- Campaign classes `timeout` (neural) and `ok_stale_retry`; exact Clopper–Pearson bounds;
  a `detected_by` split between the machine and the harness.

## What the adder composition campaign exposed, and the fixes

The first 10⁵ random additions at mix B failed ~25 %. Three causes, each fixed on
measurement: adder-internal latches were reset at 0.5× per pulse instead of 0.75×; the
2× ignition pulse was a doublet that let latches briefly run at up to +22 % and pushed
one-input ANDs over threshold in long-exposure gates (ignition now 1.8× need, latch rates
195–236 Hz); and with tight rates the two-input AND's usable window sits at 0.65
(0.75 leaks 4 %, 0.55 fails to fire). ACCEPT latency rose from 112 to 155 ms on the channel
and from 245 to 358 ms on the adder.

## Campaigns on the final build (mix B)

| circuit | transactions | wrong values | raised by the machine | harness-only | non-ok observed / 95 % upper | ACCEPT p99 |
|---|---|---|---|---|---|---|
| 4-bit channel + watchdog | 100,000 | 0 | 0 | 1 stale-activity | 1.0 × 10⁻⁵ / 4.7 × 10⁻⁵ | 170 ms |
| 4-bit ripple adder, random operands | 100,002 | **0** | 30 faults, 1 timeout | 24 no-accept, 9 stale-activity | 6.4 × 10⁻⁴ / 7.9 × 10⁻⁴ | 388 ms |

Exact 95 % upper limit on silent wrong values: 3.0 × 10⁻⁵ for each run.

## What did not work

The stale monitor (false STALE in ~2 % of perturbed transactions, wrong values through
mid-transaction resets); a second unconditional reset train (no gain at the ±20 % margin
edge); AND fractions 0.55 and 0.75 on the final build; the first two adder campaign
launches (wrong period and watchdog for the slower build; a `sed` that silently did nothing).

## Status

The M1 channel contract is **frozen** for A2/B (`docs/contracts/channel_4bit.yaml`). The
adder's residual 6.4 × 10⁻⁴ is now half neurally detected; the 24 missing completions that
the watchdog did not convert are the first item of the A2 gate work, together with a wider-
margin AND. Then the ALU, word register with staged commit, RAM, ROM, control machine.


---

# A2 step 6: ALU and word register with staged commit

**Date:** 2026-09-14. Full note: `docs/a2_alu_register.md`; contracts: `docs/contracts/`;
campaign summaries: `docs/a2/`.

## What was built

- **ALU** (`drosophilos/lib/alu.py`, 4-bit: 1,191 neurons): units ADDER (ADD and SUB through
  B xor SUB), AND, OR, XOR, PASSB (MOV); flags C, Z, V; one-hot unit select. Bit-exact against
  `isa/semantics.py` on all 256 signed 4-bit pairs; every op exact on the reference
  simulator, 1-bit exhaustive and 4-bit corners. ACCEPT 300 ms (MOV) to 520 ms (ADD/SUB).
- **Word register with staged commit** (`drosophilos/lib/staged.py`, 4-bit: 321 neurons):
  a stage (channel consumer with completion and fault latch) and a master with its own
  completion (W_M = readable). A COMMIT token is granted only against a complete, fault-free
  stage; the commit clears the master, copies through veto relays, and W_M's rise clears the
  stage. A faulty or timed-out word is discarded and the master is untouched. Early, late
  (2 s), duplicate and premature COMMIT behaviours are tested and stated. Commit 247 ms.
- **Accumulator** (1,410 neurons): ALU → stage → master → ALU operand A. The host supplies
  (B, op) and one COMMIT per instruction and reads the master; the value that feeds back is
  never touched by the host. 720–910 ms per instruction.
- Two new library primitives: the **veto relay** (a one-shot AND of a later operand's rail
  against an earlier operand's other rail, no exposure window) and the **light relay hold**
  (an ignited latch keeps its own ignition relay quiet through a −0.25× interneuron).

## What the campaigns exposed, and what changed

Five mechanisms, none visible in clean runs, each found from the spike anatomy of one
perturbed node and fixed on measurement:

1. **A level is not a token.** The master's rails restarted eager gates 25 ms after the
   stage's reset, before their relays had re-armed: 17 % of instructions hung. The master
   now enters the ALU only through an ACTIVE-gated operand gate that lives with the producer.
2. **Edge relays re-fire on slow sources.** A rate-mode gate fires at 25–40 Hz and its
   relay's inhibition decays between spikes; every gate-fed latch got an extra ignition pulse
   every ~43 ms and ran ~10 % fast, which is the regime in which one-input rate-mode ANDs
   leak. Fix: the ignited latch holds its relay. This most likely also explains the adder's
   residual 3 × 10⁻⁴.
3. **A refused word is not a hang.** The campaign classifier filed a FAULT-ACCEPT without
   completion as `no_accept`; corrected, and the adder campaign's 24 "missing completions"
   are re-labelled as an upper bound on hangs (most are neural refusals).
4. **The veto relay.** Rate-mode ANDs with ordered inputs (the result mux, the flag gating,
   the operand gate, the master copy, the logic units, the adder's first XOR) leaked on the
   ±10 % latch-rate spread (a MOV refused after 150 ms; master faults from copy gates firing
   on COPY alone). All are now veto relays: 3 neurons and ~5 ms instead of 5 and ~35, and
   nothing to leak. Routing B through the B xor SUB stage makes it the later operand by
   ≥ 50 ms, which is what the ordering needs.
5. **A relay's head start is ~5 ms.** A hold or veto through the relay's 2.2× inhibitor
   paralysed the relay for ~86 ms, longer than the gap between a reset and the next load;
   every relay whose target had held in the previous transaction failed. Holds are now
   −0.25× and vetoes −0.5×, with a stated recovery budget (40 / 55 / 86 ms).

## Campaigns on the final build (mix B)

| block | transactions | wrong values | refused / hung | non-ok observed / 95 % upper | ACCEPT p99 |
|---|---|---|---|---|---|
| 4-bit ALU, random operations | 5,000 | 0 | 0 | 0 / 6.0 × 10⁻⁴ | 497 ms |
| 4-bit staged register, random words | 5,000 | 0 (stage and master) | 0 | 0 / 6.0 × 10⁻⁴ | 170 ms |
| 4-bit accumulator, random programs | 2,000 | 0 (stage and master) | 0 | 0 / 1.5 × 10⁻³ | 490 ms |
| 4-bit channel, re-validation after the relay fix (freeze re-issued) | 100,000 | 0 | 0 | 0 / 3.0 × 10⁻⁵ | 170 ms |

Before the fixes the same harness measured, at a harsher mix: accumulator 7 hangs in 40,
ALU 6 refusals in 500, register 10 non-ok in 5,000 (2 master faults). These are composition
checks with a recorded trial count, not attempts on the silent-error bound.

## What did not work

Rate-mode ANDs as the operand gate, the mux and the copy (leaks under the latch-rate
spread); the hold through the 2.2× inhibitor (relay paralysis); a direct host load of the
master at power-up in the campaign (its W_M rise resets the stage mid-instruction; programs
now start with MOV). Two runner bugs were mine, not the circuit's: reading W_M's dying train
as commit-done, and injecting the "opposite rail" of the producer word instead of the staged
word.

## What remains rate-mode, and why it matters

B xor SUB, the adder's sum and carry, and the zero tree have inputs with no fixed order and
stay rate-mode ANDs at 0.65. They expose one input for a whole transaction when the other
rail never comes, and at the harsher mix they still produced ~0.4 % refusals (never a wrong
value) before the campaigns above. An ordered full adder (per-bit delay of x so that the
carry is always the earlier operand) would move the whole adder onto veto relays; it is the
next gate-work item, ahead of the RAM and the control machine.


---

# A2 step 6b: the ordered datapath (adder on veto relays only)

**Date:** 2026-09-14. Full note: `docs/a2_alu_register.md` §4; contracts: `docs/contracts/`;
campaign summaries: `docs/a2/`.

## What was built

Every AND in the ALU's datapath is now a veto relay, because the arrival order of its two
inputs is fixed by construction with delay chains (`protocol/celement.py::add_delay_chain`):

- **B is delayed 6 hops** (~32 ms) before bx = SUB xor B^d, so SUB is the earlier operand.
- **A enters through an operand gate** driven by ACTIVE (an OR-latch over the unit-select
  rails) delayed 11 hops, so A^d (~70 ms) is the later operand against bx (36–46 ms). The
  same gate tokenises a master's level or a producer's rails.
- **Each carry is delayed 5 hops** before the next stage reads it, so the carry-in is the
  later operand against that stage's x = bx xor A^d. Generate and kill relays are driven by
  A^d and vetoed by bx; propagate relays by the delayed carry, vetoed by x; the sum is an
  ordered XOR. 29 ms per stage.
- **V** from the delayed top carry with A^d and bx vetoes (p_top = 1 → 0; generate → not c;
  kill → c). **Z** on the consumer: its completion node over the R bits, delayed 3 hops,
  drives Z1 vetoed by every R rail 1; Z0 is an OR of the R rail-1 latches.

`Gates.ripple_adder_ordered`, `Gates.operand_gate`, `Gates.delayed`, `Gates.overflow_ordered`,
`alu.add_zero_flag`; `build_adder_channel(ordered=True)`.

| block | neurons | ACCEPT (clean model) |
|---|---|---|
| 4-bit ripple adder, ordered | 594 (rate-mode: 603) | 330–384 ms (rate-mode: 360) |
| 4-bit ALU | 1,150 (was 1,191) | 241 ms (MOV) – 524 ms (SUB, full propagate, Z = 1) |
| 4-bit accumulator | 1,323 (was 1,410) | 660–830 ms per instruction |

## What went wrong on the way

The first Z drove its relay from the top result bit delayed 32 ms, assuming the top sum is
last. In a ripple adder a lower sum can wait on a long propagate chain while the top carry
was decided early by a kill: at the harsh mix every ALU refusal (13 in 500) and the one
accumulator refusal were Z double rails. The spread between result bits grows with the
width, so no delay fixes it; "all bits valid" is a completion, and the consumer's tree
already has that node. The corrected Z costs no gates and one tree level of latency.

## Campaigns on the final build (mix B)

| block | transactions | ok | wrong values | refused / hung | non-ok observed / 95 % upper | ACCEPT p99 / max |
|---|---|---|---|---|---|---|
| 4-bit ripple adder, ordered, random operands and carry-in | 30,000 | 30,000 | 0 | 0 | 0 / 1.0 × 10⁻⁴ | 397 / 420 ms |
| 4-bit ALU, random (A, B, op) | 5,000 | 5,000 | 0 | 0 | 0 / 6.0 × 10⁻⁴ | 516 / 555 ms |
| 4-bit accumulator, random programs | 2,000 | 2,000 | 0 (stage and master) | 0 | 0 / 1.5 × 10⁻³ | 509 / 535 ms |

For comparison, the rate-mode adder's 10⁵ campaign at the same mix (`a2_liveness.md` §4)
observed 6.4 × 10⁻⁴ non-ok (30 faults, 1 timeout, 24 refused-or-hung, 9 stale), all of them
refusals or hangs and none a wrong sum. Probes at the harsher B+ mix on the final ordered
build: ALU 500/500, accumulator 120/120 clean (the delay-based Z had 13/500 and 1/120).

## What is still rate-mode

The completion trees (bounded exposure: every bit becomes valid), the fault gates (0.55 per
rail), ACTIVE's OR, and the register's grant AND(COMMIT, W_S), whose exposure is the control
machine's reaction time. The carry chain is linear in the width (29 ms per bit); wide words
will need carry-lookahead or bit-serial arithmetic, which the capacity report decides.


---

# A2/B: data RAM and the control machine

**Date:** 2026-09-15. Full note: `docs/a2_ram_control.md`; contracts: `docs/contracts/ram_8x4.yaml`;
campaign summaries: `docs/a2/`.

## What was built

- **Data RAM** (`drosophilos/lib/ram.py`): W words × n bits, each word a master register.
  Word-select is a veto relay vetoed by the mismatching address rails, so decoding is exact
  with no threshold margin (hierarchical predecoding would only cut fan-in). Write port
  (select → word reset → READY → copy relays → completion → "written" pulse) and read port
  (per-rail relays into the destination). 8 × 4 block: 1,626 neurons, write done 506 ms,
  read 417 ms. An unwritten read double-rails every bit and is refused.
- **Control machine** (`drosophilos/lib/control.py`): a program of 8 words held in latch-only
  neural memory and loaded by the host as the initial image; a one-hot PC ring and a
  one-hot FSM ring (FETCH → COMMIT → NEXT) whose kills are edge-triggered; an instruction
  register of kill pairs; the accumulator with an ordered grant; the RAM as data memory.
  ISA: MOV/ADD/SUB/AND/OR/XOR immediate, LOAD, STORE, JZ, JNZ, HALT. Every instruction
  commits the accumulator, so the committed master is the architectural state after each
  instruction. A Python reference executes the same ISA; random programs are checked
  against it commit by commit and word by word. 4,230 neurons; 920–1,130 ms per
  instruction.

Only the host's allowed actions are used: it loads the program and data image, lights PC
line 0, and reads spikes. Fetch, decode, execute, commit, load, store, branch and halt are
neural.

## What the composition exposed

Five mechanisms, found from spike anatomy and fixed on measurement:

1. A lit line must not inhibit its successor: ring kills are edge-triggered trains, not
   continuous inhibition (the first FSM never left FETCH).
2. A reset train paralyses a latch for ~80 ms, so the instruction register cannot be cleared
   by a train and reloaded 20 ms later; its bits are kill pairs.
3. A relay must be driven ≥ 55 ms after any of its veto rails dies: the previous PC line's
   "not this word" veto blocked the instruction fetch at +34 ms. The fetch stages are spaced
   accordingly (+82, +156, +177 ms).
4. A "written" or "committed" pulse counts only in COMMIT: the data image's completions at
   power-up were advancing the PC.
5. A relay that feeds an edge-detected trigger must be doublet-free: one node's word-select
   fired twice 3.6 ms apart, three reset trains stacked, and the word could not hold its
   copy. Such relays now drive their inhibitor at the head-start drive.

## Campaigns on the final build (mix B)

| block | transactions | ok | wrong values | refused / hung | non-ok observed / 95 % upper | wall |
|---|---|---|---|---|---|---|
| 8 × 4 data RAM, random writes and reads (every word written first) | 4,000 | 4,000 | 0 | 0 | 0 / 7.5 × 10⁻⁴ | 96 min, 3 CPU threads |
| control machine, random 8-word programs (probe) | 10 programs / 65 instructions | 65 | 0 | 0 | 0 / — | 39 min, 4 CPU threads |
| control machine, 100 random programs (713 instructions), fixed ALU | 100 programs | 98 | 0 | 2 (1 short, 1 no halt: fail-stop) | 2.0 % / 6.2 % (silent wrong ≤ 3.0 × 10⁻²) | 262 min on one H200 |

Before the doublet fix (§3.5) the RAM block's first 10-node probe had one node with 11
non-ok in 20 (a word that could not hold its copy); after it, 200/200 and then 4,000/4,000.

## Stage B exit: a compiled DrosoC program on the neural machine

- **Compiler path** (`drosophilos/compiler/`, `drosophilos/isa/ir.py`): pycparser front end for
  the v0 DrosoC subset → three-address IR → (a) the IR interpreter on `isa/semantics` (the
  oracle), (b) lowering to the accumulator machine (calls inlined, ports memory-mapped),
  (c) a clang + UBSan golden harness printing the canonical state and the emitted pixels.
- **Three-way agreement.** A program that reads an input, sums in a loop through a call,
  masks, branches, stores and emits a pixel (17 IR ops, 31 machine words): C reference, IR
  interpreter and the lowered program's reference agree on canonical state and pixel for four
  fresh inputs; the neural machine v1 (8-bit, 32 program words, 12,450 neurons) executes it
  in 37 instructions with the same state and pixel, commit for commit.
- **Safe-point interrupts.** A pending-interrupt pair set by the host, taken at NEXT (after
  the commit and any store), PC saved to a link ring, `IRET` restores it, nested interrupts
  masked. Two interrupts during a loop: the handler runs twice and returns to the interrupted
  word each time; the main program's result is unchanged.
- **Capacity report** (`bench/capacity.py`): program-image bits, live state bits, neurons by
  class, critical path. For the test program: 713 image bits, 59 live bits, 12,450 neurons of
  which 7,392 are the program image (~230 per instruction word), 55 instructions ≈ 58 s for
  input 4.

## What the machine is not yet

Fail-stop only (a refused instruction halts the machine in FETCH); no call stack (calls are
inlined, so no recursion); about one second per instruction, of which ~180 ms is fetch
spacing imposed by veto residuals; the program image costs ~230 neurons per word, which
argues for resident kernels (spatial dataflow) for the hot loops. Stage C (FlyLink, two
nodes) is next.


---

# Stage C, first milestone: two nodes over FlyLink

**Date:** 2026-09-15. Note: `docs/stage_c_flylink.md`.

`cluster/flylink.py` transports spike events between simulated nodes (host role: transport
only; a packet is source node, source neuron, step; the receiver's mapping is fixed at link
creation; modeled positive delay). A machine exports an output port word as one edge relay
per rail and takes an input port word's completion as an interrupt. Demo 1 shape on the
clean model: node A reads an input, computes, branches, stores neurally and sends its result;
node B's arrival interrupt runs a handler that loads it, adds one and emits the pixel. A's
word complete at 9.1 s, B received it 21 ms later, the pixel landed at 13.2 s, values as the
references compute them. Not yet: sequence/epoch, credits, acknowledgement, retransmit,
duplicate rejection, corruption, backpressure (the rest of Stage C). Rule found: a program
must write the accumulator before any OR-based instruction (the lowering emits `MOV 0`).


---

# Smoke test: Hello, World on the neural machine

**Date:** 2026-09-15. `examples/hello.c`, `drosophilos/bench/hello.py`.

`examples/hello.c` calls `out_pixel()` thirteen times. The DrosoC front end compiles it to
29 machine words (a MOV/STORE pair per character after the accumulator prologue); machine
v1 (8-bit, 32 program words, 11,230 neurons) executed the 28 instructions in 29.8 s of
neural time (22 minutes of wall time on three CPU threads of the reference simulator).

How it is known to have worked, with nothing taken on trust from the machine:

1. The same C source, compiled by clang with the runtime shim and UBSan, printed the byte
   list 72, 101, 108, … — `'Hello, World!'`. That is the reference.
2. The IR interpreter, running the front end's output on the arithmetic semantics, produced
   the same list; the lowered program's Python machine reference produced the same list.
3. In the neural run the host did three things: ignited the program image into the
   instruction latches at step 1, lit PC line 0, and read spikes. Everything else (fetch,
   decode, the ALU pass, the commit, the store into the port word, the PC advance) was
   neurons. The decoder watched one neuron, the completion latch of the output port word
   (the root of that word's C-element tree), and each time it started firing after a
   silence it read which rail neuron of each of the word's eight bits had fired in the last
   9.4 ms. Those thirteen decoded bytes, in arrival order, were compared with the reference
   list and were equal, character for character. Any fault would have shown as a double
   rail (decoded as `?`), a missing completion (a shorter string), or a wrong byte.
4. The timing is consistent with the measured instruction cycle: 28 instructions in 29.8 s
   is 1.06 s each.


---

# A frame of a Doom-like view rendered in the substrate

**Date:** 2026-09-15. `examples/frame.c`, `examples/game.c`, `lib/kernel.py`,
`bench/render_frame.py`, `bench/render_game.py`; `docs/a3_kernels.md` §9.

![the frame, computed by the neural machine](docs/img/frame160_neural.png)

The picture above was computed by the neural machine: 160 × 100 pixels, one token per
pixel, on 128 copies of a 16-cell pixel kernel (64,948 LIF neurons each, 8.3 million in
all) on one H200 of Juno. Every one of the 16,000 pixels equals the reference renderer's
(`docs/img/frame160_reference.png`); 411 s of neural time, 2 h 13 min of wall time. The
host dealt the pixel tokens to the copies and put a palette on the decoded colour indices;
the wall distances, the perspective heights, the ceiling / wall / floor decisions and the
distance shading were all spikes.

Three frames with the view turning between them (`examples/game.c`: a tick kernel turns the
heading, replicated in every copy; 80 × 50 pixels, 128 copies): 12,000 pixels, all correct,
344 s of neural time, 2 h 3 min of wall time (`docs/img/game80_0_neural.png`, `_1_`, `_2_`).
The same program at 8 × 5 on four CPU copies: correct, 2 h 43 min of wall time for 139 s of
neural time — the simulator, not the substrate, is the clock.

How far this is from the fly's own wiring (Profile 2): the render kernel needs ~5,300 latch
pairs; MCNS has 1,223 reciprocal cholinergic pairs strong enough at the H0 weight policy and
12,315 at four times that scale (`docs/capacity_doom.md` §5). A 4-bit adder fits the H0
policy in count; a kernel needs a larger weight scale or added edges. Every result here
carries the label *Profile 3*.

**Stage H, first measurement (2026-09-15, `docs/h1_placement.md`):** a motif-level placement
search (`connectome/embed_netlist.py`) puts the 4-bit adder's netlist — 614 neurons, 1,146
edges, 1,144 distinct — onto real MCNS neurons under the H0 rules with 704 edges (61 %; 701 of
the distinct pairs after an audit correction) carried by the fly's
own synapses and 612 neurons placed, in 65 s; the neuron-by-neuron greedy tool carried 9 %.
Of the edges between motifs (latch pairs, relays with their inhibitors, delay chains, vetoes)
79 % are carried. What the wiring lacks is specific: the three broadcast neurons (reset and
watchdog-cancel, 258 edges, 7 carried — no inhibitory neuron in MCNS reaches 115 targets at
the required strength), one relay fan-out node, and about 30 relay-inhibitor edges. The
placed circuit was then simulated (`connectome/embed_image.py`): with the 442 missing edges
added as labelled Profile 3 edges and the 17,845 parasitic anatomical edges among the hosts
zeroed, the adder on real fly neurons computes all 50 additions with the netlist's own
timing; with only the carried 61 % it computes none (the channel stalls before its ACCEPT);
with the parasitic edges kept at anatomical weight it computes none either (they light the
output rails within 15 ms). The 191 missing logic edges are what a first sum needs; the 251
broadcast (reset) edges are what the second needs. That is the exact size of the gap between
the fly's wiring and a working adder under the H0 weight rules. Loosening the weight bound
(k_max 4 → 8) or splitting each broadcast neuron into a tree of identical copies
(`Netlist.split_hubs`, no timing change) each shrinks the gap by about a fifth (to 337 or 346
added edges) and they do not add up; the bare placement still computes nothing, and a reset
that reaches half its latches is no reset (`docs/h1_placement.md`, the last three sections).
Inside the whole simulated brain (166,702 neurons, `bench/h1_fullgraph.py`, Juno) the same image
computes 10 / 10 additions only when the hosts' 327,497 inputs from the rest of the brain are
silenced; with them live it computes 0 / 30 under a silent, a 2 Hz Poisson or a burst surround —
its own output spikes wake 24,000 surround neurons, and 20,000–45,000 spikes per millisecond come
back into hosts that average ~530 external inputs. With the hosts' outputs into the brain
silenced instead (302,263 edges) and nothing outside driven, it computes 10 / 10: in a quiet
brain the storm is the circuit's own echo. Scoring quietness in the placement search cut the
hosts' exposure only 45 % at a cost of a third of the coverage (latch-capable neurons are the
brain's hubs), and that mapping still computes 0 / 20 in the whole brain. The simulated brain
under a 2 Hz sensory drive storms on its own at these parameters (~1.9 M spikes per 1.2 s);
Stage H0's 15 quiet hosts survived it, 614 hubs do not. And the ceiling: MCNS has only 343
*disjoint* reciprocal pairs strong enough for a latch at the H0 weight bound (1,047 at ×8,
2,896 at ×16); one 8-bit kernel cell needs 375, the pixel kernel ~5,300, the three Doom
kernels ~35,000. On the fly's own synapses, one neuron per designed neuron, the wiring holds
about one cell; 3- and 4-neuron loops as latches add a few percent, no more
(`docs/capacity_doom.md` §5, "The ceiling"). Against matched controls the fly's wiring is
specific: a random graph with the same degrees has 24 strong reciprocal pairs to the fly's
1,223 and carries 50 % of the adder to the fly's 60 %; permuting the synapse strengths over
the real edges drops it to 44 %; and permuting the transmitter labels *raises* it to 65 %,
because many of the fly's strongest reciprocal pairs are inhibitory and useless to an
excitatory latch (`docs/h1_placement.md`, shuffled controls). Counting those inhibitory pairs
as flip-flop latches instead — 358 / 1,105 / 3,717 disjoint driven pairs at ×4 / ×8 / ×16, on
hub neurons too (a mean of 6,677 external input synapses at the H0 bound, 2,137 at ×16) and in
the optic lobe's medulla columns at ×16 — would roughly
double the supply, to about one pixel kernel's worth at ×16. **That latch now exists**
(`protocol/flipflop.py`, `docs/a1_flipflop.md`, `docs/contracts/flipflop.yaml`): two inhibitory
neurons on a 58 mV bias in mutual inhibition, the firing member at 212.8 Hz — the excitatory
latch's exact period, so every primitive that reads a rail (edge relays, veto relays, rate
gates) sees the same statistics — bistable from 0.8× the loop weight, holding 2 s in either
state, switched by trains of three pulses and never by one standard ignition pulse; its noise
margins are wider than the latch's (1.8× ignite on the silent member against the latch's 1.1×
need; 2.25× loop on the firing member against 2×). What does not carry over: anything that
ignites or resets a latch by a single pulse into its members needs a three-relay adapter
(`add_set_chain` / `add_clear_chain`), and a flip-flop must be pulsed once at power-on. Per-neuron
bias is now a field of the netlist (a Profile 2 parameter edit, counted in the manifest),
carried by the topology into both simulators. Because its members are inhibitory, nothing
downstream could read it with the right sign; an excitatory proxy neuron silenced by the
CLEAR member fixes that (it fires step for step with the SET member, 18–23 ms after SET,
silent within 18 ms of CLEAR), and the fly has a proxy for almost every usable pair (358 /
1,030 / 3,712 disjoint pair-and-proxy triples at ×4 / ×8 / ×16). A dual-rail register on
flip-flop storage crosses the four-phase handshake (48 fail-stop errors and no wrong value in
10,000 mix-B transfers, against ~1 in 100,000 for the latch register); the placement motif for
flip-flops is in, and with the proxy it is the better fit for the fly: a toy of 32 flip-flops
with proxies, joined by relays, places with **219 of 220 edges carried (99.5 %)** — every
triple and every proxy readout on real neurons — where the same toy of 32 excitatory latches
reaches 86 % (`docs/h1_placement.md`, "Placing flip-flops"). The hosts are hubs (a mean of
6,500 external input synapses), so the whole-brain exposure problem stands. The register
now reads its rails through the proxies (233 neurons for 4 bits, accept latency 166 ms against
152 for the u-readout and 154 for the latch register): 200 random transfers with no error, and
in 10,000 mix-B transfers 37 fail-stops (0.37 %) and no wrong value on a clean transfer (two
wrong values were cascades after a fail-stop under the harness's fixed load period, which a
producer waiting for READY cannot reach). Every traced fail-stop has one cause: under ±4 %
weights a flip-flop can fall into *lockstep*, both members firing alternately instead of one
silencing the other. Mapped and fixed (`docs/a1_flipflop.md`, "Lockstep"): the alternating
orbit is an attractor a SET train falls into when the v → u inhibition is ≥ 1.125× nominal, so
u first fires on the train's last pulse; a four-pulse SET train instead of three takes the
mix-B lockstep rate from 0.1–0.5 % of SETs to 0 in 16,000, with the contract unchanged. The
register's campaign re-run on the four-pulse train (Juno 408918, the same 10,000 mix-B
transfers): **0 fail-stops and 0 wrong values** (upper 95 % limit 3 × 10⁻⁴ each) — the
flip-flop register now matches the latch register's channel on this campaign. The 4-bit adder
with its output register on flip-flops (664 neurons) is placed on MCNS at 62.8 % of its edges
carried (the latch adder: 61.6 %), its ten pairs on driven inhibitory pairs, and on its image
with every designed edge present adds 50 / 50 words fresh and chained with no fault, 12 ms
slower to accept than the latch adder (`docs/h1_placement.md`, "The adder on flip-flops";
`bench/h1_ffadder.py`). Carried edges alone still do not compute (condition B).

**The first perturbation campaigns on kernels (2026-09-16, Juno H200, 100 copies × 8 tokens,
mix B).** With no perturbation every copy of every block is correct. Under mix B the render
block (ROM tables, one reader per value) was 799 / 800; the fan-out block (one value read by
two cells), the tick state kernel and the 16-bit perspective kernel each lost outputs in some
copies, and the diagnostic run with every copy's outputs kept shows the failure's shape: **a
duplicated output** — one token's value emitted twice (`…, 203, 203, 83` for `…, 203, 83, 11`),
which shifts the rest of the stream. In eight copies of the fan-out block one did it once; in
eight of the perspective kernel, one. That is a silent error at the protocol level, not a
fail-stop, and it is the "guard-doublet" risk the second review round documented for two-reader
cells. Its mechanism is being traced at the spike level and the fix will carry its own test;
the rates with the time ceiling removed (`docs/a3_kernels.md` §10.3): render 799 / 800,
fan-out 775 / 800, perspective 760 / 800 — and the tick state kernel 666 / 1,600, with 94 of 100
copies failing, a third of them stalling at their second token, and every duplicated run of a
state cell leaving the state silently wrong from then on. The mechanism (a cached "both sources
ready" flag replayed after a two-order guard's doublet) is fixed with end-to-end vetoes on the
final guard; rerun on the fix: tick 1,598 / 1,600 and fan-out 795 / 800 with no wrong values,
render unchanged at 799 / 800, and the 16-bit perspective kernel unchanged at 760 / 800 — its
duplicate has another cause. A second remedy (one-hot request guards with wider margins)
stalled copies under noise and was reverted. The duplicate was then localized from a spike
dump to a request rail left dark by a failed re-ignition (§10.5); the third remedy, re-igniting
it from the row's own `ACT^d` pulse, removed every perspective duplicate (714 / 800, 0 wrong)
but stalled copies in all four blocks and corrupted a tick state, and was reverted too. The
perspective duplicate stays open, with its mechanism known; a fourth, gated remedy (a veto relay
re-lighting the rail only while no request is pending) gave the first campaign with no wrong value
in any block (perspective 779 / 800) but stalled 22 of the tick kernel's 100 copies, and is kept
buildable but off (`docs/a3_kernels.md` §10.5). (The first campaign's large "missing"
counts were the runner's 30 s ceiling: the tick kernel needs 52 s of neural time for eight
tokens and the perspective kernel 39 s.)

**And with textures, under neural pacing (2026-09-16):** `examples/doom2.c` — the same engine
with an 8 × 8 brick texture read by hit position and row — rendered two 40 × 25 frames on the
H200 (8 copies of 686,351 neurons) with the frame's phase order kept by the substrate's own
phase gates and no host barrier: all 2,000 pixels equal to the reference, no fault, 1,982 s
of neural time, 7 h 53 min of wall time (`docs/img/doom2_40_0_neural.png`). The frame loop's
ordering label for this run is *neural*; the token dealing stays hybrid. Its cost is visible:
about 7 s of neural time per token against 3 s under host pacing, because the compiler's
wrapping counter was a three-cell feedback loop every token waited for. A one-hot ring
counter advanced by done pulses replaced it: the two-pass renderer under neural pacing now
runs within 4 % of its host-paced time (29.4 s against 30.6 s per two frames) with a third
fewer neurons, and the 4 × 3 Doom sample went from 279 s to 255 s; the 40 × 25 textured run is
being re-measured on the ring.

Beside it, the Doom-shaped program grew textures (`examples/doom2.c`) and one sprite thing,
occluded by nearer walls (`examples/doom3.c`, four kernels, 1.31 M neurons per copy) that
then chases the player one step per tick with collision (`examples/doom4.c`), validated by
the kernel oracle at 160 × 100 (`docs/img/doom3_0_reference.png`, `doom4_2_reference.png`).

**Stage F2, first step (later the same day):** the frame's phase order — columns, then pixels,
then the tick — is now enforced inside the substrate by phase gates on the input registers
and wrapping counters the compiler adds, and the host only deals tokens in program order;
the two-pass renderer ran two frames correctly with no host barrier at all, and so did the
Doom-shaped program itself (two frames of a 4 × 3 sample, one CPU copy, 279 s of neural
time; `docs/a3_kernels.md` §11).

What this is: the plan's Stage E1 shape (exact integer rendering, pixel records streamed,
assets compiled into ROM relays) and Stage D shape (a world-update kernel driving it) on
the batched simulator that stands in for a cluster of brains, under the hybrid control
plane (the host paces streams and deals tokens; the benchmark label says so). What it is
not yet: a level with a moving player (`examples/doom1.c` compiles to three kernels — a
ray-walking column pass into RAM buffers, a pixel pass, a tick that moves the player — but
one copy of its three kernels is 443k neurons at 16 bits, about three brains' worth, so it
runs on the simulator but not yet in a brain's budget. **It runs**: a one-node validation on
Juno rendered two frames of a 2 × 3 sample, every pixel equal to the reference, with the
tick moving the player between them — 142 s of neural time, 1 h 56 min of wall time on a
CPU core. **And at 40 × 25 on the H200** (32 copies of the three kernels, 460,311 neurons
each, 14.7 million in all): two frames, the player moved by the tick between them, all
2,000 pixels equal to the reference, no fault, no timeout — 310 s of neural time, 2 h 31 min
of wall time (`docs/img/doom40_0_neural.png`, `_1_`; `docs/a2/doom1_40_h200.json`). That is
a Doom-shaped engine — level, ray casting, a z-buffer, a moving player — running in the
substrate with the game update and the renderer both in spikes, at the label *Profile 3,
hybrid pacing*); textures and sprites (`doom2.c`–`doom4.c`, oracle-validated, neural runs next); Profile 2 wiring; TMR.

The mix-B perturbation campaign on the control machine (the sequencer, 4-bit, fixed ALU)
also finished on the H200: 100 random programs, 713 instructions, 98 ok, 1 short, 1 no
halt — both fail-stop — and no silent wrong value (95 % upper limit 3.0 × 10⁻²);
262 minutes of wall time (`docs/a2/machine_4bit_B_summary.json`).

# Toward Stage D: multiplier, indexed addressing, arrays, exact channel

**Date:** 2026-09-15.

**Where things stand at the end of the day.** Programs that run neurally, all matching the
C reference and the IR interpreter: Hello World on the sequencer (31.8 s of neural time,
57 s of wall time); the toy world update on the sequencer (154 s) and as a 13-cell state
kernel (6 s per tick, eight ticks of fresh input); the toy renderer's column loop as a
four-cell kernel (1.18 s per column) and the whole two-loop program as a tick kernel and a
column kernel over two frames; a 16-bit perspective column with a reciprocal table and a
multiply (6.6 s per column); a 32-bit three-cell kernel; the exact channel between two
machines, clean and with a dropped rail event. Compiled and reference-checked, with the
neural runs in progress on the Apple GPU and Juno's H200: a 160 × 100 Doom-like frame one
pixel per token on 32 and 128 copies of the pixel kernel, and a three-frame slideshow whose
tick kernel turns the view. The bullets below are the day's steps in order.

- **Multiplier**: an n × n array of ordered ripple-adder rows on veto relays; each row's output
  re-timed through its completion so the next row's operand bits rise together (the first
  build refused every product: a low-stage generate overtook a late high bit). 4-bit ALU with
  MUL: 2,451 neurons, products correct, ACCEPT ~1 s. `*` in DrosoC.
- **Indexed addressing**: an index word X feeds a second read and write port (LOADI, ADDI…,
  STOREI); `static u8 a[N]` and `a[i]` in DrosoC lower through it. The array-sum program agrees
  across the C reference, the IR interpreter and the lowered reference (38 words); the indexed
  instructions ran neurally on a 16-word machine (STOREI, LOADI, ADDI).
- **CLR** (empty a port word), a **send timer** with a hardware-written status word, and
  **store-pending as a two-rail pair** so hardware-written words never advance the PC. A
  no-link send timed out twice and the handler retransmitted once (neural test).
- **Exact channel** over FlyLink (alternating bit, immediate ACK on a reverse link, retransmit
  on timeout, duplicate rejection by sequence): programs written (`tests/test_exact_channel.py`).
  The first clean-link run delivered the payload once and acknowledged it, but the sender never
  recorded the ACK: its handler `LOAD`ed the timer's status word to tell a reply from a
  timeout, and that word was empty (the timer had never written it; an earlier version emptied
  it with `CLR`), so the read double-railed and the machine halted as a fault. The status word
  is now a proper memory word: 0 at power-up, written to 1 by the timer through a write port of
  its own (the two ports sharing the word veto each other's copies), stored back to 0 by the
  handler. Both scenarios now pass on the reference simulator: clean link — received 5,
  delivered once, pixel 5, acked, no retries, 4 rail events each way; one dropped rail event —
  the timeout fires, the handler re-sends once (retries 1), delivered once and acked.
  A third scenario (2026-09-15, sol worker, fable review, `docs/stage_c_flylink.md`): the ACK
  arrives *after* the send timer has expired but *before* the timeout interrupt is taken. The
  two requests must not merge into one pending flag, so the machine gained a one-deep
  interrupt queue (INTQ) with a "settled" gate (INTS) — the review measured that without the
  gate a request landing 5–30 ms after the interrupt's clear was lost outright — and the
  timer is cancelled both by the ACK's arrival and by its consumption (cancelling only on
  consumption made the clean link retry once on Juno). Clean, dropped-event and late-ACK
  scenarios all pass (Juno 405152): late ACK — two handler entries, one retry, delivered
  once, acked.
- **Kernel compiler** (`compiler/kernel.py`, `lib/kernel.py`, `bench/run_kernel.py`): a loop
  body of the IR becomes a resident pipeline, one cell per operation, the induction variable
  streamed by the host, loop-invariant variables as image-time parameters, arrays as memories
  addressed by the index (the base folds away). The renderer's column loop compiles to four
  cells (ADD, AND, LOAD map, LOAD htab; 10,820 neurons at 8 bits with the two tables as ROM
  relays, 13,727 with RAM masters) and the kernel reference
  equals the IR interpreter's pixels for three headings. **Neural: 8 columns, 8 correct
  pixels, one per 1.18 s self-paced, 4.5 s latency** — against ~32 s per column on the
  sequencer, a ~27× gain at the machine's neuron count. Getting there took a handshake
  between cells (request / idle / commit-pending kill pairs and a two-relay "both true"
  element) after four measured failures of the un-handshaked pipeline; `docs/a3_kernels.md`.
- **The world update as a state kernel**: `examples/tick.c`'s tick (calls inlined, four `if`s
  as select cells, three loop-carried variables as feedback edges with image-lit initial
  state, two outputs per tick) compiles to 13 cells (28,420 neurons) and runs three ticks
  neurally with every output correct, one tick per 6.0 s against ~45 s on the sequencer.
  The builder now joins a cell's requests over every source and gates a commit on every
  reader. With the velocity read every tick (`examples/tick2.c`) the token is the input and
  eight ticks of varied input, wall wrap and clamp included, come out right; 32-bit cells run
  (a three-cell 32-bit kernel, 1.99 s per token) once the input register's watchdog scales
  with the width (`docs/a3_kernels.md` §4–5).
- **E1 shape, 16 bits**: constant shifts as wiring cells, ROM tables as read relays, and
  `examples/render2.c` (distance from the map, reciprocal table, multiply, shift) renders
  eight column heights correctly as a six-cell kernel of 47,775 neurons at 6.6 s per column;
  the array multiplier is the bottleneck (`docs/a3_kernels.md` §5.1).
- **Game loop and render loop together**: `compile_program` splits a two-level loop nest into
  a tick kernel and a column kernel with their own token streams; the columns read the tick's
  state (the heading) as a parameter edge, and the host paces the frames (columns, then the
  tick once the pixels are out, then the next frame once the state has landed: the plan's
  hybrid control plane, labelled). `examples/render.c` runs neurally as two frames of eight
  columns with the heading advancing between them, every pixel and frame record equal to
  the interpreter's, on 16,601 neurons at 14.6 s per frame (`docs/a3_kernels.md` §7).
- **Second review** (Kimi stalled; the `claude-fable` fallback reviewed `26fa035`): the empty
  status word found independently; the refusal window after completion corrected to ~25 ms;
  a commit watchdog added (a hung COMMIT is now a counted timeout); the written/cleared pulses
  into NEXT are address-qualified (a hardware-completed word no longer advances a pending
  STORE); an early `return` no longer falls through; the golden shim's ports follow the
  width. `docs/a2_ram_control.md` §3.2.
- **A latent hole in the sequencer's ALU, closed**: a deselected unit's late outputs (the
  adder's long carry ripple under a MOV, the multiplier under anything) reached the result
  mux after the producer had reset, because the deselect veto was the producer's level. The
  toy world update faulted at its sixth instruction on the machine; the unit select is now
  also a token held in the ALU's reset domain and the mux is vetoed by both
  (`docs/a2_alu_register.md` §2.6).
- **Runner speed**: polling a live simulator through its sorted trace was quadratic; every
  runner now reads the per-step spike lists (`protocol.token.decode_recent`,
  `recent_spikes`). A 13 s kernel run: 25 s wall instead of 50+ minutes; Hello World (31.8 s
  neural, 12.7k neurons): 57 s wall on one CPU core instead of 1,878 s. The simulator was never
  the bottleneck; the polling was.

- **Stage D shape, on the references**: `examples/tick.c`, a toy world update in the shape of
  `p_tick` (player moves by the input velocity with wall collision, a monster steps toward the
  player, contact costs health, three ticks, one pixel record per entity per tick and the
  health at the end): 65 machine words, 133 instructions per run; the C reference, the IR
  interpreter and the machine reference agree on all outputs and the canonical state for
  three inputs. **Neural (sequencer): all seven outputs match** — [25, 88, 30, 86, 35, 84,
  100] in 154 s of neural time (936 s wall on one core) — after the ALU hole below was
  closed; the first attempt faulted at the sixth instruction.
- **Stage E1 shape, on the references**: `examples/render.c`, a toy renderer in the shape of
  the `r_segs` column loop (for each of 8 columns the wall distance from a map array indexed by
  the heading, the height from a lookup table, one pixel record per column, a frame record,
  two frames): 82 machine words, 28 data words, 517 instructions for two frames; the three
  references agree on every pixel for two headings. **Neural (sequencer, Juno): all 18
  outputs match** in 586 s of neural time (3,269 s wall on one CPU core; a first attempt was
  OOM-killed at 16 GB before the runners trimmed old spikes). On this sequencer a frame is
  ~250 instructions, about five minutes of neural time; that number is the case for resident
  kernels (spatial dataflow) and the cluster rather than a faster sequencer.
